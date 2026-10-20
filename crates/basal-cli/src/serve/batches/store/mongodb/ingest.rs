//! Discovery and validation leases, bounded ingestion, seal and validation failure (MongoDB).

use std::time::Duration;

use mongodb::bson::{doc, Bson, Document};

use super::super::error::{InvalidValue, StoreError};
use super::super::ops::common::{millis, phase_free, phase_lease_valid, signed, unsigned, BatchRow, QuotaDelta};
use super::super::ops::ingest::{MAX_APPEND, MAX_TARGETS};
use super::super::policy::record_reserve;
use super::super::types::{
    AppendOutcome, BatchIssue, BatchRecord, BatchRef, BatchStatus, CatalogEntry, ClaimId, ClaimToken, DiscoveryLease,
    Fingerprint, InputFile, LeaseKind, LeaseTiming, ModelName, PlannerVersion, SealRequest, TargetResolution,
    TerminalStatus, UnixMillis, ValidatedRecord, ValidationLease, WorkerId, WorkerTarget, RESULT_MAX,
};
use super::common::{
    apply_quota, count_batch_items, end_phase, insert_claim, lock_batch, lock_quota, read_batch, set_batch,
};
use super::convert::{artifact, batch_ref, digest};
use super::driver::{
    bin, bytes, corrupt, int, mtx, text, Core, Tx, BATCHES, CATALOG, CATALOG_REVISIONS, CLAIMS, DEFICITS, FILES, ITEMS,
    SCHEDULE,
};
use super::lifecycle::{items_of, terminate_unsealed};

pub(super) async fn input_file(tx: &mut Tx<'_>, batch: &BatchRow) -> Result<InputFile, StoreError> {
    let doc = tx
        .one(FILES, doc! { "_id": batch.record.input_file_id.as_str(), "owner_id": batch.record.owner.as_str() })
        .await?
        .ok_or(corrupt("pinned input file missing"))?;
    Ok(InputFile {
        artifact: artifact(text(&doc, "artifact_ref")?)?,
        bytes: super::driver::uint(&doc, "bytes")?,
        sha256: digest(bin(&doc, "sha256")?)?,
    })
}

/// Take a phase lease on a locked, eligible batch.
pub(super) async fn take_phase(
    tx: &mut Tx<'_>,
    batch: &BatchRow,
    kind: LeaseKind,
    worker: &WorkerId,
    lease: Duration,
    now: i64,
) -> Result<(ClaimToken, LeaseTiming), StoreError> {
    let claim_id = ClaimId::new_random();
    let generation = batch.phase_generation + 1;
    let lease_until = now.saturating_add(millis(lease));
    if let Some(previous) = &batch.phase_claim_id {
        tx.update(
            CLAIMS,
            doc! { "_id": previous.as_str(), "state": "active" },
            doc! { "$set": { "state": "expired" } },
        )
        .await?;
    }
    let key = batch.batch_ref();
    insert_claim(
        tx,
        claim_id.as_str(),
        kind,
        worker.as_str(),
        generation,
        &key,
        batch.record.control_epoch,
        lease_until,
        now,
    )
    .await?;
    set_batch(
        tx,
        &key,
        doc! { "$set": {
            "phase_kind": kind.as_db(),
            "phase_claim_id": claim_id.as_str(),
            "phase_generation": generation,
            "phase_lease_until": lease_until,
        } },
    )
    .await?;
    let token = ClaimToken {
        claim_id,
        worker: worker.clone(),
        kind,
        batch: key,
        generation,
        control_epoch: batch.record.control_epoch,
    };
    let timing = LeaseTiming {
        server_now: UnixMillis(now),
        lease_until: UnixMillis(lease_until),
        batch_expires_at: batch.record.expires_at,
    };
    Ok((token, timing))
}

/// Pick a candidate from the snapshot, then lock it (a concurrent taker write-conflicts) and re-check it.
async fn lock_candidate(
    tx: &mut Tx<'_>,
    filter: Document,
    sort: Document,
) -> Result<Option<(BatchRow, i64)>, StoreError> {
    let Some(candidate) = tx.one_sorted(BATCHES, filter, sort).await? else { return Ok(None) };
    let key = batch_ref(&candidate)?;
    let (batch, now) = lock_batch(tx, &key).await?;
    if !phase_free(&batch, now) {
        return Ok(None);
    }
    Ok(Some((batch, now)))
}

fn phase_open_filter(now: i64) -> Document {
    doc! { "$or": [{ "phase_claim_id": Bson::Null }, { "phase_lease_until": { "$lt": now } }] }
}

async fn claim_discovery_tx(
    tx: &mut Tx<'_>,
    worker: &WorkerId,
    now: i64,
) -> Result<Option<DiscoveryLease>, StoreError> {
    let mut filter = doc! { "status": "validating", "fingerprint": Bson::Null, "expires_at": { "$gt": now } };
    filter.extend(phase_open_filter(now));
    let Some((batch, now)) = lock_candidate(tx, filter, doc! { "created_at": 1_i32 }).await? else { return Ok(None) };
    if batch.record.status != BatchStatus::Validating || batch.record.fingerprint.is_some() {
        return Ok(None);
    }
    let input = input_file(tx, &batch).await?;
    let lease = tx.settings().discovery_lease;
    let (token, timing) = take_phase(tx, &batch, LeaseKind::Discovery, worker, lease, now).await?;
    Ok(Some(DiscoveryLease { token, timing, input }))
}

pub(super) async fn claim_discovery(core: &Core, worker: &WorkerId) -> Result<Option<DiscoveryLease>, StoreError> {
    let now = core.server_now().await?;
    mtx!(core, tx => claim_discovery_tx(tx, worker, now))
}

async fn resolve_target_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    model: Option<&ModelName>,
) -> Result<TargetResolution, StoreError> {
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    if batch.record.status != BatchStatus::Validating || !phase_lease_valid(&batch, token, LeaseKind::Discovery, now) {
        return Err(StoreError::LeaseStale);
    }
    let revision = batch.record.catalog_revision.0;
    let name = match model {
        Some(name) => name.as_str().to_owned(),
        None => text(
            &tx.one(CATALOG_REVISIONS, doc! { "_id": revision }).await?.ok_or(corrupt("pinned catalog missing"))?,
            "default_model",
        )?,
    };
    let Some(row) = tx.one(CATALOG, doc! { "revision": revision, "model_name": name.as_str() }).await? else {
        return Ok(TargetResolution::ModelNotInCatalog);
    };
    let entry = CatalogEntry {
        model_name: ModelName::parse(&text(&row, "model_name")?).map_err(|_| corrupt("catalog model"))?,
        fingerprint: Fingerprint::parse(&text(&row, "fingerprint")?).map_err(|_| corrupt("catalog fingerprint"))?,
        planner_version: PlannerVersion::parse(&text(&row, "planner_version")?)
            .map_err(|_| corrupt("catalog planner"))?,
    };
    set_batch(
        tx,
        &token.batch,
        doc! { "$set": {
            "model_name": entry.model_name.as_str(),
            "fingerprint": entry.fingerprint.as_str(),
            "planner_version": entry.planner_version.as_str(),
        } },
    )
    .await?;
    end_phase(tx, &batch).await?;
    Ok(TargetResolution::Resolved(entry))
}

pub(super) async fn resolve_target(
    core: &Core,
    token: &ClaimToken,
    model: Option<&ModelName>,
) -> Result<TargetResolution, StoreError> {
    mtx!(core, tx => resolve_target_tx(tx, token, model))
}

async fn claim_validation_tx(
    tx: &mut Tx<'_>,
    worker: &WorkerId,
    targets: &[WorkerTarget],
    now: i64,
) -> Result<Option<ValidationLease>, StoreError> {
    let matches: Vec<Document> = targets
        .iter()
        .map(|t| doc! { "fingerprint": t.fingerprint.as_str(), "planner_version": t.planner_version.as_str() })
        .collect();
    let filter = doc! {
        "status": "validating",
        "expires_at": { "$gt": now },
        "$and": [{ "$or": matches }, phase_open_filter(now)],
    };
    let Some((batch, now)) = lock_candidate(tx, filter, doc! { "created_at": 1_i32 }).await? else { return Ok(None) };
    let (Some(model), Some(fingerprint), Some(planner)) =
        (batch.record.model_name.clone(), batch.record.fingerprint.clone(), batch.planner_version.clone())
    else {
        return Err(corrupt("validating batch without resolved target"));
    };
    if batch.record.status != BatchStatus::Validating
        || !targets.iter().any(|t| t.fingerprint == fingerprint && t.planner_version == planner)
    {
        return Ok(None);
    }
    let input = input_file(tx, &batch).await?;
    let ingested = count_batch_items(tx, &batch.batch_ref()).await?;
    let lease = tx.settings().validation_lease;
    let (token, timing) = take_phase(tx, &batch, LeaseKind::Validation, worker, lease, now).await?;
    Ok(Some(ValidationLease {
        token,
        timing,
        input,
        target: CatalogEntry { model_name: model, fingerprint, planner_version: planner },
        ingested: unsigned(ingested)?,
    }))
}

pub(super) async fn claim_validation(
    core: &Core,
    worker: &WorkerId,
    targets: &[WorkerTarget],
) -> Result<Option<ValidationLease>, StoreError> {
    if targets.is_empty() || targets.len() > MAX_TARGETS {
        return Err(StoreError::Invalid(InvalidValue::new("targets", "must list 1..16 targets")));
    }
    let now = core.server_now().await?;
    mtx!(core, tx => claim_validation_tx(tx, worker, targets, now))
}

async fn lock_validating(tx: &mut Tx<'_>, token: &ClaimToken) -> Result<(BatchRow, i64), StoreError> {
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    if batch.record.status != BatchStatus::Validating
        || !phase_lease_valid(&batch, token, LeaseKind::Validation, now)
        || now >= batch.record.expires_at.0
    {
        return Err(StoreError::LeaseStale);
    }
    Ok((batch, now))
}

async fn append_validated_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    records: &[ValidatedRecord],
) -> Result<AppendOutcome, StoreError> {
    let (batch, _now) = lock_validating(tx, token).await?;
    let scope = items_of(&token.batch);
    let mut staged_reserve = batch.staged_reserve;
    let mut ingested = count_batch_items(tx, &token.batch).await?;
    let overhead = unsigned(batch.item_overhead)?;
    let mut outcome = None;
    for record in records {
        if record.line_no == 0 || record.length == 0 || record.cost == 0 || record.result_bound == 0 {
            return Err(StoreError::IngestRejected("record fields out of range"));
        }
        let line_no = signed(record.line_no)?;
        let mut key = scope.clone();
        key.insert("line_no", line_no);
        if let Some(existing) = tx.one(ITEMS, key.clone()).await? {
            let same = bin(&existing, "custom_id")? == record.custom_id.as_bytes()
                && int(&existing, "input_offset")? == signed(record.offset)?
                && int(&existing, "input_length")? == signed(record.length)?
                && bin(&existing, "input_sha256")? == record.input_sha256.0;
            if !same {
                return Err(StoreError::IngestConflict { line_no: record.line_no });
            }
            continue;
        }
        let mut by_custom = scope.clone();
        by_custom.insert("custom_id", bytes(record.custom_id.as_bytes()));
        if let Some(first) = tx.one(ITEMS, by_custom).await? {
            outcome = Some(AppendOutcome::DuplicateCustomId {
                line_no: record.line_no,
                first_line_no: super::driver::uint(&first, "line_no")?,
            });
            break;
        }
        let reserve = signed(record_reserve(&record.custom_id, overhead))?;
        staged_reserve = staged_reserve.saturating_add(reserve);
        if staged_reserve > batch.reserve_records || ingested >= batch.max_records {
            return Err(StoreError::IngestRejected("records exceed the completion reserve of the batch"));
        }
        let mut item = key;
        item.extend(doc! {
            "custom_id": bytes(record.custom_id.as_bytes()),
            "input_offset": signed(record.offset)?,
            "input_length": signed(record.length)?,
            "input_sha256": bytes(&record.input_sha256.0),
            "state": "pending",
            "generation": 0_i64,
            "claim_id": Bson::Null,
            "failure_attempts": 0_i64,
            "abandon_attempts": 0_i64,
            "eligible_at": 0_i64,
            "requires_solo": false,
            "cost": signed(record.cost)?,
            "record_reserve": reserve,
            "result_bound": signed(record.result_bound.min(RESULT_MAX))?,
            "host_bound": signed(record.host_bound)?,
            "result_kind": Bson::Null,
            "result_inline": Bson::Null,
            "result_ref": Bson::Null,
            "result_bytes": Bson::Null,
            "result_sha256": Bson::Null,
            "accepted_at": Bson::Null,
            "quota_used": 0_i64,
        });
        tx.insert(ITEMS, item).await?;
        ingested += 1;
    }
    set_batch(tx, &token.batch, doc! { "$set": { "staged_reserve": staged_reserve } }).await?;
    Ok(outcome.unwrap_or(AppendOutcome::Appended { ingested: unsigned(ingested)? }))
}

pub(super) async fn append_validated(
    core: &Core,
    token: &ClaimToken,
    records: &[ValidatedRecord],
) -> Result<AppendOutcome, StoreError> {
    if records.len() > MAX_APPEND {
        return Err(StoreError::IngestRejected("too many records in one append"));
    }
    mtx!(core, tx => append_validated_tx(tx, token, records))
}

async fn seal_input_tx(tx: &mut Tx<'_>, token: &ClaimToken, request: &SealRequest) -> Result<BatchRecord, StoreError> {
    let owner = &token.batch.owner;
    lock_quota(tx, owner).await?;
    let unlocked = read_batch(tx, &token.batch).await?.ok_or(StoreError::NotFound)?;
    let shard = unlocked.record.fingerprint.clone().ok_or(StoreError::LeaseStale)?;
    if shard != request.fingerprint {
        return Err(StoreError::SealRejected("fingerprint differs from the resolved target"));
    }
    // Defaults are replaced by the claim parameters of the first worker of the shard.
    let (schedule, _) = tx
        .lock_or_create(
            SCHEDULE,
            shard.as_str(),
            doc! { "turn": 0_i64, "cursor_owner": Bson::Null, "quantum": 1_i64, "credit_cap": 1_i64, "version": 0_i64 },
        )
        .await?;
    let schedule_turn = int(&schedule, "turn")?;
    let (batch, now) = lock_validating(tx, token).await?;
    let input = input_file(tx, &batch).await?;
    if input.sha256 != request.input_sha256 {
        return Err(StoreError::SealRejected("input digest differs from the stored file"));
    }
    let total = signed(request.total)?;
    let rows = tx
        .aggregate(
            ITEMS,
            vec![
                doc! { "$match": items_of(&token.batch) },
                doc! { "$group": {
                    "_id": Bson::Null,
                    "n": { "$sum": 1_i64 },
                    "max_line": { "$max": "$line_no" },
                    "reserves": { "$sum": "$record_reserve" },
                } },
            ],
        )
        .await?;
    let (count, max_line, reserves) = match rows.first() {
        Some(row) => (int(row, "n")?, int(row, "max_line")?, int(row, "reserves")?),
        None => (0, 0, 0),
    };
    if total < 1 || count != total || max_line != total {
        return Err(StoreError::SealRejected("ingested records are incomplete"));
    }
    let completion = batch.object_charge.saturating_mul(2).saturating_add(reserves);
    if completion > batch.completion_reserved {
        return Err(corrupt("record reserves exceed the completion reserve"));
    }
    let deficit_key = doc! { "shard": shard.as_str(), "owner_id": owner.as_str() };
    let batch_turn = match tx.one(DEFICITS, deficit_key.clone()).await? {
        Some(deficit) => int(&deficit, "batch_turn")?,
        None => {
            let mut deficit = deficit_key;
            deficit.extend(doc! {
                "credit": 0_i64,
                "last_turn": schedule_turn,
                "reserved_cost": 0_i64,
                "batch_turn": 0_i64,
                "batch_cursor": Bson::Null,
                "quota_blocked_version": Bson::Null,
                "blocked_until": Bson::Null,
            });
            tx.insert(DEFICITS, deficit).await?;
            0
        }
    };
    end_phase(tx, &batch).await?;
    set_batch(
        tx,
        &token.batch,
        doc! { "$set": {
            "status": "in_progress",
            "sealed": true,
            "total": total,
            "queued": total,
            "in_progress_at": now,
            "completion_reserved": completion,
            "sched_credit": 0_i64,
            "sched_turn": batch_turn,
        } },
    )
    .await?;
    apply_quota(tx, owner, QuotaDelta { reserved: completion - batch.completion_reserved, ..QuotaDelta::default() })
        .await?;
    Ok(read_batch(tx, &token.batch).await?.ok_or(corrupt("sealed batch missing"))?.record)
}

/// Seal (quota → shard → batch): lease, deadline, completeness, input digest and fingerprint are checked before the
/// completion reserve is trimmed and `total` becomes public.
pub(super) async fn seal_input(
    core: &Core,
    token: &ClaimToken,
    request: &SealRequest,
) -> Result<BatchRecord, StoreError> {
    mtx!(core, tx => seal_input_tx(tx, token, request))
}

async fn fail_validation_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    issues: &[BatchIssue],
    truncated: bool,
) -> Result<BatchRecord, StoreError> {
    lock_quota(tx, &token.batch.owner).await?;
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    if batch.record.status != BatchStatus::Validating || !phase_lease_valid(&batch, token, token.kind, now) {
        return Err(StoreError::LeaseStale);
    }
    terminate_unsealed(tx, &batch, TerminalStatus::Failed, issues, truncated, now).await?;
    let key: &BatchRef = &token.batch;
    Ok(read_batch(tx, key).await?.ok_or(corrupt("failed batch missing"))?.record)
}

pub(super) async fn fail_validation(
    core: &Core,
    token: &ClaimToken,
    issues: &[BatchIssue],
    truncated: bool,
) -> Result<BatchRecord, StoreError> {
    if !matches!(token.kind, LeaseKind::Discovery | LeaseKind::Validation) {
        return Err(StoreError::LeaseStale);
    }
    mtx!(core, tx => fail_validation_tx(tx, token, issues, truncated))
}
