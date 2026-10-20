//! Discovery and validation leases, bounded ingestion of validated records, seal and validation failure.

use super::super::error::StoreError;
use super::super::policy::record_reserve;
use super::super::sql::{args, db_now, SqlRow, SqlTx, Val};
use super::super::types::{
    AppendOutcome, BatchIssue, BatchRecord, BatchRef, BatchStatus, CatalogEntry, ClaimId, ClaimToken, DiscoveryLease,
    Fingerprint, InputFile, LeaseKind, LeaseTiming, ModelName, PlannerVersion, SealRequest, StoreSettings,
    TargetResolution, UnixMillis, ValidatedRecord, ValidationLease, WorkerId, WorkerTarget,
};
use super::common::{
    apply_quota, artifact_from_db, batch_row, corrupt, digest_from_db, end_phase, insert_claim, load_batch, lock_batch,
    lock_quota, millis, phase_free, phase_lease_valid, scalar, signed, unsigned, BatchRow, QuotaDelta, BATCH_COLUMNS,
};
use super::lifecycle::terminate_unsealed;

/// Largest number of records accepted by one `append_validated` call.
pub(in super::super) const MAX_APPEND: usize = 1024;

/// Largest number of fingerprints a worker may declare in one validation claim.
pub(in super::super) const MAX_TARGETS: usize = 16;

pub(in super::super) async fn input_file<T: SqlTx>(tx: &mut T, batch: &BatchRow) -> Result<InputFile, StoreError> {
    let row = tx
        .fetch_optional(
            "SELECT artifact_ref, bytes, sha256 FROM batch_files WHERE owner_id = $1 AND id = $2",
            args![batch.record.owner.as_str(), batch.record.input_file_id.as_str()],
        )
        .await?
        .ok_or(corrupt("pinned input file missing"))?;
    Ok(InputFile {
        artifact: artifact_from_db(row.text_at(0)?)?,
        bytes: row.u64_at(1)?,
        sha256: digest_from_db(row.blob_at(2)?)?,
    })
}

/// Take a phase lease on a locked, eligible batch and return its token.
pub(in super::super) async fn take_phase<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRow,
    kind: LeaseKind,
    worker: &WorkerId,
    lease: std::time::Duration,
    now: i64,
) -> Result<(ClaimToken, LeaseTiming), StoreError> {
    let claim_id = ClaimId::new_random();
    let generation = batch.phase_generation + 1;
    let lease_until = now.saturating_add(millis(lease));
    if let Some(previous) = &batch.phase_claim_id {
        tx.execute(
            "UPDATE batch_claims SET state = 'expired' WHERE id = $1 AND state = 'active'",
            args![previous.as_str()],
        )
        .await?;
    }
    let batch_ref = batch.batch_ref();
    insert_claim(
        tx,
        claim_id.as_str(),
        kind,
        worker.as_str(),
        generation,
        &batch_ref,
        batch.record.control_epoch,
        lease_until,
        now,
    )
    .await?;
    tx.execute(
        "UPDATE batches SET phase_kind = $3, phase_claim_id = $4, phase_generation = $5, phase_lease_until = $6 \
         WHERE owner_id = $1 AND id = $2",
        args![
            batch_ref.owner.as_str(),
            batch_ref.id.as_str(),
            kind.as_db(),
            claim_id.as_str(),
            generation,
            lease_until
        ],
    )
    .await?;
    let token = ClaimToken {
        claim_id,
        worker: worker.clone(),
        kind,
        batch: batch_ref,
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

/// Any worker may take a short discovery lease to read the first record and resolve its model.
pub(in super::super) async fn claim_discovery<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    worker: &WorkerId,
) -> Result<Option<DiscoveryLease>, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}{BATCH_COLUMNS} FROM batches{} WHERE status = 'validating' AND fingerprint IS NULL \
         AND (phase_claim_id IS NULL OR phase_lease_until < $1) AND expires_at > $1 \
         ORDER BY created_at{}{}",
        d.top("1"),
        d.skip_hint,
        d.limit("1"),
        d.skip_locked
    );
    let Some(row) = tx.fetch_optional(&sql, args![now]).await? else { return Ok(None) };
    let batch = batch_row(&row)?;
    let now = db_now(tx).await?;
    if !phase_free(&batch, now) {
        return Ok(None);
    }
    let input = input_file(tx, &batch).await?;
    let (token, timing) = take_phase(tx, &batch, LeaseKind::Discovery, worker, settings.discovery_lease, now).await?;
    Ok(Some(DiscoveryLease { token, timing, input }))
}

/// Resolve the model of the first record (or the default model) in the pinned catalog revision.
pub(in super::super) async fn resolve_target<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    model: Option<&ModelName>,
) -> Result<TargetResolution, StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::Validating || !phase_lease_valid(&batch, token, LeaseKind::Discovery, now) {
        return Err(StoreError::LeaseStale);
    }
    let revision = batch.record.catalog_revision.0;
    let name = match model {
        Some(name) => name.as_str().to_owned(),
        None => tx
            .fetch_optional("SELECT default_model FROM batch_catalog_revisions WHERE revision = $1", args![revision])
            .await?
            .ok_or(corrupt("pinned catalog revision missing"))?
            .text_at(0)?,
    };
    let Some(row) = tx
        .fetch_optional(
            &format!(
                "SELECT model_name, fingerprint, planner_version FROM batch_catalog WHERE revision = $1 AND {}",
                tx.dialect().exact_eq("model_name", "$2")
            ),
            args![revision, name.as_str()],
        )
        .await?
    else {
        return Ok(TargetResolution::ModelNotInCatalog);
    };
    let entry = CatalogEntry {
        model_name: ModelName::parse(&row.text_at(0)?).map_err(|_| corrupt("catalog model"))?,
        fingerprint: Fingerprint::parse(&row.text_at(1)?).map_err(|_| corrupt("catalog fingerprint"))?,
        planner_version: PlannerVersion::parse(&row.text_at(2)?).map_err(|_| corrupt("catalog planner"))?,
    };
    tx.execute(
        "UPDATE batches SET model_name = $3, fingerprint = $4, planner_version = $5 WHERE owner_id = $1 AND id = $2",
        args![
            token.batch.owner.as_str(),
            token.batch.id.as_str(),
            entry.model_name.as_str(),
            entry.fingerprint.as_str(),
            entry.planner_version.as_str()
        ],
    )
    .await?;
    end_phase(tx, &batch).await?;
    Ok(TargetResolution::Resolved(entry))
}

/// Full validation lease, only for a worker declaring the batch's fingerprint and planner version.
pub(in super::super) async fn claim_validation<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    worker: &WorkerId,
    targets: &[WorkerTarget],
) -> Result<Option<ValidationLease>, StoreError> {
    if targets.is_empty() || targets.len() > MAX_TARGETS {
        return Err(StoreError::Invalid(super::super::error::InvalidValue::new("targets", "must list 1..16 targets")));
    }
    let now = db_now(tx).await?;
    let matches = (0..targets.len())
        .map(|i| format!("(fingerprint = ${} AND planner_version = ${})", 2 + 2 * i, 3 + 2 * i))
        .collect::<Vec<_>>()
        .join(" OR ");
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}{BATCH_COLUMNS} FROM batches{} WHERE status = 'validating' AND ({matches}) \
         AND (phase_claim_id IS NULL OR phase_lease_until < $1) AND expires_at > $1 \
         ORDER BY created_at{}{}",
        d.top("1"),
        d.skip_hint,
        d.limit("1"),
        d.skip_locked
    );
    let mut values: Vec<Val> = args![now];
    for target in targets {
        values.push(Val::from(target.fingerprint.as_str()));
        values.push(Val::from(target.planner_version.as_str()));
    }
    let Some(row) = tx.fetch_optional(&sql, values).await? else { return Ok(None) };
    let batch = batch_row(&row)?;
    let now = db_now(tx).await?;
    let (Some(model), Some(fingerprint), Some(planner)) =
        (batch.record.model_name.clone(), batch.record.fingerprint.clone(), batch.planner_version.clone())
    else {
        return Err(corrupt("validating batch without resolved target"));
    };
    // A worker with a diverging planner contract must not validate this batch.
    if !targets.iter().any(|t| t.fingerprint == fingerprint && t.planner_version == planner) || !phase_free(&batch, now)
    {
        return Ok(None);
    }
    let input = input_file(tx, &batch).await?;
    let ingested = scalar(
        tx,
        &format!("SELECT {} FROM batch_items WHERE owner_id = $1 AND batch_id = $2", tx.dialect().count_all),
        args![batch.record.owner.as_str(), batch.record.id.as_str()],
    )
    .await?;
    let (token, timing) = take_phase(tx, &batch, LeaseKind::Validation, worker, settings.validation_lease, now).await?;
    Ok(Some(ValidationLease {
        token,
        timing,
        input,
        target: CatalogEntry { model_name: model, fingerprint, planner_version: planner },
        ingested: unsigned(ingested)?,
    }))
}

async fn lock_validating<T: SqlTx>(tx: &mut T, token: &ClaimToken) -> Result<(BatchRow, i64), StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::Validating
        || !phase_lease_valid(&batch, token, LeaseKind::Validation, now)
        || now >= batch.record.expires_at.0
    {
        return Err(StoreError::LeaseStale);
    }
    Ok((batch, now))
}

/// Append validated records idempotently by line number. A re-appended line must be identical; a binary duplicate
/// `custom_id` ends the append with [`AppendOutcome::DuplicateCustomId`] (earlier records of the call stay).
pub(in super::super) async fn append_validated<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    records: &[ValidatedRecord],
) -> Result<AppendOutcome, StoreError> {
    if records.len() > MAX_APPEND {
        return Err(StoreError::IngestRejected("too many records in one append"));
    }
    let (batch, _now) = lock_validating(tx, token).await?;
    let owner = batch.record.owner.as_str().to_owned();
    let id = batch.record.id.as_str().to_owned();
    let mut staged_reserve = batch.staged_reserve;
    let d = tx.dialect();
    let mut ingested = scalar(
        tx,
        &format!("SELECT {} FROM batch_items WHERE owner_id = $1 AND batch_id = $2", d.count_all),
        args![owner.as_str(), id.as_str()],
    )
    .await?;
    let duplicate = format!(
        "SELECT line_no FROM batch_items WHERE owner_id = $1 AND batch_id = $2 AND {}",
        d.exact_eq("custom_id", "$3")
    );
    let overhead = unsigned(batch.item_overhead)?;
    let mut outcome = None;
    for record in records {
        if record.line_no == 0 || record.length == 0 || record.cost == 0 || record.result_bound == 0 {
            return Err(StoreError::IngestRejected("record fields out of range"));
        }
        let line_no = signed(record.line_no)?;
        if let Some(existing) = tx
            .fetch_optional(
                "SELECT custom_id, input_offset, input_length, input_sha256 FROM batch_items \
                 WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
                args![owner.as_str(), id.as_str(), line_no],
            )
            .await?
        {
            let same = existing.blob_at(0)? == record.custom_id.as_bytes()
                && existing.i64_at(1)? == signed(record.offset)?
                && existing.i64_at(2)? == signed(record.length)?
                && existing.blob_at(3)? == record.input_sha256.0;
            if !same {
                return Err(StoreError::IngestConflict { line_no: record.line_no });
            }
            continue;
        }
        if let Some(first) =
            tx.fetch_optional(&duplicate, args![owner.as_str(), id.as_str(), record.custom_id.as_bytes()]).await?
        {
            outcome =
                Some(AppendOutcome::DuplicateCustomId { line_no: record.line_no, first_line_no: first.u64_at(0)? });
            break;
        }
        let reserve = signed(record_reserve(&record.custom_id, overhead))?;
        staged_reserve = staged_reserve.saturating_add(reserve);
        if staged_reserve > batch.reserve_records || ingested >= batch.max_records {
            return Err(StoreError::IngestRejected("records exceed the completion reserve of the batch"));
        }
        tx.execute(
            "INSERT INTO batch_items (owner_id, batch_id, line_no, custom_id, input_offset, input_length, input_sha256, \
             state, cost, record_reserve, result_bound, host_bound) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending', $8, $9, $10, $11)",
            args![
                owner.as_str(),
                id.as_str(),
                line_no,
                record.custom_id.as_bytes(),
                signed(record.offset)?,
                signed(record.length)?,
                record.input_sha256.0.to_vec(),
                signed(record.cost)?,
                reserve,
                signed(record.result_bound.min(super::super::types::RESULT_MAX))?,
                signed(record.host_bound)?
            ],
        )
        .await?;
        ingested += 1;
    }
    tx.execute(
        "UPDATE batches SET staged_reserve = $3 WHERE owner_id = $1 AND id = $2",
        args![owner.as_str(), id.as_str(), staged_reserve],
    )
    .await?;
    Ok(outcome.unwrap_or(AppendOutcome::Appended { ingested: unsigned(ingested)? }))
}

/// Seal after complete validation (quota → shard → batch): checks lease, deadline, completeness, input digest
/// and fingerprint, trims the completion reserve to the exact record reserves and only then publishes `total`.
pub(in super::super) async fn seal_input<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    request: &SealRequest,
) -> Result<BatchRecord, StoreError> {
    let owner = &token.batch.owner;
    lock_quota(tx, owner).await?;
    // The fingerprint is immutable once resolved, so the shard can be locked before the batch.
    let unlocked = load_batch(tx, owner, &token.batch.id, false).await?.ok_or(StoreError::NotFound)?;
    let shard = unlocked.record.fingerprint.clone().ok_or(StoreError::LeaseStale)?;
    if shard != request.fingerprint {
        return Err(StoreError::SealRejected("fingerprint differs from the resolved target"));
    }
    let schedule_turn = lock_shard_for_seal(tx, &shard).await?;
    let (batch, now) = lock_validating(tx, token).await?;
    let input = input_file(tx, &batch).await?;
    if input.sha256 != request.input_sha256 {
        return Err(StoreError::SealRejected("input digest differs from the stored file"));
    }
    let total = signed(request.total)?;
    let row = tx
        .fetch_optional(
            &format!(
                "SELECT {}, CAST(COALESCE(MAX(line_no), 0) AS BIGINT), \
                 CAST(COALESCE(SUM(record_reserve), 0) AS BIGINT) \
                 FROM batch_items WHERE owner_id = $1 AND batch_id = $2",
                tx.dialect().count_all
            ),
            args![owner.as_str(), batch.record.id.as_str()],
        )
        .await?
        .ok_or(corrupt("aggregate returned no row"))?;
    let (count, max_line, reserves) = (row.i64_at(0)?, row.i64_at(1)?, row.i64_at(2)?);
    if total < 1 || count != total || max_line != total {
        return Err(StoreError::SealRejected("ingested records are incomplete"));
    }
    let completion = batch.object_charge.saturating_mul(2).saturating_add(reserves);
    if completion > batch.completion_reserved {
        return Err(corrupt("record reserves exceed the completion reserve"));
    }
    let sql =
        tx.dialect().insert_if_absent("batch_deficits", &["shard", "owner_id", "last_turn"], &["$1", "$2", "$3"], 2);
    tx.execute(&sql, args![shard.as_str(), owner.as_str(), schedule_turn]).await?;
    let batch_turn = scalar(
        tx,
        "SELECT batch_turn FROM batch_deficits WHERE shard = $1 AND owner_id = $2",
        args![shard.as_str(), owner.as_str()],
    )
    .await?;
    end_phase(tx, &batch).await?;
    tx.execute(
        "UPDATE batches SET status = 'in_progress', sealed = 1, total = $3, queued = $3, in_progress_at = $4, \
         completion_reserved = $5, sched_credit = 0, sched_turn = $6 WHERE owner_id = $1 AND id = $2",
        args![owner.as_str(), batch.record.id.as_str(), total, now, completion, batch_turn],
    )
    .await?;
    apply_quota(tx, owner, QuotaDelta { reserved: completion - batch.completion_reserved, ..QuotaDelta::default() })
        .await?;
    Ok(load_batch(tx, owner, &batch.record.id, false).await?.ok_or(corrupt("sealed batch missing"))?.record)
}

async fn lock_shard_for_seal<T: SqlTx>(tx: &mut T, shard: &Fingerprint) -> Result<i64, StoreError> {
    // Defaults are replaced by the claim parameters of the first worker of the shard.
    let d = tx.dialect();
    let sql = d.insert_if_absent("batch_schedule", &["shard", "quantum", "credit_cap"], &["$1", "1", "1"], 1);
    tx.execute(&sql, args![shard.as_str()]).await?;
    let sql = format!("SELECT turn FROM batch_schedule{} WHERE shard = $1{}", d.lock_hint, d.for_update);
    let row = tx.fetch_optional(&sql, args![shard.as_str()]).await?.ok_or(corrupt("shard row missing"))?;
    row.i64_at(0)
}

/// Fail an unsealed batch with bounded issues (discovery or validation lease).
pub(in super::super) async fn fail_validation<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    token: &ClaimToken,
    issues: &[BatchIssue],
    truncated: bool,
) -> Result<BatchRecord, StoreError> {
    if !matches!(token.kind, LeaseKind::Discovery | LeaseKind::Validation) {
        return Err(StoreError::LeaseStale);
    }
    lock_quota(tx, &token.batch.owner).await?;
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::Validating || !phase_lease_valid(&batch, token, token.kind, now) {
        return Err(StoreError::LeaseStale);
    }
    terminate_unsealed(tx, settings, &batch, super::super::types::TerminalStatus::Failed, issues, truncated, now)
        .await?;
    let batch_ref: &BatchRef = &token.batch;
    Ok(load_batch(tx, &batch_ref.owner, &batch_ref.id, false).await?.ok_or(corrupt("failed batch missing"))?.record)
}
