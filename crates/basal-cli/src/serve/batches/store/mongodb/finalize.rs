//! Finalization lease with export reservation and fencing, export reads, publication with the terminal
//! transaction, post-publication cleanup and metadata retention (MongoDB).

use mongodb::bson::{doc, Bson, Document};

use super::super::error::StoreError;
use super::super::ops::common::{millis, phase_free, phase_lease_valid, signed, unsigned, BatchRow, QuotaDelta};
use super::super::types::{
    ArtifactRef, BatchRecord, BatchRef, BatchStatus, ClaimToken, CleanupPage, CustomId, ExportItem, ExportKeys,
    ExportPayload, FileId, FinalizationClaim, FinalizationLease, LeaseKind, LeaseTiming, OperationId, PreviousExport,
    PublishRequest, SessionId, Sha256Digest, StoredObject, TerminalStatus, UnixMillis, WorkerId,
};
use super::common::{apply_quota, count_batch_items, end_phase, lock_batch, lock_quota, read_batch, set_batch};
use super::convert::{artifact, batch_ref, digest};
use super::driver::{
    bin, bytes, corrupt, int, mtx, opt_bin, opt_int, opt_text, text, Core, Tx, BATCHES, CLAIMS, ERRORS, FILES, ITEMS,
    OPERATIONS, SESSIONS, SETTLEMENTS,
};
use super::ingest::take_phase;
use super::lifecycle::{items_of, refs};

struct ExportSession {
    id: SessionId,
    keys: ExportKeys,
    generation: i64,
    lease_until: i64,
    reserved: i64,
}

async fn export_session(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<Option<ExportSession>, StoreError> {
    let Some(doc) = tx
        .one_sorted(
            SESSIONS,
            doc! { "owner_id": batch.owner.as_str(), "batch_id": batch.id.as_str(), "kind": "export" },
            doc! { "generation": -1_i32 },
        )
        .await?
    else {
        return Ok(None);
    };
    Ok(Some(ExportSession {
        id: SessionId::parse(&text(&doc, "_id")?).map_err(|_| corrupt("session id"))?,
        keys: ExportKeys {
            output: artifact(text(&doc, "artifact_ref")?)?,
            error: artifact(opt_text(&doc, "secondary_ref")?.ok_or(corrupt("export error key"))?)?,
        },
        generation: int(&doc, "generation")?,
        lease_until: int(&doc, "lease_until")?,
        reserved: int(&doc, "reserved_bytes")?,
    }))
}

async fn insert_export_session(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    keys: &ExportKeys,
    reserved: i64,
    lease_until: i64,
    now: i64,
) -> Result<SessionId, StoreError> {
    let session = SessionId::new_random();
    tx.insert(
        SESSIONS,
        doc! {
            "_id": session.as_str(),
            "owner_id": token.batch.owner.as_str(),
            "kind": "export",
            "artifact_ref": keys.output.as_str(),
            "secondary_ref": keys.error.as_str(),
            "generation": token.generation,
            "phase": "open",
            "lease_until": lease_until,
            "reserved_bytes": reserved,
            "max_bytes": 0_i64,
            "file_id": Bson::Null,
            "batch_id": token.batch.id.as_str(),
            "claim_id": Bson::Null,
            "line_no": Bson::Null,
            "filename": Bson::Null,
            "expires_after_secs": Bson::Null,
            "created_at": now,
        },
    )
    .await?;
    Ok(session)
}

async fn claim_finalization_tx(
    tx: &mut Tx<'_>,
    worker: &WorkerId,
    keys: &ExportKeys,
    now: i64,
) -> Result<FinalizationClaim, StoreError> {
    let settings = *tx.settings();
    let Some(candidate) = tx
        .one_sorted(
            BATCHES,
            doc! {
                "status": "finalizing",
                "accounting_pending": 0_i64,
                "$or": [{ "phase_claim_id": Bson::Null }, { "phase_lease_until": { "$lt": now } }],
            },
            doc! { "finalizing_at": 1_i32 },
        )
        .await?
    else {
        return Ok(FinalizationClaim::Idle);
    };
    let (batch, now) = lock_batch(tx, &batch_ref(&candidate)?).await?;
    if batch.record.status != BatchStatus::Finalizing || batch.accounting_pending != 0 || !phase_free(&batch, now) {
        return Ok(FinalizationClaim::Idle);
    }
    let target = batch.record.terminal_target.ok_or(corrupt("finalizing batch without target"))?;
    let (token, timing) =
        take_phase(tx, &batch, LeaseKind::Finalization, worker, settings.finalization_lease, now).await?;
    if let Some(previous) = export_session(tx, &token.batch).await? {
        return Ok(FinalizationClaim::NeedsFencing {
            token,
            timing,
            previous: PreviousExport {
                session: previous.id,
                keys: previous.keys,
                writer_stopped_after: UnixMillis(previous.lease_until.saturating_add(millis(settings.io_guard))),
            },
        });
    }
    let files = batch.object_charge.saturating_mul(2);
    let expired_lines = if target == TerminalStatus::Expired {
        let mut expired = items_of(&token.batch);
        expired.insert("state", "expired");
        let reserves = tx.sum(ITEMS, expired.clone(), "record_reserve").await?;
        let count = tx.count(ITEMS, expired).await?;
        reserves - count * batch.item_overhead
    } else {
        0
    };
    let from_completion = files.saturating_add(expired_lines);
    if from_completion > batch.completion_reserved {
        return Err(corrupt("export reserve exceeds the completion reserve"));
    }
    let reserved = batch.export_reserved.saturating_add(from_completion);
    insert_export_session(tx, &token, keys, reserved, timing.lease_until.0, now).await?;
    set_batch(
        tx,
        &token.batch,
        doc! { "$inc": { "completion_reserved": -from_completion }, "$set": { "export_reserved": 0_i64 } },
    )
    .await?;
    Ok(FinalizationClaim::Ready(FinalizationLease {
        token,
        timing,
        target,
        keys: keys.clone(),
        total: batch.record.counts.total,
    }))
}

/// Take the finalization lease of a batch whose records are final and accounted.
pub(super) async fn claim_finalization(
    core: &Core,
    worker: &WorkerId,
    keys: &ExportKeys,
) -> Result<FinalizationClaim, StoreError> {
    let now = core.server_now().await?;
    mtx!(core, tx => claim_finalization_tx(tx, worker, keys, now))
}

async fn lock_finalizing(tx: &mut Tx<'_>, token: &ClaimToken) -> Result<(BatchRow, i64), StoreError> {
    let (batch, now) = lock_batch(tx, &token.batch).await?;
    if batch.record.status != BatchStatus::Finalizing || !phase_lease_valid(&batch, token, LeaseKind::Finalization, now)
    {
        return Err(StoreError::LeaseStale);
    }
    Ok((batch, now))
}

async fn adopt_export_tx(
    tx: &mut Tx<'_>,
    token: &ClaimToken,
    previous: &SessionId,
    keys: &ExportKeys,
) -> Result<FinalizationLease, StoreError> {
    let io_guard = millis(tx.settings().io_guard);
    let (batch, now) = lock_finalizing(tx, token).await?;
    let old = export_session(tx, &token.batch).await?.ok_or(StoreError::LeaseStale)?;
    if &old.id != previous || old.generation >= token.generation {
        return Err(StoreError::LeaseStale);
    }
    if old.lease_until.saturating_add(io_guard) > now {
        return Err(StoreError::FencingPending);
    }
    let lease_until = batch.phase_lease_until.unwrap_or(now);
    insert_export_session(tx, token, keys, old.reserved, lease_until, now).await?;
    tx.delete_one(SESSIONS, doc! { "_id": old.id.as_str() }).await?;
    Ok(FinalizationLease {
        token: token.clone(),
        timing: LeaseTiming {
            server_now: UnixMillis(now),
            lease_until: UnixMillis(lease_until),
            batch_expires_at: batch.record.expires_at,
        },
        target: batch.record.terminal_target.ok_or(corrupt("finalizing batch without target"))?,
        keys: keys.clone(),
        total: batch.record.counts.total,
    })
}

/// Move the reservation of an earlier export generation after its objects were deleted and its writer is cut off.
pub(super) async fn adopt_export(
    core: &Core,
    token: &ClaimToken,
    previous: &SessionId,
    keys: &ExportKeys,
) -> Result<FinalizationLease, StoreError> {
    mtx!(core, tx => adopt_export_tx(tx, token, previous, keys))
}

fn export_item(doc: &Document) -> Result<ExportItem, StoreError> {
    let line_no = super::driver::uint(doc, "line_no")?;
    match text(doc, "state")?.as_str() {
        "expired" => {
            return Ok(ExportItem::Expired {
                line_no,
                custom_id: CustomId::from_bytes(bin(doc, "custom_id")?).map_err(|_| corrupt("custom_id"))?,
            });
        }
        "cancelled" => return Ok(ExportItem::Cancelled { line_no }),
        "succeeded" | "failed" => {}
        _ => return Err(corrupt("unfinished item in a finalizing batch")),
    }
    let payload = match (opt_bin(doc, "result_inline")?, opt_text(doc, "result_ref")?) {
        (Some(inline), None) => ExportPayload::Inline(inline),
        (None, Some(object_ref)) => ExportPayload::Stored {
            artifact: artifact(object_ref)?,
            object: StoredObject {
                bytes: unsigned(opt_int(doc, "result_bytes")?.ok_or(corrupt("result bytes"))?)?,
                sha256: digest(opt_bin(doc, "result_sha256")?.ok_or(corrupt("result sha256"))?)?,
            },
        },
        _ => return Err(corrupt("item result payload")),
    };
    match opt_text(doc, "result_kind")?.as_deref() {
        Some("output") => Ok(ExportItem::Output { line_no, payload }),
        Some("error") => Ok(ExportItem::Error { line_no, payload }),
        _ => Err(corrupt("item result kind")),
    }
}

/// Committed records for the export writer in line order (keyset paging, bounded). Read-only: the lease is checked
/// against the primary clock; items of a finalizing batch no longer change.
pub(super) async fn read_export_items(
    core: &Core,
    token: &ClaimToken,
    after_line: u64,
    limit: u32,
) -> Result<Vec<ExportItem>, StoreError> {
    let now = core.server_now().await?;
    let doc = core
        .read_one(BATCHES, doc! { "_id": token.batch.id.as_str(), "owner_id": token.batch.owner.as_str() })
        .await?
        .ok_or(StoreError::NotFound)?;
    let batch = super::convert::batch_row(&doc)?;
    if batch.record.status != BatchStatus::Finalizing || !phase_lease_valid(&batch, token, LeaseKind::Finalization, now)
    {
        return Err(StoreError::LeaseStale);
    }
    let mut filter = items_of(&token.batch);
    filter.insert("line_no", doc! { "$gt": signed(after_line)? });
    let docs = core.read_many(ITEMS, filter, doc! { "line_no": 1_i32 }, i64::from(limit.clamp(1, 4096))).await?;
    docs.iter().map(export_item).collect()
}

#[expect(clippy::too_many_arguments, reason = "one file document with its fields")]
async fn insert_result_file(
    tx: &mut Tx<'_>,
    batch: &BatchRef,
    purpose: &str,
    object_ref: &ArtifactRef,
    filename: &str,
    object: StoredObject,
    charge: i64,
    now: i64,
    expires_at: i64,
) -> Result<(FileId, i64), StoreError> {
    let id = FileId::new_random();
    let size = signed(object.bytes)?;
    let charged = size.saturating_add(charge);
    tx.insert(
        FILES,
        doc! {
            "_id": id.as_str(),
            "owner_id": batch.owner.as_str(),
            "purpose": purpose,
            "filename": filename,
            "artifact_ref": object_ref.as_str(),
            "bytes": size,
            "charged_bytes": charged,
            "sha256": bytes(&object.sha256.0),
            "created_at": now,
            "expires_at": expires_at,
            "state": "active",
            "pin_count": 0_i64,
            "deleted_at": Bson::Null,
            "gc_after": Bson::Null,
            "batch_id": batch.id.as_str(),
        },
    )
    .await?;
    Ok((id, charged))
}

async fn publish_files_tx(
    tx: &mut Tx<'_>,
    operation: &OperationId,
    token: &ClaimToken,
    request: &PublishRequest,
    digest_value: Sha256Digest,
) -> Result<BatchRecord, StoreError> {
    let settings = *tx.settings();
    let owner = token.batch.owner.as_str();
    if let Some(recorded) =
        tx.one(OPERATIONS, doc! { "owner_id": owner, "kind": "publish", "operation_id": operation.as_str() }).await?
    {
        if bin(&recorded, "request_sha256")? != digest_value.0 {
            return Err(StoreError::IdempotencyConflict);
        }
        return read_batch(tx, &token.batch).await?.map(|b| b.record).ok_or(StoreError::NotFound);
    }
    lock_quota(tx, &token.batch.owner).await?;
    let (batch, now) = lock_finalizing(tx, token).await?;
    if batch.accounting_pending != 0 {
        return Err(StoreError::PublishRejected("settlements are still pending"));
    }
    let target = batch.record.terminal_target.ok_or(corrupt("finalizing batch without target"))?;
    let session = export_session(tx, &token.batch).await?.ok_or(StoreError::LeaseStale)?;
    if session.generation != token.generation {
        return Err(StoreError::LeaseStale);
    }
    let charge = batch.object_charge;
    let needed = [&request.output, &request.error]
        .iter()
        .filter_map(|f| f.as_ref())
        .map(|f| signed(f.object.bytes).map(|b| b.saturating_add(charge)))
        .sum::<Result<i64, _>>()?;
    if needed > session.reserved {
        return Err(StoreError::PublishRejected("export exceeds its reservation"));
    }
    let retention = batch.record.output_expires_after.map_or(millis(settings.file_retention), millis);
    let expires_at = now.saturating_add(retention);
    let (mut used, mut published) = (0i64, 0i64);
    let mut output_id = None;
    let mut error_id = None;
    if let Some(file) = &request.output {
        let (id, charged) = insert_result_file(
            tx,
            &token.batch,
            "batch_output",
            &session.keys.output,
            &file.filename,
            file.object,
            charge,
            now,
            expires_at,
        )
        .await?;
        output_id = Some(id);
        used += charged;
        published += 1;
    }
    if let Some(file) = &request.error {
        let (id, charged) = insert_result_file(
            tx,
            &token.batch,
            "batch_output",
            &session.keys.error,
            &file.filename,
            file.object,
            charge,
            now,
            expires_at,
        )
        .await?;
        error_id = Some(id);
        used += charged;
        published += 1;
    }
    let items = count_batch_items(tx, &token.batch).await?;
    let completion = items.saturating_mul(batch.item_overhead).min(batch.completion_reserved);
    let completion_release = batch.completion_reserved - completion;
    let metadata_expires_at = now.saturating_add(millis(settings.metadata_retention)).max(expires_at);
    end_phase(tx, &batch).await?;
    let changed = tx
        .update(
            BATCHES,
            doc! { "_id": token.batch.id.as_str(), "owner_id": owner, "released": false },
            doc! { "$set": {
                "status": target.as_db(),
                "terminal_at": now,
                "output_file_id": super::driver::opt_str(output_id.as_ref().map(FileId::as_str)),
                "error_file_id": super::driver::opt_str(error_id.as_ref().map(FileId::as_str)),
                "released": true,
                "cleanup_state": "pending",
                "metadata_expires_at": metadata_expires_at,
                "completion_reserved": completion,
                "export_reserved": 0_i64,
            } },
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("publication lost its batch"));
    }
    tx.update(
        FILES,
        doc! { "_id": batch.record.input_file_id.as_str(), "owner_id": owner, "pin_count": { "$gt": 0_i64 } },
        doc! { "$inc": { "pin_count": -1_i64 } },
    )
    .await?;
    tx.delete_one(SESSIONS, doc! { "_id": session.id.as_str() }).await?;
    tx.insert(
        OPERATIONS,
        doc! {
            "owner_id": owner,
            "kind": "publish",
            "operation_id": operation.as_str(),
            "request_sha256": bytes(&digest_value.0),
            "batch_id": token.batch.id.as_str(),
            "outcome": "published",
            "created_at": now,
        },
    )
    .await?;
    apply_quota(
        tx,
        &token.batch.owner,
        QuotaDelta {
            used,
            reserved: -(session.reserved + completion_release),
            active: -1,
            files: published - 2,
            ..QuotaDelta::default()
        },
    )
    .await?;
    Ok(read_batch(tx, &token.batch).await?.ok_or(corrupt("published batch"))?.record)
}

/// Publish the export files and close the batch in one terminal transaction (quota → batch → files).
pub(super) async fn publish_files(
    core: &Core,
    operation: &OperationId,
    token: &ClaimToken,
    request: &PublishRequest,
    digest_value: Sha256Digest,
) -> Result<BatchRecord, StoreError> {
    mtx!(core, tx => publish_files_tx(tx, operation, token, request, digest_value))
}

pub(super) async fn cleanup_batches(core: &Core, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let docs = core
        .read_many(
            BATCHES,
            doc! { "cleanup_state": "pending" },
            doc! { "cleanup_state": 1_i32, "terminal_at": 1_i32 },
            i64::from(limit),
        )
        .await?;
    refs(&docs)
}

/// Control ledgers may contain many generations for each item; bound every terminal transaction separately.
const LEDGER_PAGE: i64 = 512;

async fn delete_ledger_page(tx: &mut Tx<'_>, collection: &str, scope: Document) -> Result<bool, StoreError> {
    let page = tx
        .aggregate(
            collection,
            vec![doc! { "$match": scope }, doc! { "$project": { "_id": 1_i32 } }, doc! { "$limit": LEDGER_PAGE }],
        )
        .await?;
    if page.is_empty() {
        return Ok(false);
    }
    let ids = page
        .iter()
        .map(|row| row.get("_id").cloned().ok_or(corrupt("control ledger id")))
        .collect::<Result<Vec<_>, StoreError>>()?;
    tx.delete_many(collection, doc! { "_id": { "$in": ids } }).await?;
    Ok(true)
}

/// Next bounded page of item documents to delete; the caller deletes `artifacts` before `cleanup_commit`.
pub(super) async fn cleanup_page(core: &Core, batch: &BatchRef, limit: u32) -> Result<CleanupPage, StoreError> {
    let row = core
        .read_one(BATCHES, doc! { "_id": batch.id.as_str(), "owner_id": batch.owner.as_str() })
        .await?
        .ok_or(StoreError::NotFound)?;
    if text(&row, "cleanup_state")? != "pending" {
        return Ok(CleanupPage { artifacts: Vec::new(), through_line_no: None });
    }
    let docs =
        core.read_many(ITEMS, items_of(batch), doc! { "line_no": 1_i32 }, i64::from(limit.clamp(1, 4096))).await?;
    let mut artifacts = Vec::new();
    let mut through = None;
    for doc in &docs {
        through = Some(super::driver::uint(doc, "line_no")?);
        if let Some(object_ref) = opt_text(doc, "result_ref")? {
            artifacts.push(artifact(object_ref)?);
        }
    }
    Ok(CleanupPage { artifacts, through_line_no: through })
}

async fn cleanup_commit_tx(
    tx: &mut Tx<'_>,
    batch: &BatchRef,
    through_line_no: Option<u64>,
) -> Result<bool, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let (row, _) = lock_batch(tx, batch).await?;
    if row.cleanup_state != "pending" || !row.released {
        return Ok(row.cleanup_state == "done");
    }
    if let Some(through) = through_line_no {
        let mut page = items_of(batch);
        page.insert("line_no", doc! { "$lte": signed(through)? });
        let count = tx.count(ITEMS, page.clone()).await?;
        let used = tx.sum(ITEMS, page.clone(), "quota_used").await?;
        tx.delete_many(ITEMS, page).await?;
        let overhead = count.saturating_mul(row.item_overhead).min(row.completion_reserved);
        set_batch(tx, batch, doc! { "$inc": { "completion_reserved": -overhead } }).await?;
        apply_quota(tx, &batch.owner, QuotaDelta { used: -used, reserved: -overhead, ..QuotaDelta::default() }).await?;
    }
    if count_batch_items(tx, batch).await? > 0 || row.accounting_pending > 0 {
        return Ok(false);
    }
    let scope = items_of(batch);
    let mut internal = scope.clone();
    internal.insert("kind", doc! { "$in": ["commit", "publish"] });
    // Short circuit after a nonempty page: one transaction removes at most LEDGER_PAGE control rows.
    if delete_ledger_page(tx, SETTLEMENTS, scope.clone()).await?
        || delete_ledger_page(tx, CLAIMS, scope).await?
        || delete_ledger_page(tx, OPERATIONS, internal).await?
    {
        return Ok(false);
    }
    set_batch(tx, batch, doc! { "$set": { "cleanup_state": "done" } }).await?;
    Ok(true)
}

/// Delete item documents up to `through_line_no` (objects already deleted) and release their bytes; with `None` and
/// no items left, delete a bounded control-ledger page; mark done only once all ledgers are empty.
pub(super) async fn cleanup_commit(
    core: &Core,
    batch: &BatchRef,
    through_line_no: Option<u64>,
) -> Result<bool, StoreError> {
    mtx!(core, tx => cleanup_commit_tx(tx, batch, through_line_no))
}

async fn purge_metadata_tx(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<bool, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let Some((row, now)) = super::common::try_lock_batch(tx, batch).await? else { return Ok(false) };
    if row.cleanup_state != "done" || row.metadata_expires_at.is_none_or(|at| at > now) {
        return Ok(false);
    }
    let scope = items_of(batch);
    if tx.count(SESSIONS, scope.clone()).await? > 0 {
        return Ok(false);
    }
    // Keep the batch and its metadata charge until all pages are gone; retries release quota exactly once.
    if delete_ledger_page(tx, ERRORS, scope.clone()).await? || delete_ledger_page(tx, OPERATIONS, scope).await? {
        return Ok(false);
    }
    tx.delete_one(BATCHES, doc! { "_id": batch.id.as_str(), "owner_id": batch.owner.as_str() }).await?;
    apply_quota(
        tx,
        &batch.owner,
        QuotaDelta {
            used: -row.errors_bytes,
            reserved: -(row.completion_reserved + row.export_reserved),
            retained: -1,
            ..QuotaDelta::default()
        },
    )
    .await?;
    Ok(true)
}

/// Purge metadata (with Idempotency-Key entries) of at most `limit` batches past their retention.
pub(super) async fn purge_metadata(core: &Core, limit: u32) -> Result<u64, StoreError> {
    let now = core.server_now().await?;
    let docs = core
        .read_many(
            BATCHES,
            doc! { "cleanup_state": "done", "metadata_expires_at": { "$lte": now } },
            doc! { "metadata_expires_at": 1_i32 },
            i64::from(limit),
        )
        .await?;
    let mut purged = 0;
    for batch in refs(&docs)? {
        if mtx!(core, tx => purge_metadata_tx(tx, &batch))? {
            purged += 1;
        }
    }
    Ok(purged)
}
