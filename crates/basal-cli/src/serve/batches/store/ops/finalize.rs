//! Finalization lease with export reservation and fencing of earlier generations, export reads, publication with
//! the terminal transaction, post-publication cleanup and metadata retention.

use super::super::error::StoreError;
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    ArtifactRef, BatchRecord, BatchRef, BatchStatus, ClaimToken, CleanupPage, CustomId, ExportItem, ExportKeys,
    ExportPayload, FileId, FinalizationClaim, FinalizationLease, LeaseKind, LeaseTiming, OperationId, PreviousExport,
    PublishRequest, SessionId, Sha256Digest, StoreSettings, StoredObject, TerminalStatus, UnixMillis, WorkerId,
};
use super::common::{
    apply_quota, artifact_from_db, batch_row, corrupt, digest_from_db, end_phase, load_batch, lock_batch, lock_quota,
    millis, phase_free, phase_lease_valid, scalar, signed, unsigned, BatchRow, QuotaDelta, BATCH_COLUMNS,
};
use super::ingest::take_phase;
use super::lifecycle::refs;

struct ExportSession {
    id: SessionId,
    keys: ExportKeys,
    generation: i64,
    lease_until: i64,
    reserved: i64,
}

async fn export_session<T: SqlTx>(tx: &mut T, batch: &BatchRef) -> Result<Option<ExportSession>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}id, artifact_ref, secondary_ref, generation, lease_until, reserved_bytes FROM artifact_sessions \
         WHERE owner_id = $1 AND batch_id = $2 AND kind = 'export' ORDER BY generation DESC{}",
        d.top("1"),
        d.limit("1")
    );
    let Some(row) = tx.fetch_optional(&sql, args![batch.owner.as_str(), batch.id.as_str()]).await? else {
        return Ok(None);
    };
    Ok(Some(ExportSession {
        id: SessionId::parse(&row.text_at(0)?).map_err(|_| corrupt("session id"))?,
        keys: ExportKeys {
            output: artifact_from_db(row.text_at(1)?)?,
            error: artifact_from_db(row.opt_text_at(2)?.ok_or(corrupt("export error key"))?)?,
        },
        generation: row.i64_at(3)?,
        lease_until: row.i64_at(4)?,
        reserved: row.i64_at(5)?,
    }))
}

async fn insert_export_session<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    keys: &ExportKeys,
    reserved: i64,
    lease_until: i64,
    now: i64,
) -> Result<SessionId, StoreError> {
    let session = SessionId::new_random();
    tx.execute(
        "INSERT INTO artifact_sessions (id, owner_id, kind, artifact_ref, secondary_ref, generation, phase, \
         lease_until, reserved_bytes, batch_id, created_at) VALUES ($1, $2, 'export', $3, $4, $5, 'open', $6, $7, $8, $9)",
        args![
            session.as_str(),
            token.batch.owner.as_str(),
            keys.output.as_str(),
            keys.error.as_str(),
            token.generation,
            lease_until,
            reserved,
            token.batch.id.as_str(),
            now
        ],
    )
    .await?;
    Ok(session)
}

/// Take the finalization lease of a batch whose records are all final and accounted. The first generation moves
/// the export reserve (accepted lines, expiry lines when expired, file charges) from the batch to its export
/// session; a later generation must first fence the earlier writer (see [`adopt_export`]).
pub(in super::super) async fn claim_finalization<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    worker: &WorkerId,
    keys: &ExportKeys,
) -> Result<FinalizationClaim, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}{BATCH_COLUMNS} FROM batches{} WHERE status = 'finalizing' AND accounting_pending = 0 \
         AND (phase_claim_id IS NULL OR phase_lease_until < $1) ORDER BY finalizing_at{}{}",
        d.top("1"),
        d.skip_hint,
        d.limit("1"),
        d.skip_locked
    );
    let Some(row) = tx.fetch_optional(&sql, args![now]).await? else { return Ok(FinalizationClaim::Idle) };
    let batch = batch_row(&row)?;
    let now = db_now(tx).await?;
    if !phase_free(&batch, now) {
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
        scalar(
            tx,
            &format!(
                "SELECT CAST(COALESCE(SUM(record_reserve), 0) AS BIGINT) - {} * $3 FROM batch_items \
                 WHERE owner_id = $1 AND batch_id = $2 AND state = 'expired'",
                d.count_all
            ),
            args![batch.record.owner.as_str(), batch.record.id.as_str(), batch.item_overhead],
        )
        .await?
    } else {
        0
    };
    let from_completion = files.saturating_add(expired_lines);
    if from_completion > batch.completion_reserved {
        return Err(corrupt("export reserve exceeds the completion reserve"));
    }
    let reserved = batch.export_reserved.saturating_add(from_completion);
    insert_export_session(tx, &token, keys, reserved, timing.lease_until.0, now).await?;
    tx.execute(
        "UPDATE batches SET completion_reserved = completion_reserved - $3, export_reserved = 0 \
         WHERE owner_id = $1 AND id = $2",
        args![batch.record.owner.as_str(), batch.record.id.as_str(), from_completion],
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

async fn lock_finalizing<T: SqlTx>(tx: &mut T, token: &ClaimToken) -> Result<(BatchRow, i64), StoreError> {
    let batch = lock_batch(tx, &token.batch).await?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::Finalizing || !phase_lease_valid(&batch, token, LeaseKind::Finalization, now)
    {
        return Err(StoreError::LeaseStale);
    }
    Ok((batch, now))
}

/// Move the reservation of an earlier export generation to this lease. The caller has deleted the earlier
/// objects; the move is refused before `writer_stopped_after`, so one reserve never covers two writers.
pub(in super::super) async fn adopt_export<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    token: &ClaimToken,
    previous: &SessionId,
    keys: &ExportKeys,
) -> Result<FinalizationLease, StoreError> {
    let (batch, now) = lock_finalizing(tx, token).await?;
    let old = export_session(tx, &token.batch).await?.ok_or(StoreError::LeaseStale)?;
    if &old.id != previous || old.generation >= token.generation {
        return Err(StoreError::LeaseStale);
    }
    if old.lease_until.saturating_add(millis(settings.io_guard)) > now {
        return Err(StoreError::FencingPending);
    }
    let lease_until = batch.phase_lease_until.unwrap_or(now);
    insert_export_session(tx, token, keys, old.reserved, lease_until, now).await?;
    tx.execute("DELETE FROM artifact_sessions WHERE id = $1", args![old.id.as_str()]).await?;
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

/// Committed records for the export writer in line order (keyset paging, bounded).
pub(in super::super) async fn read_export_items<T: SqlTx>(
    tx: &mut T,
    token: &ClaimToken,
    after_line: u64,
    limit: u32,
) -> Result<Vec<ExportItem>, StoreError> {
    let batch = load_batch(tx, &token.batch.owner, &token.batch.id, false).await?.ok_or(StoreError::NotFound)?;
    let now = db_now(tx).await?;
    if batch.record.status != BatchStatus::Finalizing || !phase_lease_valid(&batch, token, LeaseKind::Finalization, now)
    {
        return Err(StoreError::LeaseStale);
    }
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}line_no, state, custom_id, result_kind, result_inline, result_ref, result_bytes, result_sha256 \
         FROM batch_items WHERE owner_id = $1 AND batch_id = $2 AND line_no > $3 ORDER BY line_no{}",
        d.top("$4"),
        d.limit("$4")
    );
    let rows = tx
        .fetch_all(
            &sql,
            args![
                token.batch.owner.as_str(),
                token.batch.id.as_str(),
                signed(after_line)?,
                i64::from(limit.clamp(1, 4096))
            ],
        )
        .await?;
    rows.iter()
        .map(|row| {
            let line_no = row.u64_at(0)?;
            let state = row.text_at(1)?;
            match state.as_str() {
                "expired" => {
                    return Ok(ExportItem::Expired {
                        line_no,
                        custom_id: CustomId::from_bytes(row.blob_at(2)?).map_err(|_| corrupt("custom_id"))?,
                    });
                }
                "cancelled" => return Ok(ExportItem::Cancelled { line_no }),
                "succeeded" | "failed" => {}
                _ => return Err(corrupt("unfinished item in a finalizing batch")),
            }
            let payload = match (row.opt_blob_at(4)?, row.opt_text_at(5)?) {
                (Some(inline), None) => ExportPayload::Inline(inline),
                (None, Some(artifact)) => ExportPayload::Stored {
                    artifact: artifact_from_db(artifact)?,
                    object: StoredObject {
                        bytes: unsigned(row.opt_i64_at(6)?.ok_or(corrupt("result bytes"))?)?,
                        sha256: digest_from_db(row.opt_blob_at(7)?.ok_or(corrupt("result sha256"))?)?,
                    },
                },
                _ => return Err(corrupt("item result payload")),
            };
            match row.opt_text_at(3)?.as_deref() {
                Some("output") => Ok(ExportItem::Output { line_no, payload }),
                Some("error") => Ok(ExportItem::Error { line_no, payload }),
                _ => Err(corrupt("item result kind")),
            }
        })
        .collect()
}

#[expect(clippy::too_many_arguments, reason = "one file row insert with its columns")]
async fn insert_result_file<T: SqlTx>(
    tx: &mut T,
    owner: &str,
    batch_id: &str,
    purpose: &str,
    artifact: &ArtifactRef,
    filename: &str,
    object: StoredObject,
    charge: i64,
    now: i64,
    expires_at: i64,
) -> Result<(FileId, i64), StoreError> {
    let id = FileId::new_random();
    let bytes = signed(object.bytes)?;
    let charged = bytes.saturating_add(charge);
    tx.execute(
        "INSERT INTO batch_files (owner_id, id, purpose, filename, artifact_ref, bytes, charged_bytes, sha256, \
         created_at, expires_at, state, pin_count, batch_id) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'active', 0, $11)",
        args![
            owner,
            id.as_str(),
            purpose,
            filename,
            artifact.as_str(),
            bytes,
            charged,
            object.sha256.0.to_vec(),
            now,
            expires_at,
            batch_id
        ],
    )
    .await?;
    Ok((id, charged))
}

/// Publish the export files and close the batch in one terminal transaction (quota → batch → files). Requires the
/// current finalization lease and `accounting_pending = 0`. Takes no new quota: file charges come from the export
/// session; the rest of the completion reserve except item row charges is released.
pub(in super::super) async fn publish_files<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    operation: &OperationId,
    token: &ClaimToken,
    request: &PublishRequest,
    digest: Sha256Digest,
) -> Result<BatchRecord, StoreError> {
    let owner = token.batch.owner.as_str();
    let id = token.batch.id.as_str();
    if let Some(row) = tx
        .fetch_optional(
            &format!(
                "SELECT request_sha256 FROM batch_operations WHERE owner_id = $1 AND kind = 'publish' AND {}",
                tx.dialect().exact_eq("operation_id", "$2")
            ),
            args![owner, operation.as_str()],
        )
        .await?
    {
        if row.blob_at(0)? != digest.0 {
            return Err(StoreError::IdempotencyConflict);
        }
        return load_batch(tx, &token.batch.owner, &token.batch.id, false)
            .await?
            .map(|b| b.record)
            .ok_or(StoreError::NotFound);
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
    let mut used = 0i64;
    let mut published = 0i64;
    let mut output_id = None;
    let mut error_id = None;
    if let Some(file) = &request.output {
        let (file_id, charged) = insert_result_file(
            tx,
            owner,
            id,
            "batch_output",
            &session.keys.output,
            &file.filename,
            file.object,
            charge,
            now,
            expires_at,
        )
        .await?;
        output_id = Some(file_id);
        used += charged;
        published += 1;
    }
    if let Some(file) = &request.error {
        // Public purpose of both result files is `batch_output` (OpenAI Files contract); the batch tells them apart
        // through `output_file_id` / `error_file_id`. Rows written as `batch_error` earlier still decode.
        let (file_id, charged) = insert_result_file(
            tx,
            owner,
            id,
            "batch_output",
            &session.keys.error,
            &file.filename,
            file.object,
            charge,
            now,
            expires_at,
        )
        .await?;
        error_id = Some(file_id);
        used += charged;
        published += 1;
    }
    let count_all = tx.dialect().count_all;
    let items = scalar(
        tx,
        &format!("SELECT {count_all} FROM batch_items WHERE owner_id = $1 AND batch_id = $2"),
        args![owner, id],
    )
    .await?;
    let completion = items.saturating_mul(batch.item_overhead).min(batch.completion_reserved);
    let completion_release = batch.completion_reserved - completion;
    let metadata_expires_at = now.saturating_add(millis(settings.metadata_retention)).max(expires_at);
    end_phase(tx, &batch).await?;
    let changed = tx
        .execute(
            "UPDATE batches SET status = $3, terminal_at = $4, output_file_id = $5, error_file_id = $6, released = 1, \
             cleanup_state = 'pending', metadata_expires_at = $7, completion_reserved = $8, export_reserved = 0 \
             WHERE owner_id = $1 AND id = $2 AND released = 0",
            args![
                owner,
                id,
                target.as_db(),
                now,
                output_id.as_ref().map(|f| f.as_str().to_owned()),
                error_id.as_ref().map(|f| f.as_str().to_owned()),
                metadata_expires_at,
                completion
            ],
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("publication lost its batch"));
    }
    tx.execute(
        "UPDATE batch_files SET pin_count = pin_count - 1 WHERE owner_id = $1 AND id = $2 AND pin_count > 0",
        args![owner, batch.record.input_file_id.as_str()],
    )
    .await?;
    tx.execute("DELETE FROM artifact_sessions WHERE id = $1", args![session.id.as_str()]).await?;
    tx.execute(
        "INSERT INTO batch_operations (owner_id, kind, operation_id, request_sha256, batch_id, outcome, created_at) \
         VALUES ($1, 'publish', $2, $3, $4, 'published', $5)",
        args![owner, operation.as_str(), digest.0.to_vec(), id, now],
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
    Ok(load_batch(tx, &token.batch.owner, &token.batch.id, false).await?.ok_or(corrupt("published batch"))?.record)
}

// ---------------------------------------------------------------------------------------------------------------
// Cleanup and retention
// ---------------------------------------------------------------------------------------------------------------

pub(in super::super) async fn cleanup_batches<T: SqlTx>(tx: &mut T, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}owner_id, id FROM batches WHERE cleanup_state = 'pending' ORDER BY terminal_at{}",
        d.top("$1"),
        d.limit("$1")
    );
    let rows = tx.fetch_all(&sql, args![i64::from(limit)]).await?;
    refs(&rows)
}

/// Each terminal control transaction removes at most this many ledger rows, regardless of retry history.
const LEDGER_PAGE: u32 = 512;

/// Remove one bounded page selected by the full primary key. Empty tables act as durable phase markers:
/// settlements precede claims, then internal operations; no crash-sensitive cursor is needed.
async fn delete_ledger_page<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRef,
    table: &str,
    columns: &[&str],
    internal_only: bool,
) -> Result<bool, StoreError> {
    let predicate = if internal_only {
        "owner_id = $1 AND batch_id = $2 AND kind IN ('commit', 'publish')"
    } else {
        "owner_id = $1 AND batch_id = $2"
    };
    let sql = tx.dialect().delete_page(table, columns, predicate, LEDGER_PAGE);
    Ok(tx.execute(&sql, args![batch.owner.as_str(), batch.id.as_str()]).await? > 0)
}

/// Next bounded page of item rows to delete; the caller deletes `artifacts` before `cleanup_commit`.
pub(in super::super) async fn cleanup_page<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRef,
    limit: u32,
) -> Result<CleanupPage, StoreError> {
    let row = load_batch(tx, &batch.owner, &batch.id, false).await?.ok_or(StoreError::NotFound)?;
    if row.cleanup_state != "pending" {
        return Ok(CleanupPage { artifacts: Vec::new(), through_line_no: None });
    }
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}line_no, result_ref FROM batch_items WHERE owner_id = $1 AND batch_id = $2 ORDER BY line_no{}",
        d.top("$3"),
        d.limit("$3")
    );
    let rows =
        tx.fetch_all(&sql, args![batch.owner.as_str(), batch.id.as_str(), i64::from(limit.clamp(1, 4096))]).await?;
    let mut artifacts = Vec::new();
    let mut through = None;
    for row in &rows {
        through = Some(row.u64_at(0)?);
        if let Some(artifact) = row.opt_text_at(1)? {
            artifacts.push(artifact_from_db(artifact)?);
        }
    }
    Ok(CleanupPage { artifacts, through_line_no: through })
}

/// Delete item rows up to `through_line_no` (their objects are already gone) and release their bytes; with `None`
/// and no rows left, delete a bounded control-ledger page, marking done only after every ledger is empty.
/// Returns true when cleanup of the batch is complete. Safe to repeat after a crash.
pub(in super::super) async fn cleanup_commit<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRef,
    through_line_no: Option<u64>,
) -> Result<bool, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let row = lock_batch(tx, batch).await?;
    if row.cleanup_state != "pending" || !row.released {
        return Ok(row.cleanup_state == "done");
    }
    let owner = batch.owner.as_str();
    let id = batch.id.as_str();
    let count_all = tx.dialect().count_all;
    if let Some(through) = through_line_no {
        let through = signed(through)?;
        let agg = tx
            .fetch_optional(
                &format!(
                    "SELECT {count_all}, CAST(COALESCE(SUM(quota_used), 0) AS BIGINT) FROM batch_items \
                     WHERE owner_id = $1 AND batch_id = $2 AND line_no <= $3"
                ),
                args![owner, id, through],
            )
            .await?
            .ok_or(corrupt("aggregate returned no row"))?;
        let (count, used) = (agg.i64_at(0)?, agg.i64_at(1)?);
        tx.execute(
            "DELETE FROM batch_items WHERE owner_id = $1 AND batch_id = $2 AND line_no <= $3",
            args![owner, id, through],
        )
        .await?;
        let overhead = count.saturating_mul(row.item_overhead).min(row.completion_reserved);
        tx.execute(
            "UPDATE batches SET completion_reserved = completion_reserved - $3 WHERE owner_id = $1 AND id = $2",
            args![owner, id, overhead],
        )
        .await?;
        apply_quota(tx, &batch.owner, QuotaDelta { used: -used, reserved: -overhead, ..QuotaDelta::default() }).await?;
    }
    let remaining = scalar(
        tx,
        &format!("SELECT {count_all} FROM batch_items WHERE owner_id = $1 AND batch_id = $2"),
        args![owner, id],
    )
    .await?;
    if remaining > 0 || row.accounting_pending > 0 {
        return Ok(false);
    }
    if delete_ledger_page(tx, batch, "batch_settlements", &["claim_id", "line_no"], false).await?
        || delete_ledger_page(tx, batch, "batch_claims", &["id"], false).await?
        || delete_ledger_page(tx, batch, "batch_operations", &["owner_id", "kind", "operation_id"], true).await?
    {
        return Ok(false);
    }
    tx.execute("UPDATE batches SET cleanup_state = 'done' WHERE owner_id = $1 AND id = $2", args![owner, id]).await?;
    Ok(true)
}

pub(in super::super) async fn metadata_candidates<T: SqlTx>(
    tx: &mut T,
    limit: u32,
) -> Result<Vec<BatchRef>, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}owner_id, id FROM batches WHERE cleanup_state = 'done' AND metadata_expires_at <= $1 \
         ORDER BY metadata_expires_at{}",
        d.top("$2"),
        d.limit("$2")
    );
    let rows = tx.fetch_all(&sql, args![now, i64::from(limit)]).await?;
    refs(&rows)
}

/// After retention, purge bounded metadata pages; release the retained slot only in the final transaction.
pub(in super::super) async fn purge_metadata<T: SqlTx>(tx: &mut T, batch: &BatchRef) -> Result<bool, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let Some(row) = load_batch(tx, &batch.owner, &batch.id, true).await? else { return Ok(false) };
    let now = db_now(tx).await?;
    if row.cleanup_state != "done" || row.metadata_expires_at.is_none_or(|at| at > now) {
        return Ok(false);
    }
    let owner = batch.owner.as_str();
    let id = batch.id.as_str();
    let sessions = scalar(
        tx,
        &format!("SELECT {} FROM artifact_sessions WHERE owner_id = $1 AND batch_id = $2", tx.dialect().count_all),
        args![owner, id],
    )
    .await?;
    if sessions > 0 {
        return Ok(false);
    }
    // Retention has elapsed: purge client operation ids and errors in bounded transactions too. Keep the
    // batch and its full metadata charge until the final page commits, so retries cannot double-release it.
    if delete_ledger_page(tx, batch, "batch_errors", &["owner_id", "batch_id", "seq"], false).await?
        || delete_ledger_page(tx, batch, "batch_operations", &["owner_id", "kind", "operation_id"], false).await?
    {
        return Ok(false);
    }
    tx.execute("DELETE FROM batches WHERE owner_id = $1 AND id = $2", args![owner, id]).await?;
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
