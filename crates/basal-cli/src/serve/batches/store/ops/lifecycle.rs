//! Batch create, reads, cancel, expiry, the idempotent early terminal transaction and draining.

use super::super::error::StoreError;
use super::super::policy::{bounded_issue, completion_reserve, EXPIRED_CODE, EXPIRED_MESSAGE, ISSUE_CHARGE_BYTES};
use super::super::sql::{args, db_now, SqlRow, SqlTx};
use super::super::types::{
    BatchId, BatchIssue, BatchRecord, BatchRef, BatchStatus, DrainProgress, Idempotency, NewBatch, OwnerId, Page,
    PageRequest, QuotaPolicySpec, StoreSettings, TerminalStatus, COMPLETION_WINDOW,
};
use super::common::{
    active_policy, admit, apply_quota, batch_row, corrupt, end_phase, file_row, load_batch, lock_batch, lock_quota,
    metadata_to_db, millis, owner_limits, scalar, unsigned, BatchRow, QuotaDelta, BATCH_COLUMNS, FILE_COLUMNS,
};

/// Outcome of `create_batch`: a new batch or the batch recorded for the same Idempotency-Key and request.
pub(in super::super) enum Created {
    New(BatchRecord),
    Replayed(BatchRecord),
}

async fn replay_create<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    idempotency: &Idempotency,
) -> Result<Option<BatchRecord>, StoreError> {
    let Some(row) = tx
        .fetch_optional(
            &format!(
                "SELECT request_sha256, batch_id FROM batch_operations WHERE owner_id = $1 AND kind = 'create' \
                 AND {}",
                tx.dialect().exact_eq("operation_id", "$2")
            ),
            args![owner.as_str(), idempotency.key.as_str()],
        )
        .await?
    else {
        return Ok(None);
    };
    if row.blob_at(0)? != idempotency.request_sha256.0 {
        return Err(StoreError::IdempotencyConflict);
    }
    let id = super::common::batch_id_from_db(row.text_at(1)?)?;
    // The ledger entry and the batch are purged together, so the batch exists.
    let batch = load_batch(tx, owner, &id, false).await?.ok_or(corrupt("idempotency entry without batch"))?;
    Ok(Some(batch.record))
}

/// Pin the input, count the batch as active and retained, reserve output/error file slots and the completion
/// reserve, all in one transaction (quota → file).
pub(in super::super) async fn create_batch<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    idempotency: Option<&Idempotency>,
    spec: &NewBatch,
) -> Result<Created, StoreError> {
    if spec.endpoint != "/v1/basal" {
        return Err(StoreError::Invalid(super::super::error::InvalidValue::new("endpoint", "must be /v1/basal")));
    }
    if let Some(key) = idempotency {
        if let Some(existing) = replay_create(tx, owner, key).await? {
            return Ok(Created::Replayed(existing));
        }
    }
    let usage = lock_quota(tx, owner).await?;
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {FILE_COLUMNS} FROM batch_files{} WHERE owner_id = $1 AND id = $2 AND state = 'active' \
         AND (expires_at IS NULL OR expires_at > $3){}",
        d.lock_hint, d.for_update
    );
    let file = tx
        .fetch_optional(&sql, args![owner.as_str(), spec.input_file_id.as_str(), now])
        .await?
        .map(|row| file_row(&row))
        .transpose()?
        .ok_or(StoreError::NotFound)?;
    if file.record.purpose != super::super::types::FilePurpose::Batch {
        return Err(StoreError::InputUnusable("input file purpose must be batch"));
    }
    let policy = active_policy(tx).await?;
    let limits = owner_limits(tx, policy.revision, owner).await?;
    let spec_policy = QuotaPolicySpec {
        object_charge_bytes: unsigned(policy.object_charge)?,
        item_overhead_bytes: unsigned(policy.item_overhead)?,
        max_upload_bytes: unsigned(policy.max_upload)?,
        max_records_per_batch: unsigned(policy.max_records)?,
        installation: limits,
        owners: Vec::new(),
    };
    let reserve = completion_reserve(file.record.bytes, &spec_policy);
    let completion = super::common::signed(reserve.total())?;
    let delta = QuotaDelta { reserved: completion, active: 1, retained: 1, files: 2, ..QuotaDelta::default() };
    admit(&usage, &limits, &delta)?;
    let catalog =
        scalar(tx, "SELECT CAST(COALESCE(MAX(revision), 0) AS BIGINT) FROM batch_catalog_revisions", args![]).await?;
    if catalog == 0 {
        return Err(StoreError::CatalogMissing);
    }
    let id = BatchId::new_random();
    let expires_at = now.saturating_add(millis(COMPLETION_WINDOW));
    let output_after = spec.output_expires_after.map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    tx.execute(
        "INSERT INTO batches (owner_id, id, input_file_id, endpoint, catalog_revision, status, created_at, expires_at, \
         object_charge, item_overhead, max_records, reserve_records, completion_reserved, metadata, \
         output_expires_after_secs) VALUES ($1, $2, $3, $4, $5, 'validating', $6, $7, $8, $9, $10, $11, $12, $13, $14)",
        args![
            owner.as_str(),
            id.as_str(),
            spec.input_file_id.as_str(),
            spec.endpoint.as_str(),
            catalog,
            now,
            expires_at,
            policy.object_charge,
            policy.item_overhead,
            super::common::signed(reserve.max_records)?,
            super::common::signed(reserve.records)?,
            completion,
            metadata_to_db(&spec.metadata)?,
            output_after
        ],
    )
    .await?;
    tx.execute(
        "UPDATE batch_files SET pin_count = pin_count + 1 WHERE owner_id = $1 AND id = $2",
        args![owner.as_str(), spec.input_file_id.as_str()],
    )
    .await?;
    if let Some(key) = idempotency {
        tx.execute(
            "INSERT INTO batch_operations (owner_id, kind, operation_id, request_sha256, batch_id, outcome, created_at) \
             VALUES ($1, 'create', $2, $3, $4, 'created', $5)",
            args![owner.as_str(), key.key.as_str(), key.request_sha256.0.to_vec(), id.as_str(), now],
        )
        .await?;
    }
    apply_quota(tx, owner, delta).await?;
    let batch = load_batch(tx, owner, &id, false).await?.ok_or(corrupt("created batch missing"))?;
    Ok(Created::New(batch.record))
}

pub(in super::super) async fn get_batch<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    id: &BatchId,
) -> Result<BatchRecord, StoreError> {
    load_batch(tx, owner, id, false).await?.map(|b| b.record).ok_or(StoreError::NotFound)
}

pub(in super::super) async fn list_batches<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    page: &PageRequest<BatchId>,
) -> Result<Page<BatchRecord>, StoreError> {
    let limit = i64::from(page.limit.clamp(1, 100));
    let (cursor_at, cursor_id) = match &page.after {
        Some(after) => {
            let row = tx
                .fetch_optional(
                    "SELECT created_at FROM batches WHERE owner_id = $1 AND id = $2",
                    args![owner.as_str(), after.as_str()],
                )
                .await?
                .ok_or(StoreError::NotFound)?;
            (Some(row.i64_at(0)?), Some(after.as_str().to_owned()))
        }
        None => (None, None),
    };
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}{BATCH_COLUMNS} FROM batches WHERE owner_id = $1 \
         AND ($2 IS NULL OR created_at < $2 OR (created_at = $2 AND id < $3)) \
         ORDER BY created_at DESC, id DESC{}",
        d.top("$4"),
        d.limit("$4")
    );
    let rows = tx.fetch_all(&sql, args![owner.as_str(), cursor_at, cursor_id, limit + 1]).await?;
    let has_more = rows.len() as i64 > limit;
    let items = rows.iter().take(limit as usize).map(|r| batch_row(r).map(|b| b.record)).collect::<Result<_, _>>()?;
    Ok(Page { items, has_more })
}

pub(in super::super) async fn batch_errors<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    id: &BatchId,
) -> Result<Vec<BatchIssue>, StoreError> {
    if load_batch(tx, owner, id, false).await?.is_none() {
        return Err(StoreError::NotFound);
    }
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}code, message, param, line_no FROM batch_errors WHERE owner_id = $1 AND batch_id = $2 \
         ORDER BY seq{}",
        d.top("$3"),
        d.limit("$3")
    );
    let rows = tx.fetch_all(&sql, args![owner.as_str(), id.as_str(), BatchIssue::MAX_PER_BATCH as i64]).await?;
    rows.iter()
        .map(|row| {
            Ok(BatchIssue {
                code: row.text_at(0)?,
                message: row.text_at(1)?,
                param: row.opt_text_at(2)?,
                line: row.opt_i64_at(3)?.map(unsigned).transpose()?,
            })
        })
        .collect()
}

/// Terminal transaction before seal (validation failure, cancel or expiry): quota → batch → input file.
/// Sets status, `terminal_at` and `released`, releases the active slot, the output/error slots, the pin and every
/// unused reserve; the row charge of staged items stays until cleanup deletes them. Guarded by `released = 0`.
pub(in super::super) async fn terminate_unsealed<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    batch: &BatchRow,
    target: TerminalStatus,
    issues: &[BatchIssue],
    truncated: bool,
    now: i64,
) -> Result<(), StoreError> {
    if batch.released || batch.sealed {
        return Err(corrupt("early terminal on a sealed or released batch"));
    }
    let owner = &batch.record.owner;
    let id = &batch.record.id;
    let mut stored = 0usize;
    let mut issue_bytes = 0i64;
    for (seq, issue) in issues.iter().take(BatchIssue::MAX_PER_BATCH).enumerate() {
        let issue = bounded_issue(issue);
        tx.execute(
            "INSERT INTO batch_errors (owner_id, batch_id, seq, code, message, param, line_no) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            args![
                owner.as_str(),
                id.as_str(),
                seq as i64,
                issue.code,
                issue.message,
                issue.param,
                issue.line.map(super::common::signed).transpose()?
            ],
        )
        .await?;
        stored += 1;
        issue_bytes += ISSUE_CHARGE_BYTES as i64;
    }
    let truncated = truncated || issues.len() > BatchIssue::MAX_PER_BATCH;
    let staged = scalar(
        tx,
        &format!("SELECT {} FROM batch_items WHERE owner_id = $1 AND batch_id = $2", tx.dialect().count_all),
        args![owner.as_str(), id.as_str()],
    )
    .await?;
    let completion = staged.saturating_mul(batch.item_overhead);
    let metadata_expires_at = now.saturating_add(millis(settings.metadata_retention));
    end_phase(tx, batch).await?;
    let changed = tx
        .execute(
            "UPDATE batches SET status = $3, terminal_target = $3, terminal_at = $4, released = 1, \
             cleanup_state = 'pending', metadata_expires_at = $5, completion_reserved = $6, errors_bytes = $7, \
             error_count = $8, errors_truncated = $9, control_epoch = control_epoch + 1, queued = 0 \
             WHERE owner_id = $1 AND id = $2 AND released = 0",
            args![
                owner.as_str(),
                id.as_str(),
                target.as_db(),
                now,
                metadata_expires_at,
                completion,
                issue_bytes,
                stored as i64,
                truncated
            ],
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("terminal transaction lost its batch"));
    }
    tx.execute(
        "UPDATE batch_files SET pin_count = pin_count - 1 WHERE owner_id = $1 AND id = $2 AND pin_count > 0",
        args![owner.as_str(), batch.record.input_file_id.as_str()],
    )
    .await?;
    let released = batch.completion_reserved.saturating_sub(completion);
    apply_quota(
        tx,
        owner,
        QuotaDelta { used: issue_bytes, reserved: -released, active: -1, files: -2, ..QuotaDelta::default() },
    )
    .await
}

fn expiry_issue() -> BatchIssue {
    BatchIssue { code: EXPIRED_CODE.to_owned(), message: EXPIRED_MESSAGE.to_owned(), param: None, line: None }
}

/// Close the barrier of a sealed batch: no further commits or claims, remaining items drain in small chunks.
async fn close_barrier<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRow,
    target: TerminalStatus,
    now: i64,
) -> Result<(), StoreError> {
    let cancelling_at = (target == TerminalStatus::Cancelled).then_some(now);
    tx.execute(
        "UPDATE batches SET status = 'draining', terminal_target = $3, control_epoch = control_epoch + 1, \
         cancelling_at = COALESCE(cancelling_at, $4) WHERE owner_id = $1 AND id = $2",
        args![batch.record.owner.as_str(), batch.record.id.as_str(), target.as_db(), cancelling_at],
    )
    .await?;
    Ok(())
}

/// Apply the deadline under batch control. Returns true if this call closed the batch.
async fn apply_expiry<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    batch: &BatchRow,
    now: i64,
) -> Result<bool, StoreError> {
    if batch.record.terminal_target.is_some() || now < batch.record.expires_at.0 {
        return Ok(false);
    }
    match batch.record.status {
        BatchStatus::Validating => {
            terminate_unsealed(tx, settings, batch, TerminalStatus::Expired, &[expiry_issue()], false, now).await?;
            Ok(true)
        }
        BatchStatus::InProgress => {
            close_barrier(tx, batch, TerminalStatus::Expired, now).await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Cancel (quota → batch → file). A cancel after the deadline without an earlier target becomes expiry; cancel of
/// a terminal batch or of one already closing is an idempotent read of the current resource.
pub(in super::super) async fn request_cancel<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    owner: &OwnerId,
    id: &BatchId,
) -> Result<BatchRecord, StoreError> {
    lock_quota(tx, owner).await?;
    let batch = load_batch(tx, owner, id, true).await?.ok_or(StoreError::NotFound)?;
    let now = db_now(tx).await?;
    if !apply_expiry(tx, settings, &batch, now).await? {
        match (batch.record.status, batch.record.terminal_target) {
            (BatchStatus::Validating, None) => {
                terminate_unsealed(tx, settings, &batch, TerminalStatus::Cancelled, &[], false, now).await?;
            }
            (BatchStatus::InProgress, None) => close_barrier(tx, &batch, TerminalStatus::Cancelled, now).await?,
            (BatchStatus::Finalizing, Some(TerminalStatus::Completed)) => {
                // Every record already has a result: the export is the same, only the reported target changes.
                tx.execute(
                    "UPDATE batches SET terminal_target = 'cancelled', cancelling_at = $3, \
                     control_epoch = control_epoch + 1 WHERE owner_id = $1 AND id = $2",
                    args![owner.as_str(), id.as_str(), now],
                )
                .await?;
            }
            _ => {}
        }
    }
    let batch = load_batch(tx, owner, id, false).await?.ok_or(corrupt("cancelled batch missing"))?;
    Ok(batch.record)
}

/// Batches past their deadline without a close reason (bounded).
pub(in super::super) async fn due_expirations<T: SqlTx>(tx: &mut T, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}owner_id, id FROM batches WHERE status IN ('validating', 'in_progress') AND expires_at <= $1 \
         ORDER BY expires_at{}",
        d.top("$2"),
        d.limit("$2")
    );
    let rows = tx.fetch_all(&sql, args![now, i64::from(limit)]).await?;
    refs(&rows)
}

pub(in super::super) fn refs<R: SqlRow>(rows: &[R]) -> Result<Vec<BatchRef>, StoreError> {
    rows.iter()
        .map(|row| {
            Ok(BatchRef {
                owner: super::common::owner_from_db(row.text_at(0)?)?,
                id: super::common::batch_id_from_db(row.text_at(1)?)?,
            })
        })
        .collect()
}

/// Expire one batch (quota → batch → file); a no-op before the deadline or once a target is set.
pub(in super::super) async fn expire_batch<T: SqlTx>(
    tx: &mut T,
    settings: &StoreSettings,
    batch: &BatchRef,
) -> Result<BatchRecord, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let row = lock_batch(tx, batch).await?;
    let now = db_now(tx).await?;
    apply_expiry(tx, settings, &row, now).await?;
    Ok(load_batch(tx, &batch.owner, &batch.id, false).await?.ok_or(corrupt("expired batch missing"))?.record)
}

pub(in super::super) async fn draining_batches<T: SqlTx>(tx: &mut T, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}owner_id, id FROM batches WHERE status = 'draining' ORDER BY expires_at{}",
        d.top("$1"),
        d.limit("$1")
    );
    let rows = tx.fetch_all(&sql, args![i64::from(limit)]).await?;
    refs(&rows)
}

/// Decide the open settlement row of a claimed item that will not be committed by its claim. Moves the source
/// half later (in settlement) to an existing result session of that attempt, so an uploaded object stays covered.
pub(in super::super) async fn decide_open_settlement<T: SqlTx>(
    tx: &mut T,
    claim_id: &str,
    line_no: i64,
    kind: &str,
    batch: &BatchRow,
    now: i64,
) -> Result<bool, StoreError> {
    decide_open_settlement_with_cost(tx, claim_id, line_no, kind, batch, now, None).await
}

pub(in super::super) async fn decide_open_settlement_with_cost<T: SqlTx>(
    tx: &mut T,
    claim_id: &str,
    line_no: i64,
    kind: &str,
    batch: &BatchRow,
    now: i64,
    actual_cost: Option<i64>,
) -> Result<bool, StoreError> {
    let orphan = tx
        .fetch_optional(
            "SELECT id FROM artifact_sessions WHERE claim_id = $1 AND line_no = $2 AND kind = 'result'",
            args![claim_id, line_no],
        )
        .await?
        .map(|row| row.text_at(0))
        .transpose()?;
    if let Some(session) = &orphan {
        tx.execute(
            "UPDATE artifact_sessions SET phase = 'orphaned' WHERE id = $1 AND phase = 'open'",
            args![session.as_str()],
        )
        .await?;
    }
    let changed = tx
        .execute(
            "UPDATE batch_settlements SET kind = $3, orphan_session_id = $4, decided_at = $5, actual_cost = COALESCE($6, estimated_cost) \
             WHERE claim_id = $1 AND line_no = $2 AND kind IS NULL",
            args![claim_id, line_no, kind, orphan, now, actual_cost],
        )
        .await?;
    if changed == 1 {
        tx.execute(
            "UPDATE batches SET accounting_pending = accounting_pending + 1 WHERE owner_id = $1 AND id = $2",
            args![batch.record.owner.as_str(), batch.record.id.as_str()],
        )
        .await?;
    }
    Ok(changed == 1)
}

/// Materialize up to `limit` remaining items of a draining batch as cancelled or expired and revoke their claims.
/// When none remain the batch moves to finalizing (finalization still waits for `accounting_pending = 0`).
pub(in super::super) async fn advance_drain<T: SqlTx>(
    tx: &mut T,
    batch: &BatchRef,
    limit: u32,
) -> Result<DrainProgress, StoreError> {
    let row = lock_batch(tx, batch).await?;
    if row.record.status != BatchStatus::Draining {
        return Ok(DrainProgress { materialized: 0, status: row.record.status });
    }
    let target = row.record.terminal_target.ok_or(corrupt("draining batch without target"))?;
    let item_state = match target {
        TerminalStatus::Cancelled => "cancelled",
        TerminalStatus::Expired => "expired",
        _ => return Err(corrupt("draining toward an unexpected target")),
    };
    let now = db_now(tx).await?;
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}line_no, state, claim_id FROM batch_items WHERE owner_id = $1 AND batch_id = $2 \
         AND state IN ('pending', 'claimed') ORDER BY line_no{}",
        d.top("$3"),
        d.limit("$3")
    );
    let items = tx.fetch_all(&sql, args![batch.owner.as_str(), batch.id.as_str(), i64::from(limit)]).await?;
    let mut pending = 0i64;
    for item in &items {
        let line_no = item.i64_at(0)?;
        if item.text_at(1)? == "pending" {
            pending += 1;
        } else if let Some(claim) = item.opt_text_at(2)? {
            decide_open_settlement(tx, &claim, line_no, "revoked", &row, now).await?;
        }
        tx.execute(
            "UPDATE batch_items SET state = $4, claim_id = NULL WHERE owner_id = $1 AND batch_id = $2 AND line_no = $3",
            args![batch.owner.as_str(), batch.id.as_str(), line_no, item_state],
        )
        .await?;
    }
    let count = items.len() as i64;
    let column = if target == TerminalStatus::Cancelled { "cancelled" } else { "expired" };
    tx.execute(
        &format!(
            "UPDATE batches SET {column} = {column} + $3, queued = queued - $4, mutation_seq = mutation_seq + 1 \
             WHERE owner_id = $1 AND id = $2"
        ),
        args![batch.owner.as_str(), batch.id.as_str(), count, pending],
    )
    .await?;
    let remaining = scalar(
        tx,
        &format!(
            "SELECT {} FROM batch_items WHERE owner_id = $1 AND batch_id = $2 AND state IN ('pending', 'claimed')",
            d.count_all
        ),
        args![batch.owner.as_str(), batch.id.as_str()],
    )
    .await?;
    let mut status = BatchStatus::Draining;
    if remaining == 0 {
        tx.execute(
            "UPDATE batch_claims SET state = 'finished' WHERE owner_id = $1 AND batch_id = $2 AND kind = 'execution' \
             AND state = 'active'",
            args![batch.owner.as_str(), batch.id.as_str()],
        )
        .await?;
        tx.execute(
            "UPDATE batches SET status = 'finalizing', finalizing_at = $3 WHERE owner_id = $1 AND id = $2",
            args![batch.owner.as_str(), batch.id.as_str(), now],
        )
        .await?;
        status = BatchStatus::Finalizing;
    }
    Ok(DrainProgress { materialized: unsigned(count)?, status })
}
