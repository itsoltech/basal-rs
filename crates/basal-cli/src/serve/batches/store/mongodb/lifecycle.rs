//! Batch create, reads, cancel, expiry, the idempotent early terminal transaction and draining (MongoDB).

use mongodb::bson::{doc, Bson, Document};

use super::super::error::{InvalidValue, StoreError};
use super::super::ops::common::{admit, metadata_to_db, millis, signed, unsigned, BatchRow, QuotaDelta};
use super::super::ops::lifecycle::Created;
use super::super::policy::{bounded_issue, completion_reserve, EXPIRED_CODE, EXPIRED_MESSAGE, ISSUE_CHARGE_BYTES};
use super::super::types::{
    BatchId, BatchIssue, BatchRecord, BatchRef, BatchStatus, DrainProgress, FilePurpose, Idempotency, NewBatch,
    OwnerId, Page, PageRequest, QuotaPolicySpec, TerminalStatus, COMPLETION_WINDOW,
};
use super::common::{
    active_policy, apply_quota, count_batch_items, end_phase, lock_batch, lock_quota, owner_limits, read_batch,
    set_batch,
};
use super::convert::{batch_ref, batch_row, file_row};
use super::driver::{
    bin, bytes, corrupt, int, mtx, opt_int, opt_str, opt_text, text, Core, Tx, BATCHES, CATALOG_REVISIONS, CLAIMS,
    ERRORS, FILES, ITEMS, OPERATIONS, SESSIONS, SETTLEMENTS,
};

async fn replay_create(
    tx: &mut Tx<'_>,
    owner: &OwnerId,
    idempotency: &Idempotency,
) -> Result<Option<BatchRecord>, StoreError> {
    let Some(doc) = tx
        .one(
            OPERATIONS,
            doc! { "owner_id": owner.as_str(), "kind": "create", "operation_id": idempotency.key.as_str() },
        )
        .await?
    else {
        return Ok(None);
    };
    if bin(&doc, "request_sha256")? != idempotency.request_sha256.0 {
        return Err(StoreError::IdempotencyConflict);
    }
    let batch = BatchRef { owner: owner.clone(), id: super::convert::batch_id(text(&doc, "batch_id")?)? };
    let row = read_batch(tx, &batch).await?.ok_or(corrupt("idempotency entry without batch"))?;
    Ok(Some(row.record))
}

fn new_batch_doc(owner: &OwnerId, id: &BatchId, spec: &NewBatch, values: &[(&str, Bson)]) -> Document {
    let mut doc = doc! {
        "_id": id.as_str(),
        "owner_id": owner.as_str(),
        "input_file_id": spec.input_file_id.as_str(),
        "endpoint": spec.endpoint.as_str(),
        "model_name": Bson::Null,
        "fingerprint": Bson::Null,
        "planner_version": Bson::Null,
        "status": "validating",
        "terminal_target": Bson::Null,
        "in_progress_at": Bson::Null,
        "finalizing_at": Bson::Null,
        "cancelling_at": Bson::Null,
        "terminal_at": Bson::Null,
        "control_epoch": 0_i64,
        "mutation_seq": 0_i64,
        "sealed": false,
        "total": 0_i64,
        "completed": 0_i64,
        "failed": 0_i64,
        "cancelled": 0_i64,
        "expired": 0_i64,
        "queued": 0_i64,
        "accounting_pending": 0_i64,
        "staged_reserve": 0_i64,
        "export_reserved": 0_i64,
        "errors_bytes": 0_i64,
        "error_count": 0_i64,
        "errors_truncated": false,
        "output_file_id": Bson::Null,
        "error_file_id": Bson::Null,
        "released": false,
        "cleanup_state": "none",
        "metadata_expires_at": Bson::Null,
        "phase_kind": Bson::Null,
        "phase_claim_id": Bson::Null,
        "phase_generation": 0_i64,
        "phase_lease_until": Bson::Null,
        "sched_credit": 0_i64,
        "sched_turn": 0_i64,
    };
    for (key, value) in values {
        doc.insert(*key, value.clone());
    }
    doc
}

/// Pin the input, count the batch as active and retained, reserve output/error slots and the completion reserve.
async fn create_batch_tx(
    tx: &mut Tx<'_>,
    owner: &OwnerId,
    idempotency: Option<&Idempotency>,
    spec: &NewBatch,
) -> Result<Created, StoreError> {
    if spec.endpoint != "/v1/basal" {
        return Err(StoreError::Invalid(InvalidValue::new("endpoint", "must be /v1/basal")));
    }
    if let Some(key) = idempotency {
        if let Some(existing) = replay_create(tx, owner, key).await? {
            return Ok(Created::Replayed(existing));
        }
    }
    let (usage, _) = lock_quota(tx, owner).await?;
    let (file_doc, now) = tx
        .lock(FILES, doc! { "_id": spec.input_file_id.as_str(), "owner_id": owner.as_str(), "state": "active" })
        .await?
        .ok_or(StoreError::NotFound)?;
    let file = file_row(&file_doc)?;
    if file.record.expires_at.is_some_and(|at| at.0 <= now) {
        return Err(StoreError::NotFound);
    }
    if file.record.purpose != FilePurpose::Batch {
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
    let completion = signed(reserve.total())?;
    let delta = QuotaDelta { reserved: completion, active: 1, retained: 1, files: 2, ..QuotaDelta::default() };
    admit(&usage, &limits, &delta)?;
    let catalog = tx
        .one_sorted(CATALOG_REVISIONS, doc! {}, doc! { "_id": -1_i32 })
        .await?
        .map(|d| int(&d, "_id"))
        .transpose()?
        .ok_or(StoreError::CatalogMissing)?;
    let id = BatchId::new_random();
    let expires_at = now.saturating_add(millis(COMPLETION_WINDOW));
    let output_after = spec.output_expires_after.map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX));
    let doc = new_batch_doc(
        owner,
        &id,
        spec,
        &[
            ("catalog_revision", Bson::Int64(catalog)),
            ("created_at", Bson::Int64(now)),
            ("expires_at", Bson::Int64(expires_at)),
            ("object_charge", Bson::Int64(policy.object_charge)),
            ("item_overhead", Bson::Int64(policy.item_overhead)),
            ("max_records", Bson::Int64(signed(reserve.max_records)?)),
            ("reserve_records", Bson::Int64(signed(reserve.records)?)),
            ("completion_reserved", Bson::Int64(completion)),
            ("metadata", Bson::String(metadata_to_db(&spec.metadata)?)),
            ("output_expires_after_secs", super::driver::opt_i64(output_after)),
        ],
    );
    tx.insert(BATCHES, doc).await?;
    tx.update(FILES, doc! { "_id": spec.input_file_id.as_str() }, doc! { "$inc": { "pin_count": 1_i64 } }).await?;
    if let Some(key) = idempotency {
        tx.insert(
            OPERATIONS,
            doc! {
                "owner_id": owner.as_str(),
                "kind": "create",
                "operation_id": key.key.as_str(),
                "request_sha256": bytes(&key.request_sha256.0),
                "batch_id": id.as_str(),
                "outcome": "created",
                "created_at": now,
            },
        )
        .await?;
    }
    apply_quota(tx, owner, delta).await?;
    let row = read_batch(tx, &BatchRef { owner: owner.clone(), id }).await?.ok_or(corrupt("created batch missing"))?;
    Ok(Created::New(row.record))
}

pub(super) async fn create_batch(
    core: &Core,
    owner: &OwnerId,
    idempotency: Option<&Idempotency>,
    spec: &NewBatch,
) -> Result<Created, StoreError> {
    let mut attempt = 0;
    loop {
        match mtx!(core, tx => create_batch_tx(tx, owner, idempotency, spec)) {
            // A concurrent create with the same key won the unique ledger document: replay it.
            Err(error) if error.unique_violation() && idempotency.is_some() && attempt == 0 => attempt += 1,
            other => return other,
        }
    }
}

pub(super) async fn get_batch(core: &Core, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
    let doc = core.read_one(BATCHES, doc! { "_id": id.as_str(), "owner_id": owner.as_str() }).await?;
    doc.map(|d| batch_row(&d).map(|b| b.record)).transpose()?.ok_or(StoreError::NotFound)
}

pub(super) async fn list_batches(
    core: &Core,
    owner: &OwnerId,
    page: &PageRequest<BatchId>,
) -> Result<Page<BatchRecord>, StoreError> {
    let limit = i64::from(page.limit.clamp(1, 100));
    let mut filter = doc! { "owner_id": owner.as_str() };
    if let Some(after) = &page.after {
        let cursor = core
            .read_one(BATCHES, doc! { "_id": after.as_str(), "owner_id": owner.as_str() })
            .await?
            .ok_or(StoreError::NotFound)?;
        let at = int(&cursor, "created_at")?;
        filter.insert(
            "$or",
            vec![doc! { "created_at": { "$lt": at } }, doc! { "created_at": at, "_id": { "$lt": after.as_str() } }],
        );
    }
    let docs = core.read_many(BATCHES, filter, doc! { "created_at": -1_i32, "_id": -1_i32 }, limit + 1).await?;
    let has_more = docs.len() as i64 > limit;
    let items = docs.iter().take(limit as usize).map(|d| batch_row(d).map(|b| b.record)).collect::<Result<_, _>>()?;
    Ok(Page { items, has_more })
}

pub(super) async fn batch_errors(core: &Core, owner: &OwnerId, id: &BatchId) -> Result<Vec<BatchIssue>, StoreError> {
    if core.read_one(BATCHES, doc! { "_id": id.as_str(), "owner_id": owner.as_str() }).await?.is_none() {
        return Err(StoreError::NotFound);
    }
    let docs = core
        .read_many(
            ERRORS,
            doc! { "owner_id": owner.as_str(), "batch_id": id.as_str() },
            doc! { "seq": 1_i32 },
            BatchIssue::MAX_PER_BATCH as i64,
        )
        .await?;
    docs.iter()
        .map(|d| {
            Ok(BatchIssue {
                code: text(d, "code")?,
                message: text(d, "message")?,
                param: opt_text(d, "param")?,
                line: opt_int(d, "line_no")?.map(unsigned).transpose()?,
            })
        })
        .collect()
}

/// Terminal transaction before seal (validation failure, cancel or expiry); quota and batch are locked by the caller.
pub(super) async fn terminate_unsealed(
    tx: &mut Tx<'_>,
    batch: &BatchRow,
    target: TerminalStatus,
    issues: &[BatchIssue],
    truncated: bool,
    now: i64,
) -> Result<(), StoreError> {
    if batch.released || batch.sealed {
        return Err(corrupt("early terminal on a sealed or released batch"));
    }
    let batch_key = batch.batch_ref();
    let mut stored = 0i64;
    for (seq, issue) in issues.iter().take(BatchIssue::MAX_PER_BATCH).enumerate() {
        let issue = bounded_issue(issue);
        tx.insert(
            ERRORS,
            doc! {
                "owner_id": batch_key.owner.as_str(),
                "batch_id": batch_key.id.as_str(),
                "seq": seq as i64,
                "code": issue.code,
                "message": issue.message,
                "param": opt_str(issue.param.as_deref()),
                "line_no": super::driver::opt_i64(issue.line.map(signed).transpose()?),
            },
        )
        .await?;
        stored += 1;
    }
    let issue_bytes = stored * ISSUE_CHARGE_BYTES as i64;
    let truncated = truncated || issues.len() > BatchIssue::MAX_PER_BATCH;
    let staged = count_batch_items(tx, &batch_key).await?;
    let completion = staged.saturating_mul(batch.item_overhead);
    let metadata_expires_at = now.saturating_add(millis(tx.settings().metadata_retention));
    end_phase(tx, batch).await?;
    let changed = tx
        .update(
            BATCHES,
            doc! { "_id": batch_key.id.as_str(), "owner_id": batch_key.owner.as_str(), "released": false },
            doc! {
                "$set": {
                    "status": target.as_db(),
                    "terminal_target": target.as_db(),
                    "terminal_at": now,
                    "released": true,
                    "cleanup_state": "pending",
                    "metadata_expires_at": metadata_expires_at,
                    "completion_reserved": completion,
                    "errors_bytes": issue_bytes,
                    "error_count": stored,
                    "errors_truncated": truncated,
                    "queued": 0_i64,
                },
                "$inc": { "control_epoch": 1_i64 },
            },
        )
        .await?;
    if changed != 1 {
        return Err(corrupt("terminal transaction lost its batch"));
    }
    tx.update(
        FILES,
        doc! { "_id": batch.record.input_file_id.as_str(), "owner_id": batch_key.owner.as_str(), "pin_count": { "$gt": 0_i64 } },
        doc! { "$inc": { "pin_count": -1_i64 } },
    )
    .await?;
    let released = batch.completion_reserved.saturating_sub(completion);
    apply_quota(
        tx,
        &batch_key.owner,
        QuotaDelta { used: issue_bytes, reserved: -released, active: -1, files: -2, ..QuotaDelta::default() },
    )
    .await
}

fn expiry_issue() -> BatchIssue {
    BatchIssue { code: EXPIRED_CODE.to_owned(), message: EXPIRED_MESSAGE.to_owned(), param: None, line: None }
}

/// Close the barrier of a sealed batch: a real write of the shared control document with a new epoch.
async fn close_barrier(tx: &mut Tx<'_>, batch: &BatchRow, target: TerminalStatus, now: i64) -> Result<(), StoreError> {
    let mut set = doc! { "status": "draining", "terminal_target": target.as_db() };
    if target == TerminalStatus::Cancelled && batch.record.cancelling_at.is_none() {
        set.insert("cancelling_at", now);
    }
    set_batch(tx, &batch.batch_ref(), doc! { "$set": set, "$inc": { "control_epoch": 1_i64 } }).await
}

async fn apply_expiry(tx: &mut Tx<'_>, batch: &BatchRow, now: i64) -> Result<bool, StoreError> {
    if batch.record.terminal_target.is_some() || now < batch.record.expires_at.0 {
        return Ok(false);
    }
    match batch.record.status {
        BatchStatus::Validating => {
            terminate_unsealed(tx, batch, TerminalStatus::Expired, &[expiry_issue()], false, now).await?;
            Ok(true)
        }
        BatchStatus::InProgress => {
            close_barrier(tx, batch, TerminalStatus::Expired, now).await?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

async fn request_cancel_tx(tx: &mut Tx<'_>, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
    let key = BatchRef { owner: owner.clone(), id: id.clone() };
    lock_quota(tx, owner).await?;
    let (batch, now) = lock_batch(tx, &key).await?;
    if !apply_expiry(tx, &batch, now).await? {
        match (batch.record.status, batch.record.terminal_target) {
            (BatchStatus::Validating, None) => {
                terminate_unsealed(tx, &batch, TerminalStatus::Cancelled, &[], false, now).await?;
            }
            (BatchStatus::InProgress, None) => close_barrier(tx, &batch, TerminalStatus::Cancelled, now).await?,
            (BatchStatus::Finalizing, Some(TerminalStatus::Completed)) => {
                set_batch(
                    tx,
                    &key,
                    doc! {
                        "$set": { "terminal_target": "cancelled", "cancelling_at": now },
                        "$inc": { "control_epoch": 1_i64 },
                    },
                )
                .await?;
            }
            _ => {}
        }
    }
    Ok(read_batch(tx, &key).await?.ok_or(corrupt("cancelled batch missing"))?.record)
}

/// Cancel (quota → batch → file); idempotent, becomes expiry after the deadline.
pub(super) async fn request_cancel(core: &Core, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
    mtx!(core, tx => request_cancel_tx(tx, owner, id))
}

pub(super) fn refs(docs: &[Document]) -> Result<Vec<BatchRef>, StoreError> {
    docs.iter().map(batch_ref).collect()
}

pub(super) async fn due_expirations(core: &Core, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let now = core.server_now().await?;
    let docs = core
        .read_many(
            BATCHES,
            doc! { "status": { "$in": ["validating", "in_progress"] }, "expires_at": { "$lte": now } },
            doc! { "expires_at": 1_i32 },
            i64::from(limit),
        )
        .await?;
    refs(&docs)
}

async fn expire_batch_tx(tx: &mut Tx<'_>, batch: &BatchRef) -> Result<BatchRecord, StoreError> {
    lock_quota(tx, &batch.owner).await?;
    let (row, now) = lock_batch(tx, batch).await?;
    apply_expiry(tx, &row, now).await?;
    Ok(read_batch(tx, batch).await?.ok_or(corrupt("expired batch missing"))?.record)
}

pub(super) async fn expire_batch(core: &Core, batch: &BatchRef) -> Result<BatchRecord, StoreError> {
    mtx!(core, tx => expire_batch_tx(tx, batch))
}

pub(super) async fn draining_batches(core: &Core, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
    let docs =
        core.read_many(BATCHES, doc! { "status": "draining" }, doc! { "expires_at": 1_i32 }, i64::from(limit)).await?;
    refs(&docs)
}

/// Decide the open settlement of a claimed item that its claim will not commit; an uploaded result object of that
/// attempt keeps one `result_bound` through its (now orphaned) session until GC.
pub(super) async fn decide_open_settlement(
    tx: &mut Tx<'_>,
    claim_id: &str,
    line_no: i64,
    kind: &str,
    batch: &BatchRow,
    now: i64,
) -> Result<bool, StoreError> {
    decide_open_settlement_with_cost(tx, claim_id, line_no, kind, batch, now, None).await
}

pub(super) async fn decide_open_settlement_with_cost(
    tx: &mut Tx<'_>,
    claim_id: &str,
    line_no: i64,
    kind: &str,
    batch: &BatchRow,
    now: i64,
    actual_cost: Option<i64>,
) -> Result<bool, StoreError> {
    let cost = match actual_cost {
        Some(cost) => cost,
        None => match tx.one(SETTLEMENTS, doc! { "claim_id": claim_id, "line_no": line_no, "kind": Bson::Null }).await?
        {
            Some(row) => int(&row, "estimated_cost")?,
            None => return Ok(false),
        },
    };
    let orphan = tx
        .one(SESSIONS, doc! { "claim_id": claim_id, "line_no": line_no, "kind": "result" })
        .await?
        .map(|d| text(&d, "_id"))
        .transpose()?;
    if let Some(session) = &orphan {
        tx.update(
            SESSIONS,
            doc! { "_id": session.as_str(), "phase": "open" },
            doc! { "$set": { "phase": "orphaned" } },
        )
        .await?;
    }
    let changed = tx
        .update(
            SETTLEMENTS,
            doc! { "claim_id": claim_id, "line_no": line_no, "kind": Bson::Null },
            doc! { "$set": { "kind": kind, "orphan_session_id": opt_str(orphan.as_deref()), "decided_at": now, "actual_cost": cost } },
        )
        .await?;
    if changed == 1 {
        set_batch(tx, &batch.batch_ref(), doc! { "$inc": { "accounting_pending": 1_i64 } }).await?;
    }
    Ok(changed == 1)
}

async fn advance_drain_tx(tx: &mut Tx<'_>, batch: &BatchRef, limit: u32) -> Result<DrainProgress, StoreError> {
    let (row, now) = lock_batch(tx, batch).await?;
    if row.record.status != BatchStatus::Draining {
        return Ok(DrainProgress { materialized: 0, status: row.record.status });
    }
    let target = row.record.terminal_target.ok_or(corrupt("draining batch without target"))?;
    let (item_state, counter) = match target {
        TerminalStatus::Cancelled => ("cancelled", "cancelled"),
        TerminalStatus::Expired => ("expired", "expired"),
        _ => return Err(corrupt("draining toward an unexpected target")),
    };
    let scope = doc! { "owner_id": batch.owner.as_str(), "batch_id": batch.id.as_str() };
    let mut open = scope.clone();
    open.insert("state", doc! { "$in": ["pending", "claimed"] });
    let items = tx.many(ITEMS, open.clone(), doc! { "line_no": 1_i32 }, i64::from(limit)).await?;
    let mut pending = 0i64;
    for item in &items {
        let line_no = int(item, "line_no")?;
        if text(item, "state")? == "pending" {
            pending += 1;
        } else if let Some(claim) = opt_text(item, "claim_id")? {
            decide_open_settlement(tx, &claim, line_no, "revoked", &row, now).await?;
        }
        let mut key = scope.clone();
        key.insert("line_no", line_no);
        tx.update(ITEMS, key, doc! { "$set": { "state": item_state, "claim_id": Bson::Null } }).await?;
    }
    let count = items.len() as i64;
    set_batch(tx, batch, doc! { "$inc": { counter: count, "queued": -pending, "mutation_seq": 1_i64 } }).await?;
    let mut status = BatchStatus::Draining;
    if tx.count(ITEMS, open).await? == 0 {
        tx.update_many(
            CLAIMS,
            doc! { "owner_id": batch.owner.as_str(), "batch_id": batch.id.as_str(), "kind": "execution", "state": "active" },
            doc! { "$set": { "state": "finished" } },
        )
        .await?;
        set_batch(tx, batch, doc! { "$set": { "status": "finalizing", "finalizing_at": now } }).await?;
        status = BatchStatus::Finalizing;
    }
    Ok(DrainProgress { materialized: unsigned(count)?, status })
}

pub(super) async fn advance_drain(core: &Core, batch: &BatchRef, limit: u32) -> Result<DrainProgress, StoreError> {
    mtx!(core, tx => advance_drain_tx(tx, batch, limit))
}

/// Filter of the items of one batch.
pub(super) fn items_of(batch: &BatchRef) -> Document {
    doc! { "owner_id": batch.owner.as_str(), "batch_id": batch.id.as_str() }
}
