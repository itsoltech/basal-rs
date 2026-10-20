//! Row mapping, quota accounting and lease checks shared by the domain operations.
//!
//! Lock order of every operation (each takes the subset it needs, never an earlier lock after a later one):
//! `batch_quota` (owner) → `batch_schedule` (shard) → `batches` (control) → claims/items/settlements (ascending)
//! → `batch_files` / `artifact_sessions`. Every item, claim and settlement mutation holds its batch row lock first,
//! so item rows themselves need no row locks.

use std::time::Duration;

use super::super::error::{QuotaResource, StoreError};
use super::super::sql::{args, to_db, SqlRow, SqlTx};
use super::super::types::{
    ArtifactRef, BatchId, BatchMetadata, BatchRecord, BatchRef, BatchStatus, ClaimToken, FileId, FilePurpose,
    FileRecord, FileState, Fingerprint, LeaseKind, ModelName, OwnerId, PlannerVersion, QuotaLimits, QuotaUsage,
    RequestCounts, Revision, Sha256Digest, TerminalStatus, UnixMillis,
};

pub(in super::super) fn millis(duration: Duration) -> i64 {
    i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
}

pub(in super::super) fn corrupt(what: &'static str) -> StoreError {
    StoreError::Corrupt(what)
}

pub(in super::super) fn owner_from_db(value: String) -> Result<OwnerId, StoreError> {
    OwnerId::parse(&value).map_err(|_| corrupt("owner id"))
}

pub(in super::super) fn batch_id_from_db(value: String) -> Result<BatchId, StoreError> {
    BatchId::parse(&value).map_err(|_| corrupt("batch id"))
}

pub(in super::super) fn file_id_from_db(value: String) -> Result<FileId, StoreError> {
    FileId::parse(&value).map_err(|_| corrupt("file id"))
}

pub(in super::super) fn artifact_from_db(value: String) -> Result<ArtifactRef, StoreError> {
    ArtifactRef::parse(&value).map_err(|_| corrupt("artifact ref"))
}

pub(in super::super) fn digest_from_db(value: Vec<u8>) -> Result<Sha256Digest, StoreError> {
    Sha256Digest::from_slice(&value).map_err(|_| corrupt("sha256"))
}

fn u64_from(value: i64) -> Result<u64, StoreError> {
    u64::try_from(value).map_err(|_| corrupt("negative value"))
}

// ---------------------------------------------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------------------------------------------

pub(in super::super) const FILE_COLUMNS: &str =
    "owner_id, id, purpose, filename, artifact_ref, bytes, sha256, created_at, expires_at, state, pin_count";

pub(in super::super) struct FileRow {
    pub record: FileRecord,
    pub pin_count: i64,
}

pub(in super::super) fn file_row<R: SqlRow>(row: &R) -> Result<FileRow, StoreError> {
    let purpose = match row.text_at(2)?.as_str() {
        "batch" => FilePurpose::Batch,
        "batch_output" => FilePurpose::BatchOutput,
        "batch_error" => FilePurpose::BatchError,
        _ => return Err(corrupt("file purpose")),
    };
    let state = match row.text_at(9)?.as_str() {
        "active" => FileState::Active,
        "deleted" => FileState::Deleted,
        _ => return Err(corrupt("file state")),
    };
    Ok(FileRow {
        record: FileRecord {
            owner: owner_from_db(row.text_at(0)?)?,
            id: file_id_from_db(row.text_at(1)?)?,
            purpose,
            filename: row.text_at(3)?,
            artifact: artifact_from_db(row.text_at(4)?)?,
            bytes: row.u64_at(5)?,
            sha256: digest_from_db(row.blob_at(6)?)?,
            created_at: UnixMillis(row.i64_at(7)?),
            expires_at: row.opt_i64_at(8)?.map(UnixMillis),
            state,
        },
        pin_count: row.i64_at(10)?,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------------------------------------------

pub(in super::super) const BATCH_COLUMNS: &str =
    "owner_id, id, input_file_id, endpoint, catalog_revision, model_name, \
     fingerprint, planner_version, status, terminal_target, created_at, expires_at, in_progress_at, finalizing_at, \
     cancelling_at, terminal_at, control_epoch, sealed, total, completed, failed, cancelled, expired, queued, \
     accounting_pending, object_charge, item_overhead, max_records, reserve_records, staged_reserve, \
     completion_reserved, export_reserved, errors_bytes, error_count, errors_truncated, output_file_id, \
     error_file_id, released, cleanup_state, metadata_expires_at, metadata, output_expires_after_secs, phase_kind, \
     phase_claim_id, phase_generation, phase_lease_until, sched_credit";

/// Batch row with its internal accounting and lease columns.
#[derive(Clone, Debug)]
pub(in super::super) struct BatchRow {
    pub record: BatchRecord,
    pub planner_version: Option<PlannerVersion>,
    pub sealed: bool,
    pub cancelled: u64,
    pub accounting_pending: i64,
    pub object_charge: i64,
    pub item_overhead: i64,
    pub max_records: i64,
    pub reserve_records: i64,
    pub staged_reserve: i64,
    pub completion_reserved: i64,
    pub export_reserved: i64,
    pub errors_bytes: i64,
    pub released: bool,
    pub cleanup_state: String,
    pub metadata_expires_at: Option<i64>,
    pub phase_kind: Option<String>,
    pub phase_claim_id: Option<String>,
    pub phase_generation: i64,
    pub phase_lease_until: Option<i64>,
    pub sched_credit: i64,
}

impl BatchRow {
    pub(in super::super) fn batch_ref(&self) -> BatchRef {
        BatchRef { owner: self.record.owner.clone(), id: self.record.id.clone() }
    }

    /// Records with a final item state (public `failed` includes expired).
    pub(in super::super) fn finished(&self) -> u64 {
        self.record.counts.completed + self.record.counts.failed + self.cancelled
    }
}

fn metadata_from_db(text: &str) -> Result<BatchMetadata, StoreError> {
    let pairs: Vec<(String, String)> = serde_json::from_str(text).map_err(|_| corrupt("batch metadata"))?;
    BatchMetadata::new(pairs).map_err(|_| corrupt("batch metadata"))
}

pub(in super::super) fn metadata_to_db(metadata: &BatchMetadata) -> Result<String, StoreError> {
    serde_json::to_string(metadata.pairs()).map_err(|_| corrupt("batch metadata"))
}

pub(in super::super) fn batch_row<R: SqlRow>(row: &R) -> Result<BatchRow, StoreError> {
    let status = BatchStatus::from_db(&row.text_at(8)?).ok_or(corrupt("batch status"))?;
    let terminal_target = match row.opt_text_at(9)? {
        Some(value) => Some(TerminalStatus::from_db(&value).ok_or(corrupt("terminal target"))?),
        None => None,
    };
    let failed = row.u64_at(20)?;
    let expired = row.u64_at(22)?;
    let record = BatchRecord {
        owner: owner_from_db(row.text_at(0)?)?,
        id: batch_id_from_db(row.text_at(1)?)?,
        input_file_id: file_id_from_db(row.text_at(2)?)?,
        endpoint: row.text_at(3)?,
        catalog_revision: Revision(row.i64_at(4)?),
        model_name: row.opt_text_at(5)?.map(|v| ModelName::parse(&v)).transpose().map_err(|_| corrupt("model"))?,
        fingerprint: row
            .opt_text_at(6)?
            .map(|v| Fingerprint::parse(&v))
            .transpose()
            .map_err(|_| corrupt("fingerprint"))?,
        status,
        terminal_target,
        created_at: UnixMillis(row.i64_at(10)?),
        expires_at: UnixMillis(row.i64_at(11)?),
        in_progress_at: row.opt_i64_at(12)?.map(UnixMillis),
        finalizing_at: row.opt_i64_at(13)?.map(UnixMillis),
        cancelling_at: row.opt_i64_at(14)?.map(UnixMillis),
        terminal_at: row.opt_i64_at(15)?.map(UnixMillis),
        control_epoch: row.i64_at(16)?,
        counts: RequestCounts { total: row.u64_at(18)?, completed: row.u64_at(19)?, failed: failed + expired },
        output_file_id: row.opt_text_at(35)?.map(file_id_from_db).transpose()?,
        error_file_id: row.opt_text_at(36)?.map(file_id_from_db).transpose()?,
        metadata: metadata_from_db(&row.text_at(40)?)?,
        output_expires_after: row.opt_i64_at(41)?.map(|s| Duration::from_secs(s.max(0).unsigned_abs())),
        error_count: row.u64_at(33)?,
        errors_truncated: row.bool_at(34)?,
    };
    Ok(BatchRow {
        record,
        planner_version: row
            .opt_text_at(7)?
            .map(|v| PlannerVersion::parse(&v))
            .transpose()
            .map_err(|_| corrupt("planner version"))?,
        sealed: row.bool_at(17)?,
        cancelled: row.u64_at(21)?,
        accounting_pending: row.i64_at(24)?,
        object_charge: row.i64_at(25)?,
        item_overhead: row.i64_at(26)?,
        max_records: row.i64_at(27)?,
        reserve_records: row.i64_at(28)?,
        staged_reserve: row.i64_at(29)?,
        completion_reserved: row.i64_at(30)?,
        export_reserved: row.i64_at(31)?,
        errors_bytes: row.i64_at(32)?,
        released: row.bool_at(37)?,
        cleanup_state: row.text_at(38)?,
        metadata_expires_at: row.opt_i64_at(39)?,
        phase_kind: row.opt_text_at(42)?,
        phase_claim_id: row.opt_text_at(43)?,
        phase_generation: row.i64_at(44)?,
        phase_lease_until: row.opt_i64_at(45)?,
        sched_credit: row.i64_at(46)?,
    })
}

/// Read a batch of an owner; `lock` takes the batch control row lock.
pub(in super::super) async fn load_batch<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    id: &BatchId,
    lock: bool,
) -> Result<Option<BatchRow>, StoreError> {
    let d = tx.dialect();
    let (hint, suffix) = if lock { (d.lock_hint, d.for_update) } else { ("", "") };
    let sql = format!("SELECT {BATCH_COLUMNS} FROM batches{hint} WHERE owner_id = $1 AND id = $2{suffix}");
    match tx.fetch_optional(&sql, args![owner.as_str(), id.as_str()]).await? {
        Some(row) => Ok(Some(batch_row(&row)?)),
        None => Ok(None),
    }
}

pub(in super::super) async fn lock_batch<T: SqlTx>(tx: &mut T, batch: &BatchRef) -> Result<BatchRow, StoreError> {
    load_batch(tx, &batch.owner, &batch.id, true).await?.ok_or(StoreError::NotFound)
}

// ---------------------------------------------------------------------------------------------------------------
// Policy and quota
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub(in super::super) struct PolicyRow {
    pub revision: i64,
    pub object_charge: i64,
    pub item_overhead: i64,
    pub max_upload: i64,
    pub max_records: i64,
}

pub(in super::super) async fn active_policy<T: SqlTx>(tx: &mut T) -> Result<PolicyRow, StoreError> {
    let d = tx.dialect();
    let sql = format!(
        "SELECT {}revision, object_charge_bytes, item_overhead_bytes, max_upload_bytes, max_records_per_batch \
         FROM batch_quota_policy ORDER BY revision DESC{}",
        d.top("1"),
        d.limit("1")
    );
    let row = tx.fetch_optional(&sql, args![]).await?.ok_or(StoreError::PolicyMissing)?;
    Ok(PolicyRow {
        revision: row.i64_at(0)?,
        object_charge: row.i64_at(1)?,
        item_overhead: row.i64_at(2)?,
        max_upload: row.i64_at(3)?,
        max_records: row.i64_at(4)?,
    })
}

pub(in super::super) async fn owner_limits<T: SqlTx>(
    tx: &mut T,
    revision: i64,
    owner: &OwnerId,
) -> Result<QuotaLimits, StoreError> {
    let row = tx
        .fetch_optional(
            "SELECT max_bytes, max_active_batches, max_retained_batches, max_files, max_open_upload_sessions \
             FROM batch_quota_limits WHERE revision = $1 AND owner_id = $2",
            args![revision, owner.as_str()],
        )
        .await?
        .ok_or(StoreError::OwnerNotProvisioned)?;
    Ok(QuotaLimits {
        max_bytes: row.u64_at(0)?,
        max_active_batches: row.u64_at(1)?,
        max_retained_batches: row.u64_at(2)?,
        max_files: row.u64_at(3)?,
        max_open_upload_sessions: row.u64_at(4)?,
    })
}

pub(in super::super) const QUOTA_COLUMNS: &str =
    "used_bytes, reserved_bytes, active_batches, retained_batches, files, open_sessions, version";

pub(in super::super) fn quota_usage<R: SqlRow>(row: &R) -> Result<QuotaUsage, StoreError> {
    Ok(QuotaUsage {
        used_bytes: row.u64_at(0)?,
        reserved_bytes: row.u64_at(1)?,
        active_batches: row.u64_at(2)?,
        retained_batches: row.u64_at(3)?,
        files: row.u64_at(4)?,
        open_sessions: row.u64_at(5)?,
        version: row.i64_at(6)?,
    })
}

/// Lock (creating if absent) the quota row of an owner: the first lock of every quota-changing operation.
pub(in super::super) async fn lock_quota<T: SqlTx>(tx: &mut T, owner: &OwnerId) -> Result<QuotaUsage, StoreError> {
    let d = tx.dialect();
    tx.execute(&d.insert_if_absent("batch_quota", &["owner_id"], &["$1"], 1), args![owner.as_str()]).await?;
    let sql = format!("SELECT {QUOTA_COLUMNS} FROM batch_quota{} WHERE owner_id = $1{}", d.lock_hint, d.for_update);
    let row = tx.fetch_optional(&sql, args![owner.as_str()]).await?.ok_or(corrupt("quota row"))?;
    quota_usage(&row)
}

/// Signed change of an owner's quota row.
#[derive(Clone, Copy, Debug, Default)]
pub(in super::super) struct QuotaDelta {
    pub used: i64,
    pub reserved: i64,
    pub active: i64,
    pub retained: i64,
    pub files: i64,
    pub sessions: i64,
}

impl QuotaDelta {
    fn is_zero(&self) -> bool {
        self.used == 0
            && self.reserved == 0
            && self.active == 0
            && self.retained == 0
            && self.files == 0
            && self.sessions == 0
    }
}

/// Apply a delta in the transaction that changes the owner row of those bytes; a negative result violates the
/// CHECK constraints and aborts the transaction instead of hiding drift.
pub(in super::super) async fn apply_quota<T: SqlTx>(
    tx: &mut T,
    owner: &OwnerId,
    delta: QuotaDelta,
) -> Result<(), StoreError> {
    if delta.is_zero() {
        return Ok(());
    }
    let changed = tx
        .execute(
            "UPDATE batch_quota SET used_bytes = used_bytes + $2, reserved_bytes = reserved_bytes + $3, \
             active_batches = active_batches + $4, retained_batches = retained_batches + $5, files = files + $6, \
             open_sessions = open_sessions + $7, version = version + 1 WHERE owner_id = $1",
            args![
                owner.as_str(),
                delta.used,
                delta.reserved,
                delta.active,
                delta.retained,
                delta.files,
                delta.sessions
            ],
        )
        .await
        .map_err(|error| match error {
            StoreError::Database(db) if db.kind == super::super::error::DbErrorKind::ConstraintViolation => {
                corrupt("quota would become negative")
            }
            other => other,
        })?;
    if changed != 1 {
        return Err(corrupt("quota row missing"));
    }
    Ok(())
}

/// Check every increasing dimension of `add` against the owner limits.
pub(in super::super) fn admit(usage: &QuotaUsage, limits: &QuotaLimits, add: &QuotaDelta) -> Result<(), StoreError> {
    let over = |current: u64, delta: i64, limit: u64| delta > 0 && current.saturating_add(delta.unsigned_abs()) > limit;
    let bytes_now = usage.used_bytes.saturating_add(usage.reserved_bytes);
    let bytes_delta = add.used.saturating_add(add.reserved);
    if over(bytes_now, bytes_delta, limits.max_bytes) {
        return Err(StoreError::QuotaExceeded(QuotaResource::Bytes));
    }
    if over(usage.active_batches, add.active, limits.max_active_batches) {
        return Err(StoreError::QuotaExceeded(QuotaResource::ActiveBatches));
    }
    if over(usage.retained_batches, add.retained, limits.max_retained_batches) {
        return Err(StoreError::QuotaExceeded(QuotaResource::RetainedBatches));
    }
    if over(usage.files, add.files, limits.max_files) {
        return Err(StoreError::QuotaExceeded(QuotaResource::Files));
    }
    if over(usage.open_sessions, add.sessions, limits.max_open_upload_sessions) {
        return Err(StoreError::QuotaExceeded(QuotaResource::UploadSessions));
    }
    Ok(())
}

/// Free bytes of an owner under its limits.
pub(in super::super) fn free_bytes(usage: &QuotaUsage, limits: &QuotaLimits) -> u64 {
    limits.max_bytes.saturating_sub(usage.used_bytes.saturating_add(usage.reserved_bytes))
}

pub(in super::super) fn signed(value: u64) -> Result<i64, StoreError> {
    to_db(value)
}

pub(in super::super) fn unsigned(value: i64) -> Result<u64, StoreError> {
    u64_from(value)
}

// ---------------------------------------------------------------------------------------------------------------
// Leases
// ---------------------------------------------------------------------------------------------------------------

/// Phase lease (discovery/validation/finalization) is held by this token at `now`.
pub(in super::super) fn phase_lease_valid(batch: &BatchRow, token: &ClaimToken, kind: LeaseKind, now: i64) -> bool {
    token.kind == kind
        && token.batch.owner == batch.record.owner
        && token.batch.id == batch.record.id
        && batch.phase_kind.as_deref() == Some(kind.as_db())
        && batch.phase_claim_id.as_deref() == Some(token.claim_id.as_str())
        && batch.phase_generation == token.generation
        && batch.phase_lease_until.is_some_and(|until| until >= now)
}

/// The phase slot is free or its lease has expired.
pub(in super::super) fn phase_free(batch: &BatchRow, now: i64) -> bool {
    batch.phase_claim_id.is_none() || batch.phase_lease_until.is_none_or(|until| until < now)
}

/// Finish the claim row of a phase lease and clear the batch phase slot (batch row must be locked).
pub(in super::super) async fn end_phase<T: SqlTx>(tx: &mut T, batch: &BatchRow) -> Result<(), StoreError> {
    if let Some(claim) = &batch.phase_claim_id {
        tx.execute(
            "UPDATE batch_claims SET state = 'finished' WHERE id = $1 AND state = 'active'",
            args![claim.as_str()],
        )
        .await?;
    }
    tx.execute(
        "UPDATE batches SET phase_kind = NULL, phase_claim_id = NULL, phase_lease_until = NULL \
         WHERE owner_id = $1 AND id = $2",
        args![batch.record.owner.as_str(), batch.record.id.as_str()],
    )
    .await?;
    Ok(())
}

/// Insert the claim row of a new lease.
#[expect(clippy::too_many_arguments, reason = "one row insert with its columns")]
pub(in super::super) async fn insert_claim<T: SqlTx>(
    tx: &mut T,
    claim_id: &str,
    kind: LeaseKind,
    worker: &str,
    generation: i64,
    batch: &BatchRef,
    control_epoch: i64,
    lease_until: i64,
    now: i64,
) -> Result<(), StoreError> {
    tx.execute(
        "INSERT INTO batch_claims (id, kind, worker_id, generation, owner_id, batch_id, control_epoch, lease_until, \
         state, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'active', $9)",
        args![
            claim_id,
            kind.as_db(),
            worker,
            generation,
            batch.owner.as_str(),
            batch.id.as_str(),
            control_epoch,
            lease_until,
            now
        ],
    )
    .await?;
    Ok(())
}

/// Count rows of a bounded query that returns one BIGINT.
pub(in super::super) async fn scalar<T: SqlTx>(
    tx: &mut T,
    sql: &str,
    args: Vec<super::super::sql::Val>,
) -> Result<i64, StoreError> {
    let row = tx.fetch_optional(sql, args).await?.ok_or(corrupt("aggregate returned no row"))?;
    row.i64_at(0)
}
