//! Mapping between MongoDB documents and the shared domain rows. Field names equal the SQL column names; integers
//! are Int64, timestamps are Int64 milliseconds, byte strings use the generic binary subtype.

use std::time::Duration;

use mongodb::bson::Document;

use super::super::error::StoreError;
use super::super::ops::common::{BatchRow, FileRow};
use super::super::types::{
    ArtifactRef, BatchId, BatchMetadata, BatchRecord, BatchRef, BatchStatus, FileId, FilePurpose, FileRecord,
    FileState, Fingerprint, ModelName, OwnerId, PlannerVersion, QuotaUsage, RequestCounts, Revision, Sha256Digest,
    TerminalStatus, UnixMillis,
};
use super::driver::{bin, corrupt, flag, int, opt_int, opt_text, text, uint};

pub(super) fn owner(value: String) -> Result<OwnerId, StoreError> {
    OwnerId::parse(&value).map_err(|_| corrupt("owner id"))
}

pub(super) fn batch_id(value: String) -> Result<BatchId, StoreError> {
    BatchId::parse(&value).map_err(|_| corrupt("batch id"))
}

pub(super) fn file_id(value: String) -> Result<FileId, StoreError> {
    FileId::parse(&value).map_err(|_| corrupt("file id"))
}

pub(super) fn artifact(value: String) -> Result<ArtifactRef, StoreError> {
    ArtifactRef::parse(&value).map_err(|_| corrupt("artifact ref"))
}

pub(super) fn digest(value: Vec<u8>) -> Result<Sha256Digest, StoreError> {
    Sha256Digest::from_slice(&value).map_err(|_| corrupt("sha256"))
}

pub(super) fn batch_ref(doc: &Document) -> Result<BatchRef, StoreError> {
    Ok(BatchRef { owner: owner(text(doc, "owner_id")?)?, id: batch_id(text(doc, "_id")?)? })
}

pub(super) fn file_row(doc: &Document) -> Result<FileRow, StoreError> {
    let purpose = match text(doc, "purpose")?.as_str() {
        "batch" => FilePurpose::Batch,
        "batch_output" => FilePurpose::BatchOutput,
        "batch_error" => FilePurpose::BatchError,
        _ => return Err(corrupt("file purpose")),
    };
    let state = match text(doc, "state")?.as_str() {
        "active" => FileState::Active,
        "deleted" => FileState::Deleted,
        _ => return Err(corrupt("file state")),
    };
    Ok(FileRow {
        record: FileRecord {
            owner: owner(text(doc, "owner_id")?)?,
            id: file_id(text(doc, "_id")?)?,
            purpose,
            filename: text(doc, "filename")?,
            artifact: artifact(text(doc, "artifact_ref")?)?,
            bytes: uint(doc, "bytes")?,
            sha256: digest(bin(doc, "sha256")?)?,
            created_at: UnixMillis(int(doc, "created_at")?),
            expires_at: opt_int(doc, "expires_at")?.map(UnixMillis),
            state,
        },
        pin_count: int(doc, "pin_count")?,
    })
}

fn metadata(doc: &Document) -> Result<BatchMetadata, StoreError> {
    let pairs: Vec<(String, String)> =
        serde_json::from_str(&text(doc, "metadata")?).map_err(|_| corrupt("batch metadata"))?;
    BatchMetadata::new(pairs).map_err(|_| corrupt("batch metadata"))
}

pub(super) fn batch_row(doc: &Document) -> Result<BatchRow, StoreError> {
    let status = BatchStatus::from_db(&text(doc, "status")?).ok_or(corrupt("batch status"))?;
    let terminal_target = match opt_text(doc, "terminal_target")? {
        Some(value) => Some(TerminalStatus::from_db(&value).ok_or(corrupt("terminal target"))?),
        None => None,
    };
    let failed = uint(doc, "failed")?;
    let expired = uint(doc, "expired")?;
    let record = BatchRecord {
        owner: owner(text(doc, "owner_id")?)?,
        id: batch_id(text(doc, "_id")?)?,
        input_file_id: file_id(text(doc, "input_file_id")?)?,
        endpoint: text(doc, "endpoint")?,
        catalog_revision: Revision(int(doc, "catalog_revision")?),
        model_name: opt_text(doc, "model_name")?
            .map(|v| ModelName::parse(&v))
            .transpose()
            .map_err(|_| corrupt("model"))?,
        fingerprint: opt_text(doc, "fingerprint")?
            .map(|v| Fingerprint::parse(&v))
            .transpose()
            .map_err(|_| corrupt("fingerprint"))?,
        status,
        terminal_target,
        created_at: UnixMillis(int(doc, "created_at")?),
        expires_at: UnixMillis(int(doc, "expires_at")?),
        in_progress_at: opt_int(doc, "in_progress_at")?.map(UnixMillis),
        finalizing_at: opt_int(doc, "finalizing_at")?.map(UnixMillis),
        cancelling_at: opt_int(doc, "cancelling_at")?.map(UnixMillis),
        terminal_at: opt_int(doc, "terminal_at")?.map(UnixMillis),
        control_epoch: int(doc, "control_epoch")?,
        counts: RequestCounts {
            total: uint(doc, "total")?,
            completed: uint(doc, "completed")?,
            failed: failed + expired,
        },
        output_file_id: opt_text(doc, "output_file_id")?.map(file_id).transpose()?,
        error_file_id: opt_text(doc, "error_file_id")?.map(file_id).transpose()?,
        metadata: metadata(doc)?,
        output_expires_after: opt_int(doc, "output_expires_after_secs")?
            .map(|s| Duration::from_secs(s.max(0).unsigned_abs())),
        error_count: uint(doc, "error_count")?,
        errors_truncated: flag(doc, "errors_truncated")?,
    };
    Ok(BatchRow {
        record,
        planner_version: opt_text(doc, "planner_version")?
            .map(|v| PlannerVersion::parse(&v))
            .transpose()
            .map_err(|_| corrupt("planner version"))?,
        sealed: flag(doc, "sealed")?,
        cancelled: uint(doc, "cancelled")?,
        accounting_pending: int(doc, "accounting_pending")?,
        object_charge: int(doc, "object_charge")?,
        item_overhead: int(doc, "item_overhead")?,
        max_records: int(doc, "max_records")?,
        reserve_records: int(doc, "reserve_records")?,
        staged_reserve: int(doc, "staged_reserve")?,
        completion_reserved: int(doc, "completion_reserved")?,
        export_reserved: int(doc, "export_reserved")?,
        errors_bytes: int(doc, "errors_bytes")?,
        released: flag(doc, "released")?,
        cleanup_state: text(doc, "cleanup_state")?,
        metadata_expires_at: opt_int(doc, "metadata_expires_at")?,
        phase_kind: opt_text(doc, "phase_kind")?,
        phase_claim_id: opt_text(doc, "phase_claim_id")?,
        phase_generation: int(doc, "phase_generation")?,
        phase_lease_until: opt_int(doc, "phase_lease_until")?,
        sched_credit: int(doc, "sched_credit")?,
    })
}

pub(super) fn quota_usage(doc: &Document) -> Result<QuotaUsage, StoreError> {
    Ok(QuotaUsage {
        used_bytes: uint(doc, "used_bytes")?,
        reserved_bytes: uint(doc, "reserved_bytes")?,
        active_batches: uint(doc, "active_batches")?,
        retained_batches: uint(doc, "retained_batches")?,
        files: uint(doc, "files")?,
        open_sessions: uint(doc, "open_sessions")?,
        version: int(doc, "version")?,
    })
}
