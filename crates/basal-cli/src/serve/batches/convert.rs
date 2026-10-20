//! Mappings between the store's domain records, the artifact store and the wire DTOs, and of their errors to
//! HTTP answers. Store errors never reach the client verbatim: missing and foreign resources are one 404, driver
//! failures a 503, and internal inconsistencies a 500 without details.

use super::artifacts::{self, ArtifactError, ArtifactKey};
use super::dto::{
    self, ApiError, BatchErrorEntry, BatchErrors, BatchObject, FileObject, Metadata, RequestCounts, Resource, Throttle,
};
use super::store::{self, BatchIssue, BatchRecord, BatchStatus, FileRecord, StoreError, TerminalStatus};

/// Seconds a client should wait after a 503 or 429 of the Batch API.
pub(crate) const RETRY_AFTER_S: u32 = 5;

/// HTTP answer of a store error for an operation on `resource`.
pub(crate) fn store_error(error: &StoreError, resource: Resource) -> ApiError {
    match error {
        StoreError::NotFound => ApiError::not_found(resource),
        StoreError::FileInUse => ApiError::file_in_use(),
        StoreError::IdempotencyConflict => ApiError::idempotency_conflict(),
        StoreError::QuotaExceeded(_) => ApiError::throttled(Throttle::Quota, RETRY_AFTER_S),
        StoreError::InputUnusable(reason) => {
            ApiError::invalid_request(format!("The input file cannot be used: {reason}."))
                .with_param("input_file_id")
                .with_code("invalid_input_file")
        }
        StoreError::Invalid(value) => ApiError::invalid_request(value.to_string()),
        StoreError::SessionClosed => {
            ApiError::invalid_request("The upload took longer than its session allows.").with_code("upload_expired")
        }
        e if e.is_unavailable() => ApiError::unavailable(RETRY_AFTER_S),
        StoreError::OwnerNotProvisioned
        | StoreError::PolicyMissing
        | StoreError::CatalogMissing
        | StoreError::Config(_) => ApiError::unavailable(RETRY_AFTER_S),
        _ => ApiError::internal(),
    }
}

/// HTTP answer of an artifact error.
pub(crate) fn artifact_error(error: &ArtifactError) -> ApiError {
    match error {
        ArtifactError::LimitExceeded { limit } => ApiError::payload_too_large(*limit),
        ArtifactError::DeadlineExceeded => {
            ApiError::invalid_request("The upload took longer than allowed.").with_code("upload_expired")
        }
        e if e.is_unavailable() => ApiError::unavailable(RETRY_AFTER_S),
        _ => ApiError::internal(),
    }
}

/// The store's reference of an artifact key (its text form).
pub(crate) fn artifact_ref(key: &ArtifactKey) -> Result<store::ArtifactRef, StoreError> {
    store::ArtifactRef::parse(&key.to_string()).map_err(StoreError::Invalid)
}

/// The artifact key of a stored reference; a reference this server did not write is refused.
pub(crate) fn artifact_key(reference: &store::ArtifactRef) -> Result<ArtifactKey, ArtifactError> {
    ArtifactKey::parse(reference.as_str())
}

/// The store's form of a finished object.
pub(crate) fn stored(object: &artifacts::StoredObject) -> store::StoredObject {
    store::StoredObject { bytes: object.len, sha256: store::Sha256Digest(object.sha256.0) }
}

/// The artifact store's form of a recorded object.
pub(crate) fn object(
    reference: &store::ArtifactRef,
    recorded: &store::StoredObject,
) -> Result<artifacts::StoredObject, ArtifactError> {
    Ok(artifacts::StoredObject {
        key: artifact_key(reference)?,
        len: recorded.bytes,
        sha256: artifacts::Sha256Digest(recorded.sha256.0),
    })
}

pub(crate) fn file_object(file: &FileRecord) -> FileObject {
    FileObject {
        id: file.id.as_str().to_string(),
        object: dto::FILE,
        bytes: file.bytes,
        created_at: file.created_at.as_unix_seconds(),
        expires_at: file.expires_at.map(store::UnixMillis::as_unix_seconds),
        filename: file.filename.clone(),
        purpose: match file.purpose {
            store::FilePurpose::Batch => dto::FilePurpose::Batch,
            // Output and error files are both `batch_output` publicly (OpenAI FileObject); the kind is internal.
            store::FilePurpose::BatchOutput | store::FilePurpose::BatchError => dto::FilePurpose::BatchOutput,
        },
        status: "processed",
        status_details: None,
    }
}

/// Store filter of a public purpose. `batch_output` lists every result file: the store records output and error
/// files with purpose `batch_output`.
pub(crate) fn file_purpose(purpose: dto::FilePurpose) -> store::FilePurpose {
    match purpose {
        dto::FilePurpose::Batch => store::FilePurpose::Batch,
        dto::FilePurpose::BatchOutput => store::FilePurpose::BatchOutput,
    }
}

/// The public resource of a batch with its stored validation issues (`issues` when `error_count > 0`).
pub(crate) fn batch_object(batch: &BatchRecord, issues: &[BatchIssue]) -> BatchObject {
    let at = |t: Option<store::UnixMillis>| t.map(store::UnixMillis::as_unix_seconds);
    let terminal = |status: BatchStatus| if batch.status == status { at(batch.terminal_at) } else { None };
    let errors = (!issues.is_empty() || batch.errors_truncated).then(|| {
        let mut data: Vec<BatchErrorEntry> = issues
            .iter()
            .map(|i| BatchErrorEntry {
                code: i.code.clone(),
                message: i.message.clone(),
                param: i.param.clone(),
                line: i.line,
            })
            .collect();
        if batch.errors_truncated && data.last().is_none_or(|e| e.code != dto::codes::ERRORS_TRUNCATED) {
            data.push(BatchErrorEntry {
                code: dto::codes::ERRORS_TRUNCATED.into(),
                message: "More errors were found; the list stops here.".into(),
                param: None,
                line: None,
            });
        }
        BatchErrors { object: dto::LIST, data }
    });
    let cancelling =
        batch.terminal_target == Some(TerminalStatus::Cancelled) || matches!(batch.status, BatchStatus::Cancelled);
    BatchObject {
        id: batch.id.as_str().to_string(),
        object: dto::BATCH,
        endpoint: batch.endpoint.clone(),
        model: batch.model_name.as_ref().map(|m| m.as_str().to_string()),
        errors,
        input_file_id: batch.input_file_id.as_str().to_string(),
        completion_window: dto::COMPLETION_WINDOW_NAME,
        status: batch.status.public_name(batch.terminal_target),
        output_file_id: batch.output_file_id.as_ref().map(|f| f.as_str().to_string()),
        error_file_id: batch.error_file_id.as_ref().map(|f| f.as_str().to_string()),
        created_at: batch.created_at.as_unix_seconds(),
        in_progress_at: at(batch.in_progress_at),
        expires_at: batch.expires_at.as_unix_seconds(),
        finalizing_at: at(batch.finalizing_at),
        completed_at: terminal(BatchStatus::Completed),
        failed_at: terminal(BatchStatus::Failed),
        expired_at: terminal(BatchStatus::Expired),
        cancelling_at: if cancelling { at(batch.cancelling_at) } else { None },
        cancelled_at: terminal(BatchStatus::Cancelled),
        request_counts: RequestCounts {
            total: batch.counts.total,
            completed: batch.counts.completed,
            failed: batch.counts.failed,
        },
        metadata: (!batch.metadata.pairs().is_empty()).then(|| Metadata(batch.metadata.pairs().to_vec())),
    }
}

/// A bounded validation issue of one line.
pub(crate) fn issue(code: &str, message: &str, line: Option<u64>) -> BatchIssue {
    BatchIssue {
        code: dto::truncate_utf8(code, BatchIssue::MAX_CODE_BYTES).to_string(),
        message: dto::truncate_utf8(message, BatchIssue::MAX_MESSAGE_BYTES).to_string(),
        param: None,
        line,
    }
}
