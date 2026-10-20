//! Durable Batch/Files API. Resources require a bearer token configured through `ClientOptions`.
//! Input and output content stays raw JSONL bytes; the SDK never executes tasks or polls automatically.

use std::hash::{BuildHasher, Hasher};

use http::{
    header::{HeaderMap, HeaderValue, ACCEPT},
    Method,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Client, DecisionRequest, Error};

/// One stored input or result file. Timestamps are Unix seconds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileObject {
    /// Server-generated file id.
    pub id: String,
    /// Resource kind (`file`).
    pub object: String,
    /// Stored content length.
    pub bytes: u64,
    /// Creation time.
    pub created_at: i64,
    /// Optional expiry time.
    pub expires_at: Option<i64>,
    /// User-supplied name; not a local path.
    pub filename: String,
    /// `batch` for inputs, `batch_output` for results.
    pub purpose: String,
    /// Compatibility status (`processed`). Validation happens when a batch is created.
    pub status: String,
    /// Optional server details.
    pub status_details: Option<Value>,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Confirmation of a file deletion.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeletedFile {
    /// Deleted file id.
    pub id: String,
    /// Resource kind (`file`).
    pub object: String,
    /// Deletion confirmation.
    pub deleted: bool,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One cursor page, ordered by creation time and id descending.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchPage<T> {
    /// Resource kind (`list`).
    pub object: String,
    /// Resources in this page.
    pub data: Vec<T>,
    /// First id, absent on an empty page.
    pub first_id: Option<String>,
    /// Pass as `after` to request the next page when `has_more` is true.
    pub last_id: Option<String>,
    /// Whether another page exists.
    pub has_more: bool,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Limits and cursor of one list request. The SDK never auto-paginates.
#[derive(Clone, Debug, Default)]
pub struct BatchListOptions<'a> {
    /// Cursor of the same resource kind.
    pub after: Option<&'a str>,
    /// Page size 1–100 (server default 20).
    pub limit: Option<u32>,
}

/// Optional upload retention. The input itself is always uploaded with purpose `batch`.
#[derive(Clone, Debug, Default)]
pub struct FileUploadOptions {
    /// Lifetime from file creation, 3600–2592000 seconds; absent uses the server default.
    pub expires_after_seconds: Option<u32>,
}

/// Retention of each output/error file from its creation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchExpiresAfter {
    /// Supported anchor: `created_at`.
    pub anchor: String,
    /// Lifetime, 3600–2592000 seconds.
    pub seconds: u32,
}

impl BatchExpiresAfter {
    /// Retain each result file for `seconds` after its creation.
    pub fn created_at(seconds: u32) -> Self {
        Self { anchor: "created_at".into(), seconds }
    }
}

/// Create a batch from an uploaded JSONL input. Use [`Self::new`] for native defaults.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateBatchRequest {
    /// Uploaded input file.
    pub input_file_id: String,
    /// Only `/v1/basal` is currently supported.
    pub endpoint: String,
    /// Only `24h` is currently supported; this is an expiry window, not a completion guarantee.
    pub completion_window: String,
    /// Up to 16 string pairs; server validates key/value limits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::BTreeMap<String, String>>,
    /// Optional result-file retention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_expires_after: Option<BatchExpiresAfter>,
}

impl CreateBatchRequest {
    /// Native `/v1/basal` batch with a 24-hour inference window.
    pub fn new(input_file_id: impl Into<String>) -> Self {
        Self {
            input_file_id: input_file_id.into(),
            endpoint: "/v1/basal".into(),
            completion_window: "24h".into(),
            metadata: None,
            output_expires_after: None,
        }
    }
}

/// Counts after validation. Failed also includes expired records.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    /// Fully validated record count (zero until validation finishes).
    pub total: u64,
    /// Successful records.
    pub completed: u64,
    /// Failed or expired records.
    pub failed: u64,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A batch-level validation error.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchErrorEntry {
    /// Server error code.
    pub code: String,
    /// Error description.
    pub message: String,
    /// Affected parameter when known.
    pub param: Option<String>,
    /// One-based input line when known.
    pub line: Option<u64>,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Bounded list of validation errors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchErrors {
    /// Resource kind (`list`).
    pub object: String,
    /// Validation errors.
    pub data: Vec<BatchErrorEntry>,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Durable batch state; nullable timestamps are Unix seconds. Future status strings remain readable.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Batch {
    /// Batch id.
    pub id: String,
    /// Resource kind (`batch`).
    pub object: String,
    /// Native target endpoint.
    pub endpoint: String,
    /// Effective model, once resolved.
    pub model: Option<String>,
    /// Validation errors, when present.
    pub errors: Option<BatchErrors>,
    /// Uploaded JSONL input id.
    pub input_file_id: String,
    /// Inference window (`24h`).
    pub completion_window: String,
    /// validating/failed/in_progress/finalizing/completed/expired/cancelling/cancelled.
    pub status: String,
    /// Successful results file, when published.
    pub output_file_id: Option<String>,
    /// Errors file, when published.
    pub error_file_id: Option<String>,
    /// Creation time.
    pub created_at: i64,
    /// Validation completed and processing became possible.
    pub in_progress_at: Option<i64>,
    /// End of inference window.
    pub expires_at: i64,
    /// Export started.
    pub finalizing_at: Option<i64>,
    /// Completed time.
    pub completed_at: Option<i64>,
    /// Failed time.
    pub failed_at: Option<i64>,
    /// Expired time.
    pub expired_at: Option<i64>,
    /// Cancellation requested time.
    pub cancelling_at: Option<i64>,
    /// Cancellation finalized time.
    pub cancelled_at: Option<i64>,
    /// Validated/completed/failed counts.
    pub request_counts: BatchRequestCounts,
    /// Owner-provided metadata.
    pub metadata: Option<std::collections::BTreeMap<String, String>>,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One JSONL input record. Serialize each record followed by LF; upload exact bytes if already encoded.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchInputLine {
    /// Caller id, unique within the file.
    pub custom_id: String,
    /// `POST`.
    pub method: String,
    /// `/v1/basal`.
    pub url: String,
    /// Native request body.
    pub body: DecisionRequest,
}

impl BatchInputLine {
    /// Wrap a native request for a JSONL batch input.
    pub fn new(custom_id: impl Into<String>, body: DecisionRequest) -> Self {
        Self { custom_id: custom_id.into(), method: "POST".into(), url: "/v1/basal".into(), body }
    }
}

/// One result JSONL line; native success and error response bodies remain arbitrary JSON objects.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResultLine {
    /// Generated request id.
    pub id: String,
    /// Original caller id.
    pub custom_id: String,
    /// An executed native request response.
    pub response: Option<BatchResultResponse>,
    /// A record which could not be executed or retained.
    pub error: Option<BatchResultError>,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Response portion of a result line.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResultResponse {
    /// Native HTTP status.
    pub status_code: u16,
    /// Generated request id.
    pub request_id: String,
    /// Native response (including native error bodies).
    pub body: Value,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Non-execution error of a result record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchResultError {
    /// Error code, including `batch_expired`.
    pub code: String,
    /// Description.
    pub message: String,
    /// Future response fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn id_component(id: &str, prefix: &str) -> Result<(), Error> {
    let suffix = id.strip_prefix(prefix).ok_or(Error::InvalidRequest("invalid resource id"))?;
    if suffix.len() != 32 || !suffix.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(Error::InvalidRequest("invalid resource id"));
    }
    Ok(())
}

fn list_path(base: &str, options: &BatchListOptions<'_>, prefix: &str, purpose: Option<&str>) -> Result<String, Error> {
    let mut params = Vec::new();
    if let Some(after) = options.after {
        id_component(after, prefix)?;
        params.push(format!("after={after}"));
    }
    if let Some(limit) = options.limit {
        if !(1..=100).contains(&limit) {
            return Err(Error::InvalidRequest("list limit must be 1-100"));
        }
        params.push(format!("limit={limit}"));
    }
    if let Some(purpose) = purpose {
        if !matches!(purpose, "batch" | "batch_output") {
            return Err(Error::InvalidRequest("invalid file purpose"));
        }
        params.push(format!("purpose={purpose}"));
    }
    Ok(if params.is_empty() { base.into() } else { format!("{base}?{}", params.join("&")) })
}

impl Client {
    /// Upload caller-owned JSONL bytes (at most 200 MB); memory includes a multipart copy of the input.
    /// No file paths or URLs are opened. Expiry fields precede the file part.
    pub async fn upload_file(
        &self,
        filename: &str,
        data: &[u8],
        options: FileUploadOptions,
    ) -> Result<FileObject, Error> {
        if filename.is_empty() || filename.len() > 255 || filename.chars().any(char::is_control) {
            return Err(Error::InvalidRequest("filename must be 1-255 bytes without control characters"));
        }
        if data.len() > 200_000_000 {
            return Err(Error::InvalidRequest("input file exceeds 200 MB"));
        }
        if options.expires_after_seconds.is_some_and(|s| !(3600..=2_592_000).contains(&s)) {
            return Err(Error::InvalidRequest("file expiry must be 3600-2592000 seconds"));
        }
        // A per-call randomized standard-library hasher provides an unpredictable boundary without adding a
        // dependency. Check the actual input for collisions before constructing the multipart body.
        let random = std::collections::hash_map::RandomState::new();
        let mut counter = 0u64;
        let boundary = loop {
            let mut h = random.build_hasher();
            h.write_u64(counter);
            let candidate = format!("basal-sdk-{:016x}-{counter:016x}", h.finish());
            if !data.windows(candidate.len()).any(|w| w == candidate.as_bytes()) {
                break candidate;
            }
            counter = counter.checked_add(1).ok_or(Error::InvalidRequest("multipart boundary unavailable"))?;
        };
        let mut form = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n");
        if let Some(seconds) = options.expires_after_seconds {
            for (name, value) in
                [("expires_after[anchor]", "created_at".to_string()), ("expires_after[seconds]", seconds.to_string())]
            {
                form.push_str(&format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
                ));
            }
        }
        let filename = filename.replace('\\', "\\\\").replace('"', "\\\"");
        form.push_str(&format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: application/jsonl\r\n\r\n"));
        let mut body = form.into_bytes();
        body.extend_from_slice(data);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let content_type = format!("multipart/form-data; boundary={boundary}");
        let bytes =
            self.request_bytes(Method::POST, "v1/files", Some(body), Some(&content_type), HeaderMap::new()).await?;
        serde_json::from_slice(&bytes).map_err(Error::Protocol)
    }

    /// Read one page of files; optional purpose is `batch` or `batch_output`.
    pub async fn list_files(
        &self,
        options: BatchListOptions<'_>,
        purpose: Option<&str>,
    ) -> Result<BatchPage<FileObject>, Error> {
        self.send(Method::GET, &list_path("v1/files", &options, "file_", purpose)?, None).await
    }
    /// Retrieve file metadata.
    pub async fn get_file(&self, id: &str) -> Result<FileObject, Error> {
        id_component(id, "file_")?;
        self.send(Method::GET, &format!("v1/files/{id}"), None).await
    }
    /// Delete a file (the server may retain physical bytes until guarded cleanup).
    pub async fn delete_file(&self, id: &str) -> Result<DeletedFile, Error> {
        id_component(id, "file_")?;
        self.send(Method::DELETE, &format!("v1/files/{id}"), None).await
    }
    /// Download exact JSONL bytes, bounded by `ClientOptions::max_response_bytes` (default 8 MiB).
    /// Increase that explicit cap when downloading larger files. Dropping the future cancels I/O.
    pub async fn download_file(&self, id: &str) -> Result<Vec<u8>, Error> {
        id_component(id, "file_")?;
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
        self.request_bytes(Method::GET, &format!("v1/files/{id}/content"), None, None, headers).await
    }
    /// Create a batch; an optional owner-scoped idempotency key is 1–128 visible ASCII bytes.
    /// Retrying is the caller's decision; reuse the same key only with the same request.
    pub async fn create_batch(
        &self,
        request: &CreateBatchRequest,
        idempotency_key: Option<&str>,
    ) -> Result<Batch, Error> {
        let mut headers = HeaderMap::new();
        if let Some(key) = idempotency_key {
            if key.is_empty() || key.len() > 128 || !key.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
                return Err(Error::InvalidRequest("invalid idempotency key"));
            }
            let value = HeaderValue::from_str(key).map_err(|_| Error::InvalidRequest("invalid idempotency key"))?;
            headers.insert("idempotency-key", value);
        }
        let bytes = self
            .request_bytes(
                Method::POST,
                "v1/batches",
                Some(serde_json::to_vec(request).map_err(Error::Encode)?),
                Some("application/json"),
                headers,
            )
            .await?;
        serde_json::from_slice(&bytes).map_err(Error::Protocol)
    }
    /// Read one page of batches.
    pub async fn list_batches(&self, options: BatchListOptions<'_>) -> Result<BatchPage<Batch>, Error> {
        self.send(Method::GET, &list_path("v1/batches", &options, "batch_", None)?, None).await
    }
    /// Retrieve a batch without polling.
    pub async fn get_batch(&self, id: &str) -> Result<Batch, Error> {
        id_component(id, "batch_")?;
        self.send(Method::GET, &format!("v1/batches/{id}"), None).await
    }
    /// Request cancellation; partial result files may be published later.
    pub async fn cancel_batch(&self, id: &str) -> Result<Batch, Error> {
        id_component(id, "batch_")?;
        self.send(Method::POST, &format!("v1/batches/{id}/cancel"), None).await
    }
    /// Read Batch health. The server exposes this route without authentication.
    pub async fn batch_health(&self) -> Result<Value, Error> {
        self.send(Method::GET, "health/batches", None).await
    }
}
