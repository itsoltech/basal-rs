//! Wire contract of the Batch and Files routes (OpenAI-like; `contracts/batches/openapi.json`).
//!
//! Resource ids are the store's random UUID v4 ids with a prefix (`file_`, `batch_`; record lines use
//! `batch_req_`); they encode no machine, path or owner. Times are Unix seconds. Errors are
//! `{"error": {message, type, param, code}}`. Requests are parsed strictly: unknown fields and parameters are refused,
//! never ignored. Only the native `/v1/basal` endpoint is accepted; System One batches are refused explicitly.
//!
//! Record result lines keep the native response: its body is the `serde_json::Value` of the online endpoint,
//! written with the same serializer (key order and number text preserved), never re-normalized by the database.

use std::borrow::Cow;

use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::config::{EXPIRES_AFTER_MAX_S, EXPIRES_AFTER_MIN_S};
use super::store::{FileId, RESULT_MAX};

/// Largest body of a control request (create, cancel).
pub(crate) const MAX_CONTROL_BODY_BYTES: usize = 64 * 1024;
/// Largest text field of the upload form (`purpose`, `expires_after[...]`).
pub(crate) const MAX_FORM_TEXT_BYTES: usize = 256;
/// Longest upload file name, in bytes.
const MAX_FILENAME_BYTES: usize = 255;
const MAX_METADATA_PAIRS: usize = 16;
const MAX_METADATA_KEY_CHARS: usize = 64;
const MAX_METADATA_VALUE_CHARS: usize = 512;
/// The only accepted `endpoint` of a batch and `url` of its records.
pub(crate) const BASAL_ENDPOINT: &str = "/v1/basal";
const SYSTEMONE_ENDPOINT: &str = "/v1/systemone";
/// The only accepted `completion_window`.
const COMPLETION_WINDOW: &str = "24h";
const LIST_DEFAULT_LIMIT: u32 = 20;
const LIST_MAX_LIMIT: u32 = 100;
const MAX_ERROR_TEXT_BYTES: usize = 256;
/// Longest accepted list cursor (resource ids are 37–38 bytes).
const MAX_CURSOR_BYTES: usize = 64;

/// A new `batch_req_<32 hex>` id of one record line (random UUID v4).
pub(crate) fn new_request_id() -> String {
    format!("batch_req_{}", uuid::Uuid::new_v4().simple())
}

// ---------------------------------------------------------------------------------------------------------------
// Errors

/// Resource named by a uniform 404.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Resource {
    File,
    Batch,
}

/// An OpenAI-like error answer: `{"error": {"message", "type", "param", "code"}}` with its HTTP status (and
/// `Retry-After` for 429/503). Messages carry no secrets, paths or other owners' data.
#[derive(Clone, Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: Cow<'static, str>,
    param: Option<Cow<'static, str>>,
    code: Option<&'static str>,
    retry_after_s: Option<u32>,
}

/// Why a 429 is answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Throttle {
    /// The owner's quota (bytes, active/retained batches, files, upload sessions) is used up.
    Quota,
    /// Too many concurrent uploads, downloads or store operations.
    Concurrency,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<Cow<'static, str>>) -> Self {
        Self { status, kind, message: message.into(), param: None, code: None, retry_after_s: None }
    }

    pub(crate) fn with_param(mut self, param: impl Into<Cow<'static, str>>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub(crate) fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    /// `Retry-After` of at least one second.
    fn with_retry_after(mut self, retry_after_s: u32) -> Self {
        self.retry_after_s = Some(retry_after_s.max(1));
        self
    }

    /// 400 `invalid_request_error`; the message is cut to 256 bytes.
    pub(crate) fn invalid_request(message: impl Into<Cow<'static, str>>) -> Self {
        let message = match message.into() {
            Cow::Owned(s) => Cow::Owned(truncate_utf8(&s, MAX_ERROR_TEXT_BYTES).to_string()),
            m => m,
        };
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }

    /// 400 for a method the Batch/Files API does not have on a route.
    pub(crate) fn unsupported_route() -> Self {
        Self::invalid_request("Unsupported route or method for the Batch API.").with_code("invalid_route")
    }

    /// 404: the same answer for a missing resource, another owner's resource and a malformed id.
    pub(crate) fn not_found(resource: Resource) -> Self {
        let name = match resource {
            Resource::File => "File",
            Resource::Batch => "Batch",
        };
        Self::new(StatusCode::NOT_FOUND, "invalid_request_error", format!("No such {name} object."))
    }

    /// 413: the upload exceeds `limit` bytes (also detected without `Content-Length`).
    pub(crate) fn payload_too_large(limit: u64) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, "invalid_request_error", format!("The file exceeds {limit} bytes."))
            .with_code("file_too_large")
    }

    /// 408: the upload's total read deadline elapsed, including multipart framing and metadata.
    pub(crate) fn upload_timeout() -> Self {
        Self::new(StatusCode::REQUEST_TIMEOUT, "invalid_request_error", "The upload exceeded its time limit.")
            .with_code("upload_timeout")
    }

    /// 413: multipart framing or a transport chunk exceeds the admitted parser-buffer capacity.
    pub(crate) fn multipart_too_large(limit: u64) -> Self {
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            format!("Multipart parsing exceeds the {limit} byte upload buffer."),
        )
        .with_code("multipart_too_large")
    }

    /// 400: a list cursor that is malformed or not an id of the listed objects (also missing or another owner's).
    pub(crate) fn bad_cursor() -> Self {
        Self::invalid_request("`after` must be an id of the listed objects.").with_param("after")
    }

    /// 409: an `Idempotency-Key` reused with different request data.
    pub(crate) fn idempotency_conflict() -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "The Idempotency-Key was already used with different request data.",
        )
        .with_param("Idempotency-Key")
        .with_code("idempotency_key_reused")
    }

    /// 409: DELETE of an input file pinned by an active batch.
    pub(crate) fn file_in_use() -> Self {
        Self::new(StatusCode::CONFLICT, "invalid_request_error", "The file is the input of an active batch.")
            .with_code("file_in_use")
    }

    /// 429 with `Retry-After`; the online 529 overload answer is separate and unchanged.
    pub(crate) fn throttled(why: Throttle, retry_after_s: u32) -> Self {
        let (code, message) = match why {
            Throttle::Quota => ("quota_exceeded", "The batch quota of this owner is used up."),
            Throttle::Concurrency => ("concurrency_limit_exceeded", "Too many concurrent batch operations."),
        };
        Self::new(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error", message)
            .with_code(code)
            .with_retry_after(retry_after_s)
    }

    /// 503: the batch database or artifact storage is unavailable, or its schema or policy differs (inference is
    /// unaffected).
    pub(crate) fn unavailable(retry_after_s: u32) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", "The Batch API is unavailable.")
            .with_code("batch_service_unavailable")
            .with_retry_after(retry_after_s)
    }

    /// 500 without details (they go to the server log, not to the client).
    pub(crate) fn internal() -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", "Internal error of the Batch API.")
    }

    fn body(&self) -> Value {
        serde_json::json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }})
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.status.as_u16(), self.message)
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(self.body())).into_response();
        if let Some(s) = self.retry_after_s {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from(s));
        }
        response
    }
}

/// `s` cut to at most `max` bytes at a character boundary.
pub(crate) fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ---------------------------------------------------------------------------------------------------------------
// Files

/// `purpose` of a file (OpenAI FileObject values). Uploads accept only `batch`; both result files of a batch (output
/// and error lines) are `batch_output`, written by the server and read-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FilePurpose {
    Batch,
    BatchOutput,
}

/// A constant `"object"` value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tag(&'static str);

impl Serialize for Tag {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.0)
    }
}

pub(crate) const FILE: Tag = Tag("file");
pub(crate) const BATCH: Tag = Tag("batch");
pub(crate) const LIST: Tag = Tag("list");

/// A file resource (inputs and results share the format).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct FileObject {
    pub(crate) id: String,
    pub(crate) object: Tag,
    pub(crate) bytes: u64,
    pub(crate) created_at: i64,
    pub(crate) expires_at: Option<i64>,
    /// As given at upload (inputs) or `batch_<id>_output.jsonl` / `batch_<id>_error.jsonl` (results); never a path.
    pub(crate) filename: String,
    pub(crate) purpose: FilePurpose,
    /// Always `processed` (kept for OpenAI SDK compatibility; the field is deprecated there).
    pub(crate) status: &'static str,
    pub(crate) status_details: Option<String>,
}

/// Answer of `DELETE /v1/files/{id}` (logical deletion; physical removal follows after the download grace).
#[derive(Clone, Debug, Serialize)]
pub(crate) struct DeletedFile {
    pub(crate) id: String,
    pub(crate) object: Tag,
    pub(crate) deleted: bool,
}

/// Name of a result file of a batch.
pub(crate) fn result_filename(batch: &str, output: bool) -> String {
    format!("{batch}_{}.jsonl", if output { "output" } else { "error" })
}

/// Check the file name of an upload: 1–255 bytes without control characters. It is stored and returned as given,
/// never used as a path.
pub(crate) fn check_filename(name: Option<&str>) -> Result<String, ApiError> {
    let name = name.ok_or_else(|| {
        ApiError::invalid_request("The `file` part needs a filename.").with_param("file").with_code("missing_filename")
    })?;
    if name.is_empty() || name.len() > MAX_FILENAME_BYTES || name.chars().any(char::is_control) {
        return Err(ApiError::invalid_request("The filename must have 1-255 bytes without control characters.")
            .with_param("file")
            .with_code("invalid_filename"));
    }
    Ok(name.to_string())
}

/// Fields of the multipart upload form; any other field is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UploadField {
    File,
    Purpose,
    ExpiresAfterAnchor,
    ExpiresAfterSeconds,
}

impl UploadField {
    pub(crate) fn classify(name: Option<&str>) -> Result<Self, ApiError> {
        match name {
            Some("file") => Ok(Self::File),
            Some("purpose") => Ok(Self::Purpose),
            Some("expires_after[anchor]") => Ok(Self::ExpiresAfterAnchor),
            Some("expires_after[seconds]") => Ok(Self::ExpiresAfterSeconds),
            Some(other) => Err(ApiError::invalid_request(format!(
                "Unknown upload field `{}`.",
                truncate_utf8(other, 64).escape_debug()
            ))
            .with_code("unknown_parameter")),
            None => Err(ApiError::invalid_request("Every upload field needs a name.").with_code("unknown_parameter")),
        }
    }
}

/// `purpose` of an upload: only `batch`.
pub(crate) fn check_upload_purpose(value: &str) -> Result<(), ApiError> {
    match value {
        "batch" => Ok(()),
        "batch_output" => {
            Err(ApiError::invalid_request("Result files are written by the server; upload with purpose `batch`.")
                .with_param("purpose")
                .with_code("invalid_purpose"))
        }
        _ => Err(ApiError::invalid_request("Only purpose `batch` is supported.")
            .with_param("purpose")
            .with_code("invalid_purpose")),
    }
}

/// `expires_after[anchor]`: only `created_at`.
pub(crate) fn check_expires_anchor(value: &str, param: &'static str) -> Result<(), ApiError> {
    if value == "created_at" {
        Ok(())
    } else {
        Err(ApiError::invalid_request("The expiry anchor must be `created_at`.").with_param(param))
    }
}

/// `expires_after[seconds]`: 3600–2592000.
pub(crate) fn parse_expires_seconds(value: &str, param: &'static str) -> Result<u64, ApiError> {
    let s: u64 = value
        .parse()
        .map_err(|_| ApiError::invalid_request("The expiry must be a whole number of seconds.").with_param(param))?;
    check_expires_seconds(s, param)
}

fn check_expires_seconds(s: u64, param: &'static str) -> Result<u64, ApiError> {
    if !(EXPIRES_AFTER_MIN_S..=EXPIRES_AFTER_MAX_S).contains(&s) {
        return Err(ApiError::invalid_request(format!(
            "The expiry must be within {EXPIRES_AFTER_MIN_S}..={EXPIRES_AFTER_MAX_S} seconds."
        ))
        .with_param(param));
    }
    Ok(s)
}

// ---------------------------------------------------------------------------------------------------------------
// Batches

/// `request_counts`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub(crate) struct RequestCounts {
    pub(crate) total: u64,
    pub(crate) completed: u64,
    pub(crate) failed: u64,
}

/// One entry of `errors.data` (validation of the input file, or why a batch failed).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct BatchErrorEntry {
    pub(crate) code: String,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
    /// 1-based line of the input file.
    pub(crate) line: Option<u64>,
}

/// `errors`: `{"object": "list", "data": [...]}`.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct BatchErrors {
    pub(crate) object: Tag,
    pub(crate) data: Vec<BatchErrorEntry>,
}

/// Error codes of `errors.data` and of record lines (stable wire values).
pub(crate) mod codes {
    pub(crate) const EMPTY_FILE: &str = "empty_file";
    pub(crate) const EMPTY_LINE: &str = "empty_line";
    pub(crate) const INVALID_JSON_LINE: &str = "invalid_json_line";
    pub(crate) const LINE_TOO_LARGE: &str = "line_too_large";
    pub(crate) const TOO_MANY_RECORDS: &str = "too_many_records";
    pub(crate) const INVALID_CUSTOM_ID: &str = "invalid_custom_id";
    pub(crate) const DUPLICATE_CUSTOM_ID: &str = "duplicate_custom_id";
    pub(crate) const INVALID_METHOD: &str = "invalid_method";
    pub(crate) const INVALID_URL: &str = "invalid_url";
    pub(crate) const STREAM_UNSUPPORTED: &str = "stream_unsupported";
    pub(crate) const INVALID_BODY: &str = "invalid_body";
    pub(crate) const MODEL_NOT_FOUND: &str = "model_not_found";
    pub(crate) const MIXED_MODELS: &str = "mixed_models";
    pub(crate) const BUDGET_EXCEEDED: &str = "budget_exceeded";
    pub(crate) const HOST_BUDGET_EXCEEDED: &str = "host_budget_exceeded";
    pub(crate) const UNBOUNDED_FEATURE: &str = "unbounded_feature";
    pub(crate) const INPUT_CHANGED: &str = "input_changed";
    pub(crate) const ERRORS_TRUNCATED: &str = "errors_truncated";
    /// Record line of an execution failure without a native answer (internal error of the attempt).
    pub(crate) const EXECUTION_FAILED: &str = "execution_failed";
}

/// A batch resource. Timestamps of states not reached are null; `expires_at` is always set.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct BatchObject {
    pub(crate) id: String,
    pub(crate) object: Tag,
    /// Always `/v1/basal`.
    pub(crate) endpoint: String,
    /// Effective model of the records once discovery resolved it.
    pub(crate) model: Option<String>,
    pub(crate) errors: Option<BatchErrors>,
    pub(crate) input_file_id: String,
    /// Always `24h`.
    pub(crate) completion_window: &'static str,
    /// `validating`, `failed`, `in_progress`, `finalizing`, `completed`, `expired`, `cancelling`, `cancelled`.
    pub(crate) status: &'static str,
    pub(crate) output_file_id: Option<String>,
    pub(crate) error_file_id: Option<String>,
    pub(crate) created_at: i64,
    pub(crate) in_progress_at: Option<i64>,
    pub(crate) expires_at: i64,
    pub(crate) finalizing_at: Option<i64>,
    pub(crate) completed_at: Option<i64>,
    pub(crate) failed_at: Option<i64>,
    pub(crate) expired_at: Option<i64>,
    pub(crate) cancelling_at: Option<i64>,
    pub(crate) cancelled_at: Option<i64>,
    pub(crate) request_counts: RequestCounts,
    pub(crate) metadata: Option<Metadata>,
}

pub(crate) const COMPLETION_WINDOW_NAME: &str = COMPLETION_WINDOW;

/// Batch `metadata`: at most 16 string pairs, keys up to 64 and values up to 512 characters, in the client's order;
/// duplicate keys are refused.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Metadata(pub(crate) Vec<(String, String)>);

impl Serialize for Metadata {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_map(self.0.iter().map(|(k, v)| (k, v)))
    }
}

impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Metadata;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object of at most 16 string values")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Metadata, A::Error> {
                let mut pairs: Vec<(String, String)> = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if pairs.len() == MAX_METADATA_PAIRS {
                        return Err(de::Error::custom("metadata has more than 16 pairs"));
                    }
                    if key.is_empty() || key.chars().count() > MAX_METADATA_KEY_CHARS {
                        return Err(de::Error::custom("metadata keys have 1-64 characters"));
                    }
                    if pairs.iter().any(|(k, _)| *k == key) {
                        return Err(de::Error::custom("metadata has a duplicate key"));
                    }
                    let value: String = map.next_value()?;
                    if value.chars().count() > MAX_METADATA_VALUE_CHARS {
                        return Err(de::Error::custom("metadata values have at most 512 characters"));
                    }
                    pairs.push((key, value));
                }
                Ok(Metadata(pairs))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawExpiresAfter {
    anchor: String,
    seconds: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCreate {
    input_file_id: String,
    endpoint: String,
    completion_window: String,
    #[serde(default)]
    metadata: Option<Metadata>,
    #[serde(default)]
    output_expires_after: Option<RawExpiresAfter>,
}

/// A checked `POST /v1/batches` body.
#[derive(Clone, Debug)]
pub(crate) struct CreateBatchRequest {
    pub(crate) input_file_id: FileId,
    pub(crate) metadata: Metadata,
    /// `output_expires_after.seconds`, anchored at the creation of each result file.
    pub(crate) output_expires_after_s: Option<u64>,
}

impl CreateBatchRequest {
    /// Parse and check a create body: required `input_file_id`, `endpoint=/v1/basal`, `completion_window=24h`;
    /// optional `metadata` and `output_expires_after`; any other field is a 400. A malformed `input_file_id` is the
    /// same 404 as a missing file.
    pub(crate) fn parse(body: &[u8]) -> Result<Self, ApiError> {
        if body.len() > MAX_CONTROL_BODY_BYTES {
            return Err(ApiError::invalid_request("The request body is too large."));
        }
        let raw: RawCreate = serde_json::from_slice(body)
            .map_err(|e| ApiError::invalid_request(format!("Invalid request body: {e}")))?;
        match raw.endpoint.as_str() {
            BASAL_ENDPOINT => {}
            SYSTEMONE_ENDPOINT => {
                return Err(ApiError::invalid_request(
                    "System One batches are not supported; use endpoint /v1/basal with native Basal records.",
                )
                .with_param("endpoint")
                .with_code("unsupported_endpoint"))
            }
            _ => {
                return Err(ApiError::invalid_request("The only supported endpoint is /v1/basal.")
                    .with_param("endpoint")
                    .with_code("unsupported_endpoint"))
            }
        }
        if raw.completion_window != COMPLETION_WINDOW {
            return Err(ApiError::invalid_request("The only supported completion_window is 24h.")
                .with_param("completion_window")
                .with_code("unsupported_completion_window"));
        }
        let output_expires_after_s = match raw.output_expires_after {
            Some(e) => {
                check_expires_anchor(&e.anchor, "output_expires_after.anchor")?;
                Some(check_expires_seconds(e.seconds, "output_expires_after.seconds")?)
            }
            None => None,
        };
        let input_file_id = FileId::parse(&raw.input_file_id).map_err(|_| ApiError::not_found(Resource::File))?;
        Ok(Self { input_file_id, metadata: raw.metadata.unwrap_or_default(), output_expires_after_s })
    }

    /// SHA-256 of the request data, stored with its Idempotency-Key: a replay with another digest is a 409.
    /// Metadata keys are sorted (a map), other fields are length-prefixed.
    pub(crate) fn digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        let mut field = |bytes: &[u8]| {
            h.update((bytes.len() as u64).to_be_bytes());
            h.update(bytes);
        };
        field(b"basal-batch-create/v1");
        field(self.input_file_id.as_str().as_bytes());
        field(BASAL_ENDPOINT.as_bytes());
        field(COMPLETION_WINDOW.as_bytes());
        let mut pairs: Vec<_> = self.metadata.0.iter().collect();
        pairs.sort();
        field(&(pairs.len() as u64).to_be_bytes());
        for (k, v) in pairs {
            field(k.as_bytes());
            field(v.as_bytes());
        }
        match self.output_expires_after_s {
            Some(s) => field(&s.to_be_bytes()),
            None => field(b"no-output-expiry"),
        }
        h.finalize().into()
    }
}

/// `Idempotency-Key` of a create request: one header of 1–128 visible ASCII bytes (owner-scoped by the store).
pub(crate) fn idempotency_key(headers: &HeaderMap) -> Result<Option<&str>, ApiError> {
    let mut values = headers.get_all("idempotency-key").iter();
    let Some(value) = values.next() else { return Ok(None) };
    let bad = || {
        ApiError::invalid_request("Idempotency-Key must be one header of 1-128 visible ASCII characters.")
            .with_param("Idempotency-Key")
    };
    if values.next().is_some() {
        return Err(bad());
    }
    let b = value.as_bytes();
    if b.is_empty() || b.len() > 128 || !b.iter().all(|c| (0x21..=0x7e).contains(c)) {
        return Err(bad());
    }
    std::str::from_utf8(b).map(Some).map_err(|_| bad())
}

/// Cancel takes no options: an empty body or `{}`.
pub(crate) fn check_cancel_body(body: &[u8]) -> Result<(), ApiError> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(());
    }
    match serde_json::from_slice::<serde_json::Map<String, Value>>(body) {
        Ok(map) if map.is_empty() => Ok(()),
        Ok(_) => Err(ApiError::invalid_request("Cancel takes no options.").with_code("unknown_parameter")),
        Err(e) => Err(ApiError::invalid_request(format!("Invalid request body: {e}"))),
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Lists

/// Query of `GET /v1/files` and `GET /v1/batches`: `after` (an id of the listed kind), `limit` 1–100 (default 20),
/// and for files `purpose`. Lists are ordered by `created_at, id` descending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ListQuery {
    /// Raw cursor; the caller parses it as an id of the listed kind (malformed: 400).
    pub(crate) after: Option<String>,
    pub(crate) limit: u32,
    pub(crate) purpose: Option<FilePurpose>,
}

impl ListQuery {
    /// Parse the raw query string; unknown, repeated or malformed parameters are a 400.
    pub(crate) fn parse(query: Option<&str>, files: bool) -> Result<Self, ApiError> {
        let mut q = Self { after: None, limit: LIST_DEFAULT_LIMIT, purpose: None };
        let (mut seen_after, mut seen_limit, mut seen_purpose) = (false, false, false);
        let repeated = |seen: &mut bool, param: &'static str| {
            if std::mem::replace(seen, true) {
                Err(ApiError::invalid_request("A query parameter is repeated.").with_param(param))
            } else {
                Ok(())
            }
        };
        let malformed = || ApiError::invalid_request("Malformed query string.");
        for pair in query.unwrap_or("").split('&').filter(|p| !p.is_empty()) {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            let key = percent_decode(key).ok_or_else(malformed)?;
            let value = percent_decode(value).ok_or_else(malformed)?;
            match key.as_str() {
                "after" => {
                    repeated(&mut seen_after, "after")?;
                    if value.is_empty() || value.len() > MAX_CURSOR_BYTES || !value.is_ascii() {
                        return Err(ApiError::bad_cursor());
                    }
                    q.after = Some(value);
                }
                "limit" => {
                    repeated(&mut seen_limit, "limit")?;
                    q.limit = value
                        .parse::<u32>()
                        .ok()
                        .filter(|l| (1..=LIST_MAX_LIMIT).contains(l))
                        .ok_or_else(|| ApiError::invalid_request("`limit` must be 1-100.").with_param("limit"))?;
                }
                "purpose" if files => {
                    repeated(&mut seen_purpose, "purpose")?;
                    q.purpose = Some(match value.as_str() {
                        "batch" => FilePurpose::Batch,
                        "batch_output" => FilePurpose::BatchOutput,
                        _ => {
                            return Err(ApiError::invalid_request("`purpose` must be batch or batch_output.")
                                .with_param("purpose"))
                        }
                    });
                }
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "Unknown query parameter `{}`.",
                        truncate_utf8(other, 64).escape_debug()
                    ))
                    .with_code("unknown_parameter"))
                }
            }
        }
        Ok(q)
    }
}

/// `%XX` and `+` decoding of one query component; `None` for malformed escapes or invalid UTF-8.
fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let hex = b.get(i + 1..i + 3)?;
                out.push(u8::from_str_radix(std::str::from_utf8(hex).ok()?, 16).ok()?);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// `{"object": "list", "data", "first_id", "last_id", "has_more"}`.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ListPage<T> {
    object: Tag,
    data: Vec<T>,
    first_id: Option<String>,
    last_id: Option<String>,
    has_more: bool,
}

impl<T> ListPage<T> {
    pub(crate) fn new(data: Vec<T>, has_more: bool, id: impl Fn(&T) -> &str) -> Self {
        let first_id = data.first().map(|t| id(t).to_string());
        let last_id = data.last().map(|t| id(t).to_string());
        Self { object: LIST, data, first_id, last_id, has_more }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Record result lines

/// Outcome of one record in a result file.
#[derive(Clone, Copy, Debug)]
pub(crate) enum RecordOutcome<'a> {
    /// The native endpoint answered (200, or its error status such as 422 with its error body).
    Response { status_code: u16, body: &'a Value },
    /// The record was not executed or its result was not kept (`result_too_large`, `host_budget_exceeded`, ...).
    Error { code: &'a str, message: &'a str },
}

/// Write one JSONL result line (with its LF):
/// `{"id":"batch_req_…","custom_id":…,"response":{"status_code":…,"request_id":"batch_req_…","body":…},"error":null}`
/// or `{"id":…,"custom_id":…,"response":null,"error":{"code":…,"message":…}}` (the shape of the store's expiry
/// line). The body is written by the same serializer as the online answer.
pub(crate) fn write_result_line<W: std::io::Write>(
    out: &mut W,
    request_id: &str,
    custom_id: &str,
    outcome: RecordOutcome<'_>,
) -> std::io::Result<()> {
    out.write_all(b"{\"id\":")?;
    serde_json::to_writer(&mut *out, request_id)?;
    out.write_all(b",\"custom_id\":")?;
    serde_json::to_writer(&mut *out, custom_id)?;
    match outcome {
        RecordOutcome::Response { status_code, body } => {
            write!(out, ",\"response\":{{\"status_code\":{status_code},\"request_id\":")?;
            serde_json::to_writer(&mut *out, request_id)?;
            out.write_all(b",\"body\":")?;
            serde_json::to_writer(&mut *out, body)?;
            out.write_all(b"},\"error\":null}\n")?;
        }
        RecordOutcome::Error { code, message } => {
            out.write_all(b",\"response\":null,\"error\":{\"code\":")?;
            serde_json::to_writer(&mut *out, code)?;
            out.write_all(b",\"message\":")?;
            serde_json::to_writer(&mut *out, message)?;
            out.write_all(b"}}\n")?;
        }
    }
    Ok(())
}

/// Counts written bytes.
struct Counter(u64);

impl std::io::Write for Counter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len() as u64);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn line_len(custom_id: &str, outcome: RecordOutcome<'_>) -> u64 {
    let mut c = Counter(0);
    let id = new_request_id();
    // Writing to a counter cannot fail; a failure would only shorten the bound, so it is treated as the maximum.
    match write_result_line(&mut c, &id, custom_id, outcome) {
        Ok(()) => c.0,
        Err(_) => RESULT_MAX,
    }
}

/// Upper bound of a native error body (`{"error": message}` of an execution error) of one record.
const ERROR_BODY_BOUND: u64 = 16 * 1024;

/// Upper bound of the finished result line of a record whose native body is at most `body_bound` bytes: the larger
/// of the success line, a native error line (status and body up to 16 KiB) and a non-execution error line (code and
/// 256-byte message), at most 64 MiB.
pub(crate) fn result_line_bound(custom_id: &str, body_bound: u64) -> u64 {
    // A `null` body (4 bytes) stands for the body; status codes have three digits.
    let null = Value::Null;
    let envelope = line_len(custom_id, RecordOutcome::Response { status_code: 200, body: &null }).saturating_sub(4);
    let message = "m".repeat(MAX_ERROR_TEXT_BYTES);
    let error_line = line_len(custom_id, RecordOutcome::Error { code: codes::EXECUTION_FAILED, message: &message })
        .saturating_add(64);
    envelope.saturating_add(body_bound.max(ERROR_BODY_BOUND)).max(error_line).min(RESULT_MAX)
}
