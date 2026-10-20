//! Resource facades and opt-in Batch workflow helpers. Existing flat methods stay available.

use std::{
    collections::BTreeMap,
    fmt,
    hash::{BuildHasher, Hasher},
    io::Write,
    time::Duration,
};

use http::{
    header::{HeaderMap, HeaderValue, ACCEPT},
    Method,
};

use crate::{
    Batch, BatchExpiresAfter, BatchInputLine, BatchListOptions, BatchPage, BatchResultLine, Client, CreateBatchRequest,
    DeletedFile, Error, FileObject, FileUploadOptions,
};

/// A simplified create request. The SDK serializes and uploads these records before creating the batch.
/// JSONL is buffered in memory, limited to 50,000 records and 200 MB; the upload adds a multipart copy.
pub struct CreateBatchFromRequests<'a> {
    /// Native requests and unique caller identifiers.
    pub requests: &'a [BatchInputLine],
    /// Optional owner metadata.
    pub metadata: Option<BTreeMap<String, String>>,
    /// Optional output and error file retention.
    pub output_expires_after: Option<BatchExpiresAfter>,
}

impl<'a> CreateBatchFromRequests<'a> {
    /// Prepare native `/v1/basal` records with the server's 24-hour expiry window.
    pub fn new(requests: &'a [BatchInputLine]) -> Self {
        Self { requests, metadata: None, output_expires_after: None }
    }
}

/// Mutually exclusive input paths for [`Batches::create`]. Prefer the `From` conversions at the call site.
pub enum BatchCreateInput<'a> {
    /// Reference a previously uploaded file.
    File(&'a CreateBatchRequest),
    /// Serialize/upload native requests, then create a batch.
    Requests(&'a CreateBatchFromRequests<'a>),
}

impl<'a> From<&'a CreateBatchRequest> for BatchCreateInput<'a> {
    fn from(value: &'a CreateBatchRequest) -> Self {
        Self::File(value)
    }
}

impl<'a> From<&'a CreateBatchFromRequests<'a>> for BatchCreateInput<'a> {
    fn from(value: &'a CreateBatchFromRequests<'a>) -> Self {
        Self::Requests(value)
    }
}

/// Explicit polling settings. The deadline covers both retrieval requests and sleeps.
#[derive(Clone, Copy, Debug)]
pub struct BatchWaitOptions {
    /// Whole client-side wait budget (default ten minutes); timeout never cancels the server job.
    pub timeout: Duration,
    /// Delay between successful nonterminal observations (default five seconds).
    pub poll_interval: Duration,
}

impl Default for BatchWaitOptions {
    fn default() -> Self {
        Self { timeout: Duration::from_secs(600), poll_interval: Duration::from_secs(5) }
    }
}

/// Per-file buffering allowance for the results cursor.
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchResultsOptions {
    /// Maximum bytes in each downloaded file, defaulting to the client's bound (8 MiB by default).
    pub max_bytes: Option<usize>,
}

/// Typed failures of the opt-in workflow helpers. Debug/Display omit documents, bodies and keys.
pub enum BatchOperationError {
    /// Ordinary request/transport failure, with its original details explicitly available.
    Request(Error),
    /// Upload succeeded but creation did not return a usable batch. Creation may have succeeded remotely.
    /// Resume using this file id and the same key; never automatically delete the uploaded input.
    Submission {
        /// Uploaded file, usable with `CreateBatchRequest::new`.
        input_file_id: String,
        /// Reuse this key together with the same metadata and output retention.
        idempotency_key: String,
        /// Original request failure.
        source: Error,
    },
    /// The entire wait budget elapsed; no server cancellation was requested.
    WaitTimeout {
        /// Durable batch identifier.
        batch_id: String,
        /// Last successfully observed state, when any retrieval finished.
        last_batch: Option<Box<Batch>>,
    },
    /// Results require a terminal batch. Inspect this state or explicitly call `wait`.
    NotReady {
        /// Current batch, including counters and validation errors.
        batch: Box<Batch>,
    },
    /// One downloaded JSONL record did not match the result contract.
    ResultLine {
        /// File containing the malformed record.
        file_id: String,
        /// One-based physical line number.
        line: usize,
        /// Parser or envelope validation failure.
        source: Error,
    },
}

impl From<Error> for BatchOperationError {
    fn from(value: Error) -> Self {
        Self::Request(value)
    }
}

impl fmt::Display for BatchOperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Request(error) => fmt::Display::fmt(error, f),
            Self::Submission { .. } => {
                f.write_str("Basal batch creation failed after upload; recovery context is available")
            }
            Self::WaitTimeout { .. } => f.write_str("Basal batch wait timed out; server work continues"),
            Self::NotReady { .. } => f.write_str("Basal batch results are not ready"),
            Self::ResultLine { line, .. } => write!(f, "invalid Basal batch result at line {line}"),
        }
    }
}

impl fmt::Debug for BatchOperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for BatchOperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(source) | Self::Submission { source, .. } | Self::ResultLine { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Files resource. Borrows the pooled client; never opens local paths or remote URLs.
#[derive(Clone, Copy)]
pub struct Files<'a> {
    client: &'a Client,
}

/// Batches resource. Workflow polling happens only on an explicit `wait` call.
#[derive(Clone, Copy)]
pub struct Batches<'a> {
    client: &'a Client,
}

impl Client {
    /// Files resource with `create`, `retrieve`, `list`, `delete` and `content` operations.
    pub fn files(&self) -> Files<'_> {
        Files { client: self }
    }

    /// Batches resource with both create paths, retrieval and opt-in workflow helpers.
    pub fn batches(&self) -> Batches<'_> {
        Batches { client: self }
    }
}

impl Files<'_> {
    /// Upload JSONL bytes with purpose `batch`. Identical to the existing flat upload method.
    pub async fn create(&self, filename: &str, data: &[u8], options: FileUploadOptions) -> Result<FileObject, Error> {
        self.client.upload_file(filename, data, options).await
    }
    /// Read file metadata.
    pub async fn retrieve(&self, id: &str) -> Result<FileObject, Error> {
        self.client.get_file(id).await
    }
    /// Read one page; never automatically paginates.
    pub async fn list(
        &self,
        options: BatchListOptions<'_>,
        purpose: Option<&str>,
    ) -> Result<BatchPage<FileObject>, Error> {
        self.client.list_files(options, purpose).await
    }
    /// Request deletion.
    pub async fn delete(&self, id: &str) -> Result<DeletedFile, Error> {
        self.client.delete_file(id).await
    }
    /// Download bytes within the client's existing timeout and byte cap.
    pub async fn content(&self, id: &str) -> Result<Vec<u8>, Error> {
        self.client.download_file(id).await
    }
    /// Override the successful download byte cap for this call only. HTTP errors keep the client-wide cap.
    pub async fn content_with_limit(&self, id: &str, max_bytes: usize) -> Result<Vec<u8>, Error> {
        super::batches::id_component(id, "file_")?;
        if max_bytes == 0 {
            return Err(Error::InvalidRequest("download max_bytes must be positive"));
        }
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/octet-stream"));
        self.client
            .request_bytes_bounded(Method::GET, &format!("v1/files/{id}/content"), (None, None), headers, max_bytes)
            .await
    }
}

impl<'a> Batches<'a> {
    /// Create from an uploaded file or from native requests (via `CreateBatchFromRequests`).
    /// The requests path uploads buffered JSONL and creates the batch, without polling or retries.
    /// Post-upload failure exposes a recoverable file id/key in `BatchOperationError::Submission`.
    pub async fn create<'r>(
        &self,
        input: impl Into<BatchCreateInput<'r>>,
        idempotency_key: Option<&str>,
    ) -> Result<Batch, BatchOperationError> {
        match input.into() {
            BatchCreateInput::File(request) => {
                self.client.create_batch(request, idempotency_key).await.map_err(Into::into)
            }
            BatchCreateInput::Requests(request) => {
                validate_key(idempotency_key)?;
                validate_common(request)?;
                let data = encode_requests(request.requests)?;
                let key = idempotency_key.map(str::to_owned).unwrap_or_else(new_key);
                let file = self.client.upload_file("batch-input.jsonl", &data, FileUploadOptions::default()).await?;
                let mut create = CreateBatchRequest::new(&file.id);
                create.metadata = request.metadata.clone();
                create.output_expires_after = request.output_expires_after.clone();
                self.client.create_batch(&create, Some(&key)).await.map_err(|source| BatchOperationError::Submission {
                    input_file_id: file.id,
                    idempotency_key: key,
                    source,
                })
            }
        }
    }
    /// Retrieve the latest state without polling.
    pub async fn retrieve(&self, id: &str) -> Result<Batch, Error> {
        self.client.get_batch(id).await
    }
    /// Retrieve one page without automatic pagination.
    pub async fn list(&self, options: BatchListOptions<'_>) -> Result<BatchPage<Batch>, Error> {
        self.client.list_batches(options).await
    }
    /// Explicitly request server cancellation; partial results may remain.
    pub async fn cancel(&self, id: &str) -> Result<Batch, Error> {
        self.client.cancel_batch(id).await
    }

    /// Wait until completed/failed/expired/cancelled. HTTP errors propagate without retry.
    /// Timeout or dropping this future stops client-side polling, never the server batch.
    pub async fn wait(&self, id: &str, options: BatchWaitOptions) -> Result<Batch, BatchOperationError> {
        super::batches::id_component(id, "batch_")?;
        if options.timeout.is_zero()
            || options.poll_interval.is_zero()
            || tokio::time::Instant::now().checked_add(options.timeout).is_none()
            || tokio::time::Instant::now().checked_add(options.poll_interval).is_none()
        {
            return Err(Error::InvalidRequest("wait durations must be positive and representable").into());
        }
        let mut last_batch = None;
        let polling = async {
            loop {
                let batch = self.retrieve(id).await?;
                if terminal(&batch.status) {
                    return Ok(batch);
                }
                last_batch = Some(Box::new(batch));
                tokio::time::sleep(options.poll_interval).await;
            }
        };
        match tokio::time::timeout(options.timeout, polling).await {
            Ok(result) => result,
            Err(_) => Err(BatchOperationError::WaitTimeout { batch_id: id.to_owned(), last_batch }),
        }
    }

    /// Read terminal output and error files through a bounded, per-file buffered async cursor.
    /// This never waits for completion. Inspect `cursor.batch()` for batch-level validation errors.
    pub async fn results(&self, id: &str) -> Result<BatchResults<'a>, BatchOperationError> {
        self.results_with_options(id, BatchResultsOptions::default()).await
    }

    /// Results cursor with an explicit per-file byte allowance.
    pub async fn results_with_options(
        &self,
        id: &str,
        options: BatchResultsOptions,
    ) -> Result<BatchResults<'a>, BatchOperationError> {
        if options.max_bytes == Some(0) {
            return Err(Error::InvalidRequest("download max_bytes must be positive").into());
        }
        let batch = self.retrieve(id).await?;
        if !terminal(&batch.status) {
            return Err(BatchOperationError::NotReady { batch: Box::new(batch) });
        }
        let mut files = Vec::new();
        for file in [&batch.output_file_id, &batch.error_file_id].into_iter().flatten() {
            if !files.contains(file) {
                files.push(file.clone());
            }
        }
        Ok(BatchResults {
            client: self.client,
            batch,
            files,
            next_file: 0,
            current: Vec::new(),
            offset: 0,
            line: 0,
            max_bytes: options.max_bytes,
            finished: false,
        })
    }
}

/// Result cursor. Buffers one file at a time; parsing does not create a second collection of all records.
/// No background task or connection is retained between `next` calls. Dropping an in-flight `next`
/// cancels that request; discard the cursor after cancellation rather than resuming uncertain I/O.
pub struct BatchResults<'a> {
    client: &'a Client,
    batch: Batch,
    files: Vec<String>,
    next_file: usize,
    current: Vec<u8>,
    offset: usize,
    line: usize,
    max_bytes: Option<usize>,
    finished: bool,
}

impl BatchResults<'_> {
    /// Final batch, including counters and any batch-level validation errors.
    pub fn batch(&self) -> &Batch {
        &self.batch
    }

    /// Read the next record, output file first then error file. Ends permanently on a read/parse error.
    /// Record order is not input order; correlate by `custom_id`.
    pub async fn next(&mut self) -> Result<Option<BatchResultLine>, BatchOperationError> {
        if self.finished {
            return Ok(None);
        }
        loop {
            if self.offset < self.current.len() {
                let rest = &self.current[self.offset..];
                let length = rest.iter().position(|b| *b == b'\n').unwrap_or(rest.len());
                let record = parse_result(&rest[..length]);
                self.offset += length + usize::from(length < rest.len());
                self.line += 1;
                return record.map(Some).map_err(|source| {
                    self.finished = true;
                    self.current = Vec::new();
                    BatchOperationError::ResultLine {
                        file_id: self.files[self.next_file - 1].clone(),
                        line: self.line,
                        source,
                    }
                });
            }
            // Release the preceding file's allocation before buffering the next download.
            self.current = Vec::new();
            if self.next_file == self.files.len() {
                self.finished = true;
                return Ok(None);
            }
            let id = &self.files[self.next_file];
            // Mark the cursor exhausted before awaiting I/O. Cancellation cannot retry a consumed response.
            self.finished = true;
            let bytes = match self.max_bytes {
                Some(limit) => self.client.files().content_with_limit(id, limit).await?,
                None => self.client.files().content(id).await?,
            };
            self.finished = false;
            self.current = bytes;
            self.offset = 0;
            self.line = 0;
            self.next_file += 1;
        }
    }
}

fn terminal(status: &str) -> bool {
    matches!(status, "completed" | "failed" | "expired" | "cancelled")
}

fn validate_key(key: Option<&str>) -> Result<(), Error> {
    if key.is_some_and(|k| k.is_empty() || k.len() > 128 || !k.bytes().all(|b| (0x21..=0x7e).contains(&b))) {
        return Err(Error::InvalidRequest("invalid idempotency key"));
    }
    Ok(())
}

fn validate_common(request: &CreateBatchFromRequests<'_>) -> Result<(), Error> {
    if request.metadata.as_ref().is_some_and(|m| {
        m.len() > 16 || m.iter().any(|(k, v)| k.is_empty() || k.chars().count() > 64 || v.chars().count() > 512)
    }) {
        return Err(Error::InvalidRequest("invalid batch metadata"));
    }
    if request
        .output_expires_after
        .as_ref()
        .is_some_and(|e| e.anchor != "created_at" || !(3600..=2_592_000).contains(&e.seconds))
    {
        return Err(Error::InvalidRequest("invalid output retention"));
    }
    Ok(())
}

fn new_key() -> String {
    let random = std::collections::hash_map::RandomState::new();
    let mut a = random.build_hasher();
    a.write_u8(0);
    let mut b = random.build_hasher();
    b.write_u8(1);
    format!("basal-sdk-{:016x}{:016x}", a.finish(), b.finish())
}

struct JsonlBuffer {
    bytes: Vec<u8>,
    limit_hit: bool,
}

impl Write for JsonlBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > 200_000_000usize.saturating_sub(self.bytes.len()) {
            self.limit_hit = true;
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "batch input exceeds 200 MB"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode_requests(requests: &[BatchInputLine]) -> Result<Vec<u8>, Error> {
    if requests.is_empty() || requests.len() > 50_000 {
        return Err(Error::InvalidRequest("batch requires 1-50000 requests"));
    }
    let mut ids = std::collections::HashSet::new();
    let mut out = JsonlBuffer { bytes: Vec::new(), limit_hit: false };
    for record in requests {
        if record.custom_id.is_empty() || record.custom_id.len() > 256 || !ids.insert(record.custom_id.as_str()) {
            return Err(Error::InvalidRequest("batch custom_id must be unique and contain 1-256 UTF-8 bytes"));
        }
        if record.method != "POST"
            || record.url != "/v1/basal"
            || record.body.extra.get("stream").and_then(serde_json::Value::as_bool) == Some(true)
        {
            return Err(Error::InvalidRequest("batch records require nonstreaming POST /v1/basal"));
        }
        let encoded = serde_json::to_writer(&mut out, record);
        if out.limit_hit {
            return Err(Error::InvalidRequest("batch input exceeds 200 MB"));
        }
        encoded.map_err(Error::Encode)?;
        out.write_all(b"\n").map_err(|_| Error::InvalidRequest("batch input exceeds 200 MB"))?;
    }
    Ok(out.bytes)
}

fn parse_result(line: &[u8]) -> Result<BatchResultLine, Error> {
    let value: serde_json::Value = serde_json::from_slice(line).map_err(Error::Protocol)?;
    if value.get("response").is_none() || value.get("error").is_none() {
        return Err(Error::InvalidResponse("batch result requires response and error fields"));
    }
    let record: BatchResultLine = serde_json::from_value(value).map_err(Error::Protocol)?;
    if record.id.is_empty()
        || record.custom_id.is_empty()
        || record.response.is_some() == record.error.is_some()
        || record
            .response
            .as_ref()
            .is_some_and(|r| !(100..=599).contains(&r.status_code) || r.request_id.is_empty() || !r.body.is_object())
        || record.error.as_ref().is_some_and(|e| e.code.is_empty())
    {
        return Err(Error::InvalidResponse("invalid batch result envelope"));
    }
    Ok(record)
}
