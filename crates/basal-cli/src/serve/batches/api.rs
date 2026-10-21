//! HTTP routes of the Batch and Files API.
//!
//! Routes do not authenticate callers. All requests use the configured storage namespace through a server-set
//! `store::OwnerId` extension; every store call stays scoped to it. Mutations run as supervised tasks
//! (they finish even when the client disconnects).
//!
//! Upload (`POST /v1/files`): the session (file slot, session slot, the largest upload plus its object charge) is
//! reserved before the first byte. The body streams through a bounded buffer into a fresh artifact key and is
//! refused at the limit (413, also without `Content-Length`). `finish` makes the object durable before
//! `complete_upload` publishes the file. A handler that ends early abandons the session, so GC removes its partial
//! object after the I/O guard. `expires_after[...]` fields must precede the file part; `purpose` may come before
//! or after it, and an invalid purpose abandons the upload.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::multipart::Field;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};

use super::artifacts::{ArtifactKey, ArtifactKind, ArtifactWriter};
use super::convert::{self, store_error};
use super::dto::{
    self, check_cancel_body, check_expires_anchor, check_filename, check_upload_purpose, idempotency_key,
    parse_expires_seconds, ApiError, BatchObject, CreateBatchRequest, DeletedFile, FileObject, ListPage, ListQuery,
    Resource, Throttle, UploadField, MAX_CONTROL_BODY_BYTES, MAX_FORM_TEXT_BYTES,
};
use super::service::{health, Shared};
use super::store::{
    BatchId, BatchMetadata, BatchRecord, CreateOutcome, FileId, FileRecord, Idempotency, NewBatch, NewUpload,
    OperationId, OwnerId, PageRequest, SessionId, Sha256Digest, Store, StoreError,
};
use super::upload::UploadBody;

/// Multipart framing (boundaries, part headers, text fields) allowed on top of the file limit before a declared
/// `Content-Length` is refused early. The actual streamed body has the same total limit.
const MULTIPART_OVERHEAD: u64 = 64 * 1024;

type ApiResult<T> = Result<T, ApiError>;

pub(super) fn router(shared: Arc<Shared>) -> Router<()> {
    let control = DefaultBodyLimit::max(MAX_CONTROL_BODY_BYTES);
    Router::new()
        .route("/v1/files", get(list_files).post(upload).layer(DefaultBodyLimit::disable()).fallback(unsupported))
        .route("/v1/files/{file_id}", get(get_file).delete(delete_file).fallback(unsupported))
        .route("/v1/files/{file_id}/content", get(download).fallback(unsupported))
        .route("/v1/batches", get(list_batches).post(create_batch).layer(control).fallback(unsupported))
        .route("/v1/batches/{batch_id}", get(get_batch).fallback(unsupported))
        .route("/v1/batches/{batch_id}/cancel", post(cancel_batch).layer(control).fallback(unsupported))
        .route_layer(Extension(shared.settings.owner.clone()))
        .route("/health/batches", get(health_handler))
        .with_state(shared)
}

async fn unsupported() -> ApiError {
    ApiError::unsupported_route()
}

async fn health_handler(State(s): State<Arc<Shared>>) -> Response {
    let mut body = health(&s);
    if let Some(store) = s.store() {
        match store.backlog(32).await {
            Ok(backlog) => {
                body["backlog"] = backlog
                    .iter()
                    .map(|b| {
                        serde_json::json!({
                            "fingerprint": b.fingerprint.as_ref().map(|f| f.as_str()),
                            "status": b.status.as_db(),
                            "batches": b.batches,
                            "queued": b.queued,
                        })
                    })
                    .collect();
            }
            Err(e) => body["backlog_error"] = serde_json::Value::String(e.to_string()),
        }
    }
    let status = if body["status"] == "ok" { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (status, Json(body)).into_response()
}

// ---------------------------------------------------------------------------------------------------------------
// Files

fn file_id(raw: &str) -> ApiResult<FileId> {
    FileId::parse(raw).map_err(|_| ApiError::not_found(Resource::File))
}

async fn list_files(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    uri: Uri,
) -> ApiResult<Json<ListPage<FileObject>>> {
    let q = ListQuery::parse(uri.query(), true)?;
    let after = q.after.as_deref().map(FileId::parse).transpose().map_err(|_| ApiError::bad_cursor())?;
    let store = s.store_or_503()?;
    let page = store
        .list_files(&owner, q.purpose.map(convert::file_purpose), &PageRequest { after, limit: q.limit })
        .await
        .map_err(|e| list_error(&e, Resource::File))?;
    let data = page.items.iter().map(convert::file_object).collect();
    Ok(Json(ListPage::new(data, page.has_more, |f: &FileObject| &f.id)))
}

/// An unknown cursor (missing or another owner's) is the same 400.
fn list_error(e: &StoreError, resource: Resource) -> ApiError {
    match e {
        StoreError::NotFound => ApiError::bad_cursor(),
        e => store_error(e, resource),
    }
}

async fn get_file(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    Path(id): Path<String>,
) -> ApiResult<Json<FileObject>> {
    let id = file_id(&id)?;
    let file = s.store_or_503()?.get_file(&owner, &id).await.map_err(|e| store_error(&e, Resource::File))?;
    Ok(Json(convert::file_object(&file)))
}

async fn delete_file(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    Path(id): Path<String>,
) -> ApiResult<Json<DeletedFile>> {
    let id = file_id(&id)?;
    let store = s.store_or_503()?;
    let file = s
        .mutate(async move { store.delete_file(&owner, &id).await })
        .await?
        .map_err(|e| store_error(&e, Resource::File))?;
    Ok(Json(DeletedFile { id: file.id.as_str().to_string(), object: dto::FILE, deleted: true }))
}

async fn download(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let id = file_id(&id)?;
    let file = s.store_or_503()?.get_file(&owner, &id).await.map_err(|e| store_error(&e, Resource::File))?;
    let permit = Arc::clone(&s.downloads)
        .try_acquire_owned()
        .map_err(|_| ApiError::throttled(Throttle::Concurrency, convert::RETRY_AFTER_S))?;
    let recorded = super::store::StoredObject { bytes: file.bytes, sha256: file.sha256 };
    let object = convert::object(&file.artifact, &recorded).map_err(|e| convert::artifact_error(&e))?;
    let deadline = Instant::now() + Duration::from_secs(s.settings.retention.download_max_duration_s);
    let reader = s.artifacts.read(&object, Some(deadline)).await.map_err(|e| match e {
        // The database lists the file but its object is gone: not the client's fault, and not retriable by it.
        super::artifacts::ArtifactError::NotFound => ApiError::internal(),
        e => convert::artifact_error(&e),
    })?;
    let chunk = usize::try_from(s.settings.host.upload_buffer_bytes).unwrap_or(1 << 20);
    let mut response = reader.with_permit(permit).cancel_on(s.stop.clone()).into_body(chunk).into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/jsonl"));
    headers.insert(header::CONTENT_LENGTH, HeaderValue::from(file.bytes));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("private, no-store"));
    Ok(response)
}

/// An upload between its session reservation and its publication. Dropped unfinished, it abandons the session
/// in a background task (the writer discards its partial object on its own; GC removes what is left).
struct PendingUpload {
    shared: Arc<Shared>,
    store: Store,
    owner: OwnerId,
    session: SessionId,
    writer: Option<ArtifactWriter>,
    published: bool,
}

impl PendingUpload {
    async fn start(
        shared: &Arc<Shared>,
        store: &Store,
        owner: &OwnerId,
        filename: String,
        expires_after_s: Option<u64>,
        permit: tokio::sync::OwnedSemaphorePermit,
        deadline: Instant,
    ) -> ApiResult<Self> {
        let key = ArtifactKey::generate(ArtifactKind::Upload);
        let artifact = convert::artifact_ref(&key).map_err(|e| store_error(&e, Resource::File))?;
        let request = NewUpload {
            owner: owner.clone(),
            artifact,
            filename,
            expires_after: expires_after_s.map(Duration::from_secs),
        };
        let reserve_store = store.clone();
        let session = shared
            .mutate(async move { reserve_store.reserve_upload(&request).await })
            .await?
            .map_err(|e| store_error(&e, Resource::File))?;
        let mut pending = Self {
            shared: Arc::clone(shared),
            store: store.clone(),
            owner: owner.clone(),
            session: session.id.clone(),
            writer: None,
            published: false,
        };
        let limit = session.max_bytes.min(shared.settings.limits.max_file_bytes);
        let writer = shared
            .artifacts
            .create(key, limit, deadline, Some(permit))
            .await
            .map_err(|e| convert::artifact_error(&e))?;
        pending.writer = Some(writer);
        Ok(pending)
    }

    async fn write(&mut self, chunk: &[u8]) -> ApiResult<()> {
        let writer = self.writer.as_mut().ok_or_else(ApiError::internal)?;
        writer.write(chunk).await.map_err(|e| convert::artifact_error(&e))
    }

    /// Make the object durable, then publish the file in the database.
    async fn publish(mut self) -> ApiResult<FileRecord> {
        let writer = self.writer.take().ok_or_else(ApiError::internal)?;
        let object = writer.finish().await.map_err(|e| convert::artifact_error(&e))?;
        let (store, owner, session) = (self.store.clone(), self.owner.clone(), self.session.clone());
        let stored = convert::stored(&object);
        let file = self
            .shared
            .mutate(async move { store.complete_upload(&owner, &session, stored).await })
            .await?
            .map_err(|e| store_error(&e, Resource::File))?;
        self.published = true;
        Ok(file)
    }
}

impl Drop for PendingUpload {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        drop(self.writer.take());
        let (store, owner, session) = (self.store.clone(), self.owner.clone(), self.session.clone());
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                // Idempotent; when it fails, the session lapses at its lease end and GC releases it.
                let _ = store.abort_upload(&owner, &session).await;
            });
        }
    }
}

async fn read_text(mut field: Field<'_>, input: &UploadBody) -> ApiResult<String> {
    let name = field.name().unwrap_or("field").to_string();
    let mut bytes = Vec::new();
    while let Some(chunk) = input.read(field.chunk()).await? {
        if bytes.len() + chunk.len() > MAX_FORM_TEXT_BYTES {
            return Err(ApiError::invalid_request(format!("The `{name}` field is too long.")));
        }
        bytes.extend_from_slice(&chunk);
        input.delivered(chunk.len());
    }
    String::from_utf8(bytes).map_err(|_| ApiError::invalid_request(format!("The `{name}` field is not UTF-8.")))
}

async fn upload(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    request: Request,
) -> ApiResult<Json<FileObject>> {
    let deadline = Instant::now() + Duration::from_secs(s.settings.host.upload_max_duration_s);
    let limit = s.settings.limits.max_file_bytes;
    let declared = request.headers().get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()?.parse::<u64>().ok());
    if declared.is_some_and(|len| len > limit.saturating_add(MULTIPART_OVERHEAD)) {
        return Err(ApiError::payload_too_large(limit));
    }
    let store = s.admitting_store()?;
    let mut permit = Some(
        Arc::clone(&s.uploads)
            .try_acquire_owned()
            .map_err(|_| ApiError::throttled(Throttle::Concurrency, convert::RETRY_AFTER_S))?,
    );
    let (parts, body) = request.into_parts();
    let input = UploadBody::new(
        body,
        limit.saturating_add(MULTIPART_OVERHEAD),
        s.settings.host.upload_buffer_bytes,
        MULTIPART_OVERHEAD,
        deadline,
        s.stop.clone(),
    );
    let request = Request::from_parts(parts, input.body());
    let mut multipart = Multipart::from_request(request, &())
        .await
        .map_err(|_| ApiError::invalid_request("Invalid multipart upload."))?;
    let mut purpose: Option<String> = None;
    let (mut anchor, mut seconds): (Option<String>, Option<u64>) = (None, None);
    let mut pending: Option<PendingUpload> = None;
    while let Some(mut field) = input.read(multipart.next_field()).await? {
        let kind = UploadField::classify(field.name())?;
        let after_file = pending.is_some();
        let duplicate = || ApiError::invalid_request("An upload field is repeated.").with_code("duplicate_parameter");
        match kind {
            UploadField::Purpose => {
                if purpose.is_some() {
                    return Err(duplicate());
                }
                let value = read_text(field, &input).await?;
                check_upload_purpose(&value)?;
                purpose = Some(value);
            }
            UploadField::ExpiresAfterAnchor | UploadField::ExpiresAfterSeconds if after_file => {
                return Err(ApiError::invalid_request("`expires_after[...]` fields must precede the `file` part.")
                    .with_param("expires_after"));
            }
            UploadField::ExpiresAfterAnchor => {
                if anchor.is_some() {
                    return Err(duplicate());
                }
                let value = read_text(field, &input).await?;
                check_expires_anchor(&value, "expires_after[anchor]")?;
                anchor = Some(value);
            }
            UploadField::ExpiresAfterSeconds => {
                if seconds.is_some() {
                    return Err(duplicate());
                }
                seconds = Some(parse_expires_seconds(&read_text(field, &input).await?, "expires_after[seconds]")?);
            }
            UploadField::File => {
                if after_file {
                    return Err(duplicate());
                }
                if anchor.is_some() != seconds.is_some() {
                    return Err(ApiError::invalid_request(
                        "`expires_after[anchor]` and `expires_after[seconds]` go together.",
                    )
                    .with_param("expires_after"));
                }
                let filename = check_filename(field.file_name())?;
                let slot = permit.take().ok_or_else(ApiError::internal)?;
                let mut upload = PendingUpload::start(&s, &store, &owner, filename, seconds, slot, deadline).await?;
                loop {
                    let chunk = input.read(field.chunk()).await?;
                    let Some(chunk) = chunk else { break };
                    input.wait(upload.write(&chunk)).await?;
                    input.delivered(chunk.len());
                }
                pending = Some(upload);
            }
        }
    }
    drop(multipart);
    input.finish().await?;
    let upload = pending.ok_or_else(|| ApiError::invalid_request("The `file` part is missing.").with_param("file"))?;
    if purpose.is_none() {
        return Err(ApiError::invalid_request("The `purpose` field is missing.").with_param("purpose"));
    }
    let file = upload.publish().await?;
    Ok(Json(convert::file_object(&file)))
}

// ---------------------------------------------------------------------------------------------------------------
// Batches

fn batch_id(raw: &str) -> ApiResult<BatchId> {
    BatchId::parse(raw).map_err(|_| ApiError::not_found(Resource::Batch))
}

/// The public resource, with `errors.data` read only for batches that recorded issues.
async fn render(store: &Store, batch: &BatchRecord) -> ApiResult<BatchObject> {
    let issues = if batch.error_count > 0 {
        store.batch_errors(&batch.owner, &batch.id).await.map_err(|e| store_error(&e, Resource::Batch))?
    } else {
        Vec::new()
    };
    Ok(convert::batch_object(batch, &issues))
}

async fn create_batch(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult<Json<BatchObject>> {
    let request = CreateBatchRequest::parse(&body)?;
    let idempotency = match idempotency_key(&headers)? {
        Some(key) => Some(Idempotency {
            key: OperationId::parse(key).map_err(|e| ApiError::invalid_request(e.to_string()))?,
            request_sha256: Sha256Digest(request.digest()),
        }),
        None => None,
    };
    let spec = NewBatch {
        input_file_id: request.input_file_id.clone(),
        endpoint: dto::BASAL_ENDPOINT.to_string(),
        metadata: BatchMetadata::new(request.metadata.0.clone())
            .map_err(|e| ApiError::invalid_request(e.to_string()).with_param("metadata"))?,
        output_expires_after: request.output_expires_after_s.map(Duration::from_secs),
    };
    let store = s.admitting_store()?;
    let create_store = store.clone();
    let outcome = s
        .mutate(async move { create_store.create_batch(&owner, idempotency.as_ref(), &spec).await })
        .await?
        .map_err(|e| store_error(&e, Resource::File))?;
    let batch = match outcome {
        CreateOutcome::Created(b) | CreateOutcome::Replayed(b) => b,
    };
    Ok(Json(render(&store, &batch).await?))
}

async fn get_batch(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    Path(id): Path<String>,
) -> ApiResult<Json<BatchObject>> {
    let id = batch_id(&id)?;
    let store = s.store_or_503()?;
    let batch = store.get_batch(&owner, &id).await.map_err(|e| store_error(&e, Resource::Batch))?;
    Ok(Json(render(&store, &batch).await?))
}

async fn list_batches(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    uri: Uri,
) -> ApiResult<Json<ListPage<BatchObject>>> {
    let q = ListQuery::parse(uri.query(), false)?;
    let after = q.after.as_deref().map(BatchId::parse).transpose().map_err(|_| ApiError::bad_cursor())?;
    let store = s.store_or_503()?;
    let page = store
        .list_batches(&owner, &PageRequest { after, limit: q.limit })
        .await
        .map_err(|e| list_error(&e, Resource::Batch))?;
    let mut data = Vec::with_capacity(page.items.len());
    for batch in &page.items {
        data.push(render(&store, batch).await?);
    }
    Ok(Json(ListPage::new(data, page.has_more, |b: &BatchObject| &b.id)))
}

async fn cancel_batch(
    State(s): State<Arc<Shared>>,
    Extension(owner): Extension<OwnerId>,
    Path(id): Path<String>,
    body: Bytes,
) -> ApiResult<Json<BatchObject>> {
    check_cancel_body(&body)?;
    let id = batch_id(&id)?;
    let store = s.store_or_503()?;
    let cancel_store = store.clone();
    let batch = s
        .mutate(async move { cancel_store.request_cancel(&owner, &id).await })
        .await?
        .map_err(|e| store_error(&e, Resource::Batch))?;
    Ok(Json(render(&store, &batch).await?))
}
