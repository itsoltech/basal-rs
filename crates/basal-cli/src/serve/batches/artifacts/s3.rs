//! Artifacts in a private S3-compatible bucket (`object_store` 0.14, AWS client only).
//!
//! Objects live at `<prefix>/v1/<kind>/<hex>`. The client always uses HTTPS (an optional private CA), signed
//! requests and `x-amz-checksum-sha256` on every upload request, so the service checks each request body; the
//! object's own SHA-256 is computed here over the stream and recorded by the database — the ETag is not a SHA-256
//! and is never used as one.
//!
//! Writes: an object up to one part is a single `PUT` with `If-None-Match: *` (unless `conditional_put: false`);
//! a larger one is a multipart upload of `part_size_bytes` parts, one buffered at a time, completed only after the
//! last part, then checked by `HEAD` (size). A key is fresh for every attempt and checked absent before a
//! multipart upload starts; `object_store` 0.14 completes multipart uploads without a precondition, so immutability
//! of large objects rests on those unique keys under database fencing. An abandoned multipart upload is aborted
//! by the writer's cleanup; after a crash the bucket needs a lifecycle rule that aborts incomplete multipart
//! uploads (see the deployment documentation).
//!
//! Reads fetch bounded ranges (4 MiB) one after another, so a slow client never holds a long remote request and
//! memory stays bounded.

use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use object_store::aws::{AmazonS3, AmazonS3Builder, AmazonS3ConfigKey, Checksum, S3ConditionalPut};
use object_store::path::Path as ObjectPath;
use object_store::{
    BackoffConfig, Certificate, ClientOptions, GetOptions, MultipartUpload, ObjectStore, ObjectStoreExt, PutMode,
    PutMultipartOptions, PutOptions, PutPayload, RetryConfig,
};
use tokio::io::{AsyncRead, ReadBuf};

use super::{blocking, ArtifactError, ArtifactKey, BoxFuture, RemoteFailure};
use crate::serve::batches::config::{S3Credentials, S3Settings};

/// Size of one ranged read request.
const READ_CHUNK: u64 = 4 * 1024 * 1024;

/// Credential variables of the AWS provider chain that `aws_default_chain` passes to the client. Endpoint,
/// region, HTTP and signing variables are deliberately not read from the environment.
const CREDENTIAL_VARS: [(&str, AmazonS3ConfigKey); 9] = [
    ("AWS_ACCESS_KEY_ID", AmazonS3ConfigKey::AccessKeyId),
    ("AWS_SECRET_ACCESS_KEY", AmazonS3ConfigKey::SecretAccessKey),
    ("AWS_SESSION_TOKEN", AmazonS3ConfigKey::Token),
    ("AWS_WEB_IDENTITY_TOKEN_FILE", AmazonS3ConfigKey::WebIdentityTokenFile),
    ("AWS_ROLE_ARN", AmazonS3ConfigKey::RoleArn),
    ("AWS_ROLE_SESSION_NAME", AmazonS3ConfigKey::RoleSessionName),
    ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", AmazonS3ConfigKey::ContainerCredentialsRelativeUri),
    ("AWS_CONTAINER_CREDENTIALS_FULL_URI", AmazonS3ConfigKey::ContainerCredentialsFullUri),
    ("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", AmazonS3ConfigKey::ContainerAuthorizationTokenFile),
];

/// A private bucket (and key prefix) of artifacts.
pub(crate) struct S3Store {
    s3: Arc<AmazonS3>,
    prefix: String,
    part_size: usize,
    conditional_put: bool,
}

impl std::fmt::Debug for S3Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Store").field("part_size", &self.part_size).finish_non_exhaustive()
    }
}

fn remote(op: &'static str) -> impl FnOnce(object_store::Error) -> ArtifactError {
    move |e| match e {
        object_store::Error::NotFound { .. } => ArtifactError::NotFound,
        object_store::Error::AlreadyExists { .. } => ArtifactError::AlreadyExists,
        object_store::Error::PermissionDenied { .. } | object_store::Error::Unauthenticated { .. } => {
            ArtifactError::Remote { op, failure: RemoteFailure::Denied }
        }
        object_store::Error::NotSupported { .. } | object_store::Error::NotImplemented { .. } => {
            ArtifactError::Remote { op, failure: RemoteFailure::Unsupported }
        }
        _ => ArtifactError::Remote { op, failure: RemoteFailure::Unavailable },
    }
}

impl S3Store {
    /// Build the client: credential sources and the CA file are read here (on the blocking pool).
    pub(crate) async fn open(settings: S3Settings) -> Result<Self, ArtifactError> {
        let prefix = settings.prefix.clone();
        let part_size = usize::try_from(settings.part_size_bytes)
            .map_err(|_| ArtifactError::Config("part_size_bytes does not fit this platform".into()))?;
        let conditional_put = settings.conditional_put;
        let s3 = blocking(move || build(&settings)).await?;
        Ok(Self { s3: Arc::new(s3), prefix, part_size, conditional_put })
    }

    fn location(&self, key: &ArtifactKey) -> Result<ObjectPath, ArtifactError> {
        self.path(&key.to_string())
    }

    fn path(&self, rest: &str) -> Result<ObjectPath, ArtifactError> {
        let full = if self.prefix.is_empty() { rest.to_string() } else { format!("{}/{rest}", self.prefix) };
        ObjectPath::parse(full).map_err(|_| ArtifactError::InvalidKey)
    }

    pub(crate) async fn create(&self, key: ArtifactKey, max_len: u64) -> Result<S3Writer, ArtifactError> {
        let location = self.location(&key)?;
        match self.s3.head(&location).await {
            Ok(_) => return Err(ArtifactError::AlreadyExists),
            Err(object_store::Error::NotFound { .. }) => {}
            Err(e) => return Err(remote("check")(e)),
        }
        let first_part = usize::try_from(max_len).unwrap_or(usize::MAX).min(self.part_size);
        Ok(S3Writer {
            s3: self.s3.clone(),
            location,
            part_size: self.part_size,
            first_part,
            conditional_put: self.conditional_put,
            buf: Vec::new(),
            upload: None,
        })
    }

    pub(crate) async fn open_range(
        &self,
        key: &ArtifactKey,
        object_len: u64,
        range: Range<u64>,
    ) -> Result<Pin<Box<dyn AsyncRead + Send>>, ArtifactError> {
        let location = self.location(key)?;
        // A missing object or a different size is reported at open (NotFound / Integrity), not as a read failure.
        let meta = self.s3.head(&location).await.map_err(remote("read"))?;
        if meta.size != object_len {
            return Err(ArtifactError::Integrity);
        }
        if range.is_empty() {
            return Ok(Box::pin(tokio::io::empty()));
        }
        Ok(Box::pin(RangeReader {
            s3: self.s3.clone(),
            location,
            object_len,
            next: range.start,
            end: range.end,
            current: Bytes::new(),
            pending: None,
        }))
    }

    pub(crate) async fn delete(&self, key: &ArtifactKey) -> Result<(), ArtifactError> {
        match self.s3.delete(&self.location(key)?).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(remote("delete")(e)),
        }
    }

    pub(crate) async fn probe(&self) -> Result<(), ArtifactError> {
        let location = self.path(&format!("v1/probe/{}", uuid::Uuid::new_v4().simple()))?;
        let body: &'static [u8] = b"basal-probe";
        let mode = if self.conditional_put { PutMode::Create } else { PutMode::Overwrite };
        let opts = PutOptions { mode, ..PutOptions::default() };
        self.s3.put_opts(&location, PutPayload::from(body), opts).await.map_err(remote("probe write"))?;
        let checked = async {
            let back = self.s3.get_range(&location, 0..body.len() as u64).await.map_err(remote("probe read"))?;
            if back.as_ref() != body {
                return Err(ArtifactError::Integrity);
            }
            Ok(())
        }
        .await;
        let deleted = self.s3.delete(&location).await.map_err(remote("probe delete"));
        checked?;
        deleted
    }
}

fn build(settings: &S3Settings) -> Result<AmazonS3, ArtifactError> {
    let timeout = Duration::from_millis(settings.request_timeout_ms);
    let mut client = ClientOptions::new()
        .with_timeout(timeout)
        .with_connect_timeout(Duration::from_millis(settings.connect_timeout_ms));
    if let Some(ca) = &settings.ca_file {
        let pem = std::fs::read(ca).map_err(ArtifactError::io("read CA file"))?;
        let certs = Certificate::from_pem_bundle(&pem)
            .map_err(|_| ArtifactError::Config("the CA file is not a PEM certificate bundle".into()))?;
        if certs.is_empty() {
            return Err(ArtifactError::Config("the CA file holds no certificate".into()));
        }
        for cert in certs {
            client = client.with_root_certificate(cert);
        }
    }
    let retry = RetryConfig {
        backoff: BackoffConfig::default(),
        max_retries: settings.max_retries as usize,
        retry_timeout: timeout.saturating_mul(settings.max_retries + 1),
    };
    let mut builder = AmazonS3Builder::new()
        .with_bucket_name(&settings.bucket)
        .with_region(&settings.region)
        .with_client_options(client)
        .with_allow_http(false)
        .with_skip_signature(false)
        .with_unsigned_payload(false)
        .with_retry(retry)
        .with_checksum_algorithm(Checksum::SHA256)
        .with_conditional_put(if settings.conditional_put {
            S3ConditionalPut::ETagMatch
        } else {
            S3ConditionalPut::Disabled
        })
        .with_virtual_hosted_style_request(settings.virtual_hosted_style);
    if let Some(endpoint) = &settings.endpoint {
        builder = builder.with_endpoint(endpoint);
    }
    match &settings.credentials {
        S3Credentials::AwsDefaultChain => {
            for (var, key) in CREDENTIAL_VARS {
                if let Some(value) = std::env::var(var).ok().filter(|v| !v.is_empty()) {
                    builder = builder.with_config(key, value);
                }
            }
        }
        S3Credentials::Static { access_key_id, secret_access_key, session_token } => {
            let field = "batches.artifacts.credentials";
            let id = access_key_id.read(field).map_err(|e| ArtifactError::Config(e.to_string()))?;
            let secret = secret_access_key.read(field).map_err(|e| ArtifactError::Config(e.to_string()))?;
            builder = builder.with_access_key_id(id.expose()).with_secret_access_key(secret.expose());
            if let Some(token) = session_token {
                let token = token.read(field).map_err(|e| ArtifactError::Config(e.to_string()))?;
                builder = builder.with_token(token.expose());
            }
        }
    }
    builder.build().map_err(|_| ArtifactError::Config("the S3 client could not be built from the configuration".into()))
}

/// Writer of one object: single `PUT` or multipart upload.
pub(crate) struct S3Writer {
    s3: Arc<AmazonS3>,
    location: ObjectPath,
    part_size: usize,
    /// Capacity of the first buffer: at most one part, at most the object's limit.
    first_part: usize,
    conditional_put: bool,
    buf: Vec<u8>,
    upload: Option<Box<dyn MultipartUpload>>,
}

impl S3Writer {
    pub(crate) async fn write(&mut self, mut data: &[u8]) -> Result<(), ArtifactError> {
        while !data.is_empty() {
            if self.buf.capacity() == 0 {
                let want = if self.upload.is_none() { self.first_part } else { self.part_size };
                self.buf.reserve_exact(want.max(1));
            }
            let take = (self.part_size - self.buf.len()).min(data.len());
            self.buf.extend_from_slice(&data[..take]);
            data = &data[take..];
            if self.buf.len() == self.part_size {
                self.flush_part().await?;
            }
        }
        Ok(())
    }

    /// Upload the buffered part (starting the multipart upload on the first one); the buffer is released before the
    /// next one is allocated.
    async fn flush_part(&mut self) -> Result<(), ArtifactError> {
        let upload = match &mut self.upload {
            Some(u) => u,
            None => {
                let u = self
                    .s3
                    .put_multipart_opts(&self.location, PutMultipartOptions::default())
                    .await
                    .map_err(remote("start upload"))?;
                self.upload.insert(u)
            }
        };
        let part = std::mem::take(&mut self.buf);
        upload.put_part(PutPayload::from(part)).await.map_err(remote("upload part"))
    }

    pub(crate) async fn finish(mut self, len: u64, deadline: Instant) -> Result<(), ArtifactError> {
        let result = self.finish_inner(len, deadline).await;
        if result.is_err() {
            self.discard().await;
        }
        result
    }

    async fn finish_inner(&mut self, len: u64, deadline: Instant) -> Result<(), ArtifactError> {
        if self.upload.is_none() {
            let mode = if self.conditional_put { PutMode::Create } else { PutMode::Overwrite };
            let payload = PutPayload::from(std::mem::take(&mut self.buf));
            self.s3
                .put_opts(&self.location, payload, PutOptions { mode, ..PutOptions::default() })
                .await
                .map_err(remote("write"))?;
        } else {
            if !self.buf.is_empty() {
                self.flush_part().await?;
            }
            if Instant::now() >= deadline {
                return Err(ArtifactError::DeadlineExceeded);
            }
            let Some(upload) = self.upload.as_mut() else { return Err(ArtifactError::Closed) };
            upload.complete().await.map_err(remote("complete upload"))?;
            self.upload = None;
        }
        let meta = self.s3.head(&self.location).await.map_err(remote("check"))?;
        if meta.size != len {
            return Err(ArtifactError::Integrity);
        }
        Ok(())
    }

    /// Abort an unfinished multipart upload (its parts are removed by the service); a single `PUT` leaves nothing.
    pub(crate) async fn discard(mut self) {
        self.buf = Vec::new();
        if let Some(mut upload) = self.upload.take() {
            let _ = upload.abort().await;
        }
    }
}

/// Sequential ranged reads of one object, each checked against the object's recorded size.
struct RangeReader {
    s3: Arc<AmazonS3>,
    location: ObjectPath,
    object_len: u64,
    next: u64,
    end: u64,
    current: Bytes,
    pending: Option<BoxFuture<Result<Bytes, ArtifactError>>>,
}

impl RangeReader {
    fn fetch(&self) -> BoxFuture<Result<Bytes, ArtifactError>> {
        let (s3, location, object_len) = (self.s3.clone(), self.location.clone(), self.object_len);
        let range = self.next..(self.next + READ_CHUNK).min(self.end);
        Box::pin(async move {
            let want = range.end - range.start;
            let got =
                s3.get_opts(&location, GetOptions::new().with_range(Some(range))).await.map_err(remote("read"))?;
            if got.meta.size != object_len {
                return Err(ArtifactError::Integrity);
            }
            let bytes = got.bytes().await.map_err(remote("read"))?;
            if bytes.len() as u64 != want {
                return Err(ArtifactError::Integrity);
            }
            Ok(bytes)
        })
    }
}

impl AsyncRead for RangeReader {
    fn poll_read(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        loop {
            if !this.current.is_empty() {
                let n = this.current.len().min(buf.remaining());
                buf.put_slice(&this.current.split_to(n));
                return Poll::Ready(Ok(()));
            }
            if this.next >= this.end {
                return Poll::Ready(Ok(()));
            }
            let fetch = match &mut this.pending {
                Some(f) => f,
                None => {
                    let f = this.fetch();
                    this.pending.insert(f)
                }
            };
            match fetch.as_mut().poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => {
                    this.pending = None;
                    match result {
                        Ok(bytes) => {
                            this.next += bytes.len() as u64;
                            this.current = bytes;
                        }
                        Err(ArtifactError::Integrity) => {
                            return Poll::Ready(Err(io::Error::new(io::ErrorKind::InvalidData, "size mismatch")))
                        }
                        // Deleted while being read: keeps its kind, so the reader reports NotFound.
                        Err(ArtifactError::NotFound) => {
                            return Poll::Ready(Err(io::Error::new(io::ErrorKind::NotFound, "object not found")))
                        }
                        Err(_) => return Poll::Ready(Err(io::Error::other("remote read failed"))),
                    }
                }
            }
        }
    }
}
