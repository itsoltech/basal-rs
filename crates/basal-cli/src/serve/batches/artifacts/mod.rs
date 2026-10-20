//! Immutable artifacts of the Batch API: uploaded inputs, large record results and exported result files.
//!
//! [`ArtifactStore`] writes and reads objects under generated keys ([`ArtifactKey`]: kind + random UUID v4), never
//! under client names or paths. A key is written once: a second write of the same key fails
//! ([`ArtifactError::AlreadyExists`]) instead of replacing it. The database owns everything else — sessions, pins,
//! references, reservations and the decision when an object may be deleted:
//!
//! 1. the store creates an `artifact_sessions` row with the key (and reserves quota) before the first byte;
//! 2. [`ArtifactStore::create`] → [`ArtifactWriter::write`]… → [`ArtifactWriter::finish`] makes the object durable
//!    and returns its length and SHA-256 ([`StoredObject`]), without publishing anything;
//! 3. a database transaction publishes the reference (or the losing lease leaves the object orphaned);
//! 4. GC calls [`ArtifactStore::delete`] only after the database allows it (session ended past its grace, file
//!    deleted past the download grace); deletion is idempotent and also removes the partial data of a crashed write.
//!
//! Writers are bounded: a size limit checked before data is accepted, and a deadline after which they refuse to
//! write or finish, so a fenced writer cannot recreate an object after the orphan grace. A writer dropped without
//! `finish` (handler cancelled, error) removes its partial data in a background task that keeps the caller's
//! concurrency permit until it is done. Reads stream with a bounded buffer, never load an object whole, check the
//! recorded length and SHA-256 (of the object or of the requested range) and fail at the end on a mismatch; a read
//! with a deadline (downloads) is cut off when it passes, even if the client stopped reading.
//!
//! Errors are typed and carry no credentials, URLs or object keys.

mod filesystem;
mod s3;

use std::future::Future;
use std::io;
use std::ops::Range;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};
use std::time::Instant;

use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, ReadBuf};
use tokio::sync::OwnedSemaphorePermit;

use super::config::{hex, ArtifactSettings};

pub(crate) use filesystem::FilesystemStore;
pub(crate) use s3::S3Store;

/// What an artifact holds; the first segment of its key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ArtifactKind {
    /// An uploaded input file.
    Upload,
    /// A record result too large to keep inline.
    Result,
    /// An exported output or error file.
    Export,
}

impl ArtifactKind {
    pub(crate) const ALL: [Self; 3] = [Self::Upload, Self::Result, Self::Export];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Upload => "upload",
            Self::Result => "result",
            Self::Export => "export",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }
}

/// Key of an immutable object: `v1/<kind>/<32 lowercase hex>` (a random UUID v4). The database stores it as text;
/// [`ArtifactKey::parse`] accepts nothing else, so a stored key can never name a path outside the store.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ArtifactKey {
    kind: ArtifactKind,
    id: uuid::Uuid,
}

impl ArtifactKey {
    /// A new key (one per write attempt: a retried write gets a new key).
    pub(crate) fn generate(kind: ArtifactKind) -> Self {
        Self { kind, id: uuid::Uuid::new_v4() }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, ArtifactError> {
        let rest = s.strip_prefix("v1/").ok_or(ArtifactError::InvalidKey)?;
        let (kind, hex) = rest.split_once('/').ok_or(ArtifactError::InvalidKey)?;
        let kind = ArtifactKind::parse(kind).ok_or(ArtifactError::InvalidKey)?;
        if hex.len() != 32 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return Err(ArtifactError::InvalidKey);
        }
        let id = uuid::Uuid::try_parse(hex).map_err(|_| ArtifactError::InvalidKey)?;
        if id.get_version() != Some(uuid::Version::Random) {
            return Err(ArtifactError::InvalidKey);
        }
        Ok(Self { kind, id })
    }

    pub(crate) fn kind(&self) -> ArtifactKind {
        self.kind
    }

    /// The 32 hex digits of the key.
    fn hex(&self) -> String {
        self.id.simple().to_string()
    }
}

impl std::fmt::Display for ArtifactKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v1/{}/{}", self.kind.as_str(), self.id.simple())
    }
}

impl std::fmt::Debug for ArtifactKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ArtifactKey({self})")
    }
}

/// SHA-256 of an object or of a byte range of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Sha256Digest(pub(crate) [u8; 32]);

impl std::fmt::Display for Sha256Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&hex(&self.0))
    }
}

impl std::fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sha256Digest({self})")
    }
}

/// A durable object, not yet referenced by the database.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StoredObject {
    pub(crate) key: ArtifactKey,
    pub(crate) len: u64,
    pub(crate) sha256: Sha256Digest,
}

/// Why an object storage request failed (no URL, bucket or credentials).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RemoteFailure {
    /// Credentials refused or missing permission.
    Denied,
    /// The service does not support a required operation (for example conditional writes).
    Unsupported,
    /// Network, timeout, service error or an unexpected answer.
    Unavailable,
}

impl std::fmt::Display for RemoteFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Denied => "access denied",
            Self::Unsupported => "operation not supported by the service",
            Self::Unavailable => "service unavailable",
        })
    }
}

/// Errors of the artifact store. None contains a credential, URL, object key or local path.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ArtifactError {
    #[error("invalid artifact key")]
    InvalidKey,
    #[error("the artifact already exists")]
    AlreadyExists,
    #[error("the artifact does not exist")]
    NotFound,
    #[error("the artifact exceeds its limit of {limit} bytes")]
    LimitExceeded { limit: u64 },
    #[error("the artifact operation passed its deadline")]
    DeadlineExceeded,
    #[error("the artifact does not match its recorded length or SHA-256")]
    Integrity,
    #[error("the range {start}..{end} is outside the artifact of {len} bytes")]
    InvalidRange { start: u64, end: u64, len: u64 },
    #[error("the artifact directory is not private: {0}")]
    Insecure(&'static str),
    #[error("artifact storage configuration: {0}")]
    Config(String),
    #[error("artifact {op} failed")]
    Io {
        op: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("object storage {op} failed: {failure}")]
    Remote { op: &'static str, failure: RemoteFailure },
    #[error("the artifact writer was closed after an earlier error")]
    Closed,
    #[error("an artifact task stopped unexpectedly")]
    Task,
}

/// How a caller should treat an [`ArtifactError`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorClass {
    /// Deterministic for this object or request: retrying gives the same answer (missing object, length or SHA-256
    /// mismatch, range outside the object, malformed key, size limit).
    Permanent,
    /// May succeed later: I/O, an unavailable or timed-out remote, a passed deadline, a stopped task.
    Transient,
    /// The store itself is misconfigured or refused (permissions, unsupported operation): not per record.
    Configuration,
}

impl ArtifactError {
    /// Classification for retries: see [`ErrorClass`].
    pub(crate) fn class(&self) -> ErrorClass {
        match self {
            Self::InvalidKey
            | Self::NotFound
            | Self::Integrity
            | Self::InvalidRange { .. }
            | Self::LimitExceeded { .. } => ErrorClass::Permanent,
            Self::Insecure(_)
            | Self::Config(_)
            | Self::Remote { failure: RemoteFailure::Denied | RemoteFailure::Unsupported, .. } => {
                ErrorClass::Configuration
            }
            Self::AlreadyExists
            | Self::DeadlineExceeded
            | Self::Io { .. }
            | Self::Remote { failure: RemoteFailure::Unavailable, .. }
            | Self::Closed
            | Self::Task => ErrorClass::Transient,
        }
    }

    /// The typed error of a failed read of an [`ArtifactReader`]: integrity, deadline and missing-object failures
    /// keep their meaning through the `io::Error` of `AsyncRead`.
    pub(crate) fn from_read(op: &'static str, error: io::Error) -> Self {
        match error.kind() {
            io::ErrorKind::InvalidData => Self::Integrity,
            io::ErrorKind::TimedOut => Self::DeadlineExceeded,
            io::ErrorKind::NotFound => Self::NotFound,
            _ => Self::Io { op, source: error },
        }
    }

    /// The storage cannot serve now: Batch/Files answer 503 (online inference is unaffected).
    pub(crate) fn is_unavailable(&self) -> bool {
        matches!(self, Self::Io { .. } | Self::Remote { .. } | Self::Task | Self::Insecure(_) | Self::Config(_))
    }

    fn io(op: &'static str) -> impl FnOnce(io::Error) -> Self {
        move |source| Self::Io { op, source }
    }
}

/// The artifact store of a deployment: a private local directory or a private S3-compatible bucket. Cheap to clone.
#[derive(Clone, Debug)]
pub(crate) enum ArtifactStore {
    Filesystem(Arc<FilesystemStore>),
    S3(Arc<S3Store>),
}

impl ArtifactStore {
    /// Open the configured store: create or check the private directory, or build the S3 client (reading its
    /// credential sources). Does not touch objects; see [`ArtifactStore::probe`]. `buffer_bytes` bounds the buffer of
    /// one write or read stream (`batches.host.upload_buffer_bytes`).
    pub(crate) async fn open(settings: &ArtifactSettings, buffer_bytes: usize) -> Result<Self, ArtifactError> {
        match settings {
            ArtifactSettings::Filesystem { root } => {
                Ok(Self::Filesystem(Arc::new(FilesystemStore::open(root.clone(), buffer_bytes).await?)))
            }
            ArtifactSettings::S3(s3) => Ok(Self::S3(Arc::new(S3Store::open((**s3).clone()).await?))),
        }
    }

    /// Backend label for metrics and health.
    pub(crate) fn backend_label(&self) -> &'static str {
        match self {
            Self::Filesystem(_) => "filesystem",
            Self::S3(_) => "s3",
        }
    }

    /// Start writing `key`: at most `max_len` bytes, finished before `deadline`. `permit` (an upload or finalization
    /// slot) lives with the writer and with its cleanup. The key must be fresh: its session row exists, and no
    /// other writer uses it.
    pub(crate) async fn create(
        &self,
        key: ArtifactKey,
        max_len: u64,
        deadline: Instant,
        permit: Option<OwnedSemaphorePermit>,
    ) -> Result<ArtifactWriter, ArtifactError> {
        if Instant::now() >= deadline {
            return Err(ArtifactError::DeadlineExceeded);
        }
        let sink = match self {
            Self::Filesystem(fs) => Sink::Filesystem(fs.create(key).await?),
            Self::S3(s3) => Sink::S3(s3.create(key, max_len).await?),
        };
        Ok(ArtifactWriter {
            key,
            limit: max_len,
            deadline,
            written: 0,
            hasher: Sha256::new(),
            sink: Some(sink),
            permit,
        })
    }

    /// Read a whole object, checking its recorded length and SHA-256 (`object`); a mismatch fails the read at the end.
    /// With `deadline` the read is cut off when it passes.
    pub(crate) async fn read(
        &self,
        object: &StoredObject,
        deadline: Option<Instant>,
    ) -> Result<ArtifactReader, ArtifactError> {
        let inner = self.open_range(&object.key, object.len, 0..object.len).await?;
        Ok(ArtifactReader::verified(inner, object.len, object.sha256, deadline))
    }

    /// Read `range` of an object of `object_len` bytes. With `range_sha256` (the digest a validator recorded for a
    /// record line) the range is verified like a whole object; without, only its length is checked.
    pub(crate) async fn read_range(
        &self,
        key: &ArtifactKey,
        object_len: u64,
        range: Range<u64>,
        range_sha256: Option<Sha256Digest>,
        deadline: Option<Instant>,
    ) -> Result<ArtifactReader, ArtifactError> {
        let len = range.end.saturating_sub(range.start);
        let inner = self.open_range(key, object_len, range).await?;
        let reader = match range_sha256 {
            Some(digest) => ArtifactReader::verified(inner, len, digest, deadline),
            None => ArtifactReader::with_length(inner, len, deadline),
        };
        Ok(reader)
    }

    async fn open_range(
        &self,
        key: &ArtifactKey,
        object_len: u64,
        range: Range<u64>,
    ) -> Result<Pin<Box<dyn AsyncRead + Send>>, ArtifactError> {
        if range.start > range.end || range.end > object_len {
            return Err(ArtifactError::InvalidRange { start: range.start, end: range.end, len: object_len });
        }
        match self {
            Self::Filesystem(fs) => fs.open_range(key, object_len, range).await,
            Self::S3(s3) => s3.open_range(key, object_len, range).await,
        }
    }

    /// Remove the object and any partial data of `key`. Idempotent; the removal is durable when it returns, so the
    /// database may then release the bytes. Call only when the database allows it.
    pub(crate) async fn delete(&self, key: &ArtifactKey) -> Result<(), ArtifactError> {
        match self {
            Self::Filesystem(fs) => fs.delete(key).await,
            Self::S3(s3) => s3.delete(key).await,
        }
    }

    /// Check that the store can durably write, read and delete now (a small probe object under its own prefix).
    pub(crate) async fn probe(&self) -> Result<(), ArtifactError> {
        match self {
            Self::Filesystem(fs) => fs.probe().await,
            Self::S3(s3) => s3.probe().await,
        }
    }
}

enum Sink {
    Filesystem(filesystem::FsWriter),
    S3(s3::S3Writer),
}

impl Sink {
    async fn write(&mut self, data: &[u8]) -> Result<(), ArtifactError> {
        match self {
            Self::Filesystem(w) => w.write(data).await,
            Self::S3(w) => w.write(data).await,
        }
    }

    async fn finish(self, len: u64, deadline: Instant) -> Result<(), ArtifactError> {
        match self {
            Self::Filesystem(w) => w.finish(deadline).await,
            Self::S3(w) => w.finish(len, deadline).await,
        }
    }

    /// Remove partial data (background-safe, idempotent).
    async fn discard(self) {
        match self {
            Self::Filesystem(w) => w.discard().await,
            Self::S3(w) => w.discard().await,
        }
    }

    /// Remove partial data without a runtime (best effort).
    fn discard_blocking(self) {
        match self {
            Self::Filesystem(w) => w.discard_blocking(),
            // Without a runtime the multipart upload cannot be aborted; the bucket's lifecycle rule for incomplete
            // multipart uploads removes its parts.
            Self::S3(_) => {}
        }
    }
}

/// Writes one object. Bytes are hashed as they arrive; [`ArtifactWriter::finish`] makes the object durable under
/// its key. Dropping the writer before `finish` (or any failed call) discards the partial data in the background.
pub(crate) struct ArtifactWriter {
    key: ArtifactKey,
    limit: u64,
    deadline: Instant,
    written: u64,
    hasher: Sha256,
    /// `None` once finished, discarded or failed.
    sink: Option<Sink>,
    permit: Option<OwnedSemaphorePermit>,
}

impl std::fmt::Debug for ArtifactWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArtifactWriter")
            .field("key", &self.key)
            .field("limit", &self.limit)
            .field("written", &self.written)
            .field("open", &self.sink.is_some())
            .finish_non_exhaustive()
    }
}

impl ArtifactWriter {
    /// Move the deadline, for a writer whose lease was renewed (export). A later deadline only follows a renewal
    /// the database accepted; the orphan grace still covers the remote request time after it.
    pub(crate) fn set_deadline(&mut self, deadline: Instant) {
        self.deadline = deadline;
    }

    /// Append `data`. The limit and deadline are checked before the data is accepted; a refusal or error closes the
    /// writer and discards what was written ([`ArtifactError::LimitExceeded`] is the 413 of an upload).
    pub(crate) async fn write(&mut self, data: &[u8]) -> Result<(), ArtifactError> {
        if self.sink.is_none() {
            return Err(ArtifactError::Closed);
        }
        if Instant::now() >= self.deadline {
            self.close();
            return Err(ArtifactError::DeadlineExceeded);
        }
        if self.written.checked_add(data.len() as u64).is_none_or(|n| n > self.limit) {
            self.close();
            return Err(ArtifactError::LimitExceeded { limit: self.limit });
        }
        let Some(sink) = self.sink.as_mut() else { return Err(ArtifactError::Closed) };
        if let Err(e) = sink.write(data).await {
            self.close();
            return Err(e);
        }
        self.hasher.update(data);
        self.written += data.len() as u64;
        Ok(())
    }

    /// Make the object durable under its key and return its length and SHA-256. Nothing is published: the caller
    /// records the result in the database. The last step runs in its own task, so cancelling the caller does not
    /// interrupt it half-way; its permit is released when it ends. On error the partial data is discarded.
    pub(crate) async fn finish(mut self) -> Result<StoredObject, ArtifactError> {
        let sink = self.sink.take().ok_or(ArtifactError::Closed)?;
        let permit = self.permit.take();
        let (key, len, deadline) = (self.key, self.written, self.deadline);
        let sha256 = Sha256Digest(std::mem::take(&mut self.hasher).finalize().into());
        let task = tokio::spawn(async move {
            let _permit = permit;
            if Instant::now() >= deadline {
                sink.discard().await;
                return Err(ArtifactError::DeadlineExceeded);
            }
            sink.finish(len, deadline).await
        });
        task.await.map_err(|_| ArtifactError::Task)??;
        Ok(StoredObject { key, len, sha256 })
    }

    /// Close after a refusal or error: discard in the background, keeping the permit until that is done.
    fn close(&mut self) {
        if let Some(sink) = self.sink.take() {
            discard_in_background(sink, self.permit.take());
        }
    }
}

impl Drop for ArtifactWriter {
    fn drop(&mut self) {
        self.close();
    }
}

fn discard_in_background(sink: Sink, permit: Option<OwnedSemaphorePermit>) {
    match tokio::runtime::Handle::try_current() {
        Ok(rt) => {
            rt.spawn(async move {
                let _permit = permit;
                sink.discard().await;
            });
        }
        Err(_) => sink.discard_blocking(),
    }
}

type BoxedRead = Pin<Box<dyn AsyncRead + Send>>;

/// Source and admission slot share one lifetime, including when an HTTP body is never polled again.
struct ReaderSource {
    inner: Option<BoxedRead>,
    permit: Option<OwnedSemaphorePermit>,
    waker: Option<Waker>,
}

fn close_reader(source: &Mutex<ReaderSource>) {
    let (inner, permit, waker) = {
        let mut source = source.lock().unwrap_or_else(PoisonError::into_inner);
        (source.inner.take(), source.permit.take(), source.waker.take())
    };
    drop(inner);
    drop(permit);
    if let Some(waker) = waker {
        waker.wake();
    }
}

/// A verified, bounded read of an object or range. Implements `AsyncRead`; [`ArtifactReader::into_body`] streams it
/// as an HTTP body. Errors carry only their kind (no paths or URLs).
pub(crate) struct ArtifactReader {
    /// Taken away by the deadline watchdog, which closes the file or remote stream even while nobody polls.
    inner: Arc<Mutex<ReaderSource>>,
    /// Deadline and stop watchers; aborted when the reader is dropped.
    watchdogs: Vec<tokio::task::AbortHandle>,
    deadline: Option<Instant>,
    expected_len: u64,
    /// `None` when only the length is checked.
    expected_sha256: Option<Sha256Digest>,
    hasher: Option<Sha256>,
    read: u64,
    done: bool,
}

impl ArtifactReader {
    /// A read of `len` bytes whose SHA-256 must be `sha256`.
    fn verified(inner: BoxedRead, len: u64, sha256: Sha256Digest, deadline: Option<Instant>) -> Self {
        let mut r = Self::with_length(inner, len, deadline);
        r.expected_sha256 = Some(sha256);
        r.hasher = Some(Sha256::new());
        r
    }

    /// A read of exactly `len` bytes.
    fn with_length(inner: BoxedRead, len: u64, deadline: Option<Instant>) -> Self {
        let inner = Arc::new(Mutex::new(ReaderSource { inner: Some(inner), permit: None, waker: None }));
        let watchdog = deadline.and_then(|deadline| {
            let weak = Arc::downgrade(&inner);
            let rt = tokio::runtime::Handle::try_current().ok()?;
            let task = rt.spawn(async move {
                tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;
                if let Some(inner) = weak.upgrade() {
                    close_reader(&inner);
                }
            });
            Some(task.abort_handle())
        });
        Self {
            inner,
            watchdogs: watchdog.into_iter().collect(),
            deadline,
            expected_len: len,
            expected_sha256: None,
            hasher: None,
            read: 0,
            done: false,
        }
    }

    /// Close the source when `stop` is cancelled (service shutdown), also while nobody polls.
    pub(crate) fn cancel_on(mut self, stop: tokio_util::sync::CancellationToken) -> Self {
        let weak = Arc::downgrade(&self.inner);
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            let task = rt.spawn(async move {
                stop.cancelled().await;
                if let Some(inner) = weak.upgrade() {
                    close_reader(&inner);
                }
            });
            self.watchdogs.push(task.abort_handle());
        }
        self
    }

    /// Keep `permit` (a download slot) until EOF, drop, deadline or shutdown, even for an unpolled body.
    pub(crate) fn with_permit(self, permit: OwnedSemaphorePermit) -> Self {
        {
            let mut source = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            // A watchdog may have closed the source before the caller attached its permit.
            if source.inner.is_some() {
                source.permit = Some(permit);
            }
        }
        self
    }

    /// The whole read as an HTTP body, in chunks of at most `chunk` bytes.
    pub(crate) fn into_body(self, chunk: usize) -> axum::body::Body {
        axum::body::Body::from_stream(tokio_util::io::ReaderStream::with_capacity(self, chunk.max(1)))
    }

    /// Read everything into memory; for bounded reads only (record lines), refused above `max` bytes.
    pub(crate) async fn read_all(mut self, max: usize) -> Result<Vec<u8>, ArtifactError> {
        use tokio::io::AsyncReadExt;
        if self.expected_len > max as u64 {
            return Err(ArtifactError::LimitExceeded { limit: max as u64 });
        }
        let mut out = Vec::with_capacity(self.expected_len as usize);
        self.read_to_end(&mut out).await.map_err(|e| ArtifactError::from_read("read", e))?;
        Ok(out)
    }

    fn fail(&mut self, kind: io::ErrorKind, message: &'static str) -> Poll<io::Result<()>> {
        self.done = true;
        close_reader(&self.inner);
        Poll::Ready(Err(io::Error::new(kind, message)))
    }
}

impl AsyncRead for ArtifactReader {
    fn poll_read(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if this.done || buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        if this.deadline.is_some_and(|d| Instant::now() >= d) {
            return this.fail(io::ErrorKind::TimedOut, "artifact read passed its deadline");
        }
        let before = buf.filled().len();
        let polled = {
            let mut guard = this.inner.lock().unwrap_or_else(PoisonError::into_inner);
            guard.waker = Some(cx.waker().clone());
            match guard.inner.as_mut() {
                Some(inner) => inner.as_mut().poll_read(cx, buf),
                None => Poll::Ready(Err(io::Error::new(io::ErrorKind::TimedOut, "closed"))),
            }
        };
        match polled {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(e)) => {
                let kind = match e.kind() {
                    k @ (io::ErrorKind::TimedOut | io::ErrorKind::InvalidData | io::ErrorKind::NotFound) => k,
                    _ => io::ErrorKind::Other,
                };
                this.fail(kind, "artifact read failed")
            }
            Poll::Ready(Ok(())) => {
                let new = &buf.filled()[before..];
                if new.is_empty() {
                    let digest_ok = match (this.hasher.take(), this.expected_sha256) {
                        (Some(h), Some(expected)) => <[u8; 32]>::from(h.finalize()) == expected.0,
                        _ => true,
                    };
                    if this.read != this.expected_len || !digest_ok {
                        return this.fail(io::ErrorKind::InvalidData, "artifact integrity check failed");
                    }
                    this.done = true;
                    close_reader(&this.inner);
                    return Poll::Ready(Ok(()));
                }
                this.read += new.len() as u64;
                if this.read > this.expected_len {
                    buf.set_filled(before);
                    return this.fail(io::ErrorKind::InvalidData, "artifact integrity check failed");
                }
                if let Some(h) = &mut this.hasher {
                    h.update(new);
                }
                Poll::Ready(Ok(()))
            }
        }
    }
}

impl Drop for ArtifactReader {
    fn drop(&mut self) {
        for w in self.watchdogs.drain(..) {
            w.abort();
        }
        close_reader(&self.inner);
    }
}

/// Run blocking filesystem calls on the blocking pool.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, ArtifactError> + Send + 'static,
) -> Result<T, ArtifactError> {
    tokio::task::spawn_blocking(f).await.map_err(|_| ArtifactError::Task)?
}

/// A boxed `Send` future.
type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
