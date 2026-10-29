//! Typed store errors. Driver errors are reduced to a redacted [`DbError`]: engine, kind, SQLSTATE/SQLite code
//! and constraint name only, never the driver message (it may echo client values) or connection details.

use std::fmt;

use super::types::{BackendKind, OwnerId, SchemaState};

/// A value failed validation at the store boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid {field}: {reason}")]
pub(crate) struct InvalidValue {
    pub field: &'static str,
    pub reason: &'static str,
}

impl InvalidValue {
    pub(crate) const fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

/// Quota dimension that refused an admission step (upload, create or claim).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuotaResource {
    Bytes,
    ActiveBatches,
    RetainedBatches,
    Files,
    UploadSessions,
}

impl fmt::Display for QuotaResource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bytes => "bytes",
            Self::ActiveBatches => "active batches",
            Self::RetainedBatches => "retained batches",
            Self::Files => "files",
            Self::UploadSessions => "upload sessions",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DbErrorKind {
    /// Connection, pool or transport failure.
    Unavailable,
    /// Lock timeout, deadlock, serialization failure or SQLite busy: the transaction may be retried.
    Busy,
    /// Statement timeout or cancellation.
    Timeout,
    UniqueViolation,
    ConstraintViolation,
    /// Stored value with an unexpected type or shape.
    Decode,
    Other,
}

/// Redacted database failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DbError {
    pub backend: BackendKind,
    pub kind: DbErrorKind,
    /// SQLSTATE (PostgreSQL) or extended result code (SQLite).
    pub code: Option<String>,
    pub constraint: Option<String>,
}

impl fmt::Display for DbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?} error", self.backend.as_str(), self.kind)?;
        if let Some(code) = &self.code {
            write!(f, " (code {code})")?;
        }
        if let Some(constraint) = &self.constraint {
            write!(f, " on {constraint}")?;
        }
        Ok(())
    }
}

impl std::error::Error for DbError {}

#[derive(Debug, thiserror::Error)]
pub(crate) enum StoreError {
    #[error(transparent)]
    Invalid(#[from] InvalidValue),
    /// Missing resource or a resource of another owner: the API answers both with the same 404.
    #[error("resource not found")]
    NotFound,
    #[error("file is pinned by an active batch")]
    FileInUse,
    #[error("idempotency key was used with a different request")]
    IdempotencyConflict,
    #[error("quota exceeded: {0}")]
    QuotaExceeded(QuotaResource),
    #[error("owner has no quota in the active policy")]
    OwnerNotProvisioned,
    /// An API key for a disabled user.
    #[error("the API user is disabled")]
    UserDisabled,
    #[error("no quota policy is installed")]
    PolicyMissing,
    #[error("no model catalog is installed")]
    CatalogMissing,
    #[error("input file cannot be used: {0}")]
    InputUnusable(&'static str),
    #[error("lease or claim is stale")]
    LeaseStale,
    #[error("upload session is closed or expired")]
    SessionClosed,
    #[error("validated line {line_no} differs from the line ingested earlier")]
    IngestConflict { line_no: u64 },
    #[error("ingestion rejected: {0}")]
    IngestRejected(&'static str),
    #[error("seal rejected: {0}")]
    SealRejected(&'static str),
    #[error("result rejected: {0}")]
    ResultInvalid(&'static str),
    #[error("publication rejected: {0}")]
    PublishRejected(&'static str),
    #[error("finalization takeover is not yet safe")]
    FencingPending,
    #[error("policy rejected: {0}")]
    PolicyInvalid(&'static str),
    #[error("policy limits of owner {0} are below its current commitments")]
    PolicyBelowCommitted(OwnerId),
    #[error("batch storage schema is not usable: {0:?}")]
    Schema(SchemaState),
    #[error("invalid store configuration: {0}")]
    Config(&'static str),
    #[error("batch storage unavailable")]
    Unavailable(#[source] DbError),
    #[error("batch storage busy")]
    Busy(#[source] DbError),
    #[error("outcome of the database commit is unknown; reconcile with the same operation id")]
    CommitUnknown(#[source] DbError),
    #[error("batch storage operation failed")]
    Database(#[source] DbError),
    #[error("stored batch data is inconsistent: {0}")]
    Corrupt(&'static str),
}

impl StoreError {
    /// The transaction was rolled back and running the same domain operation again is safe.
    pub(crate) fn is_retryable(&self) -> bool {
        matches!(self, Self::Busy(_))
    }

    /// Batch/Files should answer 503 without affecting online inference.
    pub(crate) fn is_unavailable(&self) -> bool {
        matches!(
            self,
            Self::Unavailable(_) | Self::Busy(_) | Self::CommitUnknown(_) | Self::Schema(_) | Self::Database(_)
        )
    }

    pub(crate) fn from_db(error: DbError) -> Self {
        match error.kind {
            DbErrorKind::Unavailable => Self::Unavailable(error),
            DbErrorKind::Busy => Self::Busy(error),
            DbErrorKind::Timeout => Self::Busy(error),
            _ => Self::Database(error),
        }
    }

    pub(crate) fn unique_violation(&self) -> bool {
        matches!(self, Self::Database(DbError { kind: DbErrorKind::UniqueViolation, .. }))
    }
}

/// Map a driver error without its message.
pub(super) fn redact_sqlx(backend: BackendKind, error: &sqlx::Error) -> DbError {
    use sqlx::Error as E;
    let (kind, code, constraint) = match error {
        E::Database(db) => {
            let code = db.code().map(|c| c.into_owned());
            let constraint = db.constraint().map(str::to_owned);
            (classify_database(backend, code.as_deref(), db.kind()), code, constraint)
        }
        E::Io(_) | E::Tls(_) | E::Protocol(_) | E::PoolTimedOut | E::PoolClosed | E::WorkerCrashed => {
            (DbErrorKind::Unavailable, None, None)
        }
        E::ColumnDecode { .. } | E::ColumnNotFound(_) | E::ColumnIndexOutOfBounds { .. } | E::Decode(_) => {
            (DbErrorKind::Decode, None, None)
        }
        E::RowNotFound | E::TypeNotFound { .. } | E::Configuration(_) => (DbErrorKind::Other, None, None),
        _ => (DbErrorKind::Other, None, None),
    };
    DbError { backend, kind, code, constraint }
}

fn classify_database(backend: BackendKind, code: Option<&str>, kind: sqlx::error::ErrorKind) -> DbErrorKind {
    use sqlx::error::ErrorKind as K;
    match kind {
        K::UniqueViolation => return DbErrorKind::UniqueViolation,
        K::ForeignKeyViolation | K::NotNullViolation | K::CheckViolation => {
            return DbErrorKind::ConstraintViolation;
        }
        _ => {}
    }
    let Some(code) = code else { return DbErrorKind::Other };
    match backend {
        // 40001 serialization_failure, 40P01 deadlock_detected, 55P03 lock_not_available, 57014 query_canceled
        // (statement_timeout), 08* connection exceptions, 57P01..57P03 admin shutdown / cannot connect now.
        BackendKind::Postgresql => match code {
            "40001" | "40P01" | "55P03" => DbErrorKind::Busy,
            "57014" => DbErrorKind::Timeout,
            c if c.starts_with("08") || c.starts_with("57P") => DbErrorKind::Unavailable,
            _ => DbErrorKind::Other,
        },
        // Primary codes 5 SQLITE_BUSY and 6 SQLITE_LOCKED, including their extended variants.
        // MongoDB errors never pass through sqlx (see `mongodb::driver`).
        BackendKind::Mongodb => DbErrorKind::Other,
        // SQL Server errors never pass through sqlx (see `mssql`).
        BackendKind::Mssql => DbErrorKind::Other,
        BackendKind::Sqlite => match code.parse::<i64>() {
            Ok(value) if matches!(value & 0xff, 5 | 6) => DbErrorKind::Busy,
            Ok(value) if matches!(value & 0xff, 10 | 13 | 14) => DbErrorKind::Unavailable,
            _ => DbErrorKind::Other,
        },
    }
}
