//! Domain records of the durable Batch store, independent of HTTP DTOs and database drivers.
//!
//! Identifiers are validated when constructed, so adapters only ever bind checked values. Every timestamp is
//! [`UnixMillis`] read from the database clock of the transaction that wrote it (PostgreSQL `clock_timestamp()` after
//! the row locks, SQLite the clock of the local adapter after `BEGIN IMMEDIATE`). Line numbers are 1-based and equal
//! the record ordinal, because the validator rejects empty lines.

use std::fmt;
use std::time::Duration;

use super::error::InvalidValue;

/// Milliseconds since the Unix epoch, taken from the database clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct UnixMillis(pub i64);

impl UnixMillis {
    /// Whole seconds, the unit of the OpenAI-like timestamps.
    pub(crate) fn as_unix_seconds(self) -> i64 {
        self.0.div_euclid(1000)
    }
}

/// SHA-256 digest of bytes or of a canonical request description.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Sha256Digest(pub [u8; 32]);

impl Sha256Digest {
    pub(crate) fn from_slice(bytes: &[u8]) -> Result<Self, InvalidValue> {
        <[u8; 32]>::try_from(bytes).map(Self).map_err(|_| InvalidValue::new("sha256", "must have 32 bytes"))
    }

    pub(crate) fn to_hex(self) -> String {
        hex(&self.0)
    }
}

impl fmt::Debug for Sha256Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sha256({})", self.to_hex())
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

fn check_len(field: &'static str, value: &str, max: usize) -> Result<(), InvalidValue> {
    if value.is_empty() || value.len() > max {
        return Err(InvalidValue::new(field, "has an invalid length"));
    }
    Ok(())
}

fn check_ascii(field: &'static str, value: &str, allowed: impl Fn(u8) -> bool) -> Result<(), InvalidValue> {
    if value.bytes().all(allowed) {
        Ok(())
    } else {
        Err(InvalidValue::new(field, "contains a disallowed character"))
    }
}

macro_rules! string_id {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(crate) struct $name(String);

        impl $name {
            pub(crate) fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({:?})", stringify!($name), self.0)
            }
        }
    };
}

string_id! {
    /// Storage namespace of batches, files and quota; set by server configuration.
    /// 1–128 bytes of `[A-Za-z0-9._:@-]`; never derived from request headers.
    OwnerId
}

impl OwnerId {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("owner_id", value, 128)?;
        check_ascii("owner_id", value, |b| b.is_ascii_alphanumeric() || b"._:@-".contains(&b))?;
        Ok(Self(value.to_owned()))
    }
}

fn parse_prefixed_uuid(field: &'static str, prefix: &str, value: &str) -> Result<String, InvalidValue> {
    let Some(rest) = value.strip_prefix(prefix) else {
        return Err(InvalidValue::new(field, "has an unexpected prefix"));
    };
    if rest.len() != 32 || !rest.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(InvalidValue::new(field, "is not a resource identifier"));
    }
    Ok(value.to_owned())
}

fn random_suffix() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

macro_rules! resource_id {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        string_id! { $(#[$doc])* $name }

        impl $name {
            /// A new random UUID v4 identifier; it never encodes host names or paths.
            pub(crate) fn new_random() -> Self {
                Self(format!("{}{}", $prefix, random_suffix()))
            }

        }
    };
}

resource_id! {
    /// Public batch identifier `batch_<32 hex>`.
    BatchId, "batch_"
}
resource_id! {
    /// Public file identifier `file_<32 hex>`.
    FileId, "file_"
}
resource_id! {
    /// Lease or execution claim, unique per acquisition.
    ClaimId, "claim_"
}
resource_id! {
    /// Random identity of one worker process, drawn at every start.
    WorkerId, "worker_"
}
resource_id! {
    /// Artifact session (upload, large result or export) protecting an object from GC.
    SessionId, "session_"
}

impl BatchId {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        parse_prefixed_uuid("BatchId", "batch_", value).map(Self)
    }
}
impl FileId {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        parse_prefixed_uuid("FileId", "file_", value).map(Self)
    }
}
impl SessionId {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        parse_prefixed_uuid("SessionId", "session_", value).map(Self)
    }
}

string_id! {
    /// Idempotency-Key of a client or internal operation identifier: 1–128 bytes of printable ASCII.
    OperationId
}

impl OperationId {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("operation_id", value, 128)?;
        check_ascii("operation_id", value, |b| (0x20..0x7f).contains(&b))?;
        Ok(Self(value.to_owned()))
    }

    /// Internal operation identifier for worker commits and publications.
    pub(crate) fn new_internal() -> Self {
        Self(format!("op_{}", random_suffix()))
    }
}

string_id! {
    /// Effective model fingerprint: 1–128 bytes of `[A-Za-z0-9._:-]`, without local paths.
    Fingerprint
}

impl Fingerprint {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("fingerprint", value, 128)?;
        check_ascii("fingerprint", value, |b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))?;
        Ok(Self(value.to_owned()))
    }
}

string_id! {
    /// Model name of the deployment catalog: 1–256 bytes of UTF-8 without control characters.
    ModelName
}

impl ModelName {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("model", value, 256)?;
        if value.chars().any(char::is_control) {
            return Err(InvalidValue::new("model", "contains a control character"));
        }
        Ok(Self(value.to_owned()))
    }
}

string_id! {
    /// Planner contract version recorded in the catalog: 1–64 bytes of `[A-Za-z0-9._:-]`.
    PlannerVersion
}

impl PlannerVersion {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("planner_version", value, 64)?;
        check_ascii("planner_version", value, |b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))?;
        Ok(Self(value.to_owned()))
    }
}

string_id! {
    /// Opaque artifact key from the ArtifactStore (1–512 bytes of printable ASCII without spaces).
    /// Internal only: it is never returned by the HTTP API.
    ArtifactRef
}

impl ArtifactRef {
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        check_len("artifact_ref", value, 512)?;
        check_ascii("artifact_ref", value, |b| (0x21..0x7f).contains(&b))?;
        Ok(Self(value.to_owned()))
    }
}

/// `custom_id` of a record: 1–256 bytes of UTF-8, compared and indexed as exact bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct CustomId(Vec<u8>);

impl CustomId {
    pub(crate) const MAX_BYTES: usize = 256;

    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Result<Self, InvalidValue> {
        if bytes.is_empty() || bytes.len() > Self::MAX_BYTES {
            return Err(InvalidValue::new("custom_id", "must contain 1..256 bytes"));
        }
        if std::str::from_utf8(&bytes).is_err() {
            return Err(InvalidValue::new("custom_id", "must be UTF-8"));
        }
        Ok(Self(bytes))
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub(crate) fn as_str(&self) -> &str {
        // Construction checked UTF-8 and the bytes are never mutated.
        std::str::from_utf8(&self.0).unwrap_or_default()
    }
}

impl fmt::Debug for CustomId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Client data: only the length is shown.
        write!(f, "CustomId({} bytes)", self.0.len())
    }
}

/// Immutable revision of the model catalog or of the quota policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Revision(pub i64);

/// Database engine behind a [`super::Store`]; also the bounded metric label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackendKind {
    Sqlite,
    Postgresql,
    Mssql,
    Mongodb,
}

impl BackendKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Sqlite => "sqlite",
            Self::Postgresql => "postgresql",
            Self::Mssql => "mssql",
            Self::Mongodb => "mongodb",
        }
    }
}

/// Owner-scoped address of a batch; workers obtain it only from a lease or claim.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BatchRef {
    pub owner: OwnerId,
    pub id: BatchId,
}

// ---------------------------------------------------------------------------------------------------------------
// Policy and catalog (installed by administrative operations, read-only at runtime)
// ---------------------------------------------------------------------------------------------------------------

/// Finite quota limits of one owner, or of the whole installation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuotaLimits {
    pub max_bytes: u64,
    pub max_active_batches: u64,
    pub max_retained_batches: u64,
    pub max_files: u64,
    pub max_open_upload_sessions: u64,
}

/// Shared quota policy of a deployment, installed by `basal batches migrate`.
///
/// The constants that size reservations live here, not in per-host configuration, so every host computes the same
/// reservations. Installation requires Σ owner limits ≤ installation limits (no overcommit in V1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QuotaPolicySpec {
    /// Charge added to every file and reserved for each output/error file slot.
    pub object_charge_bytes: u64,
    /// Fixed charge per batch item row, released only when cleanup deletes the row.
    pub item_overhead_bytes: u64,
    /// Upper bound of one uploaded input file; reserved before the first uploaded byte.
    pub max_upload_bytes: u64,
    /// Upper bound of records in one input file; bounds the completion reserve at create.
    pub max_records_per_batch: u64,
    pub installation: QuotaLimits,
    pub owners: Vec<(OwnerId, QuotaLimits)>,
}

/// Active policy as stored in the database.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QuotaPolicy {
    pub revision: Revision,
    pub spec_sha256: Sha256Digest,
    pub spec: QuotaPolicySpec,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CatalogEntry {
    pub model_name: ModelName,
    pub fingerprint: Fingerprint,
    pub planner_version: PlannerVersion,
}

/// Content of a catalog revision: V1 maps each model name to exactly one fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CatalogSpec {
    pub default_model: ModelName,
    pub entries: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Catalog {
    pub revision: Revision,
    pub spec_sha256: Sha256Digest,
    pub spec: CatalogSpec,
}

/// Outcome of an administrative install: a new immutable revision, or the identical latest one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InstallOutcome {
    Installed(Revision),
    Unchanged(Revision),
}

/// Current usage of an owner as kept in `batch_quota`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct QuotaUsage {
    pub used_bytes: u64,
    pub reserved_bytes: u64,
    pub active_batches: u64,
    pub retained_batches: u64,
    pub files: u64,
    pub open_sessions: u64,
    pub version: i64,
}

/// Difference between `batch_quota` and the sum of its owner rows (doctor check).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct QuotaDrift {
    pub owner: OwnerId,
    pub recorded: QuotaUsage,
    pub derived: QuotaUsage,
}

// ---------------------------------------------------------------------------------------------------------------
// Files and upload sessions
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilePurpose {
    /// Uploaded input (`purpose=batch`), the only purpose accepted by upload.
    Batch,
    BatchOutput,
    BatchError,
}

impl FilePurpose {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::BatchOutput => "batch_output",
            Self::BatchError => "batch_error",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FileState {
    Active,
    /// Logically deleted or expired: reads return 404, the object waits for `gc_after`.
    Deleted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileRecord {
    pub owner: OwnerId,
    pub id: FileId,
    pub purpose: FilePurpose,
    /// Client file name, 1–255 bytes, display only (never a path).
    pub filename: String,
    pub bytes: u64,
    pub sha256: Sha256Digest,
    pub created_at: UnixMillis,
    pub expires_at: Option<UnixMillis>,
    pub state: FileState,
    /// Internal artifact key; must not be exposed by the API.
    pub artifact: ArtifactRef,
}

/// Request to reserve quota for an upload before its first byte.
#[derive(Clone, Debug)]
pub(crate) struct NewUpload {
    pub owner: OwnerId,
    pub artifact: ArtifactRef,
    pub filename: String,
    /// Optional client `expires_after` anchored at file creation; default is the file retention.
    pub expires_after: Option<Duration>,
}

/// Open upload session; the reservation covers `policy.max_upload_bytes + object_charge`.
#[derive(Clone, Debug)]
pub(crate) struct UploadSession {
    pub id: SessionId,
    pub max_bytes: u64,
}

/// Bytes actually stored by the artifact writer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoredObject {
    pub bytes: u64,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PageRequest<C> {
    /// Resource id to continue after (exclusive), as in OpenAI `after`.
    pub after: Option<C>,
    /// 1–100.
    pub limit: u32,
}

/// One page in stable `created_at DESC, id DESC` order.
#[derive(Clone, Debug)]
pub(crate) struct Page<T> {
    pub items: Vec<T>,
    pub has_more: bool,
}

// ---------------------------------------------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------------------------------------------

/// Internal batch status; [`BatchStatus::public_name`] gives the OpenAI-like status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BatchStatus {
    Validating,
    InProgress,
    /// Barrier closed by cancel or expiry; remaining items are being materialized in small chunks.
    Draining,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelled,
}

impl BatchStatus {
    pub(crate) fn as_db(self) -> &'static str {
        match self {
            Self::Validating => "validating",
            Self::InProgress => "in_progress",
            Self::Draining => "draining",
            Self::Finalizing => "finalizing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Expired => "expired",
            Self::Cancelled => "cancelled",
        }
    }

    pub(crate) fn from_db(value: &str) -> Option<Self> {
        Some(match value {
            "validating" => Self::Validating,
            "in_progress" => Self::InProgress,
            "draining" => Self::Draining,
            "finalizing" => Self::Finalizing,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelled" => Self::Cancelled,
            _ => return None,
        })
    }

    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Expired | Self::Cancelled)
    }

    /// OpenAI-like status: draining and finalizing toward cancelled read `cancelling`.
    pub(crate) fn public_name(self, target: Option<TerminalStatus>) -> &'static str {
        match (self, target) {
            (Self::Draining | Self::Finalizing, Some(TerminalStatus::Cancelled)) => "cancelling",
            (Self::Draining | Self::Finalizing, _) => "finalizing",
            (other, _) => other.as_db(),
        }
    }
}

/// Close reason decided under batch control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TerminalStatus {
    Completed,
    Failed,
    Expired,
    Cancelled,
}

impl TerminalStatus {
    pub(crate) fn as_db(self) -> &'static str {
        self.status().as_db()
    }

    pub(crate) fn status(self) -> BatchStatus {
        match self {
            Self::Completed => BatchStatus::Completed,
            Self::Failed => BatchStatus::Failed,
            Self::Expired => BatchStatus::Expired,
            Self::Cancelled => BatchStatus::Cancelled,
        }
    }

    pub(crate) fn from_db(value: &str) -> Option<Self> {
        Some(match value {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "expired" => Self::Expired,
            "cancelled" => Self::Cancelled,
            _ => return None,
        })
    }
}

/// Validated, bounded batch metadata: at most 16 pairs, keys ≤ 64 and values ≤ 512 characters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BatchMetadata(Vec<(String, String)>);

impl BatchMetadata {
    pub(crate) const MAX_PAIRS: usize = 16;
    pub(crate) const MAX_KEY_CHARS: usize = 64;
    pub(crate) const MAX_VALUE_CHARS: usize = 512;

    pub(crate) fn new(pairs: Vec<(String, String)>) -> Result<Self, InvalidValue> {
        if pairs.len() > Self::MAX_PAIRS {
            return Err(InvalidValue::new("metadata", "has more than 16 pairs"));
        }
        for (index, (key, value)) in pairs.iter().enumerate() {
            if key.chars().count() > Self::MAX_KEY_CHARS || value.chars().count() > Self::MAX_VALUE_CHARS {
                return Err(InvalidValue::new("metadata", "key or value is too long"));
            }
            if pairs[..index].iter().any(|(other, _)| other == key) {
                return Err(InvalidValue::new("metadata", "has a duplicate key"));
            }
        }
        Ok(Self(pairs))
    }

    pub(crate) fn pairs(&self) -> &[(String, String)] {
        &self.0
    }
}

/// Specification of a new batch, already validated against the HTTP contract.
#[derive(Clone, Debug)]
pub(crate) struct NewBatch {
    pub input_file_id: FileId,
    /// Always `/v1/basal` in V1; stored to keep the record self-describing.
    pub endpoint: String,
    pub metadata: BatchMetadata,
    /// `output_expires_after.seconds` (anchor: file creation), 3600–2592000; default file retention otherwise.
    pub output_expires_after: Option<Duration>,
}

/// Idempotency-Key with the digest of the canonical request it was used for.
#[derive(Clone, Debug)]
pub(crate) struct Idempotency {
    pub key: OperationId,
    pub request_sha256: Sha256Digest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BatchRecord {
    pub owner: OwnerId,
    pub id: BatchId,
    pub input_file_id: FileId,
    pub endpoint: String,
    pub catalog_revision: Revision,
    pub model_name: Option<ModelName>,
    pub fingerprint: Option<Fingerprint>,
    pub status: BatchStatus,
    pub terminal_target: Option<TerminalStatus>,
    pub created_at: UnixMillis,
    pub expires_at: UnixMillis,
    pub in_progress_at: Option<UnixMillis>,
    pub finalizing_at: Option<UnixMillis>,
    pub cancelling_at: Option<UnixMillis>,
    pub terminal_at: Option<UnixMillis>,
    /// Public counts: `total` stays 0 until seal; expired records count as failed.
    pub counts: RequestCounts,
    pub output_file_id: Option<FileId>,
    pub error_file_id: Option<FileId>,
    pub metadata: BatchMetadata,
    pub output_expires_after: Option<Duration>,
    pub control_epoch: i64,
    /// Validation issues stored for `errors.data` (see [`Store::batch_errors`](super::Store::batch_errors)).
    pub error_count: u64,
    pub errors_truncated: bool,
}

/// One entry of `errors.data`; message ≤ 256 bytes, code ≤ 64 bytes, at most 1000 per batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BatchIssue {
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    pub line: Option<u64>,
}

impl BatchIssue {
    pub(crate) const MAX_CODE_BYTES: usize = 64;
    pub(crate) const MAX_MESSAGE_BYTES: usize = 256;
    pub(crate) const MAX_PER_BATCH: usize = 1000;
}

// ---------------------------------------------------------------------------------------------------------------
// Leases, validation and execution
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaseKind {
    Discovery,
    Validation,
    Execution,
    Finalization,
}

impl LeaseKind {
    pub(crate) fn as_db(self) -> &'static str {
        match self {
            Self::Discovery => "discovery",
            Self::Validation => "validation",
            Self::Execution => "execution",
            Self::Finalization => "finalization",
        }
    }
}

/// Proof of a lease or claim. Every mutation re-checks it under batch control: claim row, worker, generation,
/// unexpired lease and the batch control epoch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ClaimToken {
    pub claim_id: ClaimId,
    pub worker: WorkerId,
    pub kind: LeaseKind,
    pub batch: BatchRef,
    /// Phase generation of the batch (discovery/validation/finalization) or the claim generation (execution).
    pub generation: i64,
    pub control_epoch: i64,
}

/// Lease timing reported by the database clock; the caller derives a monotonic local deadline from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeaseTiming {
    pub server_now: UnixMillis,
    pub lease_until: UnixMillis,
    /// Batch deadline (`expires_at`).
    pub batch_expires_at: UnixMillis,
}

/// Input file of a leased batch.
#[derive(Clone, Debug)]
pub(crate) struct InputFile {
    pub artifact: ArtifactRef,
    pub bytes: u64,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Debug)]
pub(crate) struct DiscoveryLease {
    pub token: ClaimToken,
    pub timing: LeaseTiming,
    pub input: InputFile,
}

/// Result of resolving the first record's model against the pinned catalog revision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TargetResolution {
    /// Batch now has its fingerprint; the discovery lease is finished.
    Resolved(CatalogEntry),
    /// Data error of the client: the caller fails the batch with the line number.
    ModelNotInCatalog,
}

/// Capability of a worker that may validate (and later execute) batches of these fingerprints.
#[derive(Clone, Debug)]
pub(crate) struct WorkerTarget {
    pub fingerprint: Fingerprint,
    pub planner_version: PlannerVersion,
}

#[derive(Clone, Debug)]
pub(crate) struct ValidationLease {
    pub token: ClaimToken,
    pub timing: LeaseTiming,
    pub input: InputFile,
    pub target: CatalogEntry,
    /// Items already appended by an earlier lease; re-appending identical lines is idempotent.
    pub ingested: u64,
}

/// A validated record: byte range of its line in the original JSONL (without the line separator) and
/// estimates of the planner. The store adds `record_reserve` from the custom_id itself.
#[derive(Clone, Debug)]
pub(crate) struct ValidatedRecord {
    pub line_no: u64,
    pub custom_id: CustomId,
    pub offset: u64,
    pub length: u64,
    pub input_sha256: Sha256Digest,
    /// DRR cost estimate (opaque positive units of the planner).
    pub cost: u64,
    /// Upper bound of the finished JSONL line incl. LF; at least the error line, at most 64 MiB.
    pub result_bound: u64,
    /// `retained_bound + exec_bound` of the record for the execution pool `P`.
    pub host_bound: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AppendOutcome {
    Appended {
        ingested: u64,
    },
    /// Binary-exact duplicate of an earlier line: the caller fails the batch.
    DuplicateCustomId {
        line_no: u64,
        first_line_no: u64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct SealRequest {
    pub total: u64,
    /// SHA-256 of the whole input file as read by the validator; must equal the stored file digest.
    pub input_sha256: Sha256Digest,
    pub fingerprint: Fingerprint,
}

/// DRR parameters of a shard (fingerprint), supplied by the planner configuration.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShardParams {
    /// Credit added to every active owner per turn.
    pub quantum: u64,
    /// Highest admissible solo record cost; the store enforces `credit_cap ≥ max_record_cost + quantum`.
    pub max_record_cost: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ClaimRequest {
    pub worker: WorkerId,
    pub target: WorkerTarget,
    pub shard: ShardParams,
    /// Upper bound of items in the microseries (1 when the profile has no composition-independence proof).
    pub max_items: u32,
    /// Upper bound of the summed input bytes read for the claim (lookahead limit).
    pub max_input_bytes: u64,
    /// Free bytes of the execution pool `P`; records whose `host_bound` exceeds it are not claimed.
    pub host_budget_free: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ClaimedItem {
    pub line_no: u64,
    pub custom_id: CustomId,
    pub offset: u64,
    pub length: u64,
    pub input_sha256: Sha256Digest,
    pub requires_solo: bool,
    pub failure_attempts: u32,
    pub cost: u64,
    pub result_bound: u64,
    pub host_bound: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct WorkClaim {
    pub token: ClaimToken,
    pub timing: LeaseTiming,
    pub input: InputFile,
    pub items: Vec<ClaimedItem>,
}

#[derive(Clone, Debug)]
pub(crate) enum ClaimOutcome {
    Claimed(Box<WorkClaim>),
    /// No sealed work for this fingerprint.
    Idle,
    /// The DRR choice needs more of `P` than is free; nothing was claimed or charged.
    NeedsHostBudget {
        required: u64,
    },
    /// Every candidate owner lacks quota for `2 × result_bound`; retry after the backoff.
    QuotaBlocked {
        retry_after: Duration,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Renewal {
    pub timing: LeaseTiming,
    /// False once cancel/expiry closed the barrier or the batch left the leased phase.
    pub accepting: bool,
    pub control_epoch: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaseUpdate<T> {
    Applied(T),
    /// The lease expired, was taken over or the batch control changed.
    Stale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReleaseReason {
    Shutdown,
    Requeue,
    PauseLimit,
    MemoryPressure,
    HostBudget,
    LeaseLost,
}

impl ReleaseReason {
    pub(crate) fn as_db(self) -> &'static str {
        match self {
            Self::Shutdown => "shutdown",
            Self::Requeue => "requeue",
            Self::PauseLimit => "pause_limit",
            Self::MemoryPressure => "memory_pressure",
            Self::HostBudget => "host_budget",
            Self::LeaseLost => "lease_lost",
        }
    }
}

/// Absolute measured gate time of an unfinished record, including failed-group work.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AbandonedCost {
    pub line_no: u64,
    pub actual_cost: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ReleaseReport {
    pub requeued: u32,
    pub estimated: u32,
    pub next_eligible_at: Option<UnixMillis>,
}

/// Where the finished JSONL line of a record goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResultDisposition {
    /// Success line for the output file.
    Output,
    /// Terminal error line for the error file (data error, `result_too_large`, `host_budget_exceeded`, …).
    Error,
    /// Real execution failure: retried while attempts remain, else the line becomes the terminal error.
    Retry,
}

/// Finished JSONL line incl. LF: inline up to [`INLINE_RESULT_MAX`], otherwise an already stored object.
#[derive(Clone, Debug)]
pub(crate) enum ResultPayload {
    Inline(Vec<u8>),
    Stored { session: SessionId, artifact: ArtifactRef, object: StoredObject },
}

/// Largest result line kept inline in the database.
pub(crate) const INLINE_RESULT_MAX: usize = 64 * 1024;

/// Largest finished JSONL line of one record.
pub(crate) const RESULT_MAX: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct ItemResult {
    pub line_no: u64,
    pub disposition: ResultDisposition,
    pub payload: ResultPayload,
    /// Measured GPU cost share (same units as the estimate), used by settlement.
    pub actual_cost: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RejectReason {
    /// Item is not held by this claim generation (lease lost or taken over).
    Stale,
    AlreadyTerminal,
    /// Cancel or expiry closed the barrier.
    ControlClosed,
    DeadlinePassed,
    ExceedsResultBound,
    /// Line not part of this claim.
    NotClaimed,
}

impl RejectReason {
    pub(crate) fn as_db(self) -> &'static str {
        match self {
            Self::Stale => "stale",
            Self::AlreadyTerminal => "terminal",
            Self::ControlClosed => "control",
            Self::DeadlinePassed => "deadline",
            Self::ExceedsResultBound => "bound",
            Self::NotClaimed => "not_claimed",
        }
    }

    pub(crate) fn from_db(value: &str) -> Option<Self> {
        Some(match value {
            "stale" => Self::Stale,
            "terminal" => Self::AlreadyTerminal,
            "control" => Self::ControlClosed,
            "deadline" => Self::DeadlinePassed,
            "bound" => Self::ExceedsResultBound,
            "not_claimed" => Self::NotClaimed,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ItemCommitOutcome {
    Accepted,
    /// `Retry` result with attempts left: the item is queued again without a result.
    Requeued,
    Rejected(RejectReason),
}

#[derive(Clone, Debug)]
pub(crate) struct CommitReport {
    pub items: Vec<(u64, ItemCommitOutcome)>,
    /// True when the operation id was already recorded and the stored outcome is returned.
    pub replayed: bool,
    pub batch_status: BatchStatus,
}

/// Progress of the bounded settlement supervisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) struct AccountingProgress {
    pub settled: u64,
    pub remaining: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DrainProgress {
    pub materialized: u64,
    pub status: BatchStatus,
}

// ---------------------------------------------------------------------------------------------------------------
// Finalization and cleanup
// ---------------------------------------------------------------------------------------------------------------

/// Keys the artifact writer will use for this finalization generation.
#[derive(Clone, Debug)]
pub(crate) struct ExportKeys {
    pub output: ArtifactRef,
    pub error: ArtifactRef,
}

#[derive(Clone, Debug)]
pub(crate) struct FinalizationLease {
    pub token: ClaimToken,
    pub timing: LeaseTiming,
    pub target: TerminalStatus,
    pub keys: ExportKeys,
    pub total: u64,
}

/// Export session of an earlier finalization generation that must be fenced before its reservation moves.
#[derive(Clone, Debug)]
pub(crate) struct PreviousExport {
    pub session: SessionId,
    pub keys: ExportKeys,
    /// Earliest database time at which its writer is assumed stopped (lease end + I/O guard).
    pub writer_stopped_after: UnixMillis,
}

#[derive(Clone, Debug)]
pub(crate) enum FinalizationClaim {
    Ready(FinalizationLease),
    /// The lease is held, but objects of `previous` must be deleted before
    /// [`Store::adopt_export`](super::Store::adopt_export) moves its reservation.
    NeedsFencing {
        token: ClaimToken,
        timing: LeaseTiming,
        previous: PreviousExport,
    },
    Idle,
}

#[derive(Clone, Debug)]
pub(crate) enum ExportPayload {
    Inline(Vec<u8>),
    Stored { artifact: ArtifactRef, object: StoredObject },
}

/// One record for the export writer, in line order.
#[derive(Clone, Debug)]
pub(crate) enum ExportItem {
    Output {
        line_no: u64,
        payload: ExportPayload,
    },
    Error {
        line_no: u64,
        payload: ExportPayload,
    },
    /// Expired without a result: the writer emits [`super::policy::write_expired_line`].
    Expired {
        line_no: u64,
        custom_id: CustomId,
    },
    /// Cancelled without a result: no line is written.
    Cancelled {
        line_no: u64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct PublishedFile {
    pub object: StoredObject,
    pub filename: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct PublishRequest {
    pub output: Option<PublishedFile>,
    pub error: Option<PublishedFile>,
}

/// Bounded cleanup step after publication or an early terminal state.
#[derive(Clone, Debug)]
pub(crate) struct CleanupPage {
    /// Delete these objects, then call `cleanup_commit` with `through_line_no`.
    pub artifacts: Vec<ArtifactRef>,
    pub through_line_no: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct FileGcCandidate {
    pub owner: OwnerId,
    pub file_id: FileId,
    pub artifact: ArtifactRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SessionKind {
    Upload,
    Result,
    Export,
}

impl SessionKind {
    pub(crate) fn from_db(value: &str) -> Option<Self> {
        Some(match value {
            "upload" => Self::Upload,
            "result" => Self::Result,
            "export" => Self::Export,
            _ => return None,
        })
    }
}

/// Abandoned or orphaned session: delete `artifacts`, then call `release_session`.
#[derive(Clone, Debug)]
pub(crate) struct SessionGcCandidate {
    pub session: SessionId,
    pub owner: OwnerId,
    pub artifacts: Vec<ArtifactRef>,
}

// ---------------------------------------------------------------------------------------------------------------
// Settings, schema and diagnostics
// ---------------------------------------------------------------------------------------------------------------

/// Lease, retention and retry settings of the runtime (per process configuration).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StoreSettings {
    pub discovery_lease: Duration,
    pub validation_lease: Duration,
    pub execution_lease: Duration,
    pub finalization_lease: Duration,
    pub upload_lease: Duration,
    /// Bounded time of artifact I/O after a lease ends; protects objects of a cut-off writer from GC and fencing.
    pub io_guard: Duration,
    pub file_retention: Duration,
    pub metadata_retention: Duration,
    pub download_grace: Duration,
    pub max_failure_attempts: u32,
    /// Bounded backoff of an owner without quota in DRR.
    pub quota_blocked_backoff: Duration,
    /// Largest negative credit (debt) of an owner or batch.
    pub max_debt: u64,
}

impl Default for StoreSettings {
    fn default() -> Self {
        const DAY: u64 = 86_400;
        Self {
            discovery_lease: Duration::from_secs(60),
            validation_lease: Duration::from_secs(120),
            execution_lease: Duration::from_secs(60),
            finalization_lease: Duration::from_secs(120),
            upload_lease: Duration::from_secs(3600),
            io_guard: Duration::from_secs(600),
            file_retention: Duration::from_secs(30 * DAY),
            metadata_retention: Duration::from_secs(30 * DAY),
            download_grace: Duration::from_secs(2 * 3600),
            max_failure_attempts: 3,
            quota_blocked_backoff: Duration::from_secs(5),
            max_debt: 1 << 40,
        }
    }
}

/// Completion window of every batch (24 h, public API accepts only `24h`).
pub(crate) const COMPLETION_WINDOW: Duration = Duration::from_secs(86_400);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SchemaState {
    /// No schema objects: a fresh database.
    Empty,
    Current {
        version: i64,
    },
    /// Older schema: `basal batches migrate` is required (SQLite migrates locally on open).
    Outdated {
        version: i64,
        expected: i64,
    },
    /// Newer schema than this binary: refused, never downgraded.
    TooNew {
        version: i64,
        expected: i64,
    },
    /// A migration row is not `applied` or its checksum differs from the embedded script.
    Damaged {
        version: i64,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct DoctorReport {
    pub backend: BackendKind,
    pub server_version: String,
    pub schema: SchemaState,
    pub quota_policy: Option<Revision>,
    pub catalog: Option<Revision>,
    pub batches_by_status: Vec<(BatchStatus, u64)>,
    pub expired_claims: u64,
    pub unaccounted_settlements: u64,
    pub pending_cleanup: u64,
    pub sessions_open: u64,
}

/// Backlog of one fingerprint and status; `fingerprint` is `None` for batches whose model is not yet resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BacklogEntry {
    pub fingerprint: Option<Fingerprint>,
    pub status: BatchStatus,
    pub batches: u64,
    /// Records waiting for a claim (in-progress batches).
    pub queued: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StoreCapabilities {
    pub backend: BackendKind,
    pub transactional_ddl: bool,
    pub row_locks: bool,
    pub skip_locked: bool,
    pub server_clock: bool,
}

string_id! {
    /// Public part of an API key: 16 lowercase hex characters (64 random bits), also its primary key.
    ApiKeyId
}

impl ApiKeyId {
    pub(crate) const LEN: usize = 16;

    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        if value.len() != Self::LEN || !value.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(InvalidValue::new("key_id", "must be 16 lowercase hex characters"));
        }
        Ok(Self(value.to_owned()))
    }
}

/// Route groups an API key may use; a non-empty set.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ApiScopes(u8);

impl ApiScopes {
    /// `/v1/basal`, `/v1/systemone`, `/v1/decisions`, `/v1/models`.
    pub(crate) const INFERENCE: Self = Self(1);
    /// `/v1/files*`, `/v1/batches*`.
    pub(crate) const BATCHES: Self = Self(2);
    /// `/mcp`.
    pub(crate) const MCP: Self = Self(4);
    /// `/metrics` when `auth.protect_metrics` is set.
    pub(crate) const METRICS: Self = Self(8);
    const NAMES: [(Self, &'static str); 4] =
        [(Self::INFERENCE, "inference"), (Self::BATCHES, "batches"), (Self::MCP, "mcp"), (Self::METRICS, "metrics")];
    /// Longest stored form (`inference,batches,mcp,metrics`).
    const MAX_TEXT: usize = 64;

    /// Scopes of a comma list of names (`inference,batches`); empty, unknown or repeated names are refused.
    pub(crate) fn parse(value: &str) -> Result<Self, InvalidValue> {
        let mut bits = 0u8;
        for name in value.split(',') {
            let (scope, _) = Self::NAMES
                .iter()
                .find(|(_, n)| *n == name.trim())
                .ok_or_else(|| InvalidValue::new("scopes", "names must be inference, batches, mcp or metrics"))?;
            if bits & scope.0 != 0 {
                return Err(InvalidValue::new("scopes", "repeats a name"));
            }
            bits |= scope.0;
        }
        Ok(Self(bits))
    }

    pub(crate) fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Canonical comma list in fixed order, the stored form.
    pub(crate) fn to_text(self) -> String {
        let text = Self::NAMES.iter().filter(|(s, _)| self.contains(*s)).map(|(_, n)| *n).collect::<Vec<_>>().join(",");
        debug_assert!(text.len() <= Self::MAX_TEXT);
        text
    }
}

impl fmt::Debug for ApiScopes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ApiScopes({})", self.to_text())
    }
}

/// Longest API user or key name in bytes.
pub(crate) const MAX_API_NAME: usize = 128;

/// Check an operator-chosen user or key name: 1–128 bytes without control characters.
pub(crate) fn check_api_name(name: &str) -> Result<(), InvalidValue> {
    check_len("name", name, MAX_API_NAME)?;
    if name.chars().any(char::is_control) {
        return Err(InvalidValue::new("name", "contains a control character"));
    }
    Ok(())
}

/// A user of the API keys. Its id is the Batch storage namespace ([`OwnerId`]) shared by all its keys.
#[derive(Clone, Debug)]
pub(crate) struct ApiUserRecord {
    pub id: OwnerId,
    pub name: String,
    pub created_at: UnixMillis,
    /// Set while the user is disabled: none of its keys is accepted.
    pub disabled_at: Option<UnixMillis>,
}

/// A key to insert for an existing, enabled user. `token_sha256` is the SHA-256 of the whole bearer value; the token
/// itself never reaches the store.
#[derive(Clone, Debug)]
pub(crate) struct NewApiKey {
    pub id: ApiKeyId,
    pub user: OwnerId,
    pub name: String,
    pub token_sha256: Sha256Digest,
    pub scopes: ApiScopes,
    /// Lifetime from the database clock at insertion; `None`: no expiry.
    pub expires_in: Option<Duration>,
}

/// A stored API key (never the token).
#[derive(Clone, Debug)]
pub(crate) struct ApiKeyRecord {
    pub id: ApiKeyId,
    pub user: OwnerId,
    pub name: String,
    pub token_sha256: Sha256Digest,
    pub scopes: ApiScopes,
    pub created_at: UnixMillis,
    pub expires_at: Option<UnixMillis>,
    pub revoked_at: Option<UnixMillis>,
    pub last_used_at: Option<UnixMillis>,
}

impl ApiKeyRecord {
    /// Usable at database time `now`: not revoked and not expired.
    pub(crate) fn active_at(&self, now: UnixMillis) -> bool {
        self.revoked_at.is_none() && self.expires_at.is_none_or(|e| now < e)
    }
}

/// A key read for authentication with the state of its user and the database time of the read.
#[derive(Clone, Debug)]
pub(crate) struct ApiKeyLookup {
    pub record: ApiKeyRecord,
    pub user_disabled: bool,
    pub now: UnixMillis,
}
