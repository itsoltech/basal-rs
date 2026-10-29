//! `batches:` section of `basal serve --config FILE` (Batch API; disabled by default) and its checked form.
//!
//! ```yaml
//! database:                          # root section of the server's database (`crate::database`)
//!   backend: postgresql
//!   connection_string_env: BASAL_DATABASE_URL
//!   migration_connection_string_env: BASAL_DATABASE_MIGRATION_URL
//!   tls: {mode: verify_full}
//! batches:
//!   enabled: true
//!   profile: production              # local (default) | production
//!   artifacts:
//!     backend: s3                    # filesystem | s3
//!     bucket: basal-batches
//!     prefix: prod
//!     region: eu-central-1
//!     credentials: {source: aws_default_chain}
//!   owner: acme                      # shared storage namespace; not caller identity
//!   quota:
//!     revision: 1
//!     installation: {max_bytes: 107374182400, max_active_batches: 100, max_retained_batches: 10000,
//!                    max_files: 10000, max_open_upload_sessions: 20}
//!     owners:
//!       acme: {max_bytes: 53687091200, max_active_batches: 50, max_retained_batches: 5000,
//!              max_files: 5000, max_open_upload_sessions: 10}
//! ```
//!
//! [`BatchConfig`] is the strict YAML shape (unknown fields are refused everywhere); [`BatchConfig::validate`] turns
//! it into [`BatchSettings`] and fails closed. The database, its credentials and migrations are the root `database:`
//! section; [`BatchSettings::database`] picks the one the Batch API uses. The local profile
//! (`basal serve --batches`) only uses SQLite and the filesystem, a shared storage namespace,
//! with finite default quota; without a root `database:` section its SQLite file is `<data_dir>/batches.sqlite3`.
//! The production profile needs a remote root database, explicit artifacts and quota policy. Secrets
//! (S3 keys) come from one environment variable or one file each, are read only when a component connects, and
//! never appear in `Debug` output or error messages.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::store;
use crate::config_support::{absolute, existing_file, invalid, within};
pub(crate) use crate::config_support::{effective_uid, ensure_private_dir, ConfigError, SecretSource};
use crate::database::{DatabaseSection, DatabaseSettings, Target};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;

/// Largest accepted input file (bytes); a configuration may only lower it.
pub(crate) const MAX_FILE_BYTES: u64 = 200_000_000;
/// Most records of one input file; a configuration may only lower it.
pub(crate) const MAX_RECORDS: u32 = 50_000;
/// Largest JSONL line of one record result (`result_too_large` above it).
pub(crate) const MAX_RESULT_BYTES: u64 = 64 * MIB;
/// Most validation errors kept for a batch (the list says when it stops earlier).
pub(crate) const MAX_VALIDATION_ERRORS: u32 = 1000;
/// Most owners of one quota policy.
const MAX_POLICY_OWNERS: usize = 10_000;
/// Upper bound of byte quota values, so sums over all owners cannot overflow.
const MAX_QUOTA_BYTES: u64 = 1 << 50;
const DAY_S: u64 = 86_400;

/// Configuration profile: what the server may default and which backends it accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Profile {
    /// One machine: SQLite and filesystem artifacts under the data directory, with finite default quota.
    #[default]
    Local,
    /// Shared deployment: PostgreSQL, SQL Server or MongoDB over verified TLS, explicit artifacts and quota policy.
    Production,
}

/// `batches:` as written in the YAML file.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BatchConfig {
    #[serde(default)]
    pub(crate) enabled: bool,
    #[serde(default)]
    pub(crate) profile: Profile,
    /// Moved to the root `database:` section (`database.auto_migrate`); refused here with that hint.
    #[serde(default)]
    auto_migrate: Option<serde::de::IgnoredAny>,
    /// Accept new uploads and batches (default true). `false`: they answer 503 with `Retry-After`; reads,
    /// downloads, deletes and cancels stay available. Read at start (no hot reload).
    #[serde(default)]
    accepting: Option<bool>,
    /// Take new discovery, validation and execution claims (default true). `false`: no new claims; expiry, drains,
    /// settlement, finalization, GC, cleanup, retention and lease recovery continue. Read at start.
    #[serde(default)]
    processing: Option<bool>,
    /// Directory of local state (default: `<data dir>/batches`, `paths::data_dir`).
    #[serde(default)]
    pub(crate) data_dir: Option<PathBuf>,
    /// Moved to the root `database:` section; refused here with that hint.
    #[serde(default)]
    storage: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    artifacts: Option<ArtifactConfig>,
    /// Shared storage namespace for all callers (default local); must be listed in quota.owners.
    #[serde(default)]
    owner: Option<String>,
    #[serde(default)]
    quota: Option<QuotaConfig>,
    #[serde(default)]
    limits: LimitsConfig,
    #[serde(default)]
    host: HostConfig,
    #[serde(default)]
    execution: ExecutionConfig,
    #[serde(default)]
    leases: LeaseConfig,
    #[serde(default)]
    retention: RetentionConfig,
}

/// What the configuration is checked against.
#[derive(Clone, Debug)]
pub(crate) struct ValidateContext {
    /// `paths::data_dir()`: the default parent of the local batch directory.
    pub(crate) default_data_dir: PathBuf,
}

/// A checked Batch API configuration.
#[derive(Debug, Clone)]
pub(crate) struct BatchSettings {
    pub(crate) profile: Profile,
    /// Directory of local state: `batches.data_dir`, else `<data dir>/batches` in the local profile.
    pub(crate) data_dir: Option<PathBuf>,
    /// Operator flag: admit new uploads and batches.
    pub(crate) accepting: bool,
    /// Operator flag: take new discovery, validation and execution claims.
    pub(crate) processing: bool,
    pub(crate) artifacts: ArtifactSettings,
    /// Shared storage namespace for every HTTP caller; not authentication.
    pub(crate) owner: store::OwnerId,
    pub(crate) quota: QuotaPolicy,
    pub(crate) limits: Limits,
    pub(crate) host: HostLimits,
    pub(crate) execution: ExecutionLimits,
    pub(crate) leases: Leases,
    pub(crate) retention: Retention,
}

impl BatchSettings {
    /// Lease, retention and retry settings of the store, derived from this configuration.
    pub(crate) fn store_settings(&self) -> store::StoreSettings {
        let ms = std::time::Duration::from_millis;
        let s = std::time::Duration::from_secs;
        store::StoreSettings {
            discovery_lease: ms(self.leases.discovery_ms),
            validation_lease: ms(self.leases.validation_ms),
            execution_lease: ms(self.leases.execution_ms),
            finalization_lease: ms(self.leases.finalization_ms),
            // The upload writer refuses to finish after `upload_max_duration_s`; the session outlives it by the
            // safety margin, so no renewal is needed while a client streams.
            upload_lease: s(self.host.upload_max_duration_s) + ms(self.leases.safety_margin_ms),
            io_guard: s(self.retention.orphan_grace_s),
            file_retention: s(self.retention.file_ttl_s),
            metadata_retention: s(self.retention.metadata_ttl_s),
            download_grace: s(self.retention.download_grace_s),
            max_failure_attempts: self.execution.max_attempts,
            ..store::StoreSettings::default()
        }
    }

    /// The database of the Batch API: the root `database:` section when it is written. Without one, the local
    /// profile keeps the SQLite file of `basal serve --batches`, `<data_dir>/batches.sqlite3` (automatic migration),
    /// so an existing queue stays in use. A disabled section is refused, never replaced by another database; the
    /// local profile uses SQLite and the production profile a remote database.
    pub(crate) fn database(&self, root: &DatabaseSection) -> Result<DatabaseSettings, ConfigError> {
        let settings = match root {
            DatabaseSection::Enabled(settings) => settings.clone(),
            DatabaseSection::Disabled => {
                return Err(invalid(
                    "batches.enabled",
                    "the Batch API needs a database, but `database.enabled` is false; enable the database or \
                     disable batches",
                ))
            }
            DatabaseSection::Absent => match (&self.data_dir, self.profile) {
                (Some(dir), Profile::Local) => DatabaseSettings::sqlite(dir.join("batches.sqlite3")),
                _ => {
                    return Err(invalid(
                        "database",
                        "the production Batch profile needs the root `database:` section (postgresql, mssql or \
                         mongodb)",
                    ))
                }
            },
        };
        match (self.profile, &settings.target) {
            (Profile::Production, Target::Sqlite { .. }) => Err(invalid(
                "database.backend",
                "sqlite is for the local Batch profile; production uses postgresql, mssql or mongodb",
            )),
            (Profile::Local, Target::Remote(_)) => Err(invalid(
                "database.backend",
                "the local Batch profile uses sqlite; remote databases need `batches.profile: production`",
            )),
            _ => Ok(settings),
        }
    }
}

impl BatchConfig {
    /// The same host execution pool used at startup, for host-PLE doctor's aggregate headroom check.
    pub(crate) fn host_exec_pool_bytes(&self) -> Result<u64, ConfigError> {
        if self.enabled {
            Ok(self.host.validate()?.exec_pool_bytes)
        } else {
            Ok(0)
        }
    }
    /// `basal serve --batches [--batch-data-dir DIR]`: enable the API; a directory from the command line replaces
    /// the one of the file.
    pub(crate) fn apply_cli(&mut self, enable: bool, data_dir: Option<PathBuf>) {
        self.enabled |= enable;
        if let Some(dir) = data_dir {
            // A relative directory from the command line (`./data`) is the one below the working directory; the
            // directory itself is then created or checked as private (0700, this user, not a symlink).
            self.data_dir = Some(match (dir.is_relative(), std::env::current_dir()) {
                (true, Ok(cwd)) => cwd.join(dir),
                _ => dir,
            });
        }
    }

    /// Refuse `batches.storage` and `batches.auto_migrate`, which moved to the root `database:` section, so that
    /// they are never silently ignored next to it (also when the API is disabled).
    pub(crate) fn check_moved_settings(&self) -> Result<(), ConfigError> {
        for (field, set) in
            [("batches.storage", self.storage.is_some()), ("batches.auto_migrate", self.auto_migrate.is_some())]
        {
            if set {
                return Err(invalid(
                    field,
                    "moved to the root `database:` section (backend, path, connection strings, tls, pool, \
                     auto_migrate); see docs/DATABASE.md",
                ));
            }
        }
        Ok(())
    }

    /// Check the configuration; `None` when the API is disabled (nothing else is then checked or opened, except
    /// [`BatchConfig::check_moved_settings`]).
    pub(crate) fn validate(&self, ctx: &ValidateContext) -> Result<Option<BatchSettings>, ConfigError> {
        self.check_moved_settings()?;
        if !self.enabled {
            return Ok(None);
        }
        let profile = self.profile;
        let data_dir = match (&self.data_dir, profile) {
            (Some(dir), _) => Some(absolute("batches.data_dir", dir)?),
            (None, Profile::Local) => Some(absolute("batches.data_dir", &ctx.default_data_dir.join("batches"))?),
            (None, Profile::Production) => None,
        };
        let artifacts = match &self.artifacts {
            Some(a) => a.validate(profile)?,
            None => match (&data_dir, profile) {
                (Some(dir), Profile::Local) => ArtifactSettings::Filesystem { root: dir.join("artifacts") },
                _ => return Err(invalid("batches.artifacts", "required in the production profile")),
            },
        };
        let owner = store::OwnerId::parse(self.owner.as_deref().unwrap_or("local"))
            .map_err(|e| invalid("batches.owner", e.to_string()))?;
        let quota = match &self.quota {
            Some(q) => q.validate()?,
            None if profile == Profile::Local => QuotaPolicy::local_default(owner.as_str()),
            None => return Err(invalid("batches.quota", "required in the production profile")),
        };
        if quota.owner(owner.as_str()).is_none() {
            return Err(invalid("batches.owner", "must have an entry in batches.quota.owners"));
        }
        let limits = self.limits.validate()?;
        let host = self.host.validate()?;
        let execution = self.execution.validate(&limits)?;
        let leases = self.leases.validate()?;
        let remote_timeout_ms = match &artifacts {
            ArtifactSettings::S3(s3) => s3.request_timeout_ms.saturating_mul(u64::from(s3.max_retries) + 1),
            ArtifactSettings::Filesystem { .. } => 0,
        };
        let retention = self.retention.validate(&host, &leases, remote_timeout_ms)?;
        Ok(Some(BatchSettings {
            profile,
            data_dir,
            accepting: self.accepting.unwrap_or(true),
            processing: self.processing.unwrap_or(true),
            artifacts,
            owner,
            quota,
            limits,
            host,
            execution,
            leases,
            retention,
        }))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Artifacts

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ArtifactBackend {
    Filesystem,
    S3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum CredentialSource {
    /// Standard AWS provider chain from the credential variables only (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/
    /// `AWS_SESSION_TOKEN`, web identity, ECS/EKS container credentials), else the instance metadata service.
    AwsDefaultChain,
    /// Access key from one source each.
    Static,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct S3CredentialsConfig {
    source: CredentialSource,
    #[serde(default)]
    access_key_id_env: Option<String>,
    #[serde(default)]
    access_key_id_file: Option<PathBuf>,
    #[serde(default)]
    secret_access_key_env: Option<String>,
    #[serde(default)]
    secret_access_key_file: Option<PathBuf>,
    #[serde(default)]
    session_token_env: Option<String>,
    #[serde(default)]
    session_token_file: Option<PathBuf>,
}

/// `batches.artifacts`: `backend: filesystem` with `root`, or a private S3-compatible bucket.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactConfig {
    backend: ArtifactBackend,
    #[serde(default)]
    root: Option<PathBuf>,
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default)]
    region: Option<String>,
    /// HTTPS endpoint of an S3-compatible service (default: AWS for the region).
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    credentials: Option<S3CredentialsConfig>,
    /// PEM file of a private CA of the endpoint.
    #[serde(default)]
    ca_file: Option<PathBuf>,
    #[serde(default)]
    virtual_hosted_style: Option<bool>,
    /// Single-part writes with `If-None-Match: *` (default true). Without it a key is still never reused, but the
    /// service no longer refuses a second write of the same key.
    #[serde(default)]
    conditional_put: Option<bool>,
    #[serde(default)]
    part_size_bytes: Option<u64>,
    #[serde(default)]
    request_timeout_ms: Option<u64>,
    #[serde(default)]
    connect_timeout_ms: Option<u64>,
    #[serde(default)]
    max_retries: Option<u32>,
}

/// Checked `batches.artifacts`.
#[derive(Clone, Debug)]
pub(crate) enum ArtifactSettings {
    /// Private directory (0700, owned by the server user).
    Filesystem {
        root: PathBuf,
    },
    S3(Box<S3Settings>),
}

/// S3 credentials of the server (the bucket is private; clients download through the server).
#[derive(Clone, Debug)]
pub(crate) enum S3Credentials {
    AwsDefaultChain,
    Static { access_key_id: SecretSource, secret_access_key: SecretSource, session_token: Option<SecretSource> },
}

#[derive(Clone, Debug)]
pub(crate) struct S3Settings {
    pub(crate) bucket: String,
    /// Key prefix without leading/trailing `/` (may be empty).
    pub(crate) prefix: String,
    pub(crate) region: String,
    /// `https://` endpoint; `None` is AWS.
    pub(crate) endpoint: Option<String>,
    pub(crate) credentials: S3Credentials,
    pub(crate) ca_file: Option<PathBuf>,
    pub(crate) virtual_hosted_style: bool,
    pub(crate) conditional_put: bool,
    pub(crate) part_size_bytes: u64,
    pub(crate) request_timeout_ms: u64,
    pub(crate) connect_timeout_ms: u64,
    pub(crate) max_retries: u32,
}

impl ArtifactConfig {
    fn validate(&self, profile: Profile) -> Result<ArtifactSettings, ConfigError> {
        let s3_fields = self.bucket.is_some()
            || self.prefix.is_some()
            || self.region.is_some()
            || self.endpoint.is_some()
            || self.credentials.is_some()
            || self.ca_file.is_some()
            || self.virtual_hosted_style.is_some()
            || self.conditional_put.is_some()
            || self.part_size_bytes.is_some()
            || self.request_timeout_ms.is_some()
            || self.connect_timeout_ms.is_some()
            || self.max_retries.is_some();
        match self.backend {
            ArtifactBackend::Filesystem => {
                if s3_fields {
                    return Err(invalid("batches.artifacts", "filesystem takes only `root`"));
                }
                let root = self.root.as_ref().ok_or_else(|| invalid("batches.artifacts.root", "required"))?;
                Ok(ArtifactSettings::Filesystem { root: absolute("batches.artifacts.root", root)? })
            }
            ArtifactBackend::S3 => {
                if profile == Profile::Local {
                    return Err(invalid(
                        "batches.artifacts.backend",
                        "the local profile uses the filesystem; s3 needs `profile: production`",
                    ));
                }
                if self.root.is_some() {
                    return Err(invalid("batches.artifacts.root", "only for filesystem"));
                }
                let bucket = self.bucket.as_deref().ok_or_else(|| invalid("batches.artifacts.bucket", "required"))?;
                let bucket_ok = (3..=63).contains(&bucket.len())
                    && bucket.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'.' || b == b'-')
                    && bucket.as_bytes()[0].is_ascii_alphanumeric()
                    && bucket.as_bytes()[bucket.len() - 1].is_ascii_alphanumeric();
                if !bucket_ok {
                    return Err(invalid("batches.artifacts.bucket", "not a valid bucket name"));
                }
                let prefix = self.prefix.clone().unwrap_or_default();
                let prefix_ok = prefix.is_empty()
                    || (prefix.len() <= 512
                        && prefix.split('/').all(|seg| {
                            !seg.is_empty()
                                && seg != "."
                                && seg != ".."
                                && seg.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                        }));
                if !prefix_ok {
                    return Err(invalid(
                        "batches.artifacts.prefix",
                        "segments of A-Z, a-z, 0-9, `.`, `_`, `-` separated by `/`, without `.`/`..` or empty segments",
                    ));
                }
                let region = self.region.clone().ok_or_else(|| invalid("batches.artifacts.region", "required"))?;
                if region.is_empty()
                    || region.len() > 64
                    || !region.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
                {
                    return Err(invalid("batches.artifacts.region", "not a valid region name"));
                }
                let endpoint = match &self.endpoint {
                    Some(e) => {
                        let rest = e.strip_prefix("https://").ok_or_else(|| {
                            invalid("batches.artifacts.endpoint", "must be an https:// URL (TLS is required)")
                        })?;
                        let host_ok = !rest.is_empty()
                            && !rest.contains(['@', '?', '#', ' ', '\t', '\n', '\r'])
                            && !rest.starts_with('/');
                        if !host_ok {
                            return Err(invalid(
                                "batches.artifacts.endpoint",
                                "must be https://host[:port][/path] without credentials, query or fragment",
                            ));
                        }
                        Some(e.trim_end_matches('/').to_string())
                    }
                    None => None,
                };
                let creds = self
                    .credentials
                    .as_ref()
                    .ok_or_else(|| invalid("batches.artifacts.credentials", "required: {source: aws_default_chain}"))?;
                let credentials = creds.validate()?;
                let ca_file = match &self.ca_file {
                    Some(p) => Some(existing_file("batches.artifacts.ca_file", p)?),
                    None => None,
                };
                Ok(ArtifactSettings::S3(Box::new(S3Settings {
                    bucket: bucket.to_string(),
                    prefix,
                    region,
                    endpoint,
                    credentials,
                    ca_file,
                    virtual_hosted_style: self.virtual_hosted_style.unwrap_or(false),
                    conditional_put: self.conditional_put.unwrap_or(true),
                    part_size_bytes: within(
                        "batches.artifacts.part_size_bytes",
                        self.part_size_bytes.unwrap_or(8 * MIB),
                        5 * MIB,
                        256 * MIB,
                    )?,
                    request_timeout_ms: within(
                        "batches.artifacts.request_timeout_ms",
                        self.request_timeout_ms.unwrap_or(60_000),
                        1_000,
                        600_000,
                    )?,
                    connect_timeout_ms: within(
                        "batches.artifacts.connect_timeout_ms",
                        self.connect_timeout_ms.unwrap_or(10_000),
                        100,
                        60_000,
                    )?,
                    max_retries: within("batches.artifacts.max_retries", self.max_retries.unwrap_or(3), 0, 10)?,
                })))
            }
        }
    }
}

impl S3CredentialsConfig {
    fn validate(&self) -> Result<S3Credentials, ConfigError> {
        let access = SecretSource::from_pair(
            "batches.artifacts.credentials.access_key_id",
            &self.access_key_id_env,
            &self.access_key_id_file,
        )?;
        let secret = SecretSource::from_pair(
            "batches.artifacts.credentials.secret_access_key",
            &self.secret_access_key_env,
            &self.secret_access_key_file,
        )?;
        let token = SecretSource::from_pair(
            "batches.artifacts.credentials.session_token",
            &self.session_token_env,
            &self.session_token_file,
        )?;
        match self.source {
            CredentialSource::AwsDefaultChain => {
                if access.is_some() || secret.is_some() || token.is_some() {
                    return Err(invalid(
                        "batches.artifacts.credentials",
                        "aws_default_chain takes no key sources; use source: static",
                    ));
                }
                Ok(S3Credentials::AwsDefaultChain)
            }
            CredentialSource::Static => match (access, secret) {
                (Some(access_key_id), Some(secret_access_key)) => {
                    Ok(S3Credentials::Static { access_key_id, secret_access_key, session_token: token })
                }
                _ => Err(invalid(
                    "batches.artifacts.credentials",
                    "static needs access_key_id_(env|file) and secret_access_key_(env|file)",
                )),
            },
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Quota policy

/// Limits of one owner, or (as `installation`) of the whole deployment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OwnerLimits {
    /// `used_bytes + reserved_bytes` may not exceed it.
    pub(crate) max_bytes: u64,
    pub(crate) max_active_batches: u32,
    /// Batches whose metadata is still kept (active ones included).
    pub(crate) max_retained_batches: u32,
    /// Input and result files, including logically deleted ones until GC.
    pub(crate) max_files: u32,
    pub(crate) max_open_upload_sessions: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QuotaConfig {
    revision: u64,
    #[serde(default = "default_object_charge")]
    object_charge_bytes: u64,
    #[serde(default = "default_item_overhead")]
    item_overhead_bytes: u64,
    installation: OwnerLimits,
    owners: BTreeMap<String, OwnerLimits>,
}

fn default_object_charge() -> u64 {
    4 * KIB
}

fn default_item_overhead() -> u64 {
    512
}

/// The shared, versioned quota policy. `basal batches install` installs it in the database; every host checks that
/// the database holds the same revision and digest before it admits uploads, batches or claims. Owners' sums never
/// exceed the installation limits (no overcommit), so there is no shared hot row for the installation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct QuotaPolicy {
    pub(crate) revision: u64,
    /// Bytes charged to every file and batch on top of its content, so empty uploads use finite quota.
    pub(crate) object_charge_bytes: u64,
    /// Bytes of a batch item row, reserved with its record reserve until cleanup removes the row.
    pub(crate) item_overhead_bytes: u64,
    pub(crate) installation: OwnerLimits,
    pub(crate) owners: BTreeMap<String, OwnerLimits>,
}

impl QuotaPolicy {
    /// Policy of the local profile: the configured namespace with 10 GiB, 10 active batches, 1000 retained batches and files,
    /// 2 open upload sessions; revision 1.
    pub(crate) fn local_default(owner: &str) -> Self {
        let limits = OwnerLimits {
            max_bytes: 10 * GIB,
            max_active_batches: 10,
            max_retained_batches: 1000,
            max_files: 1000,
            max_open_upload_sessions: 2,
        };
        Self {
            revision: 1,
            object_charge_bytes: default_object_charge(),
            item_overhead_bytes: default_item_overhead(),
            installation: limits,
            owners: BTreeMap::from([(owner.to_owned(), limits)]),
        }
    }

    /// Limits of an owner; `None` for an owner outside the policy (it may not use the API).
    pub(crate) fn owner(&self, owner: &str) -> Option<&OwnerLimits> {
        self.owners.get(owner)
    }

    /// The store's form of the policy: these limits plus the per-batch bounds of `limits` (largest upload, most
    /// records), which also size reservations. Hosts compare `store::policy_digest` of this spec with the installed
    /// policy; `revision` is the operator's label, the database assigns its own revision.
    pub(crate) fn to_store_spec(&self, limits: &Limits) -> Result<store::QuotaPolicySpec, ConfigError> {
        let convert = |l: &OwnerLimits| store::QuotaLimits {
            max_bytes: l.max_bytes,
            max_active_batches: u64::from(l.max_active_batches),
            max_retained_batches: u64::from(l.max_retained_batches),
            max_files: u64::from(l.max_files),
            max_open_upload_sessions: u64::from(l.max_open_upload_sessions),
        };
        let owners = self
            .owners
            .iter()
            .map(|(name, l)| {
                store::OwnerId::parse(name)
                    .map(|owner| (owner, convert(l)))
                    .map_err(|_| invalid(format!("batches.quota.owners.{}", name.escape_debug()), "invalid owner id"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(store::QuotaPolicySpec {
            object_charge_bytes: self.object_charge_bytes,
            item_overhead_bytes: self.item_overhead_bytes,
            max_upload_bytes: limits.max_file_bytes,
            max_records_per_batch: u64::from(limits.max_records),
            installation: convert(&self.installation),
            owners,
        })
    }
}

impl QuotaConfig {
    fn validate(&self) -> Result<QuotaPolicy, ConfigError> {
        if self.revision == 0 {
            return Err(invalid("batches.quota.revision", "starts at 1"));
        }
        within("batches.quota.object_charge_bytes", self.object_charge_bytes, 1, MIB)?;
        within("batches.quota.item_overhead_bytes", self.item_overhead_bytes, 1, 64 * KIB)?;
        if self.owners.is_empty() || self.owners.len() > MAX_POLICY_OWNERS {
            return Err(invalid("batches.quota.owners", format!("1..={MAX_POLICY_OWNERS} owners")));
        }
        let check = |field: String, l: &OwnerLimits| -> Result<(), ConfigError> {
            if l.max_bytes < 2 * self.object_charge_bytes || l.max_bytes > MAX_QUOTA_BYTES {
                return Err(invalid(
                    format!("{field}.max_bytes"),
                    format!("must be within {}..={MAX_QUOTA_BYTES}", 2 * self.object_charge_bytes),
                ));
            }
            for (name, v) in [
                ("max_active_batches", l.max_active_batches),
                ("max_retained_batches", l.max_retained_batches),
                ("max_files", l.max_files),
                ("max_open_upload_sessions", l.max_open_upload_sessions),
            ] {
                if v == 0 {
                    return Err(invalid(format!("{field}.{name}"), "must be at least 1"));
                }
            }
            if l.max_active_batches > l.max_retained_batches {
                return Err(invalid(format!("{field}.max_active_batches"), "must not exceed max_retained_batches"));
            }
            if l.max_open_upload_sessions > l.max_files {
                return Err(invalid(format!("{field}.max_open_upload_sessions"), "must not exceed max_files"));
            }
            Ok(())
        };
        check("batches.quota.installation".into(), &self.installation)?;
        let mut sum = [0u128; 5];
        for (name, limits) in &self.owners {
            if store::OwnerId::parse(name).is_err() {
                return Err(invalid(
                    format!("batches.quota.owners.{}", name.escape_debug()),
                    "owner ids are 1-128 bytes of A-Z, a-z, 0-9, `._:@-`",
                ));
            }
            check(format!("batches.quota.owners.{name}"), limits)?;
            for (s, v) in sum.iter_mut().zip(fields(limits)) {
                *s += v;
            }
        }
        let names =
            ["max_bytes", "max_active_batches", "max_retained_batches", "max_files", "max_open_upload_sessions"];
        for ((name, s), cap) in names.iter().zip(sum).zip(fields(&self.installation)) {
            if s > cap {
                return Err(invalid(
                    format!("batches.quota.installation.{name}"),
                    format!("the owners' sum {s} exceeds the installation limit {cap} (no overcommit)"),
                ));
            }
        }
        Ok(QuotaPolicy {
            revision: self.revision,
            object_charge_bytes: self.object_charge_bytes,
            item_overhead_bytes: self.item_overhead_bytes,
            installation: self.installation,
            owners: self.owners.clone(),
        })
    }
}

fn fields(l: &OwnerLimits) -> [u128; 5] {
    [
        u128::from(l.max_bytes),
        u128::from(l.max_active_batches),
        u128::from(l.max_retained_batches),
        u128::from(l.max_files),
        u128::from(l.max_open_upload_sessions),
    ]
}

/// Lowercase hex of bytes.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[usize::from(b >> 4)] as char);
        s.push(DIGITS[usize::from(b & 15)] as char);
    }
    s
}

// ---------------------------------------------------------------------------------------------------------------
// Limits

/// Contract limits of input files and results.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LimitsConfig {
    max_file_bytes: u64,
    max_records: u32,
    max_result_bytes: u64,
    inline_result_bytes: u64,
    max_validation_errors: u32,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_file_bytes: MAX_FILE_BYTES,
            max_records: MAX_RECORDS,
            max_result_bytes: MAX_RESULT_BYTES,
            inline_result_bytes: 64 * KIB,
            max_validation_errors: MAX_VALIDATION_ERRORS,
        }
    }
}

/// Checked contract limits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    pub(crate) max_file_bytes: u64,
    pub(crate) max_records: u32,
    pub(crate) max_result_bytes: u64,
    /// Results up to this size are stored inline in the item; larger ones as artifacts.
    pub(crate) inline_result_bytes: u64,
    pub(crate) max_validation_errors: u32,
}

impl LimitsConfig {
    fn validate(&self) -> Result<Limits, ConfigError> {
        let l = Limits {
            max_file_bytes: within("batches.limits.max_file_bytes", self.max_file_bytes, 1, MAX_FILE_BYTES)?,
            max_records: within("batches.limits.max_records", self.max_records, 1, MAX_RECORDS)?,
            max_result_bytes: within("batches.limits.max_result_bytes", self.max_result_bytes, KIB, MAX_RESULT_BYTES)?,
            inline_result_bytes: within("batches.limits.inline_result_bytes", self.inline_result_bytes, 0, MIB)?,
            max_validation_errors: within(
                "batches.limits.max_validation_errors",
                self.max_validation_errors,
                1,
                MAX_VALIDATION_ERRORS,
            )?,
        };
        if l.inline_result_bytes > l.max_result_bytes {
            return Err(invalid("batches.limits.inline_result_bytes", "must not exceed max_result_bytes"));
        }
        Ok(l)
    }
}

/// Host memory and concurrency of the batch service (separate from the online limits).
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct HostConfig {
    plan_slot_bytes: u64,
    exec_pool_bytes: u64,
    planning_workers: u32,
    upload_concurrency: u32,
    upload_buffer_bytes: u64,
    upload_max_duration_s: u64,
    download_concurrency: u32,
}

impl Default for HostConfig {
    fn default() -> Self {
        Self {
            plan_slot_bytes: 512 * MIB,
            exec_pool_bytes: 512 * MIB,
            planning_workers: 1,
            upload_concurrency: 2,
            upload_buffer_bytes: MIB,
            upload_max_duration_s: 3600,
            download_concurrency: 16,
        }
    }
}

/// Checked host limits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostLimits {
    /// Slot `R` of each planning worker (`PlanBudget` of one record).
    pub(crate) plan_slot_bytes: u64,
    /// Execution pool `P`: plans in lookahead/microbatches, continuations and result buffers.
    pub(crate) exec_pool_bytes: u64,
    /// Dedicated CPU planning threads (validation and execution), not the Tokio blocking pool.
    pub(crate) planning_workers: u32,
    pub(crate) upload_concurrency: u32,
    /// Buffer of one upload or download stream.
    pub(crate) upload_buffer_bytes: u64,
    /// Longest upload; the writer refuses to finish after it, and its session lease covers it.
    pub(crate) upload_max_duration_s: u64,
    pub(crate) download_concurrency: u32,
}

impl HostConfig {
    fn validate(&self) -> Result<HostLimits, ConfigError> {
        Ok(HostLimits {
            plan_slot_bytes: within("batches.host.plan_slot_bytes", self.plan_slot_bytes, 16 * MIB, 1024 * GIB)?,
            exec_pool_bytes: within(
                "batches.host.exec_pool_bytes",
                self.exec_pool_bytes,
                16 * MIB,
                u64::from(u32::MAX),
            )?,
            planning_workers: within("batches.host.planning_workers", self.planning_workers, 1, 64)?,
            upload_concurrency: within("batches.host.upload_concurrency", self.upload_concurrency, 1, 64)?,
            upload_buffer_bytes: within(
                "batches.host.upload_buffer_bytes",
                self.upload_buffer_bytes,
                64 * KIB,
                16 * MIB,
            )?,
            upload_max_duration_s: within("batches.host.upload_max_duration_s", self.upload_max_duration_s, 60, DAY_S)?,
            download_concurrency: within("batches.host.download_concurrency", self.download_concurrency, 1, 1024)?,
        })
    }
}

/// Background execution admission (initial values to be tuned by measurement).
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ExecutionConfig {
    lookahead_records: u32,
    lookahead_bytes: u64,
    microbatch_requests: u32,
    microbatch_tokens: u32,
    commit_items: u32,
    commit_inline_bytes: u64,
    commit_buffer_microbatches: u32,
    max_attempts: u32,
    pause_limit_ms: u64,
    poll_idle_min_ms: u64,
    poll_idle_max_ms: u64,
    cancel_poll_ms: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            lookahead_records: 64,
            lookahead_bytes: 64 * MIB,
            microbatch_requests: 32,
            microbatch_tokens: 2048,
            commit_items: 32,
            commit_inline_bytes: MIB,
            commit_buffer_microbatches: 2,
            max_attempts: 3,
            pause_limit_ms: 2_000,
            poll_idle_min_ms: 100,
            poll_idle_max_ms: 1_000,
            cancel_poll_ms: 1_000,
        }
    }
}

/// Checked execution admission. Multi-request microbatches additionally need a profile with evidence that results
/// do not depend on their composition; otherwise the executor runs records alone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExecutionLimits {
    pub(crate) lookahead_records: u32,
    pub(crate) lookahead_bytes: u64,
    pub(crate) microbatch_requests: u32,
    /// Admission tokens of a text microbatch (further bounded by the model's `max_batch_tokens`).
    pub(crate) microbatch_tokens: u32,
    pub(crate) commit_items: u32,
    pub(crate) commit_inline_bytes: u64,
    pub(crate) commit_buffer_microbatches: u32,
    /// Attempts of a record after real (non-data) errors.
    pub(crate) max_attempts: u32,
    pub(crate) pause_limit_ms: u64,
    pub(crate) poll_idle_min_ms: u64,
    pub(crate) poll_idle_max_ms: u64,
    pub(crate) cancel_poll_ms: u64,
}

impl ExecutionConfig {
    fn validate(&self, limits: &Limits) -> Result<ExecutionLimits, ConfigError> {
        let e = ExecutionLimits {
            lookahead_records: within("batches.execution.lookahead_records", self.lookahead_records, 1, 1024)?,
            lookahead_bytes: within("batches.execution.lookahead_bytes", self.lookahead_bytes, MIB, 16 * GIB)?,
            microbatch_requests: within("batches.execution.microbatch_requests", self.microbatch_requests, 1, 256)?,
            microbatch_tokens: within("batches.execution.microbatch_tokens", self.microbatch_tokens, 1, 1 << 20)?,
            commit_items: within("batches.execution.commit_items", self.commit_items, 1, 256)?,
            commit_inline_bytes: within(
                "batches.execution.commit_inline_bytes",
                self.commit_inline_bytes,
                KIB,
                16 * MIB,
            )?,
            commit_buffer_microbatches: within(
                "batches.execution.commit_buffer_microbatches",
                self.commit_buffer_microbatches,
                1,
                16,
            )?,
            max_attempts: within("batches.execution.max_attempts", self.max_attempts, 1, 10)?,
            pause_limit_ms: within("batches.execution.pause_limit_ms", self.pause_limit_ms, 10, 600_000)?,
            poll_idle_min_ms: within("batches.execution.poll_idle_min_ms", self.poll_idle_min_ms, 10, 60_000)?,
            poll_idle_max_ms: within("batches.execution.poll_idle_max_ms", self.poll_idle_max_ms, 10, 60_000)?,
            cancel_poll_ms: within("batches.execution.cancel_poll_ms", self.cancel_poll_ms, 10, 60_000)?,
        };
        if e.poll_idle_min_ms > e.poll_idle_max_ms {
            return Err(invalid("batches.execution.poll_idle_min_ms", "must not exceed poll_idle_max_ms"));
        }
        if e.microbatch_requests > e.lookahead_records {
            return Err(invalid("batches.execution.microbatch_requests", "must not exceed lookahead_records"));
        }
        if limits.inline_result_bytes > e.commit_inline_bytes {
            return Err(invalid("batches.limits.inline_result_bytes", "must not exceed execution.commit_inline_bytes"));
        }
        Ok(e)
    }
}

/// Lease durations (milliseconds).
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LeaseConfig {
    discovery_ms: u64,
    validation_ms: u64,
    validation_renew_ms: u64,
    execution_ms: u64,
    execution_renew_ms: u64,
    finalization_ms: u64,
    finalization_renew_ms: u64,
    safety_margin_ms: u64,
    shutdown_grace_ms: u64,
}

impl Default for LeaseConfig {
    fn default() -> Self {
        Self {
            discovery_ms: 30_000,
            validation_ms: 120_000,
            validation_renew_ms: 20_000,
            execution_ms: 60_000,
            execution_renew_ms: 10_000,
            finalization_ms: 120_000,
            finalization_renew_ms: 20_000,
            safety_margin_ms: 5_000,
            shutdown_grace_ms: 30_000,
        }
    }
}

/// Checked leases: each lease survives a failed renewal (`2 × renew + safety ≤ lease`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Leases {
    pub(crate) discovery_ms: u64,
    pub(crate) validation_ms: u64,
    pub(crate) validation_renew_ms: u64,
    pub(crate) execution_ms: u64,
    pub(crate) execution_renew_ms: u64,
    pub(crate) finalization_ms: u64,
    pub(crate) finalization_renew_ms: u64,
    /// Subtracted from server deadlines for clock and I/O uncertainty.
    pub(crate) safety_margin_ms: u64,
    /// Budget of database/HTTP cleanup at shutdown; driver timeouts must fit in it.
    pub(crate) shutdown_grace_ms: u64,
}

impl Leases {
    /// Longest lease of any kind.
    pub(crate) fn longest_ms(&self) -> u64 {
        self.discovery_ms.max(self.validation_ms).max(self.execution_ms).max(self.finalization_ms)
    }
}

impl LeaseConfig {
    fn validate(&self) -> Result<Leases, ConfigError> {
        let l = Leases {
            discovery_ms: within("batches.leases.discovery_ms", self.discovery_ms, 1_000, 600_000)?,
            validation_ms: within("batches.leases.validation_ms", self.validation_ms, 1_000, 3_600_000)?,
            validation_renew_ms: within("batches.leases.validation_renew_ms", self.validation_renew_ms, 100, 600_000)?,
            execution_ms: within("batches.leases.execution_ms", self.execution_ms, 1_000, 3_600_000)?,
            execution_renew_ms: within("batches.leases.execution_renew_ms", self.execution_renew_ms, 100, 600_000)?,
            finalization_ms: within("batches.leases.finalization_ms", self.finalization_ms, 1_000, 3_600_000)?,
            finalization_renew_ms: within(
                "batches.leases.finalization_renew_ms",
                self.finalization_renew_ms,
                100,
                600_000,
            )?,
            safety_margin_ms: within("batches.leases.safety_margin_ms", self.safety_margin_ms, 100, 60_000)?,
            shutdown_grace_ms: within("batches.leases.shutdown_grace_ms", self.shutdown_grace_ms, 1_000, 600_000)?,
        };
        for (name, lease, renew) in [
            ("validation", l.validation_ms, l.validation_renew_ms),
            ("execution", l.execution_ms, l.execution_renew_ms),
            ("finalization", l.finalization_ms, l.finalization_renew_ms),
        ] {
            if 2 * renew + l.safety_margin_ms > lease {
                return Err(invalid(
                    format!("batches.leases.{name}_ms"),
                    format!("must be at least 2 × {name}_renew_ms + safety_margin_ms"),
                ));
            }
        }
        if 2 * l.safety_margin_ms > l.discovery_ms {
            return Err(invalid("batches.leases.discovery_ms", "must be at least 2 × safety_margin_ms"));
        }
        Ok(l)
    }
}

/// Retention clocks (seconds).
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RetentionConfig {
    file_ttl_s: u64,
    metadata_ttl_s: u64,
    download_grace_s: u64,
    download_max_duration_s: u64,
    orphan_grace_s: u64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            file_ttl_s: 30 * DAY_S,
            metadata_ttl_s: 30 * DAY_S,
            download_grace_s: 7_200,
            download_max_duration_s: 3_600,
            orphan_grace_s: 7_200,
        }
    }
}

/// Checked retention.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Retention {
    /// Default lifetime of files (inputs; outputs without `output_expires_after`).
    pub(crate) file_ttl_s: u64,
    /// Batch metadata and its Idempotency-Key after `terminal_at` (at least until its last result file expires).
    pub(crate) metadata_ttl_s: u64,
    /// Physical GC of a deleted/expired file starts this long after `deleted_at`.
    pub(crate) download_grace_s: u64,
    /// A download is cut off after this long (less than `download_grace_s`).
    pub(crate) download_max_duration_s: u64,
    /// Objects of ended sessions are removed only this long after the session's lease ended; longer than any
    /// writer may still write (upload duration, leases, remote request time).
    pub(crate) orphan_grace_s: u64,
}

/// `output_expires_after.seconds` and file `expires_after.seconds` bounds.
pub(crate) const EXPIRES_AFTER_MIN_S: u64 = 3_600;
pub(crate) const EXPIRES_AFTER_MAX_S: u64 = 30 * DAY_S;

impl RetentionConfig {
    fn validate(&self, host: &HostLimits, leases: &Leases, remote_timeout_ms: u64) -> Result<Retention, ConfigError> {
        let r = Retention {
            file_ttl_s: within("batches.retention.file_ttl_s", self.file_ttl_s, EXPIRES_AFTER_MIN_S, 365 * DAY_S)?,
            metadata_ttl_s: within("batches.retention.metadata_ttl_s", self.metadata_ttl_s, 3_600, 365 * DAY_S)?,
            download_grace_s: within("batches.retention.download_grace_s", self.download_grace_s, 60, 7 * DAY_S)?,
            download_max_duration_s: within(
                "batches.retention.download_max_duration_s",
                self.download_max_duration_s,
                10,
                7 * DAY_S,
            )?,
            orphan_grace_s: within("batches.retention.orphan_grace_s", self.orphan_grace_s, 60, 7 * DAY_S)?,
        };
        if r.download_max_duration_s >= r.download_grace_s {
            return Err(invalid("batches.retention.download_max_duration_s", "must be less than download_grace_s"));
        }
        let longest_writer_ms = (host.upload_max_duration_s * 1000).max(leases.longest_ms());
        let needed_ms = longest_writer_ms + remote_timeout_ms + leases.safety_margin_ms;
        if r.orphan_grace_s * 1000 < needed_ms {
            return Err(invalid(
                "batches.retention.orphan_grace_s",
                format!(
                    "must cover the longest writer: at least {} s (upload or lease, remote requests, safety margin)",
                    needed_ms.div_ceil(1000)
                ),
            ));
        }
        Ok(r)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Background qualification of a served model

/// Composition of background microseries that measurements qualified for a profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Composition {
    /// One request per forward: results must equal online solo bitwise (T8 matrix).
    #[default]
    Solo,
    /// Several requests per forward: only with evidence that a record's result does not depend on the
    /// composition or order of its microseries.
    Group,
}

/// `batch_profile:` of one served model (root model configuration). Background execution stays disabled until an
/// operator records measured evidence: explicit device capacities and the online reserve per device, plus the
/// qualified composition. Nothing here is guessed from free device memory.
///
/// ```yaml
/// batch_profile:
///   background: true
///   evidence: reports/batches-qualification-2026-11-02/README.md   # report of the measurements
///   device_bytes: 8589934592        # bytes background may use per device of the instance (measured peak + margin)
///   online_reserve_bytes: 17179869184  # bytes kept free for online work per device
///   allocation_granularity_bytes: 2097152  # device allocator rounding of one allocation (measured)
///   cost_ns_per_token: 1200         # measured device-gate ns per packed token (DRR cost coefficient)
///   composition: solo               # solo (default) | group
///   max_group_requests: 8           # group only
/// ```
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackgroundProfileConfig {
    /// Required for a host-PLE qualification; absent preserves existing device qualifications.
    #[serde(default)]
    ple_placement: Option<basal_gpu::PleMode>,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    evidence: Option<String>,
    #[serde(default)]
    device_bytes: Option<u64>,
    #[serde(default)]
    online_reserve_bytes: Option<u64>,
    #[serde(default)]
    allocation_granularity_bytes: Option<u64>,
    #[serde(default)]
    cost_ns_per_token: Option<u64>,
    #[serde(default)]
    composition: Composition,
    #[serde(default)]
    max_group_requests: Option<u32>,
}

/// Checked background qualification of a profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BackgroundQualification {
    /// No background forwards for this profile; the reason is shown in health.
    Disabled(String),
    Qualified {
        /// Evidence reference given by the operator (a report path or identifier).
        evidence: String,
        /// Device bytes background work may hold per device of the instance.
        device_bytes: u64,
        /// Device bytes kept for online work per device.
        online_reserve_bytes: u64,
        /// Allocation granularity of the device allocator: every estimated allocation may round up to it.
        allocation_granularity_bytes: u64,
        /// Measured device-gate time per packed token (ns), the linear coefficient of the DRR cost estimate. The
        /// actual cost charged later is the gate ownership time of the work (`StopToken::device_time`), not
        /// hardware kernel busy time.
        cost_ns_per_token: u64,
        composition: Composition,
        /// Most requests of one microseries (1 for solo).
        max_requests: u32,
    },
}

impl BackgroundQualification {
    pub(crate) fn not_configured() -> Self {
        Self::Disabled("no batch_profile qualification configured".into())
    }
}

impl BackgroundProfileConfig {
    /// Qualification must match placement; host also requires an explicit placement-specific opt-in.
    pub(crate) fn validate(
        &self,
        model: &str,
        ple: basal_gpu::PleMode,
        batching: basal_core::Batching,
    ) -> Result<BackgroundQualification, ConfigError> {
        if self.background && batching != basal_core::Batching::Tree {
            return Err(invalid(format!("models.{model}.batch_profile.background"),
                "SOAM-off background execution is not qualified in this packing-only slice; Tree qualification cannot be reused for Budget"));
        }
        if self.ple_placement.is_some_and(|value| value != ple)
            || (self.background && ple == basal_gpu::PleMode::Host && self.ple_placement.is_none())
        {
            return Err(invalid(format!("models.{model}.batch_profile.ple_placement"),
                "qualification must name the actual PLE placement; host requires explicit `ple_placement: host` and placement-specific measured evidence"));
        }
        let field = |name: &str| format!("models.{model}.batch_profile.{name}");
        if !self.background {
            let configured = self.evidence.is_some()
                || self.device_bytes.is_some()
                || self.online_reserve_bytes.is_some()
                || self.allocation_granularity_bytes.is_some()
                || self.cost_ns_per_token.is_some()
                || self.max_group_requests.is_some()
                || self.composition != Composition::Solo;
            if configured {
                return Err(invalid(field("background"), "qualification fields need `background: true`"));
            }
            return Ok(BackgroundQualification::Disabled("background disabled in batch_profile".into()));
        }
        let evidence = self
            .evidence
            .as_deref()
            .filter(|e| !e.trim().is_empty() && e.len() <= 512 && !e.chars().any(char::is_control))
            .ok_or_else(|| invalid(field("evidence"), "required: the report of the qualification measurements"))?;
        let device_bytes = self
            .device_bytes
            .filter(|&b| b > 0)
            .ok_or_else(|| invalid(field("device_bytes"), "required: measured device bytes of background work"))?;
        let online_reserve_bytes = self
            .online_reserve_bytes
            .filter(|&b| b > 0)
            .ok_or_else(|| invalid(field("online_reserve_bytes"), "required: device bytes kept for online work"))?;
        let allocation_granularity_bytes =
            self.allocation_granularity_bytes.filter(|&b| b > 0 && b <= device_bytes).ok_or_else(|| {
                invalid(
                    field("allocation_granularity_bytes"),
                    "required: measured allocator granularity (≤ device_bytes)",
                )
            })?;
        let cost_ns_per_token =
            self.cost_ns_per_token.filter(|&c| (1..=1_000_000_000).contains(&c)).ok_or_else(|| {
                invalid(field("cost_ns_per_token"), "required: measured device-gate ns per packed token (1..=1e9)")
            })?;
        let max_requests = match (self.composition, self.max_group_requests) {
            (Composition::Solo, None) => 1,
            (Composition::Solo, Some(_)) => {
                return Err(invalid(field("max_group_requests"), "only with composition: group"));
            }
            (Composition::Group, Some(n)) if (2..=256).contains(&n) => n,
            (Composition::Group, _) => {
                return Err(invalid(field("max_group_requests"), "required for group: 2..=256"));
            }
        };
        Ok(BackgroundQualification::Qualified {
            evidence: evidence.to_string(),
            device_bytes,
            online_reserve_bytes,
            allocation_granularity_bytes,
            cost_ns_per_token,
            composition: self.composition,
            max_requests,
        })
    }
}
