//! `database:` section of the server configuration (strict YAML shape) and its checked form.
//!
//! [`DatabaseConfig`] refuses unknown fields; [`DatabaseConfig::section`] turns the optional section into a
//! [`DatabaseSection`] and fails closed. The checks do no network I/O and read no secret: connection strings are read
//! from their one environment variable or file only when a connection is opened, and never appear in `Debug`
//! output or error messages. Every remote connection uses TLS with certificate and host name verification, which a
//! connection string cannot weaken, and a bounded pool.

use std::path::PathBuf;

use serde::Deserialize;

use crate::config_support::{absolute, existing_file, invalid, within, ConfigError, SecretSource, SecretString};

const DAY_MS: u64 = 86_400_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Sqlite,
    Postgresql,
    Mssql,
    Mongodb,
}

/// `database:` as written in the YAML file: `backend: sqlite` with `path`, or a remote database with exactly one
/// connection string source, an optional separate migration account, `tls` and `pool`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DatabaseConfig {
    /// Use the database (default true: writing the section is the opt-in). `false` opens nothing, and a consumer
    /// that needs a database (the Batch API) refuses to start instead of choosing another one.
    #[serde(default = "enabled_by_default")]
    enabled: bool,
    #[serde(default)]
    backend: Option<Backend>,
    /// Migrate the schema when the server starts (default: true for sqlite, false for remote databases). Remote
    /// migrations need the separate migration account; runtime credentials never run DDL.
    #[serde(default)]
    auto_migrate: Option<bool>,
    /// SQLite database file.
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    connection_string_env: Option<String>,
    #[serde(default)]
    connection_string_file: Option<PathBuf>,
    /// Credentials of `basal database migrate` and `auto_migrate` (a separate account from the runtime one).
    #[serde(default)]
    migration_connection_string_env: Option<String>,
    #[serde(default)]
    migration_connection_string_file: Option<PathBuf>,
    #[serde(default)]
    tls: Option<TlsConfig>,
    #[serde(default)]
    pool: Option<PoolConfig>,
}

fn enabled_by_default() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TlsMode {
    /// Encrypted connection, certificate chain and host name verified.
    VerifyFull,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TlsConfig {
    mode: TlsMode,
    /// PEM file of a private CA (otherwise the system roots).
    #[serde(default)]
    ca_file: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PoolConfig {
    max_connections: u32,
    min_connections: u32,
    acquire_timeout_ms: u64,
    connect_timeout_ms: u64,
    idle_timeout_ms: u64,
    max_lifetime_ms: u64,
    statement_timeout_ms: u64,
    lock_timeout_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            min_connections: 0,
            acquire_timeout_ms: 5_000,
            connect_timeout_ms: 10_000,
            idle_timeout_ms: 600_000,
            max_lifetime_ms: 1_800_000,
            statement_timeout_ms: 15_000,
            lock_timeout_ms: 5_000,
        }
    }
}

/// What the root `database:` section selects.
#[derive(Clone, Debug)]
pub(crate) enum DatabaseSection {
    /// No section: the server opens no database (a consumer may define its own local default).
    Absent,
    /// `enabled: false`: no database is opened, and consumers that need one refuse to start.
    Disabled,
    Enabled(DatabaseSettings),
}

/// A checked database configuration.
#[derive(Clone, Debug)]
pub(crate) struct DatabaseSettings {
    pub(crate) target: Target,
    /// Migrate the schema before the first consumer uses the database.
    pub(crate) auto_migrate: bool,
}

/// Where the database is.
#[derive(Clone, Debug)]
pub(crate) enum Target {
    /// Local SQLite database file (bundled SQLite, WAL, FULL synchronous: the store adapter's choice).
    Sqlite {
        path: PathBuf,
    },
    Remote(RemoteDatabase),
}

/// Remote database engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RemoteEngine {
    Postgresql,
    Mssql,
    Mongodb,
}

impl RemoteEngine {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Postgresql => "postgresql",
            Self::Mssql => "mssql",
            Self::Mongodb => "mongodb",
        }
    }
}

/// A remote database: the connection string sources, verified TLS and a bounded pool.
#[derive(Clone, Debug)]
pub(crate) struct RemoteDatabase {
    pub(crate) engine: RemoteEngine,
    pub(crate) connection: SecretSource,
    pub(crate) migration: Option<SecretSource>,
    pub(crate) tls: TlsSettings,
    pub(crate) pool: PoolSettings,
}

/// TLS of a remote database: always encrypted with certificate and host name verification.
#[derive(Clone, Debug)]
pub(crate) struct TlsSettings {
    /// PEM file of a private CA; `None` uses the system roots.
    pub(crate) ca_file: Option<PathBuf>,
}

/// Bounds of a connection pool (all finite and non-zero, except `min_connections`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolSettings {
    pub(crate) max_connections: u32,
    pub(crate) min_connections: u32,
    pub(crate) acquire_timeout_ms: u64,
    pub(crate) connect_timeout_ms: u64,
    pub(crate) idle_timeout_ms: u64,
    /// SQL Server pool only.
    pub(crate) max_lifetime_ms: u64,
    pub(crate) statement_timeout_ms: u64,
    pub(crate) lock_timeout_ms: u64,
}

impl DatabaseSettings {
    /// A local SQLite file with the defaults of a written `backend: sqlite` section (automatic migration).
    pub(crate) fn sqlite(path: PathBuf) -> Self {
        Self { target: Target::Sqlite { path }, auto_migrate: true }
    }

    /// Engine label for logs, metrics and health (no addresses).
    pub(crate) fn engine_label(&self) -> &'static str {
        match &self.target {
            Target::Sqlite { .. } => "sqlite",
            Target::Remote(r) => r.engine.as_str(),
        }
    }
}

impl RemoteDatabase {
    /// Read the runtime connection string and refuse one whose options would disable or weaken TLS (the adapter
    /// still sets the TLS options itself). Blocking when the source is a file.
    pub(crate) fn runtime_connection_string(&self) -> Result<SecretString, ConfigError> {
        let secret = self.connection.read("database.connection_string")?;
        check_connection_tls(self.engine, &secret, "database.connection_string")?;
        Ok(secret)
    }

    /// The connection string of the migration account, distinct from the runtime one. Blocking when the source
    /// is a file.
    pub(crate) fn migration_connection_string(&self) -> Result<SecretString, ConfigError> {
        let field = "database.migration_connection_string";
        let source = self.migration.as_ref().ok_or_else(|| {
            invalid(field, "required for migrations and administration (a separate account from the runtime one)")
        })?;
        let secret = source.read(field)?;
        check_connection_tls(self.engine, &secret, field)?;
        Ok(secret)
    }
}

/// Refuse connection string options that disable encryption or certificate/host verification, or that would set
/// a second CA source next to `tls.ca_file`. The message names the option, never the string.
fn check_connection_tls(engine: RemoteEngine, secret: &SecretString, field: &'static str) -> Result<(), ConfigError> {
    for (key, value) in connection_options(secret.expose()) {
        let key = key.replace([' ', '_'], "").to_ascii_lowercase();
        let value = value.trim().trim_matches(|c| c == '\'' || c == '"').to_ascii_lowercase();
        let falsy = matches!(value.as_str(), "false" | "no" | "0" | "off" | "disable" | "disabled" | "optional");
        let truthy = matches!(value.as_str(), "true" | "yes" | "1" | "on");
        let refused = match engine {
            RemoteEngine::Postgresql => match key.as_str() {
                "sslmode" => value != "verify-full",
                "sslrootcert" | "sslcrl" | "sslcrldir" => true,
                _ => false,
            },
            RemoteEngine::Mssql => match key.as_str() {
                "encrypt" => falsy,
                "trustservercertificate" => truthy,
                "trustservercertificateca" | "hostnameincertificate" => true,
                _ => false,
            },
            RemoteEngine::Mongodb => match key.as_str() {
                "tls" | "ssl" => falsy,
                "tlsinsecure" | "tlsallowinvalidcertificates" | "tlsallowinvalidhostnames" => truthy,
                "tlscafile" | "tlsdisableocspendpointcheck" | "tlsdisablecertificaterevocationcheck" => true,
                _ => false,
            },
        };
        if refused {
            return Err(invalid(
                field,
                format!("option `{key}` would weaken or duplicate TLS settings; set them in `database.tls`"),
            ));
        }
    }
    Ok(())
}

/// `key=value` pairs of a URI query, a PostgreSQL key/value string or an ADO.NET string (best effort; only used to
/// find TLS options).
fn connection_options(s: &str) -> impl Iterator<Item = (&str, &str)> {
    let options = match s.find('?') {
        Some(i) if s.contains("://") => &s[i + 1..],
        _ => s,
    };
    // PostgreSQL key/value strings separate pairs by spaces; ADO.NET strings by `;` (values may hold spaces).
    let spaces_separate = !s.contains(';');
    options
        .split(move |c: char| c == '&' || c == ';' || (spaces_separate && c.is_whitespace()))
        .filter_map(|pair| pair.split_once('='))
        .map(|(k, v)| (k.trim(), v))
}

impl DatabaseConfig {
    /// Check the optional root section. A disabled section is not checked further (nothing is opened); an enabled
    /// one reads only the metadata of `tls.ca_file`.
    pub(crate) fn section(config: Option<&Self>) -> Result<DatabaseSection, ConfigError> {
        match config {
            None => Ok(DatabaseSection::Absent),
            Some(c) if !c.enabled => Ok(DatabaseSection::Disabled),
            Some(c) => c.validate().map(DatabaseSection::Enabled),
        }
    }

    fn validate(&self) -> Result<DatabaseSettings, ConfigError> {
        let backend = self.backend.ok_or_else(|| {
            invalid("database.backend", "required: sqlite, postgresql, mssql or mongodb (or set enabled: false)")
        })?;
        let remote_fields = self.connection_string_env.is_some()
            || self.connection_string_file.is_some()
            || self.migration_connection_string_env.is_some()
            || self.migration_connection_string_file.is_some()
            || self.tls.is_some()
            || self.pool.is_some();
        let engine = match backend {
            Backend::Sqlite => {
                if remote_fields {
                    return Err(invalid("database", "sqlite takes only `path` (and `enabled`, `auto_migrate`)"));
                }
                let path = self.path.as_ref().ok_or_else(|| invalid("database.path", "required for sqlite"))?;
                return Ok(DatabaseSettings {
                    target: Target::Sqlite { path: absolute("database.path", path)? },
                    auto_migrate: self.auto_migrate.unwrap_or(true),
                });
            }
            Backend::Postgresql => RemoteEngine::Postgresql,
            Backend::Mssql => RemoteEngine::Mssql,
            Backend::Mongodb => RemoteEngine::Mongodb,
        };
        if self.path.is_some() {
            return Err(invalid("database.path", "only for sqlite"));
        }
        let connection = SecretSource::from_pair(
            "database.connection_string",
            &self.connection_string_env,
            &self.connection_string_file,
        )?
        .ok_or_else(|| invalid("database", "set connection_string_env or connection_string_file (one source)"))?;
        let migration = SecretSource::from_pair(
            "database.migration_connection_string",
            &self.migration_connection_string_env,
            &self.migration_connection_string_file,
        )?;
        if migration.as_ref() == Some(&connection) {
            return Err(invalid(
                "database.migration_connection_string",
                "must be a separate account from the runtime connection string",
            ));
        }
        let auto_migrate = self.auto_migrate.unwrap_or(false);
        if auto_migrate && migration.is_none() {
            return Err(invalid(
                "database.migration_connection_string",
                "required when database.auto_migrate is true; set a separate migration account source",
            ));
        }
        let tls = self.tls.as_ref().ok_or_else(|| invalid("database.tls", "required: {mode: verify_full}"))?;
        let TlsMode::VerifyFull = tls.mode;
        let ca_file = match &tls.ca_file {
            Some(p) => Some(existing_file("database.tls.ca_file", p)?),
            None => None,
        };
        let default_pool = PoolConfig::default();
        let pool = self.pool.as_ref().unwrap_or(&default_pool);
        let pool = PoolSettings {
            max_connections: within("database.pool.max_connections", pool.max_connections, 1, 256)?,
            min_connections: within("database.pool.min_connections", pool.min_connections, 0, 256)?,
            acquire_timeout_ms: within("database.pool.acquire_timeout_ms", pool.acquire_timeout_ms, 100, 60_000)?,
            connect_timeout_ms: within("database.pool.connect_timeout_ms", pool.connect_timeout_ms, 100, 60_000)?,
            idle_timeout_ms: within("database.pool.idle_timeout_ms", pool.idle_timeout_ms, 1_000, DAY_MS)?,
            max_lifetime_ms: within("database.pool.max_lifetime_ms", pool.max_lifetime_ms, 1_000, DAY_MS)?,
            statement_timeout_ms: within(
                "database.pool.statement_timeout_ms",
                pool.statement_timeout_ms,
                100,
                300_000,
            )?,
            lock_timeout_ms: within("database.pool.lock_timeout_ms", pool.lock_timeout_ms, 10, 60_000)?,
        };
        if pool.min_connections > pool.max_connections {
            return Err(invalid("database.pool.min_connections", "must not exceed max_connections"));
        }
        if pool.lock_timeout_ms > pool.statement_timeout_ms {
            return Err(invalid("database.pool.lock_timeout_ms", "must not exceed statement_timeout_ms"));
        }
        Ok(DatabaseSettings {
            target: Target::Remote(RemoteDatabase {
                engine,
                connection,
                migration,
                tls: TlsSettings { ca_file },
                pool,
            }),
            auto_migrate,
        })
    }
}
