//! The server's database: the optional root `database:` section, its connection options and schema migrations.
//!
//! ```yaml
//! database:
//!   enabled: true                    # default when the section is written
//!   backend: postgresql              # sqlite | postgresql | mssql | mongodb
//!   auto_migrate: false              # default: true for sqlite, false for remote databases
//!   connection_string_env: BASAL_DATABASE_URL          # or connection_string_file (exactly one)
//!   migration_connection_string_env: BASAL_DATABASE_MIGRATION_URL
//!   tls: {mode: verify_full, ca_file: /etc/basal/db-ca.pem}
//!   pool: {max_connections: 16}
//! ```
//!
//! Without the section no database is opened. One configuration serves every consumer of the process: the Batch API
//! (`serve::batches`) and the API keys (`serve::auth`). The Batch store's schema (tables, migration ledger and
//! checksums; since version 3 also `api_keys`) is the one schema this binary registers. This module owns the configuration, the connection options built from it, and
//! when and with which account that schema is migrated: [`migrate`] (`basal database migrate`) and
//! [`Database::prepare`] (`auto_migrate` at server start, at most one successful run per process). The schema's
//! migrators and every domain query stay in the consumer's native store adapters. Each consumer opens its own
//! bounded pool from these options; there is no process-wide shared pool.

pub(crate) mod cli;
mod config;

use std::sync::Arc;

pub(crate) use config::{DatabaseConfig, DatabaseSection, DatabaseSettings, RemoteDatabase, RemoteEngine, Target};

use crate::config_support::{ensure_private_dir, ConfigError, SecretString};
use crate::serve::batches::store::{self, SchemaState, Store, StoreError};

/// Opening or migrating the database failed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum DatabaseError {
    /// A connection string could not be read or would weaken TLS.
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("database directory {}: {error}", crate::term::P(path))]
    Directory {
        path: std::path::PathBuf,
        #[source]
        error: std::io::Error,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("database startup task stopped")]
    Task,
}

impl DatabaseError {
    /// Retrying cannot help without a change of the configuration, the files or the schema.
    pub(crate) fn is_permanent(&self) -> bool {
        match self {
            Self::Directory { .. } | Self::Store(StoreError::Schema(_) | StoreError::Config(_)) => true,
            Self::Config(_) | Self::Store(_) | Self::Task => false,
        }
    }
}

/// Which account a connection uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Account {
    /// The runtime account: domain operations only, never DDL.
    Runtime,
    /// The separate migration account: schema migrations and installation of administrative records.
    Migration,
}

/// Connection options of one account of the configured database, with its TLS and pool bounds. SQLite options never
/// create the file; only [`migrate`] does.
#[derive(Debug)]
pub(crate) enum ConnectOptions {
    Sqlite(store::SqliteOptions),
    Postgresql(store::PostgresOptions),
    Mssql(store::MssqlOptions),
    Mongodb(store::MongoOptions),
}

/// Connection options of `account`. Reads the secret (a file read runs on the blocking pool); for SQLite creates or
/// checks the private parent directory of the file (0700, this user, not a symlink).
pub(crate) async fn options(settings: &DatabaseSettings, account: Account) -> Result<ConnectOptions, DatabaseError> {
    let target = settings.target.clone();
    tokio::task::spawn_blocking(move || options_blocking(&target, account)).await.map_err(|_| DatabaseError::Task)?
}

fn options_blocking(target: &Target, account: Account) -> Result<ConnectOptions, DatabaseError> {
    match target {
        Target::Sqlite { path } => {
            if let Some(dir) = path.parent() {
                ensure_private_dir(dir).map_err(|error| DatabaseError::Directory { path: dir.to_path_buf(), error })?;
            }
            let mut options = store::SqliteOptions::local(path.clone());
            options.create = false;
            Ok(ConnectOptions::Sqlite(options))
        }
        Target::Remote(remote) => {
            let secret = match account {
                Account::Runtime => remote.runtime_connection_string()?,
                Account::Migration => remote.migration_connection_string()?,
            };
            Ok(match remote.engine {
                RemoteEngine::Postgresql => ConnectOptions::Postgresql(postgres_options(&secret, remote)),
                RemoteEngine::Mssql => ConnectOptions::Mssql(mssql_options(&secret, remote)),
                RemoteEngine::Mongodb => ConnectOptions::Mongodb(mongo_options(&secret, remote)),
            })
        }
    }
}

/// Migrate the registered schema (the Batch store's) to this binary's version with the migration account: creates a
/// missing SQLite file (0600) or schema, applies pending migrations, keeps a current one. Idempotent; a newer or
/// damaged schema is refused and left unchanged. Independent of `auto_migrate` and of the consumers' settings.
/// PostgreSQL and SQL Server serialize concurrent migrators with a lock; MongoDB resumes an interrupted migration but
/// expects one migrator at a time.
pub(crate) async fn migrate(settings: &DatabaseSettings) -> Result<SchemaState, DatabaseError> {
    let state = match options(settings, Account::Migration).await? {
        ConnectOptions::Sqlite(mut options) => {
            options.create = true;
            let store = Store::open_sqlite(&options, store::StoreSettings::default()).await?;
            let state = store.schema_state().await;
            store.close().await;
            state?
        }
        ConnectOptions::Postgresql(options) => Store::migrate_postgresql(&options).await?,
        ConnectOptions::Mssql(options) => Store::migrate_mssql(&options).await?,
        ConnectOptions::Mongodb(options) => Store::migrate_mongodb(&options).await?,
    };
    Ok(state)
}

/// The database of one server process, shared by its consumers.
#[derive(Debug)]
pub(crate) struct Database {
    settings: DatabaseSettings,
    /// Schema state after the first successful automatic migration; a failure is not cached, so the next
    /// [`Database::prepare`] tries again.
    migrated: tokio::sync::OnceCell<SchemaState>,
}

impl Database {
    pub(crate) fn new(settings: DatabaseSettings) -> Arc<Self> {
        Arc::new(Self { settings, migrated: tokio::sync::OnceCell::new() })
    }

    pub(crate) fn settings(&self) -> &DatabaseSettings {
        &self.settings
    }

    /// Call before a consumer's first connection. With `auto_migrate`, [`migrate`] runs once per process (concurrent
    /// callers wait for the same run; after a success later calls return at once, a failure is retried by the next
    /// call). Without it nothing is done, and the consumers' runtime connections refuse a missing or outdated schema.
    pub(crate) async fn prepare(&self) -> Result<(), DatabaseError> {
        if !self.settings.auto_migrate {
            return Ok(());
        }
        self.migrated
            .get_or_try_init(|| async {
                let state = migrate(&self.settings).await?;
                crate::note!("database schema ({}): {}", self.settings.engine_label(), describe_schema(&state));
                Ok(state)
            })
            .await
            .map(|_| ())
    }

    /// Connection options of `account` (see [`options`]).
    pub(crate) async fn options(&self, account: Account) -> Result<ConnectOptions, DatabaseError> {
        options(&self.settings, account).await
    }
}

/// Text of a schema state for messages.
pub(crate) fn describe_schema(state: &SchemaState) -> String {
    match state {
        SchemaState::Empty => "empty".into(),
        SchemaState::Current { version } => format!("current (version {version})"),
        SchemaState::Outdated { version, expected } => format!("outdated (version {version}, expected {expected})"),
        SchemaState::TooNew { version, expected } => format!("newer than this binary ({version} > {expected})"),
        SchemaState::Damaged { version } => format!("damaged (version {version})"),
    }
}

/// PostgreSQL options of a connection string and the configured TLS and pool bounds.
fn postgres_options(secret: &SecretString, remote: &RemoteDatabase) -> store::PostgresOptions {
    let mut options = store::PostgresOptions::new(
        store::ConnectionSecret::new(secret.expose().to_string()),
        store::PgTls::VerifyFull { ca_file: remote.tls.ca_file.clone() },
    );
    options.max_connections = remote.pool.max_connections;
    options.acquire_timeout = std::time::Duration::from_millis(remote.pool.acquire_timeout_ms);
    options.statement_timeout = std::time::Duration::from_millis(remote.pool.statement_timeout_ms);
    options.lock_timeout = std::time::Duration::from_millis(remote.pool.lock_timeout_ms);
    options
}

/// MongoDB options of a connection string and the configured TLS and pool bounds (the connection string names
/// the replica set and database).
fn mongo_options(secret: &SecretString, remote: &RemoteDatabase) -> store::MongoOptions {
    let ms = std::time::Duration::from_millis;
    let mut options = store::MongoOptions::new(
        store::ConnectionSecret::new(secret.expose().to_string()),
        store::MongoTls::VerifyFull { ca_file: remote.tls.ca_file.clone() },
    );
    options.max_pool_size = remote.pool.max_connections;
    options.min_pool_size = remote.pool.min_connections;
    options.max_concurrent_operations = remote.pool.max_connections as usize;
    options.server_selection_timeout = ms(remote.pool.acquire_timeout_ms);
    options.connect_timeout = ms(remote.pool.connect_timeout_ms);
    options.max_idle_time = ms(remote.pool.idle_timeout_ms);
    options.operation_timeout = ms(remote.pool.statement_timeout_ms);
    options
}

/// SQL Server options of an ADO.NET connection string and the configured TLS and pool bounds (TDS 7.4 with
/// `Encrypt=Required`; the string cannot weaken TLS).
fn mssql_options(secret: &SecretString, remote: &RemoteDatabase) -> store::MssqlOptions {
    let ms = std::time::Duration::from_millis;
    let mut options = store::MssqlOptions::new(
        store::ConnectionSecret::new(secret.expose().to_string()),
        store::MssqlTls::VerifyFull { ca_file: remote.tls.ca_file.clone() },
    );
    options.max_connections = remote.pool.max_connections;
    options.min_connections = remote.pool.min_connections;
    options.acquire_timeout = ms(remote.pool.acquire_timeout_ms);
    options.connect_timeout = ms(remote.pool.connect_timeout_ms);
    options.idle_timeout = Some(ms(remote.pool.idle_timeout_ms));
    options.max_lifetime = Some(ms(remote.pool.max_lifetime_ms));
    options.statement_timeout = ms(remote.pool.statement_timeout_ms);
    // Validated ≤ statement_timeout_ms by the configuration.
    options.lock_timeout = ms(remote.pool.lock_timeout_ms);
    options
}
