//! PostgreSQL adapter: READ COMMITTED transactions with explicit `FOR UPDATE` row locks in the shared lock order,
//! `FOR UPDATE SKIP LOCKED` only to pick work, and `clock_timestamp()` read after the locks. Connections enforce
//! certificate and host name verification unless the server is local; statement, lock and idle-in-transaction
//! timeouts are set per connection. Runtime never creates schema objects: `migrate` is an explicit operation.

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgRow, PgSslMode};
use sqlx::{AssertSqlSafe, Postgres, Row, Transaction};

use super::error::{redact_sqlx, StoreError};
use super::sql::{Dialect, SqlRow, SqlTx, Val, POSTGRESQL};
use super::types::BackendKind;

const BACKEND: BackendKind = BackendKind::Postgresql;

/// Serializes concurrent `basal batches migrate` runs (`pg_advisory_xact_lock` key, ASCII "basalbat").
const MIGRATION_LOCK: i64 = 0x6261_7361_6c62_6174;

/// Connection string held only in memory; never printed.
#[derive(Clone)]
pub(crate) struct ConnectionSecret(String);

impl ConnectionSecret {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }

    /// The secret for a driver's own parser; callers must not log or format it.
    pub(super) fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ConnectionSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ConnectionSecret(<redacted>)")
    }
}

/// TLS requirement; a URI cannot weaken it.
#[derive(Clone, Debug)]
pub(crate) enum PgTls {
    /// Encrypted connection with certificate and host name verification; optional private CA bundle.
    VerifyFull { ca_file: Option<PathBuf> },
    /// Plaintext, accepted only for a loopback host or a Unix socket.
    #[expect(
        dead_code,
        reason = "loopback-only operator qualification mode; production configuration requires verified TLS"
    )]
    LocalPlaintext,
}

#[derive(Clone, Debug)]
pub(crate) struct PostgresOptions {
    pub connection: ConnectionSecret,
    pub tls: PgTls,
    pub max_connections: u32,
    pub acquire_timeout: Duration,
    pub statement_timeout: Duration,
    pub lock_timeout: Duration,
}

impl PostgresOptions {
    pub(crate) fn new(connection: ConnectionSecret, tls: PgTls) -> Self {
        Self {
            connection,
            tls,
            max_connections: 8,
            acquire_timeout: Duration::from_secs(10),
            statement_timeout: Duration::from_secs(15),
            lock_timeout: Duration::from_secs(5),
        }
    }
}

fn db_error(error: sqlx::Error) -> StoreError {
    StoreError::from_db(redact_sqlx(BACKEND, &error))
}

fn is_local_host(host: &str) -> bool {
    host.starts_with('/') || matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]")
}

fn connect_options(options: &PostgresOptions) -> Result<PgConnectOptions, StoreError> {
    // The parse error may quote the URI; it is replaced by a fixed message.
    let parsed = PgConnectOptions::from_str(&options.connection.0)
        .map_err(|_| StoreError::Config("invalid PostgreSQL connection string"))?;
    let parsed = match &options.tls {
        PgTls::VerifyFull { ca_file } => {
            let verified = parsed.ssl_mode(PgSslMode::VerifyFull);
            match ca_file {
                Some(path) => verified.ssl_root_cert(path),
                None => verified,
            }
        }
        PgTls::LocalPlaintext => {
            if !is_local_host(parsed.get_host()) {
                return Err(StoreError::Config("plaintext PostgreSQL is allowed only for a local server"));
            }
            parsed.ssl_mode(PgSslMode::Disable)
        }
    };
    let ms = |d: Duration| d.as_millis().to_string();
    Ok(parsed.application_name("basal-batches").options([
        ("statement_timeout", ms(options.statement_timeout)),
        ("lock_timeout", ms(options.lock_timeout)),
        ("idle_in_transaction_session_timeout", ms(options.statement_timeout.saturating_mul(2))),
    ]))
}

#[derive(Clone, Debug)]
pub(super) struct PgDb {
    pool: sqlx::PgPool,
}

impl PgDb {
    pub(super) async fn connect(options: &PostgresOptions) -> Result<Self, StoreError> {
        let connect = connect_options(options)?;
        let pool = PgPoolOptions::new()
            .max_connections(options.max_connections.max(1))
            .acquire_timeout(options.acquire_timeout)
            .connect_with(connect)
            .await
            .map_err(db_error)?;
        Ok(Self { pool })
    }

    pub(super) async fn begin(&self) -> Result<PgTx, StoreError> {
        self.pool.begin_with("BEGIN ISOLATION LEVEL READ COMMITTED").await.map(PgTx).map_err(db_error)
    }

    /// Begin the migration transaction holding the cluster-wide migration lock.
    pub(super) async fn begin_migration(&self) -> Result<PgTx, StoreError> {
        let mut tx = self.begin().await?;
        tx.execute("SELECT pg_advisory_xact_lock($1)", vec![Val::I64(Some(MIGRATION_LOCK))]).await?;
        Ok(tx)
    }

    pub(super) async fn server_version(&self) -> Result<String, StoreError> {
        let row = sqlx::query("SHOW server_version").fetch_one(&self.pool).await.map_err(db_error)?;
        row.try_get::<String, _>(0).map_err(db_error)
    }

    pub(super) async fn close(&self) {
        self.pool.close().await;
    }
}

pub(super) struct PgTx(Transaction<'static, Postgres>);

impl PgTx {
    pub(super) async fn commit(self) -> Result<(), StoreError> {
        self.0.commit().await.map_err(|error| StoreError::CommitUnknown(redact_sqlx(BACKEND, &error)))
    }

    pub(super) async fn rollback(self) {
        // Dropping an unfinished transaction also rolls it back when the connection is reused.
        let _ = self.0.rollback().await;
    }
}

fn bind<'q>(
    mut query: sqlx::query::Query<'q, Postgres, sqlx::postgres::PgArguments>,
    args: Vec<Val>,
) -> sqlx::query::Query<'q, Postgres, sqlx::postgres::PgArguments> {
    for value in args {
        query = match value {
            Val::I64(v) => query.bind(v),
            Val::Text(v) => query.bind(v),
            Val::Blob(v) => query.bind(v),
        };
    }
    query
}

impl SqlTx for PgTx {
    type Row = PgRow;

    fn dialect(&self) -> Dialect {
        POSTGRESQL
    }

    async fn execute(&mut self, sql: &str, args: Vec<Val>) -> Result<u64, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.execute(&mut *self.0).await.map(|r| r.rows_affected()).map_err(db_error)
    }

    async fn fetch_optional(&mut self, sql: &str, args: Vec<Val>) -> Result<Option<PgRow>, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.fetch_optional(&mut *self.0).await.map_err(db_error)
    }

    async fn fetch_all(&mut self, sql: &str, args: Vec<Val>) -> Result<Vec<PgRow>, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.fetch_all(&mut *self.0).await.map_err(db_error)
    }

    async fn execute_script(&mut self, sql: &'static str) -> Result<(), StoreError> {
        sqlx::raw_sql(sql).execute(&mut *self.0).await.map(|_| ()).map_err(db_error)
    }
}

impl SqlRow for PgRow {
    fn i64_at(&self, index: usize) -> Result<i64, StoreError> {
        self.try_get(index).map_err(db_error)
    }

    fn opt_i64_at(&self, index: usize) -> Result<Option<i64>, StoreError> {
        self.try_get(index).map_err(db_error)
    }

    fn text_at(&self, index: usize) -> Result<String, StoreError> {
        self.try_get(index).map_err(db_error)
    }

    fn opt_text_at(&self, index: usize) -> Result<Option<String>, StoreError> {
        self.try_get(index).map_err(db_error)
    }

    fn blob_at(&self, index: usize) -> Result<Vec<u8>, StoreError> {
        self.try_get(index).map_err(db_error)
    }

    fn opt_blob_at(&self, index: usize) -> Result<Option<Vec<u8>>, StoreError> {
        self.try_get(index).map_err(db_error)
    }
}
