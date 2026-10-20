//! SQLite adapter: one local database file, WAL journal with `synchronous=FULL`, write transactions begin with
//! `BEGIN IMMEDIATE` so every domain operation holds the single writer lock from its first statement, and the clock
//! is the adapter's (`julianday('now')`) read after that lock. Local start creates and migrates the file.

use std::path::PathBuf;
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteRow, SqliteSynchronous};
use sqlx::{AssertSqlSafe, Row, Sqlite, Transaction};

use super::error::{redact_sqlx, StoreError};
use super::sql::{Dialect, SqlRow, SqlTx, Val, SQLITE};
use super::types::BackendKind;

const BACKEND: BackendKind = BackendKind::Sqlite;

/// Local SQLite database.
#[derive(Clone, Debug)]
pub(crate) struct SqliteOptions {
    pub path: PathBuf,
    /// Create the file (mode 0600 on Unix) and apply migrations when missing: the local profile.
    pub create: bool,
    pub busy_timeout: Duration,
    pub max_connections: u32,
}

impl SqliteOptions {
    pub(crate) fn local(path: PathBuf) -> Self {
        Self { path, create: true, busy_timeout: Duration::from_secs(5), max_connections: 4 }
    }
}

fn db_error(error: sqlx::Error) -> StoreError {
    StoreError::from_db(redact_sqlx(BACKEND, &error))
}

#[derive(Clone, Debug)]
pub(super) struct SqliteDb {
    pool: sqlx::SqlitePool,
}

impl SqliteDb {
    pub(super) async fn open(options: &SqliteOptions) -> Result<Self, StoreError> {
        if options.create {
            create_private_file(&options.path).await?;
        }
        let connect = SqliteConnectOptions::new()
            .filename(&options.path)
            .create_if_missing(false)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Full)
            .foreign_keys(true)
            .busy_timeout(options.busy_timeout);
        let pool = SqlitePoolOptions::new()
            .max_connections(options.max_connections.max(1))
            .acquire_timeout(options.busy_timeout.saturating_mul(2))
            .connect_with(connect)
            .await
            .map_err(db_error)?;
        Ok(Self { pool })
    }

    pub(super) async fn begin(&self) -> Result<SqliteTx, StoreError> {
        self.pool.begin_with("BEGIN IMMEDIATE").await.map(SqliteTx).map_err(db_error)
    }

    pub(super) async fn server_version(&self) -> Result<String, StoreError> {
        let row = sqlx::query("SELECT sqlite_version()").fetch_one(&self.pool).await.map_err(db_error)?;
        row.try_get::<String, _>(0).map_err(db_error)
    }

    pub(super) async fn close(&self) {
        self.pool.close().await;
    }
}

/// Create the database file without following an existing path's contents and with owner-only permissions; an
/// existing file is kept as is.
async fn create_private_file(path: &std::path::Path) -> Result<(), StoreError> {
    let mut open = tokio::fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    open.mode(0o600);
    match open.open(path).await {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(_) => Err(StoreError::Config("cannot create the local batch database file")),
    }
}

pub(super) struct SqliteTx(Transaction<'static, Sqlite>);

impl SqliteTx {
    pub(super) async fn commit(self) -> Result<(), StoreError> {
        self.0.commit().await.map_err(|error| StoreError::CommitUnknown(redact_sqlx(BACKEND, &error)))
    }

    pub(super) async fn rollback(self) {
        // A failed rollback leaves the connection to the pool's own cleanup; the transaction is not committed.
        let _ = self.0.rollback().await;
    }
}

fn bind<'q>(
    mut query: sqlx::query::Query<'q, Sqlite, sqlx::sqlite::SqliteArguments>,
    args: Vec<Val>,
) -> sqlx::query::Query<'q, Sqlite, sqlx::sqlite::SqliteArguments> {
    for value in args {
        query = match value {
            Val::I64(v) => query.bind(v),
            Val::Text(v) => query.bind(v),
            Val::Blob(v) => query.bind(v),
        };
    }
    query
}

impl SqlTx for SqliteTx {
    type Row = SqliteRow;

    fn dialect(&self) -> Dialect {
        SQLITE
    }

    async fn execute(&mut self, sql: &str, args: Vec<Val>) -> Result<u64, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.execute(&mut *self.0).await.map(|r| r.rows_affected()).map_err(db_error)
    }

    async fn fetch_optional(&mut self, sql: &str, args: Vec<Val>) -> Result<Option<SqliteRow>, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.fetch_optional(&mut *self.0).await.map_err(db_error)
    }

    async fn fetch_all(&mut self, sql: &str, args: Vec<Val>) -> Result<Vec<SqliteRow>, StoreError> {
        let query = bind(sqlx::query(AssertSqlSafe(sql.to_owned())), args);
        query.fetch_all(&mut *self.0).await.map_err(db_error)
    }

    async fn execute_script(&mut self, sql: &'static str) -> Result<(), StoreError> {
        sqlx::raw_sql(sql).execute(&mut *self.0).await.map(|_| ()).map_err(db_error)
    }
}

impl SqlRow for SqliteRow {
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
