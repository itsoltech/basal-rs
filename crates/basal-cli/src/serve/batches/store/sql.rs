//! Minimal transaction interface shared by the SQL adapters.
//!
//! The domain operations in `ops` are written once against [`SqlTx`]; SQLite, PostgreSQL and SQL Server differ only
//! in the [`Dialect`] fragments and in how a transaction begins. This keeps the semantics of all engines identical
//! without a generic ORM. Every statement is plain SQL with `$n` parameters, which SQLite and PostgreSQL accept as
//! is; the SQL Server adapter maps each `$n` token to `@Pn` (and nothing else) with a strict lexer. Every other
//! difference is an explicit fragment here, and the SQLite/PostgreSQL fragments render exactly the original SQL:
//!
//! | Difference | SQLite / PostgreSQL | SQL Server |
//! | --- | --- | --- |
//! | Row bound | `LIMIT n` ([`Dialect::limit`]) | `TOP (n)` ([`Dialect::top`]) |
//! | Row lock | `FOR UPDATE` suffix (PostgreSQL) | `WITH (UPDLOCK, READCOMMITTEDLOCK)` table hint |
//! | Work pick | `FOR UPDATE SKIP LOCKED` suffix | `WITH (UPDLOCK, READPAST, READCOMMITTEDLOCK)` hint |
//! | Absent-key insert | `ON CONFLICT … DO NOTHING` | `INSERT … SELECT … WHERE NOT EXISTS (… UPDLOCK, HOLDLOCK)` |
//! | Bounded ordered update | row-value `IN (SELECT … LIMIT)` | updatable CTE with `TOP … ORDER BY` |
//! | Exact key equality | `=` | `=` plus equal `DATALENGTH` (SQL Server pads trailing spaces/zero bytes) |
//! | Row count | `COUNT(*)` | `COUNT_BIG(*)` |
//! | Clock | `julianday('now')` / `clock_timestamp()` | `SYSUTCDATETIME()` |
//!
//! No statement uses `RETURNING`/`OUTPUT`, `MERGE` or engine-specific upserts beyond the guarded insert above.
//! Integer columns are BIGINT (INTEGER in SQLite) and always travel as `i64`; booleans are 0/1. Aggregates are cast
//! to BIGINT so PostgreSQL does not return NUMERIC and SQL Server does not return INT.

use std::future::Future;

use super::error::StoreError;

/// Owned statement argument. Typed nulls keep PostgreSQL parameter types stable.
#[derive(Clone, Debug)]
pub(super) enum Val {
    I64(Option<i64>),
    Text(Option<String>),
    Blob(Option<Vec<u8>>),
}

impl From<i64> for Val {
    fn from(value: i64) -> Self {
        Self::I64(Some(value))
    }
}

impl From<Option<i64>> for Val {
    fn from(value: Option<i64>) -> Self {
        Self::I64(value)
    }
}

impl From<bool> for Val {
    fn from(value: bool) -> Self {
        Self::I64(Some(i64::from(value)))
    }
}

impl From<&str> for Val {
    fn from(value: &str) -> Self {
        Self::Text(Some(value.to_owned()))
    }
}

impl From<String> for Val {
    fn from(value: String) -> Self {
        Self::Text(Some(value))
    }
}

impl From<&String> for Val {
    fn from(value: &String) -> Self {
        Self::Text(Some(value.clone()))
    }
}

impl From<Option<&str>> for Val {
    fn from(value: Option<&str>) -> Self {
        Self::Text(value.map(str::to_owned))
    }
}

impl From<Option<String>> for Val {
    fn from(value: Option<String>) -> Self {
        Self::Text(value)
    }
}

impl From<&[u8]> for Val {
    fn from(value: &[u8]) -> Self {
        Self::Blob(Some(value.to_vec()))
    }
}

impl From<Vec<u8>> for Val {
    fn from(value: Vec<u8>) -> Self {
        Self::Blob(Some(value))
    }
}

impl From<Option<Vec<u8>>> for Val {
    fn from(value: Option<Vec<u8>>) -> Self {
        Self::Blob(value)
    }
}

/// `args![a, b, …]` builds the argument vector of one statement.
macro_rules! args {
    () => { Vec::<$crate::serve::batches::store::sql::Val>::new() };
    ($($value:expr),+ $(,)?) => { vec![$($crate::serve::batches::store::sql::Val::from($value)),+] };
}
pub(super) use args;

/// Typed column access by position; decode failures are reported as corrupt stored data.
pub(super) trait SqlRow {
    fn i64_at(&self, index: usize) -> Result<i64, StoreError>;
    fn opt_i64_at(&self, index: usize) -> Result<Option<i64>, StoreError>;
    fn text_at(&self, index: usize) -> Result<String, StoreError>;
    fn opt_text_at(&self, index: usize) -> Result<Option<String>, StoreError>;
    fn blob_at(&self, index: usize) -> Result<Vec<u8>, StoreError>;
    fn opt_blob_at(&self, index: usize) -> Result<Option<Vec<u8>>, StoreError>;

    fn u64_at(&self, index: usize) -> Result<u64, StoreError> {
        u64::try_from(self.i64_at(index)?).map_err(|_| StoreError::Corrupt("negative counter"))
    }

    fn bool_at(&self, index: usize) -> Result<bool, StoreError> {
        Ok(self.i64_at(index)? != 0)
    }
}

/// Engine of a [`Dialect`]; selects the shape of fragments that are not plain strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Engine {
    Sqlite,
    Postgresql,
    Mssql,
}

/// SQL fragments that differ between engines.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dialect {
    engine: Engine,
    /// Expression of the current database time in milliseconds since the epoch.
    pub now_ms: &'static str,
    /// Row lock of a selected row (empty for SQLite: `BEGIN IMMEDIATE` already holds the write lock; empty for
    /// SQL Server, which locks with [`Self::lock_hint`]).
    pub for_update: &'static str,
    /// Row lock skipping rows locked by other workers; only used to pick work.
    pub skip_locked: &'static str,
    /// Table hint placed right after the name of a row-locked table (SQL Server); pairs with `for_update`.
    pub lock_hint: &'static str,
    /// Table hint placed right after the name of the table a worker picks from (SQL Server); pairs with
    /// `skip_locked`. `READCOMMITTEDLOCK` is required for `READPAST` when READ_COMMITTED_SNAPSHOT is on.
    pub skip_hint: &'static str,
    /// Count of all rows as BIGINT.
    pub count_all: &'static str,
    /// Count of tables named `$1` in the current schema.
    pub table_exists: &'static str,
    /// Creation of the migration ledger when it does not exist (migrations only).
    pub schema_table: &'static str,
}

pub(super) const SQLITE: Dialect = Dialect {
    engine: Engine::Sqlite,
    now_ms: "CAST(ROUND((julianday('now') - 2440587.5) * 86400000.0) AS INTEGER)",
    for_update: "",
    skip_locked: "",
    lock_hint: "",
    skip_hint: "",
    count_all: "COUNT(*)",
    table_exists: "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = $1",
    schema_table: SCHEMA_TABLE_PORTABLE,
};

pub(super) const POSTGRESQL: Dialect = Dialect {
    engine: Engine::Postgresql,
    now_ms: "CAST(FLOOR(EXTRACT(EPOCH FROM clock_timestamp()) * 1000) AS BIGINT)",
    for_update: " FOR UPDATE",
    skip_locked: " FOR UPDATE SKIP LOCKED",
    lock_hint: "",
    skip_hint: "",
    count_all: "COUNT(*)",
    table_exists: "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = current_schema() \
                   AND table_name = $1",
    schema_table: SCHEMA_TABLE_PORTABLE,
};

/// SQL Server: READ COMMITTED transactions (with or without READ_COMMITTED_SNAPSHOT). Row locks are update locks
/// held to the end of the transaction; `READCOMMITTEDLOCK` makes the read a locking read of the latest committed
/// row also when READ_COMMITTED_SNAPSHOT is on. `ROWLOCK` is not combined with it (same hint group). The clock
/// is `SYSUTCDATETIME()` read after the locks, in whole milliseconds since the epoch.
pub(super) const MSSQL: Dialect = Dialect {
    engine: Engine::Mssql,
    now_ms: "DATEDIFF_BIG(MILLISECOND, CAST('1970-01-01T00:00:00' AS DATETIME2(7)), SYSUTCDATETIME())",
    for_update: "",
    skip_locked: "",
    lock_hint: " WITH (UPDLOCK, READCOMMITTEDLOCK)",
    skip_hint: " WITH (UPDLOCK, READPAST, READCOMMITTEDLOCK)",
    count_all: "COUNT_BIG(*)",
    table_exists: "SELECT COUNT_BIG(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = SCHEMA_NAME() \
                   AND TABLE_NAME = $1 AND TABLE_TYPE = 'BASE TABLE'",
    schema_table: "IF OBJECT_ID(N'batch_schema', N'U') IS NULL CREATE TABLE batch_schema (version BIGINT NOT NULL \
                   PRIMARY KEY, migration_id NVARCHAR(128) COLLATE Latin1_General_100_BIN2 NOT NULL, \
                   checksum NVARCHAR(64) COLLATE Latin1_General_100_BIN2 NOT NULL, \
                   state NVARCHAR(16) COLLATE Latin1_General_100_BIN2 NOT NULL, applied_at BIGINT NOT NULL)",
};

const SCHEMA_TABLE_PORTABLE: &str = "CREATE TABLE IF NOT EXISTS batch_schema (version BIGINT PRIMARY KEY, \
     migration_id TEXT NOT NULL, checksum TEXT NOT NULL, state TEXT NOT NULL, applied_at BIGINT NOT NULL)";

impl Dialect {
    /// Row bound placed right after `SELECT` (or `SELECT DISTINCT`): `TOP (n) ` on SQL Server, empty elsewhere.
    pub(super) fn top(self, n: &str) -> String {
        match self.engine {
            Engine::Mssql => format!("TOP ({n}) "),
            Engine::Sqlite | Engine::Postgresql => String::new(),
        }
    }

    /// Row bound placed after `ORDER BY` (before `for_update`): ` LIMIT n`, empty on SQL Server.
    pub(super) fn limit(self, n: &str) -> String {
        match self.engine {
            Engine::Mssql => String::new(),
            Engine::Sqlite | Engine::Postgresql => format!(" LIMIT {n}"),
        }
    }

    /// Delete at most `limit` rows. Names and predicates are internal SQL constants; only values are bound.
    /// Tuple membership keeps composite keys bounded on SQLite/PostgreSQL; SQL Server has native DELETE TOP.
    pub(super) fn delete_page(self, table: &str, columns: &[&str], predicate: &str, limit: u32) -> String {
        match self.engine {
            Engine::Mssql => format!("DELETE TOP ({limit}) FROM {table} WHERE {predicate}"),
            Engine::Sqlite | Engine::Postgresql => {
                let keys = columns.join(", ");
                format!("DELETE FROM {table} WHERE ({keys}) IN (SELECT {keys} FROM {table} WHERE {predicate} LIMIT {limit})")
            }
        }
    }

    /// Exact equality of a key column with a parameter. SQL Server ignores trailing spaces of strings and trailing
    /// zero bytes of binaries in `=`, so it also compares the stored lengths (the unique keys of those columns
    /// include the same length, see the SQL Server migration).
    pub(super) fn exact_eq(self, column: &str, param: &str) -> String {
        match self.engine {
            Engine::Mssql => format!("({column} = {param} AND DATALENGTH({column}) = DATALENGTH({param}))"),
            Engine::Sqlite | Engine::Postgresql => format!("{column} = {param}"),
        }
    }

    /// Insert one row unless a row with the same key exists. `key` is the number of leading `columns` that form
    /// the unique key. SQL Server takes a key-range update lock on the probed key (`UPDLOCK, HOLDLOCK`) until the
    /// end of the transaction, so concurrent inserts of the same key serialize instead of failing.
    pub(super) fn insert_if_absent(self, table: &str, columns: &[&str], values: &[&str], key: usize) -> String {
        let column_list = columns.join(", ");
        let value_list = values.join(", ");
        match self.engine {
            Engine::Mssql => {
                let probe = columns
                    .iter()
                    .zip(values)
                    .take(key)
                    .map(|(column, value)| format!("{column} = {value}"))
                    .collect::<Vec<_>>()
                    .join(" AND ");
                format!(
                    "INSERT INTO {table} ({column_list}) SELECT {value_list} WHERE NOT EXISTS \
                     (SELECT 1 FROM {table} WITH (UPDLOCK, HOLDLOCK) WHERE {probe})"
                )
            }
            Engine::Sqlite | Engine::Postgresql => {
                let conflict = columns[..key].join(", ");
                format!("INSERT INTO {table} ({column_list}) VALUES ({value_list}) ON CONFLICT ({conflict}) DO NOTHING")
            }
        }
    }

    /// Update at most `limit` rows chosen by `filter` in `order`. `key` lists the primary key columns of `table`.
    pub(super) fn bounded_update(
        self,
        table: &str,
        key: &str,
        set: &str,
        filter: &str,
        order: &str,
        limit: &str,
    ) -> String {
        match self.engine {
            Engine::Mssql => format!(
                "WITH chosen AS (SELECT TOP ({limit}) * FROM {table} WHERE {filter} ORDER BY {order}) \
                 UPDATE chosen SET {set}"
            ),
            Engine::Sqlite | Engine::Postgresql => format!(
                "UPDATE {table} SET {set} WHERE ({key}) IN (SELECT {key} FROM {table} WHERE {filter} \
                 ORDER BY {order} LIMIT {limit})"
            ),
        }
    }
}

/// One open transaction of an adapter. Every domain operation runs inside exactly one.
pub(super) trait SqlTx: Send {
    type Row: SqlRow + Send;

    fn dialect(&self) -> Dialect;

    fn execute(&mut self, sql: &str, args: Vec<Val>) -> impl Future<Output = Result<u64, StoreError>> + Send;

    fn fetch_optional(
        &mut self,
        sql: &str,
        args: Vec<Val>,
    ) -> impl Future<Output = Result<Option<Self::Row>, StoreError>> + Send;

    /// Callers always bound the result with `LIMIT` or a bounded key set.
    fn fetch_all(
        &mut self,
        sql: &str,
        args: Vec<Val>,
    ) -> impl Future<Output = Result<Vec<Self::Row>, StoreError>> + Send;

    /// Run an embedded multi-statement migration script inside the transaction.
    fn execute_script(&mut self, sql: &'static str) -> impl Future<Output = Result<(), StoreError>> + Send;
}

/// Current database time, read after the locks of the operation were taken.
pub(super) async fn db_now<T: SqlTx>(tx: &mut T) -> Result<i64, StoreError> {
    let sql = format!("SELECT {}", tx.dialect().now_ms);
    let row = tx.fetch_optional(&sql, args![]).await?.ok_or(StoreError::Corrupt("clock query returned no row"))?;
    row.i64_at(0)
}

pub(super) fn to_db(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| StoreError::Invalid(super::error::InvalidValue::new("number", "is too large")))
}

/// `$first, $first+1, …` placeholders for a bounded IN list.
pub(super) fn placeholders(first: usize, count: usize) -> String {
    (first..first + count).map(|n| format!("${n}")).collect::<Vec<_>>().join(", ")
}
