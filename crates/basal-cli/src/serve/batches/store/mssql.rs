//! SQL Server adapter over TDS (tiberius) with a bounded bb8 pool.
//!
//! Transactions are READ COMMITTED (`SET TRANSACTION ISOLATION LEVEL READ COMMITTED; BEGIN TRANSACTION`) and run
//! the shared `ops` with the [`MSSQL`] dialect: update-lock table hints in the shared lock order, `READPAST` only to
//! pick work, `SYSUTCDATETIME()` read after the locks. Both READ_COMMITTED_SNAPSHOT settings are supported: every
//! row lock is a locking read (`READCOMMITTEDLOCK`), plain reads see committed data.
//!
//! Connection safety:
//! - A connection is reused only when it is idle: no request in flight and no open transaction. A future dropped
//!   during a request or inside a transaction (cancellation, client timeout), an I/O or protocol error, or any
//!   failure of `COMMIT`/`ROLLBACK` leaves the connection marked, and the pool discards it
//!   ([`ManageConnection::has_broken`]); closing the session makes the server roll the transaction back.
//! - Every session sets `XACT_ABORT ON`, `LOCK_TIMEOUT` and the SET options required by filtered and computed-column
//!   indexes. Each statement is bounded by the statement timeout on the client (the server has no statement
//!   timeout); lock waits end earlier with error 1222.
//! - After a failed statement the transaction is unusable: later statements and `COMMIT` are refused (with
//!   `XACT_ABORT ON` the server may already have rolled back, and a further statement would run outside it).
//! - A failure while committing is [`StoreError::CommitUnknown`]; the operation ledger reconciles it with the same
//!   operation id. Only busy errors (deadlock victim 1205, lock timeout 1222, …) before the commit are retried.
//!
//! TLS: `VerifyFull` always encrypts (TDS 7.4 `Required`, or TDS 8.0 `Strict` when configured) and verifies the
//! certificate chain and the host name with rustls; the connection string cannot weaken this. tiberius 0.13 adds
//! `ca_file` to the system roots (`trust_cert_ca` is additive, there is no CA-only mode), so a server certificate
//! from either the private CA or a public root is accepted; the host name is verified in both cases.
//! `LocalPlaintext` (no TLS at all) is accepted only for a loopback server. Runtime never runs DDL: `migrate`
//! is an explicit operation holding `sp_getapplock`.

use std::fmt;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bb8::{ManageConnection, PooledConnection, RunError};
use tiberius::error::{Error as TdsError, IoErrorKind};
use tiberius::{Client, ColumnData, Config, EncryptionLevel, Query, Row};
use tokio::net::TcpStream;
use tokio_util::compat::{Compat, TokioAsyncWriteCompatExt};

use super::error::{DbError, DbErrorKind, StoreError};
use super::postgresql::ConnectionSecret;
use super::sql::{Dialect, SqlRow, SqlTx, Val, MSSQL};
use super::types::BackendKind;

const BACKEND: BackendKind = BackendKind::Mssql;

/// `sp_getapplock` resource serializing concurrent `basal batches migrate` runs.
const MIGRATION_LOCK: &str = "basal-batches-migrate";

/// Largest number of parameters of one SQL Server statement.
const MAX_PARAMETERS: usize = 2100;

/// TLS requirement; the connection string cannot weaken it.
#[derive(Clone, Debug)]
pub(crate) enum MssqlTls {
    /// Encrypted connection with certificate chain and host name verification. `ca_file` (PEM or DER) is trusted
    /// in addition to the system roots (tiberius 0.13 has no CA-only trust mode).
    VerifyFull { ca_file: Option<PathBuf> },
    /// No TLS, accepted only for a loopback host.
    #[expect(
        dead_code,
        reason = "loopback-only operator qualification mode; production configuration requires verified TLS"
    )]
    LocalPlaintext,
}

/// SQL Server connection: an ADO.NET connection string (server, database, SQL login) plus the TLS mode and
/// bounds. The string must not carry `TrustServerCertificate`, `TrustServerCertificateCA` or
/// `HostNameInCertificate`; `Encrypt` in the string is overridden by `tls`.
#[derive(Clone, Debug)]
pub(crate) struct MssqlOptions {
    pub connection: ConnectionSecret,
    pub tls: MssqlTls,
    /// TDS 8.0 strict encryption (TLS before the pre-login; SQL Server 2022 or later). Only with `VerifyFull`.
    pub strict_encryption: bool,
    pub max_connections: u32,
    pub min_connections: u32,
    pub acquire_timeout: Duration,
    pub connect_timeout: Duration,
    pub idle_timeout: Option<Duration>,
    pub max_lifetime: Option<Duration>,
    /// Client bound of one statement; the connection is discarded when it elapses.
    pub statement_timeout: Duration,
    /// Server `LOCK_TIMEOUT`; keep it below `statement_timeout`.
    pub lock_timeout: Duration,
}

impl MssqlOptions {
    pub(crate) fn new(connection: ConnectionSecret, tls: MssqlTls) -> Self {
        Self {
            connection,
            tls,
            strict_encryption: false,
            max_connections: 8,
            min_connections: 0,
            acquire_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(10),
            idle_timeout: Some(Duration::from_secs(600)),
            max_lifetime: Some(Duration::from_secs(1800)),
            statement_timeout: Duration::from_secs(15),
            lock_timeout: Duration::from_secs(5),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------------------------

fn db_error(kind: DbErrorKind, code: Option<String>) -> DbError {
    DbError { backend: BACKEND, kind, code, constraint: None }
}

fn local_error(kind: DbErrorKind, code: &'static str) -> StoreError {
    StoreError::from_db(db_error(kind, Some(code.to_owned())))
}

/// Server error numbers that matter to the store. Everything else is `Other`.
fn classify_server(number: u32) -> DbErrorKind {
    match number {
        // Deadlock victim, lock timeout, lock resources, snapshot update conflicts, memory grant timeout.
        1205 | 1222 | 1204 | 3960 | 3961 | 8645 => DbErrorKind::Busy,
        // Primary key / unique index violation.
        2627 | 2601 => DbErrorKind::UniqueViolation,
        // CHECK / FOREIGN KEY conflict, NULL into a NOT NULL column.
        547 | 515 => DbErrorKind::ConstraintViolation,
        // Login failed, database unavailable, transport-level errors, Azure SQL transient unavailability.
        18456 | 18452 | 4060 | 233 | 64 | 10053 | 10054 | 10060 | 40197 | 40501 | 40613 | 49918 | 49919 | 49920
        | 10928 | 10929 => DbErrorKind::Unavailable,
        _ => DbErrorKind::Other,
    }
}

/// Map a driver error without its message (it may echo values, host names or the login).
fn redact(error: &TdsError) -> DbError {
    match error {
        TdsError::Server(token) => db_error(classify_server(token.code()), Some(token.code().to_string())),
        TdsError::Io { kind: IoErrorKind::TimedOut, .. } => db_error(DbErrorKind::Timeout, Some("io_timeout".into())),
        TdsError::Io { .. } | TdsError::Tls(_) | TdsError::Protocol(_) | TdsError::Routing { .. } => {
            db_error(DbErrorKind::Unavailable, None)
        }
        TdsError::Conversion(_) | TdsError::Encoding(_) | TdsError::Utf8 | TdsError::Utf16 | TdsError::ParseInt(_) => {
            db_error(DbErrorKind::Decode, None)
        }
        _ => db_error(DbErrorKind::Other, None),
    }
}

/// A server error leaves the TDS stream in sync (the driver reads the response to its end); every other driver
/// error may leave a partly read response or a closed socket.
fn keeps_stream_in_sync(error: &TdsError) -> bool {
    matches!(error, TdsError::Server(_))
}

fn statement_error(error: &TdsError) -> StoreError {
    StoreError::from_db(redact(error))
}

// ---------------------------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------------------------

/// Refuse connection string keys that would bypass or redirect certificate verification. Over-approximates the
/// tiberius parser (any occurrence of the key text is refused, also inside a quoted value), so no spelling it
/// accepts can slip through; such keys belong in `MssqlTls`.
fn refuse_tls_overrides(secret: &str) -> Result<(), StoreError> {
    let lower = secret.to_ascii_lowercase();
    if ["trustservercertificate", "hostnameincertificate", "hostname in certificate"]
        .iter()
        .any(|key| lower.contains(key))
    {
        return Err(StoreError::Config(
            "the SQL Server connection string must not set TrustServerCertificate[CA] or HostNameInCertificate",
        ));
    }
    Ok(())
}

fn is_loopback(config: &Config) -> bool {
    let addr = config.get_addr();
    let Some((host, _port)) = addr.rsplit_once(':') else { return false };
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost") || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn build_config(options: &MssqlOptions) -> Result<Config, StoreError> {
    let secret = options.connection.expose();
    refuse_tls_overrides(secret)?;
    // The parse error may quote the string; it is replaced by a fixed message.
    let mut config =
        Config::from_ado_string(secret).map_err(|_| StoreError::Config("invalid SQL Server connection string"))?;
    config.application_name("basal-batches");
    config.handshake_timeout(Some(options.connect_timeout));
    config.command_timeout(Some(options.statement_timeout));
    match &options.tls {
        MssqlTls::VerifyFull { ca_file } => {
            config.encryption(if options.strict_encryption {
                EncryptionLevel::Strict
            } else {
                EncryptionLevel::Required
            });
            if let Some(path) = ca_file {
                let path = path.to_str().ok_or(StoreError::Config("the SQL Server CA path must be UTF-8"))?;
                config.trust_cert_ca(path);
            }
        }
        MssqlTls::LocalPlaintext => {
            if options.strict_encryption {
                return Err(StoreError::Config("strict encryption needs TLS verification"));
            }
            if !is_loopback(&config) {
                return Err(StoreError::Config("plaintext SQL Server is allowed only for a local server"));
            }
            config.encryption(EncryptionLevel::NotSupported);
        }
    }
    Ok(config)
}

/// Milliseconds for the INT arguments of `SET LOCK_TIMEOUT` and `sp_getapplock` (1 ..= `i32::MAX`).
fn int_millis(duration: Duration) -> i32 {
    i32::try_from(duration.as_millis()).unwrap_or(i32::MAX).max(1)
}

/// Session settings of every connection: READ COMMITTED, abort the transaction on any run-time error, bounded lock
/// waits, row counts for DML, and the SET options required by filtered and computed-column indexes.
fn session_init(lock_timeout: Duration) -> String {
    let lock_ms = int_millis(lock_timeout);
    format!(
        "SET ANSI_NULLS ON; SET ANSI_PADDING ON; SET ANSI_WARNINGS ON; SET ARITHABORT ON; \
         SET CONCAT_NULL_YIELDS_NULL ON; SET QUOTED_IDENTIFIER ON; SET NUMERIC_ROUNDABORT OFF; \
         SET NOCOUNT OFF; SET IMPLICIT_TRANSACTIONS OFF; SET XACT_ABORT ON; SET LOCK_TIMEOUT {lock_ms}; \
         SET TRANSACTION ISOLATION LEVEL READ COMMITTED"
    )
}

// ---------------------------------------------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------------------------------------------

/// One TDS session. `in_flight` and `in_tx` are set before a request or transaction starts and cleared only when it
/// finished cleanly, so a dropped future leaves them set and the pool discards the session.
pub(super) struct TdsConn {
    client: Client<Compat<TcpStream>>,
    in_flight: bool,
    in_tx: bool,
    broken: bool,
}

impl TdsConn {
    fn reusable(&self) -> bool {
        !(self.in_flight || self.in_tx || self.broken)
    }

    fn start(&mut self) -> Result<(), StoreError> {
        if self.broken || self.in_flight {
            return Err(local_error(DbErrorKind::Unavailable, "connection_unusable"));
        }
        self.in_flight = true;
        Ok(())
    }

    fn finish<T>(&mut self, result: Result<Result<T, TdsError>, tokio::time::error::Elapsed>) -> Result<T, StoreError> {
        match result {
            Ok(Ok(value)) => {
                self.in_flight = false;
                Ok(value)
            }
            Ok(Err(error)) => {
                if keeps_stream_in_sync(&error) {
                    self.in_flight = false;
                } else {
                    self.broken = true;
                }
                Err(statement_error(&error))
            }
            Err(_elapsed) => {
                self.broken = true;
                Err(local_error(DbErrorKind::Timeout, "statement_timeout"))
            }
        }
    }

    /// Run a statement without parameters (session setup, transaction control, migration scripts, diagnostics).
    async fn simple(&mut self, sql: &str, limit: Duration) -> Result<Vec<Row>, StoreError> {
        self.start()?;
        let client = &mut self.client;
        let result = tokio::time::timeout(limit, async move {
            let stream = client.simple_query(sql).await?;
            stream.into_first_result().await
        })
        .await;
        self.finish(result)
    }

    async fn execute(&mut self, query: Query<'_>, limit: Duration) -> Result<u64, StoreError> {
        self.start()?;
        let client = &mut self.client;
        let result = tokio::time::timeout(limit, async move { query.execute(client).await.map(|r| r.total()) }).await;
        self.finish(result)
    }

    async fn query(&mut self, query: Query<'_>, limit: Duration) -> Result<Vec<Row>, StoreError> {
        self.start()?;
        let client = &mut self.client;
        let result = tokio::time::timeout(limit, async move {
            let stream = query.query(client).await?;
            stream.into_first_result().await
        })
        .await;
        self.finish(result)
    }
}

/// bb8 manager: opens verified sessions and keeps only idle, healthy ones.
pub(super) struct Manager {
    config: Config,
    plaintext: bool,
    session_init: String,
    connect_timeout: Duration,
    statement_timeout: Duration,
}

impl Manager {
    async fn open(&self) -> Result<TdsConn, StoreError> {
        let unavailable = |code: &'static str| local_error(DbErrorKind::Unavailable, code);
        let mut config = self.config.clone();
        // One routing hop (Azure SQL gateway redirect) is followed; TLS then verifies the redirected host.
        for hop in 0..2 {
            let tcp = tokio::time::timeout(self.connect_timeout, TcpStream::connect(config.get_addr()))
                .await
                .map_err(|_| unavailable("connect_timeout"))?
                .map_err(|_| unavailable("connect"))?;
            tcp.set_nodelay(true).map_err(|_| unavailable("connect"))?;
            let connected =
                tokio::time::timeout(self.connect_timeout, Client::connect(config.clone(), tcp.compat_write()))
                    .await
                    .map_err(|_| unavailable("handshake_timeout"))?;
            match connected {
                Ok(client) => {
                    let mut conn = TdsConn { client, in_flight: false, in_tx: false, broken: false };
                    conn.simple(&self.session_init, self.statement_timeout).await?;
                    return Ok(conn);
                }
                // A plaintext session must never be redirected away from the loopback server.
                Err(TdsError::Routing { host, port }) if hop == 0 && !self.plaintext => {
                    config.host(host);
                    config.port(port);
                }
                Err(error) => return Err(StoreError::Unavailable(redact(&error))),
            }
        }
        Err(unavailable("routing"))
    }
}

impl ManageConnection for Manager {
    type Connection = TdsConn;
    type Error = StoreError;

    async fn connect(&self) -> Result<TdsConn, StoreError> {
        self.open().await
    }

    async fn is_valid(&self, conn: &mut TdsConn) -> Result<(), StoreError> {
        if !conn.reusable() {
            return Err(local_error(DbErrorKind::Unavailable, "connection_unusable"));
        }
        conn.simple("SELECT 1", self.statement_timeout).await.map(|_| ())
    }

    fn has_broken(&self, conn: &mut TdsConn) -> bool {
        !conn.reusable()
    }
}

fn pool_error(error: RunError<StoreError>) -> StoreError {
    match error {
        RunError::User(error) => error,
        RunError::TimedOut => local_error(DbErrorKind::Unavailable, "pool_timeout"),
    }
}

/// Pooled SQL Server store. `close` drops the pool; sessions still lent to transactions close when returned.
#[derive(Clone)]
pub(super) struct MssqlDb {
    pool: Arc<Mutex<Option<bb8::Pool<Manager>>>>,
    statement_timeout: Duration,
    lock_timeout: Duration,
}

impl fmt::Debug for MssqlDb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The pool's manager holds the connection configuration with the login; it is never printed.
        f.write_str("MssqlDb { .. }")
    }
}

impl MssqlDb {
    /// Open one session directly (bb8 reports later connect failures only as an acquire timeout), so
    /// configuration, TLS and login failures surface here with their cause, then build the pool around it.
    pub(super) async fn connect(options: &MssqlOptions) -> Result<Self, StoreError> {
        let config = build_config(options)?;
        let manager = Manager {
            config,
            plaintext: matches!(options.tls, MssqlTls::LocalPlaintext),
            session_init: session_init(options.lock_timeout),
            connect_timeout: options.connect_timeout,
            statement_timeout: options.statement_timeout,
        };
        let first = manager.open().await?;
        let max = options.max_connections.max(1);
        let pool = bb8::Pool::builder()
            .max_size(max)
            .min_idle((options.min_connections > 0).then_some(options.min_connections.min(max)))
            .connection_timeout(options.acquire_timeout)
            .idle_timeout(options.idle_timeout)
            .max_lifetime(options.max_lifetime)
            .retry_connection(false)
            .test_on_check_out(true)
            .build_unchecked(manager);
        pool.add(first).map_err(|_| local_error(DbErrorKind::Unavailable, "pool_add"))?;
        Ok(Self {
            pool: Arc::new(Mutex::new(Some(pool))),
            statement_timeout: options.statement_timeout,
            lock_timeout: options.lock_timeout,
        })
    }

    fn pool(&self) -> Result<bb8::Pool<Manager>, StoreError> {
        // The guard is held only to clone the handle, never across an await.
        let guard = self.pool.lock().map_err(|_| local_error(DbErrorKind::Unavailable, "pool_poisoned"))?;
        guard.clone().ok_or_else(|| local_error(DbErrorKind::Unavailable, "shutdown"))
    }

    async fn checkout(&self) -> Result<PooledConnection<'static, Manager>, StoreError> {
        self.pool()?.get_owned().await.map_err(pool_error)
    }

    pub(super) async fn begin(&self) -> Result<MssqlTx, StoreError> {
        let mut conn = self.checkout().await?;
        // Marked before the request: if BEGIN is interrupted the session is discarded, not reused.
        conn.in_tx = true;
        conn.simple("SET TRANSACTION ISOLATION LEVEL READ COMMITTED; BEGIN TRANSACTION", self.statement_timeout)
            .await?;
        Ok(MssqlTx { conn, statement_timeout: self.statement_timeout, failed: false })
    }

    /// Begin the migration transaction holding the database-wide migration application lock (transaction owner,
    /// released by commit or rollback).
    pub(super) async fn begin_migration(&self) -> Result<MssqlTx, StoreError> {
        let mut tx = self.begin().await?;
        let wait = int_millis(self.lock_timeout);
        let sql = format!(
            "DECLARE @granted INT; EXEC @granted = sp_getapplock @Resource = N'{MIGRATION_LOCK}', \
             @LockMode = N'Exclusive', @LockOwner = N'Transaction', @LockTimeout = {wait}; \
             SELECT CAST(@granted AS BIGINT)"
        );
        let granted = match tx.raw_scalar(&sql).await {
            Ok(value) => value,
            Err(error) => {
                tx.rollback().await;
                return Err(error);
            }
        };
        // 0 granted, 1 granted after waiting; negative values are timeout, cancel, deadlock or error.
        if granted < 0 {
            tx.rollback().await;
            return Err(StoreError::Busy(db_error(DbErrorKind::Busy, Some(format!("applock:{granted}")))));
        }
        Ok(tx)
    }

    pub(super) async fn server_version(&self) -> Result<String, StoreError> {
        let mut conn = self.checkout().await?;
        let rows = conn
            .simple("SELECT CAST(SERVERPROPERTY('ProductVersion') AS NVARCHAR(128))", self.statement_timeout)
            .await?;
        let row = rows.into_iter().next().ok_or(StoreError::Corrupt("server version query returned no row"))?;
        MssqlRow(row).text_at(0)
    }

    /// Stop lending sessions; idle sessions close with the last pool handle.
    pub(super) async fn close(&self) {
        if let Ok(mut guard) = self.pool.lock() {
            guard.take();
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------------------------------------------

/// Map the `$n` parameter tokens of a shared statement to SQL Server's `@Pn`, and nothing else. String literals,
/// quoted identifiers and comments are copied unchanged; a `$` that is not a parameter token and any `@` outside
/// them are refused, so a statement can never mix the two parameter styles or reference session variables.
/// Returns the translated text and the highest parameter number.
fn to_tds(sql: &str) -> Result<(String, usize), StoreError> {
    let refuse = || local_error(DbErrorKind::Other, "statement_placeholder");
    let mut out = String::with_capacity(sql.len() + 16);
    let mut chars = sql.chars().peekable();
    let mut highest = 0usize;
    while let Some(c) = chars.next() {
        match c {
            '\'' | '"' | '[' => {
                let close = if c == '[' { ']' } else { c };
                out.push(c);
                loop {
                    let next = chars.next().ok_or_else(refuse)?;
                    out.push(next);
                    if next == close {
                        // A doubled delimiter is an escaped one inside the literal or identifier.
                        if chars.peek() == Some(&close) {
                            out.push(close);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
            }
            '-' if chars.peek() == Some(&'-') => {
                out.push(c);
                for next in chars.by_ref() {
                    out.push(next);
                    if next == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                out.push(c);
                out.push(chars.next().ok_or_else(refuse)?);
                let mut previous = ' ';
                loop {
                    let next = chars.next().ok_or_else(refuse)?;
                    out.push(next);
                    if previous == '*' && next == '/' {
                        break;
                    }
                    previous = next;
                }
            }
            '$' => {
                let mut digits = String::new();
                while let Some(d) = chars.peek().copied().filter(char::is_ascii_digit) {
                    digits.push(d);
                    chars.next();
                }
                let number: usize = digits.parse().map_err(|_| refuse())?;
                if number == 0 {
                    return Err(refuse());
                }
                highest = highest.max(number);
                out.push_str("@P");
                out.push_str(&digits);
            }
            '@' => return Err(refuse()),
            _ => out.push(c),
        }
    }
    Ok((out, highest))
}

fn bind(sql: &str, args: Vec<Val>) -> Result<Query<'static>, StoreError> {
    let (text, highest) = to_tds(sql)?;
    if highest > args.len() || args.len() > MAX_PARAMETERS {
        return Err(local_error(DbErrorKind::Other, "statement_parameters"));
    }
    let mut query = Query::new(text);
    for value in args {
        match value {
            Val::I64(v) => query.bind(v),
            Val::Text(v) => query.bind(v),
            Val::Blob(v) => query.bind(v),
        }
    }
    Ok(query)
}

// ---------------------------------------------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------------------------------------------

/// One open READ COMMITTED transaction on a pooled session. Dropping it without `commit`/`rollback` discards the
/// session (the server rolls back when it closes).
pub(super) struct MssqlTx {
    conn: PooledConnection<'static, Manager>,
    statement_timeout: Duration,
    /// A statement failed: the transaction may already be rolled back by `XACT_ABORT`.
    failed: bool,
}

impl MssqlTx {
    fn usable(&self) -> Result<(), StoreError> {
        if self.failed {
            return Err(local_error(DbErrorKind::Other, "transaction_aborted"));
        }
        Ok(())
    }

    fn track<T>(&mut self, result: Result<T, StoreError>) -> Result<T, StoreError> {
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    async fn raw_scalar(&mut self, sql: &str) -> Result<i64, StoreError> {
        self.usable()?;
        let rows = self.conn.simple(sql, self.statement_timeout).await;
        let row = self.track(rows)?.into_iter().next().ok_or(StoreError::Corrupt("scalar query returned no row"))?;
        MssqlRow(row).i64_at(0)
    }

    pub(super) async fn commit(mut self) -> Result<(), StoreError> {
        if self.failed {
            // Known not committed: the operation already failed inside this transaction.
            self.rollback().await;
            return Err(local_error(DbErrorKind::Other, "transaction_aborted"));
        }
        match self.conn.simple("COMMIT TRANSACTION", self.statement_timeout).await {
            Ok(_) => {
                self.conn.in_tx = false;
                Ok(())
            }
            Err(error) => {
                // Whatever happened, this session never returns to the pool.
                self.conn.broken = true;
                let db = match error {
                    StoreError::Unavailable(db)
                    | StoreError::Busy(db)
                    | StoreError::Database(db)
                    | StoreError::CommitUnknown(db) => db,
                    _ => db_error(DbErrorKind::Other, Some("commit".into())),
                };
                Err(StoreError::CommitUnknown(db))
            }
        }
    }

    pub(super) async fn rollback(mut self) {
        if self.conn.broken || self.conn.in_flight {
            return;
        }
        match self.conn.simple("IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION", self.statement_timeout).await {
            Ok(_) => self.conn.in_tx = false,
            Err(_) => self.conn.broken = true,
        }
    }
}

impl SqlTx for MssqlTx {
    type Row = MssqlRow;

    fn dialect(&self) -> Dialect {
        MSSQL
    }

    async fn execute(&mut self, sql: &str, args: Vec<Val>) -> Result<u64, StoreError> {
        self.usable()?;
        let query = bind(sql, args);
        let result = match query {
            Ok(query) => self.conn.execute(query, self.statement_timeout).await,
            Err(error) => Err(error),
        };
        self.track(result)
    }

    async fn fetch_optional(&mut self, sql: &str, args: Vec<Val>) -> Result<Option<MssqlRow>, StoreError> {
        let rows = self.fetch_all(sql, args).await?;
        Ok(rows.into_iter().next())
    }

    async fn fetch_all(&mut self, sql: &str, args: Vec<Val>) -> Result<Vec<MssqlRow>, StoreError> {
        self.usable()?;
        let query = bind(sql, args);
        let result = match query {
            Ok(query) => self.conn.query(query, self.statement_timeout).await,
            Err(error) => Err(error),
        };
        Ok(self.track(result)?.into_iter().map(MssqlRow).collect())
    }

    async fn execute_script(&mut self, sql: &'static str) -> Result<(), StoreError> {
        self.usable()?;
        let result = self.conn.simple(sql, self.statement_timeout).await.map(|_| ());
        self.track(result)
    }
}

/// Result row; integers widen losslessly from the narrower SQL Server integer types.
pub(super) struct MssqlRow(Row);

fn decode_error() -> StoreError {
    StoreError::Database(db_error(DbErrorKind::Decode, None))
}

impl MssqlRow {
    fn cell(&self, index: usize) -> Result<&ColumnData<'static>, StoreError> {
        self.0.get_column_data(index).map_err(|_| decode_error())
    }
}

impl SqlRow for MssqlRow {
    fn i64_at(&self, index: usize) -> Result<i64, StoreError> {
        self.opt_i64_at(index)?.ok_or_else(decode_error)
    }

    fn opt_i64_at(&self, index: usize) -> Result<Option<i64>, StoreError> {
        match self.cell(index)? {
            ColumnData::I64(v) => Ok(*v),
            ColumnData::I32(v) => Ok(v.map(i64::from)),
            ColumnData::I16(v) => Ok(v.map(i64::from)),
            ColumnData::U8(v) => Ok(v.map(i64::from)),
            ColumnData::Bit(v) => Ok(v.map(i64::from)),
            _ => Err(decode_error()),
        }
    }

    fn text_at(&self, index: usize) -> Result<String, StoreError> {
        self.opt_text_at(index)?.ok_or_else(decode_error)
    }

    fn opt_text_at(&self, index: usize) -> Result<Option<String>, StoreError> {
        match self.cell(index)? {
            ColumnData::String(v) => Ok(v.as_ref().map(|s| s.to_string())),
            _ => Err(decode_error()),
        }
    }

    fn blob_at(&self, index: usize) -> Result<Vec<u8>, StoreError> {
        self.opt_blob_at(index)?.ok_or_else(decode_error)
    }

    fn opt_blob_at(&self, index: usize) -> Result<Option<Vec<u8>>, StoreError> {
        match self.cell(index)? {
            ColumnData::Binary(v) => Ok(v.as_ref().map(|b| b.to_vec())),
            _ => Err(decode_error()),
        }
    }
}
