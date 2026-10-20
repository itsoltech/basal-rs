//! Durable Batch store: typed domain operations over SQLite, PostgreSQL, SQL Server or MongoDB.
//!
//! [`Store`] is the only entry point. Every method is one complete domain operation in one short transaction (or a
//! bounded sequence of them); there are no public `begin/execute/commit` calls. The SQL engines run the same
//! operations from `ops` with explicit dialect fragments (see `sql.rs`), so the lifecycle, quota and fencing
//! semantics are identical. SQL Server uses the native TDS adapter in `mssql.rs`.
//!
//! MongoDB has its own native adapter (`mongodb/`): collection-level domain operations over small documents with the
//! same invariants, replica-set transactions and real writes of the shared batch control document; no SQL emulation
//! and no batch-sized documents. Every `Store` method dispatches explicitly to the SQL operations (`Db::Sql`, through
//! the SQL-only `in_tx!`) or to the native MongoDB store (`Db::Mongodb`). The domain types in [`types`], the error
//! contract in [`error`] and the pure rules in `policy` (reservation sizes, expiry line, DRR, digests) are shared by
//! every engine. The SQL Server differences (`TOP` instead of `LIMIT`, guarded inserts instead of `ON CONFLICT`,
//! `UPDLOCK`/`READPAST`/`READCOMMITTEDLOCK` table hints instead of `FOR UPDATE [SKIP LOCKED]`, `@Pn` parameters,
//! length-aware key equality, `VARBINARY(256)` custom ids) are explicit [`sql::Dialect`] fragments; its transaction
//! type adds lock/statement timeouts and never returns an uncertain TDS session to the pool.
//!
//! Invariants kept by all operations:
//! - Owner scoping: every read and write is filtered by the owner from authentication, or by a worker token whose
//!   claim row and batch control are re-checked under the batch lock. A resource of another owner is `NotFound`.
//! - Fencing: execution claims carry a random claim id, phase leases a per-batch generation; cancel and expiry bump
//!   `control_epoch`. Commits are accepted only before the deadline, under the batch lock, after re-reading the
//!   database clock. A known operation id returns its recorded outcome even after the lease ended.
//! - Quota: `used + reserved ≤ limit` is checked whenever a reservation grows (upload, create, claim). Each reserved
//!   or used byte has exactly one owner row and moves in the transaction that changes that row; finalization,
//!   expiry and cleanup never need new quota. `doctor`'s drift check recomputes the totals from the owner rows.
//! - Lock order: quota (owner) → schedule (shard) → batch control → items/claims/settlements → files/sessions.
//!   Commit, cancel, expiry and release never touch quota or shard rows; settlement takes quota → shard → batch.
//! - Bounded transactions: claims, commits, drains, settlements, cleanup and GC take explicit limits; request
//!   bodies stay in artifacts and are addressed by byte ranges.
//!
//! Cancellation: dropping a future before its commit rolls the transaction back; a failure during the commit
//! itself is reported as [`StoreError::CommitUnknown`] and must be reconciled with the same operation id.

mod error;
mod mongodb;
mod mssql;
mod ops;
mod policy;
mod postgresql;
mod sql;
mod sqlite;
mod types;

use std::time::Duration;

pub(crate) use error::{InvalidValue, StoreError};
pub(crate) use mongodb::{MongoOptions, MongoTls};
pub(crate) use mssql::{MssqlOptions, MssqlTls};
pub(crate) use policy::{policy_digest, write_expired_line};
pub(crate) use postgresql::{ConnectionSecret, PgTls, PostgresOptions};
pub(crate) use sqlite::SqliteOptions;
pub(crate) use types::*;

use ops::admin::Migration;
use ops::claims::ClaimAttempt;
use ops::lifecycle::Created;

const SQLITE_MIGRATIONS: &[Migration] = &[
    Migration { version: 1, id: "0001_init", sql: include_str!("migrations/sqlite/0001_init.sql") },
    Migration { version: 2, id: "0002_abandonment", sql: include_str!("migrations/sqlite/0002_abandonment.sql") },
];

const POSTGRESQL_MIGRATIONS: &[Migration] = &[
    Migration { version: 1, id: "0001_init", sql: include_str!("migrations/postgresql/0001_init.sql") },
    Migration { version: 2, id: "0002_abandonment", sql: include_str!("migrations/postgresql/0002_abandonment.sql") },
];

const MSSQL_MIGRATIONS: &[Migration] = &[
    Migration { version: 1, id: "0001_init", sql: include_str!("migrations/mssql/0001_init.sql") },
    Migration { version: 2, id: "0002_abandonment", sql: include_str!("migrations/mssql/0002_abandonment.sql") },
];

/// Retries of a transaction rolled back by a lock timeout, deadlock or busy database.
const MAX_TX_RETRIES: u32 = 3;
/// Attempts of one `claim_items` call (changed DRR choice or quota-blocked owners).
const MAX_CLAIM_ATTEMPTS: usize = 8;

/// SQL engines; they share the domain operations in `ops`.
#[derive(Clone, Debug)]
enum SqlDb {
    Sqlite(sqlite::SqliteDb),
    Postgresql(postgresql::PgDb),
    Mssql(mssql::MssqlDb),
}

#[derive(Clone, Debug)]
enum Db {
    Sql(SqlDb),
    Mongodb(mongodb::MongoStore),
}

/// Handle of the batch store; cheap to clone (pooled connections).
#[derive(Clone, Debug)]
pub(crate) struct Store {
    db: Db,
    settings: StoreSettings,
}

/// Outcome of `create_batch`.
#[derive(Clone, Debug)]
pub(crate) enum CreateOutcome {
    Created(BatchRecord),
    /// Same Idempotency-Key and request digest: the current resource of the earlier create.
    Replayed(BatchRecord),
}

/// Run `$body` (an operation future using `$tx: &mut impl SqlTx`) in a transaction of the SQL engine `$sql`
/// (`&SqlDb`); commit on success, roll back on error, retry rolled-back busy transactions a bounded number of times.
/// MongoDB never goes through this macro.
macro_rules! in_tx {
    ($sql:expr, $tx:ident => $body:expr) => {{
        let mut attempt = 0u32;
        loop {
            let result = match $sql {
                SqlDb::Sqlite(db) => match db.begin().await {
                    Ok(mut handle) => {
                        let outcome = {
                            let $tx = &mut handle;
                            $body.await
                        };
                        match outcome {
                            Ok(value) => handle.commit().await.map(|()| value),
                            Err(error) => {
                                handle.rollback().await;
                                Err(error)
                            }
                        }
                    }
                    Err(error) => Err(error),
                },
                SqlDb::Postgresql(db) => match db.begin().await {
                    Ok(mut handle) => {
                        let outcome = {
                            let $tx = &mut handle;
                            $body.await
                        };
                        match outcome {
                            Ok(value) => handle.commit().await.map(|()| value),
                            Err(error) => {
                                handle.rollback().await;
                                Err(error)
                            }
                        }
                    }
                    Err(error) => Err(error),
                },
                SqlDb::Mssql(db) => match db.begin().await {
                    Ok(mut handle) => {
                        let outcome = {
                            let $tx = &mut handle;
                            $body.await
                        };
                        match outcome {
                            Ok(value) => handle.commit().await.map(|()| value),
                            Err(error) => {
                                handle.rollback().await;
                                Err(error)
                            }
                        }
                    }
                    Err(error) => Err(error),
                },
            };
            match result {
                Err(error) if error.is_retryable() && attempt < MAX_TX_RETRIES => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_millis(20 * u64::from(attempt))).await;
                }
                other => break other,
            }
        }
    }};
}

impl Store {
    // -----------------------------------------------------------------------------------------------------------
    // Construction, schema and administration
    // -----------------------------------------------------------------------------------------------------------

    /// Open the local SQLite store; with `create` the file is created (0600) and migrated automatically.
    /// An existing newer or damaged schema is refused, never changed.
    pub(crate) async fn open_sqlite(options: &SqliteOptions, settings: StoreSettings) -> Result<Self, StoreError> {
        let db = SqlDb::Sqlite(sqlite::SqliteDb::open(options).await?);
        let state = if options.create {
            Some(in_tx!(&db, tx => ops::admin::apply_migrations(tx, SQLITE_MIGRATIONS))?)
        } else {
            None
        };
        let store = Self { db: Db::Sql(db), settings };
        let state = if let Some(state) = state { state } else { store.schema_state().await? };
        match state {
            SchemaState::Current { .. } => Ok(store),
            other => Err(StoreError::Schema(other)),
        }
    }

    /// Connect the runtime to PostgreSQL. Refuses unless the schema is exactly current; never runs DDL.
    pub(crate) async fn connect_postgresql(
        options: &PostgresOptions,
        settings: StoreSettings,
    ) -> Result<Self, StoreError> {
        let db = postgresql::PgDb::connect(options).await?;
        let store = Self { db: Db::Sql(SqlDb::Postgresql(db)), settings };
        match store.schema_state().await? {
            SchemaState::Current { .. } => Ok(store),
            other => Err(StoreError::Schema(other)),
        }
    }

    /// Apply pending PostgreSQL migrations with the migration account (`basal batches migrate`), serialized by an
    /// advisory lock. Returns the resulting schema state; a newer or damaged schema is an error.
    pub(crate) async fn migrate_postgresql(options: &PostgresOptions) -> Result<SchemaState, StoreError> {
        let db = postgresql::PgDb::connect(options).await?;
        let mut tx = db.begin_migration().await?;
        let state = match ops::admin::apply_migrations(&mut tx, POSTGRESQL_MIGRATIONS).await {
            Ok(state) => state,
            Err(error) => {
                tx.rollback().await;
                db.close().await;
                return Err(error);
            }
        };
        let committed = tx.commit().await;
        db.close().await;
        committed.map(|()| state)
    }

    /// Connect the runtime to SQL Server. Refuses unless the schema is exactly current; never runs DDL.
    pub(crate) async fn connect_mssql(options: &MssqlOptions, settings: StoreSettings) -> Result<Self, StoreError> {
        let db = mssql::MssqlDb::connect(options).await?;
        let store = Self { db: Db::Sql(SqlDb::Mssql(db)), settings };
        match store.schema_state().await? {
            SchemaState::Current { .. } => Ok(store),
            other => {
                store.close().await;
                Err(StoreError::Schema(other))
            }
        }
    }

    /// Apply pending SQL Server migrations with the migration account (`basal batches migrate`) in one transaction
    /// holding `sp_getapplock`. Returns the resulting schema state; a newer or damaged schema is an error.
    pub(crate) async fn migrate_mssql(options: &MssqlOptions) -> Result<SchemaState, StoreError> {
        let db = mssql::MssqlDb::connect(options).await?;
        let mut tx = match db.begin_migration().await {
            Ok(tx) => tx,
            Err(error) => {
                db.close().await;
                return Err(error);
            }
        };
        let state = match ops::admin::apply_migrations(&mut tx, MSSQL_MIGRATIONS).await {
            Ok(state) => state,
            Err(error) => {
                tx.rollback().await;
                db.close().await;
                return Err(error);
            }
        };
        let committed = tx.commit().await;
        db.close().await;
        committed.map(|()| state)
    }

    /// Connect the runtime to MongoDB (replica set or sharded cluster, wire version ≥ 8). Refuses unless the schema
    /// is exactly current; never creates collections or indexes. Call [`Store::close`] before shutdown.
    pub(crate) async fn connect_mongodb(options: &MongoOptions, settings: StoreSettings) -> Result<Self, StoreError> {
        let mongo = mongodb::MongoStore::connect(options, settings).await?;
        Ok(Self { db: Db::Mongodb(mongo), settings })
    }

    /// Create the MongoDB collections and indexes of pending schema versions with the migration account
    /// (`basal batches migrate`); resumes an interrupted migration with the same checksum.
    pub(crate) async fn migrate_mongodb(options: &MongoOptions) -> Result<SchemaState, StoreError> {
        mongodb::MongoStore::migrate(options).await
    }

    pub(crate) fn backend(&self) -> BackendKind {
        match self.db {
            Db::Sql(SqlDb::Sqlite(_)) => BackendKind::Sqlite,
            Db::Sql(SqlDb::Postgresql(_)) => BackendKind::Postgresql,
            Db::Sql(SqlDb::Mssql(_)) => BackendKind::Mssql,
            Db::Mongodb(_) => BackendKind::Mongodb,
        }
    }

    pub(crate) fn settings(&self) -> &StoreSettings {
        &self.settings
    }

    pub(crate) fn capabilities(&self) -> StoreCapabilities {
        if let Db::Mongodb(mongo) = &self.db {
            return mongo.capabilities();
        }
        let server = matches!(self.backend(), BackendKind::Postgresql | BackendKind::Mssql);
        StoreCapabilities {
            backend: self.backend(),
            transactional_ddl: true,
            row_locks: server,
            skip_locked: server,
            server_clock: server,
        }
    }

    pub(crate) async fn schema_state(&self) -> Result<SchemaState, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.schema_state().await,
            Db::Sql(sql) => {
                let migrations = match sql {
                    SqlDb::Sqlite(_) => SQLITE_MIGRATIONS,
                    SqlDb::Postgresql(_) => POSTGRESQL_MIGRATIONS,
                    SqlDb::Mssql(_) => MSSQL_MIGRATIONS,
                };
                in_tx!(sql, tx => ops::admin::schema_state(tx, migrations))
            }
        }
    }

    /// Connection, schema, active revisions and bounded counters; no DSN, owners or batch ids are reported.
    pub(crate) async fn doctor(&self) -> Result<DoctorReport, StoreError> {
        let server_version = match &self.db {
            Db::Sql(SqlDb::Sqlite(db)) => db.server_version().await?,
            Db::Sql(SqlDb::Postgresql(db)) => db.server_version().await?,
            Db::Sql(SqlDb::Mssql(db)) => db.server_version().await?,
            Db::Mongodb(mongo) => mongo.server_version().await?,
        };
        let schema = self.schema_state().await?;
        let mut report = DoctorReport {
            backend: self.backend(),
            server_version,
            schema,
            quota_policy: None,
            catalog: None,
            batches_by_status: Vec::new(),
            expired_claims: 0,
            unaccounted_settlements: 0,
            pending_cleanup: 0,
            sessions_open: 0,
        };
        if !matches!(schema, SchemaState::Current { .. }) {
            return Ok(report);
        }
        report.quota_policy = self.active_quota_policy().await?.map(|p| p.revision);
        report.catalog = self.latest_catalog().await?.map(|c| c.revision);
        let counters = match &self.db {
            Db::Mongodb(mongo) => mongo.counters().await?,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::counters(tx))?,
        };
        report.batches_by_status = counters.batches_by_status;
        report.expired_claims = counters.expired_claims;
        report.unaccounted_settlements = counters.unaccounted_settlements;
        report.pending_cleanup = counters.pending_cleanup;
        report.sessions_open = counters.sessions_open;
        Ok(report)
    }

    /// Bounded backlog by fingerprint and status (validating and in-progress) for health and metrics; a validating
    /// fingerprint without a compatible worker shows up here. Contains no owner, batch or client identifiers.
    pub(crate) async fn backlog(&self, limit: u32) -> Result<Vec<BacklogEntry>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.backlog(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::backlog(tx, limit)),
        }
    }

    /// Owners whose `batch_quota` totals differ from their owner rows (bounded by `owner_limit`).
    pub(crate) async fn quota_drift(&self, owner_limit: u32) -> Result<Vec<QuotaDrift>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.quota_drift(owner_limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::quota_drift(tx, owner_limit)),
        }
    }

    pub(crate) async fn quota_usage(&self, owner: &OwnerId) -> Result<QuotaUsage, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.quota_usage(owner).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::quota_usage_of(tx, owner)),
        }
    }

    pub(crate) async fn install_catalog(&self, spec: &CatalogSpec) -> Result<InstallOutcome, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.install_catalog(spec).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::install_catalog(tx, spec)),
        }
    }

    /// Install a quota policy revision (administrative; run with batch admission paused).
    pub(crate) async fn install_quota_policy(&self, spec: &QuotaPolicySpec) -> Result<InstallOutcome, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.install_quota_policy(spec).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::install_quota_policy(tx, spec)),
        }
    }

    pub(crate) async fn latest_catalog(&self) -> Result<Option<Catalog>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.latest_catalog().await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::latest_catalog(tx)),
        }
    }

    /// Read an exact retained catalog revision; callers expecting pinned metadata must reject absence.
    pub(crate) async fn catalog_revision(&self, revision: Revision) -> Result<Option<Catalog>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.catalog_revision(revision).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::catalog_revision(tx, revision)),
        }
    }

    pub(crate) async fn active_quota_policy(&self) -> Result<Option<QuotaPolicy>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.active_quota_policy().await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::admin::active_quota_policy(tx)),
        }
    }

    /// Local profile: install the catalog and policy once when none exist; existing revisions are kept so an earlier
    /// queue is never re-targeted. Returns the active revisions.
    pub(crate) async fn bootstrap_local(
        &self,
        catalog: &CatalogSpec,
        policy: &QuotaPolicySpec,
    ) -> Result<(Revision, Revision), StoreError> {
        let catalog = match self.latest_catalog().await? {
            Some(existing) => existing.revision,
            None => match self.install_catalog(catalog).await? {
                InstallOutcome::Installed(r) | InstallOutcome::Unchanged(r) => r,
            },
        };
        let policy = match self.active_quota_policy().await? {
            Some(existing) => existing.revision,
            None => match self.install_quota_policy(policy).await? {
                InstallOutcome::Installed(r) | InstallOutcome::Unchanged(r) => r,
            },
        };
        Ok((catalog, policy))
    }

    /// Close pools. For MongoDB this also stops the operation supervisor and waits for in-flight operations, so it
    /// must be called before runtime shutdown.
    pub(crate) async fn close(&self) {
        match &self.db {
            Db::Sql(SqlDb::Sqlite(db)) => db.close().await,
            Db::Sql(SqlDb::Postgresql(db)) => db.close().await,
            Db::Sql(SqlDb::Mssql(db)) => db.close().await,
            Db::Mongodb(mongo) => mongo.close().await,
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Files
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn reserve_upload(&self, upload: &NewUpload) -> Result<UploadSession, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.reserve_upload(upload).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::reserve_upload(tx, &settings, upload)),
        }
    }

    pub(crate) async fn complete_upload(
        &self,
        owner: &OwnerId,
        session: &SessionId,
        object: StoredObject,
    ) -> Result<FileRecord, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.complete_upload(owner, session, object).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::complete_upload(tx, &settings, owner, session, object)),
        }
    }

    pub(crate) async fn abort_upload(&self, owner: &OwnerId, session: &SessionId) -> Result<(), StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.abort_upload(owner, session).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::abort_upload(tx, owner, session)),
        }
    }

    pub(crate) async fn get_file(&self, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.get_file(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::get_file(tx, owner, id)),
        }
    }

    pub(crate) async fn list_files(
        &self,
        owner: &OwnerId,
        purpose: Option<FilePurpose>,
        page: &PageRequest<FileId>,
    ) -> Result<Page<FileRecord>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.list_files(owner, purpose, page).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::list_files(tx, owner, purpose, page)),
        }
    }

    pub(crate) async fn delete_file(&self, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.delete_file(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::delete_file(tx, &settings, owner, id)),
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Batches
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn create_batch(
        &self,
        owner: &OwnerId,
        idempotency: Option<&Idempotency>,
        spec: &NewBatch,
    ) -> Result<CreateOutcome, StoreError> {
        let sql = match &self.db {
            Db::Mongodb(mongo) => {
                return match mongo.create_batch(owner, idempotency, spec).await? {
                    Created::New(batch) => Ok(CreateOutcome::Created(batch)),
                    Created::Replayed(batch) => Ok(CreateOutcome::Replayed(batch)),
                };
            }
            Db::Sql(sql) => sql,
        };
        let mut attempt = 0;
        loop {
            match in_tx!(sql, tx => ops::lifecycle::create_batch(tx, owner, idempotency, spec)) {
                Ok(Created::New(batch)) => return Ok(CreateOutcome::Created(batch)),
                Ok(Created::Replayed(batch)) => return Ok(CreateOutcome::Replayed(batch)),
                // A concurrent create with the same key won the unique ledger row: replay it.
                Err(error) if error.unique_violation() && idempotency.is_some() && attempt == 0 => attempt += 1,
                Err(error) => return Err(error),
            }
        }
    }

    pub(crate) async fn get_batch(&self, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.get_batch(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::get_batch(tx, owner, id)),
        }
    }

    pub(crate) async fn list_batches(
        &self,
        owner: &OwnerId,
        page: &PageRequest<BatchId>,
    ) -> Result<Page<BatchRecord>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.list_batches(owner, page).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::list_batches(tx, owner, page)),
        }
    }

    /// `errors.data` of a batch (at most 1000 entries; see `BatchRecord::errors_truncated`).
    pub(crate) async fn batch_errors(&self, owner: &OwnerId, id: &BatchId) -> Result<Vec<BatchIssue>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.batch_errors(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::batch_errors(tx, owner, id)),
        }
    }

    pub(crate) async fn request_cancel(&self, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.request_cancel(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::request_cancel(tx, &settings, owner, id)),
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Discovery and validation
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn claim_discovery(&self, worker: &WorkerId) -> Result<Option<DiscoveryLease>, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.claim_discovery(worker).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::claim_discovery(tx, &settings, worker)),
        }
    }

    pub(crate) async fn resolve_target(
        &self,
        lease: &ClaimToken,
        model: Option<&ModelName>,
    ) -> Result<TargetResolution, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.resolve_target(lease, model).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::resolve_target(tx, lease, model)),
        }
    }

    pub(crate) async fn claim_validation(
        &self,
        worker: &WorkerId,
        targets: &[WorkerTarget],
    ) -> Result<Option<ValidationLease>, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.claim_validation(worker, targets).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::claim_validation(tx, &settings, worker, targets)),
        }
    }

    /// Append at most 1024 validated records (idempotent per line).
    pub(crate) async fn append_validated(
        &self,
        lease: &ClaimToken,
        records: &[ValidatedRecord],
    ) -> Result<AppendOutcome, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.append_validated(lease, records).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::append_validated(tx, lease, records)),
        }
    }

    pub(crate) async fn seal_input(
        &self,
        lease: &ClaimToken,
        request: &SealRequest,
    ) -> Result<BatchRecord, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.seal_input(lease, request).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::seal_input(tx, lease, request)),
        }
    }

    pub(crate) async fn fail_validation(
        &self,
        lease: &ClaimToken,
        issues: &[BatchIssue],
        truncated: bool,
    ) -> Result<BatchRecord, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.fail_validation(lease, issues, truncated).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::ingest::fail_validation(tx, &settings, lease, issues, truncated)),
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Execution
    // -----------------------------------------------------------------------------------------------------------

    /// Claim the next microseries by cost DRR. Each attempt is one short transaction; quota-blocked owners are
    /// recorded and skipped in a new transaction, never by taking another quota lock under the shard lock.
    pub(crate) async fn claim_items(&self, request: &ClaimRequest) -> Result<ClaimOutcome, StoreError> {
        let sql = match &self.db {
            Db::Mongodb(mongo) => return mongo.claim_items(request).await,
            Db::Sql(sql) => sql,
        };
        let settings = self.settings;
        let mut excluded: Vec<OwnerId> = Vec::new();
        for _ in 0..MAX_CLAIM_ATTEMPTS {
            // Retry and Blocked commit too: Blocked persists the quota block, Retry wrote nothing but the quota row.
            let outcome = in_tx!(sql, tx => ops::claims::claim_attempt(tx, &settings, request, &excluded));
            match outcome {
                // Work exists only for owners skipped in this call because of their quota.
                Ok(ClaimAttempt::Done(ClaimOutcome::Idle)) if !excluded.is_empty() => {
                    return Ok(ClaimOutcome::QuotaBlocked { retry_after: settings.quota_blocked_backoff });
                }
                Ok(ClaimAttempt::Done(result)) => return Ok(result),
                Ok(ClaimAttempt::Blocked(owner)) => excluded.push(owner),
                Ok(ClaimAttempt::Retry) => {}
                Err(error) if error.is_retryable() => {}
                Err(error) => return Err(error),
            }
        }
        Ok(ClaimOutcome::QuotaBlocked { retry_after: settings.quota_blocked_backoff })
    }

    pub(crate) async fn renew_claim(&self, token: &ClaimToken) -> Result<LeaseUpdate<Renewal>, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.renew_claim(token).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::claims::renew_claim(tx, &settings, token)),
        }
    }

    pub(crate) async fn release_claim(
        &self,
        token: &ClaimToken,
        reason: ReleaseReason,
    ) -> Result<LeaseUpdate<()>, StoreError> {
        Ok(match self.release_claim_with_cost(token, reason, &[]).await? {
            LeaseUpdate::Applied(_) => LeaseUpdate::Applied(()),
            LeaseUpdate::Stale => LeaseUpdate::Stale,
        })
    }

    /// Atomically decide unfinished settlements with their measured cost (or estimate when unmeasured).
    pub(crate) async fn release_claim_with_cost(
        &self,
        token: &ClaimToken,
        reason: ReleaseReason,
        costs: &[AbandonedCost],
    ) -> Result<LeaseUpdate<ReleaseReport>, StoreError> {
        let costs = policy::abandoned_costs(costs)?;
        match &self.db {
            Db::Mongodb(mongo) => mongo.release_claim_with_cost(token, reason, &costs).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::claims::release_claim(tx, token, reason, &costs)),
        }
    }

    pub(crate) async fn mark_solo(&self, token: &ClaimToken, lines: &[u64]) -> Result<LeaseUpdate<()>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.mark_solo(token, lines).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::claims::mark_solo(tx, token, lines)),
        }
    }

    /// Open a result session before writing a large result object (see [`ResultPayload::Stored`]).
    pub(crate) async fn begin_result_upload(
        &self,
        token: &ClaimToken,
        line_no: u64,
        artifact: &ArtifactRef,
    ) -> Result<SessionId, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.begin_result_upload(token, line_no, artifact).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::claims::begin_result_upload(tx, token, line_no, artifact)),
        }
    }

    /// Commit at most 32 results of a claim under `operation`. Retrying after [`StoreError::CommitUnknown`] with the
    /// same operation id and results returns the recorded outcome without running the GPU again.
    pub(crate) async fn commit_results(
        &self,
        operation: &OperationId,
        token: &ClaimToken,
        results: &[ItemResult],
    ) -> Result<CommitReport, StoreError> {
        ops::commit::check_results(results)?;
        let digest = policy::commit_digest(token, results);
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.commit_results(operation, token, results, digest).await,
            Db::Sql(sql) => {
                in_tx!(sql, tx => ops::commit::commit_results(tx, &settings, operation, token, results, digest))
            }
        }
    }

    pub(crate) async fn owners_with_pending_settlements(&self, limit: u32) -> Result<Vec<OwnerId>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.owners_with_pending_settlements(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::commit::owners_with_pending_settlements(tx, limit)),
        }
    }

    /// Account decided settlements of one owner exactly once (one shard per call, at most `limit` rows).
    pub(crate) async fn settle_pending(&self, owner: &OwnerId, limit: u32) -> Result<AccountingProgress, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.settle_pending(owner, limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::commit::settle_pending(tx, &settings, owner, limit)),
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Deadline, cancel barrier and recovery
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn due_expirations(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.due_expirations(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::due_expirations(tx, limit)),
        }
    }

    pub(crate) async fn expire_batch(&self, batch: &BatchRef) -> Result<BatchRecord, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.expire_batch(batch).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::expire_batch(tx, &settings, batch)),
        }
    }

    pub(crate) async fn draining_batches(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.draining_batches(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::draining_batches(tx, limit)),
        }
    }

    /// Materialize at most `limit` remaining items of a cancelled or expired batch.
    pub(crate) async fn advance_drain(&self, batch: &BatchRef, limit: u32) -> Result<DrainProgress, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.advance_drain(batch, limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::lifecycle::advance_drain(tx, batch, limit)),
        }
    }

    /// Requeue items of at most `limit` expired execution claims, one transaction per claim. Returns the number
    /// of recovered claims; claims of healthy workers are never touched.
    pub(crate) async fn recover_expired_claims(&self, limit: u32) -> Result<u64, StoreError> {
        let sql = match &self.db {
            Db::Mongodb(mongo) => return mongo.recover_expired_claims(limit).await,
            Db::Sql(sql) => sql,
        };
        let expired = in_tx!(sql, tx => ops::claims::expired_claims(tx, limit))?;
        let mut recovered = 0;
        for (claim, batch) in &expired {
            if in_tx!(sql, tx => ops::claims::recover_claim(tx, claim, batch))? {
                recovered += 1;
            }
        }
        Ok(recovered)
    }

    // -----------------------------------------------------------------------------------------------------------
    // Finalization
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn claim_finalization(
        &self,
        worker: &WorkerId,
        keys: &ExportKeys,
    ) -> Result<FinalizationClaim, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.claim_finalization(worker, keys).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::claim_finalization(tx, &settings, worker, keys)),
        }
    }

    /// Take over the reservation of an earlier export generation after its objects were deleted and its writer is
    /// past `writer_stopped_after`; otherwise [`StoreError::FencingPending`].
    pub(crate) async fn adopt_export(
        &self,
        lease: &ClaimToken,
        previous: &SessionId,
        keys: &ExportKeys,
    ) -> Result<FinalizationLease, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.adopt_export(lease, previous, keys).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::adopt_export(tx, &settings, lease, previous, keys)),
        }
    }

    pub(crate) async fn read_export_items(
        &self,
        lease: &ClaimToken,
        after_line: u64,
        limit: u32,
    ) -> Result<Vec<ExportItem>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.read_export_items(lease, after_line, limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::read_export_items(tx, lease, after_line, limit)),
        }
    }

    pub(crate) async fn publish_files(
        &self,
        operation: &OperationId,
        lease: &ClaimToken,
        request: &PublishRequest,
    ) -> Result<BatchRecord, StoreError> {
        let digest = policy::publish_digest(lease, request);
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.publish_files(operation, lease, request, digest).await,
            Db::Sql(sql) => {
                in_tx!(sql, tx => ops::finalize::publish_files(tx, &settings, operation, lease, request, digest))
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------------
    // Expiry, GC, cleanup and retention (bounded supervisors)
    // -----------------------------------------------------------------------------------------------------------

    pub(crate) async fn expire_files(&self, limit: u32) -> Result<u64, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.expire_files(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::expire_files(tx, &settings, limit)),
        }
    }

    pub(crate) async fn file_gc_candidates(&self, limit: u32) -> Result<Vec<FileGcCandidate>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.file_gc_candidates(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::file_gc_candidates(tx, limit)),
        }
    }

    /// Call after the object of a GC candidate was deleted.
    pub(crate) async fn purge_file(&self, owner: &OwnerId, id: &FileId) -> Result<bool, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.purge_file(owner, id).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::purge_file(tx, owner, id)),
        }
    }

    pub(crate) async fn session_gc_candidates(&self, limit: u32) -> Result<Vec<SessionGcCandidate>, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.session_gc_candidates(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::session_gc_candidates(tx, &settings, limit)),
        }
    }

    /// Call after the objects of a session GC candidate were deleted.
    pub(crate) async fn release_session(&self, owner: &OwnerId, session: &SessionId) -> Result<bool, StoreError> {
        let settings = self.settings;
        match &self.db {
            Db::Mongodb(mongo) => mongo.release_session(owner, session).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::files::release_session(tx, &settings, owner, session)),
        }
    }

    pub(crate) async fn cleanup_batches(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.cleanup_batches(limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::cleanup_batches(tx, limit)),
        }
    }

    pub(crate) async fn cleanup_page(&self, batch: &BatchRef, limit: u32) -> Result<CleanupPage, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.cleanup_page(batch, limit).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::cleanup_page(tx, batch, limit)),
        }
    }

    /// Call after the page's artifacts were deleted; returns true when the batch cleanup is complete.
    pub(crate) async fn cleanup_commit(
        &self,
        batch: &BatchRef,
        through_line_no: Option<u64>,
    ) -> Result<bool, StoreError> {
        match &self.db {
            Db::Mongodb(mongo) => mongo.cleanup_commit(batch, through_line_no).await,
            Db::Sql(sql) => in_tx!(sql, tx => ops::finalize::cleanup_commit(tx, batch, through_line_no)),
        }
    }

    /// Purge metadata (with Idempotency-Key entries) of at most `limit` batches past their retention.
    pub(crate) async fn purge_metadata(&self, limit: u32) -> Result<u64, StoreError> {
        let sql = match &self.db {
            Db::Mongodb(mongo) => return mongo.purge_metadata(limit).await,
            Db::Sql(sql) => sql,
        };
        let candidates = in_tx!(sql, tx => ops::finalize::metadata_candidates(tx, limit))?;
        let mut purged = 0;
        for batch in &candidates {
            if in_tx!(sql, tx => ops::finalize::purge_metadata(tx, batch))? {
                purged += 1;
            }
        }
        Ok(purged)
    }
}
