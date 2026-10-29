//! Driver plumbing of the MongoDB adapter: operation supervisor, transactions, control-document locks, error
//! redaction and typed BSON access.
//!
//! Cancellation: the MongoDB driver must not have its futures dropped mid-operation. Every store operation runs to
//! completion in its own task spawned on a [`TaskTracker`], holding a semaphore permit for its whole lifetime; the
//! caller only awaits the join handle, so dropping the caller leaves the operation running to its end. Arguments
//! are moved (owned clones) into the task. `close` stops admission and waits for in-flight operations.
//!
//! Transactions: snapshot read concern, majority + journal write concern, primary selection, `maxCommitTimeMS`.
//! A "lock" is a real write to a control document (`lock_seq` increment and `touched_at = $$NOW` through an update
//! pipeline), so two transactions that touch the same quota, shard or batch control document write-conflict and one
//! retries; a snapshot can never apply a decision on stale control state. Retries are bounded:
//! `TransientTransactionError` (and write conflicts) retry the whole callback, `UnknownTransactionCommitResult`
//! retries only the commit and finally reports [`StoreError::CommitUnknown`]. Callbacks perform database work only
//! (no artifact I/O, no GPU work).
//!
//! Clock: the time of a decision is `$$NOW` evaluated by the primary in the write that locks the control document
//! (returned as `touched_at`). This assumes `$$NOW` reflects the primary's wall clock at that write; T11 must confirm
//! it on the qualified server versions, together with failover behaviour (clock skew between members is assumed
//! bounded by NTP and below the lease safety margin).

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use mongodb::bson::spec::BinarySubtype;
use mongodb::bson::{doc, Binary, Bson, Document};
use mongodb::error::{ErrorKind, WriteFailure, TRANSIENT_TRANSACTION_ERROR, UNKNOWN_TRANSACTION_COMMIT_RESULT};
use mongodb::options::{
    Acknowledgment, ReadConcern, ReadPreference, ReturnDocument, SelectionCriteria, TransactionOptions, WriteConcern,
};
use mongodb::{ClientSession, Collection, Database};
use tokio::sync::Semaphore;
use tokio_util::task::TaskTracker;

use super::super::error::{DbError, DbErrorKind, StoreError};
use super::super::types::{BackendKind, StoreSettings};

const BACKEND: BackendKind = BackendKind::Mongodb;

/// Whole-transaction retries after transient errors or write conflicts.
pub(super) const MAX_TXN_RETRIES: u32 = 4;
/// Commit retries while the outcome is unknown.
const MAX_COMMIT_RETRIES: u32 = 3;

fn db_error(kind: DbErrorKind, code: Option<String>) -> DbError {
    DbError { backend: BACKEND, kind, code, constraint: None }
}

fn error_code(error: &mongodb::error::Error) -> Option<i32> {
    match error.kind.as_ref() {
        ErrorKind::Command(command) => Some(command.code),
        ErrorKind::Write(WriteFailure::WriteError(write)) => Some(write.code),
        ErrorKind::Write(WriteFailure::WriteConcernError(concern)) => Some(concern.code),
        _ => None,
    }
}

/// Map a driver error to a redacted store error: kind and numeric server code only, never the message (it can
/// contain document values, host names or the connection string).
pub(super) fn redact(error: &mongodb::error::Error) -> StoreError {
    let code = error_code(error);
    let code_text = code.map(|c| c.to_string());
    if error.contains_label(TRANSIENT_TRANSACTION_ERROR) {
        return StoreError::Busy(db_error(DbErrorKind::Busy, code_text));
    }
    let kind = match (error.kind.as_ref(), code) {
        // 11000 duplicate key; 112 WriteConflict; 50 MaxTimeMSExpired; 251 NoSuchTransaction.
        (_, Some(11000)) => DbErrorKind::UniqueViolation,
        (_, Some(112 | 251)) => DbErrorKind::Busy,
        (_, Some(50)) => DbErrorKind::Timeout,
        (
            ErrorKind::Io(_)
            | ErrorKind::ConnectionPoolCleared { .. }
            | ErrorKind::ServerSelection { .. }
            | ErrorKind::DnsResolve { .. }
            | ErrorKind::Authentication { .. }
            | ErrorKind::InvalidTlsConfig { .. }
            | ErrorKind::Shutdown,
            _,
        ) => DbErrorKind::Unavailable,
        (ErrorKind::BsonDeserialization(_) | ErrorKind::Bson(_), _) => DbErrorKind::Decode,
        _ => DbErrorKind::Other,
    };
    StoreError::from_db(db_error(kind, code_text))
}

pub(super) fn shutdown_error() -> StoreError {
    StoreError::Unavailable(db_error(DbErrorKind::Unavailable, Some("shutdown".to_owned())))
}

fn task_failed_error() -> StoreError {
    StoreError::Database(db_error(DbErrorKind::Other, Some("task_panicked".to_owned())))
}

pub(super) fn corrupt(what: &'static str) -> StoreError {
    StoreError::Corrupt(what)
}

// ---------------------------------------------------------------------------------------------------------------
// Supervisor
// ---------------------------------------------------------------------------------------------------------------

/// Bounded set of store operations, each running to completion in its own task.
#[derive(Clone)]
pub(super) struct Supervisor {
    permits: Arc<Semaphore>,
    tracker: TaskTracker,
}

impl Supervisor {
    pub(super) fn new(max_concurrent: usize) -> Self {
        Self { permits: Arc::new(Semaphore::new(max_concurrent.max(1))), tracker: TaskTracker::new() }
    }

    /// Run `operation` in a supervised task. Dropping the returned future does not cancel the operation.
    pub(super) async fn run<R, F>(&self, operation: F) -> Result<R, StoreError>
    where
        R: Send + 'static,
        F: Future<Output = Result<R, StoreError>> + Send + 'static,
    {
        let permit = self.permits.clone().acquire_owned().await.map_err(|_| shutdown_error())?;
        if self.tracker.is_closed() {
            return Err(shutdown_error());
        }
        let handle = self.tracker.spawn(async move {
            let _permit = permit;
            operation.await
        });
        handle.await.unwrap_or_else(|_| Err(task_failed_error()))
    }

    /// Refuse new operations and wait until every started operation has finished.
    pub(super) async fn close(&self) {
        self.permits.close();
        self.tracker.close();
        self.tracker.wait().await;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Core and collections
// ---------------------------------------------------------------------------------------------------------------

pub(super) const SCHEMA: &str = "batch_schema";
pub(super) const POLICY: &str = "batch_quota_policy";
pub(super) const LIMITS: &str = "batch_quota_limits";
pub(super) const CATALOG_REVISIONS: &str = "batch_catalog_revisions";
pub(super) const CATALOG: &str = "batch_catalog";
pub(super) const QUOTA: &str = "batch_quota";
pub(super) const FILES: &str = "batch_files";
pub(super) const BATCHES: &str = "batches";
pub(super) const ERRORS: &str = "batch_errors";
pub(super) const ITEMS: &str = "batch_items";
pub(super) const CLAIMS: &str = "batch_claims";
pub(super) const SETTLEMENTS: &str = "batch_settlements";
pub(super) const OPERATIONS: &str = "batch_operations";
pub(super) const SCHEDULE: &str = "batch_schedule";
pub(super) const DEFICITS: &str = "batch_deficits";
pub(super) const SESSIONS: &str = "artifact_sessions";
pub(super) const API_USERS: &str = "api_users";
pub(super) const API_KEYS: &str = "api_keys";

/// Shared state of the adapter: client, database namespace, settings and per-operation server time limit.
pub(super) struct Core {
    pub client: mongodb::Client,
    pub db: Database,
    pub settings: StoreSettings,
    /// `maxTimeMS` of reads and aggregations, `maxCommitTimeMS` of commits.
    pub op_timeout: Duration,
}

impl Core {
    pub(super) fn coll(&self, name: &str) -> Collection<Document> {
        self.db.collection::<Document>(name)
    }

    fn transaction_options(&self) -> TransactionOptions {
        TransactionOptions::builder()
            .read_concern(ReadConcern::snapshot())
            .write_concern(majority_journal())
            .selection_criteria(SelectionCriteria::ReadPreference(ReadPreference::Primary))
            .max_commit_time(self.op_timeout)
            .build()
    }

    pub(super) async fn begin(&self) -> Result<ClientSession, StoreError> {
        let mut session = self.client.start_session().await.map_err(|e| redact(&e))?;
        session.start_transaction().with_options(self.transaction_options()).await.map_err(|e| redact(&e))?;
        Ok(session)
    }

    /// Commit with bounded retries of an unknown outcome.
    pub(super) async fn commit(&self, session: &mut ClientSession) -> Result<(), StoreError> {
        let mut attempt = 0;
        loop {
            match session.commit_transaction().await {
                Ok(()) => return Ok(()),
                Err(error) if error.contains_label(UNKNOWN_TRANSACTION_COMMIT_RESULT) => {
                    attempt += 1;
                    if attempt > MAX_COMMIT_RETRIES {
                        let code = error_code(&error).map(|c| c.to_string());
                        return Err(StoreError::CommitUnknown(db_error(DbErrorKind::Unavailable, code)));
                    }
                }
                Err(error) => return Err(redact(&error)),
            }
        }
    }

    /// Abort; a failure only means the server aborts the transaction itself (lifetime limit or session end).
    pub(super) async fn abort(&self, session: &mut ClientSession) {
        let _ = session.abort_transaction().await;
    }

    /// Database time of the primary (`hello.localTime`), used where no control document is written.
    pub(super) async fn server_now(&self) -> Result<i64, StoreError> {
        let reply = self
            .db
            .run_command(doc! { "hello": 1_i32 })
            .selection_criteria(SelectionCriteria::ReadPreference(ReadPreference::Primary))
            .await
            .map_err(|e| redact(&e))?;
        reply.get_datetime("localTime").map(|t| t.timestamp_millis()).map_err(|_| corrupt("hello localTime"))
    }
}

impl Core {
    /// Single-document read outside a transaction (majority read concern from the client options).
    pub(super) async fn read_one(&self, coll: &str, filter: Document) -> Result<Option<Document>, StoreError> {
        self.coll(coll).find_one(filter).max_time(self.op_timeout).await.map_err(|e| redact(&e))
    }

    /// Bounded, sorted read outside a transaction.
    pub(super) async fn read_many(
        &self,
        coll: &str,
        filter: Document,
        sort: Document,
        limit: i64,
    ) -> Result<Vec<Document>, StoreError> {
        let mut cursor = self
            .coll(coll)
            .find(filter)
            .sort(sort)
            .limit(limit.max(1))
            .max_time(self.op_timeout)
            .await
            .map_err(|e| redact(&e))?;
        let mut out = Vec::new();
        while cursor.advance().await.map_err(|e| redact(&e))? {
            out.push(cursor.deserialize_current().map_err(|e| redact(&e))?);
        }
        Ok(out)
    }

    /// Aggregation outside a transaction; the pipeline must be bounded.
    pub(super) async fn read_aggregate(
        &self,
        coll: &str,
        pipeline: Vec<Document>,
    ) -> Result<Vec<Document>, StoreError> {
        let mut cursor = self.coll(coll).aggregate(pipeline).max_time(self.op_timeout).await.map_err(|e| redact(&e))?;
        let mut out = Vec::new();
        while cursor.advance().await.map_err(|e| redact(&e))? {
            out.push(cursor.deserialize_current().map_err(|e| redact(&e))?);
        }
        Ok(out)
    }
}

pub(super) fn majority_journal() -> WriteConcern {
    WriteConcern::builder().w(Acknowledgment::Majority).journal(true).build()
}

/// Run `$body` (a future using `$tx: &mut Tx`) in a MongoDB transaction with bounded retries.
macro_rules! mtx {
    ($core:expr, $tx:ident => $body:expr) => {{
        let core: &$crate::serve::batches::store::mongodb::driver::Core = $core;
        let mut attempt = 0u32;
        loop {
            let result = match core.begin().await {
                Ok(mut session) => {
                    let outcome = {
                        let mut handle = $crate::serve::batches::store::mongodb::driver::Tx::new(core, &mut session);
                        let $tx = &mut handle;
                        $body.await
                    };
                    match outcome {
                        Ok(value) => core.commit(&mut session).await.map(|()| value),
                        Err(error) => {
                            core.abort(&mut session).await;
                            Err(error)
                        }
                    }
                }
                Err(error) => Err(error),
            };
            match result {
                Err(error)
                    if error.is_retryable()
                        && attempt < $crate::serve::batches::store::mongodb::driver::MAX_TXN_RETRIES =>
                {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(15 * u64::from(attempt))).await;
                }
                other => break other,
            }
        }
    }};
}
pub(super) use mtx;

// ---------------------------------------------------------------------------------------------------------------
// Transaction operations
// ---------------------------------------------------------------------------------------------------------------

/// Operations bound to one open transaction. Every read is bounded (`limit`, unique key or `$limit`).
pub(super) struct Tx<'a> {
    core: &'a Core,
    session: &'a mut ClientSession,
}

fn lock_pipeline(defaults: &Document) -> Vec<Document> {
    let mut set = Document::new();
    for (key, value) in defaults {
        set.insert(key.clone(), doc! { "$ifNull": [format!("${key}"), value.clone()] });
    }
    set.insert("lock_seq", doc! { "$add": [{ "$ifNull": ["$lock_seq", 0_i64] }, 1_i64] });
    set.insert("touched_at", doc! { "$toLong": "$$NOW" });
    vec![doc! { "$set": set }]
}

impl<'a> Tx<'a> {
    pub(super) fn new(core: &'a Core, session: &'a mut ClientSession) -> Self {
        Self { core, session }
    }

    pub(super) fn settings(&self) -> &StoreSettings {
        &self.core.settings
    }

    /// Primary clock for administrative timestamps (no control document involved).
    pub(super) async fn server_time(&self) -> Result<i64, StoreError> {
        self.core.server_now().await
    }

    /// Write-lock one control document and return it with the database time of the lock (`$$NOW`).
    pub(super) async fn lock(&mut self, coll: &str, filter: Document) -> Result<Option<(Document, i64)>, StoreError> {
        let doc = self
            .core
            .coll(coll)
            .find_one_and_update(filter, lock_pipeline(&Document::new()))
            .return_document(ReturnDocument::After)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        match doc {
            Some(doc) => {
                let now = int(&doc, "touched_at")?;
                Ok(Some((doc, now)))
            }
            None => Ok(None),
        }
    }

    /// Lock the document with `_id = id`, creating it with `defaults` when absent (unique `_id` guards races).
    pub(super) async fn lock_or_create(
        &mut self,
        coll: &str,
        id: &str,
        defaults: Document,
    ) -> Result<(Document, i64), StoreError> {
        let doc = self
            .core
            .coll(coll)
            .find_one_and_update(doc! { "_id": id }, lock_pipeline(&defaults))
            .upsert(true)
            .return_document(ReturnDocument::After)
            .session(&mut *self.session)
            .await
            .map_err(|e| match redact(&e) {
                // Two first-time creators of the same `_id`: the loser retries the whole transaction (bounded).
                StoreError::Database(db) if db.kind == DbErrorKind::UniqueViolation => StoreError::Busy(db),
                other => other,
            })?
            .ok_or(corrupt("upserted control document missing"))?;
        let now = int(&doc, "touched_at")?;
        Ok((doc, now))
    }

    pub(super) async fn one(&mut self, coll: &str, filter: Document) -> Result<Option<Document>, StoreError> {
        self.core
            .coll(coll)
            .find_one(filter)
            .max_time(self.core.op_timeout)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))
    }

    pub(super) async fn one_sorted(
        &mut self,
        coll: &str,
        filter: Document,
        sort: Document,
    ) -> Result<Option<Document>, StoreError> {
        Ok(self.many(coll, filter, sort, 1).await?.into_iter().next())
    }

    pub(super) async fn many(
        &mut self,
        coll: &str,
        filter: Document,
        sort: Document,
        limit: i64,
    ) -> Result<Vec<Document>, StoreError> {
        let mut cursor = self
            .core
            .coll(coll)
            .find(filter)
            .sort(sort)
            .limit(limit.max(1))
            .max_time(self.core.op_timeout)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        let mut out = Vec::new();
        while let Some(doc) = cursor.next(&mut *self.session).await {
            out.push(doc.map_err(|e| redact(&e))?);
        }
        Ok(out)
    }

    /// Aggregation; the pipeline must end with `$limit` or reduce to a bounded number of groups.
    pub(super) async fn aggregate(&mut self, coll: &str, pipeline: Vec<Document>) -> Result<Vec<Document>, StoreError> {
        let mut cursor = self
            .core
            .coll(coll)
            .aggregate(pipeline)
            .max_time(self.core.op_timeout)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        let mut out = Vec::new();
        while let Some(doc) = cursor.next(&mut *self.session).await {
            out.push(doc.map_err(|e| redact(&e))?);
        }
        Ok(out)
    }

    pub(super) async fn count(&mut self, coll: &str, filter: Document) -> Result<i64, StoreError> {
        let n = self
            .core
            .coll(coll)
            .count_documents(filter)
            .max_time(self.core.op_timeout)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        i64::try_from(n).map_err(|_| corrupt("count overflow"))
    }

    /// `SUM(field)` over the matching documents (missing values count as 0).
    pub(super) async fn sum(&mut self, coll: &str, filter: Document, field: &str) -> Result<i64, StoreError> {
        let rows = self
            .aggregate(
                coll,
                vec![
                    doc! { "$match": filter },
                    doc! { "$group": { "_id": Bson::Null, "s": { "$sum": format!("${field}") }, "n": { "$sum": 1_i64 } } },
                ],
            )
            .await?;
        match rows.first() {
            Some(row) => int(row, "s"),
            None => Ok(0),
        }
    }

    pub(super) async fn insert(&mut self, coll: &str, doc: Document) -> Result<(), StoreError> {
        self.core.coll(coll).insert_one(doc).session(&mut *self.session).await.map_err(|e| redact(&e))?;
        Ok(())
    }

    /// Update one document; returns the number of matched documents (0 or 1).
    pub(super) async fn update(&mut self, coll: &str, filter: Document, update: Document) -> Result<u64, StoreError> {
        let result = self
            .core
            .coll(coll)
            .update_one(filter, update)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        Ok(result.matched_count)
    }

    /// Update a bounded set (the filter always names a batch, claim or explicit keys).
    pub(super) async fn update_many(
        &mut self,
        coll: &str,
        filter: Document,
        update: Document,
    ) -> Result<u64, StoreError> {
        let result = self
            .core
            .coll(coll)
            .update_many(filter, update)
            .session(&mut *self.session)
            .await
            .map_err(|e| redact(&e))?;
        Ok(result.modified_count)
    }

    pub(super) async fn delete_one(&mut self, coll: &str, filter: Document) -> Result<u64, StoreError> {
        let result =
            self.core.coll(coll).delete_one(filter).session(&mut *self.session).await.map_err(|e| redact(&e))?;
        Ok(result.deleted_count)
    }

    pub(super) async fn delete_many(&mut self, coll: &str, filter: Document) -> Result<u64, StoreError> {
        let result =
            self.core.coll(coll).delete_many(filter).session(&mut *self.session).await.map_err(|e| redact(&e))?;
        Ok(result.deleted_count)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Typed BSON access (integers are always written as Int64)
// ---------------------------------------------------------------------------------------------------------------

pub(super) fn int(doc: &Document, key: &str) -> Result<i64, StoreError> {
    opt_int(doc, key)?.ok_or(corrupt("missing integer field"))
}

pub(super) fn opt_int(doc: &Document, key: &str) -> Result<Option<i64>, StoreError> {
    match doc.get(key) {
        None | Some(Bson::Null) => Ok(None),
        Some(Bson::Int64(v)) => Ok(Some(*v)),
        Some(Bson::Int32(v)) => Ok(Some(i64::from(*v))),
        Some(_) => Err(corrupt("integer field has another type")),
    }
}

pub(super) fn uint(doc: &Document, key: &str) -> Result<u64, StoreError> {
    u64::try_from(int(doc, key)?).map_err(|_| corrupt("negative counter"))
}

pub(super) fn text(doc: &Document, key: &str) -> Result<String, StoreError> {
    opt_text(doc, key)?.ok_or(corrupt("missing text field"))
}

pub(super) fn opt_text(doc: &Document, key: &str) -> Result<Option<String>, StoreError> {
    match doc.get(key) {
        None | Some(Bson::Null) => Ok(None),
        Some(Bson::String(v)) => Ok(Some(v.clone())),
        Some(_) => Err(corrupt("text field has another type")),
    }
}

pub(super) fn bin(doc: &Document, key: &str) -> Result<Vec<u8>, StoreError> {
    opt_bin(doc, key)?.ok_or(corrupt("missing binary field"))
}

pub(super) fn opt_bin(doc: &Document, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
    match doc.get(key) {
        None | Some(Bson::Null) => Ok(None),
        Some(Bson::Binary(binary)) if binary.subtype == BinarySubtype::Generic => Ok(Some(binary.bytes.clone())),
        Some(_) => Err(corrupt("binary field has another type")),
    }
}

pub(super) fn flag(doc: &Document, key: &str) -> Result<bool, StoreError> {
    match doc.get(key) {
        None | Some(Bson::Null) => Ok(false),
        Some(Bson::Boolean(v)) => Ok(*v),
        Some(_) => Err(corrupt("boolean field has another type")),
    }
}

/// Exact bytes (generic binary subtype): custom ids, digests, inline results.
pub(super) fn bytes(value: &[u8]) -> Bson {
    Bson::Binary(Binary { subtype: BinarySubtype::Generic, bytes: value.to_vec() })
}

pub(super) fn opt_i64(value: Option<i64>) -> Bson {
    value.map_or(Bson::Null, Bson::Int64)
}

pub(super) fn opt_str(value: Option<&str>) -> Bson {
    value.map_or(Bson::Null, |v| Bson::String(v.to_owned()))
}
