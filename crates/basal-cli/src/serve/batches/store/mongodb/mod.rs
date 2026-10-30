//! Native MongoDB adapter of the batch store.
//!
//! The same domain operations as the SQL adapters, written against MongoDB collections of small documents (no SQL
//! emulation, no batch-sized documents): `batches` holds the shared batch control document, `batch_items` one
//! document per record (byte range, binary `custom_id`, digests, bounded inline result), settlements, claims,
//! sessions, quota and DRR state in their own collections; request bodies and large results stay in artifacts.
//!
//! Requirements checked at connect: replica set or sharded cluster (transactions), wire version ≥ 8 (MongoDB 4.2:
//! update pipelines and `$$NOW`), current schema. TLS always verifies certificate and host name unless the server is
//! local and `LocalPlaintext` is chosen; a connection string cannot weaken this. Clock: decisions use `$$NOW` of the
//! write that locks the control document; selection queries use `hello.localTime` of the primary. Both are the
//! primary's clock; a primary failover may move it by the clock skew of the members (assumed bounded by NTP and
//! smaller than the lease safety margin). The 24 h deadline is always checked after the batch control lock.
//!
//! Native TTL indexes are not used: deadlines, expiry and GC are explicit, bounded operations.
//!
//! Every method runs in a supervised task (see `driver`); call [`MongoStore::close`] before process shutdown.

mod admin;
mod claims;
mod commit;
mod common;
mod convert;
pub(super) mod driver;
mod files;
mod finalize;
mod ingest;
mod keys;
mod lifecycle;
mod schema;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use mongodb::bson::{doc, Bson};
use mongodb::options::{ClientOptions, ReadConcern, ReadPreference, SelectionCriteria, ServerAddress, Tls, TlsOptions};

use super::error::StoreError;
use super::ops::admin::Counters;
use super::ops::lifecycle::Created;
use super::postgresql::ConnectionSecret;
use super::types::*;
use driver::{majority_journal, redact, Core, Supervisor};

/// TLS requirement of the MongoDB connection; options in the connection string cannot weaken it.
#[derive(Clone, Debug)]
pub(crate) enum MongoTls {
    /// TLS with certificate and host name verification; optional private CA bundle.
    VerifyFull { ca_file: Option<PathBuf> },
    /// Plaintext, accepted only when every host is loopback or a Unix socket.
    #[expect(
        dead_code,
        reason = "loopback-only operator qualification mode; production configuration requires verified TLS"
    )]
    LocalPlaintext,
}

#[derive(Clone, Debug)]
pub(crate) struct MongoOptions {
    pub connection: ConnectionSecret,
    /// Database (namespace) of the batch collections; `None` uses the database of the connection string.
    pub database: Option<String>,
    pub tls: MongoTls,
    pub max_pool_size: u32,
    pub min_pool_size: u32,
    /// Upper bound of concurrently running store operations (supervised tasks).
    pub max_concurrent_operations: usize,
    pub server_selection_timeout: Duration,
    pub connect_timeout: Duration,
    pub max_idle_time: Duration,
    /// `maxTimeMS` of reads and aggregations and `maxCommitTimeMS` of commits.
    pub operation_timeout: Duration,
}

impl MongoOptions {
    pub(crate) fn new(connection: ConnectionSecret, tls: MongoTls) -> Self {
        Self {
            connection,
            database: None,
            tls,
            max_pool_size: 16,
            min_pool_size: 0,
            max_concurrent_operations: 16,
            server_selection_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(10),
            max_idle_time: Duration::from_secs(300),
            operation_timeout: Duration::from_secs(15),
        }
    }
}

fn valid_database(name: &str) -> bool {
    !name.is_empty() && name.len() <= 63 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn is_local(address: &ServerAddress) -> bool {
    match address {
        ServerAddress::Tcp { host, .. } => matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1" | "[::1]"),
        // Unix domain sockets are local by construction.
        _ => true,
    }
}

async fn client_options(options: &MongoOptions) -> Result<(ClientOptions, String), StoreError> {
    // The parse error may quote the URI; it is replaced by a fixed message.
    let mut parsed = ClientOptions::parse(options.connection.expose())
        .await
        .map_err(|_| StoreError::Config("invalid MongoDB connection string"))?;
    let database = match options.database.clone().or_else(|| parsed.default_database.clone()) {
        Some(name) if valid_database(&name) => name,
        Some(_) => return Err(StoreError::Config("invalid MongoDB database name")),
        None => return Err(StoreError::Config("MongoDB database name is missing")),
    };
    let client_cert = match &parsed.tls {
        Some(Tls::Enabled(tls)) => tls.cert_key_file_path.clone(),
        _ => None,
    };
    parsed.tls = Some(match &options.tls {
        MongoTls::VerifyFull { ca_file } => {
            let mut tls = TlsOptions::default();
            tls.ca_file_path = ca_file.clone();
            tls.cert_key_file_path = client_cert;
            // With the rustls backend host name verification cannot be disabled; certificates are verified against
            // the CA file or the bundled web PKI roots.
            tls.allow_invalid_certificates = Some(false);
            Tls::Enabled(tls)
        }
        MongoTls::LocalPlaintext => {
            if !parsed.hosts.iter().all(is_local) {
                return Err(StoreError::Config("plaintext MongoDB is allowed only for a local server"));
            }
            Tls::Disabled
        }
    });
    parsed.app_name = Some("basal-batches".to_owned());
    parsed.max_pool_size = Some(options.max_pool_size.max(1));
    parsed.min_pool_size = Some(options.min_pool_size.min(options.max_pool_size.max(1)));
    parsed.server_selection_timeout = Some(options.server_selection_timeout);
    parsed.connect_timeout = Some(options.connect_timeout);
    parsed.max_idle_time = Some(options.max_idle_time);
    parsed.retry_reads = Some(true);
    parsed.retry_writes = Some(true);
    parsed.read_concern = Some(ReadConcern::majority());
    parsed.write_concern = Some(majority_journal());
    parsed.selection_criteria = Some(SelectionCriteria::ReadPreference(ReadPreference::Primary));
    Ok((parsed, database))
}

/// Topology able to run multi-document transactions with update pipelines.
async fn check_topology(core: &Core) -> Result<(), StoreError> {
    let hello = core.db.run_command(doc! { "hello": 1_i32 }).await.map_err(|e| redact(&e))?;
    let replica_set = matches!(hello.get("setName"), Some(Bson::String(_)));
    let mongos = matches!(hello.get("msg"), Some(Bson::String(msg)) if msg == "isdbgrid");
    let wire = driver::opt_int(&hello, "maxWireVersion")?.unwrap_or(0);
    if !(replica_set || mongos) {
        return Err(StoreError::Config("MongoDB batch storage requires a replica set or sharded cluster"));
    }
    // The owner-first scheduler uses $lookup inside a transaction. MongoDB permits sharded lookup targets in
    // transactions starting with 8.0 (wire version 25); older mongos topologies must fail before accepting work.
    if mongos && wire < 25 {
        return Err(StoreError::Config("MongoDB sharded batch storage requires MongoDB 8.0 or newer"));
    }
    if wire < 8 {
        return Err(StoreError::Config("MongoDB batch storage requires MongoDB 4.2 or newer"));
    }
    Ok(())
}

/// Native MongoDB store; cheap to clone.
#[derive(Clone)]
pub(super) struct MongoStore {
    core: Arc<Core>,
    supervisor: Supervisor,
}

impl std::fmt::Debug for MongoStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MongoStore { .. }")
    }
}

impl MongoStore {
    async fn open(options: &MongoOptions, settings: StoreSettings) -> Result<Self, StoreError> {
        let (client_options, database) = client_options(options).await?;
        let client = mongodb::Client::with_options(client_options).map_err(|e| redact(&e))?;
        let db = client.database(&database);
        let core = Arc::new(Core { client, db, settings, op_timeout: options.operation_timeout });
        let store = Self { core, supervisor: Supervisor::new(options.max_concurrent_operations) };
        if let Err(error) = store.run(|core| async move { check_topology(&core).await }).await {
            store.close().await;
            return Err(error);
        }
        Ok(store)
    }

    /// Runtime connection: topology checked, schema must be current.
    pub(super) async fn connect(options: &MongoOptions, settings: StoreSettings) -> Result<Self, StoreError> {
        let store = Self::open(options, settings).await?;
        match store.schema_state().await {
            Ok(SchemaState::Current { .. }) => Ok(store),
            Ok(other) => {
                store.close().await;
                Err(StoreError::Schema(other))
            }
            Err(error) => {
                store.close().await;
                Err(error)
            }
        }
    }

    /// Explicit migration with the migration account.
    pub(super) async fn migrate(options: &MongoOptions) -> Result<SchemaState, StoreError> {
        let store = Self::open(options, StoreSettings::default()).await?;
        let result = store.run(|core| async move { schema::migrate(&core).await }).await;
        store.close().await;
        result
    }

    /// Stop admission of new operations, wait for in-flight operations, then shut the client down.
    pub(super) async fn close(&self) {
        self.supervisor.close().await;
        self.core.client.clone().shutdown().await;
    }

    async fn run<R, F, Fut>(&self, operation: F) -> Result<R, StoreError>
    where
        R: Send + 'static,
        F: FnOnce(Arc<Core>) -> Fut,
        Fut: std::future::Future<Output = Result<R, StoreError>> + Send + 'static,
    {
        self.supervisor.run(operation(self.core.clone())).await
    }

    pub(super) fn capabilities(&self) -> StoreCapabilities {
        StoreCapabilities {
            backend: BackendKind::Mongodb,
            transactional_ddl: false,
            row_locks: true,
            skip_locked: false,
            server_clock: true,
        }
    }

    pub(super) async fn schema_state(&self) -> Result<SchemaState, StoreError> {
        self.run(|core| async move { schema::schema_state(&core).await }).await
    }

    pub(super) async fn server_version(&self) -> Result<String, StoreError> {
        self.run(|core| async move { schema::server_version(&core).await }).await
    }

    pub(super) async fn counters(&self) -> Result<Counters, StoreError> {
        self.run(|core| async move { admin::counters(&core).await }).await
    }

    pub(super) async fn backlog(&self, limit: u32) -> Result<Vec<BacklogEntry>, StoreError> {
        self.run(move |core| async move { admin::backlog(&core, limit).await }).await
    }

    pub(super) async fn quota_drift(&self, owner_limit: u32) -> Result<Vec<QuotaDrift>, StoreError> {
        self.run(move |core| async move { admin::quota_drift(&core, owner_limit).await }).await
    }

    pub(super) async fn quota_usage(&self, owner: &OwnerId) -> Result<QuotaUsage, StoreError> {
        let owner = owner.clone();
        self.run(move |core| async move { admin::quota_usage_of(&core, &owner).await }).await
    }

    pub(super) async fn install_catalog(&self, spec: &CatalogSpec) -> Result<InstallOutcome, StoreError> {
        let spec = spec.clone();
        self.run(move |core| async move { admin::install_catalog(&core, &spec).await }).await
    }

    pub(super) async fn install_quota_policy(&self, spec: &QuotaPolicySpec) -> Result<InstallOutcome, StoreError> {
        let spec = spec.clone();
        self.run(move |core| async move { admin::install_quota_policy(&core, &spec).await }).await
    }

    pub(super) async fn latest_catalog(&self) -> Result<Option<Catalog>, StoreError> {
        self.run(|core| async move { admin::latest_catalog(&core).await }).await
    }

    pub(super) async fn catalog_revision(&self, revision: Revision) -> Result<Option<Catalog>, StoreError> {
        self.run(move |core| async move { admin::catalog_revision(&core, revision).await }).await
    }

    pub(super) async fn active_quota_policy(&self) -> Result<Option<QuotaPolicy>, StoreError> {
        self.run(|core| async move { admin::active_quota_policy(&core).await }).await
    }

    // API users and keys

    pub(super) async fn create_api_user(&self, id: &OwnerId, name: &str) -> Result<Option<ApiUserRecord>, StoreError> {
        let (id, name) = (id.clone(), name.to_owned());
        self.run(move |core| async move { keys::create_user(&core, &id, &name).await }).await
    }

    pub(super) async fn list_api_users(
        &self,
        include_disabled: bool,
        limit: u32,
    ) -> Result<Vec<ApiUserRecord>, StoreError> {
        self.run(move |core| async move { keys::list_users(&core, include_disabled, limit).await }).await
    }

    pub(super) async fn set_api_user_disabled(
        &self,
        id: &OwnerId,
        disabled: bool,
    ) -> Result<ApiUserRecord, StoreError> {
        let id = id.clone();
        self.run(move |core| async move { keys::set_user_disabled(&core, &id, disabled).await }).await
    }

    pub(super) async fn create_api_key(&self, key: &NewApiKey) -> Result<ApiKeyRecord, StoreError> {
        let key = key.clone();
        self.run(move |core| async move { keys::create(&core, &key).await }).await
    }

    pub(super) async fn list_api_keys(
        &self,
        user: Option<&OwnerId>,
        include_revoked: bool,
        limit: u32,
    ) -> Result<Vec<ApiKeyRecord>, StoreError> {
        let user = user.cloned();
        self.run(move |core| async move { keys::list(&core, user.as_ref(), include_revoked, limit).await }).await
    }

    pub(super) async fn lookup_api_key(&self, id: &ApiKeyId) -> Result<Option<ApiKeyLookup>, StoreError> {
        let id = id.clone();
        self.run(move |core| async move { keys::lookup(&core, &id).await }).await
    }

    pub(super) async fn revoke_api_key(&self, id: &ApiKeyId) -> Result<ApiKeyRecord, StoreError> {
        let id = id.clone();
        self.run(move |core| async move { keys::revoke(&core, &id).await }).await
    }

    pub(super) async fn expire_api_key(&self, id: &ApiKeyId, within: Duration) -> Result<ApiKeyRecord, StoreError> {
        let id = id.clone();
        self.run(move |core| async move { keys::expire_within(&core, &id, within).await }).await
    }

    pub(super) async fn touch_api_key(&self, id: &ApiKeyId, interval: Duration) -> Result<(), StoreError> {
        let id = id.clone();
        self.run(move |core| async move { keys::touch(&core, &id, interval).await }).await
    }

    // Files

    pub(super) async fn reserve_upload(&self, upload: &NewUpload) -> Result<UploadSession, StoreError> {
        let upload = upload.clone();
        self.run(move |core| async move { files::reserve_upload(&core, &upload).await }).await
    }

    pub(super) async fn complete_upload(
        &self,
        owner: &OwnerId,
        session: &SessionId,
        object: StoredObject,
    ) -> Result<FileRecord, StoreError> {
        let (owner, session) = (owner.clone(), session.clone());
        self.run(move |core| async move { files::complete_upload(&core, &owner, &session, object).await }).await
    }

    pub(super) async fn abort_upload(&self, owner: &OwnerId, session: &SessionId) -> Result<(), StoreError> {
        let (owner, session) = (owner.clone(), session.clone());
        self.run(move |core| async move { files::abort_upload(&core, &owner, &session).await }).await
    }

    pub(super) async fn get_file(&self, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { files::get_file(&core, &owner, &id).await }).await
    }

    pub(super) async fn list_files(
        &self,
        owner: &OwnerId,
        purpose: Option<FilePurpose>,
        page: &PageRequest<FileId>,
    ) -> Result<Page<FileRecord>, StoreError> {
        let (owner, page) = (owner.clone(), page.clone());
        self.run(move |core| async move { files::list_files(&core, &owner, purpose, &page).await }).await
    }

    pub(super) async fn delete_file(&self, owner: &OwnerId, id: &FileId) -> Result<FileRecord, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { files::delete_file(&core, &owner, &id).await }).await
    }

    pub(super) async fn expire_files(&self, limit: u32) -> Result<u64, StoreError> {
        self.run(move |core| async move { files::expire_files(&core, limit).await }).await
    }

    pub(super) async fn file_gc_candidates(&self, limit: u32) -> Result<Vec<FileGcCandidate>, StoreError> {
        self.run(move |core| async move { files::file_gc_candidates(&core, limit).await }).await
    }

    pub(super) async fn purge_file(&self, owner: &OwnerId, id: &FileId) -> Result<bool, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { files::purge_file(&core, &owner, &id).await }).await
    }

    pub(super) async fn session_gc_candidates(&self, limit: u32) -> Result<Vec<SessionGcCandidate>, StoreError> {
        self.run(move |core| async move { files::session_gc_candidates(&core, limit).await }).await
    }

    pub(super) async fn release_session(&self, owner: &OwnerId, session: &SessionId) -> Result<bool, StoreError> {
        let (owner, session) = (owner.clone(), session.clone());
        self.run(move |core| async move { files::release_session(&core, &owner, &session).await }).await
    }

    // Batches

    pub(super) async fn create_batch(
        &self,
        owner: &OwnerId,
        idempotency: Option<&Idempotency>,
        spec: &NewBatch,
    ) -> Result<Created, StoreError> {
        let (owner, idempotency, spec) = (owner.clone(), idempotency.cloned(), spec.clone());
        self.run(move |core| async move { lifecycle::create_batch(&core, &owner, idempotency.as_ref(), &spec).await })
            .await
    }

    pub(super) async fn get_batch(&self, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { lifecycle::get_batch(&core, &owner, &id).await }).await
    }

    pub(super) async fn list_batches(
        &self,
        owner: &OwnerId,
        page: &PageRequest<BatchId>,
    ) -> Result<Page<BatchRecord>, StoreError> {
        let (owner, page) = (owner.clone(), page.clone());
        self.run(move |core| async move { lifecycle::list_batches(&core, &owner, &page).await }).await
    }

    pub(super) async fn batch_errors(&self, owner: &OwnerId, id: &BatchId) -> Result<Vec<BatchIssue>, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { lifecycle::batch_errors(&core, &owner, &id).await }).await
    }

    pub(super) async fn request_cancel(&self, owner: &OwnerId, id: &BatchId) -> Result<BatchRecord, StoreError> {
        let (owner, id) = (owner.clone(), id.clone());
        self.run(move |core| async move { lifecycle::request_cancel(&core, &owner, &id).await }).await
    }

    pub(super) async fn due_expirations(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        self.run(move |core| async move { lifecycle::due_expirations(&core, limit).await }).await
    }

    pub(super) async fn expire_batch(&self, batch: &BatchRef) -> Result<BatchRecord, StoreError> {
        let batch = batch.clone();
        self.run(move |core| async move { lifecycle::expire_batch(&core, &batch).await }).await
    }

    pub(super) async fn draining_batches(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        self.run(move |core| async move { lifecycle::draining_batches(&core, limit).await }).await
    }

    pub(super) async fn advance_drain(&self, batch: &BatchRef, limit: u32) -> Result<DrainProgress, StoreError> {
        let batch = batch.clone();
        self.run(move |core| async move { lifecycle::advance_drain(&core, &batch, limit).await }).await
    }

    // Discovery and validation

    pub(super) async fn claim_discovery(&self, worker: &WorkerId) -> Result<Option<DiscoveryLease>, StoreError> {
        let worker = worker.clone();
        self.run(move |core| async move { ingest::claim_discovery(&core, &worker).await }).await
    }

    pub(super) async fn resolve_target(
        &self,
        token: &ClaimToken,
        model: Option<&ModelName>,
    ) -> Result<TargetResolution, StoreError> {
        let (token, model) = (token.clone(), model.cloned());
        self.run(move |core| async move { ingest::resolve_target(&core, &token, model.as_ref()).await }).await
    }

    pub(super) async fn claim_validation(
        &self,
        worker: &WorkerId,
        targets: &[WorkerTarget],
    ) -> Result<Option<ValidationLease>, StoreError> {
        let (worker, targets) = (worker.clone(), targets.to_vec());
        self.run(move |core| async move { ingest::claim_validation(&core, &worker, &targets).await }).await
    }

    pub(super) async fn append_validated(
        &self,
        token: &ClaimToken,
        records: &[ValidatedRecord],
    ) -> Result<AppendOutcome, StoreError> {
        let (token, records) = (token.clone(), records.to_vec());
        self.run(move |core| async move { ingest::append_validated(&core, &token, &records).await }).await
    }

    pub(super) async fn seal_input(
        &self,
        token: &ClaimToken,
        request: &SealRequest,
    ) -> Result<BatchRecord, StoreError> {
        let (token, request) = (token.clone(), request.clone());
        self.run(move |core| async move { ingest::seal_input(&core, &token, &request).await }).await
    }

    pub(super) async fn fail_validation(
        &self,
        token: &ClaimToken,
        issues: &[BatchIssue],
        truncated: bool,
    ) -> Result<BatchRecord, StoreError> {
        let (token, issues) = (token.clone(), issues.to_vec());
        self.run(move |core| async move { ingest::fail_validation(&core, &token, &issues, truncated).await }).await
    }

    // Execution

    pub(super) async fn claim_items(&self, request: &ClaimRequest) -> Result<ClaimOutcome, StoreError> {
        let request = request.clone();
        self.run(move |core| async move { claims::claim_items(&core, &request).await }).await
    }

    pub(super) async fn renew_claim(&self, token: &ClaimToken) -> Result<LeaseUpdate<Renewal>, StoreError> {
        let token = token.clone();
        self.run(move |core| async move { claims::renew_claim(&core, &token).await }).await
    }

    pub(super) async fn release_claim_with_cost(
        &self,
        token: &ClaimToken,
        reason: ReleaseReason,
        costs: &std::collections::BTreeMap<i64, i64>,
    ) -> Result<LeaseUpdate<ReleaseReport>, StoreError> {
        let (token, costs) = (token.clone(), costs.clone());
        self.run(move |core| async move { claims::release_claim(&core, &token, reason, &costs).await }).await
    }

    pub(super) async fn mark_solo(&self, token: &ClaimToken, lines: &[u64]) -> Result<LeaseUpdate<()>, StoreError> {
        let (token, lines) = (token.clone(), lines.to_vec());
        self.run(move |core| async move { claims::mark_solo(&core, &token, &lines).await }).await
    }

    pub(super) async fn begin_result_upload(
        &self,
        token: &ClaimToken,
        line_no: u64,
        artifact: &ArtifactRef,
    ) -> Result<SessionId, StoreError> {
        let (token, artifact) = (token.clone(), artifact.clone());
        self.run(move |core| async move { claims::begin_result_upload(&core, &token, line_no, &artifact).await }).await
    }

    pub(super) async fn recover_expired_claims(&self, limit: u32) -> Result<u64, StoreError> {
        self.run(move |core| async move { claims::recover_expired_claims(&core, limit).await }).await
    }

    pub(super) async fn commit_results(
        &self,
        operation: &OperationId,
        token: &ClaimToken,
        results: &[ItemResult],
        digest: Sha256Digest,
    ) -> Result<CommitReport, StoreError> {
        let (operation, token, results) = (operation.clone(), token.clone(), results.to_vec());
        self.run(move |core| async move { commit::commit_results(&core, &operation, &token, &results, digest).await })
            .await
    }

    pub(super) async fn owners_with_pending_settlements(&self, limit: u32) -> Result<Vec<OwnerId>, StoreError> {
        self.run(move |core| async move { commit::owners_with_pending_settlements(&core, limit).await }).await
    }

    pub(super) async fn settle_pending(&self, owner: &OwnerId, limit: u32) -> Result<AccountingProgress, StoreError> {
        let owner = owner.clone();
        self.run(move |core| async move { commit::settle_pending(&core, &owner, limit).await }).await
    }

    // Finalization, cleanup and retention

    pub(super) async fn claim_finalization(
        &self,
        worker: &WorkerId,
        keys: &ExportKeys,
    ) -> Result<FinalizationClaim, StoreError> {
        let (worker, keys) = (worker.clone(), keys.clone());
        self.run(move |core| async move { finalize::claim_finalization(&core, &worker, &keys).await }).await
    }

    pub(super) async fn adopt_export(
        &self,
        token: &ClaimToken,
        previous: &SessionId,
        keys: &ExportKeys,
    ) -> Result<FinalizationLease, StoreError> {
        let (token, previous, keys) = (token.clone(), previous.clone(), keys.clone());
        self.run(move |core| async move { finalize::adopt_export(&core, &token, &previous, &keys).await }).await
    }

    pub(super) async fn read_export_items(
        &self,
        token: &ClaimToken,
        after_line: u64,
        limit: u32,
    ) -> Result<Vec<ExportItem>, StoreError> {
        let token = token.clone();
        self.run(move |core| async move { finalize::read_export_items(&core, &token, after_line, limit).await }).await
    }

    pub(super) async fn publish_files(
        &self,
        operation: &OperationId,
        token: &ClaimToken,
        request: &PublishRequest,
        digest: Sha256Digest,
    ) -> Result<BatchRecord, StoreError> {
        let (operation, token, request) = (operation.clone(), token.clone(), request.clone());
        self.run(move |core| async move { finalize::publish_files(&core, &operation, &token, &request, digest).await })
            .await
    }

    pub(super) async fn cleanup_batches(&self, limit: u32) -> Result<Vec<BatchRef>, StoreError> {
        self.run(move |core| async move { finalize::cleanup_batches(&core, limit).await }).await
    }

    pub(super) async fn cleanup_page(&self, batch: &BatchRef, limit: u32) -> Result<CleanupPage, StoreError> {
        let batch = batch.clone();
        self.run(move |core| async move { finalize::cleanup_page(&core, &batch, limit).await }).await
    }

    pub(super) async fn cleanup_commit(
        &self,
        batch: &BatchRef,
        through_line_no: Option<u64>,
    ) -> Result<bool, StoreError> {
        let batch = batch.clone();
        self.run(move |core| async move { finalize::cleanup_commit(&core, &batch, through_line_no).await }).await
    }

    pub(super) async fn purge_metadata(&self, limit: u32) -> Result<u64, StoreError> {
        self.run(move |core| async move { finalize::purge_metadata(&core, limit).await }).await
    }
}
