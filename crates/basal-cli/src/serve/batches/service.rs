//! The batch service of one process: startup, shared state, store connection, health and shutdown.

use std::future::Future;
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::artifacts::{ArtifactError, ArtifactStore};
use super::budget::HostPool;
use super::config::{BackgroundQualification, BatchSettings, ConfigError, Profile};
use super::dto::{ApiError, Throttle};
use super::planning::{PlanLimits, PlanningPool};
use super::profile::{ModelProfile, ResolvedProfile};
use super::store::{
    self, CatalogEntry, CatalogSpec, ModelName, QuotaPolicySpec, Store, StoreError, StoreSettings, WorkerId,
    WorkerTarget,
};
use crate::database::{Account, ConnectOptions, Database, DatabaseError, Target};

/// Own handles throughout shutdown. Cancellation of shutdown aborts supervisors instead of detaching them.
struct StoppingSupervisors(Vec<JoinHandle<()>>);

impl Drop for StoppingSupervisors {
    fn drop(&mut self) {
        for task in &self.0 {
            task.abort();
        }
    }
}

/// Starting the service failed for a local reason (configuration, artifact directory, profiles).
#[derive(Debug, thiserror::Error)]
pub(crate) enum StartError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error("batch artifacts: {0}")]
    Artifacts(#[source] ArtifactError),
    #[error("batch planning threads")]
    Planning(#[source] std::io::Error),
}

/// Availability of one dependency, for health and admission.
#[derive(Clone, Debug, Default)]
struct Component {
    available: bool,
    reason: Option<String>,
}

impl Component {
    fn up() -> Self {
        Self { available: true, reason: None }
    }

    fn down(reason: impl Into<String>) -> Self {
        Self { available: false, reason: Some(reason.into()) }
    }
}

#[derive(Debug, Default)]
struct Health {
    storage: Component,
    artifacts: Component,
    /// The installed quota policy equals this host's configuration.
    policy: Component,
    /// Fingerprints are computed (profiles are usable).
    profiles_ready: bool,
    profile_errors: Vec<String>,
    /// Latest catalog revision and whether it lists every profile of this host.
    catalog: Option<(i64, bool)>,
    /// Reported by the background executor (`BatchService::set_execution_health`).
    execution: Option<Value>,
}

/// State shared by the routes and the supervisors.
pub(super) struct Shared {
    pub(super) settings: BatchSettings,
    /// The process database; its schema is migrated (when configured) before the store connects.
    database: Arc<Database>,
    pub(super) store_settings: StoreSettings,
    pub(super) policy_spec: QuotaPolicySpec,
    pub(super) default_model: String,
    pub(super) artifacts: ArtifactStore,
    store: RwLock<Option<Store>>,
    profiles: RwLock<Vec<Arc<ResolvedProfile>>>,
    pub(super) worker: WorkerId,
    pub(super) uploads: Arc<Semaphore>,
    pub(super) downloads: Arc<Semaphore>,
    /// Store mutations started by HTTP handlers; each runs in its own task holding a permit until the database
    /// answered, so a disconnected client never drops a transaction half-way.
    ops: Arc<Semaphore>,
    pub(super) host_pool: Arc<HostPool>,
    pub(super) planning: PlanningPool,
    pub(super) stop: CancellationToken,
    health: Mutex<Health>,
}

impl Shared {
    /// The store when connected and usable.
    pub(super) fn store(&self) -> Option<Store> {
        self.store.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    pub(super) fn profiles(&self) -> Vec<Arc<ResolvedProfile>> {
        self.profiles.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Validation targets: profiles that plan bounded records. The local profile also claims the others, because no
    /// other host can validate their batches; validation then fails them with `unbounded_feature`.
    pub(super) fn targets(&self) -> Vec<WorkerTarget> {
        let profiles = self.profiles();
        match self.settings.profile {
            Profile::Local => profiles.iter().map(|p| p.claim_target()).collect(),
            Profile::Production => profiles.iter().filter_map(|p| p.target()).collect(),
        }
    }

    pub(super) fn profile_for(&self, fingerprint: &store::Fingerprint) -> Option<Arc<ResolvedProfile>> {
        self.profiles().into_iter().find(|p| &p.fingerprint == fingerprint)
    }

    /// The dependencies are ready: store and artifacts usable, the installed policy equal to this configuration,
    /// not stopping. Independent of the operator flags.
    pub(super) fn ready(&self) -> bool {
        let h = self.health.lock().unwrap_or_else(PoisonError::into_inner);
        h.storage.available && h.artifacts.available && h.policy.available && !self.stop.is_cancelled()
    }

    /// New uploads and batches are admitted: ready and `batches.accepting`.
    pub(super) fn admission_open(&self) -> bool {
        self.settings.accepting && self.ready()
    }

    /// New discovery, validation and execution claims may be taken: ready and `batches.processing`.
    pub(super) fn processing_open(&self) -> bool {
        self.settings.processing && self.ready()
    }

    /// The store for a request, or 503.
    pub(super) fn store_or_503(&self) -> Result<Store, ApiError> {
        self.store().ok_or_else(|| ApiError::unavailable(super::convert::RETRY_AFTER_S))
    }

    /// The store for an admission step (upload, create), or 503.
    pub(super) fn admitting_store(&self) -> Result<Store, ApiError> {
        if !self.admission_open() {
            return Err(ApiError::unavailable(super::convert::RETRY_AFTER_S));
        }
        self.store_or_503()
    }

    /// Run a store mutation in its own task (it completes even when the caller is dropped), bounded by the
    /// operation permits (429 when all are in use).
    pub(super) async fn mutate<T, F>(&self, op: F) -> Result<Result<T, StoreError>, ApiError>
    where
        T: Send + 'static,
        F: Future<Output = Result<T, StoreError>> + Send + 'static,
    {
        let permit = Arc::clone(&self.ops)
            .try_acquire_owned()
            .map_err(|_| ApiError::throttled(Throttle::Concurrency, super::convert::RETRY_AFTER_S))?;
        let task = tokio::spawn(async move {
            let _permit = permit;
            op.await
        });
        // At shutdown the handler answers at once; the operation still completes (or rolls back) in its task.
        tokio::select! {
            joined = task => joined.map_err(|_| ApiError::internal()),
            () = self.stop.cancelled() => Err(ApiError::unavailable(super::convert::RETRY_AFTER_S)),
        }
    }

    pub(super) fn set_artifacts(&self, component: Result<(), ArtifactError>) {
        let mut h = self.health.lock().unwrap_or_else(PoisonError::into_inner);
        h.artifacts = match component {
            Ok(()) => Component::up(),
            Err(e) => Component::down(e.to_string()),
        };
    }

    fn set_storage(&self, store: Option<Store>, component: Component) {
        *self.store.write().unwrap_or_else(PoisonError::into_inner) = store;
        self.health.lock().unwrap_or_else(PoisonError::into_inner).storage = component;
    }

    fn set_policy(&self, component: Component) {
        self.health.lock().unwrap_or_else(PoisonError::into_inner).policy = component;
    }

    /// Compare the installed policy and catalog with this host (also run periodically by the lifecycle).
    pub(super) async fn check_policy(&self, store: &Store) {
        let policy = match store.active_quota_policy().await {
            Ok(Some(p)) if p.spec_sha256 == store::policy_digest(&self.policy_spec) => Component::up(),
            Ok(Some(_)) => Component::down(
                "the installed quota policy differs from this configuration; install it with `basal batches install`",
            ),
            Ok(None) => Component::down("no quota policy is installed; run `basal batches install`"),
            Err(e) => Component::down(format!("reading the quota policy: {e}")),
        };
        self.set_policy(policy);
        let profiles = self.profiles();
        if let Ok(Some(catalog)) = store.latest_catalog().await {
            let covers = !profiles.is_empty()
                && profiles.iter().all(|p| {
                    catalog.spec.entries.iter().any(|e| {
                        e.model_name.as_str() == p.name
                            && e.fingerprint == p.fingerprint
                            && e.planner_version == p.planner_version
                    })
                });
            self.health.lock().unwrap_or_else(PoisonError::into_inner).catalog = Some((catalog.revision.0, covers));
        }
    }
}

/// The batch service; see the module docs of `serve::batches`.
pub(crate) struct BatchService {
    shared: Arc<Shared>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl BatchService {
    /// Start the service inside the Tokio runtime, before the HTTP server. Only local problems fail (configuration,
    /// the artifact directory, planning threads). An unreachable database or remote storage starts the
    /// service degraded: the new routes answer 503 until it connects. The database's automatic schema migration
    /// (`database.auto_migrate`, [`Database::prepare`]) finishes before the store is published to workers. There is
    /// never a fallback to another database.
    pub(crate) async fn start(
        settings: BatchSettings,
        database: Arc<Database>,
        profiles: Vec<ModelProfile>,
        default_model: String,
    ) -> Result<Arc<Self>, StartError> {
        // Both rustls providers are in the dependency graph; the drivers need one process default.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let policy_spec = settings.quota.to_store_spec(&settings.limits)?;
        let buffer = usize::try_from(settings.host.upload_buffer_bytes).unwrap_or(1 << 20);
        let artifacts = ArtifactStore::open(&settings.artifacts, buffer).await.map_err(StartError::Artifacts)?;
        let limits = PlanLimits::new(
            settings.host.plan_slot_bytes,
            settings.host.exec_pool_bytes,
            u64::from(settings.limits.max_records),
            settings.limits.max_file_bytes,
        );
        let planning = PlanningPool::start(settings.host.planning_workers, limits).map_err(StartError::Planning)?;
        let ops = match &database.settings().target {
            Target::Remote(r) => r.pool.max_connections as usize * 2,
            Target::Sqlite { .. } => 16,
        };
        let shared = Arc::new(Shared {
            database,
            store_settings: settings.store_settings(),
            policy_spec,
            default_model,
            artifacts,
            store: RwLock::new(None),
            profiles: RwLock::new(Vec::new()),
            worker: WorkerId::new_random(),
            uploads: Arc::new(Semaphore::new(settings.host.upload_concurrency as usize)),
            downloads: Arc::new(Semaphore::new(settings.host.download_concurrency as usize)),
            ops: Arc::new(Semaphore::new(ops)),
            host_pool: Arc::new(HostPool::new(settings.host.exec_pool_bytes)),
            planning,
            stop: CancellationToken::new(),
            health: Mutex::new(Health::default()),
            settings,
        });
        let probe = shared.artifacts.probe().await;
        if let Err(e) = &probe {
            crate::warn!("batch artifacts unavailable: {e}");
        }
        shared.set_artifacts(probe);
        let service = Arc::new(Self { shared: Arc::clone(&shared), tasks: Mutex::new(Vec::new()) });
        let startup = tokio::spawn(connect(Arc::clone(&shared), profiles));
        let ingest = tokio::spawn(super::ingest::run(Arc::clone(&shared)));
        let lifecycle = tokio::spawn(super::lifecycle::run(Arc::clone(&shared)));
        let finalizer = tokio::spawn(super::lifecycle::run_finalizer(Arc::clone(&shared)));
        service.tasks.lock().unwrap_or_else(PoisonError::into_inner).extend([startup, ingest, lifecycle, finalizer]);
        Ok(service)
    }

    /// The unauthenticated Batch/Files routes and `GET /health/batches`; merge into the root router.
    pub(crate) fn router(self: &Arc<Self>) -> axum::Router<()> {
        super::api::router(Arc::clone(&self.shared))
    }

    /// Close admission, stop the supervisors between operations, wait at most `leases.shutdown_grace_ms` for them,
    /// stop the planning threads and close the store. Unfinished leases expire and are recovered by any host.
    pub(crate) async fn shutdown(&self) {
        self.shared.stop.cancel();
        let mut tasks =
            StoppingSupervisors(std::mem::take(&mut *self.tasks.lock().unwrap_or_else(PoisonError::into_inner)));
        let grace = Duration::from_millis(self.shared.settings.leases.shutdown_grace_ms);
        let joined = tokio::time::timeout(grace, async {
            while let Some(task) = tasks.0.last_mut() {
                if let Err(e) = task.await {
                    crate::warn!("batch supervisor stopped: {e}");
                }
                tasks.0.pop();
            }
        })
        .await;
        if joined.is_err() {
            crate::warn!("batch supervisors did not stop within {} ms", grace.as_millis());
            for task in &tasks.0 {
                task.abort();
            }
            // Abort drops export writers and their LeaseKeeper; wait for that drop before closing the store.
            // Store-owned transaction supervisors finish/reconcile any already submitted native operation.
            while let Some(task) = tasks.0.pop() {
                let _ = task.await;
            }
        }
        self.shared.planning.shutdown().await;
        if let Some(store) = self.shared.store() {
            store.close().await;
        }
    }

    /// The store while connected (for the background executor).
    pub(crate) fn store(&self) -> Option<Store> {
        self.shared.store()
    }

    pub(crate) fn artifacts(&self) -> &ArtifactStore {
        &self.shared.artifacts
    }

    /// Fingerprinted profiles (empty until fingerprints are computed).
    pub(crate) fn profiles(&self) -> Vec<Arc<ResolvedProfile>> {
        self.shared.profiles()
    }

    pub(crate) fn worker_id(&self) -> &WorkerId {
        &self.shared.worker
    }

    pub(crate) fn settings(&self) -> &BatchSettings {
        &self.shared.settings
    }

    /// Cancelled when the service stops (executor tasks end their microseries and release their claims).
    pub(crate) fn stopping(&self) -> CancellationToken {
        self.shared.stop.clone()
    }

    /// The executor feeder may take a new execution claim (dependencies ready and `batches.processing`). Check it
    /// before every `claim_items`; a `false` takes no new claim, while claims already held finish, commit and
    /// release cooperatively (only shutdown stops them). Independent of `admission_open`.
    pub(crate) fn processing_open(&self) -> bool {
        self.shared.processing_open()
    }

    pub(crate) fn host_pool(&self) -> &Arc<HostPool> {
        &self.shared.host_pool
    }

    pub(crate) fn planning(&self) -> &PlanningPool {
        &self.shared.planning
    }

    /// Health of the background executor, shown under `execution` in `/health/batches` (aggregates only: no
    /// owners, batch ids or client data).
    pub(crate) fn set_execution_health(&self, health: Value) {
        self.shared.health.lock().unwrap_or_else(PoisonError::into_inner).execution = Some(health);
    }
}

pub(super) fn health(shared: &Shared) -> Value {
    let h = shared.health.lock().unwrap_or_else(PoisonError::into_inner);
    let component = |c: &Component| json!({"available": c.available, "reason": c.reason});
    let profiles: Vec<Value> = shared
        .profiles()
        .iter()
        .map(|p| {
            let qualification = match &p.qualification {
                BackgroundQualification::Disabled(reason) => json!({"qualified": false, "reason": reason}),
                BackgroundQualification::Qualified { evidence, composition, max_requests, .. } => json!({
                    "qualified": true, "evidence": evidence,
                    "composition": format!("{composition:?}").to_lowercase(), "max_requests": max_requests,
                }),
            };
            json!({
                "model": p.name,
                "fingerprint": p.fingerprint.as_str(),
                "planner_version": p.planner_version.as_str(),
                "validates": p.target().is_some(),
                "background_supported": p.background.is_supported(),
                "background_unsupported_reason": match &p.background {
                    basal_core::BackgroundSupport::Unsupported(r) => Some(r.as_str()),
                    basal_core::BackgroundSupport::Supported => None,
                },
                "qualification": qualification,
                "max_record_cost": p.max_record_cost,
                "cost_ns_per_token": p.cost_ns_per_token,
            })
        })
        .collect();
    let ready = h.storage.available && h.artifacts.available && h.policy.available && !shared.stop.is_cancelled();
    let accepting = ready && shared.settings.accepting;
    let processing = ready && shared.settings.processing;
    let status = if ready && h.profiles_ready {
        "ok"
    } else if h.storage.available {
        "degraded"
    } else {
        "unavailable"
    };
    json!({
        "status": status,
        "profile": match shared.settings.profile { Profile::Local => "local", Profile::Production => "production" },
        "storage": {"engine": shared.database.settings().engine_label(), "available": h.storage.available, "reason": h.storage.reason},
        "artifacts": {"backend": shared.artifacts.backend_label(), "available": h.artifacts.available, "reason": h.artifacts.reason},
        "policy": component(&h.policy),
        "ready": ready,
        "accepting": accepting,
        "processing": processing,
        "operator": {"accepting": shared.settings.accepting, "processing": shared.settings.processing},
        "profiles_ready": h.profiles_ready,
        "profile_errors": h.profile_errors,
        "catalog": h.catalog.map(|(revision, covers)| json!({"revision": revision, "lists_this_host": covers})),
        "profiles": profiles,
        "host_pool": {"capacity": shared.host_pool.capacity(), "free": shared.host_pool.free()},
        "execution_limits": {
            // Configured caps; the executor may use less (for example one microseries in flight).
            "microbatch_requests": shared.settings.execution.microbatch_requests,
            "microbatch_tokens": shared.settings.execution.microbatch_tokens,
            "lookahead_records": shared.settings.execution.lookahead_records,
            "commit_items": shared.settings.execution.commit_items,
            "commit_buffer_microbatches_max": shared.settings.execution.commit_buffer_microbatches,
            "max_attempts": shared.settings.execution.max_attempts,
            "pause_limit_ms": shared.settings.execution.pause_limit_ms,
        },
        "execution": h.execution,
    })
}

/// Startup task: fingerprint the profiles, connect the store (retrying with backoff), bootstrap or check the
/// installed policy and catalog.
async fn connect(shared: Arc<Shared>, profiles: Vec<ModelProfile>) {
    let limits = shared.planning.limits();
    let mut resolved = Vec::new();
    let mut errors = Vec::new();
    for profile in profiles {
        let name = profile.name().to_string();
        let started = std::time::Instant::now();
        match tokio::task::spawn_blocking(move || profile.resolve(&limits)).await {
            Ok(Ok(p)) => {
                crate::note!(
                    "batch profile {name}: fingerprint {} ({:.1} s)",
                    p.fingerprint.as_str(),
                    started.elapsed().as_secs_f64()
                );
                resolved.push(Arc::new(p));
            }
            Ok(Err(e)) => errors.push(error_chain(&e)),
            Err(_) => errors.push(format!("{name}: fingerprint task failed")),
        }
        if shared.stop.is_cancelled() {
            return;
        }
    }
    for e in &errors {
        crate::warn!("batch profile: {e}");
    }
    *shared.profiles.write().unwrap_or_else(PoisonError::into_inner) = resolved;
    {
        let mut h = shared.health.lock().unwrap_or_else(PoisonError::into_inner);
        h.profiles_ready = true;
        h.profile_errors = errors;
    }
    let mut backoff = Duration::from_secs(1);
    loop {
        match open_store(&shared).await {
            Ok(store) => {
                if let Err(e) = prepare(&shared, &store).await {
                    shared.set_storage(None, Component::down(format!("preparing the batch store: {e}")));
                    store.close().await;
                } else {
                    shared.check_policy(&store).await;
                    shared.set_storage(Some(store), Component::up());
                    crate::note!("batch store connected ({})", shared.database.settings().engine_label());
                    return;
                }
            }
            Err(Unavailable::Permanent(reason)) => {
                crate::warn!("batch store: {reason}");
                shared.set_storage(None, Component::down(reason));
                return;
            }
            Err(Unavailable::Retry(reason)) => {
                shared.set_storage(None, Component::down(reason));
            }
        }
        tokio::select! {
            () = shared.stop.cancelled() => return,
            () = tokio::time::sleep(backoff) => {}
        }
        backoff = (backoff * 2).min(Duration::from_secs(30));
    }
}

enum Unavailable {
    /// Not available in this build or configuration; no retry.
    Permanent(String),
    Retry(String),
}

fn error_chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        s.push_str(": ");
        s.push_str(&cause.to_string());
        source = cause.source();
    }
    s
}

async fn open_store(shared: &Shared) -> Result<Store, Unavailable> {
    let unavailable = |e: DatabaseError| {
        if e.is_permanent() {
            Unavailable::Permanent(error_chain(&e))
        } else {
            Unavailable::Retry(error_chain(&e))
        }
    };
    shared.database.prepare().await.map_err(unavailable)?;
    let options = shared.database.options(Account::Runtime).await.map_err(unavailable)?;
    connect_store(&options, shared.store_settings).await.map_err(|e| unavailable(e.into()))
}

/// Connect the Batch store with `options`. Refuses unless the schema is exactly current; never runs DDL and never
/// creates a SQLite file.
pub(crate) async fn connect_store(options: &ConnectOptions, settings: StoreSettings) -> Result<Store, StoreError> {
    match options {
        ConnectOptions::Sqlite(options) => Store::open_sqlite(options, settings).await,
        ConnectOptions::Postgresql(options) => Store::connect_postgresql(options, settings).await,
        ConnectOptions::Mssql(options) => Store::connect_mssql(options, settings).await,
        ConnectOptions::Mongodb(options) => Store::connect_mongodb(options, settings).await,
    }
}

/// The local catalog of this host's profiles (default model first when it is served).
pub(super) fn local_catalog(profiles: &[Arc<ResolvedProfile>], default_model: &str) -> Result<CatalogSpec, StoreError> {
    let entries = profiles
        .iter()
        .map(|p| {
            Ok(CatalogEntry {
                model_name: ModelName::parse(&p.name)?,
                fingerprint: p.fingerprint.clone(),
                planner_version: p.planner_version.clone(),
            })
        })
        .collect::<Result<Vec<_>, StoreError>>()?;
    let default = entries
        .iter()
        .find(|e| e.model_name.as_str() == default_model)
        .or_else(|| entries.first())
        .map(|e| e.model_name.clone())
        .ok_or(StoreError::Config("no model profile can be listed in the local catalog"))?;
    Ok(CatalogSpec { default_model: default, entries })
}

/// Local profile: install the catalog of this host's profiles when the latest revision does not list them, and the
/// local policy once (accepted batches keep their pinned revision). Production: nothing is installed at runtime.
async fn prepare(shared: &Shared, store: &Store) -> Result<(), StoreError> {
    if shared.settings.profile != Profile::Local {
        return Ok(());
    }
    let profiles = shared.profiles();
    if profiles.is_empty() {
        // Files still work; batches need a catalog, so create answers 503 until a profile is fingerprinted.
        if store.active_quota_policy().await?.is_none() {
            store.install_quota_policy(&shared.policy_spec).await?;
        }
        crate::warn!("batch catalog: no fingerprinted model profile; batches cannot be created");
        return Ok(());
    }
    let catalog = local_catalog(&profiles, &shared.default_model)?;
    let (catalog, policy) = store.bootstrap_local(&catalog, &shared.policy_spec).await?;
    crate::note!("batch catalog revision {}, quota policy revision {}", catalog.0, policy.0);
    Ok(())
}
