//! `basal serve`: HTTP server: POST /v1/systemone (TypeSafe System One), POST /v1/basal (upstream basal conventions),
//! GET /v1/models, GET /health, opt-in GET /metrics (Prometheus).
//!
//! One thread owns the GPU engine; HTTP handlers (tokio) send jobs to it over a channel. The scheduler plans arriving
//! requests (validation errors are answered at once), orders the waiting ones (`--schedule`: highest response ratio
//! next by default, or arrival order) and admits the first one and every other waiting request that fits
//! `max_batch_tokens` packed tokens. Idle requests start immediately; after a concurrent validated Omni Max batch a bounded
//! 10 ms collection window coalesces returning clients. They run
//! together with `Engine::run_plans`: the questions of all admitted requests share the forwards of each round (the
//! budget counts a state shared by several requests once). A request larger than the budget runs alone; requests
//! whose client disconnected while waiting are dropped. More than `max_inflight` waiting or running requests are
//! refused with 529 and `Retry-After: 1` (the TypeSafe API's status when it is overloaded).
//!
//! Long requests (`--long-tokens`, default 4096 packed tokens) go to a second lane: a second engine on the same GPU
//! sharing the weights, admitted the same way. The two lanes take turns on the GPU (`basal_core::gate`): when the
//! main lane has a batch, the long lane pauses after its current layer (once it has worked `--long-slice-ms`), the
//! main batch runs, and the long forward continues. Results do not depend on the lane or the pauses.
//!
//! Several models (`basal serve --config FILE`, [`crate::config`]) run in one process: each has its own engine,
//! queue and long lane, requests go to the model named in their `model` field (/v1/basal: the default model when the
//! field is missing). The engines of the models on one GPU (`device`) take turns on it through one gate, so short
//! batches of any of them go before long requests; models on different GPUs have separate gates and run at the same
//! time.
//!
//! A model may have replicas (`replica_devices`, CUDA): independent engines on distinct GPUs of one type under the one
//! model name, each with its own queue, scheduler, caches and long lane. A request goes to the available replica with
//! the fewest reserved (queued, planned, waiting or running) requests, ties in rotating order; the reservation lives in
//! the job until it is answered or dropped. The model-level metrics aggregate the replicas, the `basal_replica_*`
//! series split them.
//!
//! A model may instead be split by layers over several GPUs (`layer_split`, CUDA): one instance whose stages run one
//! after another, each on its GPU. It takes the gate of every GPU it uses, at most one at a time, so engines of other
//! models on one of those GPUs take turns with the stage there, and its long lane yields on every GPU.
//!
//! Timing headers (not part of the System One body): `x-basal-queue-ms` (arrival to the start of its batch),
//! `x-basal-compute-ms` (planning + forwards of its batch), `x-basal-batch-requests`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use basal_core::engine::{plan_response, PlanOutput, RequestPlan};
use basal_core::gate::Gate;
use basal_core::pack::TrieSize;
use basal_core::request::Dialect;
use basal_core::{Backend, DecideError, DeviceHealth, Engine};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, Semaphore};

pub(crate) mod batches;
pub(crate) mod collection;
mod decisions;
mod http;
mod metrics;
mod systemone;
use metrics::{LaneMetrics, Metrics, ReplicaMetrics, RequestModel};

pub(crate) const DEFAULT_MAX_LARGE_INFLIGHT: usize = 4;

/// Explicit dialect of /v1/systemone; never inferred from a request body.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SystemOneDialect {
    #[default]
    #[value(name = "typesafe")]
    #[serde(rename = "typesafe")]
    TypeSafe,
    Upstream,
}

pub struct ServeOptions {
    pub mcp: crate::mcp::Switch,
    pub systemone_dialect: SystemOneDialect,
    pub log_decisions: Option<std::path::PathBuf>,
    pub addr: SocketAddr,
    /// `release_date` of GET /v1/models (TypeSafe ModelMetadata; upstream reports its engine release date)
    pub release_date: String,
    /// Waiting + running requests of all models.
    pub max_inflight: usize,
    /// Large bodies and requests whose expanded input exceeds 128 KiB, across all models.
    pub max_large_inflight: usize,
    /// Model of /v1/basal requests without a "model" field (default: the first model).
    pub default_model: Option<String>,
    /// Least work of a long lane between two hand-overs of the GPU.
    pub long_slice_ms: u64,
    /// One log line per HTTP request.
    pub access_log: bool,
    /// Collect and expose Prometheus metrics; disabled by default.
    pub metrics: bool,
    /// The process database (`database:` or the local Batch default); `None` opens no database. Set whenever
    /// `batches` is.
    pub(crate) database: Option<std::sync::Arc<crate::database::Database>>,
    /// Checked configuration; `None` opens no batch storage and creates no background workers.
    pub(crate) batches: Option<batches::config::BatchSettings>,
}

/// One instance of a served model.
pub struct Replica<B: Backend> {
    pub engine: Engine<B>,
    /// GPU ordinals the engine was loaded on: one, or the stages of a layer split in order. Engines using the same
    /// ordinal share its gate.
    pub devices: Vec<usize>,
    /// How the main lane collects requests: the model profile's policy with the features of the engine's plan.
    pub scheduling: crate::models::SchedulingPolicy,
}

/// One served model: its instances and admission options.
pub struct ServedModel<B: Backend> {
    /// Deployment identity; the engines retain their source manifest names.
    pub name: String,
    /// At least one; several are replicas of one model on distinct GPUs, routed by load.
    pub replicas: Vec<Replica<B>>,
    pub max_batch_tokens: usize,
    pub(crate) collection: collection::CollectionPolicy,
    pub schedule: Schedule,
    /// Requests above this many packed tokens run in a second lane that yields the GPU between layers (0 = one lane).
    pub long_tokens: usize,
    /// Captured before moving the online engine; hashing and CPU planning are independent of GPU execution.
    pub(crate) batch_profile: Option<batches::ModelProfile>,
    pub(crate) batch_qualification: batches::config::BackgroundQualification,
}

struct Timing {
    queue_ms: f64,
    compute_ms: f64,
    batch: usize,
}

type LargeBodyPermit = Arc<tokio::sync::OwnedSemaphorePermit>;

#[derive(Clone)]
struct SystemOneDecodePermit(LargeBodyPermit);

type Reply = (std::result::Result<Value, DecideError>, Option<Timing>);

struct Job {
    body: Value,
    dialect: Dialect,
    arrived: Instant,
    reply: oneshot::Sender<Reply>,
    /// Lives with queued/processed media even if the HTTP handler is cancelled.
    _media_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    _large_body_permit: Option<LargeBodyPermit>,
    /// Bound queued/running jobs even after the HTTP handler disconnects.
    _job_permit: tokio::sync::OwnedSemaphorePermit,
    /// This job counted on its replica (none for a single instance without metrics), until answered or dropped.
    load: Option<Reservation>,
    /// Announced before queueing on the selected replica; lives through every round and the long lane.
    online_demand: Option<basal_core::gate::OnlineDemand>,
    /// Waiting for admission in either online lane; also released if the handler disconnects.
    queued: Option<Queued>,
}

impl Job {
    /// Release the reservation, then reply: a client that sends its next request as soon as it has the answer finds
    /// the replica's count already lowered. The reply is lost when the client is gone.
    fn answer(self, reply: Reply) {
        let Self { reply: tx, load, online_demand, queued, .. } = self;
        drop(load);
        drop(online_demand);
        drop(queued);
        let _ = tx.send(reply);
    }
}

/// Jobs reserved on one replica: queued, planned, waiting or running. A replica group routes by it; with metrics it
/// also drives the replica's inflight gauge.
struct ReplicaLoad {
    jobs: AtomicUsize,
    metrics: Option<ReplicaMetrics>,
}

/// One job counted in its replica's [`ReplicaLoad`]. Owned by the job, so the count drops wherever the job ends:
/// answered (after its batch even when the client is gone, or on a planning error), skipped in admission because its
/// client is gone, refused by a stopped worker, or dropped with a worker's queue (end, shutdown, panic).
struct Reservation(Arc<ReplicaLoad>);

impl Reservation {
    fn new(load: &Arc<ReplicaLoad>) -> Self {
        load.jobs.fetch_add(1, Ordering::Relaxed);
        if let Some(m) = &load.metrics {
            m.reserve();
        }
        Self(load.clone())
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.0.jobs.fetch_sub(1, Ordering::Relaxed);
        if let Some(m) = &self.0.metrics {
            m.release();
        }
    }
}

/// Queue of one instance of a model.
struct ReplicaRoute {
    tx: mpsc::UnboundedSender<Job>,
    /// GPU ordinals of the instance (one, or the stages of a layer split).
    devices: Vec<usize>,
    /// Device readiness reported by its engines (none when its backend does not report it).
    health: Vec<DeviceHealth>,
    /// Always set in a replica group; for a single instance only with metrics.
    load: Option<Arc<ReplicaLoad>>,
    /// Every gate of the instance, including all devices of a split.
    gates: Vec<Arc<Gate>>,
}

impl ReplicaRoute {
    /// Queue a job; a stopped worker hands it back untouched.
    fn submit(&self, mut job: Job) -> std::result::Result<(), Box<Job>> {
        job.online_demand = Some(basal_core::gate::OnlineDemand::register(self.gates.iter().cloned()));
        self.tx.send(job).map_err(|e| Box::new(e.0))?;
        if let Some(m) = self.load.as_ref().and_then(|l| l.metrics.as_ref()) {
            m.routed();
        }
        Ok(())
    }

    fn jobs(&self) -> usize {
        self.load.as_ref().map_or(0, |l| l.jobs.load(Ordering::Relaxed))
    }

    /// Why the instance takes no requests: its backend reported the device lost, or its main worker stopped (a stopped
    /// long lane is not counted: the main lane runs its requests). Not a check of the hardware: a CUDA backend reports
    /// `device_lost` only when the end of its own GPU work could not be confirmed (two failed synchronisations or
    /// context binds of that instance); a fatal CUDA error may leave the other GPUs of the process unusable too.
    fn unavailable(&self) -> Option<&'static str> {
        if self.health.iter().any(DeviceHealth::is_lost) {
            Some("device_lost")
        } else if self.tx.is_closed() {
            Some("worker_closed")
        } else {
            None
        }
    }

    fn available(&self) -> bool {
        self.unavailable().is_none()
    }
}

/// Queues of a served model.
enum Route {
    /// A single instance, used as without replicas: no availability check, no reservation unless metrics are on.
    One(ReplicaRoute),
    Group {
        replicas: Box<[ReplicaRoute]>,
        /// Start of the next least-loaded scan, advanced per request so ties rotate between replicas.
        next: AtomicUsize,
    },
}

impl Route {
    fn replicas(&self) -> &[ReplicaRoute] {
        match self {
            Self::One(r) => std::slice::from_ref(r),
            Self::Group { replicas, .. } => replicas,
        }
    }

    /// Queue a job on an instance; false when none took it (the job and its reservation are then dropped). A group
    /// picks the available replica with the fewest reserved jobs; the pick and the reservation are separate steps, so
    /// concurrent requests may pick the same replica (the balance is approximate, the counts exact). A worker that
    /// stopped hands the job back before any work on it, which then goes to another replica, at most one try per
    /// replica. A job is never retried after it was queued.
    fn send(&self, mut job: Job) -> bool {
        match self {
            Self::One(r) => {
                job.load = r.load.as_ref().map(Reservation::new);
                r.submit(job).is_ok()
            }
            Self::Group { replicas, next } => {
                let start = next.fetch_add(1, Ordering::Relaxed);
                for _ in 0..replicas.len() {
                    let Some(r) = least_loaded(replicas, start) else { break };
                    job.load = r.load.as_ref().map(Reservation::new);
                    match r.submit(job) {
                        Ok(()) => return true,
                        Err(back) => {
                            job = *back;
                            job.load = None;
                        }
                    }
                }
                false
            }
        }
    }
}

/// The available replica with the fewest reserved jobs, scanning from `start` (the first of equal ones wins).
fn least_loaded(replicas: &[ReplicaRoute], start: usize) -> Option<&ReplicaRoute> {
    let n = replicas.len();
    let mut best: Option<(&ReplicaRoute, usize)> = None;
    for i in 0..n {
        let r = &replicas[start.wrapping_add(i) % n];
        if !r.available() {
            continue;
        }
        let jobs = r.jobs();
        if best.is_none_or(|(_, least)| jobs < least) {
            best = Some((r, jobs));
        }
    }
    best.map(|(r, _)| r)
}

struct App {
    systemone_dialect: SystemOneDialect,
    decisions: Option<decisions::DecisionLog>,
    metrics: Option<Metrics>,
    /// Queues of every served model, by name.
    routes: HashMap<String, Route>,
    names: Vec<String>,
    model_metadata: Vec<Value>,
    large_bodies: Arc<Semaphore>,
    jobs: Arc<Semaphore>,
    media_requests: Arc<Semaphore>,
    systemone_decoders: Arc<Semaphore>,
    default_model: String,
    load: Arc<Mutex<LoadCounts>>,
    max_inflight: usize,
    release_date: String,
}

/// Concrete MCP access to the same online dispatcher as the HTTP adapters.
#[derive(Clone)]
pub(crate) struct McpDispatch {
    app: Arc<App>,
    stop: tokio_util::sync::CancellationToken,
}

impl McpDispatch {
    pub(crate) async fn decide(
        &self,
        body: Value,
        context: &rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<Value> {
        let parts =
            context.extensions.get::<axum::http::request::Parts>().context("MCP HTTP request context missing")?;
        let large = parts.extensions.get::<LargeBodyPermit>().cloned();
        let label = parts.extensions.get::<RequestModel>().cloned();
        tokio::select! {
            result = dispatch(self.app.clone(), body, Dialect::Upstream, Instant::now(), label, large) => {
                result.map(|(value, _)| value).map_err(|error| anyhow::anyhow!(error.tool_message()))
            }
            _ = self.stop.cancelled() => anyhow::bail!("basal server shutting down"),
        }
    }
}

fn plan_prompts(p: &RequestPlan) -> Vec<&[u32]> {
    p.prepared.iter().flat_map(|q| q.prompts()).collect()
}

/// One cost rule for routing/HRRN, admission and collection preview. Tree retains legacy trie accounting;
/// separate rows count each Prepared's actual packing, without allocating or deduplicating its prompts.
enum AdmissionCost {
    Tree(TrieSize),
    Rows(usize),
}

impl AdmissionCost {
    fn new(batching: basal_core::Batching) -> Self {
        match batching {
            basal_core::Batching::Tree => Self::Tree(TrieSize::default()),
            basal_core::Batching::Budget | basal_core::Batching::Single => Self::Rows(0),
        }
    }

    fn tokens(&self) -> usize {
        match self {
            Self::Tree(size) => size.tokens,
            Self::Rows(tokens) => *tokens,
        }
    }

    /// The first whole request may exceed the independent token budget. Overflow cannot admit a later request;
    /// an overflowing first cost saturates for routing and telemetry instead of wrapping to a small value.
    fn try_add(&mut self, plan: &RequestPlan, first: bool, limit: usize) -> bool {
        match self {
            Self::Tree(size) => {
                let prompts = plan_prompts(plan);
                if !first && size.tokens + size.growth(&prompts) > limit {
                    return false;
                }
                size.insert(&prompts);
            }
            Self::Rows(tokens) => {
                let total = plan.prepared.iter().try_fold(*tokens, |n, p| n.checked_add(p.packed.ids.len()));
                if !first && total.is_none_or(|n| n > limit) {
                    return false;
                }
                *tokens = total.unwrap_or(usize::MAX);
            }
        }
        true
    }
}

/// Order in which waiting requests are admitted to a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Schedule {
    /// Arrival order.
    Fifo,
    /// Highest response ratio next: (wait + estimated compute) / estimated compute, so short requests pass long
    /// ones and a long request moves up as it waits (no starvation).
    Hrrn,
}

/// A planned waiting request: its prompts' packed tokens (the estimate of its compute).
struct Waiting {
    job: Job,
    plan: RequestPlan,
    tokens: usize,
}

/// Admission of one lane: order of the waiting requests, budget, packed tokens per second of its recent batches
/// (exponential average, converts tokens to seconds for HRRN).
struct Lane {
    name: String,
    /// Actual execution lane, reported only when this replica has an active long lane.
    usage_lane: Option<&'static str>,
    metrics: Option<LaneMetrics>,
    schedule: Schedule,
    batching: basal_core::Batching,
    max_batch_tokens: usize,
    max_batch: Option<usize>,
    rate: f64,
    waiting: Vec<Waiting>,
}

impl Lane {
    fn new(
        name: &str,
        schedule: Schedule,
        batching: basal_core::Batching,
        max_batch_tokens: usize,
        max_batch: Option<usize>,
        metrics: Option<LaneMetrics>,
    ) -> Self {
        Self {
            name: name.into(),
            usage_lane: None,
            metrics,
            schedule,
            batching,
            max_batch_tokens,
            max_batch,
            rate: 3000.0,
            waiting: Vec::new(),
        }
    }

    /// Take the next batch: the first waiting request in schedule order always, then every other one that still fits
    /// the budget (Tree shares trie tokens; Budget counts separate packed primitive rows).
    fn admit(&mut self, start: Instant) -> (Vec<(Job, RequestPlan)>, usize) {
        self.waiting.retain(|w| !w.job.reply.is_closed()); // clients gone (their handlers were dropped)
        let (batch, tokens) = self.take(start);
        if let Some(metrics) = &self.metrics {
            metrics.waiting(self.waiting.len());
        }
        (batch, tokens)
    }

    fn take(&mut self, start: Instant) -> (Vec<(Job, RequestPlan)>, usize) {
        if self.schedule == Schedule::Hrrn {
            let rate = self.rate;
            let ratio = |w: &Waiting| (start - w.job.arrived).as_secs_f64() * rate / w.tokens as f64;
            self.waiting.sort_by(|a, b| ratio(b).total_cmp(&ratio(a)));
        }
        let mut batch = Vec::new();
        let mut size = AdmissionCost::new(self.batching);
        let mut rest = Vec::with_capacity(self.waiting.len());
        for mut w in self.waiting.drain(..) {
            if self.max_batch.is_none_or(|cap| batch.len() < cap)
                && size.try_add(&w.plan, batch.is_empty(), self.max_batch_tokens)
            {
                // Both lanes leave queued only on admission, after planning and any long-lane handoff.
                drop(w.job.queued.take());
                batch.push((w.job, w.plan));
            } else {
                rest.push(w);
            }
        }
        self.waiting = rest;
        (batch, size.tokens())
    }

    /// Run a batch and answer its requests.
    fn run<B: Backend>(
        &mut self,
        engine: &mut Engine<B>,
        batch: Vec<(Job, RequestPlan)>,
        tokens: usize,
        start: Instant,
    ) {
        let plans: Vec<&RequestPlan> = batch.iter().map(|(_, p)| p).collect();
        let result = engine.run_plans(&plans);
        let compute_ms = start.elapsed().as_secs_f64() * 1e3;
        if let Some(metrics) = &self.metrics {
            metrics.batch(compute_ms / 1e3, result.is_ok(), batch.iter().map(|(job, _)| start - job.arrived));
        }
        if compute_ms > 1.0 {
            self.rate = 0.8 * self.rate + 0.2 * (tokens as f64 / (compute_ms / 1e3));
        }
        let n = batch.len();
        match result {
            Ok(outs) => {
                for ((job, plan), out) in batch.into_iter().zip(outs) {
                    let mut resp = served_response(&self.name, &plan, &out);
                    if let (Some(lane), Ok(response)) = (self.usage_lane, &mut resp) {
                        response["usage"]["lane"] = lane.into();
                    }
                    if let (Some(metrics), Ok(v)) = (&self.metrics, &resp) {
                        metrics.usage(v);
                    }
                    let queue_ms = (start - job.arrived).as_secs_f64() * 1e3;
                    job.answer((resp, Some(Timing { queue_ms, compute_ms, batch: n })));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                crate::warn!("batch of {n} failed: {msg}");
                for (job, _) in batch {
                    job.answer((Err(DecideError::Internal(anyhow::anyhow!(msg.clone()))), None));
                }
            }
        }
    }
}

/// Final deployment identity, including native task responses already assembled by the source engine.
/// Internal continuations continue to use the source model; only the completed public response is renamed.
fn served_response(name: &str, plan: &RequestPlan, out: &PlanOutput) -> std::result::Result<Value, DecideError> {
    let mut response = plan_response(name, plan, out)?;
    response["model"] = name.into();
    Ok(response)
}

/// GPU thread of the main lane: plan arriving requests (validation errors are answered at once), pass requests above
/// `long_tokens` packed tokens to the long lane (when there is one), admit, run, answer.
fn scheduler<B: Backend>(
    mut engine: Engine<B>,
    mut rx: mpsc::UnboundedReceiver<Job>,
    mut lane: Lane,
    long: Option<(std::sync::mpsc::Sender<Waiting>, usize)>,
    scheduling: crate::models::SchedulingPolicy,
) {
    // `CoalesceReturning` (a profile policy, resolved with the engine's plan): a model idle for at least `idle` still
    // starts immediately; after a concurrent batch, collect its returning clients for at most `window`, avoiding
    // repeated 1+(N-1) forwards. This runs on the dedicated GPU thread, never on a Tokio executor worker.
    let coalesce = match scheduling {
        crate::models::SchedulingPolicy::CoalesceReturning { window, idle, .. } => Some((window, idle)),
        crate::models::SchedulingPolicy::Immediate => None,
    };
    let mut last_batch = 1;
    loop {
        let mut arrived: Vec<Job> = Vec::new();
        let wait_start = Instant::now();
        let had_backlog = !lane.waiting.is_empty();
        if lane.waiting.is_empty() {
            match rx.blocking_recv() {
                Some(j) => arrived.push(j),
                None => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            arrived.push(j);
        }
        if let Some((window, _)) =
            coalesce.filter(|&(_, idle)| last_batch > 1 && !had_backlog && wait_start.elapsed() < idle)
        {
            let until = Instant::now() + window;
            while Instant::now() < until {
                match rx.try_recv() {
                    Ok(job) => arrived.push(job),
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        std::thread::sleep(
                            until.saturating_duration_since(Instant::now()).min(std::time::Duration::from_micros(100)),
                        );
                    }
                }
            }
        }
        for job in arrived {
            plan_job(&engine, job, &mut lane, &long);
        }
        let start = Instant::now();
        let (batch, tokens) = lane.admit(start);
        last_batch = batch.len().max(1);
        if !batch.is_empty() {
            lane.run(&mut engine, batch, tokens, start);
        }
    }
}

/// Plan and route one job. False ends opt-in collection after a validation failure or cancelled job, so a timing
/// window cannot keep refilling with invalid work. Ownership and reply/permit release stay with the original job.
fn plan_job<B: Backend>(
    engine: &Engine<B>,
    mut job: Job,
    lane: &mut Lane,
    long: &Option<(std::sync::mpsc::Sender<Waiting>, usize)>,
) -> bool {
    if job.reply.is_closed() {
        return false;
    }
    let phase_start = basal_gpu::DiagnosticOverrides::process().phase_timing().then(Instant::now);
    // Dispatch hashed the ORIGINAL request. Translate only the explicit TypeSafe deployment alias for the core's
    // source-name check; missing/non-string fields still fail validation. Upstream ignores its `model` field.
    if job.dialect == Dialect::TypeSafe
        && lane.name != engine.manifest.name
        && job.body.get("model").and_then(Value::as_str) == Some(lane.name.as_str())
    {
        job.body["model"] = engine.manifest.name.clone().into();
    }
    let planned = engine.plan_request_as(&job.body, job.dialect);
    if let Some(start) = phase_start {
        eprintln!(
            "{}",
            json!({"cpu_planning":true,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
            "ok":planned.is_ok()})
        );
    }
    match planned {
        Ok(plan) => {
            let mut size = AdmissionCost::new(lane.batching);
            size.try_add(&plan, true, lane.max_batch_tokens);
            let w = Waiting { job, plan, tokens: size.tokens().max(1) };
            match long {
                Some((tx, limit)) if w.tokens > *limit => {
                    if let Err(e) = tx.send(w) {
                        lane.waiting.push(e.0); // the long lane stopped: run it here
                    }
                }
                _ => lane.waiting.push(w),
            }
            true
        }
        Err(e) => {
            job.answer((Err(e), None));
            false
        }
    }
}

/// GPU thread of the long lane: planned requests from the main lane, run between layers of which the main lane may
/// take the GPU (`Gate`).
fn long_lane<B: Backend>(mut engine: Engine<B>, rx: std::sync::mpsc::Receiver<Waiting>, mut lane: Lane) {
    loop {
        if lane.waiting.is_empty() {
            match rx.recv() {
                Ok(w) => lane.waiting.push(w),
                Err(_) => return,
            }
        }
        while let Ok(w) = rx.try_recv() {
            lane.waiting.push(w);
        }
        let start = Instant::now();
        let (batch, tokens) = lane.admit(start);
        if !batch.is_empty() {
            lane.run(&mut engine, batch, tokens, start);
        }
    }
}

fn error_response(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

/// TypeSafe System One endpoint. A body that is not JSON is a 422 `json_invalid` error, as in the TypeSafe API
/// (with any content type).
async fn systemone(
    State(app): State<Arc<App>>,
    label: Option<Extension<RequestModel>>,
    large: Option<Extension<LargeBodyPermit>>,
    Extension(SystemOneDecodePermit(decoding)): Extension<SystemOneDecodePermit>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let arrived = Instant::now();
    let large = large.map(|Extension(permit)| permit);
    match systemone::decode(body, headers, systemone::Permits::new(large.clone(), decoding)).await {
        Ok(v) => decide(app, v, Dialect::TypeSafe, arrived, label.map(|Extension(l)| l), large).await,
        Err(error) => error.response(),
    }
}

/// TypeSafe `json_invalid` error: location `["body", character offset]`, the decoder message in `ctx.error`.
fn json_invalid(body: &[u8], e: &serde_json::Error) -> Value {
    let text = String::from_utf8_lossy(body);
    // serde_json counts lines from 1 and columns (bytes) from 1; line 0 is an error before any input
    let pos = if e.line() == 0 {
        0
    } else {
        let line_start: usize = text.split_inclusive('\n').take(e.line() - 1).map(str::len).sum();
        let byte = (line_start + e.column().saturating_sub(1)).min(text.len());
        text.char_indices().take_while(|(i, _)| *i < byte).count()
    };
    let msg = e.to_string();
    let msg = msg.split(" at line ").next().unwrap_or(&msg);
    let mut chars = msg.chars();
    let msg = chars.next().map(|c| c.to_uppercase().chain(chars).collect::<String>()).unwrap_or_default();
    json!({"detail": [{"type": "json_invalid", "loc": ["body", pos], "msg": "JSON decode error", "input": {},
                       "ctx": {"error": msg}}]})
}

/// Upstream-compatible endpoint (`Server.decide` conventions): 422 for invalid requests, 503 for recognized OOM,
/// 500 for other engine failures, with `{"error": message}`. Admission retains 529.
async fn basal(
    State(app): State<Arc<App>>,
    label: Option<Extension<RequestModel>>,
    large: Option<Extension<LargeBodyPermit>>,
    body: axum::body::Bytes,
) -> Response {
    let arrived = Instant::now();
    let large = large.map(|Extension(permit)| permit);
    match decode_body(body, Dialect::Upstream, large.clone()).await {
        Ok(Ok(v)) if v.is_object() => {
            decide(app, v, Dialect::Upstream, arrived, label.map(|Extension(l)| l), large).await
        }
        Ok(Ok(_)) => {
            error_response(StatusCode::UNPROCESSABLE_ENTITY, json!({"error": "request body must be a JSON object"}))
        }
        Ok(Err(e)) => error_response(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":format!("JSON decode worker failed: {e}")}),
        ),
    }
}

async fn decode_body(
    body: axum::body::Bytes,
    dialect: Dialect,
    permit: Option<LargeBodyPermit>,
) -> Result<std::result::Result<Value, Value>, tokio::task::JoinError> {
    let large = body.len() > 2 * 1024 * 1024;
    let parse = move || {
        // A blocking decoder continues after handler cancellation; retain its allocation permit until it exits.
        let _permit = permit;
        serde_json::from_slice::<Value>(&body).map_err(|e| match dialect {
            Dialect::TypeSafe => json_invalid(&body, &e),
            Dialect::Upstream => json!({"error":e.to_string()}),
        })
    };
    if large {
        tokio::task::spawn_blocking(parse).await
    } else {
        Ok(parse())
    }
}

/// A coherent health snapshot across all online models and lanes. Durable background batches do not enter here.
#[derive(Default)]
struct LoadCounts {
    inflight: usize,
    queued: usize,
}

/// The handler and job share a waiting flag. All accesses to it are under `counts`, so admission, worker drops and
/// handler cancellation can release the same queued count exactly once. No lock spans planning, GPU work or await.
struct RequestLoad {
    counts: Arc<Mutex<LoadCounts>>,
    queued: AtomicBool,
}

impl RequestLoad {
    fn clear_queue(&self, counts: &mut LoadCounts) {
        if self.queued.swap(false, Ordering::Relaxed) {
            counts.queued -= 1;
        }
    }
}

/// One slot of `App::load`, released when the engine's reply arrives, on an early answer, or when the handler is
/// dropped (client disconnect).
struct Inflight(Arc<RequestLoad>);

impl Inflight {
    fn new(counts: &Arc<Mutex<LoadCounts>>) -> (Self, usize) {
        let mut total = counts.lock().unwrap_or_else(|e| e.into_inner());
        let previous = total.inflight;
        total.inflight += 1;
        (Self(Arc::new(RequestLoad { counts: counts.clone(), queued: AtomicBool::new(false) })), previous)
    }

    fn queue(&self) -> Queued {
        let mut counts = self.0.counts.lock().unwrap_or_else(|e| e.into_inner());
        self.0.queued.store(true, Ordering::Relaxed);
        counts.queued += 1;
        Queued(self.0.clone())
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        let mut counts = self.0.counts.lock().unwrap_or_else(|e| e.into_inner());
        self.0.clear_queue(&mut counts);
        counts.inflight -= 1;
    }
}

/// Lives with a waiting job, including planning, the long-lane channel, and stopped/panicked workers' queues.
struct Queued(Arc<RequestLoad>);

impl Drop for Queued {
    fn drop(&mut self) {
        let mut counts = self.0.counts.lock().unwrap_or_else(|e| e.into_inner());
        self.0.clear_queue(&mut counts);
    }
}

async fn decide(
    app: Arc<App>,
    body: Value,
    dialect: Dialect,
    arrived: Instant,
    label: Option<RequestModel>,
    large: Option<LargeBodyPermit>,
) -> Response {
    match dispatch(app, body, dialect, arrived, label, large).await {
        Ok((value, timing)) => {
            let mut headers = HeaderMap::new();
            if let Some(t) = timing {
                for (k, v) in [
                    ("x-basal-queue-ms", format!("{:.3}", t.queue_ms)),
                    ("x-basal-compute-ms", format!("{:.3}", t.compute_ms)),
                    ("x-basal-batch-requests", t.batch.to_string()),
                ] {
                    headers.insert(k, HeaderValue::from_str(&v).expect("numeric timing header"));
                }
            }
            (StatusCode::OK, headers, Json(value)).into_response()
        }
        Err(error) => error.response(dialect),
    }
}

/// Shared online result retains API causes until the transport maps them.
enum DispatchError {
    Api(DecideError),
    Overloaded { typesafe: &'static str, upstream: &'static str },
    Stopped,
}

impl DispatchError {
    fn overloaded(typesafe: &'static str, upstream: &'static str) -> Self {
        Self::Overloaded { typesafe, upstream }
    }

    fn response(self, dialect: Dialect) -> Response {
        match self {
            Self::Api(error) => error_body(dialect, error),
            Self::Overloaded { typesafe, upstream } => overloaded(dialect, typesafe, upstream),
            Self::Stopped => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                match dialect {
                    Dialect::Upstream => json!({"error": "engine stopped"}),
                    Dialect::TypeSafe => json!({"error": "internal", "detail": "engine stopped"}),
                },
            ),
        }
    }

    fn tool_message(self) -> String {
        match self {
            Self::Api(error) => {
                let (status, body) = error_parts(Dialect::Upstream, error);
                format!("basal server {}: {}", status.as_u16(), body["error"].as_str().unwrap_or("decision failed"))
            }
            Self::Overloaded { upstream, .. } => format!("basal server 529: {upstream}; Retry-After: 1"),
            Self::Stopped => "basal server 500: engine stopped".into(),
        }
    }
}

async fn dispatch(
    app: Arc<App>,
    body: Value,
    dialect: Dialect,
    arrived: Instant,
    label: Option<RequestModel>,
    mut large: Option<LargeBodyPermit>,
) -> std::result::Result<(Value, Option<Timing>), DispatchError> {
    // The request's model, or the default one when the field is missing or not a string (/v1/systemone then
    // answers the schema error of the field). Resolved before admission so a 529 also carries the model label;
    // unknown client-provided names keep the empty label.
    let model = resolve_model(&body, &app.default_model, |name| app.routes.contains_key(name));
    if let (Some(label), Ok(m)) = (label, model) {
        let _ = label.0.set(m.to_string());
    }
    let (slot, previous) = Inflight::new(&app.load);
    if previous >= app.max_inflight {
        // 529 as the TypeSafe API when it is overloaded; its SDKs (and most HTTP clients) retry 5xx with backoff
        return Err(DispatchError::overloaded(
            "Too many requests in flight. Retry after a short delay.",
            "overloaded: too many requests in flight",
        ));
    }
    let model = match model {
        Ok(m) => m,
        Err(m) => return Err(DispatchError::Api(basal_core::request::api_usage_error(format!("Unknown model: {m}")))),
    };
    let job_permit = match app.jobs.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return Err(DispatchError::overloaded(
                "Too many queued or running requests. Retry after a short delay.",
                "overloaded: too many queued or running requests",
            ));
        }
    };
    let expanded = match basal_core::request::prepared_input_bytes(&body) {
        Ok(n) => n,
        Err(e) => return Err(DispatchError::Api(e)),
    };
    if large.is_none() && expanded > 128 * 1024 {
        match app.large_bodies.clone().try_acquire_owned() {
            Ok(permit) => large = Some(Arc::new(permit)),
            Err(_) => {
                return Err(DispatchError::overloaded(
                    "Too many expanded requests in flight. Retry after a short delay.",
                    "overloaded: too many expanded requests in flight",
                ));
            }
        }
    }
    let (tx, rx) = oneshot::channel();
    let media_permit = if body.get("media").is_some_and(|v| !v.is_null())
        || (dialect == Dialect::TypeSafe && body.get("assets").is_some())
    {
        match app.media_requests.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                return Err(DispatchError::overloaded(
                    "Too many media requests in flight. Retry after a short delay.",
                    "overloaded: too many media requests in flight",
                ));
            }
        }
    } else {
        None
    };
    let admitted = app.metrics.as_ref().map(|metrics| metrics.admitted(model));
    // Look the route up while `model` still borrows the body, which request hashing may take below.
    let route = &app.routes[model];
    let queued = slot.queue();
    // Hash only when enabled, on the original incoming body. Moving admission permits through the blocking
    // worker keeps cancelled hashing jobs bounded and retains their large-body allocation until it is released.
    let (body, request_sha256, job_permit, large) = if app.decisions.is_some() {
        match decisions::DecisionLog::hash_request(body, job_permit, large).await {
            Ok(hashed) => (hashed.body, Some(hashed.hash), hashed.job, hashed.large),
            Err(error) => return Err(DispatchError::Api(DecideError::Internal(error))),
        }
    } else {
        (body, None, job_permit, large)
    };
    let sent = route.send(Job {
        body,
        dialect,
        arrived,
        reply: tx,
        load: None,
        online_demand: None,
        queued: Some(queued),
        _media_permit: media_permit,
        _large_body_permit: large,
        _job_permit: job_permit,
    });
    let reply = if sent { rx.await.ok() } else { None };
    let Some((result, timing)) = reply else {
        return Err(DispatchError::Stopped);
    };
    // One final dispatch point for both online dialects, after auto orders and native task continuations.
    let result = match (result, &app.decisions, request_sha256) {
        (Ok(response), Some(log), Some(hash)) => log.record(hash, response).await.map_err(DecideError::Internal),
        (result, _, _) => result,
    };
    drop(slot);
    drop(admitted);
    match result {
        Ok(mut v) => {
            // upstream basal-1.5 usage field: wall time of the request in the server
            if let Some(u) = v.get_mut("usage").and_then(Value::as_object_mut) {
                let ms = (arrived.elapsed().as_secs_f64() * 1e5).round() / 100.0;
                u.insert("latency_ms".into(), json!(ms));
            }
            Ok((v, timing))
        }
        Err(e) => Err(DispatchError::Api(e)),
    }
}

/// Error answer of an endpoint: upstream: 422 for request errors, 503 for recognized OOM, 500 otherwise, `{"error"}`;
/// TypeSafe: 422 for schema errors and
/// requests this runtime does not answer, 400 for the errors the TypeSafe API answers with 400, 500 otherwise.
fn error_body(dialect: Dialect, e: DecideError) -> Response {
    let (status, body) = error_parts(dialect, e);
    error_response(status, body)
}

/// Select the effective model against the caller's model registry. Batch workers
/// use the pinned deployment catalog; online uses the locally served routes.
fn resolve_model<'a>(
    body: &'a Value,
    default: &'a str,
    known: impl FnOnce(&str) -> bool,
) -> std::result::Result<&'a str, &'a str> {
    let model = body.get("model").and_then(Value::as_str).unwrap_or(default);
    if known(model) {
        Ok(model)
    } else {
        Err(model)
    }
}

/// Pure native error mapping shared by HTTP and durable batch results.
fn error_parts(dialect: Dialect, e: DecideError) -> (StatusCode, Value) {
    if dialect == Dialect::Upstream {
        let status = match &e {
            DecideError::Internal(error) => {
                let oom = error.chain().any(|cause| {
                    let message = cause.to_string();
                    message.to_ascii_lowercase().contains("out of memory") || message.contains("OutOfMemory")
                });
                if oom {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::INTERNAL_SERVER_ERROR
                }
            }
            DecideError::Validation(_) | DecideError::Usage(_) | DecideError::Unsupported(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
        };
        return (status, e.to_upstream_json());
    }
    match e {
        e @ DecideError::Usage(_) => (StatusCode::BAD_REQUEST, e.to_json()),
        e @ (DecideError::Validation(_) | DecideError::Unsupported(_)) => {
            (StatusCode::UNPROCESSABLE_ENTITY, e.to_json())
        }
        e => (StatusCode::INTERNAL_SERVER_ERROR, e.to_json()),
    }
}

/// TypeSafe `ModelMetadataList` (name, description, release_date) plus the upstream fields `mode` and `early_exit`
/// (no early-exit policies in this runtime).
async fn models_list(State(app): State<Arc<App>>) -> Json<Value> {
    let mut models = app.model_metadata.clone();
    for model in &mut models {
        model["release_date"] = app.release_date.clone().into();
    }
    Json(json!({ "models": models }))
}

/// Admit large or chunked inference bodies before reading/allocating them.
async fn large_body_admission(State(app): State<Arc<App>>, mut request: Request, next: Next) -> Response {
    let mcp = request.uri().path() == "/mcp";
    let typesafe =
        request.uri().path() == "/v1/systemone" && matches!(app.systemone_dialect, SystemOneDialect::TypeSafe);
    if typesafe || mcp {
        let permit = match app.systemone_decoders.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                return overloaded(
                    if typesafe { Dialect::TypeSafe } else { Dialect::Upstream },
                    "Too many System One requests decoding. Retry after a short delay.",
                    "overloaded: too many requests decoding",
                )
            }
        };
        request.extensions_mut().insert(SystemOneDecodePermit(Arc::new(permit)));
    }
    let large = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_none_or(|n| n > if mcp { 128 * 1024 } else { 2 * 1024 * 1024 })
        || (typesafe && request.headers().get(axum::http::header::CONTENT_ENCODING).is_some_and(|v| v != "identity"));
    let _permit = if large {
        match app.large_bodies.clone().try_acquire_owned() {
            Ok(permit) => Some(Arc::new(permit)),
            Err(_) => {
                let dialect = if typesafe { Dialect::TypeSafe } else { Dialect::Upstream };
                return overloaded(
                    dialect,
                    "Too many large requests in flight. Retry after a short delay.",
                    "overloaded: too many large requests in flight",
                );
            }
        }
    } else {
        None
    };
    if let Some(permit) = &_permit {
        request.extensions_mut().insert(permit.clone());
    }
    next.run(request).await
}

/// Ctrl-C, or SIGTERM (`docker stop`) on Unix.
pub(crate) async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
    crate::note!("shutting down");
}

/// Readiness. 200, or 503 once the backend of a single-instance model reports its device permanently lost (its
/// requests fail until a restart): `device_lost` with `lost_models`. A replica group is ready while one replica takes
/// requests; a group with unavailable replicas is listed in `degraded_models` (200, status still `ok`), one without
/// any in `unavailable_models` (503, `engine_stopped` unless a device is lost), each as `{model, replicas, available,
/// unavailable: [{replica, device, reason: worker_closed | device_lost}]}`. Neither checks the GPU hardware.
async fn health(State(app): State<Arc<App>>) -> Response {
    let lost: Vec<&str> = app
        .names
        .iter()
        .filter(
            |name| matches!(&app.routes[name.as_str()], Route::One(r) if r.health.iter().any(DeviceHealth::is_lost)),
        )
        .map(String::as_str)
        .collect();
    let (mut degraded, mut unavailable) = (Vec::new(), Vec::new());
    for name in &app.names {
        let Route::Group { replicas, .. } = &app.routes[name.as_str()] else { continue };
        let down: Vec<Value> = replicas
            .iter()
            .enumerate()
            // replicas run on one GPU each (a layer-split model has no replicas)
            .filter_map(|(i, r)| {
                r.unavailable().map(|why| json!({"replica": i, "device": r.devices[0], "reason": why}))
            })
            .collect();
        if down.is_empty() {
            continue;
        }
        let none_left = down.len() == replicas.len();
        let entry = json!({"model": name, "replicas": replicas.len(), "available": replicas.len() - down.len(),
                           "unavailable": down});
        if none_left {
            unavailable.push(entry)
        } else {
            degraded.push(entry)
        }
    }
    let (running, queued) = {
        let counts = app.load.lock().unwrap_or_else(|e| e.into_inner());
        (counts.inflight - counts.queued, counts.queued)
    };
    let mut body = json!({"status": "ok", "models": app.names, "default_model": app.default_model,
                          "running": running, "queued": queued});
    let status = if !lost.is_empty() {
        body["status"] = json!("device_lost");
        body["lost_models"] = json!(lost);
        StatusCode::SERVICE_UNAVAILABLE
    } else if !unavailable.is_empty() {
        body["status"] = json!("engine_stopped");
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::OK
    };
    if !degraded.is_empty() {
        body["degraded_models"] = Value::Array(degraded);
    }
    if !unavailable.is_empty() {
        body["unavailable_models"] = Value::Array(unavailable);
    }
    error_response(status, body)
}

pub fn serve<B: Backend + Send + 'static>(models: Vec<ServedModel<B>>, opts: ServeOptions) -> Result<()> {
    serve_inner(models, opts, None)
}

/// Run the same HTTP router and GPU scheduler on an owned listener, stopped by its benchmark owner.
pub(crate) fn serve_benchmark<B: Backend + Send + 'static>(
    models: Vec<ServedModel<B>>,
    opts: ServeOptions,
    listener: std::net::TcpListener,
    stop: oneshot::Receiver<()>,
) -> Result<()> {
    listener.set_nonblocking(true).context("benchmark listener")?;
    serve_inner(models, opts, Some((listener, stop)))
}

/// Preserve the dialect's overload body and the retry contract at every admission boundary.
fn overloaded(dialect: Dialect, typesafe: &str, upstream: &str) -> Response {
    let body = match dialect {
        Dialect::TypeSafe => json!({"detail": {"error_type": "overloaded_error", "message": typesafe}}),
        Dialect::Upstream => json!({"error": upstream}),
    };
    let mut response = error_response(StatusCode::from_u16(529).expect("529 is a valid HTTP status"), body);
    response.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

fn metadata_for_model<B: Backend>(name: &str, engine: &Engine<B>) -> Value {
    let manifest = &engine.manifest;
    let temperatures: serde_json::Map<String, Value> =
        manifest.temperatures.iter().map(|(k, v)| (k.as_str().into(), json!(v))).collect();
    let mut metadata = json!({
        "name": name,
        "description": "basal typed-decision model (choice / noul / score / act / multi / tool), Polish and English",
        "mode": "basal-rs",
        "early_exit": [],
        "prompt_format": manifest.prompt_format,
        "orders": manifest.orders.metadata(),
        "temperature_per_prim": temperatures
    });
    if manifest.layout != basal_core::prompt::PromptLayout::Canonical {
        metadata["layout"] = manifest.layout.as_str().into();
    }
    let thinking = engine.thinking_profile();
    if let Some(caps) = engine.backend.early_exit_caps() {
        metadata["early_exit"] = json!(caps.available());
        metadata["early_exit_default"] = caps.default_policy().into();
    }
    if thinking["mode"] == "auto" {
        metadata["think"] = json!({"default":"auto","available":["off","auto","force"],
            "limits":thinking["limits"], "calibration":thinking["calibration"]});
    }
    if let Some(warning) = crate::model_options::calibration_warning(manifest) {
        metadata["calibration_warning"] = warning.into();
    }
    let media = engine.backend.media_kinds();
    if !media.is_empty() {
        metadata["media"] = json!(media.iter().map(|kind| kind.as_str()).collect::<Vec<_>>());
    }
    metadata
}

pub(crate) fn check_admission_limits(max_inflight: usize, max_large_inflight: usize) -> Result<()> {
    anyhow::ensure!(
        max_inflight > 0 && max_inflight <= Semaphore::MAX_PERMITS,
        "max_inflight must be positive and within the semaphore limit"
    );
    anyhow::ensure!(
        max_large_inflight > 0 && max_large_inflight <= Semaphore::MAX_PERMITS,
        "max_large_inflight must be positive and within the semaphore limit"
    );
    Ok(())
}

/// Qualified allowances are configuration, never inferred from free memory. The snapshot only refuses an
/// allowance when the already-loaded models leave insufficient room for it and the qualified online reserve.
fn background_limits<B: Backend>(
    models: &[ServedModel<B>],
    enabled: bool,
) -> Result<(batches::device_budget::DeviceBudget, Vec<batches::executor::OnlineReserve>)> {
    use batches::config::BackgroundQualification;
    use batches::device_budget::{DeviceBudget, DeviceLimit};
    let mut policies = std::collections::BTreeMap::<usize, (u64, u64, u64)>::new();
    if enabled {
        for model in models {
            let BackgroundQualification::Qualified {
                device_bytes,
                online_reserve_bytes,
                allocation_granularity_bytes,
                ..
            } = model.batch_qualification
            else {
                continue;
            };
            for device in model.replicas.iter().flat_map(|replica| replica.devices.iter().copied()) {
                policies
                    .entry(device)
                    .and_modify(|policy| {
                        policy.0 = policy.0.min(device_bytes);
                        policy.1 = policy.1.max(online_reserve_bytes);
                        policy.2 = policy.2.max(allocation_granularity_bytes);
                    })
                    .or_insert((device_bytes, online_reserve_bytes, allocation_granularity_bytes));
            }
        }
    }
    let mut limits = Vec::new();
    let mut reserves = Vec::new();
    for (device, (background, online, granularity)) in policies {
        let snapshot = basal_gpu::gpu_memory(device).context("reading qualified batch device capacity")?;
        let needed = background.checked_add(online).context("qualified device limits overflow")?;
        anyhow::ensure!(needed <= snapshot.available_bytes,
            "GPU {device}: qualified background allowance and online reserve exceed available memory after loading models");
        anyhow::ensure!(granularity <= background, "GPU {device}: incompatible shared background allocation limits");
        limits.push(DeviceLimit {
            device,
            background_bytes: background,
            allocation_granularity: granularity,
            online_reserve_bytes: online,
        });
        reserves.push(batches::executor::OnlineReserve { device, bytes: online });
    }
    Ok((DeviceBudget::new(&limits)?, reserves))
}

fn serve_inner<B: Backend + Send + 'static>(
    models: Vec<ServedModel<B>>,
    opts: ServeOptions,
    owned: Option<(std::net::TcpListener, oneshot::Receiver<()>)>,
) -> Result<()> {
    anyhow::ensure!(!models.is_empty(), "no models to serve");
    check_admission_limits(opts.max_inflight, opts.max_large_inflight)?;
    let mut layout: Vec<(String, Vec<Vec<usize>>)> = Vec::with_capacity(models.len());
    for m in &models {
        let first = m.replicas.first().context("a served model needs at least one engine")?;
        let name = &m.name;
        let mut devices: Vec<Vec<usize>> = Vec::with_capacity(m.replicas.len());
        for r in &m.replicas {
            anyhow::ensure!(
                r.engine.manifest.name == first.engine.manifest.name,
                "replicas of {name} load different models"
            );
            anyhow::ensure!(!r.devices.is_empty(), "{name}: an instance without a GPU");
            for d in &r.devices {
                anyhow::ensure!(!devices.iter().flatten().any(|o| o == d), "{name}: two replicas on GPU {d}");
            }
            devices.push(r.devices.clone());
        }
        layout.push((name.clone(), devices));
    }
    let names: Vec<String> = layout.iter().map(|(n, _)| n.clone()).collect();
    let supports_media = models.iter().any(|m| m.replicas.iter().any(|r| !r.engine.backend.media_kinds().is_empty()));
    // replicas load the same model with the same options: the first one describes it
    let model_metadata: Vec<Value> =
        models.iter().map(|m| metadata_for_model(&m.name, &m.replicas[0].engine)).collect();
    for (i, n) in names.iter().enumerate() {
        anyhow::ensure!(!names[..i].contains(n), "model {n} configured twice");
    }
    let default_model = opts.default_model.clone().unwrap_or_else(|| names[0].clone());
    anyhow::ensure!(names.contains(&default_model), "default model {default_model} is not served ({names:?})");
    let metrics = opts
        .metrics
        .then(|| Metrics::new(&layout, opts.max_inflight))
        .transpose()
        .context("initializing Prometheus metrics")?;
    // Per GPU: with several engines (models or replicas of different models, a layer-split instance counting on each
    // of its GPUs) or a long lane on it, every engine on that GPU takes turns through its gate (short batches of any of
    // them before long requests, which yield between layers). GPUs do not wait for each other; a layer-split instance
    // holds one of its gates at a time.
    let instances: Vec<(&[usize], usize)> =
        models.iter().flat_map(|m| m.replicas.iter().map(move |r| (r.devices.as_slice(), m.long_tokens))).collect();
    let background_devices: std::collections::BTreeSet<usize> = models
        .iter()
        .filter(|model| {
            opts.batches.is_some()
                && matches!(model.batch_qualification, batches::config::BackgroundQualification::Qualified { .. })
        })
        .flat_map(|model| model.replicas.iter().flat_map(|replica| replica.devices.iter().copied()))
        .collect();
    let mut gates: HashMap<usize, Arc<Gate>> = HashMap::new();
    for &(devices, long_tokens) in &instances {
        for &device in devices {
            let on_device = instances.iter().filter(|(ds, _)| ds.contains(&device)).count();
            if on_device > 1 || long_tokens > 0 || background_devices.contains(&device) {
                gates
                    .entry(device)
                    .or_insert_with(|| Arc::new(Gate::new(std::time::Duration::from_millis(opts.long_slice_ms))));
            }
        }
    }
    let mut threads = Vec::new();
    let collection_stopping = Arc::new(AtomicBool::new(false));
    let batches_enabled = opts.batches.is_some();
    let (device_budget, online_reserves) = background_limits(&models, batches_enabled)?;
    // Weights/tables are now resident. Gates do not serialize host allocations: every main,
    // long and qualified background worker gets its own maximum PLE staging allowance.
    let mut ple_staging = 0u64;
    for model in &models {
        let background = batches_enabled
            && matches!(model.batch_qualification, batches::config::BackgroundQualification::Qualified { .. });
        let workers = 1 + u64::from(model.long_tokens > 0) + u64::from(background);
        for replica in &model.replicas {
            if let Some(bytes) = replica.engine.backend.describe()["ple"]["max_staging_bytes_per_call"].as_u64() {
                ple_staging = ple_staging
                    .checked_add(bytes.checked_mul(workers).context("host PLE worker estimate overflow")?)
                    .context("host PLE server staging overflow")?;
            }
        }
    }
    if ple_staging > 0 {
        let pool = opts.batches.as_ref().map_or(0, |b| b.host.exec_pool_bytes);
        crate::config::check_host_ple_headroom(
            ple_staging.checked_add(pool).context("host PLE and Batch pool overflow")?,
        )?;
        crate::note!("host PLE: {ple_staging} bytes for simultaneous worker staging, plus {pool} Batch pool bytes; source bounds, not allocator reservations or total process peaks");
    }
    // Synchronous startup, before any readiness message or HTTP runtime. Every subsequent exit joins it below.
    let decisions = opts.log_decisions.as_deref().map(decisions::DecisionWriter::open).transpose()?;
    let result = (|| -> Result<()> {
        let mut routes = HashMap::new();
        let mut batch_profiles = Vec::new();
        let mut background_lanes = Vec::new();
        for m in models {
            let ServedModel {
                name,
                replicas,
                max_batch_tokens,
                collection,
                schedule,
                long_tokens,
                batch_profile,
                batch_qualification,
            } = m;
            if let Some(profile) = batch_profile {
                batch_profiles.push(profile);
            }
            let group = replicas.len() > 1;
            let mut queues = Vec::with_capacity(replicas.len());
            for (index, Replica { mut engine, devices, scheduling }) in replicas.into_iter().enumerate() {
                // Keep the forward budget aligned with admission, including the forked long lane.
                // Otherwise a larger server batch is silently split at the engine's default 8192.
                engine.forward_max_tokens = max_batch_tokens;
                // threads and messages: the model's name, with the replica index in a group
                let label = if group { format!("{name}-r{index}") } else { name.clone() };
                let device_gates: Vec<(usize, &Arc<Gate>)> =
                    devices.iter().filter_map(|&d| gates.get(&d).map(|g| (d, g))).collect();
                let mut health: Vec<DeviceHealth> = engine.work_health();
                let (tx, rx) = mpsc::unbounded_channel();
                let batching = engine.request_batching;
                let lane = |kind| {
                    Lane::new(
                        &name,
                        schedule,
                        batching,
                        max_batch_tokens,
                        collection.max_batch(),
                        metrics.as_ref().map(|m| m.lane(&name, kind, index)),
                    )
                };
                if batches_enabled {
                    if let batches::config::BackgroundQualification::Qualified { composition, max_requests, .. } =
                        &batch_qualification
                    {
                        let mut background = engine.fork().with_context(|| format!("{label}: background engine"))?;
                        background
                            .configure_background()
                            .with_context(|| format!("{label}: background configuration"))?;
                        anyhow::ensure!(device_gates.len() == devices.len(), "background needs every device gate");
                        for &(device, gate) in &device_gates {
                            background.set_device_work_class(
                                device,
                                gate.clone(),
                                basal_core::WorkClass::Background,
                            )?;
                        }
                        anyhow::ensure!(
                            background.background_support().is_supported(),
                            "{label}: background is unsupported"
                        );
                        background_lanes.push(batches::executor::ExecutionLane {
                            model: name.clone(),
                            worker: batches::bg_lane::BgLane::start(&label, &name, background)?,
                            composition_qualified: *composition == batches::config::Composition::Group,
                            max_requests: *max_requests as usize,
                            devices: devices.clone(),
                        });
                    }
                }
                let long = if long_tokens > 0 {
                    let mut le = engine.fork().with_context(|| format!("{label}: second engine for the long lane"))?;
                    anyhow::ensure!(
                        device_gates.len() == devices.len(),
                        "{label}: the long lane needs the gate of every GPU it uses"
                    );
                    for &(d, gate) in &device_gates {
                        le.set_device_gate(d, gate.clone(), false).with_context(|| format!("{label}: long lane"))?;
                    }
                    health.extend(le.work_health());
                    let (ltx, lrx) = std::sync::mpsc::channel();
                    let mut l = lane("long");
                    l.usage_lane = Some("long");
                    threads.push(
                        std::thread::Builder::new()
                            .name(format!("{label}-long"))
                            .spawn(move || long_lane(le, lrx, l))
                            .context("starting a long-lane thread")?,
                    );
                    Some((ltx, long_tokens))
                } else {
                    None
                };
                for &(d, gate) in &device_gates {
                    engine.set_device_gate(d, gate.clone(), true).with_context(|| label.clone())?;
                }
                let mut main_lane = lane("main");
                main_lane.usage_lane = long.as_ref().map(|_| "main");
                crate::note!(
                    "{label}: effective collection {collection:?}; profile {scheduling:?}; long lane: entry cap only"
                );
                let collection_runtime = collection::CollectionRuntime {
                    max_inflight: opts.max_inflight,
                    stopping: Arc::clone(&collection_stopping),
                };
                threads.push(
                    std::thread::Builder::new()
                        .name(label)
                        .spawn(move || {
                            if collection == collection::CollectionPolicy::default() {
                                scheduler(engine, rx, main_lane, long, scheduling);
                            } else {
                                collection::scheduler(
                                    engine,
                                    rx,
                                    main_lane,
                                    long,
                                    scheduling,
                                    collection,
                                    collection_runtime,
                                );
                            }
                        })
                        .context("starting a GPU thread")?,
                );
                let on = match devices.as_slice() {
                    [device] => format!("GPU {device}"),
                    stages => format!(
                        "GPUs {} (layer split)",
                        stages.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
                    ),
                };
                crate::note!(
                    "serving {name}{} on {on} (batch {max_batch_tokens} tokens, {schedule:?}{})",
                    if group { format!(" replica {index}") } else { String::new() },
                    if long_tokens > 0 { format!(", long lane above {long_tokens} tokens") } else { String::new() }
                );
                // a group routes by its reservations; a single instance counts them only for its metrics
                let load = (group || metrics.is_some()).then(|| {
                    Arc::new(ReplicaLoad {
                        jobs: AtomicUsize::new(0),
                        metrics: metrics.as_ref().map(|m| m.replica(&name, index)),
                    })
                });
                let instance_gates = device_gates.iter().map(|(_, gate)| Arc::clone(gate)).collect();
                queues.push(ReplicaRoute { tx, devices, health, load, gates: instance_gates });
            }
            let route = if group {
                Route::Group { replicas: queues.into_boxed_slice(), next: AtomicUsize::new(0) }
            } else {
                Route::One(queues.pop().context("a served model needs at least one engine")?)
            };
            routes.insert(name, route);
        }
        let batch_default_model = default_model.clone();
        let app = Arc::new(App {
            systemone_dialect: opts.systemone_dialect,
            decisions: decisions.as_ref().map(decisions::DecisionWriter::log),
            metrics,
            routes,
            names,
            model_metadata,
            large_bodies: Arc::new(Semaphore::new(opts.max_large_inflight)),
            jobs: Arc::new(Semaphore::new(opts.max_inflight)),
            media_requests: Arc::new(Semaphore::new(4)),
            systemone_decoders: Arc::new(Semaphore::new(opts.max_inflight)),
            default_model,
            load: Arc::new(Mutex::new(LoadCounts::default())),
            max_inflight: opts.max_inflight,
            release_date: opts.release_date.clone(),
        });
        let typesafe = post(systemone)
            .layer(axum::extract::DefaultBodyLimit::max(basal_core::systemone::MAX_BODY_BYTES))
            .layer(axum::middleware::from_fn_with_state(app.clone(), large_body_admission));
        let basal = if supports_media {
            post(basal)
                .layer(axum::extract::DefaultBodyLimit::max(256 * 1024 * 1024))
                .layer(axum::middleware::from_fn_with_state(app.clone(), large_body_admission))
        } else {
            post(basal)
        };
        let systemone = match opts.systemone_dialect {
            SystemOneDialect::TypeSafe => typesafe,
            SystemOneDialect::Upstream => basal.clone(),
        };
        let router = Router::new().route("/v1/systemone", systemone).route("/v1/basal", basal);
        let router = if opts.metrics {
            router
                .route_layer(axum::middleware::from_fn_with_state(app.clone(), metrics::observe))
                .route("/metrics", get(metrics::export))
        } else {
            router
        };
        let router = router.route("/v1/models", get(models_list)).route("/health", get(health)).with_state(app.clone());
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        let collection_shutdown = Arc::clone(&collection_stopping);
        let collection_http_exit = Arc::clone(&collection_stopping);
        rt.block_on(async move {
            let (listener, stop) = match owned {
                Some((listener, stop)) => (tokio::net::TcpListener::from_std(listener)?, Some(stop)),
                None => (
                    tokio::net::TcpListener::bind(opts.addr).await.with_context(|| format!("binding {}", opts.addr))?,
                    None,
                ),
            };
            let bound_addr = listener.local_addr()?;
            // Off creates no transport, cancellation token or session manager. Stateless SDK mode never starts
            // session workers or stores sessions. The token's guard also cancels on setup / server errors.
            let mcp_stop = matches!(opts.mcp, crate::mcp::Switch::On).then(tokio_util::sync::CancellationToken::new);
            let _mcp_guard = mcp_stop.clone().map(tokio_util::sync::CancellationToken::drop_guard);
            let router = if let Some(stop) = &mcp_stop {
                let service = crate::mcp::http_service(
                    McpDispatch { app: app.clone(), stop: stop.clone() },
                    bound_addr,
                    app.default_model.clone(),
                    stop.clone(),
                );
                router.merge(
                    Router::new()
                        .route_service("/mcp", service)
                        .layer(axum::middleware::from_fn_with_state(app.clone(), large_body_admission)),
                )
            } else {
                router
            };
            let router = if opts.access_log { router.layer(axum::middleware::from_fn(access_log)) } else { router };
            // No queue sender survives the router/transport. In particular do not keep App while joining workers.
            drop(app);
            let batch_service = match opts.batches {
                Some(settings) => {
                    let database = opts.database.clone().context("the Batch API has no database")?;
                    Some(batches::BatchService::start(settings, database, batch_profiles, batch_default_model).await?)
                }
                None => None,
            };
            // Without an enabled consumer, `auto_migrate` still migrates the configured database once at start
            // (in the background: serving does not wait for it, and shutdown stops it).
            let database_init = match (&batch_service, opts.database) {
                (None, Some(database)) if database.settings().auto_migrate => Some(tokio::spawn(async move {
                    if let Err(e) = database.prepare().await {
                        crate::warn!("database: {:#}", anyhow::Error::new(e));
                    }
                })),
                _ => None,
            };
            let router = match &batch_service {
                Some(service) => router.merge(service.router()),
                None => router,
            };
            let executor = batch_service.as_ref().map(|service| {
                batches::executor::Executor::start(service.clone(), background_lanes, device_budget, online_reserves)
            });
            crate::done!(
                "serving on http://{} (ready {:.1} s after the start{})",
                bound_addr,
                crate::started().elapsed().as_secs_f64(),
                if opts.access_log { "; access log on" } else { "" }
            );
            let http_grace = batch_service
                .as_ref()
                .map(|service| std::time::Duration::from_millis(service.settings().leases.shutdown_grace_ms));
            let stop_batches = batch_service.as_ref().map(|service| service.stopping());
            let shutdown = async move {
                match stop {
                    Some(stop) => {
                        let _ = stop.await;
                    }
                    None => shutdown_signal().await,
                }
                collection_shutdown.store(true, Ordering::Relaxed);
                if let Some(stopping) = stop_batches {
                    stopping.cancel();
                }
                if let Some(stopping) = mcp_stop {
                    stopping.cancel();
                }
            };
            let result = match http_grace {
                Some(grace) => http::serve(listener, router, shutdown, grace).await,
                None => axum::serve(listener, router).with_graceful_shutdown(shutdown).await,
            }
            .context("HTTP server");
            // HTTP errors must also end collection waits before background shutdown or worker joins.
            collection_http_exit.store(true, Ordering::Relaxed);
            let execution_result = match executor {
                Some(executor) => executor.shutdown().await,
                None => Ok(()),
            };
            if let Some(service) = batch_service {
                service.shutdown().await;
            }
            if let Some(init) = database_init {
                init.abort();
                let _ = init.await;
            }
            result.and(execution_result)
        })?;
        // The router and with it every job queue are gone: the GPU threads finish their batch, return and drop their
        // engines (CUDA resources) before the process exits.
        drop(rt);
        Ok(())
    })();
    collection_stopping.store(true, Ordering::Relaxed);
    // On setup / HTTP errors the closure also drops all queue senders before any worker is joined.
    let mut panicked = false;
    for thread in threads {
        panicked |= thread.join().is_err();
    }
    let decisions_result = decisions.map(decisions::DecisionWriter::finish).transpose();
    result?;
    decisions_result?;
    anyhow::ensure!(!panicked, "GPU worker panicked");
    Ok(())
}

/// `[POST] 200 /v1/systemone 31 ms (queue 2 ms, compute 28 ms, batch 3)` per request (`--access-log`).
async fn access_log(req: Request, next: Next) -> Response {
    let (method, path) = (req.method().clone(), req.uri().path().to_string());
    let t = Instant::now();
    let resp = next.run(req).await;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let h = |k: &str| resp.headers().get(k).and_then(|v| v.to_str().ok()).map(str::to_string);
    let detail: Vec<String> = [
        ("queue", "x-basal-queue-ms", " ms"),
        ("compute", "x-basal-compute-ms", " ms"),
        ("batch", "x-basal-batch-requests", ""),
    ]
    .iter()
    .filter_map(|(name, key, unit)| {
        h(key).map(|v| match v.parse::<f64>() {
            Ok(x) if !unit.is_empty() => format!("{name} {}{unit}", crate::term::ms(x)),
            _ => format!("{name} {v}{unit}"),
        })
    })
    .collect();
    crate::term::access(method.as_str(), resp.status().as_u16(), &path, ms, &detail.join(", "));
    resp
}
