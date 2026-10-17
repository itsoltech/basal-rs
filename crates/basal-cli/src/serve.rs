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
//! field is missing), and all engines take turns on the GPU through one gate, so short batches of any model go
//! before long requests.
//!
//! Timing headers (not part of the System One body): `x-basal-queue-ms` (arrival to the start of its batch),
//! `x-basal-compute-ms` (planning + forwards of its batch), `x-basal-batch-requests`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use basal_core::engine::{plan_response, RequestPlan};
use basal_core::gate::Gate;
use basal_core::pack::TrieSize;
use basal_core::request::Dialect;
use basal_core::{Backend, DecideError, DeviceHealth, Engine};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, Semaphore};

mod metrics;
use metrics::{LaneMetrics, Metrics, RequestModel};

pub(crate) const DEFAULT_MAX_LARGE_INFLIGHT: usize = 4;

pub struct ServeOptions {
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
}

/// One served model: its engine and admission options.
pub struct ServedModel<B: Backend> {
    pub engine: Engine<B>,
    pub max_batch_tokens: usize,
    pub schedule: Schedule,
    /// Requests above this many packed tokens run in a second lane that yields the GPU between layers (0 = one lane).
    pub long_tokens: usize,
}

struct Timing {
    queue_ms: f64,
    compute_ms: f64,
    batch: usize,
}

type LargeBodyPermit = Arc<tokio::sync::OwnedSemaphorePermit>;

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
}

struct App {
    metrics: Option<Metrics>,
    /// Job queue of every served model, by name.
    routes: HashMap<String, mpsc::UnboundedSender<Job>>,
    names: Vec<String>,
    model_metadata: Vec<Value>,
    large_bodies: Arc<Semaphore>,
    jobs: Arc<Semaphore>,
    media_requests: Arc<Semaphore>,
    /// Device readiness reported by each model's engines (none when its backend does not report it).
    health: Vec<(String, Vec<DeviceHealth>)>,
    default_model: String,
    inflight: AtomicUsize,
    max_inflight: usize,
    release_date: String,
}

fn plan_prompts(p: &RequestPlan) -> Vec<&[u32]> {
    p.prepared.iter().flat_map(|q| q.prompts()).collect()
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
    metrics: Option<LaneMetrics>,
    schedule: Schedule,
    max_batch_tokens: usize,
    rate: f64,
    waiting: Vec<Waiting>,
}

impl Lane {
    fn new(schedule: Schedule, max_batch_tokens: usize, metrics: Option<LaneMetrics>) -> Self {
        Self { metrics, schedule, max_batch_tokens, rate: 3000.0, waiting: Vec::new() }
    }

    /// Take the next batch: the first waiting request in schedule order always, then every other one that still fits
    /// the budget (packed tokens of the batch, a state shared by several requests counted once).
    fn admit(&mut self, start: Instant) -> (Vec<(Job, RequestPlan)>, usize) {
        self.waiting.retain(|w| !w.job.reply.is_closed()); // clients gone (their handlers were dropped)
        if self.schedule == Schedule::Hrrn {
            let rate = self.rate;
            let ratio = |w: &Waiting| (start - w.job.arrived).as_secs_f64() * rate / w.tokens as f64;
            self.waiting.sort_by(|a, b| ratio(b).total_cmp(&ratio(a)));
        }
        let mut batch = Vec::new();
        let mut size = TrieSize::default();
        let mut rest = Vec::with_capacity(self.waiting.len());
        for w in self.waiting.drain(..) {
            let prompts = plan_prompts(&w.plan);
            if batch.is_empty() || size.tokens + size.growth(&prompts) <= self.max_batch_tokens {
                size.insert(&prompts);
                batch.push((w.job, w.plan));
            } else {
                rest.push(w);
            }
        }
        self.waiting = rest;
        (batch, size.tokens)
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
                    let resp = plan_response(&engine.manifest.name, &plan, &out);
                    if let (Some(metrics), Ok(v)) = (&self.metrics, &resp) {
                        metrics.usage(v);
                    }
                    let queue_ms = (start - job.arrived).as_secs_f64() * 1e3;
                    let _ = job.reply.send((resp, Some(Timing { queue_ms, compute_ms, batch: n })));
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                crate::warn!("batch of {n} failed: {msg}");
                for (job, _) in batch {
                    let _ = job.reply.send((Err(DecideError::Internal(anyhow::anyhow!(msg.clone()))), None));
                }
            }
        }
    }
}

/// GPU thread of the main lane: plan arriving requests (validation errors are answered at once), pass requests above
/// `long_tokens` packed tokens to the long lane (when there is one), admit, run, answer.
fn scheduler<B: Backend>(
    mut engine: Engine<B>,
    mut rx: mpsc::UnboundedReceiver<Job>,
    mut lane: Lane,
    long: Option<(std::sync::mpsc::Sender<Waiting>, usize)>,
) {
    // A model idle for at least 100 ms still starts immediately. After a concurrent validated Omni Max batch, collect
    // its returning clients for at most 10 ms, avoiding repeated 1+(N-1) forwards.
    // This runs on the dedicated GPU thread, never on a Tokio executor worker.
    let coalesce = matches!(&engine.manifest.config, basal_core::architecture::Architecture::Gemma4(_))
        && engine.manifest.name == "basal-omni-1.0-max"
        && engine.backend.describe()["attention_products"].as_str() == Some("F16");
    let window = std::time::Duration::from_millis(10);
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
        if coalesce && last_batch > 1 && !had_backlog && wait_start.elapsed() < std::time::Duration::from_millis(100) {
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
            if job.reply.is_closed() {
                continue;
            }
            match engine.plan_request_as(&job.body, job.dialect) {
                Ok(plan) => {
                    let mut size = TrieSize::default();
                    size.insert(&plan_prompts(&plan));
                    let w = Waiting { job, plan, tokens: size.tokens.max(1) };
                    match &long {
                        Some((tx, limit)) if w.tokens > *limit => {
                            if let Err(e) = tx.send(w) {
                                lane.waiting.push(e.0); // the long lane stopped: run it here
                            }
                        }
                        _ => lane.waiting.push(w),
                    }
                }
                Err(e) => {
                    let _ = job.reply.send((Err(e), None));
                }
            }
        }
        let start = Instant::now();
        let (batch, tokens) = lane.admit(start);
        last_batch = batch.len().max(1);
        if !batch.is_empty() {
            lane.run(&mut engine, batch, tokens, start);
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
    body: axum::body::Bytes,
) -> Response {
    let arrived = Instant::now();
    let large = large.map(|Extension(permit)| permit);
    match decode_body(body, Dialect::TypeSafe, large.clone()).await {
        Ok(Ok(v)) => decide(app, v, Dialect::TypeSafe, arrived, label.map(|Extension(l)| l), large).await,
        Ok(Err(e)) => error_response(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error":"JSON decode worker failed","detail":e.to_string()}),
        ),
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

/// Upstream-compatible endpoint (`Server.decide` conventions): any failure, including a body that is not JSON, is a
/// 422 `{"error": message}`.
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

/// One slot of `App::inflight`, released when the handler finishes or is dropped (client disconnect).
struct Inflight<'a>(&'a AtomicUsize);

impl Drop for Inflight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn decide(
    app: Arc<App>,
    body: Value,
    dialect: Dialect,
    arrived: Instant,
    label: Option<RequestModel>,
    mut large: Option<LargeBodyPermit>,
) -> Response {
    // The request's model, or the default one when the field is missing or not a string (/v1/systemone then
    // answers the schema error of the field). Resolved before admission so a 529 also carries the model label;
    // unknown client-provided names keep the empty label.
    let model = match body.get("model").and_then(Value::as_str) {
        Some(m) if app.routes.contains_key(m) => Ok(m),
        Some(m) => Err(m),
        None => Ok(app.default_model.as_str()),
    };
    if let (Some(label), Ok(m)) = (label, model) {
        let _ = label.0.set(m.to_string());
    }
    let slot = Inflight(&app.inflight);
    if app.inflight.fetch_add(1, Ordering::SeqCst) >= app.max_inflight {
        // 529 as the TypeSafe API when it is overloaded; its SDKs (and most HTTP clients) retry 5xx with backoff
        return overloaded(
            dialect,
            "Too many requests in flight. Retry after a short delay.",
            "overloaded: too many requests in flight",
        );
    }
    let model = match model {
        Ok(m) => m,
        Err(m) => return error_body(dialect, basal_core::request::api_usage_error(format!("Unknown model: {m}"))),
    };
    let job_permit = match app.jobs.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return overloaded(
                dialect,
                "Too many queued or running requests. Retry after a short delay.",
                "overloaded: too many queued or running requests",
            );
        }
    };
    let expanded = match basal_core::request::prepared_input_bytes(&body) {
        Ok(n) => n,
        Err(e) => return error_body(dialect, e),
    };
    if large.is_none() && expanded > 128 * 1024 {
        match app.large_bodies.clone().try_acquire_owned() {
            Ok(permit) => large = Some(Arc::new(permit)),
            Err(_) => {
                return overloaded(
                    dialect,
                    "Too many expanded requests in flight. Retry after a short delay.",
                    "overloaded: too many expanded requests in flight",
                );
            }
        }
    }
    let (tx, rx) = oneshot::channel();
    let media_permit = if body.get("media").is_some_and(|v| !v.is_null()) {
        match app.media_requests.clone().try_acquire_owned() {
            Ok(permit) => Some(permit),
            Err(_) => {
                return overloaded(
                    dialect,
                    "Too many media requests in flight. Retry after a short delay.",
                    "overloaded: too many media requests in flight",
                );
            }
        }
    } else {
        None
    };
    let admitted = app.metrics.as_ref().map(|metrics| metrics.admitted(model));
    let sent = app.routes[model].send(Job {
        body,
        dialect,
        arrived,
        reply: tx,
        _media_permit: media_permit,
        _large_body_permit: large,
        _job_permit: job_permit,
    });
    let reply = match sent {
        Ok(()) => rx.await.ok(),
        Err(_) => None,
    };
    drop(slot);
    drop(admitted);
    let Some((result, timing)) = reply else {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({"error": "internal", "detail": "engine stopped"}),
        );
    };
    match result {
        Ok(mut v) => {
            // upstream basal-1.5 usage field: wall time of the request in the server
            if let Some(u) = v.get_mut("usage").and_then(Value::as_object_mut) {
                let ms = (arrived.elapsed().as_secs_f64() * 1e5).round() / 100.0;
                u.insert("latency_ms".into(), json!(ms));
            }
            let mut headers = HeaderMap::new();
            if let Some(t) = timing {
                for (k, v) in [
                    ("x-basal-queue-ms", format!("{:.3}", t.queue_ms)),
                    ("x-basal-compute-ms", format!("{:.3}", t.compute_ms)),
                    ("x-basal-batch-requests", t.batch.to_string()),
                ] {
                    headers.insert(k, HeaderValue::from_str(&v).unwrap());
                }
            }
            (StatusCode::OK, headers, Json(v)).into_response()
        }
        Err(e) => error_body(dialect, e),
    }
}

/// Error answer of an endpoint: upstream: 422 `{"error"}` for every failure; TypeSafe: 422 for schema errors and
/// requests this runtime does not answer, 400 for the errors the TypeSafe API answers with 400, 500 otherwise.
fn error_body(dialect: Dialect, e: DecideError) -> Response {
    match e {
        e if dialect == Dialect::Upstream => error_response(StatusCode::UNPROCESSABLE_ENTITY, e.to_upstream_json()),
        e @ DecideError::Usage(_) => error_response(StatusCode::BAD_REQUEST, e.to_json()),
        e @ (DecideError::Validation(_) | DecideError::Unsupported(_)) => {
            error_response(StatusCode::UNPROCESSABLE_ENTITY, e.to_json())
        }
        e => error_response(StatusCode::INTERNAL_SERVER_ERROR, e.to_json()),
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
    let large = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .is_none_or(|n| n > 2 * 1024 * 1024);
    let _permit = if large {
        match app.large_bodies.clone().try_acquire_owned() {
            Ok(permit) => Some(Arc::new(permit)),
            Err(_) => {
                let dialect = if request.uri().path() == "/v1/basal" { Dialect::Upstream } else { Dialect::TypeSafe };
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
async fn shutdown_signal() {
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

/// Readiness: 200, or 503 once a model's device is permanently lost (its requests fail until a restart).
async fn health(State(app): State<Arc<App>>) -> Response {
    let lost: Vec<&str> = app
        .health
        .iter()
        .filter(|(_, devices)| devices.iter().any(DeviceHealth::is_lost))
        .map(|(name, _)| name.as_str())
        .collect();
    if lost.is_empty() {
        return Json(json!({"status": "ok", "models": app.names, "default_model": app.default_model})).into_response();
    }
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        json!({"status": "device_lost", "models": app.names, "default_model": app.default_model, "lost_models": lost}),
    )
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

fn metadata_for_model<B: Backend>(engine: &Engine<B>) -> Value {
    let manifest = &engine.manifest;
    let temperatures: serde_json::Map<String, Value> =
        manifest.temperatures.iter().map(|(k, v)| (k.as_str().into(), json!(v))).collect();
    let mut metadata = json!({
        "name": manifest.name,
        "description": "basal typed-decision model (choice / noul / score / act / multi / tool), Polish and English",
        "mode": "basal-rs",
        "early_exit": [],
        "prompt_format": manifest.prompt_format,
        "orders": manifest.orders.metadata(),
        "temperature_per_prim": temperatures
    });
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

fn serve_inner<B: Backend + Send + 'static>(
    models: Vec<ServedModel<B>>,
    opts: ServeOptions,
    owned: Option<(std::net::TcpListener, oneshot::Receiver<()>)>,
) -> Result<()> {
    anyhow::ensure!(!models.is_empty(), "no models to serve");
    check_admission_limits(opts.max_inflight, opts.max_large_inflight)?;
    let names: Vec<String> = models.iter().map(|m| m.engine.manifest.name.clone()).collect();
    let supports_media = models.iter().any(|m| !m.engine.backend.media_kinds().is_empty());
    let model_metadata: Vec<Value> = models.iter().map(|m| metadata_for_model(&m.engine)).collect();
    for (i, n) in names.iter().enumerate() {
        anyhow::ensure!(!names[..i].contains(n), "model {n} configured twice");
    }
    let default_model = opts.default_model.clone().unwrap_or_else(|| names[0].clone());
    anyhow::ensure!(names.contains(&default_model), "default model {default_model} is not served ({names:?})");
    let metrics = opts
        .metrics
        .then(|| Metrics::new(&names, opts.max_inflight))
        .transpose()
        .context("initializing Prometheus metrics")?;
    // One GPU: with several models or a long lane, every engine takes turns through one gate (short batches of any
    // model before long requests, which yield between layers).
    let shared = models.len() > 1 || models.iter().any(|m| m.long_tokens > 0);
    let gate = shared.then(|| Arc::new(Gate::new(std::time::Duration::from_millis(opts.long_slice_ms))));
    let mut threads = Vec::new();
    let result = (|| -> Result<()> {
        let mut routes = HashMap::new();
        let mut readiness = Vec::new();
        for m in models {
            let ServedModel { mut engine, max_batch_tokens, schedule, long_tokens } = m;
            // Keep the forward budget aligned with admission, including the forked long lane.
            // Otherwise a larger server batch is silently split at the engine's default 8192.
            engine.forward_max_tokens = max_batch_tokens;
            let name = engine.manifest.name.clone();
            let mut devices: Vec<DeviceHealth> = engine.backend.device_health().into_iter().collect();
            let (tx, rx) = mpsc::unbounded_channel();
            let lane = |kind| Lane::new(schedule, max_batch_tokens, metrics.as_ref().map(|m| m.lane(&name, kind)));
            let long = if long_tokens > 0 {
                let mut le = engine.fork().with_context(|| format!("{name}: second engine for the long lane"))?;
                le.backend.set_gate(gate.clone().unwrap(), false);
                devices.extend(le.backend.device_health());
                let (ltx, lrx) = std::sync::mpsc::channel();
                let l = lane("long");
                threads.push(
                    std::thread::Builder::new()
                        .name(format!("{name}-long"))
                        .spawn(move || long_lane(le, lrx, l))
                        .context("starting a long-lane thread")?,
                );
                Some((ltx, long_tokens))
            } else {
                None
            };
            if let Some(g) = &gate {
                engine.backend.set_gate(g.clone(), true);
            }
            let main_lane = lane("main");
            threads.push(
                std::thread::Builder::new()
                    .name(name.clone())
                    .spawn(move || scheduler(engine, rx, main_lane, long))
                    .context("starting a GPU thread")?,
            );
            crate::note!(
                "serving {name} (batch {max_batch_tokens} tokens, {schedule:?}{})",
                if long_tokens > 0 { format!(", long lane above {long_tokens} tokens") } else { String::new() }
            );
            readiness.push((name.clone(), devices));
            routes.insert(name, tx);
        }
        let app = Arc::new(App {
            metrics,
            routes,
            names,
            model_metadata,
            large_bodies: Arc::new(Semaphore::new(opts.max_large_inflight)),
            jobs: Arc::new(Semaphore::new(opts.max_inflight)),
            media_requests: Arc::new(Semaphore::new(4)),
            health: readiness,
            default_model,
            inflight: AtomicUsize::new(0),
            max_inflight: opts.max_inflight,
            release_date: opts.release_date.clone(),
        });
        let router = Router::new().route("/v1/systemone", post(systemone)).route("/v1/basal", post(basal));
        let router = if supports_media {
            router
                .layer(axum::extract::DefaultBodyLimit::max(256 * 1024 * 1024))
                .route_layer(axum::middleware::from_fn_with_state(app.clone(), large_body_admission))
        } else {
            router
        };
        let router = if opts.metrics {
            router
                .route_layer(axum::middleware::from_fn_with_state(app.clone(), metrics::observe))
                .route("/metrics", get(metrics::export))
        } else {
            router
        };
        let router = router.route("/v1/models", get(models_list)).route("/health", get(health)).with_state(app);
        let router = if opts.access_log { router.layer(axum::middleware::from_fn(access_log)) } else { router };
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async move {
            let (listener, stop) = match owned {
                Some((listener, stop)) => (tokio::net::TcpListener::from_std(listener)?, Some(stop)),
                None => (
                    tokio::net::TcpListener::bind(opts.addr).await.with_context(|| format!("binding {}", opts.addr))?,
                    None,
                ),
            };
            crate::done!(
                "serving on http://{} (ready {:.1} s after the start{})",
                listener.local_addr()?,
                crate::started().elapsed().as_secs_f64(),
                if opts.access_log { "; access log on" } else { "" }
            );
            let shutdown = async move {
                match stop {
                    Some(stop) => {
                        let _ = stop.await;
                    }
                    None => shutdown_signal().await,
                }
            };
            axum::serve(listener, router).with_graceful_shutdown(shutdown).await.context("HTTP server")
        })?;
        // The router and with it every job queue are gone: the GPU threads finish their batch, return and drop their
        // engines (CUDA resources) before the process exits.
        drop(rt);
        Ok(())
    })();
    // On setup / HTTP errors the closure also drops all queue senders before any worker is joined.
    let mut panicked = false;
    for thread in threads {
        panicked |= thread.join().is_err();
    }
    result?;
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
