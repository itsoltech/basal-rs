//! `basal serve`: HTTP server: POST /v1/systemone (TypeSafe System One), POST /v1/basal (upstream basal conventions),
//! GET /v1/models, GET /health.
//!
//! One thread owns the GPU engine; HTTP handlers (tokio) send jobs to it over a channel. The scheduler plans arriving
//! requests (validation errors are answered at once), orders the waiting ones (`--schedule`: highest response ratio
//! next by default, or arrival order) and admits the first one and every other waiting request that fits
//! `max_batch_tokens` packed tokens (no extra wait, as upstream's adaptive batching with `--wait-ms 0`). They run
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
use axum::{Json, Router};
use basal_core::engine::{plan_response, RequestPlan};
use basal_core::gate::Gate;
use basal_core::pack::TrieSize;
use basal_core::request::Dialect;
use basal_core::{Backend, DecideError, Engine};
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

pub struct ServeOptions {
    pub addr: SocketAddr,
    /// `release_date` of GET /v1/models (TypeSafe ModelMetadata; upstream reports its engine release date)
    pub release_date: String,
    /// Waiting + running requests of all models.
    pub max_inflight: usize,
    /// Model of /v1/basal requests without a "model" field (default: the first model).
    pub default_model: Option<String>,
    /// Least work of a long lane between two hand-overs of the GPU.
    pub long_slice_ms: u64,
    /// One log line per HTTP request.
    pub access_log: bool,
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

type Reply = (std::result::Result<Value, DecideError>, Option<Timing>);

struct Job {
    body: Value,
    dialect: Dialect,
    arrived: Instant,
    reply: oneshot::Sender<Reply>,
}

struct App {
    /// Job queue of every served model, by name.
    routes: HashMap<String, mpsc::UnboundedSender<Job>>,
    names: Vec<String>,
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
    schedule: Schedule,
    max_batch_tokens: usize,
    rate: f64,
    waiting: Vec<Waiting>,
}

impl Lane {
    fn new(schedule: Schedule, max_batch_tokens: usize) -> Self {
        Self { schedule, max_batch_tokens, rate: 3000.0, waiting: Vec::new() }
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
        if compute_ms > 1.0 {
            self.rate = 0.8 * self.rate + 0.2 * (tokens as f64 / (compute_ms / 1e3));
        }
        let n = batch.len();
        match result {
            Ok(outs) => {
                for ((job, plan), out) in batch.into_iter().zip(outs) {
                    let resp = plan_response(&engine.manifest.name, &plan, &out);
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
    loop {
        let mut arrived: Vec<Job> = Vec::new();
        if lane.waiting.is_empty() {
            match rx.blocking_recv() {
                Some(j) => arrived.push(j),
                None => return,
            }
        }
        while let Ok(j) = rx.try_recv() {
            arrived.push(j);
        }
        for job in arrived {
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
async fn systemone(State(app): State<Arc<App>>, body: axum::body::Bytes) -> Response {
    let arrived = Instant::now();
    match serde_json::from_slice::<Value>(&body) {
        Ok(v) => decide(app, v, Dialect::TypeSafe, arrived).await,
        Err(e) => error_response(StatusCode::UNPROCESSABLE_ENTITY, json_invalid(&body, &e)),
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
async fn basal(State(app): State<Arc<App>>, body: axum::body::Bytes) -> Response {
    let arrived = Instant::now();
    match serde_json::from_slice::<Value>(&body) {
        Ok(v) if v.is_object() => decide(app, v, Dialect::Upstream, arrived).await,
        Ok(_) => {
            error_response(StatusCode::UNPROCESSABLE_ENTITY, json!({"error": "request body must be a JSON object"}))
        }
        Err(e) => error_response(StatusCode::UNPROCESSABLE_ENTITY, json!({"error": e.to_string()})),
    }
}

/// One slot of `App::inflight`, released when the handler finishes or is dropped (client disconnect).
struct Inflight<'a>(&'a AtomicUsize);

impl Drop for Inflight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn decide(app: Arc<App>, body: Value, dialect: Dialect, arrived: Instant) -> Response {
    let slot = Inflight(&app.inflight);
    if app.inflight.fetch_add(1, Ordering::SeqCst) >= app.max_inflight {
        // 529 as the TypeSafe API when it is overloaded; its SDKs (and most HTTP clients) retry 5xx with backoff
        let body = match dialect {
            Dialect::TypeSafe => json!({"detail": {"error_type": "overloaded_error",
                                                   "message": "Too many requests in flight. Retry after a short delay."}}),
            Dialect::Upstream => json!({"error": "overloaded: too many requests in flight"}),
        };
        let mut r = error_response(StatusCode::from_u16(529).unwrap(), body);
        r.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from_static("1"));
        return r;
    }
    // The request's model, or the default one when the field is missing or not a string (/v1/systemone then
    // answers the schema error of the field).
    let model = match body.get("model").and_then(Value::as_str) {
        Some(m) if app.routes.contains_key(m) => m.to_string(),
        Some(m) => {
            return error_body(dialect, basal_core::request::api_usage_error(format!("Unknown model: {m}")));
        }
        None => app.default_model.clone(),
    };
    let (tx, rx) = oneshot::channel();
    let sent = app.routes[&model].send(Job { body, dialect, arrived, reply: tx });
    let reply = match sent {
        Ok(()) => rx.await.ok(),
        Err(_) => None,
    };
    drop(slot);
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
    let models: Vec<Value> = app
        .names
        .iter()
        .map(|name| {
            json!({"name": name, "release_date": app.release_date,
                "description": "basal typed-decision model (choice / noul / score / act / multi), Polish and English",
                "mode": "basal-rs", "early_exit": []})
        })
        .collect();
    Json(json!({ "models": models }))
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

async fn health(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"status": "ok", "models": app.names, "default_model": app.default_model}))
}

pub fn serve<B: Backend + Send + 'static>(models: Vec<ServedModel<B>>, opts: ServeOptions) -> Result<()> {
    anyhow::ensure!(!models.is_empty(), "no models to serve");
    let names: Vec<String> = models.iter().map(|m| m.engine.manifest.name.clone()).collect();
    for (i, n) in names.iter().enumerate() {
        anyhow::ensure!(!names[..i].contains(n), "model {n} configured twice");
    }
    let default_model = opts.default_model.clone().unwrap_or_else(|| names[0].clone());
    anyhow::ensure!(names.contains(&default_model), "default model {default_model} is not served ({names:?})");
    // One GPU: with several models or a long lane, every engine takes turns through one gate (short batches of any
    // model before long requests, which yield between layers).
    let shared = models.len() > 1 || models.iter().any(|m| m.long_tokens > 0);
    let gate = shared.then(|| Arc::new(Gate::new(std::time::Duration::from_millis(opts.long_slice_ms))));
    let mut routes = HashMap::new();
    let mut threads = Vec::new();
    for m in models {
        let ServedModel { mut engine, max_batch_tokens, schedule, long_tokens } = m;
        let name = engine.manifest.name.clone();
        let (tx, rx) = mpsc::unbounded_channel();
        let lane = || Lane::new(schedule, max_batch_tokens);
        let long = if long_tokens > 0 {
            let mut le = engine.fork().with_context(|| format!("{name}: second engine for the long lane"))?;
            le.backend.set_gate(gate.clone().unwrap(), false);
            let (ltx, lrx) = std::sync::mpsc::channel();
            let l = lane();
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
        let main_lane = lane();
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
        routes.insert(name, tx);
    }
    let app = Arc::new(App {
        routes,
        names,
        default_model,
        inflight: AtomicUsize::new(0),
        max_inflight: opts.max_inflight,
        release_date: opts.release_date.clone(),
    });
    let router = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/basal", post(basal))
        .route("/v1/models", get(models_list))
        .route("/health", get(health))
        .with_state(app);
    let router = if opts.access_log { router.layer(axum::middleware::from_fn(access_log)) } else { router };
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let listener =
            tokio::net::TcpListener::bind(opts.addr).await.with_context(|| format!("binding {}", opts.addr))?;
        crate::done!(
            "serving on http://{} (ready {:.1} s after the start{})",
            opts.addr,
            crate::started().elapsed().as_secs_f64(),
            if opts.access_log { "; access log on" } else { "" }
        );
        axum::serve(listener, router).with_graceful_shutdown(shutdown_signal()).await.context("HTTP server")
    })?;
    // The router and with it every job queue are gone: the GPU threads finish their batch, return and drop their
    // engines (CUDA resources) before the process exits.
    drop(rt);
    for t in threads {
        let _ = t.join();
    }
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
