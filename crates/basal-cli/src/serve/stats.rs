//! Statistics of the browser UI's server view (`/ui/api/admin/stats`, `docs/UI.md`): requests of the inference routes,
//! batches and prefix caches of every engine lane, a one-second history, hardware telemetry ([`super::telemetry`]) and
//! the most recent requests. Kept only in this process and only with the UI and accounts (`ui` with `auth`); an admin
//! resets them, a restart starts them from zero. Independent of the Prometheus registry, whose counters never reset.
//!
//! Cost on the request path: one uncontended lock per finished inference request (HTTP) and per executed batch
//! (engine lane), and reading the `usage` of each successful answer up to 64 KiB. Percentiles, history and telemetry
//! are computed by the sampler thread once a second.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::header::{CONTENT_ENCODING, CONTENT_LENGTH};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use basal_core::CacheStats;
use serde::Serialize;
use serde_json::{json, Value};

use super::metrics::{RequestModel, ENDPOINTS};
use super::telemetry::{GpuSample, HostSample};

/// Seconds of history.
const HISTORY: usize = 300;
/// Most recent requests kept.
const RECENT: usize = 200;
/// Request and response bodies up to this size are captured (`ui.capture_requests`); larger ones are noted only.
const CAPTURE_BYTES: usize = 64 * 1024;
/// Latency histogram: 8 buckets per doubling of microseconds (relative error of a quantile below 5 %).
const BUCKETS_PER_OCTAVE: f64 = 8.0;
const BUCKETS: usize = 8 * 34;

pub(super) fn unix_ms(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Log-bucketed latencies in milliseconds.
struct Histogram {
    counts: Vec<u64>,
    total: u64,
    sum_ms: f64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self { counts: vec![0; BUCKETS], total: 0, sum_ms: 0.0 }
    }
}

impl Histogram {
    fn record(&mut self, ms: f64) {
        let us = (ms * 1e3).max(1.0);
        let i = ((us.log2() * BUCKETS_PER_OCTAVE) as usize).min(BUCKETS - 1);
        self.counts[i] += 1;
        self.total += 1;
        self.sum_ms += ms;
    }

    /// Geometric middle of the bucket holding quantile `q`.
    fn quantile(&self, q: f64) -> Option<f64> {
        let rank = (q * self.total as f64).ceil().max(1.0) as u64;
        let mut seen = 0;
        for (i, &n) in self.counts.iter().enumerate() {
            seen += n;
            if n > 0 && seen >= rank {
                return Some(((i as f64 + 0.5) / BUCKETS_PER_OCTAVE).exp2() / 1e3);
            }
        }
        None
    }

    fn summary(&self) -> Value {
        json!({
            "p50": self.quantile(0.5), "p90": self.quantile(0.9), "p99": self.quantile(0.99),
            "mean": (self.total > 0).then(|| self.sum_ms / self.total as f64),
        })
    }
}

#[derive(Default)]
struct Endpoint {
    requests: u64,
    /// HTTP status 400 and above, and cancelled handlers.
    failed: u64,
    statuses: BTreeMap<String, u64>,
    latency: Histogram,
}

/// Requests finished in the current second, rolled into the history by the sampler.
#[derive(Default)]
struct Window {
    requests: usize,
    failed: u32,
    /// Of the requests answered (not cancelled).
    latencies: Vec<f64>,
}

/// A captured body, kept as received; parsed only when an admin opens the request.
#[derive(Clone)]
enum Captured {
    Body(Bytes),
    Omitted(&'static str),
}

impl Captured {
    fn json(&self) -> Value {
        match self {
            Self::Body(bytes) => {
                serde_json::from_slice(bytes).unwrap_or_else(|_| json!({"text": String::from_utf8_lossy(bytes)}))
            }
            Self::Omitted(why) => json!({"omitted": why}),
        }
    }
}

/// A finished request: its outcome and the timing the server reported; the bodies only with `ui.capture_requests`.
struct Recent {
    id: u64,
    at: u64,
    endpoint: &'static str,
    model: String,
    status: String,
    latency_ms: f64,
    timing: Timing,
    usage: Usage,
    bodies: Option<(Captured, Captured)>,
}

impl Recent {
    fn summary(&self) -> Value {
        json!({
            "id": self.id, "at": self.at, "endpoint": self.endpoint, "model": self.model, "status": self.status,
            "latency_ms": self.latency_ms, "queue_ms": self.timing.queue_ms, "compute_ms": self.timing.compute_ms,
            "batch_requests": self.timing.batch, "questions": self.usage.questions,
            "input_tokens": self.usage.input_tokens, "lane": self.usage.lane,
            "captured": self.bodies.is_some(),
        })
    }
}

/// `x-basal-*` timing headers of a successful answer.
#[derive(Default)]
struct Timing {
    queue_ms: Option<f64>,
    compute_ms: Option<f64>,
    batch: Option<u64>,
}

impl Timing {
    fn of(headers: &HeaderMap) -> Self {
        let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        Self {
            queue_ms: get("x-basal-queue-ms").and_then(|v| v.parse().ok()),
            compute_ms: get("x-basal-compute-ms").and_then(|v| v.parse().ok()),
            batch: get("x-basal-batch-requests").and_then(|v| v.parse().ok()),
        }
    }
}

/// Usage of a successful answer: `usage.questions` (the Decisions API: its number of answers), input tokens, lane.
#[derive(Default)]
struct Usage {
    questions: Option<u64>,
    input_tokens: Option<u64>,
    lane: Option<String>,
}

impl Usage {
    fn of(body: &[u8]) -> Self {
        let Ok(v) = serde_json::from_slice::<Value>(body) else { return Self::default() };
        let usage = &v["usage"];
        Self {
            questions: usage["questions"].as_u64().or_else(|| v["answers"].as_array().map(|a| a.len() as u64)),
            input_tokens: usage["input_tokens"].as_u64(),
            lane: usage["lane"].as_str().map(str::to_owned),
        }
    }
}

#[derive(Default)]
struct Http {
    endpoints: [Endpoint; ENDPOINTS.len()],
    window: Window,
    recent: VecDeque<Recent>,
    next_id: u64,
}

/// One second of the history.
#[derive(Clone, Serialize)]
struct Point {
    seq: u64,
    at: u64,
    requests: usize,
    failed: u32,
    p50: Option<f64>,
    p90: Option<f64>,
    p99: Option<f64>,
    mean: Option<f64>,
    running: usize,
    queued: usize,
    /// Per GPU of [`Telemetry::gpus`], in order.
    gpu_util: Vec<Option<u32>>,
    gpu_memory: Vec<Option<u64>>,
    cpu: Option<f64>,
    state_cache_bytes: u64,
}

#[derive(Default)]
struct History {
    points: VecDeque<Point>,
    seq: u64,
}

/// Latest hardware readings, replaced by the sampler.
#[derive(Default)]
pub(super) struct Telemetry {
    pub(super) at: u64,
    /// Running and queued requests of `/health`.
    pub(super) load: (usize, usize),
    pub(super) gpus: Vec<GpuSample>,
    pub(super) host: HostSample,
}

/// Counters of one engine lane (the main or long lane of one instance of a model), written by its GPU thread.
pub(super) struct LaneStats {
    model: String,
    replica: usize,
    lane: &'static str,
    devices: Vec<usize>,
    counters: Mutex<LaneCounters>,
}

#[derive(Default)]
struct LaneCounters {
    batches: u64,
    failed_batches: u64,
    requests: u64,
    batch_seconds: f64,
    queue_seconds: f64,
    input_tokens: u64,
    questions: u64,
    last_batch: usize,
    waiting: usize,
    /// Latest reading of the engine's prefix caches, and its counters at the last reset.
    cache: Option<CacheStats>,
    cache_base: CacheStats,
}

/// `cache` with its counters counted from `base`.
fn cache_since(cache: CacheStats, base: &CacheStats) -> CacheStats {
    CacheStats {
        state_hits: cache.state_hits.saturating_sub(base.state_hits),
        state_misses: cache.state_misses.saturating_sub(base.state_misses),
        state_inserts: cache.state_inserts.saturating_sub(base.state_inserts),
        state_evictions: cache.state_evictions.saturating_sub(base.state_evictions),
        template_hits: cache.template_hits.saturating_sub(base.template_hits),
        ..cache
    }
}

impl LaneStats {
    /// One executed batch: wall time and outcome of `run_plans`, its `n` requests and their summed queue time, the
    /// summed (input tokens, questions) of its answers and the engine's prefix caches after it.
    pub(super) fn batch(
        &self,
        seconds: f64,
        success: bool,
        n: usize,
        queued: Duration,
        usage: (u64, u64),
        cache: Option<CacheStats>,
    ) {
        let queue = queued.as_secs_f64();
        let mut c = lock(&self.counters);
        c.batches += 1;
        c.failed_batches += u64::from(!success);
        c.requests += n as u64;
        c.batch_seconds += seconds;
        c.queue_seconds += queue;
        c.input_tokens += usage.0;
        c.questions += usage.1;
        c.last_batch = n;
        if cache.is_some() {
            c.cache = cache;
        }
    }

    /// Planned requests left waiting in the lane after an admission.
    pub(super) fn waiting(&self, n: usize) {
        lock(&self.counters).waiting = n;
    }

    /// The engine's prefix caches before its first batch (template prefixes are warmed at load).
    pub(super) fn set_cache(&self, cache: Option<CacheStats>) {
        lock(&self.counters).cache = cache;
    }

    fn reset(&self) {
        let mut c = lock(&self.counters);
        let (cache, waiting) = (c.cache, c.waiting);
        *c = LaneCounters { cache, cache_base: cache.unwrap_or_default(), waiting, ..LaneCounters::default() };
    }

    fn state_cache_bytes(&self) -> u64 {
        lock(&self.counters).cache.map_or(0, |c| c.state_bytes)
    }

    fn json(&self) -> Value {
        let c = lock(&self.counters);
        let per = |x: f64, n: u64| (n > 0).then(|| x / n as f64);
        json!({
            "model": self.model, "replica": self.replica, "lane": self.lane, "devices": self.devices,
            "batches": c.batches, "failed_batches": c.failed_batches, "requests": c.requests,
            "avg_batch_requests": per(c.requests as f64, c.batches),
            "avg_batch_ms": per(c.batch_seconds * 1e3, c.batches),
            "avg_queue_ms": per(c.queue_seconds * 1e3, c.requests),
            "input_tokens": c.input_tokens, "questions": c.questions,
            "last_batch_requests": c.last_batch, "waiting": c.waiting,
            "cache": c.cache.map(|cache| cache_since(cache, &c.cache_base)),
        })
    }
}

/// The statistics of one server process.
pub(super) struct Stats {
    /// Bodies of the recent requests are kept (`ui.capture_requests`).
    capture: bool,
    started_at: u64,
    /// Last reset (`None`: counting since the start) and when counting started.
    reset_at: Mutex<(Option<u64>, Instant)>,
    /// Static description of the server and its models (`docs/UI.md`).
    config: Value,
    http: Mutex<Http>,
    lanes: Mutex<Vec<Arc<LaneStats>>>,
    history: Mutex<History>,
    telemetry: Mutex<Telemetry>,
}

impl Stats {
    pub(super) fn new(capture: bool, config: Value) -> Self {
        let now = SystemTime::now();
        let started = now.checked_sub(crate::started().elapsed()).unwrap_or(now);
        Self {
            capture,
            started_at: unix_ms(started),
            reset_at: Mutex::new((None, Instant::now())),
            config,
            http: Mutex::default(),
            lanes: Mutex::default(),
            history: Mutex::default(),
            telemetry: Mutex::default(),
        }
    }

    /// Counters of a new engine lane.
    pub(super) fn lane(&self, model: &str, replica: usize, lane: &'static str, devices: &[usize]) -> Arc<LaneStats> {
        let stats = Arc::new(LaneStats {
            model: model.to_owned(),
            replica,
            lane,
            devices: devices.to_vec(),
            counters: Mutex::default(),
        });
        let mut lanes = lock(&self.lanes);
        lanes.push(stats.clone());
        lanes.sort_by(|a, b| (&a.model, a.replica, a.lane != "main").cmp(&(&b.model, b.replica, b.lane != "main")));
        stats
    }

    /// Zero the counters, the history and the recent requests (the Prometheus registry is untouched). Request ids
    /// keep counting, so an id never names two requests.
    pub(super) fn reset(&self) {
        {
            let mut http = lock(&self.http);
            *http = Http { next_id: http.next_id, ..Http::default() };
        }
        lock(&self.history).points.clear();
        for lane in lock(&self.lanes).iter() {
            lane.reset();
        }
        *lock(&self.reset_at) = (Some(unix_ms(SystemTime::now())), Instant::now());
    }

    fn finish(&self, endpoint: usize, model: String, status: Option<StatusCode>, latency_ms: f64, recent: Partial) {
        let status_text = status.as_ref().map_or("cancelled", StatusCode::as_str).to_owned();
        let failed = status.is_none_or(|s| s.as_u16() >= 400);
        let mut http = lock(&self.http);
        let e = &mut http.endpoints[endpoint];
        e.requests += 1;
        e.failed += u64::from(failed);
        *e.statuses.entry(status_text.clone()).or_default() += 1;
        if status.is_some() {
            e.latency.record(latency_ms);
            http.window.latencies.push(latency_ms);
        }
        http.window.requests += 1;
        http.window.failed += u32::from(failed);
        http.next_id += 1;
        let id = http.next_id;
        if http.recent.len() == RECENT {
            http.recent.pop_front();
        }
        http.recent.push_back(Recent {
            id,
            at: unix_ms(SystemTime::now()),
            endpoint: ENDPOINTS[endpoint],
            model,
            status: status_text,
            latency_ms,
            timing: recent.timing,
            usage: recent.usage,
            bodies: self.capture.then(|| {
                let none = || Captured::Omitted("not captured");
                (recent.request.unwrap_or_else(none), recent.response.unwrap_or_else(none))
            }),
        });
    }

    /// Store a telemetry reading and close the current second of the history (the sampler, once a second).
    pub(super) fn tick(&self, telemetry: Telemetry) {
        let window = std::mem::take(&mut lock(&self.http).window);
        let mut latencies = window.latencies;
        latencies.sort_by(f64::total_cmp);
        let at = |q: f64| {
            (!latencies.is_empty()).then(|| latencies[((q * latencies.len() as f64).ceil() as usize).max(1) - 1])
        };
        let state_cache_bytes = lock(&self.lanes).iter().map(|l| l.state_cache_bytes()).sum();
        let mut history = lock(&self.history);
        history.seq += 1;
        let point = Point {
            seq: history.seq,
            at: telemetry.at,
            requests: window.requests,
            failed: window.failed,
            p50: at(0.5),
            p90: at(0.9),
            p99: at(0.99),
            mean: (!latencies.is_empty()).then(|| latencies.iter().sum::<f64>() / latencies.len() as f64),
            running: telemetry.load.0,
            queued: telemetry.load.1,
            gpu_util: telemetry.gpus.iter().map(|g| g.utilization_pct).collect(),
            gpu_memory: telemetry.gpus.iter().map(|g| g.memory_used_bytes).collect(),
            cpu: telemetry.host.cpu_pct,
            state_cache_bytes,
        };
        if history.points.len() == HISTORY {
            history.points.pop_front();
        }
        history.points.push_back(point);
        drop(history);
        *lock(&self.telemetry) = telemetry;
    }

    /// The view's snapshot: counters since the last reset, the history after point `since`, the latest telemetry,
    /// and with `since == 0` the static configuration.
    pub(super) fn snapshot(&self, since: u64) -> Value {
        let (reset_at, reset_instant) = *lock(&self.reset_at);
        let elapsed = reset_instant.elapsed().as_secs_f64();
        let (endpoints, total, failed, overall) = {
            let http = lock(&self.http);
            let mut overall = Histogram::default();
            let endpoints: Vec<Value> = http
                .endpoints
                .iter()
                .zip(ENDPOINTS)
                .map(|(e, name)| {
                    for (o, n) in overall.counts.iter_mut().zip(&e.latency.counts) {
                        *o += n;
                    }
                    overall.total += e.latency.total;
                    overall.sum_ms += e.latency.sum_ms;
                    json!({"endpoint": name, "requests": e.requests, "failed": e.failed, "statuses": e.statuses,
                           "latency_ms": e.latency.summary()})
                })
                .collect();
            let total: u64 = http.endpoints.iter().map(|e| e.requests).sum();
            let failed: u64 = http.endpoints.iter().map(|e| e.failed).sum();
            (endpoints, total, failed, overall)
        };
        let lanes: Vec<Value> = lock(&self.lanes).iter().map(|l| l.json()).collect();
        let (series, seq) = {
            let history = lock(&self.history);
            (history.points.iter().filter(|p| p.seq > since).cloned().collect::<Vec<_>>(), history.seq)
        };
        let telemetry = lock(&self.telemetry);
        let mut v = json!({
            "now": unix_ms(SystemTime::now()), "started_at": self.started_at, "reset_at": reset_at,
            "capture_requests": self.capture,
            "requests": {"total": total, "failed": failed,
                         "per_second": (elapsed > 0.0).then(|| total as f64 / elapsed),
                         "latency_ms": overall.summary()},
            "load": {"running": telemetry.load.0, "queued": telemetry.load.1},
            "endpoints": endpoints, "lanes": lanes,
            "telemetry": {"at": telemetry.at, "gpus": telemetry.gpus, "host": telemetry.host},
            "seq": seq, "series": series,
        });
        if since == 0 {
            v["config"] = self.config.clone();
        }
        v
    }

    /// Summaries of the recent requests, newest first.
    pub(super) fn recent(&self) -> Value {
        let http = lock(&self.http);
        let data: Vec<Value> = http.recent.iter().rev().map(Recent::summary).collect();
        json!({"capture_requests": self.capture, "data": data})
    }

    /// One recent request with its bodies (`None`: no longer kept). The bodies are parsed after the lock is released.
    pub(super) fn request(&self, id: u64) -> Option<Value> {
        let (mut v, bodies) = {
            let http = lock(&self.http);
            let r = http.recent.iter().find(|r| r.id == id)?;
            (r.summary(), r.bodies.clone())
        };
        if let Some((request, response)) = bodies {
            v["request"] = request.json();
            v["response"] = response.json();
        }
        Some(v)
    }
}

/// What the middleware learned besides the status.
#[derive(Default)]
struct Partial {
    timing: Timing,
    usage: Usage,
    request: Option<Captured>,
    response: Option<Captured>,
}

/// Records the request as cancelled when the handler future is dropped before it answered.
struct Pending<'a> {
    stats: &'a Stats,
    endpoint: usize,
    model: RequestModel,
    started: Instant,
    done: bool,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        if !self.done {
            let model = self.model.0.get().cloned().unwrap_or_default();
            let ms = self.started.elapsed().as_secs_f64() * 1e3;
            self.stats.finish(self.endpoint, model, None, ms, Partial::default());
        }
    }
}

enum Buffered {
    Kept(Bytes),
    /// Not read, with the reason.
    Skipped(&'static str, Body),
    /// Reading failed; the body is gone.
    Failed,
}

/// A body whose size is known and at most [`CAPTURE_BYTES`], read into memory.
async fn buffer(body: Body, size: Option<u64>) -> Buffered {
    match size {
        Some(n) if n <= CAPTURE_BYTES as u64 => match axum::body::to_bytes(body, CAPTURE_BYTES).await {
            Ok(bytes) => Buffered::Kept(bytes),
            Err(_) => Buffered::Failed,
        },
        Some(_) => Buffered::Skipped("larger than 64 KiB", body),
        None => Buffered::Skipped("size not known in advance", body),
    }
}

/// Installed on the inference routes (inside the Prometheus middleware, sharing its model label): counts every
/// request with its status and latency, from entry to the built response, like `basal_http_*`.
pub(super) async fn observe(State(stats): State<Arc<Stats>>, mut req: Request, next: Next) -> Response {
    let endpoint = ENDPOINTS.iter().position(|e| *e == req.uri().path()).unwrap_or(1);
    let model = match req.extensions().get::<RequestModel>() {
        Some(model) => model.clone(),
        None => {
            let model = RequestModel::default();
            req.extensions_mut().insert(model.clone());
            model
        }
    };
    let mut pending = Pending { stats: &stats, endpoint, model, started: Instant::now(), done: false };
    let mut partial = Partial::default();
    let response = if stats.capture {
        let (parts, body) = req.into_parts();
        let size = parts.headers.get(CONTENT_LENGTH).and_then(|v| v.to_str().ok()?.parse().ok());
        let encoded = parts.headers.get(CONTENT_ENCODING).is_some_and(|v| v != "identity");
        let buffered = if encoded { Buffered::Skipped("compressed body", body) } else { buffer(body, size).await };
        match buffered {
            Buffered::Kept(bytes) => {
                partial.request = Some(Captured::Body(bytes.clone()));
                next.run(Request::from_parts(parts, Body::from(bytes))).await
            }
            Buffered::Skipped(why, body) => {
                partial.request = Some(Captured::Omitted(why));
                next.run(Request::from_parts(parts, body)).await
            }
            // the client's body failed to arrive: a plain 400 (the route itself would refuse it in its own format)
            Buffered::Failed => {
                partial.request = Some(Captured::Omitted("unreadable body"));
                (StatusCode::BAD_REQUEST, "Failed to buffer the request body").into_response()
            }
        }
    } else {
        next.run(req).await
    };
    let status = response.status();
    let latency_ms = pending.started.elapsed().as_secs_f64() * 1e3;
    let read_usage = status == StatusCode::OK;
    let response = if read_usage || stats.capture {
        partial.timing = Timing::of(response.headers());
        let (parts, body) = response.into_parts();
        let size = axum::body::HttpBody::size_hint(&body).exact();
        let body = match buffer(body, size).await {
            Buffered::Kept(bytes) => {
                if read_usage {
                    partial.usage = Usage::of(&bytes);
                }
                partial.response = Some(Captured::Body(bytes.clone()));
                Body::from(bytes)
            }
            Buffered::Skipped(why, body) => {
                partial.response = Some(Captured::Omitted(why));
                body
            }
            // the answers of these routes are built in memory (`Json`, text): reading them cannot fail
            Buffered::Failed => Body::empty(),
        };
        Response::from_parts(parts, body)
    } else {
        response
    };
    pending.done = true;
    let model = pending.model.0.get().cloned().unwrap_or_default();
    stats.finish(endpoint, model, Some(status), latency_ms, partial);
    response
}
