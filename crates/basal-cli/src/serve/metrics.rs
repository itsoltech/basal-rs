//! Process-local Prometheus registry. Only configured model names, replica indices and device ordinals, and fixed
//! endpoint/lane labels are exported.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{header::CONTENT_TYPE, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
    TextEncoder,
};
use serde_json::Value;

use super::App;

const ENDPOINTS: [&str; 3] = ["/v1/systemone", "/v1/basal", "/v1/decisions"];
const SECONDS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0,
    15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 300.0, 600.0,
];

pub(super) struct Metrics {
    registry: Registry,
    requests: IntCounterVec,
    duration: HistogramVec,
    http_inflight: IntGaugeVec,
    admitted: IntGaugeVec,
    input_tokens: IntCounterVec,
    output_tokens: IntCounterVec,
    questions: IntCounterVec,
    queue: HistogramVec,
    batch_duration: HistogramVec,
    batch_size: HistogramVec,
    replica_info: IntGaugeVec,
    replica_routed: IntCounterVec,
    replica_inflight: IntGaugeVec,
    replica_available: IntGaugeVec,
    replica_waiting: IntGaugeVec,
    replica_queue: HistogramVec,
    replica_batch_duration: HistogramVec,
    replica_batch_size: HistogramVec,
    model_replicas: IntGaugeVec,
    model_replicas_available: IntGaugeVec,
}

const BATCH_SIZES: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0];

impl Metrics {
    /// Registry of the served models, each with the device ordinals of its replicas (one replica for a single
    /// instance; a layer-split instance lists the ordinals of its stages).
    pub fn new(models: &[(String, Vec<Vec<usize>>)], max_inflight: usize) -> prometheus::Result<Self> {
        let registry = Registry::new();
        let counter = |name, help, labels| {
            let c = IntCounterVec::new(Opts::new(name, help), labels)?;
            registry.register(Box::new(c.clone()))?;
            Ok::<_, prometheus::Error>(c)
        };
        let gauge = |name, help, labels| {
            let g = IntGaugeVec::new(Opts::new(name, help), labels)?;
            registry.register(Box::new(g.clone()))?;
            Ok::<_, prometheus::Error>(g)
        };
        let histogram = |name, help, labels, buckets: &[f64]| {
            let h = HistogramVec::new(HistogramOpts::new(name, help).buckets(buckets.to_vec()), labels)?;
            registry.register(Box::new(h.clone()))?;
            Ok::<_, prometheus::Error>(h)
        };
        let limit = IntGauge::new("basal_max_inflight_requests", "Configured admission limit across all models.")?;
        limit.set(max_inflight as i64);
        registry.register(Box::new(limit))?;
        let m = Self {
            requests: counter(
                "basal_http_requests_total",
                "Inference HTTP requests completed by status, or cancelled when the handler future is dropped.",
                &["endpoint", "model", "status"],
            )?,
            duration: histogram(
                "basal_http_request_duration_seconds",
                "Inference HTTP time from middleware entry through response construction, excluding cancellations.",
                &["endpoint", "model", "status"],
                SECONDS,
            )?,
            http_inflight: gauge(
                "basal_http_inflight_requests",
                "Active inference HTTP handlers, including body reads and validation.",
                &["endpoint"],
            )?,
            admitted: gauge(
                "basal_model_inflight_requests",
                "Admitted requests awaiting a model reply; disconnected handlers release their slot.",
                &["model"],
            )?,
            input_tokens: counter(
                "basal_input_tokens_total",
                "Sum of usage.input_tokens from successfully constructed model responses.",
                &["model"],
            )?,
            output_tokens: counter(
                "basal_output_tokens_total",
                "Sum of usage.output_tokens from successfully constructed model responses (no generated text).",
                &["model"],
            )?,
            questions: counter(
                "basal_questions_total",
                "Sum of usage.questions from successfully constructed model responses.",
                &["model"],
            )?,
            queue: histogram(
                "basal_queue_duration_seconds",
                "Per-request time from handler arrival to batch admission, including planning.",
                &["model", "lane"],
                SECONDS,
            )?,
            batch_duration: histogram(
                "basal_batch_duration_seconds",
                "Wall time of batch admission and run_plans, including GPU gate waits, once per batch.",
                &["model", "lane", "outcome"],
                SECONDS,
            )?,
            batch_size: histogram(
                "basal_batch_requests",
                "Number of requests admitted to each executed batch.",
                &["model", "lane"],
                BATCH_SIZES,
            )?,
            replica_info: gauge(
                "basal_replica_info",
                // a layer-split instance labels `device` with its stages' ordinals in order, comma-separated; the help
                // text stays as before
                "Configured instances of each model (1): replica index and the CUDA device ordinal it runs on.",
                &["model", "replica", "device"],
            )?,
            replica_routed: counter(
                "basal_replica_routed_requests_total",
                "Requests queued to each replica of a model.",
                &["model", "replica"],
            )?,
            replica_inflight: gauge(
                "basal_replica_inflight_requests",
                "Requests reserved on each replica: queued, planned, waiting or running, until answered or dropped, \
                 also after their client disconnected.",
                &["model", "replica"],
            )?,
            replica_available: gauge(
                "basal_replica_available",
                "Replica takes requests (1): main worker running and no device loss reported; sampled at scrape.",
                &["model", "replica"],
            )?,
            replica_waiting: gauge(
                "basal_replica_waiting_requests",
                "Planned requests left waiting in each replica lane after its latest admission.",
                &["model", "replica", "lane"],
            )?,
            replica_queue: histogram(
                "basal_replica_queue_duration_seconds",
                "Per-request time from handler arrival to batch admission on each replica lane, including planning.",
                &["model", "replica", "lane"],
                SECONDS,
            )?,
            replica_batch_duration: histogram(
                "basal_replica_batch_duration_seconds",
                "Wall time of batch admission and run_plans on each replica lane, including GPU gate waits.",
                &["model", "replica", "lane", "outcome"],
                SECONDS,
            )?,
            replica_batch_size: histogram(
                "basal_replica_batch_requests",
                "Number of requests admitted to each executed batch of each replica lane.",
                &["model", "replica", "lane"],
                BATCH_SIZES,
            )?,
            model_replicas: gauge("basal_model_replicas", "Configured replicas of each model.", &["model"])?,
            model_replicas_available: gauge(
                "basal_model_replicas_available",
                "Replicas of each model that take requests; sampled at scrape.",
                &["model"],
            )?,
            registry,
        };
        for endpoint in ENDPOINTS {
            m.http_inflight.with_label_values(&[endpoint]);
            for (name, _) in models {
                m.requests.with_label_values(&[endpoint, name, "200"]);
                m.duration.with_label_values(&[endpoint, name, "200"]);
            }
        }
        for (name, devices) in models {
            m.admitted.with_label_values(&[name]);
            m.input_tokens.with_label_values(&[name]);
            m.output_tokens.with_label_values(&[name]);
            m.questions.with_label_values(&[name]);
            m.model_replicas.with_label_values(&[name]).set(devices.len() as i64);
            m.model_replicas_available.with_label_values(&[name]);
            for (index, stages) in devices.iter().enumerate() {
                let replica = index.to_string();
                let device = stages.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
                m.replica_info.with_label_values(&[name, &replica, &device]).set(1);
                m.replica_routed.with_label_values(&[name, &replica]);
                m.replica_inflight.with_label_values(&[name, &replica]);
                m.replica_available.with_label_values(&[name, &replica]);
            }
        }
        Ok(m)
    }

    /// Handles of replica `index` of `model` for its routing and reservations.
    pub fn replica(&self, model: &str, index: usize) -> ReplicaMetrics {
        let replica = index.to_string();
        ReplicaMetrics {
            routed: self.replica_routed.with_label_values(&[model, &replica]),
            inflight: self.replica_inflight.with_label_values(&[model, &replica]),
        }
    }

    /// Availability of every replica of `model` at this scrape, in replica order.
    pub fn set_availability(&self, model: &str, available: impl Iterator<Item = bool>) {
        let mut n = 0;
        for (index, up) in available.enumerate() {
            self.replica_available.with_label_values(&[model, &index.to_string()]).set(i64::from(up));
            n += usize::from(up);
        }
        self.model_replicas_available.with_label_values(&[model]).set(n as i64);
    }

    pub fn admitted(&self, model: &str) -> GaugeGuard {
        GaugeGuard::new(self.admitted.with_label_values(&[model]))
    }

    /// Handles of a lane of replica `replica` of `model`: the model's aggregate series (shared by its replicas) and
    /// the replica's own.
    pub fn lane(&self, model: &str, lane: &str, replica: usize) -> LaneMetrics {
        let r = replica.to_string();
        LaneMetrics {
            queue: self.queue.with_label_values(&[model, lane]),
            size: self.batch_size.with_label_values(&[model, lane]),
            duration_ok: self.batch_duration.with_label_values(&[model, lane, "success"]),
            duration_error: self.batch_duration.with_label_values(&[model, lane, "error"]),
            input_tokens: self.input_tokens.with_label_values(&[model]),
            output_tokens: self.output_tokens.with_label_values(&[model]),
            questions: self.questions.with_label_values(&[model]),
            replica_queue: self.replica_queue.with_label_values(&[model, &r, lane]),
            replica_size: self.replica_batch_size.with_label_values(&[model, &r, lane]),
            replica_duration_ok: self.replica_batch_duration.with_label_values(&[model, &r, lane, "success"]),
            replica_duration_error: self.replica_batch_duration.with_label_values(&[model, &r, lane, "error"]),
            waiting: self.replica_waiting.with_label_values(&[model, &r, lane]),
        }
    }
}

/// Cached handles keep registry lookups off the scheduler's per-request path.
pub(super) struct LaneMetrics {
    queue: Histogram,
    size: Histogram,
    duration_ok: Histogram,
    duration_error: Histogram,
    input_tokens: IntCounter,
    output_tokens: IntCounter,
    questions: IntCounter,
    replica_queue: Histogram,
    replica_size: Histogram,
    replica_duration_ok: Histogram,
    replica_duration_error: Histogram,
    waiting: IntGauge,
}

impl LaneMetrics {
    /// One executed batch: its wall time by `run_plans` outcome, its size and the queue time of each request, in the
    /// model's and the replica's series.
    pub fn batch(&self, seconds: f64, success: bool, queued: impl ExactSizeIterator<Item = Duration>) {
        let (duration, replica_duration) = if success {
            (&self.duration_ok, &self.replica_duration_ok)
        } else {
            (&self.duration_error, &self.replica_duration_error)
        };
        duration.observe(seconds);
        replica_duration.observe(seconds);
        let n = queued.len() as f64;
        self.size.observe(n);
        self.replica_size.observe(n);
        for q in queued {
            let q = q.as_secs_f64();
            self.queue.observe(q);
            self.replica_queue.observe(q);
        }
    }

    /// Planned requests left waiting in the lane after an admission.
    pub fn waiting(&self, n: usize) {
        self.waiting.set(n as i64);
    }

    pub fn usage(&self, response: &Value) {
        let usage = &response["usage"];
        for (field, counter) in [
            ("input_tokens", &self.input_tokens),
            ("output_tokens", &self.output_tokens),
            ("questions", &self.questions),
        ] {
            if let Some(n) = usage[field].as_u64() {
                counter.inc_by(n);
            }
        }
    }
}

/// Routing handles of one replica, held by its [`super::ReplicaLoad`].
pub(super) struct ReplicaMetrics {
    routed: IntCounter,
    inflight: IntGauge,
}

impl ReplicaMetrics {
    pub fn routed(&self) {
        self.routed.inc();
    }

    pub fn reserve(&self) {
        self.inflight.inc();
    }

    pub fn release(&self) {
        self.inflight.dec();
    }
}

pub(super) struct GaugeGuard(IntGauge);

impl GaugeGuard {
    fn new(gauge: IntGauge) -> Self {
        gauge.inc();
        Self(gauge)
    }
}

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        self.0.dec();
    }
}

/// Filled after JSON parsing, only with a configured model name. Empty means no model could be resolved.
#[derive(Clone, Default)]
pub(super) struct RequestModel(pub Arc<OnceLock<String>>);

/// Records the request when dropped: with its status, or as `cancelled` when the handler future was dropped first.
struct HttpObservation<'a> {
    metrics: &'a Metrics,
    endpoint: &'static str,
    model: RequestModel,
    started: Instant,
    status: Option<StatusCode>,
    _inflight: GaugeGuard,
}

impl Drop for HttpObservation<'_> {
    fn drop(&mut self) {
        let model = self.model.0.get().map_or("", String::as_str);
        let status = self.status.as_ref().map_or("cancelled", StatusCode::as_str);
        let labels = [self.endpoint, model, status];
        let metrics = self.metrics;
        metrics.requests.with_label_values(&labels).inc();
        if self.status.is_some() {
            metrics.duration.with_label_values(&labels).observe(self.started.elapsed().as_secs_f64());
        }
    }
}

/// Installed only on the inference routes, so scrapes and probes cannot skew latency or request counts.
pub(super) async fn observe(State(app): State<Arc<App>>, mut req: Request, next: Next) -> Response {
    let Some(metrics) = &app.metrics else {
        return next.run(req).await;
    };
    let endpoint = ENDPOINTS.into_iter().find(|e| *e == req.uri().path()).unwrap_or(ENDPOINTS[1]);
    let model = RequestModel::default();
    req.extensions_mut().insert(model.clone());
    let inflight = GaugeGuard::new(metrics.http_inflight.with_label_values(&[endpoint]));
    let mut observation =
        HttpObservation { metrics, endpoint, model, started: Instant::now(), status: None, _inflight: inflight };
    let response = next.run(req).await;
    observation.status = Some(response.status());
    response
}

pub(super) async fn export(State(app): State<Arc<App>>) -> Response {
    let Some(metrics) = &app.metrics else {
        return StatusCode::NOT_FOUND.into_response();
    };
    for (name, route) in &app.routes {
        metrics.set_availability(name, route.replicas().iter().map(super::ReplicaRoute::available));
    }
    let encoder = TextEncoder::new();
    match encoder.encode_to_string(&metrics.registry.gather()) {
        Ok(body) => ([(CONTENT_TYPE, encoder.format_type())], body).into_response(),
        Err(e) => {
            crate::warn!("encoding Prometheus metrics: {e}");
            (StatusCode::INTERNAL_SERVER_ERROR, "metrics encoding failed").into_response()
        }
    }
}
