//! Bounded closed-loop HTTP clients. Each slot sends its next request only after the preceding one finishes.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::task::JoinSet;

use super::workload::{Request, Workload};

#[derive(Serialize)]
pub(super) struct Sample {
    pub id: String,
    pub class: String,
    pub payload_bytes: usize,
    pub input_tokens: usize,
    pub questions: usize,
    pub latency_ms: f64,
    pub status: Option<u16>,
    pub error: Option<&'static str>,
    pub queue_ms: Option<f64>,
    pub batch_compute_ms: Option<f64>,
    pub batch_requests: Option<usize>,
    pub answer_changed: bool,
}

pub(super) struct Phase {
    pub name: String,
    pub concurrency: usize,
    pub planned: usize,
    pub submitted: usize,
    pub started: Instant,
    pub ended: Option<Instant>,
    pub samples: Vec<Sample>,
    pub failure: Option<String>,
}

impl Phase {
    fn new(name: &str, concurrency: usize, planned: usize) -> Self {
        Self {
            name: name.into(),
            concurrency,
            planned,
            submitted: 0,
            started: Instant::now(),
            ended: None,
            samples: Vec::with_capacity(planned),
            failure: None,
        }
    }

    pub fn report(&self) -> Value {
        let wall = (self.ended.unwrap_or_else(Instant::now) - self.started).as_secs_f64();
        let success: Vec<&Sample> = self.samples.iter().filter(|s| s.error.is_none()).collect();
        let stats = |rows: &[&Sample]| {
            let mut statuses: BTreeMap<String, usize> = BTreeMap::new();
            let mut error_kinds: BTreeMap<&str, usize> = BTreeMap::new();
            for s in rows {
                *statuses
                    .entry(s.status.map(|v| v.to_string()).unwrap_or_else(|| "no_response".into()))
                    .or_default() += 1;
                if let Some(error) = s.error {
                    *error_kinds.entry(error).or_default() += 1;
                }
            }
            json!({"completed": rows.len(), "errors": rows.iter().filter(|s| s.error.is_some()).count(),
                "latency_ms": super::latency(&rows.iter().map(|s| s.latency_ms).collect::<Vec<_>>()),
                "successful_latency_ms": super::latency(&rows.iter().filter(|s| s.error.is_none()).map(|s| s.latency_ms).collect::<Vec<_>>()),
                "queue_ms": super::latency(&rows.iter().filter_map(|s| s.queue_ms).collect::<Vec<_>>()),
                "batch_compute_ms": super::latency(&rows.iter().filter_map(|s| s.batch_compute_ms).collect::<Vec<_>>()),
                "batch_requests": super::latency(&rows.iter().filter_map(|s| s.batch_requests.map(|v| v as f64)).collect::<Vec<_>>()),
                "payload_bytes": super::latency(&rows.iter().map(|s| s.payload_bytes as f64).collect::<Vec<_>>()), "statuses": statuses, "error_kinds": error_kinds})
        };
        let mut classes: BTreeMap<&str, Vec<&Sample>> = BTreeMap::new();
        for s in &self.samples {
            classes.entry(&s.class).or_default().push(s);
        }
        let mut r = stats(&self.samples.iter().collect::<Vec<_>>());
        r["name"] = json!(self.name);
        r["concurrency"] = json!(self.concurrency);
        r["planned_requests"] = json!(self.planned);
        r["submitted_requests"] = json!(self.submitted);
        r["cancelled_requests"] = json!(self.submitted.saturating_sub(self.samples.len()));
        r["complete"] = json!(self.ended.is_some() && self.samples.len() == self.planned && self.failure.is_none());
        r["reason"] = json!(self.failure);
        r["wall_s"] = json!(wall);
        r["successful_requests_per_second"] = json!(success.len() as f64 / wall.max(f64::MIN_POSITIVE));
        r["successful_questions_per_second"] =
            json!(success.iter().map(|s| s.questions).sum::<usize>() as f64 / wall.max(f64::MIN_POSITIVE));
        r["answer_changes_vs_first_success"] = json!(self.samples.iter().filter(|s| s.answer_changed).count());
        r["classes"] = json!(classes.into_iter().map(|(k, rows)| (k, stats(&rows))).collect::<BTreeMap<_, _>>());
        r["per_request"] = json!(self.samples);
        r
    }
}

async fn request(client: reqwest::Client, url: String, r: Arc<Request>) -> (Sample, Option<Value>) {
    let body = r.body.as_ref().clone();
    let start = Instant::now();
    let result = client.post(url).header("content-type", "application/json").body(body).send().await;
    let mut sample = Sample {
        id: r.id.clone(),
        class: r.class.clone(),
        payload_bytes: r.body.len(),
        input_tokens: r.input_tokens,
        questions: r.questions,
        latency_ms: 0.0,
        status: None,
        error: None,
        queue_ms: None,
        batch_compute_ms: None,
        batch_requests: None,
        answer_changed: false,
    };
    let mut answer = None;
    match result {
        Err(e) => sample.error = Some(if e.is_timeout() { "timeout" } else { "transport" }),
        Ok(response) => {
            let status = response.status();
            sample.status = Some(status.as_u16());
            let header = |name| response.headers().get(name).and_then(|v| v.to_str().ok());
            sample.queue_ms =
                header("x-basal-queue-ms").and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0);
            sample.batch_compute_ms =
                header("x-basal-compute-ms").and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0);
            sample.batch_requests = header("x-basal-batch-requests").and_then(|v| v.parse().ok());
            match response.bytes().await {
                Err(e) => sample.error = Some(if e.is_timeout() { "timeout" } else { "response_body" }),
                Ok(bytes) => {
                    // End-to-end latency includes reading the response. Decode / comparison stay outside it.
                    sample.latency_ms = start.elapsed().as_secs_f64() * 1e3;
                    if !status.is_success() {
                        sample.error = Some("http_status");
                    } else {
                        match serde_json::from_slice::<Value>(&bytes) {
                            Ok(v) if v["answers"].as_object().is_some_and(|a| a.len() == r.questions) => {
                                answer = Some(v["answers"].clone());
                            }
                            _ => sample.error = Some("invalid_response"),
                        }
                    }
                }
            }
        }
    }
    if sample.latency_ms == 0.0 {
        sample.latency_ms = start.elapsed().as_secs_f64() * 1e3;
    }
    (sample, answer)
}

async fn run_phase(
    client: &reqwest::Client,
    url: &str,
    workload: &Workload,
    phase: &mut Phase,
    seen: &mut BTreeMap<[u8; 32], Value>,
    abort: &AtomicBool,
    server_done: &AtomicBool,
) -> Result<()> {
    let mut jobs = JoinSet::new();
    let mut next = 0;
    let outcome = async {
        while next < workload.len() || !jobs.is_empty() {
            if abort.load(Ordering::Acquire) { bail!("resource monitor stopped the benchmark"); }
            if server_done.load(Ordering::Acquire) { bail!("benchmark server stopped unexpectedly"); }
            while next < workload.len() && jobs.len() < phase.concurrency {
                let r = workload[next].clone();
                let key = r.key;
                let client = client.clone();
                let url = url.to_owned();
                jobs.spawn(async move { (key, request(client, url, r).await) });
                phase.submitted += 1;
                next += 1;
            }
            tokio::select! {
                row = jobs.join_next() => {
                    let (key, (mut sample, answer)) = row.context("no HTTP client result")?.context("HTTP client task panicked")?;
                    if let Some(answer) = answer {
                        if let Some(first) = seen.get(&key) { sample.answer_changed = first != &answer; }
                        else { seen.insert(key, answer); }
                    }
                    phase.samples.push(sample);
                    if phase.samples.last().is_some_and(|s| matches!(s.error, Some("timeout" | "transport" | "response_body" | "invalid_response"))
                        || s.status.is_some_and(|status| status >= 500 && status != 529)) {
                        bail!("HTTP timeout / transport / response failure; stopping further load and draining server work");
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
            }
        }
        Ok(())
    }.await;
    if let Err(e) = &outcome {
        phase.failure = Some(format!("{e:#}"));
    }
    // Stop producing requests, cancel client futures, and join every task before handing back the phase.
    jobs.abort_all();
    while jobs.join_next().await.is_some() {}
    phase.ended = Some(Instant::now());
    outcome
}

pub(super) async fn phases(
    url: &str,
    workloads: &[(&str, Workload, Vec<usize>)],
    options: &super::Options,
    phases: &mut Vec<Phase>,
    abort: &AtomicBool,
    server_done: &AtomicBool,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_millis(options.timeout_ms))
        .pool_max_idle_per_host(256)
        .build()?;
    let mut seen = BTreeMap::new();
    // The listener is already owned, so readiness cannot accidentally connect to another server on this port.
    let health = url.replace("/v1/systemone", "/health");
    let ready_at = Instant::now();
    loop {
        if server_done.load(Ordering::Acquire) {
            bail!("benchmark server failed during startup");
        }
        if abort.load(Ordering::Acquire) {
            bail!("resource monitor stopped the benchmark");
        }
        if client
            .get(&health)
            .timeout(std::time::Duration::from_millis(500))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        if ready_at.elapsed().as_secs() >= 30 {
            bail!("benchmark server not ready after 30 seconds");
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    for (name, workload, levels) in workloads {
        let mut representatives = BTreeMap::new();
        for r in workload {
            representatives.entry(&r.class).or_insert(r.clone());
        }
        let warm =
            representatives.values().flat_map(|r| std::iter::repeat_n(r.clone(), options.warmup)).collect::<Workload>();
        let mut warm_phase = Phase::new(&format!("{name}-warmup"), 1, warm.len());
        run_phase(&client, url, &warm, &mut warm_phase, &mut seen, abort, server_done).await?;
        ensure_warmup(&warm_phase)?;
        for &concurrency in levels {
            crate::progress!("benchmark {name}: {} requests, concurrency {concurrency}", workload.len());
            phases.push(Phase::new(name, concurrency, workload.len()));
            let phase = phases.last_mut().context("benchmark phase")?;
            run_phase(&client, url, workload, phase, &mut seen, abort, server_done).await?;
        }
    }
    Ok(())
}

fn ensure_warmup(phase: &Phase) -> Result<()> {
    if let Some(s) = phase.samples.iter().find(|s| s.error.is_some()) {
        bail!("warmup request {} failed ({:?}, HTTP {:?})", s.id, s.error, s.status);
    }
    Ok(())
}
