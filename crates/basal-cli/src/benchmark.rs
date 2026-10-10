//! Hardware characterization through the production HTTP router, batching scheduler and long-request lane.

mod http;
mod resources;
mod workload;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use basal_core::{Backend, Engine, ModelManifest};
use basal_gpu::GpuBackend;
use clap::{Args, ValueEnum};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::serve::{Schedule, ServeOptions, ServedModel};
use crate::{GpuArgs, ModelArgs};
use resources::{host_memory, Guard, Monitor};

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Scenario {
    All,
    Sequential,
    Mixed,
}

#[derive(Args)]
pub(crate) struct Options {
    /// Replay your traffic: JSONL lines {id?, class?, request}. Replaces both workloads embedded in the binary
    #[arg(long)]
    requests: Option<PathBuf>,
    /// Sequential requests, mixed concurrent traffic, or both
    #[arg(long, value_enum, default_value_t = Scenario::All)]
    scenario: Scenario,
    /// Simultaneous HTTP clients in mixed phases (closed loop, measured at every listed level)
    #[arg(long, value_delimiter = ',', default_value = "1,4,8,16,32")]
    concurrency: Vec<usize>,
    /// Unmeasured requests per payload class, before each workload
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// HTTP request timeout; timeouts are reported and stop further load generation
    #[arg(long, default_value_t = 120_000)]
    timeout_ms: u64,
    /// Minimum available RAM to retain (GiB)
    #[arg(long, default_value_t = 2.0)]
    ram_reserve_gib: f64,
    /// Minimum free VRAM / Metal working-set headroom (GiB)
    #[arg(long, default_value_t = 1.0)]
    gpu_reserve_gib: f64,
    /// Packed tokens admitted per server batch; same setting as serve
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: usize,
    /// Server waiting + running request limit; excess requests receive 529
    #[arg(long, default_value_t = 1024)]
    max_inflight: usize,
    /// Server queue admission order
    #[arg(long, value_enum, default_value_t = Schedule::Hrrn)]
    schedule: Schedule,
    /// Packed-token threshold of the server's long-request lane (0 disables it)
    #[arg(long, default_value_t = 4096)]
    long_tokens: usize,
    /// Minimum long-lane slice before handing the GPU to short requests
    #[arg(long, default_value_t = 100)]
    long_slice_ms: u64,
    /// Save the machine-readable result to a NEW file
    #[arg(long)]
    out: Option<PathBuf>,
    /// Print only JSON to stdout; progress stays on stderr
    #[arg(long)]
    json: bool,
}

impl Options {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.concurrency.is_empty() && self.concurrency.iter().all(|c| (1..=256).contains(c)),
            "--concurrency needs levels between 1 and 256"
        );
        ensure!(self.warmup <= 10, "--warmup must be between 0 and 10 per class");
        ensure!((1..=3_600_000).contains(&self.timeout_ms), "--timeout-ms must be between 1 and 3600000");
        ensure!(
            self.max_batch_tokens > 0 && self.max_inflight > 0 && self.long_slice_ms > 0,
            "server budgets / long-slice-ms must be positive"
        );
        for (name, value) in [("ram-reserve-gib", self.ram_reserve_gib), ("gpu-reserve-gib", self.gpu_reserve_gib)] {
            ensure!(
                value.is_finite() && (0.1..=1048576.0).contains(&value),
                "--{name} must be finite and between 0.1 and 1048576"
            );
        }
        Ok(())
    }
}

/// Median p50, nearest-rank p95/p99; never substitutes a missing sample with zero.
fn latency(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut s = values.to_vec();
    s.sort_by(f64::total_cmp);
    let quantile = |p: f64| s[((s.len() as f64 * p).ceil() as usize).saturating_sub(1)];
    json!({"n": s.len(), "min": s[0], "p50": crate::stats::median(&s), "p95": quantile(0.95), "p99": quantile(0.99),
        "max": s[s.len() - 1], "mean": s.iter().sum::<f64>() / s.len() as f64})
}

/// Owns the internal server. Normal completion and early errors both stop and join it.
struct Server {
    stop: Option<oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    done: Arc<AtomicBool>,
    url: String,
}

impl Server {
    fn start(engine: Engine<GpuBackend>, options: &Options) -> Result<Self> {
        let listener =
            std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).context("binding benchmark listener")?;
        let addr = listener.local_addr()?;
        let opts = ServeOptions {
            addr,
            release_date: "2026-10-05".into(),
            max_inflight: options.max_inflight,
            default_model: None,
            long_slice_ms: options.long_slice_ms,
            access_log: false,
            metrics: false,
        };
        let model = ServedModel {
            engine,
            max_batch_tokens: options.max_batch_tokens,
            schedule: options.schedule,
            long_tokens: options.long_tokens,
        };
        let (stop, receiver) = oneshot::channel();
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        let thread = std::thread::Builder::new()
            .name("benchmark-server".into())
            .spawn(move || {
                let result = crate::serve::serve_benchmark(vec![model], opts, listener, receiver);
                flag.store(true, Ordering::Release);
                result
            })
            .context("starting benchmark HTTP server")?;
        Ok(Self { stop: Some(stop), thread: Some(thread), done, url: format!("http://{addr}/v1/systemone") })
    }
    fn finish(mut self) -> Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.thread.take().context("server handle")?.join().map_err(|_| anyhow::anyhow!("benchmark server panicked"))?
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub(crate) fn run(model: &ModelArgs, gpu_options: &GpuArgs, options: &Options) -> Result<()> {
    options.validate()?;
    let clients = if options.scenario == Scenario::Sequential {
        1
    } else {
        options.concurrency.iter().copied().max().unwrap_or(1)
    };
    resources::check_open_files(clients)?;
    let raw = workload::load(options.requests.as_deref())?;
    let mut output = options
        .out
        .as_ref()
        .map(|path| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .with_context(|| format!("creating {} (must not exist)", path.display()))
        })
        .transpose()?;
    let mut report = json!({"schema_version": 2, "runtime_version": crate::version(), "source": raw.source,
        "model": {"reference": model.model}, "complete": false, "status": "not_run", "phases": [],
        "methodology": {"transport": "loopback HTTP /v1/systemone", "traffic": "closed_loop", "p99_method": "nearest_rank",
            "warmup_per_class": options.warmup, "concurrency": options.concurrency, "timeout_ms": options.timeout_ms,
            "client_latency_scope": "send request through complete response body; excludes JSON decode and answer comparison",
            "batch_compute_scope": "batch admission, packing and forwards; shared by requests in a batch, not standalone inference latency; request validation and tokenization on the GPU thread count as queue time",
            "resource_sample_interval_ms": 200},
        "limitations": ["Built-in mixed traffic is synthetic; use --requests for application-specific payloads.",
            "Closed-loop clients wait for replies; this measures concurrency and saturation, not independent open-loop arrivals.",
            "Load generator and server share this host and process; CPU/RSS include both.",
            "Memory is sampled, not a true allocation peak; reserve checks cannot prevent an OOM inside a forward.",
            "No sustained thermal/power guarantee. Answer consistency is not upstream FP32 quality validation."]});
    let result = execute(model, gpu_options, options, raw, &mut report);
    if let Err(e) = &result {
        report["reason"] = json!(format!("{e:#}"));
    }
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(file) = &mut output {
        file.write_all(encoded.as_bytes()).context("writing benchmark report")?;
    }
    if options.json {
        print!("{encoded}");
    } else {
        print_summary(&report);
    }
    result
}

fn execute(
    model: &ModelArgs,
    g: &GpuArgs,
    options: &Options,
    raw: workload::RawWorkload,
    report: &mut Value,
) -> Result<()> {
    let host = host_memory()?;
    let gpu = basal_gpu::gpu_memory()?;
    report["device"] = json!(gpu);
    let mut guard = Guard::new(host.clone(), &gpu, options);
    let first = guard.check(&host, &gpu, 0);
    report["resources"] = guard.report();
    first?;
    let dir = model.dir()?;
    let checkpoint_bytes = std::fs::metadata(dir.join("model.safetensors")).context("checkpoint size")?.len();
    report["model"]["checkpoint_bytes"] = json!(checkpoint_bytes);
    let preflight = guard.check(&host_memory()?, &basal_gpu::gpu_memory()?, checkpoint_bytes);
    report["resources"] = guard.report();
    preflight?;
    // Same automatic / bundled GEMM selection as serve, without ambient server configuration.
    let mut config = crate::config::model_ref(&model.model)?;
    report["model"]["resolved_revision"] = json!(config.revision);
    config.dtype = g.dtype.clone();
    if let Some(table) = &g.gemm_table {
        config.gemm_table = table.to_string_lossy().into_owned();
    }
    let manifest = ModelManifest::load(&dir)?;
    let mut gpu_settings = g.clone();
    gpu_settings.gemm_table = crate::config::gemm_table(&config, &manifest, &crate::config::default_gemm_cache())?;
    let engine = crate::gpu_engine_at(&dir, &gpu_settings)?;
    if let Some(table) = &gpu_settings.gemm_table {
        crate::config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), table)?;
    }
    report["model"]["name"] = json!(engine.manifest.name);
    report["model"]["load_s"] = json!(engine.backend.load_s);
    report["backend"] = engine.backend.describe();
    // File name only: an automatic table lives in the user's cache directory, whose path identifies the machine.
    let gemm_table = gpu_settings.gemm_table.as_deref().and_then(Path::file_name).map(|n| n.to_string_lossy());
    report["settings"] = json!({"dtype": g.dtype, "kernels": g.kernels, "readout": g.readout, "prefix_cache": !g.no_prefix_cache,
        "state_cache_mb": g.state_cache_mb, "gemm_table": gemm_table, "tree_max_tokens": g.tree_max_tokens,
        "max_batch_tokens": options.max_batch_tokens, "max_inflight": options.max_inflight, "schedule": format!("{:?}", options.schedule),
        "long_tokens": options.long_tokens, "long_slice_ms": options.long_slice_ms});
    let loaded = guard.check(&host_memory()?, &engine.backend.memory()?, 0);
    report["resources"] = guard.report();
    loaded?;
    let monitor = Monitor::start(guard, engine.backend.memory_probe())?;
    let mut workloads = Vec::new();
    report["workloads"] = json!({});
    if options.scenario != Scenario::Mixed {
        let (work, description) = workload::prepare(raw.sequential, &engine, false)?;
        report["workloads"]["sequential"] = description;
        workloads.push(("sequential", work, vec![1]));
    }
    if options.scenario != Scenario::Sequential {
        let (work, description) = workload::prepare(raw.mixed, &engine, true)?;
        report["workloads"]["mixed"] = description;
        workloads.push(("mixed", work, options.concurrency.clone()));
    }
    for (name, work, _) in &workloads {
        ensure!(!work.is_empty(), "benchmark workload {name}: no requests within the model context limit");
    }
    let server = Server::start(engine, options)?;
    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let mut phases = Vec::new();
    report["status"] = json!("running");
    let measured = runtime.block_on(async {
        tokio::select! {
            result = http::phases(&server.url, &workloads, options, &mut phases, &monitor.abort, &server.done) => result,
            signal = tokio::signal::ctrl_c() => { signal.context("Ctrl-C handler")?; anyhow::bail!("benchmark interrupted"); }
        }
    });
    let now = Instant::now();
    for p in &mut phases {
        if p.ended.is_none() {
            p.ended = Some(now);
            p.failure = Some("benchmark interrupted".into());
        }
    }
    drop(runtime);
    // Stop production, then drain and join the actual server's active GPU batch before returning.
    let shutdown = server.finish();
    if let Err(e) = &shutdown {
        report["server_error"] = json!(format!("{e:#}"));
    }
    let resources = monitor.finish()?;
    report["phases"] = json!(phases
        .iter()
        .map(|p| {
            let mut r = p.report();
            r["resources"] = resources.window(p.started, p.ended.unwrap_or(now));
            r
        })
        .collect::<Vec<_>>());
    report["resources"] = resources.summary;
    let resource_failure = resources.reason;
    let complete = measured.is_ok() && shutdown.is_ok() && resource_failure.is_none();
    report["complete"] = json!(complete);
    let errors = phases.iter().any(|p| p.samples.iter().any(|s| s.error.is_some()));
    report["status"] = json!(if !complete {
        "aborted"
    } else if errors {
        "completed_with_errors"
    } else {
        "completed"
    });
    if let Some(reason) = resource_failure {
        anyhow::bail!("resource guard: {reason}");
    }
    match (measured, shutdown) {
        // The client side only sees that the server stopped; keep the server's own error chain as the cause.
        (Err(e), Err(server)) => Err(server.context(format!("{e:#}"))),
        (measured, shutdown) => measured.and(shutdown),
    }
}

fn print_summary(report: &Value) {
    let display =
        |v: &Value, divisor: f64| v.as_f64().map(|v| format!("{:.2}", v / divisor)).unwrap_or_else(|| "n/a".into());
    println!("benchmark: {}", report["status"].as_str().unwrap_or("error"));
    if let Some(workloads) = report["workloads"].as_object() {
        for (name, work) in workloads {
            if work["skipped_requests"].as_u64().is_some_and(|count| count > 0) {
                println!("{name}: skipped {} requests exceeding the model context limit", work["skipped_requests"]);
            }
        }
    }
    if let Some(phases) = report["phases"].as_array() {
        for p in phases {
            println!(
                "{} / {} clients: {}/{} completed, {} errors",
                p["name"].as_str().unwrap_or("phase"),
                p["concurrency"],
                p["completed"],
                p["planned_requests"],
                p["errors"]
            );
            if let (Some(min), Some(p50), Some(p99), Some(rate)) = (
                p["successful_latency_ms"]["min"].as_f64(),
                p["successful_latency_ms"]["p50"].as_f64(),
                p["successful_latency_ms"]["p99"].as_f64(),
                p["successful_requests_per_second"].as_f64(),
            ) {
                println!("  HTTP: min {min:.2} ms, p50 {p50:.2} ms, p99 {p99:.2} ms; {rate:.2} requests/s");
            }
            println!(
                "  p99: queue {} ms, whole-batch compute {} ms",
                display(&p["queue_ms"]["p99"], 1.0),
                display(&p["batch_compute_ms"]["p99"], 1.0)
            );
            let resources = &p["resources"];
            println!(
                "  sampled headroom: RAM {} GiB, GPU {} GiB; CPU {} cores; lifetime peak RSS {} GiB",
                display(&resources["min_available_ram_bytes_sampled"], 1073741824.0),
                display(&resources["min_available_gpu_bytes_sampled"], 1073741824.0),
                display(&resources["mean_process_cpu_cores"], 1.0),
                display(&resources["lifetime_peak_rss_bytes"], 1073741824.0)
            );
            if let Some(classes) = p["classes"].as_object() {
                for (class, stats) in classes {
                    if let Some(p99) = stats["successful_latency_ms"]["p99"].as_f64() {
                        println!(
                            "  {class}: p99 {p99:.2} ms, {} completed, {} errors",
                            stats["completed"], stats["errors"]
                        );
                    }
                }
            }
        }
    }
    if report["resources"].is_object() {
        println!(
            "Global swap growth: {} MiB",
            display(&report["resources"]["max_global_swap_growth_bytes"], 1048576.0)
        );
    }
    if let Some(reason) = report["reason"].as_str() {
        println!("Reason: {reason}");
    }
}
