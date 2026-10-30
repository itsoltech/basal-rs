//! Hardware characterization through the production HTTP router, batching scheduler and long-request lane.

mod http;
mod resources;
mod workload;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::GpuBackend;
use anyhow::{ensure, Context, Result};
use basal_core::{Backend, Engine, ModelManifest};
use basal_gpu::{PoolProbe, PoolUsage, Quantization};
use clap::{Args, ValueEnum};
use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::serve::{Replica, Schedule, ServeOptions, ServedModel};
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
    /// Minimum free VRAM / Metal working-set / Vulkan heap headroom (GiB)
    #[arg(long, default_value_t = 1.0)]
    gpu_reserve_gib: f64,
    /// Packed tokens admitted per server batch; same setting as serve
    #[arg(long, default_value_t = 8192)]
    max_batch_tokens: usize,
    /// Server waiting + running request limit; excess requests receive 529
    #[arg(long, default_value_t = 1024)]
    max_inflight: usize,
    /// Server limit for large/chunked bodies and expanded inputs above 128 KiB
    #[arg(long, default_value_t = crate::serve::DEFAULT_MAX_LARGE_INFLIGHT)]
    max_large_inflight: usize,
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
        ensure!(self.max_batch_tokens > 0 && self.long_slice_ms > 0, "server budgets / long-slice-ms must be positive");
        crate::serve::check_admission_limits(self.max_inflight, self.max_large_inflight)?;
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
    fn start(
        engine: Engine<GpuBackend>,
        devices: Vec<usize>,
        scheduling: crate::models::SchedulingPolicy,
        options: &Options,
    ) -> Result<Self> {
        let listener =
            std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).context("binding benchmark listener")?;
        let addr = listener.local_addr()?;
        let opts = ServeOptions {
            mcp: crate::mcp::Switch::Off,
            systemone_dialect: crate::serve::SystemOneDialect::TypeSafe,
            log_decisions: None,
            addr,
            release_date: "2026-10-05".into(),
            max_inflight: options.max_inflight,
            max_large_inflight: options.max_large_inflight,
            default_model: None,
            long_slice_ms: options.long_slice_ms,
            access_log: false,
            metrics: false,
            database: None,
            batches: None,
            auth: None,
            ui: None,
        };
        let model = ServedModel {
            replicas: vec![Replica { engine, devices, scheduling }],
            max_batch_tokens: options.max_batch_tokens,
            schedule: options.schedule,
            long_tokens: options.long_tokens,
            batch_profile: None,
            batch_qualification: crate::serve::batches::config::BackgroundQualification::not_configured(),
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

/// CUDA allocator-pool watermarks of the benchmark process (exact `used` / `reserved` counters of the pool every
/// tensor is allocated from), or the reason they are missing. Resets happen only outside measured intervals.
struct PoolTelemetry {
    probe: Option<PoolProbe>,
    report: Value,
}

impl PoolTelemetry {
    fn new(device: usize) -> Self {
        let scope = "CUDA stream-ordered allocator pool of the device (cuMemAllocAsync): Candle tensors, resident q8_0 or q4_k blocks, temporary f16 weights, cuBLASLt workspaces; not the CUDA context, modules, library handles or synchronous allocations. used and reserved are separate counters; *_high are watermarks since the preceding reset";
        match basal_gpu::pool_probe(device) {
            Ok(Some(probe)) => Self { probe: Some(probe), report: json!({"available": true, "scope": scope}) },
            Ok(None) => Self {
                probe: None,
                report: json!({"available": false, "gap": "no stream-ordered allocator pool (not CUDA, or the device does not support memory pools)"}),
            },
            Err(e) => Self { probe: None, report: json!({"available": false, "gap": format!("pool probe: {e:#}")}) },
        }
    }

    /// A failed reset or reading ends the telemetry with a recorded gap; the benchmark itself continues.
    fn failed(&mut self, stage: &str, e: &anyhow::Error) {
        self.probe = None;
        self.report["available"] = json!(false);
        self.report["gap"] = json!(format!("{stage}: {e:#}"));
    }

    fn reset(&mut self, stage: &str) {
        if let Some(Err(e)) = self.probe.as_ref().map(PoolProbe::reset_peaks) {
            self.failed(stage, &e);
        }
    }

    fn read(&mut self, stage: &str, synchronize: bool) -> Option<PoolUsage> {
        let probe = self.probe.as_ref()?;
        let usage = (|| {
            if synchronize {
                probe.synchronize()?;
            }
            probe.usage()
        })();
        match usage {
            Ok(u) => Some(u),
            Err(e) => {
                self.failed(stage, &e);
                None
            }
        }
    }
}

pub(crate) fn run(model: &ModelArgs, gpu_options: &GpuArgs, options: &Options) -> Result<()> {
    options.validate()?;
    // an unsupported weight format or layer split fails before any GPU check or download
    let quantization = gpu_options.quantization(&model.model)?;
    gpu_options.check_layer_split(&model.model)?;
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
    let result = execute(model, gpu_options, quantization, options, raw, &mut report);
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

/// The model directory (downloaded when missing), its checkpoint size (also reported) and manifest.
fn model_files(model: &ModelArgs, g: &GpuArgs, report: &mut Value) -> Result<(PathBuf, u64, ModelManifest)> {
    let mut configured = crate::config::model_ref(&model.model)?;
    configured.model_options = g.model_options.clone();
    let dir = if g.model_options.thinking_enabled()? || g.model_options.exits_enabled() {
        crate::config::model_dir(&configured)?
    } else {
        model.dir()?
    };
    let checkpoint_bytes = basal_core::checkpoint::bytes(&dir).context("checkpoint size")?;
    report["model"]["checkpoint_bytes"] = json!(checkpoint_bytes);
    let source = crate::config::model_ref(&model.model)?.source_name();
    let manifest = g.model_options.load_manifest(&dir, g.orders.as_deref(), &source)?;
    Ok((dir, checkpoint_bytes, manifest))
}

fn execute(
    model: &ModelArgs,
    g: &GpuArgs,
    quantization: Quantization,
    options: &Options,
    raw: workload::RawWorkload,
    report: &mut Value,
) -> Result<()> {
    // The guards' host baseline (host counters only, no device) precedes the model files, so swap growth during a
    // download still counts; the later checks sample the host again.
    let host = host_memory()?;
    // A layer split's coverage needs the model's config.json and is refused before any device context (each probe
    // below creates one), as in serve: its model files come next; one GPU keeps its device check before them.
    let split_files = if g.model_options.exits_enabled() {
        let files = model_files(model, g, report)?;
        let readout = g.readout()?;
        GpuBackend::check_early_exit(
            basal_gpu::Kernels::parse(&g.kernels)?,
            quantization,
            readout,
            g.layer_split.is_some(),
            g.media,
            &files.2.config,
        )?;
        let artifacts = g.model_options.load_exits(&files.2)?.context("trained exit artifacts")?;
        report["model"]["exit_head_resident_bytes"] =
            GpuBackend::early_exit_resident_bytes(&artifacts, &files.2.config)?.into();
        Some(files)
    } else {
        g.layer_split
            .as_ref()
            .map(|split| {
                let files = model_files(model, g, report)?;
                // Refuse unsupported architecture-specific split options before the preflight.
                crate::config::check_architecture(&files.2, quantization, true)?;
                crate::config::check_split_coverage(&files.2.name, split, files.2.config.num_layers())?;
                anyhow::Ok(files)
            })
            .transpose()?
    };
    // one GPU, or the stages of a layer split in order: the first one is the device the top-level fields describe,
    // `layer_split` reports every stage's device (only with a split)
    let devices = g.devices();
    let gpu = basal_gpu::gpu_memory(devices[0])?;
    report["device"] = json!(gpu);
    let mut guard = Guard::new(host.clone(), &gpu, options);
    let first = guard.check(&host, &gpu, 0);
    report["resources"] = guard.report();
    // the other stages' devices (a layer split)
    let mut others = Vec::new();
    for &d in &devices[1..] {
        let gpu = basal_gpu::gpu_memory(d)?;
        let mut guard = Guard::new(host.clone(), &gpu, options);
        let checked = guard.check(&host, &gpu, 0).with_context(|| format!("device {d}"));
        others.push((d, gpu, guard, checked));
    }
    if let Some(split) = &g.layer_split {
        report["layer_split"] = json!({"stages": split.stages(), "devices": []});
        report["layer_split"]["devices"] =
            json!(std::iter::once(json!({"device": devices[0], "gpu": gpu, "resources": guard.report()}))
                .chain(
                    others
                        .iter()
                        .map(|(d, gpu, guard, _)| json!({"device": d, "gpu": gpu, "resources": guard.report()}))
                )
                .collect::<Vec<_>>());
    }
    first?;
    for (_, _, _, checked) in &mut others {
        std::mem::replace(checked, Ok(()))?;
    }
    let (dir, checkpoint_bytes, manifest) = match split_files {
        Some(files) => files,
        None => model_files(model, g, report)?,
    };
    crate::check_thinking_gpu_options(g)?;
    let thinking = g.model_options.load_thinking(&manifest)?;
    let resident_models = if thinking.is_some() { 2 } else { 1 };
    if thinking.is_some() {
        report["model"]["thinking_checkpoint_bytes"] =
            basal_core::checkpoint::bytes(&manifest.dir.join("think"))?.into();
    }
    // quantization exists for Llama only: an Omni (Gemma) model is refused before the preflight
    crate::config::check_architecture(&manifest, quantization, g.layer_split.is_some())?;
    // none: the checkpoint size, as before; q8_0 / q4_k: the same with the projections at the format's block size;
    // Intel (always none): the resident weight buffers from the validated checkpoint header
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    let (weight_bytes, method) = (GpuBackend::estimated_weight_bytes(&dir, &manifest.config)?,
        "Intel: resident FP16 weight buffers estimated from the validated checkpoint header (embedding and lm_head rows are read on demand and excluded), not a measurement");
    #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
    let (weight_bytes, method) = (match &manifest.config {
            basal_core::architecture::Architecture::Llama(c) => basal_gpu::estimated_weight_bytes(c, checkpoint_bytes, quantization)?,
            // dense weights only (checked above)
            basal_core::architecture::Architecture::Gemma4(_) | basal_core::architecture::Architecture::Qwen35(_) => checkpoint_bytes,
        },
        "estimate from format arithmetic (checkpoint size; q8_0 / q4_k: projection matrices at Q8_0 / Q4_K block size with padding), not a measurement");
    // The supported pair has identical tensor shapes, file sizes and load precision, with independent weights.
    let weight_bytes = weight_bytes
        .checked_mul(resident_models)
        .and_then(|n| n.checked_add(report["model"]["exit_head_resident_bytes"].as_u64().unwrap_or(0)))
        .context("resident model/head weight estimate overflow")?;
    report["model"]["estimated_weight_bytes"] =
        json!({"bytes": weight_bytes, "quantization": quantization.as_str(), "method": method});
    report["model"]["estimated_resident_weight_bytes"] = json!(weight_bytes);
    match &g.layer_split {
        None => {
            let preflight = guard.check(&host_memory()?, &basal_gpu::gpu_memory(g.device())?, weight_bytes);
            report["resources"] = guard.report();
            preflight?;
        }
        Some(split) => {
            // per stage its part of the weights (arithmetic): the layers' share, the embedding on the first stage,
            // lm_head, final norm and the evidence head file on the last one
            let evidence = std::fs::metadata(dir.join("evidence_head.pt")).map_or(0, |m| m.len());
            let stages = split.stages();
            let last = stages.len() - 1;
            let mut checks = Vec::with_capacity(stages.len());
            for (i, stage) in stages.iter().enumerate() {
                let bytes = match &manifest.config {
                    basal_core::architecture::Architecture::Qwen35(_) => {
                        anyhow::bail!("Qwen3.5 does not support layer split")
                    }
                    basal_core::architecture::Architecture::Llama(cfg) => {
                        basal_gpu::estimated_stage_weight_bytes(
                            cfg,
                            checkpoint_bytes,
                            quantization,
                            stage.layers,
                            i == 0,
                            i == last,
                        )? + if i == last { evidence } else { 0 }
                    }
                    basal_core::architecture::Architecture::Gemma4(cfg) => GpuBackend::estimated_gemma_stage_bytes(
                        &dir,
                        cfg,
                        split.ranges(cfg.layers.len())?[i].clone(),
                        g.media,
                    )?,
                };
                let bytes = bytes.checked_mul(resident_models).context("dual-model stage estimate overflow")?;
                let stage_guard = if i == 0 { &mut guard } else { &mut others[i - 1].2 };
                let checked = stage_guard
                    .check(&host_memory()?, &basal_gpu::gpu_memory(stage.device)?, bytes)
                    .with_context(|| format!("stage {i} on device {}", stage.device));
                let entry = &mut report["layer_split"]["devices"][i];
                entry["estimated_weight_bytes"] = json!({"bytes": bytes, "layers": stage.layers,
                    "method": "resident checkpoint estimate for the stage: decoder layers, input embedding on the first, tied code embedding and final norm on the last, optional media encoders on the first; Llama projection format arithmetic and evidence head apply where present; not a runtime measurement"});
                entry["resources"] = stage_guard.report();
                checks.push(checked);
            }
            report["resources"] = guard.report();
            for c in checks {
                c?;
            }
        }
    }
    // Same automatic / bundled GEMM selection as serve, without ambient server configuration; a layer split resolves
    // it on its first device and checks it on every device before loading.
    let mut config = crate::config::model_ref(&model.model)?;
    report["model"]["resolved_revision"] = json!(config.revision);
    config.dtype = g.dtype.clone();
    if let Some(table) = &g.gemm_table {
        config.gemm_table = table.to_string_lossy().into_owned();
    }
    let mut gpu_settings = g.clone();
    let precision = crate::config::model_precision(&config.dtype, &manifest, devices[0])?;
    let resolved = crate::gemm_artifact::resolve(
        &crate::gemm_artifact::TableRequest::parse(&config.gemm_table)?,
        &manifest,
        crate::models::profile_for(&manifest),
        precision,
        &crate::config::default_gemm_cache(),
        &devices,
    )?;
    if let Some(split) = &g.layer_split {
        crate::config::check_table_on_devices(
            &manifest,
            resolved.table.as_ref(),
            &crate::config::Placement::Split(split.clone()),
        )?;
    }
    gpu_settings.gemm_table = resolved.path().map(Path::to_path_buf);
    gpu_settings.resolved_gemm = Some(resolved);
    // pool watermarks of the load alone: reset after the preflight and the GEMM table, outside load_s
    let mut pools: Vec<PoolTelemetry> = devices.iter().map(|&d| PoolTelemetry::new(d)).collect();
    for pool in &mut pools {
        pool.reset("reset before load");
    }
    let planned = crate::planned_engine_from_manifest(manifest.clone(), thinking, &gpu_settings)?;
    let scheduling = planned.scheduling();
    report["execution_plan"] = planned.plan.identity();
    let engine = planned.engine;
    // The prefix precomputation queues its temporaries' stream-ordered frees without a final synchronisation:
    // synchronise the context first (after load_s, before any measured work) so `used` counts only what stays.
    for pool in &mut pools {
        if let Some(u) = pool.read("reading after load", true) {
            pool.report["load_peak"] = json!({"used_bytes": u.used_high_bytes, "reserved_bytes": u.reserved_high_bytes,
                "scope": "load, q8_0 / q4_k warm-up (q4_k: with its host/GPU check of one matrix) and template prefix precomputation"});
            pool.report["resident_after_load"] = json!({"used_bytes": u.used_bytes, "reserved_bytes": u.reserved_bytes,
                "scope": "the loaded engine (weights, precomputed prefixes, workspaces) before any request, after a context synchronisation"});
        }
    }
    set_pools(report, &pools, g.layer_split.is_some());
    let table = gpu_settings.resolved_gemm.as_ref().and_then(|r| r.table.as_ref());
    if let Some(table) = table {
        crate::config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), table)?;
    }
    report["model"]["name"] = json!(engine.manifest.name);
    report["model"]["load_s"] = json!(engine.backend.load_s());
    report["backend"] = engine.backend.describe();
    // File name only: an automatic table lives in the user's cache directory, whose path identifies the machine; a
    // table that could not be stored (a read-only cache) by its digest.
    let gemm_table = table.map(|t| match t.path().and_then(Path::file_name) {
        Some(name) => name.to_string_lossy().into_owned(),
        None => format!("sha256:{}", t.digest()),
    });
    report["settings"] = json!({"dtype": g.dtype, "inference_mode": g.inference_mode, "quantization": quantization.as_str(), "kernels": g.kernels, "readout": g.readout, "prefix_cache": !g.no_prefix_cache,
        "state_cache_mb": g.state_cache_mb, "gemm_table": gemm_table, "tree_max_tokens": g.tree_max_tokens,
        "max_batch_tokens": options.max_batch_tokens, "max_inflight": options.max_inflight,
        "max_large_inflight": options.max_large_inflight, "schedule": format!("{:?}", options.schedule),
        "long_tokens": options.long_tokens, "long_slice_ms": options.long_slice_ms});
    if let Some(split) = &g.layer_split {
        report["settings"]["layer_split"] = json!(split.stages());
    }
    crate::model_options::record_effective_settings(
        &mut report["settings"],
        &engine.manifest,
        engine.auto_orders(),
        engine.context_overflow(),
    );
    report["settings"]["thinking"] = engine.thinking_profile();
    report["settings"]["early_exit"] =
        engine.backend.early_exit_caps().map(basal_core::early_exit::profile).unwrap_or(Value::Null);
    let probes = engine.backend.memory_probes();
    let host = host_memory()?;
    let gpu = engine.backend.memory()?;
    report["memory_after_load"] = json!({"host": host, "gpu": gpu});
    let loaded = guard.check(&host, &gpu, 0);
    report["resources"] = guard.report();
    loaded?;
    let mut monitored = vec![(devices[0], guard, engine.backend.memory_probe())];
    for ((d, _, mut other, _), (pd, probe)) in others.into_iter().zip(probes.into_iter().skip(1)) {
        anyhow::ensure!(d == pd, "memory probe of device {pd}, expected {d}");
        let loaded = other.check(&host_memory()?, &probe.snapshot()?, 0).with_context(|| format!("device {d}"));
        report["layer_split"]["devices"][monitored.len()]["resources"] = other.report();
        loaded?;
        monitored.push((d, other, probe));
    }
    // runtime watermarks: from here through the drained shutdown of every phase
    for pool in &mut pools {
        pool.reset("reset before the workloads");
    }
    // layer split: transfers of the workloads (warm-up requests included), from here through the shutdown
    let transfers = engine.backend.stage_transfers();
    if let Some(t) = &transfers {
        t.reset();
    }
    let monitor = Monitor::start(monitored)?;
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
    let server = Server::start(engine, devices.clone(), scheduling, options)?;
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
    // The server threads are joined: the engine and its forks are dropped (the monitor keeps only a device handle),
    // so after a context synchronisation (pending stream-ordered frees executed) `used` is what outlived them.
    for pool in &mut pools {
        if let Some(u) = pool.read("reading after shutdown", true) {
            pool.report["runtime_peak"] = json!({"used_bytes": u.used_high_bytes, "reserved_bytes": u.reserved_high_bytes,
                "scope": "every phase of this run, warm-up requests and the drained shutdown; not per phase"});
            pool.report["after_shutdown"] = json!({"used_bytes": u.used_bytes, "reserved_bytes": u.reserved_bytes,
                "scope": "left after the engine was dropped and the context synchronised (leak diagnostic, not resident memory)"});
        }
    }
    set_pools(report, &pools, g.layer_split.is_some());
    if let Some(t) = &transfers {
        report["layer_split"]["transfers"] = t.report();
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

/// The pool telemetry of the report: the first device's as `memory_pool`; with a layer split also every stage
/// device's in `layer_split.devices`.
fn set_pools(report: &mut Value, pools: &[PoolTelemetry], split: bool) {
    if let Some(first) = pools.first() {
        report["memory_pool"] = first.report.clone();
    }
    if split {
        for (i, pool) in pools.iter().enumerate() {
            report["layer_split"]["devices"][i]["memory_pool"] = pool.report.clone();
        }
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
    let pool = &report["memory_pool"];
    if pool["available"].as_bool() == Some(true) {
        println!(
            "GPU allocator pool (used / reserved GiB): load peak {} / {}, after load {} / {}, runtime peak {} / {}",
            display(&pool["load_peak"]["used_bytes"], 1073741824.0),
            display(&pool["load_peak"]["reserved_bytes"], 1073741824.0),
            display(&pool["resident_after_load"]["used_bytes"], 1073741824.0),
            display(&pool["resident_after_load"]["reserved_bytes"], 1073741824.0),
            display(&pool["runtime_peak"]["used_bytes"], 1073741824.0),
            display(&pool["runtime_peak"]["reserved_bytes"], 1073741824.0)
        );
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
