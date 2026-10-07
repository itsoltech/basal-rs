//! `basal` command line.
//!
//!   basal decide        --model DIR [--request FILE|-] [--timing]      System One request -> response (Metal)
//!   basal prepare       --model DIR [--request FILE|-]                 items, prompts, token ids (no GPU)
//!   basal check-prompts --model DIR --reference DIR                    prompts/tokens vs the reference (no GPU)
//!   basal export        --model DIR --inputs DIR --out DIR             per-item results in the reference format (Metal)
//!   basal compare       --a DIR --b DIR --out FILE                     tokens, numerics, decisions, quality
//!   basal bench         --model DIR --reference DIR --out FILE         latency / throughput (Metal)
//!   basal doctor        [--json]                                       what this machine has / lacks for serve
//!   basal setup         [--prefetch] [--service]                       CUDA libraries (Linux), configuration
//!   basal update        [--version X] [--check]                        newest release over this installation
//!   basal uninstall     [--models] [--dry-run] [--yes]                 remove what basal put on this machine
//!   basal init          [--model mini|4.5B|max]                        user configuration of `basal serve`
//!   basal serve         [--config FILE | --model DIR]                  HTTP server (default: user configuration)

mod bench;
mod compare;
mod config;
mod doctor;
mod export;
mod large_eval;
mod paths;
mod reference;
mod serve;
mod setup;
mod stats;
mod term;
mod uninstall;
mod update;

use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use basal_core::pack::Packed;
use basal_core::{Backend, DecideError, Engine, ModelManifest};
use basal_gpu::{GpuBackend, Kernels, Precision, Readout};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};

/// `basal --version`: release, commit and GPU backend of the build.
pub fn version() -> &'static str {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let backend = match paths::cuda_binary() {
            Some(_) if !cfg!(feature = "cuda") => "CUDA through libexec/basal/basal-cuda",
            _ => basal_gpu::BUILD_BACKEND,
        };
        format!("{} ({}, {backend})", env!("CARGO_PKG_VERSION"), env!("BASAL_GIT_SHA"))
    })
}

/// The CUDA build of the Linux package with the CUDA libraries of `basal setup` on `LD_LIBRARY_PATH`.
pub fn cuda_command(bin: &Path) -> std::process::Command {
    let mut c = std::process::Command::new(bin);
    let lib = setup::cuda_lib_dir();
    if lib.exists() {
        let mut p = lib.into_os_string();
        if let Some(old) = std::env::var_os("LD_LIBRARY_PATH").filter(|v| !v.is_empty()) {
            p.push(":");
            p.push(old);
        }
        c.env("LD_LIBRARY_PATH", p);
    }
    c
}

/// Linux package: this binary (no CUDA) runs the GPU commands through the CUDA build next to it.
#[cfg(all(target_os = "linux", not(feature = "cuda")))]
fn delegate_gpu_command() {
    use std::os::unix::process::CommandExt;
    const GPU: [&str; 9] =
        ["decide", "export", "bench-requests", "eval-large-choice", "profile", "serve", "gemm-search", "gemm", "bench"];
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if !args.get(1).and_then(|a| a.to_str()).is_some_and(|c| GPU.contains(&c)) {
        return;
    }
    if let Some(bin) = paths::cuda_binary() {
        let err = cuda_command(&bin).args(&args[1..]).exec();
        crate::warn!("running {}: {err} (`basal doctor` checks the installation)", bin.display());
        std::process::exit(1);
    }
}

#[derive(Parser)]
#[command(name = "basal", version = version(), styles = term::help_styles(), about = "Runtime of basal decision models (Rust; CUDA and Metal)")]
struct Cli {
    /// Colours in the output: auto (a terminal, without NO_COLOR), always, never
    #[arg(long, global = true, value_enum, default_value_t = term::ColorMode::Auto)]
    color: term::ColorMode,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Local model directory (config.json, basal.json, tokenizer.json, model.safetensors, CALIBRATION.json)
    #[arg(long, default_value = ".models/basal-1.5-max")]
    model: PathBuf,
}

#[derive(Args, Clone)]
struct GpuArgs {
    /// Weight/activation precision: f16 (default; closest to f32 on the reference sets), bf16 (checkpoint dtype) or
    /// f32 (numerical reference, slow)
    #[arg(long, default_value = "f16")]
    dtype: String,
    /// Letter logits: f32, or bf16 (f32 rounded to bf16, diagnostic emulation of a bf16 lm_head)
    #[arg(long, default_value = "f32")]
    readout: String,
    /// Decoder kernels: fused (default) or candle (stock candle ops, first port; for comparison)
    #[arg(long, default_value = "fused")]
    kernels: String,
    /// Do not precompute the K/V of the prompt template prefix
    #[arg(long)]
    no_prefix_cache: bool,
    /// Budget (MiB) of the cross-request state cache: K/V of the shared prefix (template + state) of single-row
    /// forwards, reused on an exact token match. 0 = off (default; benchmarks then measure cold state).
    #[arg(long, default_value_t = 0)]
    state_cache_mb: usize,
    /// Longest shared-prefix (tree) row in tokens
    #[arg(long, default_value_t = basal_core::engine::TREE_MAX_TOKENS)]
    tree_max_tokens: usize,
    /// CUDA: cuBLASLt algorithm table from `basal gemm-search` (classes not in it are tuned at first use)
    #[arg(long)]
    gemm_table: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Answer one System One request and print the response
    Decide {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        /// Request JSON file, "-" for stdin
        #[arg(long, default_value = "-")]
        request: String,
        /// Print timing (stderr)
        #[arg(long)]
        timing: bool,
    },
    /// Print the prompts, token ids and packed rows of a request (no GPU)
    Prepare {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "-")]
        request: String,
    },
    /// Compare prompts, token ids, letters and packing with a reference export (no GPU)
    CheckPrompts {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Run the inputs of a reference export and write the results in the same format (for `compare`)
    Export {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        /// Directory whose bench.jsonl / systemone.jsonl provide the inputs (e.g. the Python reference)
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Forwards of the bench items: single (one question each), budget or tree
        #[arg(long, default_value = "single")]
        batching: String,
    },
    /// Latency of whole System One requests (decide), one client
    BenchRequests {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        #[arg(long)]
        requests: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        reps: usize,
        /// How the questions of a request are run: tree (shared state prefix), budget (one row per question, upstream
        /// chunks) or single (one forward per question)
        #[arg(long, default_value = "tree")]
        batching: String,
    },
    /// Evaluate the grouped 11..255 choice strategy on reference items with added distractor options
    EvalLargeChoice {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "11,20,40,100")]
        sizes: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "0,1")]
        seeds: Vec<u64>,
        /// Fraction of wall time the GPU computes: after each question sleep `t * (1 / duty - 1)` (limits heat)
        #[arg(long, default_value_t = 1.0)]
        duty: f64,
    },
    /// Per-section forward profile on reference items (synchronises after every section; diagnostic)
    Profile {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long, default_value_t = 10)]
        n: usize,
        /// Questions per forward (budget batching); 1 = one question per forward
        #[arg(long, default_value_t = 1)]
        batch: usize,
        /// Also record the largest |value| of the residual stream after every layer (f16 range check; its reduction
        /// adds GPU time to the profile)
        #[arg(long)]
        residual_max: bool,
    },
    /// GEMM tile/swizzle tuning on the projection shapes (diagnostic)
    GemmTune {
        #[arg(long, default_value = "f16")]
        dtype: String,
        #[arg(long, default_values_t = [136usize, 200])]
        m: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
    },
    /// Check this machine for `basal serve`: GPU, driver and libraries, configuration, models, disk, network, port
    Doctor {
        /// Machine-readable report
        #[arg(long)]
        json: bool,
    },
    /// Prepare this machine for `basal serve`: on Linux the CUDA libraries (from NVIDIA), the user configuration;
    /// optionally the models and a user service
    Setup {
        /// Download the CUDA libraries again
        #[arg(long)]
        force: bool,
        /// Download the configured models now (otherwise at the first `basal serve`)
        #[arg(long)]
        prefetch: bool,
        /// Write a user service running `basal serve` (systemd on Linux, launchd on macOS)
        #[arg(long)]
        service: bool,
    },
    /// Install the newest release (or --version) over this installation; Homebrew and container installations
    /// are updated by their own tools
    Update {
        /// Release to install (default: the newest)
        #[arg(long)]
        version: Option<String>,
        /// Only report whether a newer release exists
        #[arg(long)]
        check: bool,
    },
    /// Remove what basal put on this machine: user service, configuration, caches, CUDA libraries, the binaries of a
    /// package installation; with --models also the basal models in the Hugging Face cache
    Uninstall {
        /// Also the basal models (Remek/basal-*) in the Hugging Face cache
        #[arg(long)]
        models: bool,
        /// Do not ask
        #[arg(long, short)]
        yes: bool,
        /// Only list what would be removed
        #[arg(long)]
        dry_run: bool,
    },
    /// Write a server configuration for `basal serve` (default: the user configuration file)
    Init {
        /// Served models: mini, 4.5B, max (repeat for several; default: 4.5B)
        #[arg(long = "model")]
        models: Vec<String>,
        /// Configuration file to write (default: BASAL_CONFIG or serve.yml in the user configuration directory)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Replace an existing file
        #[arg(long)]
        force: bool,
    },
    /// System One HTTP server: POST /v1/systemone, POST /v1/basal, GET /v1/models, GET /health. Without --config and
    /// --model: the user configuration (`basal init`), else basal-1.5-4.5B from Hugging Face
    Serve {
        /// YAML file with the served models and their options (see crates/basal-cli/src/config.rs); replaces the
        /// model, GPU and admission options below
        #[arg(long, conflicts_with_all = ["model", "gemm_table"])]
        config: Option<PathBuf>,
        /// Local model directory (one model; the options below apply to it)
        #[arg(long)]
        model: Option<PathBuf>,
        #[command(flatten)]
        g: GpuArgs,
        #[arg(long, default_value = "127.0.0.1:8000")]
        addr: std::net::SocketAddr,
        /// Packed tokens of the requests admitted into one batch (a larger request runs alone)
        #[arg(long, default_value_t = 8192)]
        max_batch_tokens: usize,
        /// Waiting + running requests; more are refused with 503
        #[arg(long, default_value_t = 1024)]
        max_inflight: usize,
        /// Admission order of waiting requests
        #[arg(long, value_enum, default_value_t = serve::Schedule::Hrrn)]
        schedule: serve::Schedule,
        /// Requests above this many packed tokens run in a second lane that yields the GPU between layers to
        /// shorter requests (0 = one lane)
        #[arg(long, default_value_t = 4096)]
        long_tokens: usize,
        /// Least time the long lane works between two hand-overs to the main lane
        #[arg(long, default_value_t = 100)]
        long_slice_ms: u64,
        /// `release_date` reported by GET /v1/models (default: the release date upstream basal v1.5.0 reports)
        #[arg(long, default_value = "2026-10-05")]
        release_date: String,
    },
    /// CUDA: exhaustive cuBLASLt configuration search for the projection shapes; writes the table for --gemm-table
    GemmSearch {
        #[command(flatten)]
        m: ModelArgs,
        #[arg(long, default_value = "f16")]
        dtype: String,
        #[arg(long)]
        out: PathBuf,
        /// Largest M class (classes: multiples of 16 up to 512, then of 128)
        #[arg(long, default_value_t = 3072)]
        max_m: usize,
        /// M classes instead of the default ladder (--invariant: the 25 classes of config::INVARIANT_CLASSES, 16 to
        /// 16384; otherwise multiples of 16 up to 512, then of 128 up to --max-m)
        #[arg(long, value_delimiter = ',')]
        m_classes: Vec<usize>,
        /// One algorithm per weight shape for every M, without split-K: results of a row do not depend on the
        /// batch it runs in
        #[arg(long)]
        invariant: bool,
    },
    /// GEMM micro-benchmark on the projection shapes (diagnostic)
    Gemm {
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, default_values_t = [240usize, 480])]
        m: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
    },
    /// Compare two exports (reference or runtime): tokens, decisions, logit and probability differences, answers
    Compare {
        #[arg(long)]
        a: PathBuf,
        #[arg(long)]
        b: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// upstream `facts.inject` over a JSONL corpus: {"id", "state"} -> {"id", "out"} or {"id", "error"} (diagnostic,
    /// for tools/reference/facts_parity.py)
    #[command(hide = true)]
    Facts {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Single-decision latency and throughput with the basal-bench methodology
    Bench {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 39)]
        lat_n: usize,
        /// Batching of the throughput phase (groups of 16 questions): budget (upstream chunks) or tree
        #[arg(long, default_value = "budget")]
        batching: String,
    },
}

/// Backend for commands that only prepare inputs; any forward is an error (no silent fallback).
struct NoBackend;

impl Backend for NoBackend {
    fn describe(&self) -> Value {
        json!({"backend": "none"})
    }
    fn letter_logits(&mut self, _: &[&Packed], _: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        bail!("this command does not load the model")
    }
}

fn read_request(src: &str) -> Result<Value> {
    let mut s = String::new();
    if src == "-" {
        std::io::stdin().read_to_string(&mut s)?;
    } else {
        s = std::fs::read_to_string(src).with_context(|| format!("reading {src}"))?;
    }
    Ok(serde_json::from_str(&s)?)
}

fn gpu_engine(m: &ModelArgs, g: &GpuArgs) -> Result<Engine<GpuBackend>> {
    let manifest = ModelManifest::load(&m.model)?;
    let readout = match g.readout.as_str() {
        "f32" => Readout::F32,
        "bf16" => Readout::Bf16Rounded,
        r => bail!("unknown readout {r:?} (f32, bf16)"),
    };
    let backend = GpuBackend::load(
        &m.model,
        &manifest.config,
        Precision::parse(&g.dtype)?,
        readout,
        Kernels::parse(&g.kernels)?,
    )?;
    crate::done!("{} loaded on the GPU in {:.1}s", manifest.name, backend.load_s);
    let mut backend = backend;
    if let Some(p) = &g.gemm_table {
        let n = backend.load_gemm_table(p)?;
        crate::note!("{n} GEMM classes from {}", p.display());
    }
    backend.state_cache_bytes = g.state_cache_mb << 20;
    let mut engine = Engine::new(manifest, backend)?;
    engine.mark_state = g.state_cache_mb > 0;
    engine.tree_max_tokens = g.tree_max_tokens;
    if !g.no_prefix_cache {
        let lens = engine.warm_static_prefixes()?;
        crate::note!("template prefixes precomputed ({lens:?} tokens)");
    }
    Ok(engine)
}

fn parse_batching(s: &str) -> Result<basal_core::Batching> {
    Ok(match s {
        "single" => basal_core::Batching::Single,
        "budget" => basal_core::Batching::Budget,
        "tree" => basal_core::Batching::Tree,
        b => bail!("unknown batching {b:?} (single, budget, tree)"),
    })
}

fn write_json(path: &Path, v: &Value) -> Result<()> {
    if path.exists() {
        bail!("{} exists; choose a new output path", path.display());
    }
    std::fs::write(path, serde_json::to_string_pretty(v)? + "\n")?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        term::error(&e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    delegate_gpu_command();
    let cli = Cli::parse();
    term::set_mode(cli.color);
    match cli.cmd {
        Cmd::Decide { m, g, request, timing } => {
            let body = read_request(&request)?;
            let mut engine = gpu_engine(&m, &g)?;
            match engine.decide_timed(&body) {
                Ok((resp, t)) => {
                    println!("{}", serde_json::to_string_pretty(&resp)?);
                    if timing {
                        eprintln!("timing: {t}");
                    }
                }
                Err(DecideError::Internal(e)) => return Err(e),
                Err(e) => {
                    println!("{}", serde_json::to_string_pretty(&e.to_json())?);
                    std::process::exit(2);
                }
            }
        }
        Cmd::Prepare { m, request } => {
            let engine = Engine::new(ModelManifest::load(&m.model)?, NoBackend)?;
            match engine.prepare_request(&read_request(&request)?) {
                Ok(p) => println!("{}", serde_json::to_string_pretty(&p)?),
                Err(DecideError::Internal(e)) => return Err(e),
                Err(e) => {
                    println!("{}", serde_json::to_string_pretty(&e.to_json())?);
                    std::process::exit(2);
                }
            }
        }
        Cmd::CheckPrompts { m, reference, out } => {
            let engine = Engine::new(ModelManifest::load(&m.model)?, NoBackend)?;
            let report = compare::check_prompts(&engine, &reference)?;
            let t = &report["tokens"];
            eprintln!(
                "orders compared: {}, mismatches: {}, known deviations: {}",
                t["orders_compared"],
                t["mismatches"].as_array().map_or(0, Vec::len),
                t["known_deviations"].as_array().map_or(0, Vec::len)
            );
            match out {
                Some(p) => write_json(&p, &report)?,
                None => println!("{}", serde_json::to_string_pretty(&report)?),
            }
        }
        Cmd::Export { m, g, inputs, out, batching } => {
            let mut engine = gpu_engine(&m, &g)?;
            let load_s = engine.backend.load_s;
            export::export(
                &mut engine,
                &inputs,
                &out,
                parse_batching(&batching)?,
                json!({"dtype": g.dtype, "readout": g.readout, "kernels": g.kernels, "prefix_cache": !g.no_prefix_cache, "load_s": load_s}),
            )?;
        }
        Cmd::BenchRequests { m, g, requests, out, reps, batching } => {
            let mut engine = gpu_engine(&m, &g)?;
            engine.request_batching = match batching.as_str() {
                "tree" => basal_core::Batching::Tree,
                "budget" => basal_core::Batching::Budget,
                "single" => basal_core::Batching::Single,
                b => bail!("unknown batching {b:?}"),
            };
            let mut r = bench::bench_requests(&mut engine, &requests, reps)?;
            r["backend"] = engine.backend.describe();
            r["state_cache"] =
                json!({"hits": engine.backend.state_cache_hits, "inserts": engine.backend.state_cache_inserts});
            write_json(&out, &r)?;
        }
        Cmd::EvalLargeChoice { m, g, reference, out, sizes, seeds, duty } => {
            let mut engine = gpu_engine(&m, &g)?;
            let s = large_eval::eval(&mut engine, &reference, &sizes, &seeds, duty, &out)?;
            println!("{}", serde_json::to_string_pretty(&s["sizes"])?);
        }
        Cmd::Profile { m, g, reference, n, batch, residual_max } => {
            let mut engine = gpu_engine(&m, &g)?;
            let recs = reference::read_jsonl(&reference.join("bench.jsonl"))?;
            let prep: Vec<_> =
                recs.iter().take(n + 1).map(|r| engine.prepare(reference::bench_item(r)?)).collect::<Result<_>>()?;
            engine.run(&prep[..1], basal_core::Batching::Single)?; // warm-up
            engine.backend.profile = Some(Default::default());
            if residual_max {
                engine.backend.residual_max = Some(Default::default());
            }
            // items after the warm-up one (the reference may hold fewer than n + 1)
            let n = prep.len() - 1;
            let t = std::time::Instant::now();
            for c in prep[1..].chunks(batch) {
                engine.run(c, basal_core::Batching::Budget)?;
            }
            let total = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            let prof = engine.backend.profile.take().unwrap().into_inner();
            let rmax = engine.backend.residual_max.take().map(|r| r.into_inner()).unwrap_or_default();
            let nl = engine.manifest.config.num_layers;
            let per_layer: Vec<f32> =
                (0..nl).map(|l| rmax.iter().skip(l).step_by(nl).copied().fold(0.0f32, f32::max)).collect();
            let sum: f64 = prof.values().sum::<f64>() / n as f64;
            let tokens: f64 = prep[1..].iter().map(|p| p.packed.ids.len() as f64).sum::<f64>() / n as f64;
            println!(
                "{}",
                json!({"items": n, "mean_packed_tokens_with_template": tokens, "ms_per_item_profiled": total, "sections_ms_per_item": prof.iter().map(|(k, v)| (k.to_string(), json!(v / n as f64))).collect::<serde_json::Map<_, _>>(), "sections_sum_ms": sum, "residual_abs_max_per_layer": per_layer, "residual_abs_max": per_layer.iter().copied().fold(0.0f32, f32::max)})
            );
        }
        Cmd::GemmTune { dtype, m, reps } => {
            let shapes = [(2048, 2560), (2048, 2048), (2048, 22016), (11008, 2048)];
            #[cfg(target_os = "macos")]
            for r in basal_gpu::gemm_tune(&m, &shapes, &dtype, reps)? {
                println!("{r}");
            }
            #[cfg(not(target_os = "macos"))]
            anyhow::bail!("gemm-tune tunes the Metal GEMM kernels ({dtype}, {m:?}, {reps}, {shapes:?})");
        }
        Cmd::Doctor { json } => {
            if !doctor::run(json)? {
                std::process::exit(1);
            }
        }
        Cmd::Setup { force, prefetch, service } => {
            setup::run(&setup::Options { force, prefetch, service })?;
        }
        Cmd::Update { version, check } => update::run(version, check)?,
        Cmd::Uninstall { models, yes, dry_run } => uninstall::run(&uninstall::Options { models, yes, dry_run })?,
        Cmd::Init { models, out, force } => {
            let models: Vec<&config::KnownModel> = if models.is_empty() {
                vec![config::DEFAULT_MODEL]
            } else {
                models.iter().map(|m| config::known_model(m)).collect::<Result<_>>()?
            };
            let out = out.unwrap_or_else(paths::config_file);
            if out.exists() && !force {
                bail!("{} exists (`--force` replaces it)", out.display());
            }
            if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            }
            std::fs::write(&out, config::template(&models)).with_context(|| format!("writing {}", out.display()))?;
            let gb: f64 = models.iter().map(|m| m.size_gb).sum();
            crate::done!("wrote {} ({:.1} GB of models to download at the first `basal serve`)", out.display(), gb);
        }
        Cmd::Serve {
            config,
            model,
            g,
            addr,
            max_batch_tokens,
            max_inflight,
            schedule,
            long_tokens,
            long_slice_ms,
            release_date,
        } => {
            // once a day: is there a newer release (BASAL_NO_UPDATE_CHECK=1 turns it off)
            std::thread::spawn(|| {
                if let Some(v) = update::notice() {
                    crate::warn!(
                        "version {v} is available ({} installed): {}",
                        env!("CARGO_PKG_VERSION"),
                        update::how()
                    );
                }
            });
            // --config, else --model (one model, options from the command line), else the user configuration, else
            // the built-in one
            let c = match (config, &model) {
                (Some(path), _) => Some(config::ServeConfig::load(&path)?),
                (None, Some(_)) => None,
                (None, None) => {
                    let (path, c) = config::user_config()?;
                    match path {
                        Some(p) => crate::note!("configuration {}", p.display()),
                        None => {
                            let m = config::DEFAULT_MODEL;
                            crate::note!(
                                "no configuration ({} is missing; `basal init` writes one): serving {} ({:.1} \
                                 GB, downloaded at the first start)",
                                paths::config_file().display(),
                                m.repo,
                                m.size_gb
                            );
                        }
                    }
                    Some(c)
                }
            };
            if let Some(c) = c {
                // model files (local or downloaded), then GEMM tables (a search needs no model weights in GPU
                // memory), then the models
                let dirs: Vec<PathBuf> = c.models.iter().map(config::model_dir).collect::<Result<_>>()?;
                let mut tables = Vec::new();
                for (mc, dir) in c.models.iter().zip(&dirs) {
                    let manifest = ModelManifest::load(dir)?;
                    tables.push(config::gemm_table(mc, &manifest, &c.gemm_cache)?);
                }
                let mut models = Vec::new();
                for ((mc, table), dir) in c.models.iter().zip(tables).zip(dirs) {
                    let g = GpuArgs {
                        dtype: mc.dtype.clone(),
                        readout: "f32".into(),
                        kernels: "fused".into(),
                        no_prefix_cache: mc.no_prefix_cache,
                        state_cache_mb: mc.state_cache_mb,
                        tree_max_tokens: mc.tree_max_tokens,
                        gemm_table: table.clone(),
                    };
                    let engine = gpu_engine(&ModelArgs { model: dir }, &g)?;
                    if let Some(t) = &table {
                        config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), t)?;
                    }
                    models.push(serve::ServedModel {
                        engine,
                        max_batch_tokens: mc.max_batch_tokens,
                        schedule: mc.schedule,
                        long_tokens: mc.long_tokens,
                    });
                }
                // BASAL_ADDR (e.g. 0.0.0.0:8000 in the container image) overrides `addr`
                let addr = match std::env::var("BASAL_ADDR") {
                    Ok(v) => v.parse().with_context(|| format!("BASAL_ADDR {v:?}"))?,
                    Err(_) => c.addr,
                };
                let opts = serve::ServeOptions {
                    addr,
                    release_date: c.release_date,
                    max_inflight: c.max_inflight,
                    default_model: c.default_model,
                    long_slice_ms: c.long_slice_ms,
                };
                serve::serve(models, opts)?;
            } else {
                let engine = gpu_engine(&ModelArgs { model: model.expect("--model") }, &g)?;
                if let Some(t) = &g.gemm_table {
                    config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), t)?;
                }
                let model = serve::ServedModel { engine, max_batch_tokens, schedule, long_tokens };
                let opts = serve::ServeOptions { addr, release_date, max_inflight, default_model: None, long_slice_ms };
                serve::serve(vec![model], opts)?;
            }
        }
        Cmd::GemmSearch { m, dtype, out, max_m, m_classes, invariant } => {
            #[cfg(feature = "cuda")]
            {
                let classes: Vec<usize> = if !m_classes.is_empty() {
                    m_classes
                } else if invariant {
                    config::INVARIANT_CLASSES.to_vec()
                } else {
                    (16..=512).step_by(16).chain((640..=max_m).step_by(128)).collect()
                };
                let shapes = basal_gpu::projection_shapes(&ModelManifest::load(&m.model)?.config);
                let t = basal_gpu::gemm_search(&classes, &shapes, &dtype, invariant)?;
                write_json(&out, &t)?;
            }
            #[cfg(not(feature = "cuda"))]
            bail!(
                "gemm-search needs a CUDA build ({}, {dtype}, {}, {max_m}, {m_classes:?}, {invariant})",
                m.model.display(),
                out.display()
            );
        }
        Cmd::Gemm { dtype, m, reps } => {
            let shapes = [(2048, 2560), (2048, 2048), (2048, 11008), (11008, 2048)];
            for r in basal_gpu::gemm_microbench(&m, &shapes, &dtype, reps)? {
                println!("{r}");
            }
        }
        Cmd::Compare { a, b, out } => {
            let r = compare::compare(&a, &b)?;
            write_json(&out, &r)?;
            let n = &r["bench"];
            eprintln!(
                "tokens: {} mismatches; bench argmax avg {}, cal {}; max |logit diff| {}; max TV avg {}",
                r["tokens"]["mismatches"].as_array().map_or(0, Vec::len),
                n["argmax_agreement_avg"],
                n["argmax_agreement_cal"],
                n["letter_logit_diff"]["max"],
                n["tv_avg"]["max"]
            );
        }
        Cmd::Facts { input, out } => {
            if out.exists() {
                bail!("{} exists; choose a new output path", out.display());
            }
            let src = std::fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?;
            let (mut lines, mut n) = (String::new(), 0usize);
            for (i, line) in src.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                let rec: Value =
                    serde_json::from_str(line).with_context(|| format!("{}:{}", input.display(), i + 1))?;
                let state = basal_core::pyjson::text(&rec["state"]);
                let r = match basal_core::facts::try_inject(&state) {
                    Ok(s) => json!({"id": rec["id"], "out": s}),
                    Err(e) => json!({"id": rec["id"], "error": e.0}),
                };
                lines.push_str(&serde_json::to_string(&r)?);
                lines.push('\n');
                n += 1;
            }
            std::fs::write(&out, lines)?;
            eprintln!("facts: {n} states -> {}", out.display());
        }
        Cmd::Bench { m, g, reference, out, lat_n, batching } => {
            let mut engine = gpu_engine(&m, &g)?;
            let load_s = engine.backend.load_s;
            let batch = match batching.as_str() {
                "budget" => basal_core::Batching::Budget,
                "tree" => basal_core::Batching::Tree,
                b => bail!("unknown batching {b:?}"),
            };
            // GPU condition probe before and after: the same GEMM (f16, 480x2048 @ 11008x2048^T). A drop between the
            // two marks throttling or outside load during the run.
            let probe = || -> Result<f64> {
                let r = basal_gpu::gemm_microbench(&[480], &[(2048, 11008)], "f16", 30)?;
                Ok(r.iter().filter_map(|x| x["tflops"].as_f64()).fold(0.0, f64::max))
            };
            let before = probe()?;
            let mut r = bench::bench(&mut engine, &reference, lat_n, batch)?;
            r["gpu_probe_tflops"] = json!({"before": before, "after": probe()?});
            r["load_s"] = json!(load_s);
            r["backend"] = engine.backend.describe();
            r["model"] = json!(engine.manifest.name);
            write_json(&out, &r)?;
            eprintln!(
                "lat2 median {:.1} ms, p95 {:.1} ms, lat1 {:.1} ms, {:.3} dec/s, acc {}, GEMM probe {}",
                r["lat2_ms"].as_f64().unwrap_or(f64::NAN),
                r["lat2_p95_ms"].as_f64().unwrap_or(f64::NAN),
                r["lat1_ms"].as_f64().unwrap_or(f64::NAN),
                r["dec_s"].as_f64().unwrap_or(f64::NAN),
                r["acc"],
                r["gpu_probe_tflops"]
            );
        }
    }
    Ok(())
}
