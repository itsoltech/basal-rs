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
//!   basal init          [--model REF]...                               basal-serve.yml for `basal serve`
//!   basal serve         [--model REF]... | [--config FILE]             HTTP server (default: ./basal-serve.yml)

mod bench;
mod choice_set;
mod compare;
mod config;
mod doctor;
mod export;
mod gemm_share;
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
    const GPU: [&str; 10] = [
        "decide",
        "export",
        "bench-requests",
        "eval-large-choice",
        "eval-choice-set",
        "profile",
        "serve",
        "gemm-search",
        "gemm",
        "bench",
    ];
    let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if !args.get(1).and_then(|a| a.to_str()).is_some_and(|c| GPU.contains(&c)) {
        return;
    }
    if let Some(bin) = paths::cuda_binary() {
        let err = cuda_command(&bin).args(&args[1..]).exec();
        crate::warn!("running {}: {err} (`basal doctor` checks the installation)", crate::term::P(&bin));
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
    /// Model: a local directory (config.json, basal.json, tokenizer.json, model.safetensors, CALIBRATION.json) or a
    /// Hugging Face repository: name (owner Remek; mini, 4.5B, max), owner/name, with @revision
    #[arg(long, default_value = ".models/basal-1.5-max")]
    model: String,
}

impl ModelArgs {
    /// The model directory (a repository is downloaded into the Hugging Face cache when missing).
    fn dir(&self) -> Result<PathBuf> {
        config::model_dir(&config::model_ref(&self.model)?)
    }
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
    /// Choice questions with more than 10 options from a labelled set (tools/choice-sets), with each grouped strategy
    EvalChoiceSet {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        /// JSONL: id, dataset, state, question, options, gold (index)
        #[arg(long)]
        set: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Strategies, e.g. luce2-1o,luce1-final,knockout1-1o
        #[arg(long, value_delimiter = ',', default_value = "luce2-1o")]
        strategies: Vec<String>,
        /// First N questions only
        #[arg(long)]
        limit: Option<usize>,
    },
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
        /// Strategy for more than 10 options: luce2, luce2-final, luce1-final, knockout1, knockout2, knockout3, each
        /// also with -1o (one option order before the final group); the default of the server is luce2-1o
        #[arg(long, default_value = "luce2-1o")]
        strategy: String,
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
        /// Configuration file (instead of BASAL_CONFIG or basal-serve.yml in the working directory)
        #[arg(long, conflicts_with = "models")]
        config: Option<PathBuf>,
        /// Model instead of a configuration file: name (owner Remek; mini, 4.5B, max), owner/name, @revision, or a
        /// local directory; repeat for several
        #[arg(long = "model")]
        models: Vec<String>,
        /// Machine-readable report
        #[arg(long)]
        json: bool,
    },
    /// Prepare this machine for `basal serve`: on Linux the CUDA libraries (from NVIDIA), the user configuration;
    /// optionally the models and a user service
    Setup {
        /// Configuration file (instead of BASAL_CONFIG or basal-serve.yml in the working directory)
        #[arg(long, conflicts_with = "models")]
        config: Option<PathBuf>,
        /// Model instead of a configuration file: name (owner Remek; mini, 4.5B, max), owner/name, @revision, or a
        /// local directory; repeat for several
        #[arg(long = "model")]
        models: Vec<String>,
        /// Download the CUDA libraries again
        #[arg(long)]
        force: bool,
        /// Download the models now (otherwise at the first `basal serve`)
        #[arg(long)]
        prefetch: bool,
        /// Write a user service running `basal serve` with these models or configuration (systemd on Linux, launchd
        /// on macOS)
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
    /// Send the GEMM tables generated on this machine for a GPU this build has none for to the basal-rs project (a
    /// GitHub issue per table, after a confirmation), so later builds ship them
    GemmShare {
        /// Directory of the tables (default: the GEMM cache of `basal serve`)
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Send without asking
        #[arg(long)]
        yes: bool,
        /// Only write the issue text next to each table
        #[arg(long)]
        dry_run: bool,
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
    /// Write a configuration file for `basal serve` (basal-serve.yml in the working directory)
    Init {
        /// Served models: name (owner Remek; mini, 4.5B, max), owner/name, @revision, or a local directory; repeat
        /// for several (default: basal-1.5-4.5B)
        #[arg(long = "model")]
        models: Vec<String>,
        /// Configuration file to write (default: basal-serve.yml in the working directory, which `basal serve` there
        /// reads)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Replace an existing file
        #[arg(long)]
        force: bool,
    },
    /// System One HTTP server: POST /v1/systemone, POST /v1/basal, GET /v1/models, GET /health. Models from --model,
    /// --config, basal-serve.yml in the working directory, else basal-1.5-4.5B from Hugging Face
    Serve {
        /// YAML file with the served models and their options (serve.example.yml); replaces the model, GPU and
        /// admission options below. Default: BASAL_CONFIG, else basal-serve.yml in the working directory
        #[arg(long, conflicts_with_all = ["models", "gemm_table"])]
        config: Option<PathBuf>,
        /// Model to serve: name (owner Remek; mini, 4.5B, max), owner/name, @revision, or a local directory; repeat
        /// for several (the options below apply to each). Without --model and a configuration file: basal-1.5-4.5B
        #[arg(long = "model")]
        models: Vec<String>,
        #[command(flatten)]
        g: GpuArgs,
        /// Listen address (default: the configuration's, BASAL_ADDR, 127.0.0.1:8000)
        #[arg(long)]
        addr: Option<std::net::SocketAddr>,
        /// Packed tokens of the requests admitted into one batch (a larger request runs alone)
        #[arg(long, default_value_t = 8192)]
        max_batch_tokens: usize,
        /// Waiting + running requests; more are refused with 529 (Retry-After: 1)
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
        /// One log line per HTTP request: method, status, path, time, queue and compute time (also `access_log: true`
        /// in the configuration or BASAL_ACCESS_LOG=1)
        #[arg(long)]
        access_log: bool,
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
    gpu_engine_at(&m.dir()?, g)
}

fn gpu_engine_at(dir: &Path, g: &GpuArgs) -> Result<Engine<GpuBackend>> {
    let manifest = ModelManifest::load(dir)?;
    let readout = match g.readout.as_str() {
        "f32" => Readout::F32,
        "bf16" => Readout::Bf16Rounded,
        r => bail!("unknown readout {r:?} (f32, bf16)"),
    };
    let backend =
        GpuBackend::load(dir, &manifest.config, Precision::parse(&g.dtype)?, readout, Kernels::parse(&g.kernels)?)?;
    crate::done!("{} loaded on the GPU in {:.1}s", manifest.name, backend.load_s);
    let mut backend = backend;
    if let Some(p) = &g.gemm_table {
        let n = backend.load_gemm_table(p)?;
        crate::note!("{n} GEMM classes from {}", crate::term::P(&p));
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
        bail!("{} exists; choose a new output path", crate::term::P(&path));
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

/// When this process started (serve logs the time to the first request it can take).
pub fn started() -> std::time::Instant {
    static T: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *T.get_or_init(std::time::Instant::now)
}

fn run() -> Result<()> {
    started();
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
            let engine = Engine::new(ModelManifest::load(&m.dir()?)?, NoBackend)?;
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
            let engine = Engine::new(ModelManifest::load(&m.dir()?)?, NoBackend)?;
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
        Cmd::EvalChoiceSet { m, g, set, out, strategies, limit } => {
            let mut engine = gpu_engine(&m, &g)?;
            choice_set::eval(&mut engine, &set, &strategies, limit, &out)?;
        }
        Cmd::EvalLargeChoice { m, g, reference, out, sizes, seeds, duty, strategy } => {
            let mut engine = gpu_engine(&m, &g)?;
            engine.large_choice_strategy = basal_core::large_choice::Strategy::parse(&strategy)
                .with_context(|| format!("unknown strategy {strategy:?}"))?;
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
        Cmd::Doctor { config, models, json } => {
            if !doctor::run(config, &models, json)? {
                std::process::exit(1);
            }
        }
        Cmd::Setup { config, models, force, prefetch, service } => {
            setup::run(&setup::Options { config, models, force, prefetch, service })?;
        }
        Cmd::Update { version, check } => update::run(version, check)?,
        Cmd::GemmShare { dir, yes, dry_run } => gemm_share::run(dir, yes, dry_run)?,
        Cmd::Uninstall { models, yes, dry_run } => uninstall::run(&uninstall::Options { models, yes, dry_run })?,
        Cmd::Init { models, out, force } => {
            let models = if models.is_empty() { vec![config::DEFAULT_MODEL.short.to_string()] } else { models };
            let models = models.iter().map(|m| config::model_ref(m)).collect::<Result<Vec<_>>>()?;
            let out = out.unwrap_or_else(|| PathBuf::from(paths::CONFIG_NAME));
            if out.exists() && !force {
                bail!("{} exists (`--force` replaces it)", crate::term::P(&out));
            }
            if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir).with_context(|| format!("creating {}", crate::term::P(&dir)))?;
            }
            std::fs::write(&out, config::template(&models))
                .with_context(|| format!("writing {}", crate::term::P(&out)))?;
            let gb: f64 = models.iter().filter_map(config::known_size_gb).sum();
            let models: Vec<String> = models.iter().map(config::describe).collect();
            crate::done!("wrote {} ({})", crate::term::P(&out), models.join(", "));
            if out == Path::new(paths::CONFIG_NAME) {
                crate::note!(
                    "`basal serve` in this directory uses it{}",
                    if gb > 0.0 {
                        format!("; {gb:.1} GB of models to download at the first start")
                    } else {
                        String::new()
                    }
                );
            } else {
                crate::note!("start: `basal serve --config {}`", out.display());
            }
        }
        Cmd::Serve {
            config,
            models,
            g,
            addr,
            max_batch_tokens,
            max_inflight,
            schedule,
            long_tokens,
            long_slice_ms,
            access_log,
            release_date,
        } => {
            let access_log = access_log || std::env::var("BASAL_ACCESS_LOG").is_ok_and(|v| !v.is_empty() && v != "0");
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
            let (source, mut c) = config::select(config, &models)?;
            match &source {
                config::Source::File(p) => crate::note!("configuration {}", crate::term::P(p)),
                config::Source::Models => {}
                config::Source::Builtin => crate::note!(
                    "no `--model` and no {} here: serving {} (`--model mini|4.5B|max` picks another)",
                    paths::CONFIG_NAME,
                    config::DEFAULT_MODEL.repo
                ),
            }
            if !matches!(source, config::Source::File(_)) {
                // models from the command line: the options of the command line
                for m in &mut c.models {
                    m.dtype = g.dtype.clone();
                    m.gemm_table = match &g.gemm_table {
                        Some(t) => t.display().to_string(),
                        None => "auto".into(),
                    };
                    m.max_batch_tokens = max_batch_tokens;
                    m.schedule = schedule;
                    m.long_tokens = long_tokens;
                    m.state_cache_mb = g.state_cache_mb;
                    m.tree_max_tokens = g.tree_max_tokens;
                    m.no_prefix_cache = g.no_prefix_cache;
                }
                c.max_inflight = max_inflight;
                c.long_slice_ms = long_slice_ms;
                c.release_date = release_date;
            }
            // model files (local or downloaded), then GEMM tables (a search needs no model weights in GPU memory),
            // then the models
            let dirs: Vec<PathBuf> = c.models.iter().map(config::model_dir).collect::<Result<_>>()?;
            let mut tables = Vec::new();
            for (mc, dir) in c.models.iter().zip(&dirs) {
                let manifest = ModelManifest::load(dir)?;
                tables.push(config::gemm_table(mc, &manifest, &c.gemm_cache)?);
            }
            let mut served = Vec::new();
            for ((mc, table), dir) in c.models.iter().zip(tables).zip(dirs) {
                let mg = GpuArgs {
                    dtype: mc.dtype.clone(),
                    readout: g.readout.clone(),
                    kernels: g.kernels.clone(),
                    no_prefix_cache: mc.no_prefix_cache,
                    state_cache_mb: mc.state_cache_mb,
                    tree_max_tokens: mc.tree_max_tokens,
                    gemm_table: table.clone(),
                };
                let engine = gpu_engine_at(&dir, &mg)?;
                if let Some(t) = &table {
                    config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), t)?;
                }
                served.push(serve::ServedModel {
                    engine,
                    max_batch_tokens: mc.max_batch_tokens,
                    schedule: mc.schedule,
                    long_tokens: mc.long_tokens,
                });
            }
            // --addr, else BASAL_ADDR (0.0.0.0:8000 in the container image), else the configuration's
            let addr = match (addr, std::env::var("BASAL_ADDR")) {
                (Some(a), _) => a,
                (None, Ok(v)) => v.parse().with_context(|| format!("BASAL_ADDR {v:?}"))?,
                (None, Err(_)) => c.addr,
            };
            let opts = serve::ServeOptions {
                addr,
                release_date: c.release_date,
                max_inflight: c.max_inflight,
                default_model: c.default_model,
                long_slice_ms: c.long_slice_ms,
                access_log: access_log || c.access_log,
            };
            serve::serve(served, opts)?;
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
                let shapes = basal_gpu::projection_shapes(&ModelManifest::load(&m.dir()?)?.config);
                let t = basal_gpu::gemm_search(&classes, &shapes, &dtype, invariant)?;
                write_json(&out, &t)?;
            }
            #[cfg(not(feature = "cuda"))]
            bail!(
                "gemm-search needs a CUDA build ({}, {dtype}, {}, {max_m}, {m_classes:?}, {invariant})",
                m.model,
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
                bail!("{} exists; choose a new output path", crate::term::P(&out));
            }
            let src = std::fs::read_to_string(&input).with_context(|| format!("reading {}", crate::term::P(&input)))?;
            let (mut lines, mut n) = (String::new(), 0usize);
            for (i, line) in src.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
                let rec: Value =
                    serde_json::from_str(line).with_context(|| format!("{}:{}", crate::term::P(&input), i + 1))?;
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
