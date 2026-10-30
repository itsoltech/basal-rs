//! `basal` command line.
//!
//!   basal client        --ask QUESTION | --template FILE | --request FILE   stdin -> decisions (HTTP or --local)
//!   basal decide        --model DIR [--request FILE|-] [--timing]      System One request -> response (Metal)
//!   basal prepare       --model DIR [--request FILE|-]                 items, prompts, token ids (no GPU)
//!   basal check-prompts --model DIR --reference DIR                    prompts/tokens vs the reference (no GPU)
//!   basal export        --model DIR --inputs DIR --out DIR             per-item results in the reference format (Metal)
//!   basal compare       --a DIR --b DIR --out FILE                     tokens, numerics, decisions, quality
//!   basal bench         --model DIR --reference DIR --out FILE         latency / throughput (Metal)
//!   basal benchmark     --model REF [--requests FILE] [--json]         sequential and mixed concurrent HTTP traffic
//!   basal doctor        [--json]                                       what this machine has / lacks for serve
//!   basal setup         [--prefetch] [--service]                       CUDA libraries (Linux), configuration
//!   basal update        [--version X] [--check]                        newest release over this installation
//!   basal uninstall     [--models] [--dry-run] [--yes]                 remove what basal put on this machine
//!   basal init          [--model REF]...                               basal-serve.yml for `basal serve`
//!   basal serve         [--model REF]... | [--config FILE]             HTTP server (default: ./basal-serve.yml)

mod batches;
mod bench;
mod benchmark;
mod choice_set;
mod client;
mod compare;
mod config;
mod config_support;
mod database;
mod doctor;
mod export;
mod gemm_artifact;
mod gemm_share;
mod keys;
mod large_eval;
mod mcp;
mod model_options;
mod models;
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
use basal_gpu::RuntimeBackend as GpuBackend;
use basal_gpu::{InferenceMode, Kernels, LayerSplit, Quantization, Readout};
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
fn delegate_gpu_command(cmd: &Cmd) {
    use std::os::unix::process::CommandExt;
    // Parse the command first: global flags may precede its name.
    if !matches!(
        cmd,
        Cmd::Decide { .. }
            | Cmd::Export { .. }
            | Cmd::BenchRequests { .. }
            | Cmd::EvalLargeChoice { .. }
            | Cmd::EvalChoiceSet { .. }
            | Cmd::Profile { .. }
            | Cmd::Serve { .. }
            | Cmd::GemmSearch { .. }
            | Cmd::Gemm { .. }
            | Cmd::GemmShare { .. }
            | Cmd::Bench { .. }
            | Cmd::Benchmark { .. }
    ) {
        return;
    }
    if let Some(bin) = paths::cuda_binary() {
        let args: Vec<std::ffi::OsString> = std::env::args_os().collect();
        let err = cuda_command(&bin).args(&args[1..]).exec();
        crate::warn!("running {}: {err} (`basal doctor` checks the installation)", crate::term::P(&bin));
        std::process::exit(1);
    }
}

#[derive(Parser)]
#[command(name = "basal", version = version(), styles = term::help_styles(), about = "Runtime of basal decision models (Rust; CUDA, Metal and Intel Vulkan)")]
struct Cli {
    /// Colours in the output: auto (a terminal, without NO_COLOR), always, never
    #[arg(long, global = true, value_enum, default_value_t = term::ColorMode::Auto)]
    color: term::ColorMode,
    /// /v1/systemone dialect for serve; default typesafe (YAML can also select it)
    #[arg(long, global = true, value_enum)]
    systemone_dialect: Option<serve::SystemOneDialect>,
    /// Append final online decisions to this JSONL file (serve only; YAML log_decisions)
    #[arg(long, global = true)]
    log_decisions: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct ModelArgs {
    /// Model: a local directory (config.json, basal.json, tokenizer.json, model.safetensors, CALIBRATION.json) or a
    /// Hugging Face repository: name (owner Remek; pico, mini, 4.5B, max), owner/name, with @revision
    #[arg(long, default_value = ".models/basal-1.5-max")]
    model: String,
}

impl ModelArgs {
    /// The model directory (a repository is downloaded into the Hugging Face cache when missing).
    fn dir(&self) -> Result<PathBuf> {
        config::model_dir(&config::model_ref(&self.model)?)
    }
}

const INFERENCE_MODES: [&str; 3] = ["auto", "accurate", "balanced"];

#[derive(Args, Clone)]
struct GpuArgs {
    #[command(flatten)]
    model_options: model_options::ModelOptions,
    /// Enable image/audio processors and native encoders (omni CUDA; BF16/F32 encoders)
    #[arg(long)]
    media: bool,
    /// Option orders: 1, 2, auto or `2le<N>`; default 2 for format 1, 1 for format 2
    #[arg(long)]
    orders: Option<String>,
    /// Weight/activation precision: auto (Llama f16; Gemma profile default; Qwen3.5 CUDA bf16, Metal f32), f16, bf16 or f32
    #[arg(long, default_value = "auto")]
    dtype: String,
    /// Attention arithmetic: auto, accurate (Llama original, Gemma FP32 products), or balanced (CUDA f16 Llama)
    #[arg(long, default_value = "auto", value_parser = INFERENCE_MODES)]
    inference_mode: String,
    /// Letter logits: f32, or bf16 (f32 rounded to bf16, diagnostic emulation of a bf16 lm_head)
    #[arg(long, default_value = "f32")]
    readout: String,
    /// Decoder kernels: fused (default) or candle (stock candle ops, first port; for comparison)
    #[arg(long, default_value = "fused")]
    kernels: String,
    /// Projection weights: none (default; dense weights of --dtype), q8_0 (8-bit GGML Q8_0 blocks) or q4_k (4-bit GGML
    /// Q4_K blocks), dequantized to f16 per projection: less GPU memory, more work per forward, different numerics;
    /// CUDA with --dtype f16 (auto for Llama) and --kernels fused; Llama models only (Omni is refused). `basal serve`:
    /// the format of every --model (a configuration file sets `quantization` per model instead)
    #[arg(long)]
    quantization: Option<String>,
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
    /// The GEMM table acquired by the first plan phase (`gemm_artifact::resolve`, serve and benchmark), with its
    /// digest and provenance; takes the place of `gemm_table`.
    #[arg(skip)]
    resolved_gemm: Option<gemm_artifact::ResolvedGemm>,
    /// GPU: CUDA device ordinal among the GPUs visible to this process (CUDA_VISIBLE_DEVICES renumbers them); Metal
    /// and Intel: 0 only. Default 0. `basal serve`: the device of every --model (a configuration file sets `device`
    /// per model instead)
    #[arg(long)]
    device: Option<usize>,
    /// CUDA: one model on several GPUs by consecutive layer ranges, DEVICE:LAYERS pairs in layer order (e.g. 0:30,1:30:
    /// layers 0-29 and the embedding on device 0, layers 30-59 and the head on device 1). At least two distinct GPUs
    /// of one type, counts adding up to the model's layers; Llama models only (Omni is refused); --dtype f16 (auto for
    /// Llama) and --kernels fused; instead of --device. The
    /// same computations as on one GPU, stage after stage. `basal serve`: the split of every --model (a configuration
    /// file sets `layer_split` per model instead)
    #[arg(long, conflicts_with = "device")]
    layer_split: Option<LayerSplit>,
}

impl GpuArgs {
    /// The GPU ordinal (`--device`, default 0).
    fn device(&self) -> usize {
        self.device.unwrap_or(0)
    }

    /// The devices the model uses: the --layer-split stages, else `[--device]`.
    fn devices(&self) -> Vec<usize> {
        self.layer_split.as_ref().map_or_else(|| vec![self.device()], LayerSplit::devices)
    }

    /// `--layer-split` of `model` against this build's backend, --dtype and --kernels, then its GPUs against the
    /// driver (every ordinal visible, one GPU type), before any download, GEMM table or load.
    fn check_layer_split(&self, model: &str) -> Result<()> {
        let Some(split) = &self.layer_split else { return Ok(()) };
        config::check_layer_split(model, split, &self.dtype, &self.kernels, cfg!(feature = "cuda"))?;
        config::check_same_gpus(model, &config::Placement::Split(split.clone()))
    }

    /// `--quantization` (default none) of `model`, checked with --dtype and --kernels for this build's backend.
    fn quantization(&self, model: &str) -> Result<Quantization> {
        config::check_quantization(
            model,
            self.quantization.as_deref().unwrap_or(Quantization::None.as_str()),
            &self.dtype,
            &self.kernels,
            cfg!(feature = "cuda"),
        )
    }

    /// `--readout` (f32, bf16).
    fn readout(&self) -> Result<Readout> {
        match self.readout.as_str() {
            "f32" => Ok(Readout::F32),
            "bf16" => Ok(Readout::Bf16Rounded),
            r => bail!("unknown readout {r:?} (f32, bf16)"),
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// MCP stdio bridge to a running basal server; does not load models or start a server
    Mcp(mcp::Args),
    /// Manage durable batch storage without loading a model or a GPU.
    Batches(batches::Args),
    /// Manage the configured database (`database:`) without loading a model or a GPU.
    Database(database::cli::Args),
    /// Manage API users (`auth:`) in the configured database without loading a model or a GPU.
    Users(keys::UsersArgs),
    /// Manage API keys of the users (`auth:`) without loading a model or a GPU.
    Keys(keys::KeysArgs),
    /// Use decisions in pipes and scripts: HTTP by default, --local loads a model once
    Client(Box<client::ClientArgs>),
    /// Measure sequential and mixed concurrent HTTP traffic on this hardware
    Benchmark {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        g: GpuArgs,
        #[command(flatten)]
        options: benchmark::Options,
    },
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
        /// Prepare the upstream request dialect, including multi branches
        #[arg(long)]
        upstream: bool,
        /// Option orders: 1, 2, auto or `2le<N>`
        #[arg(long)]
        orders: Option<String>,
        #[command(flatten)]
        model_options: model_options::ModelOptions,
        #[arg(long, default_value = "-")]
        request: String,
    },
    /// Compare prompts, token ids, letters and packing with a reference export (no GPU)
    CheckPrompts {
        #[command(flatten)]
        m: ModelArgs,
        #[command(flatten)]
        model_options: model_options::ModelOptions,
        /// Option orders: 1, 2, auto or `2le<N>`
        #[arg(long)]
        orders: Option<String>,
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
        /// Prepare/export the upstream endpoint dialect (default: TypeSafe)
        #[arg(long)]
        upstream: bool,
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
    /// Prepare this machine for `basal serve`: CUDA libraries for Linux CUDA builds, the user configuration;
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
    /// Print the SHA-256 of the normalized content of GEMM tables (the identity `gemm_table: sha256:<digest>` pins);
    /// with --import also store them in the content-addressed GEMM cache
    GemmDigest {
        /// Table files (`basal gemm-search` output, a bundled table, an earlier cache file)
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Store each table under its digest in the GEMM cache (never replacing a stored artifact)
        #[arg(long)]
        import: bool,
        /// GEMM cache of --import (default: the GEMM cache of `basal serve`)
        #[arg(long, requires = "import")]
        dir: Option<PathBuf>,
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
        /// Served models: pico, 2.0-mini/2.0-4.5B/2.0-max, omni-mini/omni-max, name (owner Remek), owner/name,
        /// @revision, or a local directory; mini/4.5B/max select basal-1.5. Repeat
        /// for several (default: basal-2.0-4.5B)
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
    /// --config, basal-serve.yml in the working directory, else basal-2.0-4.5B from Hugging Face
    Serve {
        /// Serve stateless streamable HTTP MCP at /mcp (default: off; YAML mcp)
        #[arg(long, value_enum)]
        mcp: Option<mcp::Switch>,
        /// YAML file with the served models and their options (serve.example.yml); replaces the model, GPU and
        /// admission options below. Default: BASAL_CONFIG, else basal-serve.yml in the working directory
        #[arg(long, conflicts_with_all = ["models", "gemm_table", "device", "replica_devices", "layer_split"])]
        config: Option<PathBuf>,
        /// Model to serve: pico, 2.0-mini/2.0-4.5B/2.0-max, omni-mini/omni-max, name (owner Remek), owner/name,
        /// @revision, or a local directory; mini/4.5B/max select basal-1.5. Repeat
        /// for several (the options below apply to each). Without --model and a configuration file: basal-2.0-4.5B
        #[arg(long = "model")]
        models: Vec<String>,
        #[command(flatten)]
        g: GpuArgs,
        /// CUDA: independent replicas of every --model under its one name, one per listed device (e.g. 0,1; distinct
        /// GPUs of one type, instead of --device). A configuration file sets `replica_devices` per model instead
        #[arg(long, value_delimiter = ',', conflicts_with_all = ["device", "layer_split"])]
        replica_devices: Option<Vec<usize>>,
        /// Listen address (default: the configuration's, BASAL_ADDR, 127.0.0.1:8000)
        #[arg(long)]
        addr: Option<std::net::SocketAddr>,
        /// Packed tokens of the requests admitted into one batch (a larger request runs alone)
        #[arg(long, default_value_t = 8192)]
        max_batch_tokens: usize,
        /// Waiting + running requests; more are refused with 529 (Retry-After: 1)
        #[arg(long, default_value_t = 1024)]
        max_inflight: usize,
        /// Large/chunked bodies and expanded input above 128 KiB; more are refused with 529. Increase only with
        /// enough host memory for their concurrent preparation and queued inputs
        #[arg(long, default_value_t = serve::DEFAULT_MAX_LARGE_INFLIGHT)]
        max_large_inflight: usize,
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
        /// Enable Prometheus metrics at GET /metrics (also `metrics: true` in the configuration; default: off)
        #[arg(long)]
        metrics: bool,
        /// Enable the durable Batch/Files API (local SQLite profile by default).
        #[arg(long)]
        batches: bool,
        /// Directory of durable batch data (with --batches or batches.enabled in YAML).
        #[arg(long)]
        batch_data_dir: Option<PathBuf>,
        /// Require API keys (`basal keys`) on every route except /health (also `auth.enabled` in YAML); needs a
        /// `database:` section or --batches for the local SQLite default
        #[arg(long)]
        auth: bool,
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
        /// CUDA device ordinal to search on (the table is valid for GPUs of its name)
        #[arg(long, default_value_t = 0)]
        device: usize,
    },
    /// GEMM micro-benchmark on the projection shapes (diagnostic)
    Gemm {
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long, default_values_t = [240usize, 480])]
        m: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        /// GPU ordinal
        #[arg(long, default_value_t = 0)]
        device: usize,
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
    // an unsupported weight format or layer split fails before the model is downloaded
    g.quantization(&m.model)?;
    g.check_layer_split(&m.model)?;
    let mut model = config::model_ref(&m.model)?;
    model.model_options = g.model_options.clone();
    gpu_engine_at(&config::model_dir(&model)?, g, &model.source_name())
}

/// Prompt planning without a GPU: the manifest with `options` and their adaptive-order thresholds, no weights.
fn planning_engine(
    m: &ModelArgs,
    orders: Option<&str>,
    options: &model_options::ModelOptions,
) -> Result<Engine<NoBackend>> {
    anyhow::ensure!(
        !options.exits_enabled(),
        "this prepare-only command has no trained-head backend; use the loaded engine for early_exit planning"
    );
    let mut model = config::model_ref(&m.model)?;
    model.model_options = options.clone();
    let manifest = options.load_manifest(&config::model_dir(&model)?, orders, &model.source_name())?;
    let thinking = options.load_thinking(&manifest)?;
    let mut engine = Engine::new(manifest, NoBackend)?;
    if let Some(t) = thinking {
        engine.set_thinking(basal_core::thinking::Thinking::new(
            t.manifest,
            NoBackend,
            t.settings,
            t.eos,
            t.calibration_label,
            t.sources,
        )?);
    }
    engine.set_auto_orders(options.auto_orders()?);
    engine.set_context_overflow(options.overflow()?);
    Ok(engine)
}

fn parse_inference_mode(value: &str) -> Result<InferenceMode> {
    Ok(match value {
        "auto" => InferenceMode::Auto,
        "accurate" => InferenceMode::Accurate,
        "balanced" => InferenceMode::Balanced,
        value => bail!("unknown inference mode {value:?} ({})", INFERENCE_MODES.join(", ")),
    })
}

/// An engine with the frozen plan and profile it was loaded with.
pub(crate) struct PlannedEngine {
    pub engine: Engine<GpuBackend>,
    pub plan: basal_gpu::ResolvedExecutionPlan,
    pub profile: &'static models::Profile,
}

impl PlannedEngine {
    /// The scheduling policy of `basal serve` for this instance: the profile's, with the features of the plan.
    pub fn scheduling(&self) -> models::SchedulingPolicy {
        self.profile.scheduling.resolve(&self.plan)
    }
}

fn gpu_engine_at(dir: &Path, g: &GpuArgs, source: &str) -> Result<Engine<GpuBackend>> {
    Ok(planned_engine_at(dir, g, source)?.engine)
}

/// Reject unsupported opt-in thinking settings before checkpoint digests or GPU weight loading.
fn check_thinking_gpu_options(g: &GpuArgs) -> Result<()> {
    if g.model_options.think_numerics()? == model_options::ThinkNumerics::Inherit {
        return Ok(());
    }
    let reproducibility = match &g.resolved_gemm {
        Some(gemm) => gemm.reproducibility,
        None => gemm_artifact::TableRequest::parse(
            &g.gemm_table.as_ref().map(|p| p.to_string_lossy()).unwrap_or("none".into()),
        )?
        .reproducibility(),
    };
    g.model_options.check_think_execution(
        Kernels::parse(&g.kernels)?,
        g.quantization.as_deref().unwrap_or("none").parse()?,
        parse_inference_mode(&g.inference_mode)?,
        g.layer_split.is_some(),
        reproducibility,
    )
}

/// Load the model of `dir` with `g`: resolve its profile, freeze the execution plan (precision, kernels, attention,
/// GEMM table and fallbacks for the instance's GPUs) before any weight is loaded, load the backend from the plan,
/// then precompute the template prefixes.
fn planned_engine_at(dir: &Path, g: &GpuArgs, source: &str) -> Result<PlannedEngine> {
    check_thinking_gpu_options(g)?;
    let manifest = g.model_options.load_manifest(dir, g.orders.as_deref(), source)?;
    let thinking = g.model_options.load_thinking(&manifest)?;
    planned_engine_from_manifest(manifest, thinking, g)
}

fn planned_engine_from_manifest(
    manifest: ModelManifest,
    thinking: Option<model_options::LoadedThinking>,
    g: &GpuArgs,
) -> Result<PlannedEngine> {
    check_thinking_gpu_options(g)?;
    let dir = manifest.dir.as_path();
    let inference_mode = parse_inference_mode(&g.inference_mode)?;
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    anyhow::ensure!(g.state_cache_mb == 0, "Intel state cache is not implemented; use --state-cache-mb 0");
    let quantization = g.quantization(&manifest.name)?;
    // Validate architecture-specific quantization and split constraints before any weight is read.
    config::check_architecture(&manifest, quantization, g.layer_split.is_some())?;
    let readout = g.readout()?;
    let profile = models::profile_for(&manifest);
    let kernels = Kernels::parse(&g.kernels)?;
    // Before precision/device/GEMM work below, heads are validated from the exact published model.
    if g.model_options.exits_enabled() {
        GpuBackend::check_early_exit(
            kernels,
            quantization,
            readout,
            g.layer_split.is_some(),
            g.media,
            &manifest.config,
        )?;
    }
    let exits = g.model_options.load_exits(&manifest)?;
    if let Some(artifacts) = &exits {
        GpuBackend::early_exit_resident_bytes(artifacts, &manifest.config)?;
    }
    let precision = config::model_precision(&g.dtype, &manifest, g.devices()[0])?;
    if let Some(split) = &g.layer_split {
        // the stages must cover the model's layers exactly: checked before any weight is read
        config::check_split_coverage(&manifest.name, split, manifest.config.num_layers())?;
    }
    let gemm = match (&g.resolved_gemm, &g.gemm_table) {
        (Some(resolved), _) => resolved.clone(),
        // `--gemm-table PATH`, or `pinned` / `sha256:<digest>` / `auto` resolved in the user's GEMM cache
        (None, Some(value)) => gemm_artifact::resolve(
            &gemm_artifact::TableRequest::parse(&value.to_string_lossy())?,
            &manifest,
            profile,
            precision,
            &config::default_gemm_cache(),
            &g.devices(),
        )?,
        (None, None) => gemm_artifact::ResolvedGemm::none(
            basal_gpu::cublaslt_version().is_some() && precision != basal_gpu::Precision::F32,
        ),
    };
    let options = basal_gpu::PlanOptions {
        mode: inference_mode,
        precision,
        readout,
        kernels,
        quantization,
        reproducibility: gemm.reproducibility,
    };
    let plan = basal_gpu::ResolvedExecutionPlan::freeze(
        profile.id,
        profile.kernel_set,
        &profile.dispatch,
        &manifest.config,
        &g.devices(),
        options,
        gemm.table.clone(),
        gemm.tables_apply,
    )
    .with_context(|| format!("model {} (profile {})", manifest.name, profile.id))?;
    crate::note!("{}: execution plan {}", manifest.name, plan_summary(&plan));
    let placement = match &g.layer_split {
        None => basal_gpu::PlannedPlacement::Single(g.device()),
        Some(split) => basal_gpu::PlannedPlacement::Split(split),
    };
    let thinking = thinking
        .map(|mut t| {
            let think_profile = models::profile_for(&t.manifest);
            let bf16 = t.numerics == model_options::ThinkNumerics::UpstreamBf16Pointwise;
            let (kernel_set, precision, readout, table, tables_apply) = if bf16 {
                let lt = basal_gpu::cublaslt_version().is_some();
                (basal_gpu::KernelSet::LlamaThinkBf16V1, basal_gpu::Precision::Bf16, Readout::Bf16Rounded, None, lt)
            } else {
                (think_profile.kernel_set, precision, Readout::F32, gemm.table.clone(), gemm.tables_apply)
            };
            let think_plan = basal_gpu::ResolvedExecutionPlan::freeze(
                think_profile.id,
                kernel_set,
                &think_profile.dispatch,
                &t.manifest.config,
                &g.devices(),
                basal_gpu::PlanOptions {
                    mode: inference_mode,
                    precision,
                    readout,
                    kernels,
                    quantization,
                    reproducibility: gemm.reproducibility,
                },
                table,
                tables_apply,
            )?;
            if bf16 {
                crate::note!("{}: thinking execution plan {}", manifest.name, plan_summary(&think_plan));
            }
            t.sources["execution_plan"] = think_plan.identity();
            let (backend, _) =
                GpuBackend::load_planned(&t.manifest.dir, &t.manifest.config, &think_plan, placement, false)
                    .context("loading separate think checkpoint for generation and P2")?;
            basal_core::thinking::Thinking::new(t.manifest, backend, t.settings, t.eos, t.calibration_label, t.sources)
        })
        .transpose()?;
    let loaded = match &exits {
        Some(exits) => GpuBackend::load_planned_with_exits(dir, &manifest.config, &plan, placement, g.media, exits),
        None => GpuBackend::load_planned(dir, &manifest.config, &plan, placement, g.media),
    };
    let (mut backend, gemm_classes) = loaded.with_context(|| match &g.layer_split {
        None => format!("loading {}", manifest.name),
        Some(split) => format!("loading {} with layer split {split}", manifest.name),
    })?;
    match &g.layer_split {
        None => crate::done!("{} loaded on GPU {} in {:.1}s", manifest.name, g.device(), backend.load_s()),
        Some(split) => {
            let ranges = split.ranges(manifest.config.num_layers())?;
            let stages: Vec<String> = split
                .devices()
                .iter()
                .zip(&ranges)
                .map(|(d, r)| format!("{d} (layers {}-{})", r.start, r.end - 1))
                .collect();
            crate::done!("{} loaded on GPUs {} in {:.1}s", manifest.name, stages.join(", "), backend.load_s());
        }
    }
    match quantization {
        Quantization::None => {}
        Quantization::Q8_0 => {
            let q = &backend.describe()["quantization"];
            crate::note!(
                "{}: {quantization} projection weights, {} matrices, {:.2} GiB of blocks ({} zero-scale, {} \
                 subnormal-scale blocks)",
                manifest.name,
                q["matrices"],
                q["allocated_bytes_with_padding"].as_f64().unwrap_or(f64::NAN) / 1073741824.0,
                q["zero_scale_blocks"],
                q["subnormal_scale_blocks"]
            );
        }
        Quantization::Q4K => {
            let q = &backend.describe()["quantization"];
            let model = &q["reconstruction_error"]["model"];
            let groups = &model["groups"];
            crate::note!(
                "{}: {quantization} projection weights, {} matrices, {:.2} GiB of blocks; reconstruction relative RMS \
                 error {} over all projections, {} flattened and {} zero groups of 32 of {}; warm-up check: layer {} \
                 {} dequantized on the GPU bitwise equal to the host",
                manifest.name,
                q["matrices"],
                q["allocated_bytes_with_padding"].as_f64().unwrap_or(f64::NAN) / 1073741824.0,
                model["relative_error_ratio"]["sqrt"],
                groups["flattened"]["count"],
                groups["zero"],
                groups["total"],
                q["warmup_check"]["layer"],
                q["warmup_check"]["projection"]
            );
        }
    }
    if let (Some(n), Some(table)) = (gemm_classes, &plan.gemm().table) {
        crate::note!("{n} GEMM classes from {}", gemm_artifact::describe(table));
    }
    backend.set_state_cache_bytes(g.state_cache_mb.checked_mul(1 << 20).context("state cache budget overflow")?)?;
    let mut engine = Engine::new(manifest, backend)?;
    if let Some(t) = thinking {
        engine.set_thinking(t);
    }
    engine.set_auto_orders(g.model_options.auto_orders()?);
    engine.set_context_overflow(g.model_options.overflow()?);
    engine.mark_state = g.state_cache_mb > 0;
    engine.tree_max_tokens = g.tree_max_tokens;
    if !g.no_prefix_cache && engine.backend.supports_prefix_cache() {
        let lens = engine.warm_static_prefixes()?;
        crate::note!("template prefixes precomputed ({lens:?} tokens)");
    }
    Ok(PlannedEngine { engine, plan, profile })
}

/// Profile, digest of the plan identity and its main decisions, for the log.
fn plan_summary(plan: &basal_gpu::ResolvedExecutionPlan) -> String {
    use sha2::Digest;
    let identity = plan.identity();
    let hash = sha2::Sha256::digest(identity.to_string().as_bytes());
    let hex: String = hash.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let gemm = plan.gemm().table.as_ref().map_or_else(|| identity["gemm"]["fallback"].to_string(), |t| t.label());
    format!(
        "{} sha256:{hex} ({}; {}; GEMM {gemm}; {})",
        plan.profile(),
        identity["kernel_set"].as_str().unwrap_or("?"),
        identity["policy"],
        identity["reproducibility"].as_str().unwrap_or("?")
    )
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
    let cli = Cli::parse();
    term::set_mode(cli.color);
    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    delegate_gpu_command(&cli.cmd);
    match cli.cmd {
        Cmd::Mcp(args) => mcp::run(args)?,
        Cmd::Batches(args) => batches::run(args)?,
        Cmd::Database(args) => database::cli::run(args)?,
        Cmd::Users(args) => keys::run_users(args)?,
        Cmd::Keys(args) => keys::run_keys(args)?,
        Cmd::Client(args) => {
            let code = match client::run(*args) {
                Ok(code) => code,
                Err(e) => {
                    term::error(&e);
                    2
                }
            };
            if code != 0 {
                std::process::exit(i32::from(code));
            }
        }
        Cmd::Benchmark { m, g, options } => benchmark::run(&m, &g, &options)?,
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
        Cmd::Prepare { m, request, upstream, orders, model_options } => {
            let engine = planning_engine(&m, orders.as_deref(), &model_options)?;
            let body = read_request(&request)?;
            let prepared = if upstream {
                engine.plan_request_as(&body, basal_core::request::Dialect::Upstream).map(|p| p.prepared)
            } else {
                engine.prepare_request(&body)
            };
            match prepared {
                Ok(p) => println!("{}", serde_json::to_string_pretty(&p)?),
                Err(DecideError::Internal(e)) => return Err(e),
                Err(e) => {
                    println!("{}", serde_json::to_string_pretty(&e.to_json())?);
                    std::process::exit(2);
                }
            }
        }
        Cmd::CheckPrompts { m, reference, out, orders, model_options } => {
            let engine = planning_engine(&m, orders.as_deref(), &model_options)?;
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
        Cmd::Export { m, g, inputs, out, batching, upstream } => {
            let mut engine = gpu_engine(&m, &g)?;
            let load_s = engine.backend.load_s();
            let mut run = json!({"dtype": g.dtype, "inference_mode": g.inference_mode, "quantization": g.quantization(&m.model)?.as_str(), "readout": g.readout, "kernels": g.kernels, "prefix_cache": !g.no_prefix_cache, "load_s": load_s});
            // only with a split: its stages as given (one GPU keeps the run object unchanged)
            if let Some(split) = &g.layer_split {
                run["layer_split"] = serde_json::to_value(split.stages())?;
            }
            export::export(
                &mut engine,
                &inputs,
                &out,
                parse_batching(&batching)?,
                if upstream { basal_core::request::Dialect::Upstream } else { basal_core::request::Dialect::TypeSafe },
                run,
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
                json!({"hits": engine.backend.state_cache_stats().0, "inserts": engine.backend.state_cache_stats().1});
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
            let transfers = engine.backend.stage_transfers(); // layer split: the profiled items only
            if let Some(t) = &transfers {
                t.reset();
            }
            engine.backend.enable_profile(residual_max)?;
            // items after the warm-up one (the reference may hold fewer than n + 1)
            let n = prep.len() - 1;
            let t = std::time::Instant::now();
            for c in prep[1..].chunks(batch) {
                engine.run(c, basal_core::Batching::Budget)?;
            }
            let total = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            let prof = engine.backend.take_profile();
            let rmax = engine.backend.take_residual_max();
            let nl = engine.manifest.config.num_layers();
            let per_layer: Vec<f32> =
                (0..nl).map(|l| rmax.iter().skip(l).step_by(nl).copied().fold(0.0f32, f32::max)).collect();
            let sum: f64 = prof.values().sum::<f64>() / n as f64;
            let tokens: f64 = prep[1..].iter().map(|p| p.packed.ids.len() as f64).sum::<f64>() / n as f64;
            let mut out = json!({"items": n, "mean_packed_tokens_with_template": tokens, "ms_per_item_profiled": total, "sections_ms_per_item": prof.iter().map(|(k, v)| (k.to_string(), json!(v / n as f64))).collect::<serde_json::Map<_, _>>(), "sections_sum_ms": sum, "residual_abs_max_per_layer": per_layer, "residual_abs_max": per_layer.iter().copied().fold(0.0f32, f32::max)});
            // quantized weights: the dequantizations are a part of the projection sections, so not added to
            // sections_sum_ms (q8_0 keeps its keys)
            match engine.backend.dequant_profile() {
                Some((Quantization::Q8_0, ms, values)) => {
                    out["q8_dequant_ms_per_item"] = json!(ms / n as f64);
                    out["q8_dequant_values_per_item"] = json!(values as f64 / n as f64);
                    out["q8_dequant_scope"] = json!("synchronised time of the q8_0 -> f16 dequantizations; included in the qkv_proj, o_proj, gate_up_proj and down_proj sections, not added to sections_sum_ms");
                }
                Some((format, ms, values)) => {
                    out["dequant_ms_per_item"] = json!(ms / n as f64);
                    out["dequant_values_per_item"] = json!(values as f64 / n as f64);
                    out["dequant_format"] = json!(format.as_str());
                    out["dequant_scope"] = json!(format!("synchronised time of the {format} -> f16 dequantizations; included in the qkv_proj, o_proj, gate_up_proj and down_proj sections, not added to sections_sum_ms"));
                }
                None => {}
            }
            if let Some(t) = &transfers {
                out["stage_transfers"] = t.report();
            }
            println!("{out}");
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
        Cmd::GemmDigest { files, import, dir } => {
            let cache = import.then(|| dir.unwrap_or_else(config::default_gemm_cache));
            gemm_artifact::digest_command(&files, cache.as_deref())?;
        }
        Cmd::Uninstall { models, yes, dry_run } => uninstall::run(&uninstall::Options { models, yes, dry_run })?,
        Cmd::Init { models, out, force } => {
            let models = if models.is_empty() { vec![models::default_model().short.to_string()] } else { models };
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
            mcp,
            config,
            models,
            g,
            replica_devices,
            addr,
            max_batch_tokens,
            max_inflight,
            max_large_inflight,
            schedule,
            long_tokens,
            long_slice_ms,
            access_log,
            metrics,
            batches,
            batch_data_dir,
            auth,
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
            c.systemone_dialect = cli.systemone_dialect.unwrap_or(c.systemone_dialect);
            if cli.log_decisions.is_some() {
                c.log_decisions = cli.log_decisions;
            }
            for model in &mut c.models {
                model.model_options.apply_cli(&g.model_options);
                model.model_options.validate()?;
                model.orders = model.orders.clone().or_else(|| g.orders.clone());
            }
            match &source {
                config::Source::File(p) => crate::note!("configuration {}", crate::term::P(p)),
                config::Source::Models => {}
                config::Source::Builtin => crate::note!(
                    "no `--model` and no {} here: serving {} (`--model 2.0-mini|2.0-max` selects another 2.0 model; \
                     `--model mini|4.5B|max` selects basal-1.5)",
                    paths::CONFIG_NAME,
                    models::default_model().repo
                ),
            }
            if matches!(source, config::Source::File(_)) {
                // the file's models have their own devices; a device of the command line would be ignored
                anyhow::ensure!(
                    g.device.is_none(),
                    "--device applies to --model; with a configuration file set `device` per model in it"
                );
                anyhow::ensure!(
                    replica_devices.is_none(),
                    "--replica-devices applies to --model; with a configuration file set `replica_devices` per model \
                     in it"
                );
                anyhow::ensure!(
                    g.quantization.is_none(),
                    "--quantization applies to --model; with a configuration file set `quantization` per model in it"
                );
                anyhow::ensure!(
                    g.layer_split.is_none(),
                    "--layer-split applies to --model; with a configuration file set `layer_split` per model in it"
                );
            } else {
                // models from the command line: the options of the command line
                for m in &mut c.models {
                    if g.dtype != "auto" {
                        m.dtype = g.dtype.clone();
                    }
                    m.inference_mode = g.inference_mode.clone();
                    m.quantization = g.quantization.clone().unwrap_or_else(|| Quantization::None.as_str().into());
                    m.device = g.device;
                    m.replica_devices.clone_from(&replica_devices);
                    m.layer_split.clone_from(&g.layer_split);
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
                c.max_large_inflight = max_large_inflight;
                c.long_slice_ms = long_slice_ms;
                c.release_date = release_date;
            }
            serve::check_admission_limits(c.max_inflight, c.max_large_inflight)?;
            // Validate database and Batch settings before downloading weights or loading a GPU.
            let addr = match (addr, std::env::var("BASAL_ADDR")) {
                (Some(a), _) => a,
                (None, Ok(v)) => v.parse().with_context(|| format!("BASAL_ADDR {v:?}"))?,
                (None, Err(_)) => c.addr,
            };
            let database_section = database::DatabaseConfig::section(c.database.as_ref())?;
            c.batches.apply_cli(batches, batch_data_dir);
            let batch_settings =
                c.batches.validate(&serve::batches::config::ValidateContext { default_data_dir: paths::data_dir() })?;
            // The consumers (Batch API, API keys) use the root database, or without one the local server database. An
            // enabled root database without an enabled consumer is only migrated at start (with auto_migrate), never
            // otherwise connected. API keys never fall back to serving without a database.
            c.auth.enabled |= auth;
            let database = match (&batch_settings, database_section) {
                (Some(batches), section) => Some(batches.database(&section)?),
                (None, database::DatabaseSection::Enabled(settings)) => Some(settings),
                (None, database::DatabaseSection::Absent) if c.auth.enabled => {
                    Some(database::DatabaseSettings::local_default(&paths::data_dir(), &c.batches.local_data_dir()))
                }
                (None, database::DatabaseSection::Disabled) if c.auth.enabled => {
                    bail!("API keys (auth.enabled or --auth) need a database, but `database.enabled` is false")
                }
                (None, database::DatabaseSection::Absent | database::DatabaseSection::Disabled) => None,
            };
            let batch_qualifications = c
                .models
                .iter()
                .map(|model| model.batch_profile.validate(&config::describe(model)))
                .collect::<Result<Vec<_>, _>>()?;
            // devices of every model's instances, its weight format and layer split; the GPUs of a replica group or
            // a layer split are compared before any download
            let placements: Vec<config::Placement> = c
                .models
                .iter()
                .map(|m| {
                    let model = config::describe(m);
                    config::check_quantization(&model, &m.quantization, &m.dtype, &g.kernels, cfg!(feature = "cuda"))?;
                    let placement = m.placement().with_context(|| format!("model {model}"))?;
                    if let Some(split) = placement.split() {
                        config::check_layer_split(&model, split, &m.dtype, &g.kernels, cfg!(feature = "cuda"))?;
                    }
                    Ok(placement)
                })
                .collect::<Result<_>>()?;
            for (m, placement) in c.models.iter().zip(&placements) {
                if matches!(placement, config::Placement::Replicas(d) if d.len() > 1) {
                    anyhow::ensure!(
                        cfg!(feature = "cuda"),
                        "model {}: several `replica_devices` need the CUDA build",
                        config::describe(m)
                    );
                }
                config::check_same_gpus(&format!("model {}", config::describe(m)), placement)?;
            }
            // model files (local or downloaded), then GEMM tables (a search needs no model weights in GPU memory),
            // then the models
            let dirs: Vec<PathBuf> = c.models.iter().map(config::model_dir).collect::<Result<_>>()?;
            let manifests: Vec<ModelManifest> = dirs
                .iter()
                .zip(&c.models)
                .map(|(d, m)| m.model_options.load_manifest(d, m.orders.as_deref(), &m.source_name()))
                .collect::<Result<_>>()?;
            // Validate every second checkpoint and calibration before any model's GEMM search or GPU load.
            let thinking: Vec<_> = c
                .models
                .iter()
                .zip(&manifests)
                .zip(&placements)
                .map(|((mc, manifest), placement)| {
                    mc.model_options.check_think_execution(
                        Kernels::parse(&g.kernels)?,
                        mc.quantization.parse()?,
                        parse_inference_mode(&mc.inference_mode)?,
                        placement.split().is_some(),
                        gemm_artifact::TableRequest::parse(&mc.gemm_table)?.reproducibility(),
                    )?;
                    mc.model_options.load_thinking(manifest)
                })
                .collect::<Result<_>>()?;
            // one engine per model name (the `model` field of requests) before any GEMM search or load
            for (i, m) in manifests.iter().enumerate() {
                anyhow::ensure!(
                    manifests[..i].iter().all(|o| o.name != m.name),
                    "model {} configured twice (one entry per model; `replica_devices` serves it on several GPUs)",
                    m.name
                );
            }
            // Validate architecture-specific placement and exact layer coverage before any GEMM table or load.
            for ((mc, manifest), placement) in c.models.iter().zip(&manifests).zip(&placements) {
                let quantization: Quantization =
                    mc.quantization.parse().with_context(|| format!("model {}", manifest.name))?;
                config::check_architecture(manifest, quantization, placement.split().is_some())?;
                if mc.model_options.exits_enabled() {
                    let readout = g.readout()?;
                    GpuBackend::check_early_exit(
                        Kernels::parse(&g.kernels)?,
                        quantization,
                        readout,
                        placement.split().is_some(),
                        mc.media,
                        &manifest.config,
                    )?;
                    if let Some(artifacts) = mc.model_options.load_exits(manifest)? {
                        GpuBackend::early_exit_resident_bytes(&artifacts, &manifest.config)?;
                    }
                }
                if let Some(split) = placement.split() {
                    config::check_split_coverage(&manifest.name, split, manifest.config.num_layers())?;
                }
            }
            // one table per model, chosen on its first device (first phase of the execution plan) and checked on
            // every device of the model before anything loads
            let mut tables = Vec::new();
            for ((mc, manifest), placement) in c.models.iter().zip(&manifests).zip(&placements) {
                let device = placement.first_device();
                let precision = config::model_precision(&mc.dtype, manifest, device)?;
                let profile = models::profile_for(manifest);
                let table = gemm_artifact::resolve(
                    &gemm_artifact::TableRequest::parse(&mc.gemm_table)?,
                    manifest,
                    profile,
                    precision,
                    &c.gemm_cache,
                    &placement.instances().concat(),
                )?;
                config::check_table_on_devices(manifest, table.table.as_ref(), placement)?;
                tables.push(table);
            }
            let mut served = Vec::new();
            for (((((mc, table), manifest), thinking), placement), batch_qualification) in
                c.models.iter().zip(tables).zip(manifests).zip(thinking).zip(placements).zip(batch_qualifications)
            {
                let instances = placement.instances();
                let mut replicas = Vec::with_capacity(instances.len());
                for devices in instances {
                    let mg = GpuArgs {
                        model_options: mc.model_options.clone(),
                        media: mc.media || g.media,
                        orders: mc.orders.clone(),
                        dtype: mc.dtype.clone(),
                        inference_mode: mc.inference_mode.clone(),
                        readout: g.readout.clone(),
                        kernels: g.kernels.clone(),
                        quantization: Some(mc.quantization.clone()),
                        no_prefix_cache: mc.no_prefix_cache,
                        state_cache_mb: mc.state_cache_mb,
                        tree_max_tokens: mc.tree_max_tokens,
                        gemm_table: None,
                        resolved_gemm: Some(table.clone()),
                        device: placement.split().is_none().then(|| devices[0]),
                        layer_split: placement.split().cloned(),
                    };
                    let planned = planned_engine_from_manifest(manifest.clone(), thinking.clone(), &mg)?;
                    let scheduling = planned.scheduling();
                    let mut engine = planned.engine;
                    engine.forward_max_tokens = mc.max_batch_tokens;
                    if let Some(t) = &table.table {
                        config::check_gemm_coverage(&engine.manifest.name, &engine.backend.gemm_table_missing(), t)?;
                    }
                    replicas.push(serve::Replica { engine, devices, scheduling });
                }
                let batch_profile = if batch_settings.is_some() {
                    Some(
                        serve::batches::ModelProfile::from_engine(
                            &replicas[0].engine,
                            table.table.as_ref().map(basal_gpu::GemmArtifact::bytes_sha256),
                        )?
                        .with_background(batch_qualification.clone()),
                    )
                } else {
                    None
                };
                served.push(serve::ServedModel {
                    replicas,
                    max_batch_tokens: mc.max_batch_tokens,
                    schedule: mc.schedule,
                    long_tokens: mc.long_tokens,
                    batch_profile,
                    batch_qualification,
                });
            }
            let opts = serve::ServeOptions {
                mcp: mcp.unwrap_or(c.mcp),
                systemone_dialect: c.systemone_dialect,
                log_decisions: c.log_decisions,
                addr,
                release_date: c.release_date,
                max_inflight: c.max_inflight,
                max_large_inflight: c.max_large_inflight,
                default_model: c.default_model,
                long_slice_ms: c.long_slice_ms,
                access_log: access_log || c.access_log,
                metrics: metrics || c.metrics,
                database: database.map(database::Database::new),
                batches: batch_settings,
                auth: c.auth.enabled.then_some(c.auth),
            };
            serve::serve(served, opts)?;
        }
        Cmd::GemmSearch { m, dtype, out, max_m, m_classes, invariant, device } => {
            #[cfg(feature = "cuda")]
            {
                let classes: Vec<usize> = if !m_classes.is_empty() {
                    m_classes
                } else if invariant {
                    config::INVARIANT_CLASSES.to_vec()
                } else {
                    (16..=512).step_by(16).chain((640..=max_m).step_by(128)).collect()
                };
                let shapes = basal_gpu::architecture_projection_shapes(&ModelManifest::load(&m.dir()?)?.config)?;
                let t = basal_gpu::gemm_search(&classes, &shapes, &dtype, invariant, device)?;
                write_json(&out, &t)?;
            }
            #[cfg(not(feature = "cuda"))]
            bail!(
                "gemm-search needs a CUDA build ({}, {dtype}, {}, {max_m}, {m_classes:?}, {invariant}, {device})",
                m.model,
                out.display()
            );
        }
        Cmd::Gemm { dtype, m, reps, device } => {
            let shapes = [(2048, 2560), (2048, 2048), (2048, 11008), (11008, 2048)];
            for r in basal_gpu::gemm_microbench(&m, &shapes, &dtype, reps, device)? {
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
            let load_s = engine.backend.load_s();
            let batch = match batching.as_str() {
                "budget" => basal_core::Batching::Budget,
                "tree" => basal_core::Batching::Tree,
                b => bail!("unknown batching {b:?}"),
            };
            // GPU condition probe before and after: the same GEMM (f16, 480x2048 @ 11008x2048^T). A drop between the
            // two marks throttling or outside load during the run. A layer split probes every stage's GPU, one after
            // another (each probe owns its device and operands and returns host values only).
            let probe = |device: usize| -> Result<f64> {
                let r = basal_gpu::gemm_microbench(&[480], &[(2048, 11008)], "f16", 30, device)?;
                Ok(r.iter().filter_map(|x| x["tflops"].as_f64()).fold(0.0, f64::max))
            };
            let devices = g.devices();
            let before = devices.iter().map(|&d| probe(d)).collect::<Result<Vec<_>>>()?;
            let mut r = bench::bench(&mut engine, &reference, lat_n, batch)?;
            let after = devices.iter().map(|&d| probe(d)).collect::<Result<Vec<_>>>()?;
            r["gpu_probe_tflops"] = json!({"before": before[0], "after": after[0]});
            if g.layer_split.is_some() {
                r["gpu_probe_tflops_per_device"] = json!(devices
                    .iter()
                    .zip(before.iter().zip(&after))
                    .map(|(d, (b, a))| json!({"device": d, "before": b, "after": a}))
                    .collect::<Vec<_>>());
            }
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
