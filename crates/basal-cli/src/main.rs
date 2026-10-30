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
mod deterministic;
mod doctor;
mod export;
mod gemm_artifact;
mod gemm_share;
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
mod think_source;
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
    // `--backend cpu` runs in this binary
    let g = match cmd {
        Cmd::Benchmark { g, .. }
        | Cmd::Decide { g, .. }
        | Cmd::Export { g, .. }
        | Cmd::BenchRequests { g, .. }
        | Cmd::EvalChoiceSet { g, .. }
        | Cmd::EvalLargeChoice { g, .. }
        | Cmd::Profile { g, .. }
        | Cmd::Serve { g, .. }
        | Cmd::Bench { g, .. } => Some(g),
        _ => None,
    };
    if g.is_some_and(|g| g.backend == Some(basal_gpu::ExecutionBackend::Cpu)) {
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
    /// Decoder kernels: fused (default; candle with --backend cpu) or candle (stock candle ops, first port; for
    /// comparison)
    #[arg(long)]
    kernels: Option<String>,
    /// gpu (default: this build's GPU backend) or cpu: a dense BF16 Llama model on the host CPU, computed in f32 with
    /// candle kernels, text only, one model. With a serve configuration file set `backend` per model instead
    #[arg(long)]
    backend: Option<basal_gpu::ExecutionBackend>,
    /// Required with --backend cpu: MiB of host working set one CPU forward may use (larger inputs are refused, longer
    /// batches run in sub-batches). With a serve configuration file set `cpu_work_mb` per model instead
    #[arg(long)]
    cpu_work_mb: Option<usize>,
    /// Last-layer row pruning: auto (current behavior), on or off (fused CUDA/Metal Llama).
    /// With a serve configuration file, set `readout_layer` per model instead.
    #[arg(long)]
    readout_layer: Option<basal_gpu::ReadoutLayer>,
    /// Projection GEMMs: auto (current behavior), on (combined q/k/v and gate/up) or off (separate q/k/v and gate/up);
    /// on/off need dense fused CUDA/Metal Llama. With a serve configuration file, set `merge_projections` per model
    /// instead.
    #[arg(long)]
    merge_projections: Option<basal_gpu::MergeProjections>,
    /// Projection weights: none (default; dense weights of --dtype), q8_0 (8-bit GGML Q8_0 blocks) or q4_k (4-bit GGML
    /// Q4_K blocks), dequantized to f16 per projection: less GPU memory, more work per forward, different numerics;
    /// CUDA with --dtype f16 (auto for Llama) and --kernels fused; Llama models only (Omni is refused). `basal serve`:
    /// the format of every --model (a configuration file sets `quantization` per model instead)
    #[arg(long)]
    quantization: Option<String>,
    /// Gemma per-layer token table: device (default) or host (single CUDA non-unified PLE models).
    /// With a serve configuration file, set `ple_placement` per model instead.
    #[arg(long)]
    ple_placement: Option<basal_gpu::PleMode>,
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
    /// Refuse host-only option/source incompatibilities before callers can discover any GPU.
    fn check_ple_source(&self, source: &str) -> Result<()> {
        if self.ple_placement != Some(basal_gpu::PleMode::Host) {
            return Ok(());
        }
        let mut model = config::model_ref(source)?;
        model.quantization = self.quantization.clone().unwrap_or_else(|| "none".into());
        model.inference_mode.clone_from(&self.inference_mode);
        model.dtype.clone_from(&self.dtype);
        model.state_cache_mb = self.state_cache_mb;
        model.model_options = self.model_options.clone();
        config::check_ple_source(basal_gpu::PleMode::Host, self.layer_split.is_some(), &model)
    }

    /// Whether `--merge-projections` is on or off (absent and auto keep every default check and its order).
    fn explicit_merge(&self) -> bool {
        self.merge_projections.is_some_and(|m| m != basal_gpu::MergeProjections::Auto)
    }

    /// `--kernels`, else candle with `--backend cpu`, else fused.
    fn kernels(&self) -> &str {
        match (&self.kernels, self.backend) {
            (Some(kernels), _) => kernels,
            (None, Some(basal_gpu::ExecutionBackend::Cpu)) => "candle",
            (None, _) => "fused",
        }
    }

    /// `source` configured with the options of the command line, as a serve configuration would set them.
    fn cpu_model(&self, source: &str) -> Result<config::ModelConfig> {
        let mut model = config::model_ref(source)?;
        model.backend = self.backend.unwrap_or_default();
        model.cpu_work_mb = self.cpu_work_mb;
        model.dtype.clone_from(&self.dtype);
        model.inference_mode.clone_from(&self.inference_mode);
        model.quantization = self.quantization.clone().unwrap_or_else(|| Quantization::None.as_str().into());
        model.ple_placement = self.ple_placement.unwrap_or_default();
        model.readout_layer = self.readout_layer.unwrap_or_default();
        model.merge_projections = self.merge_projections.unwrap_or_default();
        model.state_cache_mb = self.state_cache_mb;
        model.media = self.media;
        model.model_options = self.model_options.clone();
        model.device = self.device;
        model.layer_split.clone_from(&self.layer_split);
        model.gemm_table = self.gemm_table.as_ref().map_or_else(|| "none".into(), |t| t.display().to_string());
        Ok(model)
    }

    /// The static CPU refusals of `source` with these options, before any download, GPU or GEMM table: the CPU
    /// per-forward ceiling with `--backend cpu`, else None (refusing `--cpu-work-mb`).
    fn check_cpu(&self, source: &str) -> Result<Option<std::num::NonZeroU64>> {
        let model = self.cpu_model(source)?;
        let work = config::cpu_work(&model, self.kernels.as_deref(), &self.readout)?;
        if work.is_some() {
            model.placement()?;
        }
        Ok(work)
    }

    /// The GPU ordinal (`--device`, default 0).
    fn device(&self) -> usize {
        self.device.unwrap_or(0)
    }

    /// The devices the model uses: the --layer-split stages, else `[--device]`.
    fn devices(&self) -> Vec<usize> {
        self.layer_split.as_ref().map_or_else(|| vec![self.device()], LayerSplit::devices)
    }

    /// `--layer-split` of `model` against this build's backend, --dtype and --kernels, then its GPUs against the
    /// driver (every ordinal visible, one GPU type), before any download, GEMM table or load. With a custom thinking
    /// source the GPUs are compared after it is validated and admitted ([`admit_custom_thinking`]); with explicit
    /// merge-projections after the model's and its thinking checkpoint's configurations are validated.
    fn check_layer_split(&self, model: &str) -> Result<()> {
        let Some(split) = &self.layer_split else { return Ok(()) };
        config::check_layer_split(model, split, &self.dtype, self.kernels(), cfg!(feature = "cuda"))?;
        if self.model_options.custom_thinking() || self.explicit_merge() {
            return Ok(());
        }
        config::check_same_gpus(model, &config::Placement::Split(split.clone()))
    }

    /// `--quantization` (default none) of `model`, checked with --dtype and --kernels for this build's backend.
    fn quantization(&self, model: &str) -> Result<Quantization> {
        config::check_quantization(
            model,
            self.quantization.as_deref().unwrap_or(Quantization::None.as_str()),
            &self.dtype,
            self.kernels(),
            cfg!(feature = "cuda"),
        )
    }

    /// The options of one instance of `model` as the strict profile checks them ([`deterministic::profile`]).
    fn deterministic_request<'a>(&'a self, model: &'a str) -> deterministic::Requested<'a> {
        deterministic::Requested {
            model,
            options: &self.model_options,
            backend: self.backend.unwrap_or_default(),
            dtype: &self.dtype,
            kernels: self.kernels(),
            readout: &self.readout,
            readout_layer: self.readout_layer.unwrap_or_default(),
            quantization: self.quantization.as_deref().unwrap_or(Quantization::None.as_str()),
            ple: self.ple_placement.unwrap_or_default(),
            inference_mode: &self.inference_mode,
            instances: 1,
            split: self.layer_split.is_some(),
            media: self.media,
            state_cache_mb: self.state_cache_mb,
            gemm_table: self.gemm_table.as_ref().map_or("auto".into(), |t| t.to_string_lossy()),
            background: false,
        }
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
        /// Bench-item forwards: single (default), budget or tree; --soam controls served request answers separately
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
        /// chunks) or single (one forward per question); default tree unless --soam is explicit
        #[arg(long)]
        batching: Option<String>,
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
        /// legacy admission options below. Explicit --max-batch/--wait-ms/--batch-window-ms override its per-model
        /// collection settings. Default: BASAL_CONFIG, else basal-serve.yml in the working directory
        #[arg(long, conflicts_with_all = ["models", "name", "gemm_table", "device", "replica_devices", "layer_split"])]
        config: Option<PathBuf>,
        /// Model to serve: pico, 2.0-mini/2.0-4.5B/2.0-max, omni-mini/omni-max, name (owner Remek), owner/name,
        /// @revision, or a local directory; mini/4.5B/max select basal-1.5. Repeat
        /// for several (the options below apply to each). Without --model and a configuration file: basal-2.0-4.5B
        #[arg(long = "model")]
        models: Vec<String>,
        /// Deployment name for one model; with a configuration file set `name` per model instead
        #[arg(long)]
        name: Option<String>,
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
        /// Maximum whole requests per online batch in both lanes; absent: no additional entry ceiling
        #[arg(long)]
        max_batch: Option<usize>,
        /// Fixed main-lane collection wait in finite nonnegative milliseconds (overrides YAML)
        #[arg(long)]
        wait_ms: Option<f64>,
        /// Adaptive main-lane collection window in milliseconds; explicit 0 disables it (overrides YAML)
        #[arg(long)]
        batch_window_ms: Option<f64>,
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
        /// Explicit short-first backlog: auto enables only without a long lane; on requires it off (overrides YAML)
        #[arg(long, value_enum)]
        scheduler: Option<serve::backlog::Scheduler>,
        /// Strict long-lane token threshold in the selected basis (0 disables the legacy lane)
        #[arg(long, default_value_t = 4096)]
        long_tokens: usize,
        /// Long lane: legacy follows the threshold, on requires a positive threshold, off disables (overrides YAML)
        #[arg(long, value_enum)]
        long_lane: Option<serve::routing::LongLane>,
        /// Long routing by packed admission cost or safe longest-prompt bound (overrides YAML)
        #[arg(long, value_enum)]
        long_token_basis: Option<serve::routing::LongTokenBasis>,
        /// Least time the long lane works between two hand-overs to the main lane
        #[arg(long, default_value_t = 100)]
        long_slice_ms: u64,
        /// Root long-lane share of occupied online wall time, finite `[0,1]` (overrides YAML); absent: YAML or legacy
        #[arg(long)]
        long_share: Option<f64>,
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
        /// Shapes of the projection GEMMs: auto and on (combined q/k/v and gate/up, the default table) or off
        /// (separate q/k/v and gate/up, for `--merge-projections off`; dense fused Llama)
        #[arg(long)]
        merge_projections: Option<basal_gpu::MergeProjections>,
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
    // the strict profile's refusals precede every other check, the download and any device
    deterministic::profile(&g.deterministic_request(&m.model))?;
    // an unsupported CPU option, weight format or layer split fails before the model is downloaded
    g.check_cpu(&m.model)?;
    g.quantization(&m.model)?;
    g.check_ple_source(&m.model)?;
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
        !options.deterministic_enabled(),
        "this prepare-only command has no backend; the deterministic profile {} applies to loaded engines",
        basal_gpu::DeterministicProfile::ID
    );
    anyhow::ensure!(
        !options.exits_enabled(),
        "this prepare-only command has no trained-head backend; use the loaded engine for early_exit planning"
    );
    let mut model = config::model_ref(&m.model)?;
    model.model_options = options.clone();
    let manifest = options.load_manifest(&config::model_dir(&model)?, orders, &model.source_name())?;
    let thinking = options.load_thinking(&manifest)?;
    let mut engine = Engine::new(manifest, NoBackend)?;
    engine.request_batching = options.request_batching();
    if let Some(t) = thinking {
        engine.set_thinking(basal_core::thinking::Thinking::new(
            t.manifest,
            NoBackend,
            t.settings,
            t.eos,
            t.calibration_label,
            t.sources,
        )?);
        // a custom source's files still have the stamps of their verification
        if let Some(custom) = &t.custom {
            custom.verified.fence()?;
        }
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
    check_custom_thinking_options(g)?;
    check_merge_thinking(g.merge_projections.unwrap_or_default(), &g.model_options)?;
    if g.model_options.think_numerics()? == model_options::ThinkNumerics::Inherit {
        return Ok(());
    }
    let reproducibility = base_reproducibility(g)?;
    g.model_options.check_think_execution(
        Kernels::parse(g.kernels())?,
        g.quantization.as_deref().unwrap_or("none").parse()?,
        parse_inference_mode(&g.inference_mode)?,
        g.layer_split.is_some(),
        reproducibility,
    )
}

/// Explicit merge-projections with upstream-bf16-pointwise thinking is refused before any checkpoint read; nothing
/// for auto.
fn check_merge_thinking(merge: basal_gpu::MergeProjections, options: &model_options::ModelOptions) -> Result<()> {
    if merge == basal_gpu::MergeProjections::Auto {
        return Ok(());
    }
    anyhow::ensure!(
        options.think_numerics()? != model_options::ThinkNumerics::UpstreamBf16Pointwise,
        "merge-projections on/off is not implemented with upstream-bf16-pointwise thinking"
    );
    Ok(())
}

/// Whether merge-projections `merge` runs the Q/K/V and gate/up projections of configuration `arch` (of `model`: a
/// base or thinking checkpoint) as separate GEMMs with `kernels` and `quantization`, validated without any device
/// query. Auto is false before any parsing, so the default paths keep their checks and their order.
fn separate_projections(
    merge: basal_gpu::MergeProjections,
    arch: &basal_core::architecture::Architecture,
    kernels: &str,
    quantization: Quantization,
    model: &str,
) -> Result<bool> {
    if merge == basal_gpu::MergeProjections::Auto {
        return Ok(false);
    }
    merge.resolve(arch, Kernels::parse(kernels)?, quantization).with_context(|| format!("model {model}"))
}

/// Whether the base plan of `g` is reproducible: its resolved GEMM table, else its `--gemm-table` request.
fn base_reproducibility(g: &GpuArgs) -> Result<basal_gpu::Reproducibility> {
    Ok(match &g.resolved_gemm {
        Some(gemm) => gemm.reproducibility,
        None => gemm_artifact::TableRequest::parse(
            &g.gemm_table.as_ref().map(|p| p.to_string_lossy()).unwrap_or("none".into()),
        )?
        .reproducibility(),
    })
}

/// The custom thinking refusals of one instance with `g` (CPU backend, quantization, reproducibility, media) before
/// its descriptor is read; nothing without `think_source`.
fn check_custom_thinking_options(g: &GpuArgs) -> Result<()> {
    if !g.model_options.custom_thinking() {
        return Ok(());
    }
    g.model_options.check_custom_think_execution(
        g.quantization.as_deref().unwrap_or("none").parse()?,
        g.media,
        base_reproducibility(g)?,
        false,
        g.backend == Some(basal_gpu::ExecutionBackend::Cpu),
    )
}

/// A custom thinking checkpoint whose layer count `layers` the layer split `split` does not cover.
fn check_custom_split(split: &LayerSplit, layers: usize) -> Result<()> {
    split.ranges(layers).with_context(|| {
        format!("custom thinking checkpoint has {layers} layers; layer_split {split} does not cover them")
    })?;
    Ok(())
}

/// Custom thinking of one instance of `model` on `split` or `device`, after its source was validated: coverage of its
/// own layers by the split, host then device admission (the first device query of the custom path), then the deferred
/// GPU identity check of a split; before any precision rule, GEMM table or load. Nothing for the published pair.
fn admit_custom_thinking(
    t: &model_options::LoadedThinking,
    split: Option<&LayerSplit>,
    device: usize,
    model: &str,
) -> Result<()> {
    let Some(custom) = &t.custom else { return Ok(()) };
    let layers = t.manifest.config.num_layers();
    let stages = match split {
        None => vec![(device, 0..layers)],
        Some(split) => {
            check_custom_split(split, layers)?;
            split.devices().into_iter().zip(split.ranges(layers)?).collect()
        }
    };
    let resources = &custom.verified.resources;
    think_source::admit(resources.host_instance()?, &[(resources, stages)])?;
    if let Some(split) = split {
        config::check_same_gpus(model, &config::Placement::Split(split.clone()))?;
    }
    Ok(())
}

/// The GPU comparison of a layer split that explicit merge-projections deferred ([`GpuArgs::check_layer_split`]), after
/// the model's and its thinking checkpoint's configurations were validated; nothing otherwise, and nothing with a
/// custom thinking source, whose admission compares them ([`admit_custom_thinking`]).
fn check_merge_split(g: &GpuArgs, thinking: Option<&model_options::LoadedThinking>, model: &str) -> Result<()> {
    if !g.explicit_merge() || thinking.is_some_and(|t| t.custom.is_some()) {
        return Ok(());
    }
    let Some(split) = &g.layer_split else { return Ok(()) };
    config::check_same_gpus(model, &config::Placement::Split(split.clone()))
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
    mut thinking: Option<model_options::LoadedThinking>,
    g: &GpuArgs,
) -> Result<PlannedEngine> {
    // before the CPU early return, thinking admission and any device: also for direct callers of this function
    let strict = deterministic::profile(&g.deterministic_request(&manifest.name))?;
    if strict.is_some() {
        deterministic::check_manifest(&manifest)?;
        anyhow::ensure!(thinking.is_none(), "model {}: the deterministic profile has no thinking", manifest.name);
    }
    check_custom_thinking_options(g)?;
    if g.backend == Some(basal_gpu::ExecutionBackend::Cpu) {
        return cpu_engine_from_manifest(manifest, thinking, g);
    }
    check_thinking_gpu_options(g)?;
    let dir = manifest.dir.as_path();
    let inference_mode = parse_inference_mode(&g.inference_mode)?;
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    anyhow::ensure!(g.state_cache_mb == 0, "Intel state cache is not implemented; use --state-cache-mb 0");
    let quantization = g.quantization(&manifest.name)?;
    // Validate architecture-specific quantization and split constraints before any weight is read.
    config::check_architecture(&manifest, quantization, g.layer_split.is_some())?;
    let ple = g.ple_placement.unwrap_or_default();
    ple.resolve(&manifest.config, g.layer_split.is_some())?;
    let readout = g.readout()?;
    let profile = models::profile_for(&manifest);
    let kernels = Kernels::parse(g.kernels())?;
    let readout_layer = g.readout_layer.unwrap_or_default();
    readout_layer.resolve(&manifest.config, kernels)?;
    let merge_projections = g.merge_projections.unwrap_or_default();
    let separate =
        separate_projections(merge_projections, &manifest.config, g.kernels(), quantization, &manifest.name)?;
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
    // A custom thinking source (validated by load_thinking, always Llama, so the readout-layer rule above holds for it
    // too): its split coverage and admission, before the host-PLE device query, the precision rule and any GEMM table.
    // Explicit merge-projections validates the thinking checkpoint's own configuration first.
    let think_separate = match &thinking {
        Some(t) => separate_projections(
            merge_projections,
            &t.manifest.config,
            g.kernels(),
            quantization,
            &format!("{} thinking checkpoint", manifest.name),
        )?,
        None => false,
    };
    if let Some(t) = &thinking {
        admit_custom_thinking(t, g.layer_split.as_ref(), g.device(), &format!("model {}", manifest.name))?;
    }
    check_merge_split(g, thinking.as_ref(), &format!("model {}", manifest.name))?;
    if ple == basal_gpu::PleMode::Host {
        anyhow::ensure!(
            inference_mode != basal_gpu::InferenceMode::Balanced && g.state_cache_mb == 0,
            "host PLE Gemma does not support balanced inference or cross-request state cache"
        );
        if g.dtype != "auto" {
            basal_gpu::Precision::parse(&g.dtype)?;
        }
        let memory = GpuBackend::host_ple_memory(dir, &manifest.config)?;
        config::check_host_ple_headroom(memory.host_load_bytes(1)?)?;
        let device_bytes = memory
            .estimated_device_weight_bytes
            .checked_add(memory.estimated_device_load_scratch_bytes)
            .context("host PLE device load estimate overflow")?;
        anyhow::ensure!(basal_gpu::gpu_memory(g.device())?.available_bytes >= device_bytes,
            "host PLE estimated device weights plus load scratch need {device_bytes} bytes; this source estimate excludes allocator/context/runtime overhead");
        crate::note!(
            "{}: host PLE source estimate {} (not allocator peaks or measured VRAM savings)",
            manifest.name,
            serde_json::to_string(&memory)?
        );
    }
    let precision = config::model_precision(&g.dtype, &manifest, g.devices()[0])?;
    if let Some(split) = &g.layer_split {
        // the stages must cover the model's layers exactly: checked before any weight is read
        config::check_split_coverage(&manifest.name, split, manifest.config.num_layers())?;
    }
    let gemm = match (&g.resolved_gemm, &g.gemm_table) {
        (Some(resolved), _) => {
            anyhow::ensure!(
                resolved.table.is_none() || resolved.separate_projections == separate,
                "model {}: GEMM table was resolved for other projection shapes",
                manifest.name
            );
            resolved.clone()
        }
        // `--gemm-table PATH`, or `pinned` / `sha256:<digest>` / `auto` resolved in the user's GEMM cache
        (None, Some(value)) => gemm_artifact::resolve_for_projections(
            &gemm_artifact::TableRequest::parse(&value.to_string_lossy())?,
            &manifest,
            profile,
            precision,
            &config::default_gemm_cache(),
            &g.devices(),
            separate,
        )?,
        (None, None) => gemm_artifact::ResolvedGemm::none(
            basal_gpu::cublaslt_version().is_some() && precision != basal_gpu::Precision::F32,
        ),
    };
    // A custom thinking checkpoint has its own table for its own shapes (resolved here unless serve did), never the
    // base table; checked on the instance's GPUs.
    if let Some(model_options::LoadedThinking { manifest: think, custom: Some(custom), .. }) = &mut thinking {
        if custom.gemm.is_none() {
            custom.resolve_gemm(think, precision, &config::default_gemm_cache(), &g.devices(), think_separate)?;
        }
        anyhow::ensure!(
            custom.gemm.as_ref().is_none_or(|r| r.table.is_none() || r.separate_projections == think_separate),
            "model {}: the custom thinking GEMM table was resolved for other projection shapes",
            manifest.name
        );
        let placement = match &g.layer_split {
            None => config::Placement::Single(g.device()),
            Some(split) => config::Placement::Split(split.clone()),
        };
        config::check_table_on_devices(think, custom.gemm.as_ref().and_then(|r| r.table.as_ref()), &placement)?;
    }
    let options = basal_gpu::PlanOptions {
        mode: inference_mode,
        precision,
        readout,
        kernels,
        quantization,
        reproducibility: gemm.reproducibility,
        ple,
    };
    let plan = basal_gpu::ResolvedExecutionPlan::freeze_with_layer_controls(
        profile.id,
        profile.kernel_set,
        &profile.dispatch,
        &manifest.config,
        &g.devices(),
        options,
        gemm.table.clone(),
        gemm.tables_apply,
        readout_layer,
        merge_projections,
    )
    .with_context(|| format!("model {} (profile {})", manifest.name, profile.id))?;
    let plan = match strict {
        Some(p) => {
            plan.with_deterministic(p).with_context(|| format!("model {} (profile {})", manifest.name, profile.id))?
        }
        None => plan,
    };
    crate::note!("{}: execution plan {}", manifest.name, plan_summary(&plan));
    let placement = match &g.layer_split {
        None => basal_gpu::PlannedPlacement::Single(g.device()),
        Some(split) => basal_gpu::PlannedPlacement::Split(split),
    };
    let thinking = thinking
        .map(|mut t| {
            let custom = t.custom.take();
            // a custom checkpoint: the generic Llama profile (never a catalog profile by its basal.json name) and its
            // own GEMM table
            let (think_profile, think_gemm) = match &custom {
                None => (models::profile_for(&t.manifest), &gemm),
                Some(c) => (
                    models::generic_llama_profile(),
                    c.gemm.as_ref().context("the custom thinking GEMM table was not resolved")?,
                ),
            };
            let bf16 = t.numerics == model_options::ThinkNumerics::UpstreamBf16Pointwise;
            let (kernel_set, precision, readout, table, tables_apply) = if bf16 {
                let lt = basal_gpu::cublaslt_version().is_some();
                (basal_gpu::KernelSet::LlamaThinkBf16V1, basal_gpu::Precision::Bf16, Readout::Bf16Rounded, None, lt)
            } else {
                (think_profile.kernel_set, precision, Readout::F32, think_gemm.table.clone(), think_gemm.tables_apply)
            };
            let think_plan = basal_gpu::ResolvedExecutionPlan::freeze_with_layer_controls(
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
                    reproducibility: think_gemm.reproducibility,
                    ple: basal_gpu::PleMode::Device,
                },
                table,
                tables_apply,
                readout_layer,
                merge_projections,
            )?;
            if bf16 || custom.is_some() {
                crate::note!("{}: thinking execution plan {}", manifest.name, plan_summary(&think_plan));
            }
            t.sources["execution_plan"] = think_plan.identity();
            if let Some(custom) = custom {
                let (backend, _) = GpuBackend::load_planned_verified(
                    &custom.verified.inventory,
                    &t.manifest.config,
                    &think_plan,
                    placement,
                )
                .context("loading the custom think checkpoint for generation and P2")?;
                if let Some(table) = &think_plan.gemm().table {
                    config::check_gemm_coverage(&t.manifest.name, &backend.gemm_table_missing(), table)?;
                }
                let thinking = basal_core::thinking::Thinking::new(
                    t.manifest,
                    backend,
                    t.settings,
                    t.eos,
                    t.calibration_label,
                    t.sources,
                )?;
                custom.verified.fence()?;
                crate::note!(
                    "{}: thinking: custom source loaded (model_id {})",
                    manifest.name,
                    custom.verified.model_id
                );
                return Ok(thinking);
            }
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
    engine.request_batching = g.model_options.request_batching();
    if g.model_options.soam == Some(model_options::Soam::Off) {
        crate::note!(
            "SOAM off: Budget primitive rows, option orders packed together; queue entries remain whole requests"
        );
    }
    if let Some(t) = thinking {
        engine.set_thinking(t);
    }
    engine.set_auto_orders(g.model_options.auto_orders()?);
    engine.set_context_overflow(g.model_options.overflow()?);
    engine.mark_state = g.state_cache_mb > 0;
    engine.tree_max_tokens = g.tree_max_tokens;
    // before any fork or lane exists
    if let Some(p) = plan.deterministic() {
        engine.set_request_isolation(basal_core::engine::RequestIsolation::Whole);
        crate::note!(
            "{}: deterministic profile {}: one whole request per engine call, cuBLAS atomics not allowed, math mode \
             default, CUBLAS_WORKSPACE_CONFIG={} (not a repeatability guarantee)",
            engine.manifest.name,
            basal_gpu::DeterministicProfile::ID,
            p.workspace_config()
        );
    }
    if !g.no_prefix_cache && engine.backend.supports_prefix_cache() && plan.deterministic().is_none() {
        let lens = engine.warm_static_prefixes()?;
        crate::note!("template prefixes precomputed ({lens:?} tokens)");
    }
    Ok(PlannedEngine { engine, plan, profile })
}

/// Load the model of `manifest` on the host CPU (`--backend cpu`): the static CPU refusals of `g`, Llama only, the
/// host headroom of its source estimate, then a CPU plan frozen without any GPU query (no GPU discovery, memory query,
/// precision rule, GEMM table or Intel loader) and the shared Llama loader on `Device::Cpu`.
fn cpu_engine_from_manifest(
    manifest: ModelManifest,
    thinking: Option<model_options::LoadedThinking>,
    g: &GpuArgs,
) -> Result<PlannedEngine> {
    let work = g.check_cpu(&manifest.dir.to_string_lossy())?.context("the CPU backend needs cpu_work_mb")?;
    anyhow::ensure!(thinking.is_none(), "model {}: thinking is not implemented on the CPU backend", manifest.name);
    config::check_cpu_architecture(&manifest)?;
    let memory = GpuBackend::cpu_host_memory(&manifest.dir, &manifest.config)?;
    config::check_host_headroom(memory.startup_bytes(1, work.get())?, "the CPU backend")?;
    crate::note!(
        "{}: CPU host source estimate {} (named tensor payloads, not allocator peaks, RSS or OOM protection)",
        manifest.name,
        serde_json::to_string(&memory)?
    );
    let profile = models::profile_for(&manifest);
    let plan = basal_gpu::ResolvedExecutionPlan::freeze_cpu(
        profile.id,
        profile.kernel_set,
        &profile.dispatch,
        &manifest.config,
        work,
    )
    .with_context(|| format!("model {} (profile {})", manifest.name, profile.id))?;
    crate::note!("{}: execution plan {}", manifest.name, plan_summary(&plan));
    let (backend, _) =
        GpuBackend::load_planned(&manifest.dir, &manifest.config, &plan, basal_gpu::PlannedPlacement::Cpu, false)
            .with_context(|| format!("loading {} on the host CPU", manifest.name))?;
    crate::done!(
        "{} loaded on the host CPU in {:.1}s ({} GEMM threads, cpu work ceiling {} bytes)",
        manifest.name,
        backend.load_s(),
        plan.cpu_gemm_threads().unwrap_or(0),
        work
    );
    let mut engine = Engine::new(manifest, backend)?;
    engine.request_batching = g.model_options.request_batching();
    if g.model_options.soam == Some(model_options::Soam::Off) {
        crate::note!(
            "SOAM off: Budget primitive rows, option orders packed together; queue entries remain whole requests"
        );
    }
    engine.set_auto_orders(g.model_options.auto_orders()?);
    engine.set_context_overflow(g.model_options.overflow()?);
    engine.mark_state = false;
    engine.tree_max_tokens = g.tree_max_tokens;
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

/// Both bench-requests switches control the same request forwards; refuse conflicting explicit choices.
fn request_batching(options: &model_options::ModelOptions, batching: Option<&str>) -> Result<basal_core::Batching> {
    let explicit = batching.map(parse_batching).transpose()?;
    let soam = options.soam.map(model_options::Soam::batching);
    anyhow::ensure!(
        !matches!((explicit, soam), (Some(a), Some(b)) if a != b),
        "explicit --batching conflicts with --soam (on requires tree; off requires budget)"
    );
    Ok(explicit.unwrap_or_else(|| options.request_batching()))
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
            let batching = parse_batching(&batching)?;
            anyhow::ensure!(
                !g.model_options.deterministic_enabled() || batching == basal_core::Batching::Single,
                "export --deterministic {}: bench items run one per forward; use --batching single",
                basal_gpu::DeterministicProfile::ID
            );
            let mut engine = gpu_engine(&m, &g)?;
            let load_s = engine.backend.load_s();
            let mut run = json!({"dtype": g.dtype, "inference_mode": g.inference_mode, "quantization": g.quantization(&m.model)?.as_str(), "readout": g.readout, "kernels": g.kernels(), "prefix_cache": !g.no_prefix_cache, "load_s": load_s});
            // only with a split: its stages as given (one GPU keeps the run object unchanged)
            if let Some(split) = &g.layer_split {
                run["layer_split"] = serde_json::to_value(split.stages())?;
            }
            export::export(
                &mut engine,
                &inputs,
                &out,
                batching,
                if upstream { basal_core::request::Dialect::Upstream } else { basal_core::request::Dialect::TypeSafe },
                run,
            )?;
        }
        Cmd::BenchRequests { m, g, requests, out, reps, batching } => {
            let batching = request_batching(&g.model_options, batching.as_deref())?;
            let mut engine = gpu_engine(&m, &g)?;
            engine.request_batching = batching;
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
            anyhow::ensure!(
                !g.model_options.deterministic_enabled(),
                "profile batches independent items; the deterministic profile {} is unavailable here",
                basal_gpu::DeterministicProfile::ID
            );
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
            name,
            g,
            replica_devices,
            addr,
            max_batch_tokens,
            max_batch,
            wait_ms,
            batch_window_ms,
            max_inflight,
            max_large_inflight,
            schedule,
            scheduler,
            long_tokens,
            long_lane,
            long_token_basis,
            long_slice_ms,
            long_share,
            access_log,
            metrics,
            batches,
            batch_data_dir,
            release_date,
        } => {
            let access_log = access_log || std::env::var("BASAL_ACCESS_LOG").is_ok_and(|v| !v.is_empty() && v != "0");
            let (source, mut c) = config::select(config, &models)?;
            c.long_share = long_share.or(c.long_share);
            serve::check_long_share(c.long_share)?;
            if let Some(name) = name {
                anyhow::ensure!(
                    !matches!(source, config::Source::File(_)),
                    "--name cannot be used with a configuration file; set `name` per model in it"
                );
                anyhow::ensure!(c.models.len() == 1, "--name requires exactly one model; use YAML `name` per model");
                c.models[0].name = Some(name);
            }
            c.systemone_dialect = cli.systemone_dialect.unwrap_or(c.systemone_dialect);
            if cli.log_decisions.is_some() {
                c.log_decisions = cli.log_decisions;
            }
            for model in &mut c.models {
                model.validate_name()?;
                model.model_options.apply_cli(&g.model_options);
                model.model_options.validate()?;
                model.orders = model.orders.clone().or_else(|| g.orders.clone());
                model.max_batch = max_batch.or(model.max_batch);
                model.wait_ms = wait_ms.or(model.wait_ms);
                model.batch_window_ms = batch_window_ms.or(model.batch_window_ms);
                model.long_lane = long_lane.or(model.long_lane);
                model.long_token_basis = long_token_basis.or(model.long_token_basis);
                model.scheduler = scheduler.or(model.scheduler);
                if model.model_options.deterministic_enabled() {
                    crate::note!("{}", deterministic::apply_serving(model)?);
                }
                model.collection_policy()?;
                let origin = |cli: bool, supplied: bool| {
                    if cli {
                        "CLI"
                    } else if supplied {
                        "YAML"
                    } else {
                        "absent"
                    }
                };
                crate::note!(
                    "collection {}: max_batch={:?} ({}), wait_ms={:?} ({}), batch_window_ms={:?} ({})",
                    config::describe(model),
                    model.max_batch,
                    origin(max_batch.is_some(), model.max_batch.is_some()),
                    model.wait_ms,
                    origin(wait_ms.is_some(), model.wait_ms.is_some()),
                    model.batch_window_ms,
                    origin(batch_window_ms.is_some(), model.batch_window_ms.is_some())
                );
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
                    g.ple_placement.is_none(),
                    "--ple-placement applies to --model; with a configuration file set `ple_placement` per model in it"
                );
                anyhow::ensure!(
                    g.readout_layer.is_none(),
                    "--readout-layer applies to --model; with a configuration file set `readout_layer` per model in it"
                );
                anyhow::ensure!(
                    g.merge_projections.is_none(),
                    "--merge-projections applies to --model; with a configuration file set `merge_projections` per \
                     model in it"
                );
                anyhow::ensure!(
                    g.layer_split.is_none(),
                    "--layer-split applies to --model; with a configuration file set `layer_split` per model in it"
                );
                anyhow::ensure!(
                    g.backend.is_none() && g.cpu_work_mb.is_none(),
                    "--backend and --cpu-work-mb apply to --model; with a configuration file set `backend` and \
                     `cpu_work_mb` per model in it"
                );
            } else {
                // models from the command line: the options of the command line
                for m in &mut c.models {
                    m.backend = g.backend.unwrap_or_default();
                    m.cpu_work_mb = g.cpu_work_mb;
                    // a catalog model's GPU dtype (f16) does not apply on the CPU backend
                    if g.dtype != "auto" || m.backend == basal_gpu::ExecutionBackend::Cpu {
                        m.dtype = g.dtype.clone();
                    }
                    m.inference_mode = g.inference_mode.clone();
                    m.quantization = g.quantization.clone().unwrap_or_else(|| Quantization::None.as_str().into());
                    m.ple_placement = g.ple_placement.unwrap_or_default();
                    m.readout_layer = g.readout_layer.unwrap_or_default();
                    m.merge_projections = g.merge_projections.unwrap_or_default();
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
            // The strict profile against every model's final fields, before any placement, GPU query or download.
            for m in c.models.iter().filter(|m| m.model_options.deterministic_enabled()) {
                let model = config::describe(m);
                let background = matches!(
                    m.batch_qualification()?,
                    serve::batches::config::BackgroundQualification::Qualified { .. }
                );
                deterministic::profile(&deterministic::Requested {
                    model: &model,
                    options: &m.model_options,
                    backend: m.backend,
                    dtype: &m.dtype,
                    kernels: g.kernels(),
                    readout: &g.readout,
                    readout_layer: m.readout_layer,
                    quantization: &m.quantization,
                    ple: m.ple_placement,
                    inference_mode: &m.inference_mode,
                    instances: m.replica_devices.as_ref().map_or(1, Vec::len),
                    split: m.layer_split.is_some(),
                    media: m.media || g.media,
                    state_cache_mb: m.state_cache_mb,
                    gemm_table: m.gemm_table.as_str().into(),
                    background,
                })?;
            }
            for model in &c.models {
                let routing = model.routing_policy()?;
                crate::note!("routing {}: {}", config::describe(model), routing.identity());
                if let Some(identity) = model.scheduler_policy()?.identity() {
                    crate::note!("scheduler {}: {}", config::describe(model), identity);
                }
            }
            serve::check_admission_limits(c.max_inflight, c.max_large_inflight)?;
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
            // The Batch API uses the root database, or without one its local SQLite default. An enabled root
            // database without an enabled consumer is only migrated at start (with auto_migrate), never otherwise
            // connected.
            let database = match (&batch_settings, database_section) {
                (Some(batches), section) => Some(batches.database(&section)?),
                (None, database::DatabaseSection::Enabled(settings)) => Some(settings),
                (None, database::DatabaseSection::Absent | database::DatabaseSection::Disabled) => None,
            };
            let batch_qualifications =
                c.models.iter().map(config::ModelConfig::batch_qualification).collect::<Result<Vec<_>, _>>()?;
            // the CPU backend's static refusals and per-forward ceilings, before any download or GPU
            let cpu_works = c
                .models
                .iter()
                .map(|m| config::cpu_work(m, g.kernels.as_deref(), &g.readout))
                .collect::<Result<Vec<_>>>()?;
            if cpu_works.iter().any(Option::is_some) {
                anyhow::ensure!(
                    c.models.len() == 1,
                    "backend: cpu serves exactly one model per process in this release"
                );
                anyhow::ensure!(!g.media, "media is not implemented on the CPU backend");
            }
            // devices of every model's instances, its weight format and layer split; the GPUs of a replica group or
            // a layer split are compared before any download
            let placements: Vec<config::Placement> = c
                .models
                .iter()
                .map(|m| {
                    let model = config::describe(m);
                    config::check_ple_source(m.ple_placement, m.layer_split.is_some(), m)?;
                    config::check_quantization(&model, &m.quantization, &m.dtype, g.kernels(), cfg!(feature = "cuda"))?;
                    let placement = m.placement().with_context(|| format!("model {model}"))?;
                    if let Some(split) = placement.split() {
                        config::check_layer_split(&model, split, &m.dtype, g.kernels(), cfg!(feature = "cuda"))?;
                    }
                    Ok(placement)
                })
                .collect::<Result<_>>()?;
            // with a custom thinking source in any model, the GPUs are compared only after every custom source is
            // validated and admitted (below); with explicit merge_projections in any model, after every model's and
            // thinking checkpoint's configuration is validated
            let custom_thinking = c.models.iter().any(|m| m.model_options.custom_thinking());
            let defer_gpu_identity =
                custom_thinking || c.models.iter().any(|m| m.merge_projections != basal_gpu::MergeProjections::Auto);
            for (m, placement) in c.models.iter().zip(&placements) {
                if matches!(placement, config::Placement::Replicas(d) if d.len() > 1) {
                    anyhow::ensure!(
                        cfg!(feature = "cuda"),
                        "model {}: several `replica_devices` need the CUDA build",
                        config::describe(m)
                    );
                }
                if !defer_gpu_identity {
                    config::check_same_gpus(&format!("model {}", config::describe(m)), placement)?;
                }
            }
            // model files (local or downloaded), then GEMM tables (a search needs no model weights in GPU memory),
            // then the models
            let dirs: Vec<PathBuf> = c.models.iter().map(config::model_dir).collect::<Result<_>>()?;
            let manifests: Vec<ModelManifest> = dirs
                .iter()
                .zip(&c.models)
                .map(|(d, m)| m.model_options.load_manifest(d, m.orders.as_deref(), &m.source_name()))
                .collect::<Result<_>>()?;
            for (m, manifest) in c.models.iter().zip(&manifests) {
                if m.model_options.deterministic_enabled() {
                    deterministic::check_manifest(manifest)?;
                }
            }
            let served_names = c.model_names(&manifests)?;
            // Validate every second checkpoint and calibration before any model's GEMM search or GPU load.
            let mut thinking: Vec<_> = c
                .models
                .iter()
                .zip(&manifests)
                .zip(&placements)
                .map(|((mc, manifest), placement)| {
                    if mc.model_options.custom_thinking() {
                        mc.model_options.check_custom_think_execution(
                            mc.quantization.parse()?,
                            mc.media || g.media,
                            gemm_artifact::TableRequest::parse(&mc.gemm_table)?.reproducibility(),
                            matches!(placement, config::Placement::Replicas(d) if d.len() > 1),
                            matches!(placement, config::Placement::Cpu),
                        )?;
                    }
                    check_merge_thinking(mc.merge_projections, &mc.model_options)?;
                    mc.model_options.check_think_execution(
                        Kernels::parse(g.kernels())?,
                        mc.quantization.parse()?,
                        parse_inference_mode(&mc.inference_mode)?,
                        placement.split().is_some(),
                        gemm_artifact::TableRequest::parse(&mc.gemm_table)?.reproducibility(),
                    )?;
                    mc.model_options.load_thinking(manifest)
                })
                .collect::<Result<_>>()?;
            // Validate architecture-specific placement and exact layer coverage before any GEMM table or load; the
            // projection shape policy of every model and of its thinking checkpoint before any device query.
            let mut projection_policies = Vec::with_capacity(c.models.len());
            for (((mc, manifest), placement), t) in c.models.iter().zip(&manifests).zip(&placements).zip(&thinking) {
                let quantization: Quantization =
                    mc.quantization.parse().with_context(|| format!("model {}", manifest.name))?;
                if matches!(placement, config::Placement::Cpu) {
                    config::check_cpu_architecture(manifest)?;
                }
                config::check_architecture(manifest, quantization, placement.split().is_some())?;
                mc.ple_placement.resolve(&manifest.config, placement.split().is_some())?;
                mc.readout_layer.resolve(&manifest.config, Kernels::parse(g.kernels())?)?;
                let base = separate_projections(
                    mc.merge_projections,
                    &manifest.config,
                    g.kernels(),
                    quantization,
                    &manifest.name,
                )?;
                let think = match t {
                    Some(t) => separate_projections(
                        mc.merge_projections,
                        &t.manifest.config,
                        g.kernels(),
                        quantization,
                        &format!("{} thinking checkpoint", manifest.name),
                    )?,
                    None => false,
                };
                projection_policies.push((base, think));
                if mc.model_options.exits_enabled() {
                    let readout = g.readout()?;
                    GpuBackend::check_early_exit(
                        Kernels::parse(g.kernels())?,
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
            // All separately loaded host tables plus each worker's own PLE call reservation.
            // Check before GEMM/device loading; mapped checkpoint pages and load scratch are separate estimates.
            let mut host_ple_load = 0u64;
            for (((mc, manifest), placement), qualification) in
                c.models.iter().zip(&manifests).zip(&placements).zip(&batch_qualifications)
            {
                if mc.ple_placement == basal_gpu::PleMode::Host {
                    let memory = GpuBackend::host_ple_memory(&manifest.dir, &manifest.config)?;
                    let background = batch_settings.is_some()
                        && matches!(qualification, serve::batches::config::BackgroundQualification::Qualified { .. });
                    let workers = mc.routing_policy()?.workers(background);
                    let replicas = u64::try_from(placement.instances().len())?;
                    let bytes = memory
                        .host_load_bytes(workers)?
                        .checked_mul(replicas)
                        .context("host PLE replica estimate overflow")?;
                    host_ple_load = host_ple_load.checked_add(bytes).context("host PLE server estimate overflow")?;
                    crate::note!(
                        "{}: host PLE {} loaded replica(s), {} workers/replica; source estimate {}",
                        manifest.name,
                        replicas,
                        workers,
                        serde_json::to_string(&memory)?
                    );
                }
            }
            if host_ple_load > 0 {
                // The global Batch pool also owns plans/results; do not treat it as a shared worker buffer.
                let pool = batch_settings.as_ref().map_or(0, |b| b.host.exec_pool_bytes);
                config::check_host_ple_headroom(
                    host_ple_load.checked_add(pool).context("host PLE plus Batch pool overflow")?,
                )?;
            }
            // The CPU model (the only one): resident, load scratch and one ceiling per worker (main and long lane,
            // counted although one gate serializes them), plus the Batch pool, before anything loads.
            for ((mc, manifest), work) in c.models.iter().zip(&manifests).zip(&cpu_works) {
                let Some(work) = work else { continue };
                let memory = GpuBackend::cpu_host_memory(&manifest.dir, &manifest.config)?;
                let workers = mc.routing_policy()?.workers(false);
                let pool = batch_settings.as_ref().map_or(0, |b| b.host.exec_pool_bytes);
                let bytes =
                    memory.startup_bytes(workers, work.get())?.checked_add(pool).context("CPU estimate overflow")?;
                config::check_host_headroom(bytes, "the CPU backend")?;
                crate::note!(
                    "{}: CPU {} worker(s) with a {} byte forward ceiling each, plus {pool} Batch pool bytes; source \
                     estimate {} (named tensor payloads, not allocator peaks, RSS or OOM protection)",
                    manifest.name,
                    workers,
                    work,
                    serde_json::to_string(&memory)?
                );
            }
            // Custom thinking checkpoints: coverage of their own layers by a split, then host and device admission of
            // every custom instance together, then the GPU comparisons deferred above; before any precision rule or
            // GEMM table.
            if custom_thinking {
                let mut instances = Vec::new();
                for (t, placement) in thinking.iter().zip(&placements) {
                    let Some(model_options::LoadedThinking { manifest: think, custom: Some(custom), .. }) = t else {
                        continue;
                    };
                    let layers = think.config.num_layers();
                    if let Some(split) = placement.split() {
                        check_custom_split(split, layers)?;
                    }
                    for devices in placement.instances() {
                        let stages: think_source::Stages = match placement.split() {
                            None => devices.gpus().iter().map(|&d| (d, 0..layers)).collect(),
                            Some(split) => devices.gpus().iter().copied().zip(split.ranges(layers)?).collect(),
                        };
                        instances.push((&custom.verified.resources, stages));
                    }
                }
                if !instances.is_empty() {
                    let host = think_source::server_host(&instances.iter().map(|(r, _)| *r).collect::<Vec<_>>())?;
                    think_source::admit(host, &instances)?;
                }
            }
            if defer_gpu_identity {
                for (m, placement) in c.models.iter().zip(&placements) {
                    config::check_same_gpus(&format!("model {}", config::describe(m)), placement)?;
                }
            }
            // one table per model, chosen on its first device (first phase of the execution plan) and checked on
            // every device of the model before anything loads; none on the host CPU; a custom thinking checkpoint's
            // own table right after its base model's
            let mut tables = Vec::new();
            for ((((mc, manifest), placement), t), &(separate, think_separate)) in
                c.models.iter().zip(&manifests).zip(&placements).zip(&mut thinking).zip(&projection_policies)
            {
                let Some(device) = placement.first_device() else {
                    tables.push(gemm_artifact::ResolvedGemm::none(false));
                    continue;
                };
                let precision = config::model_precision(&mc.dtype, manifest, device)?;
                let profile = models::profile_for(manifest);
                let table = gemm_artifact::resolve_for_projections(
                    &gemm_artifact::TableRequest::parse(&mc.gemm_table)?,
                    manifest,
                    profile,
                    precision,
                    &c.gemm_cache,
                    &placement.instances().iter().flat_map(|i| i.gpus().iter().copied()).collect::<Vec<_>>(),
                    separate,
                )?;
                config::check_table_on_devices(manifest, table.table.as_ref(), placement)?;
                tables.push(table);
                if let Some(model_options::LoadedThinking { manifest: think, custom: Some(custom), .. }) = t {
                    let devices: Vec<usize> =
                        placement.instances().iter().flat_map(|i| i.gpus().iter().copied()).collect();
                    custom.resolve_gemm(think, precision, &c.gemm_cache, &devices, think_separate)?;
                    config::check_table_on_devices(
                        think,
                        custom.gemm.as_ref().and_then(|r| r.table.as_ref()),
                        placement,
                    )?;
                }
            }
            let mut served = Vec::new();
            for ((((((mc, table), manifest), thinking), placement), batch_qualification), name) in c
                .models
                .iter()
                .zip(tables)
                .zip(manifests)
                .zip(thinking)
                .zip(placements)
                .zip(batch_qualifications)
                .zip(served_names)
            {
                let instances = placement.instances();
                let mut replicas = Vec::with_capacity(instances.len());
                for devices in instances {
                    let mut mg = GpuArgs {
                        model_options: mc.model_options.clone(),
                        media: mc.media || g.media,
                        orders: mc.orders.clone(),
                        dtype: mc.dtype.clone(),
                        inference_mode: mc.inference_mode.clone(),
                        readout: g.readout.clone(),
                        kernels: g.kernels.clone(),
                        backend: None,
                        cpu_work_mb: None,
                        readout_layer: Some(mc.readout_layer),
                        merge_projections: Some(mc.merge_projections),
                        quantization: Some(mc.quantization.clone()),
                        ple_placement: Some(mc.ple_placement),
                        no_prefix_cache: mc.no_prefix_cache,
                        state_cache_mb: mc.state_cache_mb,
                        tree_max_tokens: mc.tree_max_tokens,
                        gemm_table: None,
                        resolved_gemm: None,
                        device: None,
                        layer_split: None,
                    };
                    match &devices {
                        config::InstanceDevices::Gpu(ds) => {
                            mg.resolved_gemm = Some(table.clone());
                            mg.device = placement.split().is_none().then(|| ds[0]);
                            mg.layer_split = placement.split().cloned();
                        }
                        config::InstanceDevices::Cpu => {
                            mg.backend = Some(basal_gpu::ExecutionBackend::Cpu);
                            mg.cpu_work_mb = mc.cpu_work_mb;
                        }
                    }
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
                            &name,
                            table.table.as_ref().map(basal_gpu::GemmArtifact::bytes_sha256),
                        )?
                        .with_background(batch_qualification.clone()),
                    )
                } else {
                    None
                };
                served.push(serve::ServedModel {
                    name,
                    replicas,
                    max_batch_tokens: mc.max_batch_tokens,
                    collection: mc.collection_policy()?,
                    scheduler: mc.scheduler_policy()?,
                    routing: mc.routing_policy()?,
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
                long_share: c.long_share,
                access_log: access_log || c.access_log,
                metrics: metrics || c.metrics,
                database: database.map(database::Database::new),
                batches: batch_settings,
            };
            serve::serve(served, opts)?;
        }
        Cmd::GemmSearch { m, dtype, out, max_m, m_classes, invariant, device, merge_projections } => {
            #[cfg(feature = "cuda")]
            {
                let classes: Vec<usize> = if !m_classes.is_empty() {
                    m_classes
                } else if invariant {
                    config::INVARIANT_CLASSES.to_vec()
                } else {
                    (16..=512).step_by(16).chain((640..=max_m).step_by(128)).collect()
                };
                let config = ModelManifest::load(&m.dir()?)?.config;
                let separate =
                    merge_projections.unwrap_or_default().resolve(&config, Kernels::Fused, Quantization::None)?;
                let shapes = basal_gpu::planned_projection_shapes(&config, separate)?;
                let t = basal_gpu::gemm_search(&classes, &shapes, &dtype, invariant, device)?;
                write_json(&out, &t)?;
            }
            #[cfg(not(feature = "cuda"))]
            bail!(
                "gemm-search needs a CUDA build ({}, {dtype}, {}, {max_m}, {m_classes:?}, {invariant}, {device}, \
                 {merge_projections:?})",
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
            anyhow::ensure!(
                !g.model_options.deterministic_enabled(),
                "bench batches independent items; the deterministic profile {} is unavailable here",
                basal_gpu::DeterministicProfile::ID
            );
            anyhow::ensure!(
                g.backend != Some(basal_gpu::ExecutionBackend::Cpu) && g.cpu_work_mb.is_none(),
                "bench measures GPU resources; the CPU backend has no benchmark telemetry in this release"
            );
            let batch = parse_batching(&batching)?;
            anyhow::ensure!(batch != basal_core::Batching::Single, "bench --batching requires budget or tree");
            let mut engine = gpu_engine(&m, &g)?;
            let load_s = engine.backend.load_s();
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
