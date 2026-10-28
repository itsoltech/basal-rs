//! `basal serve --config FILE` (or basal-serve.yml in the working directory): the models one server process serves
//! and their options (YAML). `basal serve --model` takes the models from the command line instead (model_ref).
//!
//! ```yaml
//! addr: 0.0.0.0:8000
//! default_model: basal-1.5-max     # /v1/basal requests without "model" (default: the first model)
//! gemm_cache: .cache/gemm          # tables written by `gemm_table: auto` (default: the user cache, paths.rs)
//! models:
//!   - repo: Remek/basal-1.5-max    # downloaded from Hugging Face at start (HF_HOME cache, HF_TOKEN)
//!     revision: be1b5ee7e7a9755a931262fa7fab4f59be0fd03c
//!     gemm_table: auto             # auto (default) | pinned | sha256:<digest> | none | path of a table
//!   - path: .models/basal-1.5-mini # or a local model directory
//!     long_tokens: 0
//!     device: 1                    # CUDA device (default 0); models on one device take turns on it
//!   - path: .models/basal-1.5-4.5B
//!     replica_devices: [2, 3]      # CUDA: one replica per device under one model name (GPUs of one type)
//!     quantization: q8_0           # none (default) | q8_0 | q4_k: 8- or 4-bit projection weights (CUDA, dtype f16)
//!   - path: .models/basal-1.5-max
//!     layer_split:                 # CUDA: consecutive layer ranges of one model on several GPUs, in layer order
//!       - {device: 4, layers: 25}  # layers 0-24 and the embedding
//!       - {device: 5, layers: 25}  # layers 25-49, final norm, lm_head, evidence head
//! ```
//!
//! A model is addressed by the name in its `basal.json` (the `model` field of requests). The optional durable features
//! have their own sections: `database:` (`crate::database`, docs/DATABASE.md) and `batches:`
//! (`serve::batches::config`, docs/BATCHES.md).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use basal_core::ModelManifest;
use basal_gpu::{Kernels, LayerSplit, Precision, Quantization};
use serde::Deserialize;
use serde_json::Value;

use crate::serve::Schedule;

/// M classes of a batch-invariant GEMM table: the forward sizes for which the fastest algorithm of the chosen
/// group of bitwise-identical algorithms is recorded.
pub const INVARIANT_CLASSES: [usize; 25] = [
    16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 3072, 4096, 6144,
    8192, 12288, 16384,
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeConfig {
    #[serde(default)]
    pub mcp: crate::mcp::Switch,
    #[serde(default)]
    pub systemone_dialect: crate::serve::SystemOneDialect,
    /// Append final successful online decisions; absent: no log file, hashing or record serialization.
    #[serde(default)]
    pub log_decisions: Option<PathBuf>,
    #[serde(default = "default_addr")]
    pub addr: SocketAddr,
    /// Waiting + running requests of all models; more are refused with 529.
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,
    /// Large bodies and requests whose expanded input exceeds 128 KiB; shared across models.
    #[serde(default = "default_max_large_inflight")]
    pub max_large_inflight: usize,
    /// `release_date` reported by GET /v1/models.
    #[serde(default = "default_release_date")]
    pub release_date: String,
    /// Model of /v1/basal requests without a "model" field (default: the first model).
    #[serde(default)]
    pub default_model: Option<String>,
    /// Least time a long-request lane works between two hand-overs of the GPU to short batches (any model).
    #[serde(default = "default_long_slice_ms")]
    pub long_slice_ms: u64,
    /// One log line per HTTP request (method, status, path, time); also `basal serve --access-log` or
    /// `BASAL_ACCESS_LOG=1`.
    #[serde(default)]
    pub access_log: bool,
    /// Collect and expose Prometheus metrics at GET /metrics; disabled unless explicitly enabled.
    #[serde(default)]
    pub metrics: bool,
    /// Database of the server's durable features (`crate::database`); absent: no database is opened.
    #[serde(default)]
    pub(crate) database: Option<crate::database::DatabaseConfig>,
    /// Durable Batch/Files API; disabled unless explicitly enabled. Uses `database`.
    #[serde(default)]
    pub(crate) batches: crate::serve::batches::config::BatchConfig,
    /// Directory of the GEMM tables of `gemm_table: auto`.
    #[serde(default = "default_gemm_cache")]
    pub gemm_cache: PathBuf,
    /// Required by `basal serve`; administration commands ([`ServeConfig::load_admin`]) accept a file without it.
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    #[serde(flatten)]
    pub(crate) model_options: crate::model_options::ModelOptions,
    /// Local model directory (config.json, basal.json, tokenizer.json, model.safetensors, CALIBRATION.json, ...).
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Hugging Face repository `owner/name`, downloaded at start into the Hugging Face cache (`HF_HOME`,
    /// default ~/.cache/huggingface; token from `HF_TOKEN`); instead of `path`.
    #[serde(default)]
    pub repo: Option<String>,
    /// Branch, tag or commit of `repo` (default: main). A commit pins the weights.
    #[serde(default)]
    pub revision: Option<String>,
    /// auto (Llama f16; Gemma profile default; Qwen3.5 CUDA bf16, Metal f32), f16, bf16 or f32.
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Format of the projection weights: `none` (default, dense weights of `dtype`), `q8_0` (GGML Q8_0 blocks) or
    /// `q4_k` (GGML Q4_K blocks; every projection's K a multiple of 256), dequantized to f16 per projection: less GPU
    /// memory, more work per forward, different numerics; CUDA with `dtype` f16 (`auto` for Llama) and Llama models only
    /// (Omni/Gemma refuses it). All replicas of the model use it.
    #[serde(default = "default_quantization")]
    pub quantization: String,
    /// GPU of the model: the CUDA device ordinal, an index among the GPUs visible to the process (after
    /// `CUDA_VISIBLE_DEVICES`; `basal doctor` lists them). Models on one device take turns on it; models on different
    /// devices run at the same time. Metal and Intel: 0 only. Not together with `replica_devices`.
    #[serde(default)]
    pub device: Option<usize>,
    /// CUDA: independent replicas of this model under its one name, one per listed device (distinct ordinals of GPUs
    /// of one type: name, compute capability and SM count). Each replica loads the weights and has its own queue,
    /// scheduler, caches and long lane; requests go to the replica with the fewest waiting and running ones. Several
    /// replicas need `gemm_table: auto` or a path, not `none`: Llama replicas share one batch-invariant table; Omni
    /// (Gemma) replicas without a matching table use the batch-independent fixed per-prompt GEMMs. One device is the
    /// same as `device`.
    #[serde(default)]
    pub replica_devices: Option<Vec<usize>>,
    /// CUDA: one model on several GPUs by consecutive layer ranges, `[{device, layers}, ...]` in layer order (at least
    /// two stages on distinct GPUs of one type, every stage with at least one layer, together exactly the model's
    /// layers); stage 0 also holds the embedding, the last stage the final norm, lm_head and evidence head. Llama
    /// models and Gemma without shared KV or per-layer input embeddings; needs explicit `dtype` f16 and fused kernels;
    /// `quantization` applies to every stage. Not together with `device` or `replica_devices`. Results are the same
    /// computations as on one GPU; the stages run one after another.
    #[serde(default)]
    pub layer_split: Option<LayerSplit>,
    /// auto, accurate (Llama original, Gemma FP32 attention), or balanced (CUDA FP16 Llama).
    #[serde(default = "default_inference_mode")]
    pub inference_mode: String,
    /// `auto` (adaptive): this machine's table of the model and GPU in `gemm_cache`, else a bundled table pinned by
    /// the model profile, else a search (Llama; Omni uses fixed per-prompt GEMMs); `pinned` (reproducible): only a
    /// bundled table pinned by the profile, none is an error; `sha256:<digest>` (reproducible): exactly that
    /// artifact (bundled or stored, `basal gemm-digest --import`); `none`: algorithms timed at first use (results may
    /// depend on the batch); otherwise the path of a table (`crate::gemm_artifact`).
    #[serde(default = "default_gemm_table")]
    pub gemm_table: String,
    /// Packed tokens of one batch (a larger request runs alone).
    #[serde(default = "default_max_batch_tokens")]
    pub max_batch_tokens: usize,
    #[serde(default = "default_schedule")]
    pub schedule: Schedule,
    /// Requests above this many packed tokens run in this model's long lane (0 = none).
    #[serde(default = "default_long_tokens")]
    pub long_tokens: usize,
    /// Budget (MiB) of the cross-request state cache (0 = off).
    #[serde(default)]
    pub state_cache_mb: usize,
    #[serde(default = "default_tree_max_tokens")]
    pub tree_max_tokens: usize,
    #[serde(default)]
    pub no_prefix_cache: bool,
    /// Upstream option orders: 1, 2, auto or `2le<N>`; absent uses the trained format's default.
    #[serde(default)]
    pub orders: Option<String>,
    /// Enable native image/audio processing for an omni model on CUDA.
    #[serde(default)]
    pub media: bool,
    /// Explicit evidence and memory limits of background batch execution; disabled by default.
    #[serde(default)]
    pub(crate) batch_profile: crate::serve::batches::config::BackgroundProfileConfig,
}

fn default_addr() -> SocketAddr {
    "127.0.0.1:8000".parse().unwrap()
}
fn default_max_inflight() -> usize {
    1024
}
fn default_max_large_inflight() -> usize {
    crate::serve::DEFAULT_MAX_LARGE_INFLIGHT
}
fn default_release_date() -> String {
    "2026-10-05".into()
}
fn default_long_slice_ms() -> u64 {
    100
}
pub fn default_gemm_cache() -> PathBuf {
    crate::paths::cache_dir().join("gemm")
}
fn default_dtype() -> String {
    "auto".into()
}
fn default_inference_mode() -> String {
    "auto".into()
}
fn default_quantization() -> String {
    Quantization::None.as_str().into()
}
fn default_gemm_table() -> String {
    "auto".into()
}
fn default_max_batch_tokens() -> usize {
    8192
}
fn default_schedule() -> Schedule {
    Schedule::Hrrn
}
fn default_long_tokens() -> usize {
    4096
}
fn default_tree_max_tokens() -> usize {
    basal_core::engine::TREE_MAX_TOKENS
}

impl ModelConfig {
    /// Actual configured source, without a routing alias or an unresolved revision suffix.
    pub(crate) fn source_name(&self) -> String {
        self.repo
            .clone()
            .unwrap_or_else(|| self.path.as_deref().unwrap_or(Path::new("")).to_string_lossy().into_owned())
    }
    /// GPU ordinal of a model without replicas (`device`, default 0).
    pub fn device(&self) -> usize {
        self.device.unwrap_or(0)
    }

    /// Where the model runs: one GPU (`device`), validated `replica_devices` (non-empty, distinct, not together with
    /// `device`; several need a GEMM table) or a `layer_split` (not together with `device` or `replica_devices`; the
    /// split itself is validated when parsed). Whether the ordinals exist and the GPUs match is checked against the
    /// driver ([`check_same_gpus`]).
    pub fn placement(&self) -> Result<Placement> {
        if let Some(split) = &self.layer_split {
            ensure!(self.device.is_none(), "give either `device` or `layer_split`, not both");
            ensure!(self.replica_devices.is_none(), "replicas of a layer-split model are not supported");
            return Ok(Placement::Split(split.clone()));
        }
        let Some(devices) = &self.replica_devices else { return Ok(Placement::Single(self.device())) };
        ensure!(self.device.is_none(), "give either `device` or `replica_devices`, not both");
        ensure!(!devices.is_empty(), "`replica_devices` is empty (give at least one CUDA device)");
        for (i, d) in devices.iter().enumerate() {
            ensure!(!devices[..i].contains(d), "`replica_devices` lists device {d} twice (one replica per GPU)");
        }
        ensure!(
            devices.len() == 1 || self.gemm_table != "none",
            "several `replica_devices` need one batch-invariant GEMM table shared by the replicas: use `gemm_table: \
             auto` or the path of a `basal gemm-search --invariant` table, not `none`"
        );
        Ok(Placement::Replicas(devices.clone()))
    }
}

/// Where a configured model runs ([`ModelConfig::placement`]).
#[derive(Clone, Debug)]
pub enum Placement {
    /// One instance on one GPU.
    Single(usize),
    /// One independent instance per device (one device: a single instance).
    Replicas(Vec<usize>),
    /// One instance whose consecutive layer ranges run on the split's devices.
    Split(LayerSplit),
}

impl Placement {
    /// The devices of every instance: one device per instance, all the split's devices for its one instance.
    pub fn instances(&self) -> Vec<Vec<usize>> {
        match self {
            Self::Single(d) => vec![vec![*d]],
            Self::Replicas(ds) => ds.iter().map(|&d| vec![d]).collect(),
            Self::Split(split) => vec![split.devices()],
        }
    }

    /// The device a table `auto` is resolved on: the first instance's first device (a split's stage 0).
    pub fn first_device(&self) -> usize {
        match self {
            Self::Single(d) => *d,
            Self::Replicas(ds) => ds.first().copied().unwrap_or(0),
            Self::Split(split) => split.devices().first().copied().unwrap_or(0),
        }
    }

    /// Several GPUs that need one GPU type and one GEMM table checked on each: a replica group or a split.
    pub fn shared_devices(&self) -> Option<Vec<usize>> {
        match self {
            Self::Single(_) => None,
            Self::Replicas(ds) => (ds.len() > 1).then(|| ds.clone()),
            Self::Split(split) => Some(split.devices()),
        }
    }

    /// The layer split, if any.
    pub fn split(&self) -> Option<&LayerSplit> {
        match self {
            Self::Split(split) => Some(split),
            _ => None,
        }
    }
}

/// The static checks of a layer split of `model` before any download: the backend (`cuda_backend`: this build's, or
/// the CUDA build of a Linux package for `basal doctor`), `dtype` and kernel set. The split's own form (stages,
/// layers, distinct devices) was checked when it was parsed.
pub fn check_layer_split(
    model: &str,
    split: &LayerSplit,
    dtype: &str,
    kernels: &str,
    cuda_backend: bool,
) -> Result<()> {
    let precision = static_precision(model, dtype)?;
    let kernels = Kernels::parse(kernels).with_context(|| format!("model {model}"))?;
    split.check(precision, kernels, cuda_backend).with_context(|| format!("model {model}: layer_split {split}"))?;
    Ok(())
}

/// A layer split of `model` covers exactly its `num_layers` layers (after `config.json` is available, before any GEMM
/// table or load).
pub fn check_split_coverage(model: &str, split: &LayerSplit, num_layers: usize) -> Result<()> {
    split.ranges(num_layers).with_context(|| format!("model {model}: layer_split {split}"))?;
    Ok(())
}

/// The weight format `quantization` of `model`, checked with its `dtype` and the kernel set against the backend
/// (`cuda_backend`: this build's, or the CUDA build of a Linux package for `basal doctor`) before any download, GEMM
/// table or load. `dtype` and `kernels` are parsed here only for a quantized format (`q8_0`, `q4_k`); otherwise they
/// are checked at load as before.
pub fn check_quantization(
    model: &str,
    quantization: &str,
    dtype: &str,
    kernels: &str,
    cuda_backend: bool,
) -> Result<Quantization> {
    let q: Quantization = quantization.parse().with_context(|| format!("model {model}"))?;
    if q != Quantization::None {
        let precision = static_precision(model, dtype)?;
        let kernels = Kernels::parse(kernels).with_context(|| format!("model {model}"))?;
        q.check(precision, kernels, cuda_backend).with_context(|| format!("model {model}"))?;
    }
    Ok(q)
}

/// Precision of `dtype` for the checks before the model's config is read: `auto` is f16 for every Llama model; an
/// Omni (Gemma) compatibility is checked by [`check_architecture`] once its config is read.
fn static_precision(model: &str, dtype: &str) -> Result<Precision> {
    if dtype == "auto" {
        return Ok(Precision::F16);
    }
    Precision::parse(dtype).with_context(|| format!("model {model}"))
}

/// Combinations this runtime does not implement for the model's architecture, refused once its config is read and
/// before any GEMM table or load. Gemma needs dense weights; a split additionally requires independent KV layers
/// and no per-layer input embeddings.
pub fn check_architecture(manifest: &ModelManifest, quantization: Quantization, split: bool) -> Result<()> {
    if let basal_core::architecture::Architecture::Qwen35(_) = &manifest.config {
        ensure!(
            quantization == Quantization::None && !split,
            "model {}: Qwen3.5 requires `quantization: none` and a single device per replica",
            manifest.name
        );
    }
    if let basal_core::architecture::Architecture::Gemma4(cfg) = &manifest.config {
        ensure!(
            quantization == Quantization::None,
            "model {}: quantization {quantization} is implemented for Llama models only; an Omni (Gemma 4) model \
             needs `quantization: none`",
            manifest.name
        );
        if split {
            basal_gpu::RuntimeBackend::check_gemma_split(cfg).with_context(|| format!("model {}", manifest.name))?;
        }
    }
    Ok(())
}

/// The GPUs of a replica group or a layer split are of one type (name, compute capability, SM count), asked from the
/// driver without creating a context (every ordinal must exist for this process), so a mixed group is refused before
/// any download, GEMM table or load.
pub fn check_same_gpus(model: &str, placement: &Placement) -> Result<()> {
    let Some(devices) = placement.shared_devices() else { return Ok(()) };
    let (role, option) = match placement {
        Placement::Split(_) => ("layer split", "`layer_split`"),
        _ => ("replica", "`replica_devices`"),
    };
    let ids = devices
        .iter()
        .map(|&d| basal_gpu::gpu_identity(d).with_context(|| format!("{model}: {role} device {d}")))
        .collect::<Result<Vec<_>>>()?;
    if ids.iter().any(|id| *id != ids[0]) {
        let gpus: Vec<String> = devices.iter().zip(&ids).map(|(d, id)| format!("device {d}: {id}")).collect();
        let what = if matches!(placement, Placement::Split(_)) { "a layer split needs" } else { "replicas need" };
        bail!(
            "{model}: {what} GPUs of one type (name, compute capability, SM count), {option} mixes them ({})",
            gpus.join("; ")
        );
    }
    Ok(())
}

/// The one GEMM table of a replica group or a layer split is batch-invariant (its cuBLASLt version, GPU name and
/// algorithms were checked on every one of its GPUs by `gemm_artifact::resolve`, before anything loads). A replica
/// group needs a table, except Omni (Gemma, `manifest`), whose GEMMs without one are fixed per prompt (independent
/// of the batch); a split without one (`gemm_table: none`) times its algorithms at first use on every GPU.
pub fn check_table_on_devices(
    manifest: &ModelManifest,
    table: Option<&basal_gpu::GemmArtifact>,
    placement: &Placement,
) -> Result<()> {
    let name = &manifest.name;
    if placement.shared_devices().is_none() {
        return Ok(());
    }
    let split = matches!(placement, Placement::Split(_));
    let Some(table) = table else {
        if split
            || matches!(
                manifest.config,
                basal_core::architecture::Architecture::Gemma4(_) | basal_core::architecture::Architecture::Qwen35(_)
            )
        {
            return Ok(());
        }
        bail!("{name}: several `replica_devices` need a batch-invariant GEMM table (CUDA build, gemm_table not none)")
    };
    let role = if split { "a layer split" } else { "replicas" };
    ensure!(
        table.table()["invariant"].as_bool() == Some(true),
        "{name}: GEMM table {} of {role} is not batch-invariant",
        table.label()
    );
    Ok(())
}

impl ServeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        Self::read(path, true)
    }

    /// The configuration of administration commands (`basal database`, `basal batches`): the same strict sections,
    /// but `models` may be absent because nothing is served.
    pub(crate) fn load_admin(path: &Path) -> Result<Self> {
        Self::read(path, false)
    }

    fn read(path: &Path, serving: bool) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", crate::term::P(&path)))?;
        Self::parse(&text, serving).with_context(|| format!("parsing {}", crate::term::P(&path)))
    }

    fn parse(text: &str, serving: bool) -> Result<Self> {
        let c: Self = serde_saphyr::from_str_with_options(text, yaml_options())?;
        ensure_single_document(text)?;
        ensure!(!serving || !c.models.is_empty(), "no models");
        for model in &c.models {
            model.model_options.validate()?;
            crate::parse_inference_mode(&model.inference_mode)?;
            crate::gemm_artifact::TableRequest::parse(&model.gemm_table)?;
        }
        crate::serve::check_admission_limits(c.max_inflight, c.max_large_inflight)?;
        Ok(c)
    }

    /// Configuration with default options for the given models (`basal serve --model`, the built-in one).
    pub fn for_models(models: Vec<ModelConfig>) -> Self {
        let mut c = Self::parse("models: [{path: x}]", true).expect("default configuration");
        c.models = models;
        c
    }
}

/// Options of the configuration parser: duplicate keys, merge keys (`<<`) and YAML 1.1 boolean words (yes/no/on/off)
/// are refused as with serde_yaml before; an error is one line with its line and column. Comments are not captured
/// (the `comments` feature is off).
fn yaml_options() -> serde_saphyr::Options {
    serde_saphyr::options! {
        duplicate_keys: serde_saphyr::DuplicateKeyPolicy::Error,
        merge_keys: serde_saphyr::MergeKeyPolicy::Error,
        strict_booleans: true,
        no_schema: false,
        legacy_octal_numbers: false,
        reject_unsupported_tags: false,
        with_snippet: false,
    }
}

/// The whole file must be one YAML document. serde-saphyr stops after the first one and ignores a syntax error that
/// follows a document end, so this reads every event once more with the same parser and limits: the first syntax
/// error is returned, and any further document, an empty one included, is refused.
fn ensure_single_document(text: &str) -> Result<()> {
    use serde_saphyr::granit_parser::{Event, Parser};

    let options = serde_saphyr::granit_parser::options! { emit_comments: false };
    let mut documents = 0_usize;
    for event in Parser::new_from_str_with_options(text, options) {
        if matches!(event?.0, Event::DocumentStart(..)) {
            documents += 1;
        }
    }
    ensure!(documents == 1, "a configuration must be one YAML document, found {documents}");
    Ok(())
}

/// A model on the command line (`basal serve --model`, `basal init --model`, the `--model` of the other commands):
/// a local model directory (an existing path, or one starting with `/`, `.` or `~`), else a Hugging Face repository
/// `owner/name` or `name` (owner Remek; 2.0-mini/2.0-4.5B/2.0-max and omni-mini/omni-max;
/// mini/4.5B/max for basal-1.5), each with an optional
/// `@revision` (branch, tag or commit). Without a revision a model of the catalog ([`crate::models`]) is the revision
/// this runtime pins, any other repository its main branch.
pub fn model_ref(s: &str) -> Result<ModelConfig> {
    let s = s.trim();
    ensure!(!s.is_empty(), "empty model");
    let expanded = match s.strip_prefix("~/") {
        Some(rest) => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(s),
    };
    let entry = if expanded.exists() || s.starts_with(['/', '.', '~']) {
        serde_json::json!({ "path": expanded })
    } else {
        let (name, rev) = match s.split_once('@') {
            Some((n, r)) => {
                ensure!(!r.is_empty(), "model {s:?}: empty revision after @");
                (n, Some(r))
            }
            None => (s, None),
        };
        let known = crate::models::find(name);
        let repo = match known {
            Some(k) => k.repo.to_string(),
            None if name.contains('/') => name.to_string(),
            None => format!("{}/{name}", crate::models::DEFAULT_OWNER),
        };
        ensure!(repo.split('/').count() == 2, "model {s:?}: expected name, owner/name or a local directory");
        let rev = rev.map(str::to_string).or_else(|| known.map(|k| k.revision.to_string()));
        // Unknown repositories need architecture-aware precision after their metadata is loaded.
        // Known Omni checkpoints also resolve precision after the GPU is known.
        let dtype = known.map_or("auto", |k| k.cli_dtype());
        serde_json::json!({ "repo": repo, "revision": rev, "dtype": dtype })
    };
    Ok(serde_json::from_value(entry)?)
}

/// Checkpoint size (GB) of a model of the catalog.
pub fn known_size_gb(m: &ModelConfig) -> Option<f64> {
    crate::models::by_repo(m.repo.as_deref()?).map(|k| k.size_gb)
}

/// `owner/name@revision` (commit shortened) or the directory of a model, for messages.
pub fn describe(m: &ModelConfig) -> String {
    match (&m.repo, &m.path) {
        (Some(r), _) => match m.revision.as_deref() {
            Some(v) if v.len() == 40 => format!("{r}@{}", &v[..8]),
            Some(v) => format!("{r}@{v}"),
            None => r.clone(),
        },
        (None, Some(p)) => crate::term::path(p),
        _ => "?".into(),
    }
}

/// Configuration file written by `basal init`.
pub fn template(models: &[ModelConfig]) -> String {
    let mut s = String::from(
        "# basal serve configuration: `basal serve` in this directory reads basal-serve.yml, elsewhere\n\
         # `basal serve --config FILE` (every option: serve.example.yml in the basal-rs repository).\n\
         # Models are downloaded from Hugging Face at the first start (HF_HOME, HF_TOKEN for private ones).\n\n\
         addr: 127.0.0.1:8000             # 0.0.0.0:8000 to serve other machines\n\
         # default_model: basal-2.0-4.5B  # /v1/basal requests without \"model\" (default: the first model)\n\
         # access_log: true               # one log line per request\n\
         # metrics: true                  # enable Prometheus at /metrics (default: false)\n\n\
         models:\n",
    );
    for m in models {
        match (&m.repo, &m.path) {
            (Some(r), _) => {
                s += &format!("  - repo: {r}\n");
                if let Some(v) = &m.revision {
                    s += &format!("    revision: {v}\n");
                }
            }
            (None, Some(p)) => s += &format!("  - path: {}\n", p.display()),
            _ => {}
        }
        if m.dtype != "auto" && m.dtype != "f16" {
            s += &format!("    dtype: {}\n", m.dtype);
        }
        if m.inference_mode != "auto" {
            s += &format!("    inference_mode: {}\n", m.inference_mode);
        }
        if m.media {
            s += "    media: true\n";
        }
        if let Some(orders) = &m.orders {
            s += &format!("    orders: {orders:?}\n");
        }
    }
    s
}

/// Files of a model directory the runtime reads (`evidence_head.pt` is optional: without it `evidence` is refused).
const MODEL_FILES: [&str; 14] = [
    "config.json",
    "basal.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "CALIBRATION.json",
    "model.safetensors",
    "evidence_head.pt",
    "CALIBRATION.*.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "processor_config.json",
    "preprocessor_config.json",
    "audio_preprocessor_config.json",
];

/// Where the configuration of `basal serve` (and `basal doctor`) comes from.
pub enum Source {
    /// a file: `--config`, `BASAL_CONFIG` or `basal-serve.yml` in the working directory
    File(PathBuf),
    /// `--model` on the command line
    Models,
    /// none of them: the default model of the catalog ([`crate::models::default_model`]) with default options
    Builtin,
}

/// The configuration: `--config FILE`, else the `--model`s, else `BASAL_CONFIG` or `basal-serve.yml` in the working
/// directory, else the default model of the catalog.
pub fn select(config: Option<PathBuf>, models: &[String]) -> Result<(Source, ServeConfig)> {
    if let Some(p) = config {
        let c = ServeConfig::load(&p)?;
        return Ok((Source::File(p), c));
    }
    if !models.is_empty() {
        let ms = models.iter().map(|m| model_ref(m)).collect::<Result<Vec<_>>>()?;
        return Ok((Source::Models, ServeConfig::for_models(ms)));
    }
    if let Some(p) = crate::paths::config_file() {
        let c = ServeConfig::load(&p)?;
        return Ok((Source::File(p), c));
    }
    Ok((Source::Builtin, ServeConfig::for_models(vec![model_ref(crate::models::default_model().short)?])))
}

/// Snapshot of `repo` at `revision` in the Hugging Face cache, without contacting the Hub.
pub fn cached_model_dir(repo: &str, revision: Option<&str>) -> Option<PathBuf> {
    let (owner, name) = repo.split_once('/')?;
    let client = hf_hub::HFClientSync::new().ok()?;
    let files: Vec<String> = MODEL_FILES.iter().map(|f| f.to_string()).collect();
    let dir = client
        .model(owner, name)
        .snapshot_download()
        .maybe_revision(revision.map(str::to_string))
        .allow_patterns(files)
        .local_files_only(true)
        .send()
        .ok()?;
    (MODEL_FILES[..5].iter().all(|f| dir.join(f).exists()) && basal_core::checkpoint::files(&dir).is_ok())
        .then_some(dir)
}

/// Directory of a configured model: `path`, or the Hugging Face snapshot of `repo` at `revision` (downloaded when
/// missing; a cached snapshot is used when the Hub cannot be reached).
pub fn model_dir(m: &ModelConfig) -> Result<PathBuf> {
    m.model_options.validate()?;
    let thinking = m.model_options.thinking_enabled()?;
    if thinking && m.repo.is_some() {
        ensure!(
            m.repo.as_deref() == Some(basal_core::thinking::THINK_REPO)
                && m.revision.as_deref() == Some(basal_core::thinking::THINK_REVISION),
            "thinking acquisition requires the published repository and immutable revision"
        );
    }
    match (&m.path, &m.repo) {
        (Some(p), None) => {
            ensure!(m.revision.is_none(), "model {}: revision needs repo", crate::term::P(&p));
            Ok(p.clone())
        }
        (None, Some(repo)) => download(repo, m.revision.as_deref(), thinking),
        (Some(_), Some(_)) => bail!("model entry: give either path or repo, not both"),
        (None, None) => bail!("model entry: needs path or repo"),
    }
}

fn download(repo: &str, revision: Option<&str>, thinking: bool) -> Result<PathBuf> {
    let (owner, name) = repo.split_once('/').with_context(|| format!("repo {repo:?}: expected owner/name"))?;
    let client = hf_hub::HFClientSync::new().context("Hugging Face client")?;
    let r = client.model(owner, name);
    let at = revision.unwrap_or("main");
    // a commit as its first 8 characters in the messages
    let at = if at.len() == 40 && at.bytes().all(|b| b.is_ascii_hexdigit()) { &at[..8] } else { at };
    let mut files: Vec<String> = MODEL_FILES.iter().map(|f| f.to_string()).collect();
    if thinking {
        files.extend(["think/*.json", "think/*.safetensors", "think/chat_template.jinja"].map(str::to_string));
    }
    crate::progress!("{repo}@{at}: fetching the model files (Hugging Face cache)");
    let t = std::time::Instant::now();
    let progress = std::sync::Arc::new(DownloadLog::new(repo));
    let fetched = r
        .snapshot_download()
        .maybe_revision(revision.map(str::to_string))
        .allow_patterns(files.clone())
        .progress(progress.clone())
        .send();
    let dir = match fetched {
        Ok(d) => d,
        Err(e) => {
            crate::warn!("{repo}@{at}: Hub not reachable or download failed ({e}); trying the cache");
            r.snapshot_download()
                .maybe_revision(revision.map(str::to_string))
                .allow_patterns(files)
                .local_files_only(true)
                .send()
                .with_context(|| format!("{repo}@{at}: download failed ({e}) and no cached copy"))?
        }
    };
    // Calibration is optional: explicit/off loads must not require the automatic file.
    for f in &MODEL_FILES[..5] {
        ensure!(dir.join(f).exists(), "{repo}@{at}: {f} missing in {}", crate::term::P(&dir));
    }
    ensure!(
        dir.join("model.safetensors").exists() || dir.join("model.safetensors.index.json").exists(),
        "{repo}@{at}: model checkpoint missing"
    );
    let s = t.elapsed().as_secs_f64();
    let bytes = progress.state.lock().unwrap().0;
    if bytes > 0 {
        crate::done!(
            "{repo}@{at}: downloaded {:.2} GB in {s:.0} s ({:.0} MB/s) into {}",
            bytes as f64 / 1e9,
            bytes as f64 / 1e6 / s.max(1e-3),
            crate::term::P(&dir)
        );
    } else {
        crate::done!("{repo}@{at}: from the Hugging Face cache in {s:.1} s ({})", crate::term::P(&dir));
    }
    Ok(dir)
}

/// Download progress on stderr, at most every 15 s.
struct DownloadLog {
    repo: String,
    state: std::sync::Mutex<(u64, std::collections::HashMap<String, u64>, std::time::Instant)>,
}

impl DownloadLog {
    fn new(repo: &str) -> Self {
        Self { repo: repo.into(), state: std::sync::Mutex::new((0, Default::default(), std::time::Instant::now())) }
    }
}

impl hf_hub::progress::ProgressHandler for DownloadLog {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent, ProgressEvent};
        let ProgressEvent::Download(ev) = event else { return };
        let mut s = self.state.lock().unwrap();
        let done = match ev {
            // the first Start has the totals of the snapshot (single files of it start again)
            DownloadEvent::Start { total_files, total_bytes } => {
                if s.0 == 0 {
                    s.0 = *total_bytes;
                    if *total_bytes == 0 {
                        crate::done!("{}: all files cached", self.repo);
                    } else {
                        crate::progress!(
                            "{}: {total_files} file(s), {:.2} GB to download",
                            self.repo,
                            *total_bytes as f64 / 1e9
                        );
                    }
                }
                return;
            }
            DownloadEvent::Progress { files } => {
                for f in files {
                    s.1.insert(f.filename.clone(), f.bytes_completed);
                }
                s.1.values().sum::<u64>()
            }
            DownloadEvent::AggregateProgress { bytes_completed, .. } => (*bytes_completed).max(s.1.values().sum()),
            _ => return,
        };
        if s.2.elapsed().as_secs() >= 15 && s.0 > 0 {
            s.2 = std::time::Instant::now();
            let total = s.0;
            crate::progress!(
                "{}: {:.2} / {:.2} GB ({:.0}%)",
                self.repo,
                done as f64 / 1e9,
                total as f64 / 1e9,
                100.0 * done as f64 / total as f64
            );
        }
    }
}

/// Precision of `value` (`auto`, f16, bf16, f32) for `manifest` on GPU `device`, by the dispatch rule of the model's
/// profile ([`crate::models`]): `auto` is f16 for Llama; for Omni (Gemma) f16 on the GPUs the profile lists, else
/// the published bf16. The GPU is asked only when the rule needs it. Explicit precision always wins; media towers
/// stay BF16. Replicas and layer splits run on GPUs of one type.
pub fn model_precision(value: &str, manifest: &ModelManifest, device: usize) -> Result<basal_gpu::Precision> {
    let profile = crate::models::profile_for(manifest);
    profile
        .dispatch
        .precision(value, &manifest.config, || basal_gpu::gpu_name(device).ok())
        .with_context(|| format!("model {} (profile {})", manifest.name, profile.id))
}

/// GEMM tables compiled into this build (crates/basal-cli/gemm-tables, reduced by build.rs; none without CUDA).
pub mod bundled {
    include!(concat!(env!("OUT_DIR"), "/gemm_tables.rs"));
}

/// GPUs with a GEMM table in this build.
pub fn bundled_gpus() -> Vec<String> {
    let mut g: Vec<String> = bundled::TABLES
        .iter()
        .filter_map(|(_, _, t)| serde_json::from_str::<Value>(t).ok()?["gpu"].as_str().map(str::to_string))
        .collect();
    g.sort();
    g.dedup();
    g
}

/// Why a GEMM table cannot be used for this GPU, cuBLASLt version, dtype and model shapes (None: it can).
pub(crate) fn table_problem(
    v: &Value,
    version: usize,
    gpu: &str,
    dtype: &str,
    shapes: &[(usize, usize)],
) -> Option<String> {
    let covered = |n: usize, k: usize| {
        v["entries"]
            .as_array()
            .is_some_and(|es| es.iter().any(|e| e["n"].as_u64() == Some(n as u64) && e["k"].as_u64() == Some(k as u64)))
    };
    if v["cublaslt_version"].as_u64() != Some(version as u64) {
        Some(format!("table for cuBLASLt {}", v["cublaslt_version"]))
    } else if v["gpu"].as_str() != Some(gpu) {
        Some(format!("table for GPU {}", v["gpu"]))
    } else if v["invariant"].as_bool() != Some(true) || v["dtype"].as_str() != Some(dtype) {
        Some("not a batch-invariant table of this dtype".into())
    } else if v["gemm_search_version"].as_u64().unwrap_or(1) < basal_gpu::gemm_search_version() {
        Some(format!("table of search version {}", v["gemm_search_version"].as_u64().unwrap_or(1)))
    } else if let Some((n, k)) = shapes.iter().copied().find(|&(n, k)| !covered(n, k)) {
        Some(format!("shape {n}x{k} missing"))
    } else {
        None
    }
}

/// The batch-invariant search of `shapes` in `dtype` on GPU `device` (CUDA builds).
pub(crate) fn search(shapes: &[(usize, usize)], dtype: &str, device: usize) -> Result<Value> {
    #[cfg(feature = "cuda")]
    return basal_gpu::gemm_search(&INVARIANT_CLASSES, shapes, dtype, true, device);
    #[cfg(not(feature = "cuda"))]
    bail!("GEMM search needs a CUDA build ({shapes:?}, {dtype}, device {device})")
}

/// Refuse a GEMM table that leaves weight shapes of the model to run-time algorithm choice.
pub fn check_gemm_coverage(name: &str, missing: &[(usize, usize)], table: &basal_gpu::GemmArtifact) -> Result<()> {
    if !missing.is_empty() {
        bail!(
            "{name}: GEMM table {} has no entries for the weight shapes {missing:?} (generate one with `basal \
             gemm-search --model ... --invariant` or use gemm_table: auto)",
            crate::gemm_artifact::describe(table)
        );
    }
    Ok(())
}
