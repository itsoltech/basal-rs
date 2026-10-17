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
//!     gemm_table: auto             # auto (default) | none | path of a `basal gemm-search` table
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
//! A model is addressed by the name in its `basal.json` (the `model` field of requests).

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
    /// Directory of the GEMM tables of `gemm_table: auto`.
    #[serde(default = "default_gemm_cache")]
    pub gemm_cache: PathBuf,
    pub models: Vec<ModelConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
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
    /// auto (Llama f16; validated Omni on Ada f16, other Gemma bf16), f16, bf16 or f32.
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
    /// models only (Omni/Gemma refuses it); needs `dtype` f16 (`auto` for Llama) and the fused kernels;
    /// `quantization` applies to every stage. Not together with `device` or `replica_devices`. Results are the same
    /// computations as on one GPU; the stages run one after another.
    #[serde(default)]
    pub layer_split: Option<LayerSplit>,
    /// auto, accurate (Llama original, Gemma FP32 attention), or balanced (CUDA FP16 Llama).
    #[serde(default = "default_inference_mode")]
    pub inference_mode: String,
    /// `auto`: the table of this model and GPU from `gemm_cache`, generated when missing or not valid here; `none`:
    /// algorithms timed at first use (results may depend on the batch); otherwise the path of a table.
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
/// Omni (Gemma) model is refused quantization and layer split by [`check_architecture`] once its config is read.
fn static_precision(model: &str, dtype: &str) -> Result<Precision> {
    if dtype == "auto" {
        return Ok(Precision::F16);
    }
    Precision::parse(dtype).with_context(|| format!("model {model}"))
}

/// Combinations this runtime does not implement for the model's architecture, refused once its config is read and
/// before any GEMM table or load: projection quantization and layer split exist for Llama models only (basal-1.5,
/// Basal 2.0 mini/4.5B/max). Omni (Gemma 4) runs on one GPU (`device`) or as `replica_devices`.
pub fn check_architecture(manifest: &ModelManifest, quantization: Quantization, split: bool) -> Result<()> {
    if matches!(manifest.config, basal_core::architecture::Architecture::Gemma4(_)) {
        ensure!(
            quantization == Quantization::None,
            "model {}: quantization {quantization} is implemented for Llama models only; an Omni (Gemma 4) model \
             needs `quantization: none`",
            manifest.name
        );
        ensure!(
            !split,
            "model {}: layer_split is implemented for Llama models only; serve an Omni (Gemma 4) model on one GPU \
             (`device`) or as `replica_devices`",
            manifest.name
        );
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

/// The one GEMM table of a replica group or a layer split is batch-invariant and valid on every one of its GPUs
/// (cuBLASLt version, GPU name and every algorithm at its M class), checked before anything loads. A replica group
/// needs a table, except Omni (Gemma, `manifest`), whose GEMMs without one are fixed per prompt (independent of the
/// batch); a split without one (`gemm_table: none`) times its algorithms at first use on every GPU.
pub fn check_table_on_devices(manifest: &ModelManifest, table: Option<&Path>, placement: &Placement) -> Result<()> {
    let name = &manifest.name;
    let Some(devices) = placement.shared_devices() else { return Ok(()) };
    let split = matches!(placement, Placement::Split(_));
    let Some(path) = table else {
        if split || matches!(manifest.config, basal_core::architecture::Architecture::Gemma4(_)) {
            return Ok(());
        }
        bail!("{name}: several `replica_devices` need a batch-invariant GEMM table (CUDA build, gemm_table not none)")
    };
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", crate::term::P(&path)))?;
    let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", crate::term::P(&path)))?;
    let role = if split { "layer split" } else { "replica" };
    for d in devices {
        basal_gpu::check_gemm_table(&v, d)
            .with_context(|| format!("{name}: GEMM table {} on {role} device {d}", crate::term::P(&path)))?;
    }
    Ok(())
}

impl ServeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", crate::term::P(&path)))?;
        Self::parse(&text).with_context(|| format!("parsing {}", crate::term::P(&path)))
    }

    fn parse(text: &str) -> Result<Self> {
        let c: Self = serde_saphyr::from_str_with_options(text, yaml_options())?;
        ensure_single_document(text)?;
        ensure!(!c.models.is_empty(), "no models");
        for model in &c.models {
            crate::parse_inference_mode(&model.inference_mode)?;
        }
        crate::serve::check_admission_limits(c.max_inflight, c.max_large_inflight)?;
        Ok(c)
    }

    /// Configuration with default options for the given models (`basal serve --model`, the built-in one).
    pub fn for_models(models: Vec<ModelConfig>) -> Self {
        let mut c = Self::parse("models: [{path: x}]").expect("default configuration");
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

/// A published basal model pinned to a metadata revision. Runtime verification is recorded separately in reports.
pub struct KnownModel {
    /// short name for `basal init --model`
    pub short: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    /// checkpoint size (download), GB
    pub size_gb: f64,
}

pub const KNOWN_MODELS: [KnownModel; 10] = [
    KnownModel {
        short: "mini",
        repo: "Remek/basal-1.5-mini",
        revision: "1978d0705ce09cd2d7d8d3e87b304468121e7255",
        size_gb: 3.1,
    },
    KnownModel {
        short: "4.5B",
        repo: "Remek/basal-1.5-4.5B",
        revision: "784a683bfadcc8865238fc0fc74a83b4000269bc",
        size_gb: 9.1,
    },
    KnownModel {
        short: "max",
        repo: "Remek/basal-1.5-max",
        revision: "be1b5ee7e7a9755a931262fa7fab4f59be0fd03c",
        size_gb: 22.3,
    },
    KnownModel {
        short: "2.0-mini",
        repo: "Remek/basal-2.0-mini",
        revision: "cb81b32ce89b0b87e62fcdad28f0435e25c953f3",
        size_gb: 3.1,
    },
    KnownModel {
        short: "2.0-4.5B",
        repo: "Remek/basal-2.0-4.5B",
        revision: "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
        size_gb: 9.1,
    },
    KnownModel {
        short: "2.0-max",
        repo: "Remek/basal-2.0-max",
        revision: "2fffc5cc41e5cf7a2f752f38f17e7b4d9a4a8252",
        size_gb: 22.3,
    },
    KnownModel {
        short: "omni-mini",
        repo: "Remek/basal-omni-1.0-mini",
        revision: "73c993dc3d9ab3cb8887ac0b1ee1cf1c92783240",
        size_gb: 15.9,
    },
    KnownModel {
        short: "omni-max",
        repo: "Remek/basal-omni-1.0-max",
        revision: "e538228c4f3ffd96dde298a073ed10247da0d637",
        size_gb: 23.9,
    },
    KnownModel {
        short: "omni-1.1-max",
        repo: "Remek/basal-omni-1.1-max",
        revision: "c8ab50cd973c116b3ca6bd9e4d574f713bab1f68",
        size_gb: 23.92,
    },
    KnownModel {
        short: "omni-ultra",
        repo: "Remek/basal-omni-1.0-ultra",
        revision: "b8e214cb42d0a7a4cb2c3a93b0ba02c65d7590e4",
        size_gb: 62.55,
    },
];

/// Model of the built-in configuration: the pinned basal-2.0-4.5B checkpoint.
/// Keep index 4 assigned to basal-2.0-4.5B when changing [`KNOWN_MODELS`].
pub const DEFAULT_MODEL: &KnownModel = &KNOWN_MODELS[4];

/// Owner of a repository given without one (`--model basal-1.5-max` is `Remek/basal-1.5-max`).
pub const DEFAULT_OWNER: &str = "Remek";

/// A model on the command line (`basal serve --model`, `basal init --model`, the `--model` of the other commands):
/// a local model directory (an existing path, or one starting with `/`, `.` or `~`), else a Hugging Face repository
/// `owner/name` or `name` (owner Remek; 2.0-mini/2.0-4.5B/2.0-max and omni-mini/omni-max;
/// mini/4.5B/max for basal-1.5), each with an optional
/// `@revision` (branch, tag or commit). Without a revision a model of [`KNOWN_MODELS`] is the revision this runtime
/// pins, any other repository its main branch.
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
        let known = KNOWN_MODELS.iter().find(|m| {
            m.short.eq_ignore_ascii_case(name)
                || m.repo.eq_ignore_ascii_case(name)
                || m.repo.rsplit('/').next().is_some_and(|n| n.eq_ignore_ascii_case(name))
        });
        let repo = match known {
            Some(k) => k.repo.to_string(),
            None if name.contains('/') => name.to_string(),
            None => format!("{DEFAULT_OWNER}/{name}"),
        };
        ensure!(repo.split('/').count() == 2, "model {s:?}: expected name, owner/name or a local directory");
        let rev = rev.map(str::to_string).or_else(|| known.map(|k| k.revision.to_string()));
        // Unknown repositories need architecture-aware precision after their metadata is loaded.
        // Known Omni checkpoints also resolve precision after the GPU is known.
        let dtype = match known {
            Some(m) if m.short.starts_with("omni-") => "auto",
            Some(_) => "f16",
            None => "auto",
        };
        serde_json::json!({ "repo": repo, "revision": rev, "dtype": dtype })
    };
    Ok(serde_json::from_value(entry)?)
}

/// Checkpoint size (GB) of a model of [`KNOWN_MODELS`].
pub fn known_size_gb(m: &ModelConfig) -> Option<f64> {
    let repo = m.repo.as_deref()?;
    KNOWN_MODELS.iter().find(|k| k.repo == repo).map(|k| k.size_gb)
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
    "CALIBRATION.o*.json",
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
    /// none of them: [`DEFAULT_MODEL`] with default options
    Builtin,
}

/// The configuration: `--config FILE`, else the `--model`s, else `BASAL_CONFIG` or `basal-serve.yml` in the working
/// directory, else [`DEFAULT_MODEL`].
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
    Ok((Source::Builtin, ServeConfig::for_models(vec![model_ref(DEFAULT_MODEL.short)?])))
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
    (MODEL_FILES[..6].iter().all(|f| dir.join(f).exists()) && basal_core::checkpoint::files(&dir).is_ok())
        .then_some(dir)
}

/// Directory of a configured model: `path`, or the Hugging Face snapshot of `repo` at `revision` (downloaded when
/// missing; a cached snapshot is used when the Hub cannot be reached).
pub fn model_dir(m: &ModelConfig) -> Result<PathBuf> {
    match (&m.path, &m.repo) {
        (Some(p), None) => {
            ensure!(m.revision.is_none(), "model {}: revision needs repo", crate::term::P(&p));
            Ok(p.clone())
        }
        (None, Some(repo)) => download(repo, m.revision.as_deref()),
        (Some(_), Some(_)) => bail!("model entry: give either path or repo, not both"),
        (None, None) => bail!("model entry: needs path or repo"),
    }
}

fn download(repo: &str, revision: Option<&str>) -> Result<PathBuf> {
    let (owner, name) = repo.split_once('/').with_context(|| format!("repo {repo:?}: expected owner/name"))?;
    let client = hf_hub::HFClientSync::new().context("Hugging Face client")?;
    let r = client.model(owner, name);
    let at = revision.unwrap_or("main");
    // a commit as its first 8 characters in the messages
    let at = if at.len() == 40 && at.bytes().all(|b| b.is_ascii_hexdigit()) { &at[..8] } else { at };
    let files: Vec<String> = MODEL_FILES.iter().map(|f| f.to_string()).collect();
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
    for f in &MODEL_FILES[..6] {
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

/// CLI auto precision: use the validated FP16 decoder for the two measured Omni
/// model names on RTX 6000 Ada (GPU `device`; replicas run on GPUs of one type). Explicit precision always wins;
/// media towers stay BF16. Other Gemma devices keep the published BF16 default.
pub fn model_precision(value: &str, manifest: &ModelManifest, device: usize) -> Result<basal_gpu::Precision> {
    if value == "auto"
        && matches!(&manifest.config, basal_core::architecture::Architecture::Gemma4(_))
        && matches!(manifest.name.as_str(), "basal-omni-1.0-mini" | "basal-omni-1.0-max")
        && basal_gpu::gpu_name(device).is_ok_and(|name| name == "NVIDIA RTX 6000 Ada Generation")
    {
        return Ok(basal_gpu::Precision::F16);
    }
    basal_gpu::model_precision(value, &manifest.config)
}

/// GEMM table of a model: `None` for `gemm_table: none` and on builds without CUDA; for `auto` the cached table of
/// this model, dtype, GPU `device` and cuBLASLt version. Llama generates it (one-time, minutes to an hour) on that GPU
/// when it is missing, was made on another GPU or cuBLASLt, or does not cover the model's weight shapes; Gemma falls
/// back to fixed per-prompt GEMMs when no matching cached or bundled table is available. A replica group or a layer
/// split resolves its one table on its first device ([`check_table_on_devices`] then checks it on every device; the
/// file name has no GPU identity, so a split needs GPUs of one type).
pub fn gemm_table(m: &ModelConfig, manifest: &ModelManifest, cache: &Path, device: usize) -> Result<Option<PathBuf>> {
    let mut resolved = m.clone();
    resolved.dtype = match model_precision(&m.dtype, manifest, device)? {
        basal_gpu::Precision::F16 => "f16",
        basal_gpu::Precision::Bf16 => "bf16",
        basal_gpu::Precision::F32 => "f32",
    }
    .into();
    let m = &resolved;
    match m.gemm_table.as_str() {
        "none" => return Ok(None),
        "auto" => {}
        p => return Ok(Some(PathBuf::from(p))),
    }
    // The algorithm search supports half precision only; FP32 uses the reference matmul path.
    if m.dtype == "f32" {
        return Ok(None);
    }
    let Some(version) = basal_gpu::cublaslt_version() else {
        return Ok(None); // Metal: no cuBLASLt tables
    };
    let gpu = basal_gpu::gpu_name(device)?;
    let shapes = basal_gpu::architecture_projection_shapes(&manifest.config)?;
    let path = cache.join(format!("{}-{}.json", manifest.name, m.dtype));
    let problem = match std::fs::read_to_string(&path) {
        Err(_) => Some("no table".to_string()),
        Ok(text) => {
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            table_problem(&v, version, &gpu, &m.dtype, &shapes)
        }
    };
    // a table compiled into this build for the same GPU, cuBLASLt version and shapes, checked on this GPU
    if problem.is_some() {
        for (file, text) in bundled::TABLES {
            let Ok(v) = serde_json::from_str::<Value>(text) else { continue };
            if table_problem(&v, version, &gpu, &m.dtype, &shapes).is_some() {
                continue;
            }
            match basal_gpu::check_gemm_table(&v, device) {
                Ok(()) => {
                    std::fs::create_dir_all(cache).with_context(|| format!("creating {}", crate::term::P(&cache)))?;
                    let tmp = path.with_extension("json.tmp");
                    std::fs::write(&tmp, text)?;
                    std::fs::rename(&tmp, &path)?;
                    crate::done!("{}: GEMM table for {gpu} from this build ({file}), no search needed", manifest.name);
                    return Ok(Some(path));
                }
                Err(e) => crate::warn!("{}: GEMM table {file} of this build rejected ({e:#})", manifest.name),
            }
        }
    }
    if matches!(manifest.config, basal_core::architecture::Architecture::Gemma4(_)) && problem.is_some() {
        crate::note!(
            "{}: using fixed per-prompt GEMMs; an invariant Gemma table for this GPU can be generated with gemm-search",
            manifest.name
        );
        return Ok(None);
    }
    if let Some(why) = problem {
        crate::progress!(
            "{}: generating the batch-invariant GEMM table for {gpu} (device {}; {why}; {} classes x {} shapes, \
             one-time) -> {}",
            manifest.name,
            device,
            INVARIANT_CLASSES.len(),
            shapes.len(),
            crate::term::P(&path)
        );
        let t = std::time::Instant::now();
        let mut table = search(&shapes, &m.dtype, device)?;
        // for `basal gemm-share`: which build and model made the table
        table["model"] = Value::from(manifest.name.clone());
        table["basal"] = Value::from(format!("{} ({})", env!("CARGO_PKG_VERSION"), env!("BASAL_GIT_SHA")));
        std::fs::create_dir_all(cache).with_context(|| format!("creating {}", crate::term::P(&cache)))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&table)? + "\n")?;
        std::fs::rename(&tmp, &path)?;
        crate::done!("{}: GEMM table done in {:.0} s", manifest.name, t.elapsed().as_secs_f64());
        crate::note!(
            "basal {} has no GEMM table of {} for {gpu}: `basal gemm-share` sends this one to the basal-rs project \
             (a GitHub issue, after your confirmation), so that later builds spare other users of this GPU the search",
            env!("CARGO_PKG_VERSION"),
            manifest.name
        );
    }
    Ok(Some(path))
}

/// GEMM tables compiled into this build (crates/basal-cli/gemm-tables, reduced by build.rs; none without CUDA).
pub mod bundled {
    include!(concat!(env!("OUT_DIR"), "/gemm_tables.rs"));
}

/// GPUs with a GEMM table in this build.
pub fn bundled_gpus() -> Vec<String> {
    let mut g: Vec<String> = bundled::TABLES
        .iter()
        .filter_map(|(_, t)| serde_json::from_str::<Value>(t).ok()?["gpu"].as_str().map(str::to_string))
        .collect();
    g.sort();
    g.dedup();
    g
}

/// Why a GEMM table cannot be used for this GPU, cuBLASLt version, dtype and model shapes (None: it can).
fn table_problem(v: &Value, version: usize, gpu: &str, dtype: &str, shapes: &[(usize, usize)]) -> Option<String> {
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

fn search(shapes: &[(usize, usize)], dtype: &str, device: usize) -> Result<Value> {
    #[cfg(feature = "cuda")]
    return basal_gpu::gemm_search(&INVARIANT_CLASSES, shapes, dtype, true, device);
    #[cfg(not(feature = "cuda"))]
    bail!("GEMM search needs a CUDA build ({shapes:?}, {dtype}, device {device})")
}

/// Refuse a GEMM table that leaves weight shapes of the model to run-time algorithm choice.
pub fn check_gemm_coverage(name: &str, missing: &[(usize, usize)], table: &Path) -> Result<()> {
    if !missing.is_empty() {
        bail!(
            "{name}: GEMM table {} has no entries for the weight shapes {missing:?} (generate one with `basal \
             gemm-search --model ... --invariant` or use gemm_table: auto)",
            crate::term::P(&table)
        );
    }
    Ok(())
}
