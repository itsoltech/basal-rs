//! `basal serve --config FILE`: the models one server process serves and their options (YAML).
//!
//! ```yaml
//! addr: 0.0.0.0:8000
//! default_model: basal-1.5-max     # /v1/basal requests without "model" (default: the first model)
//! gemm_cache: .cache/gemm          # tables written by `gemm_table: auto`
//! models:
//!   - path: .models/basal-1.5-max
//!     gemm_table: auto             # auto (default) | none | path of a `basal gemm-search` table
//!   - path: .models/basal-1.5-mini
//!     long_tokens: 0
//! ```
//!
//! A model is addressed by the name in its `basal.json` (the `model` field of requests).

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use basal_core::ModelManifest;
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
    /// Waiting + running requests of all models; more are refused with 503.
    #[serde(default = "default_max_inflight")]
    pub max_inflight: usize,
    /// `release_date` reported by GET /v1/models.
    #[serde(default = "default_release_date")]
    pub release_date: String,
    /// Model of /v1/basal requests without a "model" field (default: the first model).
    #[serde(default)]
    pub default_model: Option<String>,
    /// Least time a long-request lane works between two hand-overs of the GPU to short batches (any model).
    #[serde(default = "default_long_slice_ms")]
    pub long_slice_ms: u64,
    /// Directory of the GEMM tables of `gemm_table: auto`.
    #[serde(default = "default_gemm_cache")]
    pub gemm_cache: PathBuf,
    pub models: Vec<ModelConfig>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model directory (config.json, basal.json, tokenizer.json, model.safetensors, CALIBRATION.json).
    pub path: PathBuf,
    /// f16 (default), bf16 or f32.
    #[serde(default = "default_dtype")]
    pub dtype: String,
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
}

fn default_addr() -> SocketAddr {
    "127.0.0.1:8000".parse().unwrap()
}
fn default_max_inflight() -> usize {
    1024
}
fn default_release_date() -> String {
    "2026-10-05".into()
}
fn default_long_slice_ms() -> u64 {
    100
}
fn default_gemm_cache() -> PathBuf {
    ".cache/gemm".into()
}
fn default_dtype() -> String {
    "f16".into()
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

impl ServeConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let c: Self = serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        ensure!(!c.models.is_empty(), "{}: no models", path.display());
        Ok(c)
    }
}

/// GEMM table of a model: `None` for `gemm_table: none` and on builds without CUDA; for `auto` the cached table of
/// this model, dtype, GPU and cuBLASLt version, generated (one-time, minutes to an hour) when it is missing, was made
/// on another GPU or cuBLASLt, or does not cover the model's weight shapes.
pub fn gemm_table(m: &ModelConfig, manifest: &ModelManifest, cache: &Path) -> Result<Option<PathBuf>> {
    match m.gemm_table.as_str() {
        "none" => return Ok(None),
        "auto" => {}
        p => return Ok(Some(PathBuf::from(p))),
    }
    let Some(version) = basal_gpu::cublaslt_version() else {
        return Ok(None); // Metal: no cuBLASLt tables
    };
    let gpu = basal_gpu::gpu_name()?;
    let shapes = basal_gpu::projection_shapes(&manifest.config);
    let path = cache.join(format!("{}-{}.json", manifest.name, m.dtype));
    let problem = match std::fs::read_to_string(&path) {
        Err(_) => Some("no table".to_string()),
        Ok(text) => {
            let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            let covered = |n: usize, k: usize| {
                v["entries"].as_array().is_some_and(|es| {
                    es.iter().any(|e| e["n"].as_u64() == Some(n as u64) && e["k"].as_u64() == Some(k as u64))
                })
            };
            if v["cublaslt_version"].as_u64() != Some(version as u64) {
                Some(format!("table for cuBLASLt {}", v["cublaslt_version"]))
            } else if v["gpu"].as_str() != Some(gpu.as_str()) {
                Some(format!("table for GPU {}", v["gpu"]))
            } else if v["invariant"].as_bool() != Some(true) || v["dtype"].as_str() != Some(m.dtype.as_str()) {
                Some("not a batch-invariant table of this dtype".into())
            } else if let Some((n, k)) = shapes.iter().copied().find(|&(n, k)| !covered(n, k)) {
                Some(format!("shape {n}x{k} missing"))
            } else {
                None
            }
        }
    };
    if let Some(why) = problem {
        eprintln!(
            "basal: {}: generating the batch-invariant GEMM table for {gpu} ({why}; {} classes x {} shapes, one-time) -> {}",
            manifest.name,
            INVARIANT_CLASSES.len(),
            shapes.len(),
            path.display()
        );
        let t = std::time::Instant::now();
        let table = search(&shapes, &m.dtype)?;
        std::fs::create_dir_all(cache).with_context(|| format!("creating {}", cache.display()))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&table)? + "\n")?;
        std::fs::rename(&tmp, &path)?;
        eprintln!("basal: {}: GEMM table done in {:.0} s", manifest.name, t.elapsed().as_secs_f64());
    }
    Ok(Some(path))
}

fn search(shapes: &[(usize, usize)], dtype: &str) -> Result<Value> {
    #[cfg(feature = "cuda")]
    return basal_gpu::gemm_search(&INVARIANT_CLASSES, shapes, dtype, true);
    #[cfg(not(feature = "cuda"))]
    bail!("GEMM search needs a CUDA build ({shapes:?}, {dtype})")
}

/// Refuse a GEMM table that leaves weight shapes of the model to run-time algorithm choice.
pub fn check_gemm_coverage(name: &str, missing: &[(usize, usize)], table: &Path) -> Result<()> {
    if !missing.is_empty() {
        bail!(
            "{name}: GEMM table {} has no entries for the weight shapes {missing:?} (generate one with `basal \
             gemm-search --model ... --invariant` or use gemm_table: auto)",
            table.display()
        );
    }
    Ok(())
}
