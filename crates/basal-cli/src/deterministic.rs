//! The opt-in strict deterministic profile `llama-candle-f32.1` (docs/ARCHITECTURE.md): its typed option, the checks
//! of one instance's final merged options, environment and architecture, and the serving rewrite. Everything here
//! runs before any GPU discovery, GEMM table, model load or warm-up. Absent or `off` changes nothing. The profile
//! fixes controls; it does not establish that results repeat bit for bit.

use anyhow::{ensure, Result};
use basal_core::ModelManifest;
use basal_gpu::{DeterministicProfile, ExecutionBackend, PleMode, ReadoutLayer};
use serde::Deserialize;

use crate::config::ModelConfig;
use crate::model_options::ModelOptions;
use crate::serve::backlog::Scheduler;
use crate::serve::routing::LongLane;

/// `deterministic` of the CLI and per-model YAML.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Deterministic {
    Off,
    /// Single-device CUDA Llama, Candle F32 kernels and readout, one whole request per engine call.
    LlamaCandleF32,
}

/// The final merged options of one served instance, whatever adapter (command line, serve, benchmark) produced them.
pub(crate) struct Requested<'a> {
    /// The model as messages name it.
    pub(crate) model: &'a str,
    pub(crate) options: &'a ModelOptions,
    pub(crate) backend: ExecutionBackend,
    pub(crate) dtype: &'a str,
    pub(crate) kernels: &'a str,
    pub(crate) readout: &'a str,
    pub(crate) readout_layer: ReadoutLayer,
    pub(crate) quantization: &'a str,
    pub(crate) ple: PleMode,
    pub(crate) inference_mode: &'a str,
    /// Replica instances of the model (1 without `replica_devices`).
    pub(crate) instances: usize,
    pub(crate) split: bool,
    pub(crate) media: bool,
    pub(crate) state_cache_mb: usize,
    pub(crate) gemm_table: std::borrow::Cow<'a, str>,
    /// A qualified background batch profile.
    pub(crate) background: bool,
}

fn refuse(model: &str, what: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("model {model}: deterministic profile {} {what}", DeterministicProfile::ID)
}

/// Every thinking route and trained exits are refused under the profile, before any descriptor, digest, admission
/// or device query; nothing when it is off.
pub(crate) fn check_options(o: &ModelOptions) -> Result<()> {
    if !o.deterministic_enabled() {
        return Ok(());
    }
    let thinking = o.thinking_enabled()?
        || o.think_numerics.is_some()
        || o.think_source.is_some()
        || o.think_integrity.is_some()
        || o.think_gemm_table.is_some();
    ensure!(
        !thinking,
        "deterministic profile {} refuses thinking (think: off; no think_numerics, think_source, think_integrity or \
         think_gemm_table)",
        DeterministicProfile::ID
    );
    ensure!(
        !o.exits_enabled(),
        "deterministic profile {} refuses trained exits (early_exit, exit_heads)",
        DeterministicProfile::ID
    );
    Ok(())
}

/// None when the profile is absent or off; otherwise every option of the profile's matrix, then the read-only
/// process checks of [`DeterministicProfile::from_process`].
pub(crate) fn profile(r: &Requested<'_>) -> Result<Option<DeterministicProfile>> {
    if !r.options.deterministic_enabled() {
        return Ok(None);
    }
    let m = r.model;
    check_options(r.options)?;
    ensure!(r.backend == ExecutionBackend::Gpu, refuse(m, "runs on the CUDA GPU backend, not backend: cpu"));
    ensure!(matches!(r.dtype, "f32" | "float32"), refuse(m, format_args!("requires dtype f32, got {}", r.dtype)));
    ensure!(r.kernels == "candle", refuse(m, format_args!("requires --kernels candle, got {}", r.kernels)));
    ensure!(r.readout == "f32", refuse(m, format_args!("requires --readout f32, got {}", r.readout)));
    ensure!(r.readout_layer == ReadoutLayer::Auto, refuse(m, "requires readout_layer auto"));
    ensure!(r.quantization == "none", refuse(m, format_args!("requires quantization none, got {}", r.quantization)));
    ensure!(r.ple == PleMode::Device, refuse(m, "requires ple_placement device"));
    ensure!(
        matches!(r.inference_mode, "auto" | "accurate"),
        refuse(m, format_args!("requires inference_mode auto or accurate, got {}", r.inference_mode))
    );
    ensure!(r.instances <= 1, refuse(m, "runs one instance (no several replica_devices)"));
    ensure!(!r.split, refuse(m, "runs on one device (no layer_split)"));
    ensure!(!r.media, refuse(m, "takes text only (no media)"));
    ensure!(r.state_cache_mb == 0, refuse(m, "requires state_cache_mb 0"));
    ensure!(
        !matches!(
            crate::gemm_artifact::TableRequest::parse(&r.gemm_table)?,
            crate::gemm_artifact::TableRequest::Path(_)
        ),
        refuse(
            m,
            format_args!("uses no GEMM table file (gemm_table auto, none, pinned or sha256:), got {}", r.gemm_table)
        )
    );
    ensure!(!r.background, refuse(m, "has no background batch execution (batch_profile.background)"));
    Ok(Some(DeterministicProfile::from_process().map_err(|e| anyhow::anyhow!("model {m}: {e}"))?))
}

/// The profile runs Llama only; checked once `config.json` is read, before precision, GEMM table or load.
pub(crate) fn check_manifest(manifest: &ModelManifest) -> Result<()> {
    ensure!(
        matches!(manifest.config, basal_core::architecture::Architecture::Llama(_)),
        refuse(&manifest.name, "runs Llama models only (not Gemma/Omni or Qwen3.5)")
    );
    Ok(())
}

/// Serving of a strict model, after the CLI/YAML merge and before its collection policy: refuse explicit conflicts
/// (max_batch other than 1, positive wait_ms or batch_window_ms, long_lane on, scheduler on) and rewrite the absent
/// or accepted values to one whole request per call: max_batch 1, wait 0, window 0, no long lane, native schedule
/// (`auto` would enable the short-first backlog once the long lane is off). Returns the requested -> effective note.
pub(crate) fn apply_serving(m: &mut ModelConfig) -> Result<String> {
    let name = crate::config::describe(m);
    ensure!(
        m.max_batch.is_none_or(|n| n == 1),
        refuse(&name, format_args!("serves one whole request per call: max_batch 1, got {:?}", m.max_batch))
    );
    for (field, value) in [("wait_ms", m.wait_ms), ("batch_window_ms", m.batch_window_ms)] {
        ensure!(value.is_none_or(|v| v == 0.0), refuse(&name, format_args!("requires {field} 0, got {value:?}")));
    }
    ensure!(!matches!(m.long_lane, Some(LongLane::On)), refuse(&name, "has no long lane: long_lane off, got on"));
    ensure!(
        !matches!(m.scheduler, Some(Scheduler::On)),
        refuse(&name, "uses the native schedule: scheduler auto or off, got on")
    );
    let note = format!(
        "{name}: deterministic profile {}: max_batch {:?} -> 1, wait_ms {:?} -> 0, batch_window_ms {:?} -> 0, \
         long_lane {:?} -> off, scheduler {:?} -> native, template prefix warm-up off; one whole request per engine \
         call, packed by soam within the request",
        DeterministicProfile::ID,
        m.max_batch,
        m.wait_ms,
        m.batch_window_ms,
        m.long_lane,
        m.scheduler
    );
    m.max_batch = Some(1);
    m.wait_ms = Some(0.0);
    m.batch_window_ms = Some(0.0);
    m.long_lane = Some(LongLane::Off);
    if matches!(m.scheduler, Some(Scheduler::Auto)) {
        m.scheduler = None;
    }
    Ok(note)
}
