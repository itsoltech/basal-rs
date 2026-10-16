//! Dispatch a validated architecture to its native backend, retaining the existing Llama implementation unchanged.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use basal_core::architecture::Architecture;
use basal_core::gate::Gate;
use basal_core::pack::Packed;
use basal_core::{Backend, DeviceHealth};
use serde_json::Value;

use crate::{GpuMemory, GpuMemoryProbe, InferenceConfigError, InferenceMode, Kernels, Precision, Readout};

pub enum RuntimeBackend {
    Llama(Box<crate::GpuBackend>),
    Gemma(Box<crate::gemma::GemmaBackend>),
    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    Intel(Box<crate::intel::IntelBackend>),
}

impl RuntimeBackend {
    /// Select arithmetic before prefix warmup or lane creation. Explicit balanced mode requires a CUDA FP16 Llama.
    pub fn configure_inference(&mut self, mode: InferenceMode, model_name: &str) -> Result<(), InferenceConfigError> {
        match self {
            Self::Llama(b) => b.configure_inference(mode, model_name),
            _ => {
                if mode == InferenceMode::Balanced {
                    Err(InferenceConfigError::UnsupportedBalanced)
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn load(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        media: bool,
    ) -> Result<Self> {
        match cfg {
            Architecture::Gemma4(c) => {
                Ok(Self::Gemma(Box::new(crate::gemma::GemmaBackend::load(dir, c, precision, readout, media)?)))
            }
            Architecture::Llama(c) => {
                anyhow::ensure!(!media, "media requires a Gemma omni model");
                #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
                return Ok(Self::Intel(Box::new(crate::intel::IntelBackend::load(
                    dir, c, precision, readout, kernels,
                )?)));
                #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
                Ok(Self::Llama(Box::new(crate::GpuBackend::load(dir, c, precision, readout, kernels)?)))
            }
        }
    }

    pub fn load_s(&self) -> f64 {
        match self {
            Self::Llama(b) => b.load_s,
            Self::Gemma(b) => b.load_s,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.load_s,
        }
    }

    pub fn set_state_cache_bytes(&mut self, bytes: usize) -> Result<()> {
        match self {
            Self::Llama(b) => b.state_cache_bytes = bytes,
            Self::Gemma(_) => anyhow::ensure!(bytes == 0, "Gemma cross-request state cache is not supported; use 0"),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => {
                anyhow::ensure!(bytes == 0, "Intel cross-request state cache is not supported");
                b.state_cache_bytes = bytes;
            }
        }
        Ok(())
    }

    pub fn state_cache_stats(&self) -> (u64, u64) {
        match self {
            Self::Llama(b) => (b.state_cache_hits, b.state_cache_inserts),
            Self::Gemma(_) => (0, 0),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => (b.state_cache_hits, b.state_cache_inserts),
        }
    }

    pub fn supports_prefix_cache(&self) -> bool {
        !matches!(self, Self::Gemma(_))
    }

    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Llama(b) => b.gemm_table_missing(),
            Self::Gemma(_) => vec![],
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.gemm_table_missing(),
        }
    }

    pub fn load_gemm_table(&mut self, path: &Path) -> Result<usize> {
        match self {
            Self::Llama(b) => b.load_gemm_table(path),
            Self::Gemma(_) => anyhow::bail!("Gemma does not use Llama cuBLASLt algorithm tables"),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.load_gemm_table(path),
        }
    }

    pub fn enable_profile(&mut self, residual: bool) -> Result<()> {
        match self {
            Self::Llama(b) => {
                b.profile = Some(Default::default());
                if residual {
                    b.residual_max = Some(Default::default());
                }
            }
            Self::Gemma(_) => anyhow::bail!("per-section profiling is not implemented for Gemma"),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => {
                b.profile = Some(Default::default());
                if residual {
                    b.residual_max = Some(Default::default());
                }
            }
        }
        Ok(())
    }

    pub fn take_profile(&mut self) -> BTreeMap<&'static str, f64> {
        match self {
            Self::Llama(b) => b.profile.take().map(|p| p.into_inner()).unwrap_or_default(),
            Self::Gemma(_) => Default::default(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.profile.take().map(|p| p.into_inner()).unwrap_or_default(),
        }
    }

    pub fn take_residual_max(&mut self) -> Vec<f32> {
        match self {
            Self::Llama(b) => b.residual_max.take().map(|p| p.into_inner()).unwrap_or_default(),
            Self::Gemma(_) => vec![],
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.residual_max.take().map(|p| p.into_inner()).unwrap_or_default(),
        }
    }

    pub fn memory_probe(&self) -> GpuMemoryProbe {
        match self {
            Self::Llama(b) => b.memory_probe(),
            Self::Gemma(b) => GpuMemoryProbe::candle(b.dev.clone()),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.memory_probe(),
        }
    }
    pub fn memory(&self) -> Result<GpuMemory> {
        self.memory_probe().snapshot()
    }

    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    pub fn estimated_weight_bytes(dir: &Path, cfg: &Architecture) -> Result<u64> {
        Ok(crate::intel::IntelBackend::estimated_weight_bytes(dir, cfg.llama()?)?)
    }
}

impl Backend for RuntimeBackend {
    fn media_kinds(&self) -> Vec<basal_core::media::Kind> {
        match self {
            Self::Gemma(b) => b.media_kinds(),
            _ => vec![],
        }
    }
    fn prepare_media(&self, entries: &Value, detail: &str) -> Result<Arc<basal_core::media::ProcessedMedia>> {
        match self {
            Self::Gemma(b) => b.prepare_media(entries, detail),
            _ => anyhow::bail!("this backend does not take media"),
        }
    }
    fn begin_media(&mut self, media: &basal_core::media::ProcessedMedia) -> Result<()> {
        match self {
            Self::Gemma(b) => b.begin_media(media),
            _ => anyhow::bail!("this backend does not take media"),
        }
    }
    fn media_logits(&mut self, prompts: &[&[u32]], codes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Gemma(b) => b.media_logits(prompts, codes),
            _ => anyhow::bail!("this backend does not take media"),
        }
    }
    fn end_media(&mut self) {
        if let Self::Gemma(b) = self {
            b.end_media();
        }
    }
    fn describe(&self) -> Value {
        match self {
            Self::Llama(b) => b.describe(),
            Self::Gemma(b) => b.describe(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.describe(),
        }
    }
    fn letter_logits(&mut self, rows: &[&Packed], codes: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        match self {
            Self::Llama(b) => b.letter_logits(rows, codes),
            Self::Gemma(b) => b.letter_logits(rows, codes),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.letter_logits(rows, codes),
        }
    }
    fn add_prefix(&mut self, ids: &[u32]) -> Result<()> {
        match self {
            Self::Llama(b) => b.add_prefix(ids),
            Self::Gemma(_) => Ok(()),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.add_prefix(ids),
        }
    }
    fn has_evidence(&self) -> bool {
        match self {
            Self::Llama(b) => b.has_evidence(),
            Self::Gemma(_) => false,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.has_evidence(),
        }
    }
    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Llama(b) => b.evidence_scores(ids, t0, t1),
            Self::Gemma(_) => anyhow::bail!("Gemma model has no evidence head"),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.evidence_scores(ids, t0, t1),
        }
    }
    fn fork(&self) -> Result<Self> {
        match self {
            Self::Llama(b) => Ok(Self::Llama(Box::new(b.fork()?))),
            Self::Gemma(b) => Ok(Self::Gemma(Box::new(b.fork()?))),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => Ok(Self::Intel(Box::new(b.fork()?))),
        }
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        match self {
            Self::Llama(b) => b.set_gate(gate, urgent),
            Self::Gemma(b) => b.set_gate(gate, urgent),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_gate(gate, urgent),
        }
    }
    fn device_health(&self) -> Option<DeviceHealth> {
        match self {
            Self::Llama(b) => b.device_health(),
            Self::Gemma(_) => None,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.device_health(),
        }
    }
}
