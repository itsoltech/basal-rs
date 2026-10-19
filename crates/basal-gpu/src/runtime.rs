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

use crate::{
    GpuMemory, GpuMemoryProbe, InferenceConfigError, InferenceMode, Kernels, LayerSplit, Precision, Quantization,
    Readout, StageTransfers,
};

pub enum RuntimeBackend {
    Llama(Box<crate::GpuBackend>),
    Gemma(Box<crate::gemma::GemmaBackend>),
    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    Intel(Box<crate::intel::IntelBackend>),
}

impl RuntimeBackend {
    /// Refuse Gemma configurations whose cross-layer state is not placed by the layer-split implementation.
    pub fn check_gemma_split(cfg: &basal_core::architecture::GemmaConfig) -> Result<()> {
        crate::gemma::GemmaBackend::check_split_config(cfg)
    }

    /// Conservative resident FP16/BF16 checkpoint estimate for a Gemma stage, including first-stage media.
    pub fn estimated_gemma_stage_bytes(
        dir: &Path,
        cfg: &basal_core::architecture::GemmaConfig,
        range: std::ops::Range<usize>,
        media: bool,
    ) -> Result<u64> {
        crate::gemma::GemmaBackend::estimated_stage_bytes(dir, cfg, range, media)
    }

    /// Select arithmetic before prefix warmup or lane creation. Explicit balanced mode requires a CUDA FP16 Llama.
    pub fn configure_inference(&mut self, mode: InferenceMode, model_name: &str) -> Result<(), InferenceConfigError> {
        match self {
            Self::Llama(b) => b.configure_inference(mode, model_name),
            Self::Gemma(b) => b.configure_inference(mode, model_name),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(_) => {
                if mode == InferenceMode::Balanced {
                    Err(InferenceConfigError::UnsupportedBalanced)
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Load a model onto GPU `device`. Projection quantization exists for Llama only: an Omni (Gemma) model is refused
    /// any format other than `none` before its weights are read.
    #[expect(clippy::too_many_arguments, reason = "the load options of every architecture, validated together")]
    pub fn load(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        device: usize,
        media: bool,
    ) -> Result<Self> {
        match cfg {
            Architecture::Gemma4(c) => {
                anyhow::ensure!(
                    quantization == Quantization::None,
                    "quantization {quantization} is implemented for Llama models only; Gemma needs none"
                );
                Ok(Self::Gemma(Box::new(crate::gemma::GemmaBackend::load(dir, c, precision, readout, device, media)?)))
            }
            Architecture::Llama(c) => {
                anyhow::ensure!(!media, "media requires a Gemma omni model");
                #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
                return Ok(Self::Intel(Box::new(crate::intel::IntelBackend::load(
                    dir,
                    c,
                    precision,
                    readout,
                    kernels,
                    quantization,
                    device,
                )?)));
                #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
                Ok(Self::Llama(Box::new(crate::GpuBackend::load(
                    dir,
                    c,
                    precision,
                    readout,
                    kernels,
                    quantization,
                    device,
                )?)))
            }
        }
    }

    /// Load consecutive decoder layer ranges over several CUDA devices, without media encoders.
    pub fn load_split(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
    ) -> Result<Self> {
        Self::load_split_with_media(dir, cfg, precision, readout, kernels, quantization, split, false)
    }

    /// Load consecutive decoder layer ranges on CUDA FP16, with optional Gemma media encoders on the first stage.
    #[expect(clippy::too_many_arguments, reason = "the validated load options of each architecture and media mode")]
    pub fn load_split_with_media(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
        media: bool,
    ) -> Result<Self> {
        if let Architecture::Gemma4(c) = cfg {
            anyhow::ensure!(quantization == Quantization::None, "Gemma layer split needs quantization none");
            return Ok(Self::Gemma(Box::new(crate::gemma::GemmaBackend::load_split(
                dir, c, precision, readout, kernels, split, media,
            )?)));
        }
        anyhow::ensure!(!media, "media requires a Gemma omni model");
        let c = cfg.llama()?;
        #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
        {
            let _ = (dir, c, precision, readout, kernels, quantization);
            anyhow::bail!("layer split {split}: needs the CUDA backend, this build runs Intel Vulkan")
        }
        #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
        Ok(Self::Llama(Box::new(crate::GpuBackend::load_split(
            dir,
            c,
            precision,
            readout,
            kernels,
            quantization,
            split,
        )?)))
    }

    /// Transfer counters of a layer split (shared with the forks); None on one GPU and for other backends.
    pub fn stage_transfers(&self) -> Option<StageTransfers> {
        match self {
            Self::Llama(b) => b.stage_transfers(),
            Self::Gemma(b) => b.stage_transfers(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(_) => None,
        }
    }

    /// `q8_0` / `q4_k` while profiling: the format, synchronised dequantization time and values (Llama CUDA only).
    pub fn dequant_profile(&self) -> Option<(Quantization, f64, u64)> {
        match self {
            Self::Llama(b) => b.dequant_profile(),
            Self::Gemma(_) => None,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.dequant_profile(),
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
            Self::Gemma(b) => b.gemm_table_missing(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.gemm_table_missing(),
        }
    }

    pub fn load_gemm_table(&mut self, path: &Path) -> Result<usize> {
        match self {
            Self::Llama(b) => b.load_gemm_table(path),
            Self::Gemma(b) => b.load_gemm_table(path),
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

    /// One memory probe per device of the model, with its ordinal, in stage order (one entry on one GPU).
    pub fn memory_probes(&self) -> Vec<(usize, GpuMemoryProbe)> {
        match self {
            Self::Llama(b) => b.memory_probes(),
            Self::Gemma(b) => vec![(b.device, GpuMemoryProbe::candle(b.dev.clone()))],
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => vec![(0, b.memory_probe())],
        }
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
    fn set_device_gate(&mut self, device: usize, gate: Arc<Gate>, urgent: bool) -> Result<()> {
        match self {
            Self::Llama(b) => b.set_device_gate(device, gate, urgent),
            Self::Gemma(b) => b.set_device_gate(device, gate, urgent),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_device_gate(device, gate, urgent),
        }
    }
    fn device_health(&self) -> Option<DeviceHealth> {
        match self {
            Self::Llama(b) => b.device_health(),
            Self::Gemma(b) => b.device_health(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.device_health(),
        }
    }
}
