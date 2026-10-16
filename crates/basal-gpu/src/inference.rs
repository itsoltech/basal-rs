//! Per-backend attention policy, fixed before K/V caches and execution lanes exist.

use candle_core::Device;

use crate::{GpuBackend, Kernels, Precision, Readout};

/// Attention arithmetic selected before any template or state K/V is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceMode {
    /// Use the measured model/device policy; preserve explicit `BASAL_ATT` experiments.
    Auto,
    /// Preserve original arithmetic (CUDA Llama: hi/lo tensor cores; full FP32 uses `Precision::F32`).
    Accurate,
    /// FP16 Q/K products, hi/lo probabilities, FP32 softmax and accumulation; CUDA FP16 only.
    Balanced,
}

/// Why an inference mode cannot be selected for this backend.
#[derive(Debug)]
pub enum InferenceConfigError {
    /// Cached K/V or an execution lane already uses the current arithmetic.
    CacheInitialized,
    /// Balanced mode requires CUDA FP16 Llama, fused kernels and FP32 readout.
    UnsupportedBalanced,
}

impl std::fmt::Display for InferenceConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::CacheInitialized => "configure inference before prefix warmup or lane creation",
            Self::UnsupportedBalanced => "balanced inference requires CUDA FP16 Llama, fused kernels and FP32 readout",
        })
    }
}

impl std::error::Error for InferenceConfigError {}

impl GpuBackend {
    pub(super) fn configure_inference(
        &mut self,
        mode: InferenceMode,
        model_name: &str,
    ) -> Result<(), InferenceConfigError> {
        if !self.prefixes.is_empty() || self.gate.is_some() {
            return Err(InferenceConfigError::CacheInitialized);
        }
        let balanced = self.dev.is_cuda()
            && self.precision == Precision::F16
            && self.kernels == Kernels::Fused
            && self.readout == Readout::F32;
        let kernel = match mode {
            InferenceMode::Auto if balanced && measured_balanced(&self.dev, model_name) => {
                Some("attn_tree_tc_qk1_static")
            }
            InferenceMode::Auto => None,
            InferenceMode::Accurate => Some("attn_tree_tc"),
            InferenceMode::Balanced if balanced => Some("attn_tree_tc_qk1_static"),
            InferenceMode::Balanced => return Err(InferenceConfigError::UnsupportedBalanced),
        };
        self.attention_override = kernel;
        Ok(())
    }
}

/// Keep the original policy outside the evaluated model/device scope. A failed optional device-name query
/// also keeps the original policy. Explicit experimental kernel selection always wins over auto selection.
fn measured_balanced(dev: &Device, model_name: &str) -> bool {
    #[cfg(feature = "cuda")]
    {
        if std::env::var_os("BASAL_ATT").is_some()
            || !matches!(model_name, "basal-2.0-mini" | "basal-2.0-4.5B" | "basal-2.0-max")
        {
            return false;
        }
        match dev {
            Device::Cuda(d) => {
                d.cuda_stream().context().name().is_ok_and(|name| name == "NVIDIA RTX 6000 Ada Generation")
            }
            _ => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (dev, model_name);
        false
    }
}
