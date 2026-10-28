//! Per-backend execution policy, applied from a frozen [`ResolvedExecutionPlan`] before K/V caches and execution
//! lanes exist.

use crate::plan::{ExecutionPolicy, KernelSet, ResolvedExecutionPlan};
use crate::{GpuBackend, LlamaNumerics};

/// Attention arithmetic selected before any template or state K/V is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceMode {
    /// Use the measured model/device policy of the profile; preserve explicit `BASAL_ATT` experiments.
    Auto,
    /// High-precision attention: original hi/lo on CUDA Llama, FP32 products on Gemma.
    /// Full FP32 computation uses `Precision::F32`.
    Accurate,
    /// FP16 Q/K products, hi/lo probabilities, FP32 softmax and accumulation; CUDA FP16 only.
    Balanced,
}

/// Why an execution plan cannot be selected or applied for this backend.
#[derive(Debug)]
pub enum InferenceConfigError {
    /// Cached K/V or an execution lane already uses the current arithmetic.
    CacheInitialized,
    /// Balanced mode requires CUDA FP16 Llama, fused kernels and FP32 readout.
    UnsupportedBalanced,
    /// The plan was frozen for another family, stage count or load options than this backend.
    PlanMismatch(&'static str),
    /// The plan's kernel set is not implemented by this build.
    UnsupportedKernelSet(KernelSet),
}

impl std::fmt::Display for InferenceConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CacheInitialized => f.write_str("configure inference before prefix warmup or lane creation"),
            Self::UnsupportedBalanced => {
                f.write_str("balanced inference requires CUDA FP16 Llama, fused kernels and FP32 readout")
            }
            Self::PlanMismatch(what) => write!(f, "execution plan does not match the loaded backend: {what}"),
            Self::UnsupportedKernelSet(set) => write!(f, "kernel set {} is not implemented by this build", set.id()),
        }
    }
}

impl std::error::Error for InferenceConfigError {}

/// The checks shared by every backend before a plan is applied.
pub(crate) fn check_plan(plan: &ResolvedExecutionPlan) -> Result<(), InferenceConfigError> {
    if !KernelSet::SUPPORTED.contains(&plan.kernel_set()) {
        return Err(InferenceConfigError::UnsupportedKernelSet(plan.kernel_set()));
    }
    Ok(())
}

impl GpuBackend {
    /// Apply the Llama attention policy of `plan` (the same on every stage). Refused once template K/V or a lane
    /// exists, and for a plan of other load options.
    pub(super) fn apply_plan(&mut self, plan: &ResolvedExecutionPlan) -> Result<(), InferenceConfigError> {
        if !self.prefixes.is_empty() || self.stages.iter().any(|s| s.gate.is_some()) {
            return Err(InferenceConfigError::CacheInitialized);
        }
        check_plan(plan)?;
        if plan.precision() != self.precision
            || plan.readout() != self.readout
            || plan.kernels() != self.kernels
            || plan.quantization() != self.quantization
        {
            return Err(InferenceConfigError::PlanMismatch("precision, readout, kernels or quantization"));
        }
        let ExecutionPolicy::Llama { attention, default_kernels, .. } = plan.policy() else {
            return Err(InferenceConfigError::PlanMismatch("a Gemma policy for a Llama backend"));
        };
        if self.is_cuda() && default_kernels.len() != self.stages.len() {
            return Err(InferenceConfigError::PlanMismatch("stage count"));
        }
        let numerics = match plan.kernel_set() {
            KernelSet::V1 => LlamaNumerics::Ordinary,
            KernelSet::LlamaThinkBf16V1 if self.is_cuda() && self.stages.len() == 1 => LlamaNumerics::ThinkBf16,
            _ => return Err(InferenceConfigError::PlanMismatch("Llama kernel set or placement")),
        };
        #[cfg(feature = "cuda")]
        if numerics == LlamaNumerics::ThinkBf16 && self.stages.iter().any(|s| s.w.get().lt.is_none()) {
            return Err(InferenceConfigError::PlanMismatch("thinking requires cuBLASLt"));
        }
        #[cfg(feature = "cuda")]
        if numerics == LlamaNumerics::ThinkBf16 && plan.overrides().think_gemm().is_some() {
            for s in &self.stages {
                if let Some(lt) = &s.w.get().lt {
                    lt.set_bias_heuristic_first();
                }
            }
        }
        if numerics == LlamaNumerics::ThinkBf16 {
            let d = self.cfg.head_dim;
            let mut rope = vec![0f32; self.cfg.max_positions * d];
            for (pos, row) in rope.chunks_exact_mut(d).enumerate() {
                let (cos, sin) = row.split_at_mut(d / 2);
                crate::think_rope_angles(self.cfg.rope_theta, pos as f32, cos, sin);
            }
            self.rope = rope.into();
        }
        self.numerics = numerics;
        self.attention_override = attention.kernel_override();
        self.attention_defaults.clone_from(default_kernels);
        Ok(())
    }
}
