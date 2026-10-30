//! Dispatch a validated architecture to its native backend, retaining the existing Llama implementation unchanged.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use basal_core::architecture::Architecture;
use basal_core::execution::{BackgroundSupport, ExecutionError, StopToken, WorkClass};
use basal_core::gate::Gate;
use basal_core::pack::Packed;
use basal_core::{Backend, DeviceHealth};
use serde_json::Value;

use crate::plan::ResolvedExecutionPlan;
use crate::{
    GpuMemory, GpuMemoryProbe, InferenceConfigError, Kernels, LayerSplit, Precision, Quantization, Readout,
    StageTransfers,
};

/// Why the CPU backend refuses durable background execution.
const CPU_BACKGROUND: &str = "durable background execution is not qualified on the CPU backend";

/// Source arithmetic for one separately loaded host-PLE replica, not allocator peaks or measured savings.
#[derive(Clone, Debug, serde::Serialize)]
pub struct HostPleMemory {
    pub checkpoint_bytes: u64,
    pub estimated_device_weight_bytes: u64,
    pub host_table_bytes: u64,
    pub max_staging_bytes_per_worker: u64,
    pub checkpoint_mapping_bytes: u64,
    pub estimated_host_load_scratch_bytes: u64,
    pub estimated_device_load_scratch_bytes: u64,
}

impl HostPleMemory {
    /// Additional host payload during load, including all mapped source pages conservatively.
    /// Workers are counted independently even when their device gate serializes GPU execution.
    pub fn host_load_bytes(&self, workers: u64) -> Result<u64> {
        self.max_staging_bytes_per_worker
            .checked_mul(workers)
            .and_then(|n| n.checked_add(self.host_table_bytes))
            .and_then(|n| n.checked_add(self.checkpoint_mapping_bytes))
            .and_then(|n| n.checked_add(self.estimated_host_load_scratch_bytes))
            .ok_or_else(|| anyhow::anyhow!("host PLE load estimate overflow"))
    }
}

/// Where a planned model instance runs ([`RuntimeBackend::load_planned`]).
#[derive(Clone, Copy, Debug)]
pub enum PlannedPlacement<'a> {
    /// The whole model on one GPU.
    Single(usize),
    /// Consecutive layer ranges on the split's GPUs (CUDA).
    Split(&'a LayerSplit),
    /// The whole model on the host CPU; needs a plan frozen by [`ResolvedExecutionPlan::freeze_cpu`].
    Cpu,
}

/// A loaded model instance of one architecture and backend. `CpuLlama` is the shared Llama implementation on the
/// host CPU (every build), dispatched explicitly so that no GPU-only operation is reached through it.
pub enum RuntimeBackend {
    Qwen35(Box<crate::qwen35::Qwen35Backend>),
    Llama(Box<crate::GpuBackend>),
    CpuLlama(Box<crate::GpuBackend>),
    Gemma(Box<crate::gemma::GemmaBackend>),
    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    Intel(Box<crate::intel::IntelBackend>),
}

impl RuntimeBackend {
    /// Placement-aware single-device source estimate. The PLE source range must exist exactly once
    /// and have its validated configuration's byte length before it can be removed from device bytes.
    /// Other checkpoint bytes (including unused tensors/headers) remain conservative; generated device
    /// tensors, allocator rounding, encoders' runtime work and process overhead are outside this estimate.
    pub fn host_ple_memory(dir: &Path, cfg: &Architecture) -> Result<HostPleMemory> {
        let placement = crate::PleMode::Host.resolve(cfg, false)?;
        let Architecture::Gemma4(cfg) = cfg else { unreachable!("validated host PLE family") };
        let host_table_bytes = crate::gemma::GemmaBackend::estimated_host_ple_bytes(cfg)?;
        let crate::PlePlacement::Host { max_staging_bytes } = placement else { unreachable!("host placement") };
        let mut found = false;
        let mut largest = 0;
        let mut header_scratch = 0u64;
        for file in basal_core::checkpoint::files(dir)? {
            let reader = crate::TensorReader::open(&file)?;
            header_scratch = header_scratch.max(reader.base);
            for (name, (start, end)) in &reader.ranges {
                let bytes = end - start;
                if name == "model.language_model.embed_tokens_per_layer.weight" {
                    anyhow::ensure!(
                        !found && bytes == host_table_bytes,
                        "host PLE source range missing, duplicated or inconsistent"
                    );
                    found = true;
                } else {
                    largest = largest.max(bytes);
                }
            }
        }
        anyhow::ensure!(found, "host PLE source tensor is missing");
        let checkpoint_bytes = basal_core::checkpoint::bytes(dir)?;
        Ok(HostPleMemory {
            checkpoint_bytes,
            estimated_device_weight_bytes: checkpoint_bytes
                .checked_sub(host_table_bytes)
                .ok_or_else(|| anyhow::anyhow!("host PLE source exceeds checkpoint size"))?,
            host_table_bytes,
            max_staging_bytes_per_worker: max_staging_bytes,
            checkpoint_mapping_bytes: checkpoint_bytes,
            // Reusable bytes, CPU storage/cast and a conservative transfer copy; header parsing is separate.
            estimated_host_load_scratch_bytes: largest
                .checked_mul(4)
                .and_then(|n| n.checked_add(header_scratch))
                .ok_or_else(|| anyhow::anyhow!("host PLE load scratch estimate overflow"))?,
            estimated_device_load_scratch_bytes: largest
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("host PLE device load scratch estimate overflow"))?,
        })
    }
    /// Host source estimate of a dense BF16 Llama checkpoint on the CPU backend, before loading; validates the
    /// evidence head's tensor metadata. Not allocator peaks, measured RSS or OOM protection.
    pub fn cpu_host_memory(dir: &Path, cfg: &Architecture) -> Result<crate::CpuHostMemory> {
        crate::cpu::host_memory(dir, cfg.llama()?)
    }

    /// Validate the actual pickle tensor inventory on CPU; exact FP32 resident head bytes, before device work.
    pub fn early_exit_resident_bytes(
        artifacts: &basal_core::early_exit::ExitArtifacts,
        cfg: &Architecture,
    ) -> Result<usize> {
        let cfg = cfg.llama()?;
        Ok(crate::early_exit::ExitHeads::load(artifacts, cfg.hidden, cfg.num_layers)?.resident_bytes)
    }

    /// Refuse unsupported combinations before any device discovery, GEMM tuning or weight upload.
    pub fn check_early_exit(
        kernels: Kernels,
        quantization: Quantization,
        readout: Readout,
        split: bool,
        media: bool,
        cfg: &Architecture,
    ) -> Result<()> {
        anyhow::ensure!(cfg!(feature = "cuda") && !cfg!(target_os = "macos")
            && matches!(cfg, Architecture::Llama(_)) && kernels == Kernels::Candle
            && quantization == Quantization::None && readout == Readout::F32 && !split && !media,
            "trained early_exit supports only single-device CUDA Llama with explicit Candle kernels, quantization none, FP32 readout and text input; fused, split, Metal, Intel, Gemma and Qwen are unsupported");
        Ok(())
    }

    /// Published trained-head opt-in. Ordinary constructors/loaders remain plain and never read heads.
    pub fn load_planned_with_exits(
        dir: &Path,
        cfg: &Architecture,
        plan: &ResolvedExecutionPlan,
        placement: PlannedPlacement<'_>,
        media: bool,
        artifacts: &basal_core::early_exit::ExitArtifacts,
    ) -> Result<(Self, Option<usize>)> {
        anyhow::ensure!(!matches!(placement, PlannedPlacement::Cpu), "the CPU backend has no trained exits");
        Self::check_early_exit(
            plan.kernels(),
            plan.quantization(),
            plan.readout(),
            matches!(placement, PlannedPlacement::Split(_)),
            media,
            cfg,
        )?;
        artifacts.check_model_dir(dir)?;
        let (mut backend, classes) = Self::load_planned(dir, cfg, plan, placement, media)?;
        match &mut backend {
            Self::Llama(b) => b.enable_early_exit(artifacts)?,
            _ => anyhow::bail!("trained exits require the CUDA Llama backend"),
        }
        Ok((backend, classes))
    }
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

    /// Load a model instance with the load options of its frozen `plan` (precision, readout, kernels, weight
    /// format), then apply the plan's execution policy and GEMM table before any template K/V, gate or fork exists.
    /// Returns the backend and the number of GEMM table entries it loaded.
    pub fn load_planned(
        dir: &Path,
        cfg: &Architecture,
        plan: &ResolvedExecutionPlan,
        placement: PlannedPlacement<'_>,
        media: bool,
    ) -> Result<(Self, Option<usize>)> {
        anyhow::ensure!(
            plan.cpu_work_bytes().is_some() == matches!(placement, PlannedPlacement::Cpu),
            "the execution plan and placement disagree on the CPU backend"
        );
        let requested = if matches!(plan.ple_placement(), crate::PlePlacement::Host { .. }) {
            crate::PleMode::Host
        } else {
            crate::PleMode::Device
        };
        anyhow::ensure!(
            requested.resolve(cfg, matches!(placement, PlannedPlacement::Split(_)))? == plan.ple_placement(),
            "frozen PLE placement does not match the loaded architecture"
        );
        let (precision, readout, kernels, quantization) =
            (plan.precision(), plan.readout(), plan.kernels(), plan.quantization());
        let backend = match placement {
            PlannedPlacement::Single(device) => {
                if let crate::PlePlacement::Host { .. } = plan.ple_placement() {
                    let Architecture::Gemma4(c) = cfg else { unreachable!("validated host PLE family") };
                    anyhow::ensure!(quantization == Quantization::None, "Gemma needs quantization none");
                    Self::Gemma(Box::new(crate::gemma::GemmaBackend::load_with_ple(
                        dir,
                        c,
                        precision,
                        readout,
                        device,
                        media,
                        plan.ple_placement(),
                    )?))
                } else {
                    Self::load(dir, cfg, precision, readout, kernels, quantization, device, media, plan.kernel_set())?
                }
            }
            PlannedPlacement::Split(split) => {
                Self::load_split_with_media(dir, cfg, precision, readout, kernels, quantization, split, media)?
            }
            PlannedPlacement::Cpu => {
                anyhow::ensure!(!media, "the CPU backend takes text only");
                let Architecture::Llama(c) = cfg else { anyhow::bail!("the CPU backend runs Llama models only") };
                let (Some(ceiling), Some(gemm_threads)) = (plan.cpu_work_bytes(), plan.cpu_gemm_threads()) else {
                    anyhow::bail!("the execution plan has no CPU limits")
                };
                Self::CpuLlama(Box::new(crate::GpuBackend::load_cpu(
                    dir,
                    c,
                    crate::cpu::CpuWork { ceiling, gemm_threads },
                )?))
            }
        };
        Self::finish_planned(backend, plan)
    }

    /// Load a custom thinking checkpoint from its operator-verified `inventory` like [`RuntimeBackend::load_planned`]:
    /// dense Llama on the CUDA or Metal Llama backend, one GPU or a layer split, text only, no evidence head. A CPU
    /// placement or plan, the Intel backend, host PLE or quantization is refused before any device is touched.
    pub fn load_planned_verified(
        inventory: &basal_core::checkpoint::Inventory,
        cfg: &Architecture,
        plan: &ResolvedExecutionPlan,
        placement: PlannedPlacement<'_>,
    ) -> Result<(Self, Option<usize>)> {
        let Architecture::Llama(c) = cfg else { anyhow::bail!("custom thinking checkpoints load as Llama only") };
        anyhow::ensure!(
            !crate::INTEL_BACKEND && !matches!(placement, PlannedPlacement::Cpu) && plan.cpu_work_bytes().is_none(),
            "custom thinking needs the CUDA or Metal Llama backend"
        );
        anyhow::ensure!(
            !matches!(plan.ple_placement(), crate::PlePlacement::Host { .. }),
            "frozen PLE placement does not match the loaded architecture"
        );
        anyhow::ensure!(plan.quantization() == Quantization::None, "custom thinking supports quantization none");
        #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
        {
            let _ = (inventory, c);
            anyhow::bail!("custom thinking needs the CUDA or Metal Llama backend")
        }
        #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
        {
            let (precision, readout, kernels) = (plan.precision(), plan.readout(), plan.kernels());
            let backend = match placement {
                PlannedPlacement::Single(device) => crate::GpuBackend::load_unplanned_verified(
                    inventory,
                    c,
                    precision,
                    readout,
                    kernels,
                    Quantization::None,
                    device,
                )?,
                PlannedPlacement::Split(split) => crate::GpuBackend::load_split_unplanned_verified(
                    inventory,
                    c,
                    precision,
                    readout,
                    kernels,
                    Quantization::None,
                    split,
                )?,
                PlannedPlacement::Cpu => anyhow::bail!("custom thinking needs the CUDA or Metal Llama backend"),
            };
            Self::finish_planned(Self::Llama(Box::new(backend)), plan)
        }
    }

    /// The common tail of the planned loaders: apply the plan's execution policy, then load its GEMM table.
    fn finish_planned(mut backend: Self, plan: &ResolvedExecutionPlan) -> Result<(Self, Option<usize>)> {
        backend.apply_plan(plan)?;
        let classes = match &plan.gemm().table {
            Some(table) => Some(backend.load_gemm_table(
                table.table(),
                &table.label(),
                plan.gemm().fallback == crate::plan::GemmFallback::Refused,
            )?),
            None => None,
        };
        Ok((backend, classes))
    }

    /// Apply the execution policy of `plan`; refused once K/V or an execution lane exists.
    fn apply_plan(&mut self, plan: &ResolvedExecutionPlan) -> Result<(), InferenceConfigError> {
        match self {
            Self::Qwen35(b) => b.apply_plan(plan),
            Self::Llama(b) | Self::CpuLlama(b) => b.apply_plan(plan),
            Self::Gemma(b) => b.apply_plan(plan),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(_) => {
                crate::inference::check_plan(plan)?;
                if plan.kernel_set() != crate::KernelSet::V1 {
                    return Err(InferenceConfigError::PlanMismatch("Intel kernel set"));
                }
                match plan.policy() {
                    crate::plan::ExecutionPolicy::Llama { .. } => Ok(()),
                    crate::plan::ExecutionPolicy::Gemma { .. } | crate::plan::ExecutionPolicy::Qwen35 { .. } => {
                        Err(InferenceConfigError::PlanMismatch("a Gemma policy for the Intel backend"))
                    }
                }
            }
        }
    }

    /// Load a model onto GPU `device`, without its execution policy: only for [`RuntimeBackend::load_planned`], which
    /// applies the plan's policy next. Projection quantization exists for Llama only: an Omni (Gemma) model is refused
    /// any format other than `none` before its weights are read.
    #[expect(clippy::too_many_arguments, reason = "the load options of every architecture, validated together")]
    fn load(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        device: usize,
        media: bool,
        kernel_set: crate::KernelSet,
    ) -> Result<Self> {
        match cfg {
            Architecture::Qwen35(c) => {
                anyhow::ensure!(
                    !media && quantization == Quantization::None,
                    "Qwen3.5 requires text-only input and quantization none"
                );
                Ok(Self::Qwen35(Box::new(crate::qwen35::Qwen35Backend::load(
                    dir, c, precision, readout, device, kernel_set,
                )?)))
            }
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
                Ok(Self::Llama(Box::new(crate::GpuBackend::load_unplanned(
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

    /// Load consecutive decoder layer ranges on CUDA FP16, with optional Gemma media encoders on the first stage;
    /// like [`RuntimeBackend::load`] without the execution policy.
    #[expect(clippy::too_many_arguments, reason = "the validated load options of each architecture and media mode")]
    fn load_split_with_media(
        dir: &Path,
        cfg: &Architecture,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
        media: bool,
    ) -> Result<Self> {
        anyhow::ensure!(!matches!(cfg, Architecture::Qwen35(_)), "Qwen3.5 does not support layer split");
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
        Ok(Self::Llama(Box::new(crate::GpuBackend::load_split_unplanned(
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
            Self::Qwen35(_) | Self::CpuLlama(_) => None,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(_) => None,
        }
    }

    /// `q8_0` / `q4_k` while profiling: the format, synchronised dequantization time and values (Llama CUDA only).
    pub fn dequant_profile(&self) -> Option<(Quantization, f64, u64)> {
        match self {
            Self::Llama(b) => b.dequant_profile(),
            Self::Gemma(_) | Self::Qwen35(_) | Self::CpuLlama(_) => None,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.dequant_profile(),
        }
    }

    pub fn load_s(&self) -> f64 {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.load_s,
            Self::Gemma(b) => b.load_s,
            Self::Qwen35(b) => b.load_s,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.load_s,
        }
    }

    pub fn set_state_cache_bytes(&mut self, bytes: usize) -> Result<()> {
        match self {
            Self::Llama(b) => b.state_cache_bytes = bytes,
            Self::CpuLlama(_) => anyhow::ensure!(bytes == 0, "the CPU backend has no state cache; use 0"),
            Self::Gemma(_) | Self::Qwen35(_) => {
                anyhow::ensure!(bytes == 0, "this architecture does not support cross-request state cache; use 0")
            }
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
            Self::Llama(b) | Self::CpuLlama(b) => (b.state_cache_hits, b.state_cache_inserts),
            Self::Gemma(_) | Self::Qwen35(_) => (0, 0),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => (b.state_cache_hits, b.state_cache_inserts),
        }
    }

    pub fn supports_prefix_cache(&self) -> bool {
        !matches!(self, Self::Gemma(_) | Self::Qwen35(_) | Self::CpuLlama(_))
    }

    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Llama(b) => b.gemm_table_missing(),
            Self::Gemma(b) => b.gemm_table_missing(),
            Self::Qwen35(_) | Self::CpuLlama(_) => vec![],
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.gemm_table_missing(),
        }
    }

    /// Load the plan's validated GEMM table `v` (`name` for messages); `table_only`: a reproducible Llama plan.
    fn load_gemm_table(&mut self, v: &serde_json::Value, name: &str, table_only: bool) -> Result<usize> {
        match self {
            Self::Llama(b) => b.load_gemm_table(v, name, table_only),
            Self::CpuLlama(_) => anyhow::bail!("the CPU backend uses no GEMM tables ({name})"),
            Self::Gemma(b) => b.load_gemm_table(v, name),
            Self::Qwen35(b) => b.load_gemm_table(v, name),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.load_gemm_table(v, name),
        }
    }

    pub fn enable_profile(&mut self, residual: bool) -> Result<()> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => {
                b.profile = Some(Default::default());
                if residual {
                    b.residual_max = Some(Default::default());
                }
            }
            Self::Gemma(_) | Self::Qwen35(_) => {
                anyhow::bail!("per-section profiling is not implemented for this architecture")
            }
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
            Self::Llama(b) | Self::CpuLlama(b) => b.profile.take().map(|p| p.into_inner()).unwrap_or_default(),
            Self::Gemma(_) | Self::Qwen35(_) => Default::default(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.profile.take().map(|p| p.into_inner()).unwrap_or_default(),
        }
    }

    pub fn take_residual_max(&mut self) -> Vec<f32> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.residual_max.take().map(|p| p.into_inner()).unwrap_or_default(),
            Self::Gemma(_) | Self::Qwen35(_) => vec![],
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.residual_max.take().map(|p| p.into_inner()).unwrap_or_default(),
        }
    }

    pub fn memory_probe(&self) -> GpuMemoryProbe {
        match self {
            Self::Llama(b) => b.memory_probe(),
            // its snapshot refuses: there is no GPU memory telemetry for the host CPU
            Self::CpuLlama(_) => GpuMemoryProbe::candle(candle_core::Device::Cpu),
            Self::Gemma(b) => GpuMemoryProbe::candle(b.dev.clone()),
            Self::Qwen35(b) => GpuMemoryProbe::candle(b.dev.clone()),
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
            Self::CpuLlama(_) => vec![],
            Self::Gemma(b) => vec![(b.device, GpuMemoryProbe::candle(b.dev.clone()))],
            Self::Qwen35(b) => vec![(b.device, GpuMemoryProbe::candle(b.dev.clone()))],
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
    fn generation_logits(&mut self, prefix: &[u32], codes: &[u32]) -> Result<Vec<f32>> {
        // Only Llama specializes generation; the others keep the trait's full-prefix default.
        match self {
            // the CPU model runs the Candle padded readout of its letter path
            Self::Llama(b) | Self::CpuLlama(b) => b.generation_logits(prefix, codes),
            Self::Gemma(b) => b.generation_logits(prefix, codes),
            Self::Qwen35(b) => b.generation_logits(prefix, codes),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.generation_logits(prefix, codes),
        }
    }
    fn end_generation(&mut self) -> Result<()> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.end_generation(),
            _ => Ok(()),
        }
    }
    fn generation_work_bound(
        &self,
        shape: &basal_core::planner::DeviceWorkShape,
    ) -> basal_core::planner::DeviceWorkBound {
        match self {
            Self::Llama(b) => b.generation_work_bound(shape),
            // CPU: the Unknown bound of `device_work_bound`
            _ => self.device_work_bound(shape),
        }
    }
    fn generation_host_staging(&self) -> Option<basal_core::planner::HostStaging> {
        match self {
            Self::Llama(b) => b.generation_host_staging(),
            _ => self.host_staging(),
        }
    }
    fn early_exit_caps(&self) -> Option<&basal_core::early_exit::ExitCaps> {
        match self {
            Self::Llama(b) => b.early_exit_caps(),
            _ => None,
        }
    }
    fn letter_logits_with_policy(
        &mut self,
        rows: &[&Packed],
        letters: &[&[Vec<u32>]],
        policy: &str,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        match self {
            Self::Llama(b) => b.letter_logits_with_policy(rows, letters, policy),
            _ => anyhow::bail!("this architecture/backend has no trained early-exit implementation"),
        }
    }
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
    fn asset_logits(&mut self, prompts: &[&[u32]], codes: &[&[u32]], order: &[&[usize]]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Gemma(b) => b.asset_logits(prompts, codes, order),
            _ => anyhow::bail!("this backend does not take image assets"),
        }
    }
    fn end_media(&mut self) {
        if let Self::Gemma(b) = self {
            b.end_media();
        }
    }
    fn describe(&self) -> Value {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.describe(),
            Self::Gemma(b) => b.describe(),
            Self::Qwen35(b) => b.describe(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.describe(),
        }
    }
    fn letter_logits(&mut self, rows: &[&Packed], codes: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.letter_logits(rows, codes),
            Self::Gemma(b) => b.letter_logits(rows, codes),
            Self::Qwen35(b) => b.letter_logits(rows, codes),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.letter_logits(rows, codes),
        }
    }
    fn add_prefix(&mut self, ids: &[u32]) -> Result<()> {
        match self {
            Self::Llama(b) => b.add_prefix(ids),
            Self::Gemma(_) | Self::Qwen35(_) | Self::CpuLlama(_) => Ok(()),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.add_prefix(ids),
        }
    }
    fn has_evidence(&self) -> bool {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.has_evidence(),
            Self::Gemma(_) | Self::Qwen35(_) => false,
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.has_evidence(),
        }
    }
    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.evidence_scores(ids, t0, t1),
            Self::Gemma(_) | Self::Qwen35(_) => anyhow::bail!("this architecture has no evidence head"),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.evidence_scores(ids, t0, t1),
        }
    }
    fn fork(&self) -> Result<Self> {
        match self {
            Self::Llama(b) => Ok(Self::Llama(Box::new(b.fork()?))),
            Self::CpuLlama(b) => Ok(Self::CpuLlama(Box::new(b.fork()?))),
            Self::Gemma(b) => Ok(Self::Gemma(Box::new(b.fork()?))),
            Self::Qwen35(b) => Ok(Self::Qwen35(Box::new(b.fork()?))),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => Ok(Self::Intel(Box::new(b.fork()?))),
        }
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.set_gate(gate, urgent),
            Self::Gemma(b) => b.set_gate(gate, urgent),
            Self::Qwen35(b) => b.set_gate(gate, urgent),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_gate(gate, urgent),
        }
    }
    fn set_device_gate(&mut self, device: usize, gate: Arc<Gate>, urgent: bool) -> Result<()> {
        match self {
            Self::Llama(b) => b.set_device_gate(device, gate, urgent),
            Self::CpuLlama(_) => anyhow::bail!("the CPU backend has no device ordinal {device}; use set_gate"),
            Self::Gemma(b) => b.set_device_gate(device, gate, urgent),
            Self::Qwen35(b) => b.set_device_gate(device, gate, urgent),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_device_gate(device, gate, urgent),
        }
    }
    fn device_health(&self) -> Option<DeviceHealth> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.device_health(),
            Self::Gemma(b) => b.device_health(),
            Self::Qwen35(b) => b.device_health(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.device_health(),
        }
    }
    fn set_device_work_class(
        &mut self,
        device: usize,
        gate: Arc<Gate>,
        class: WorkClass,
    ) -> std::result::Result<(), ExecutionError> {
        match self {
            Self::Llama(b) => b.set_device_work_class(device, gate, class),
            Self::CpuLlama(_) => Err(ExecutionError::NoSuchDevice { device, devices: vec![] }),
            Self::Gemma(b) => b.set_device_work_class(device, gate, class),
            Self::Qwen35(b) => b.set_device_work_class(device, gate, class),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_device_work_class(device, gate, class),
        }
    }
    fn background_support(&self, max_forward_tokens: usize) -> BackgroundSupport {
        match self {
            Self::Llama(b) => b.background_support(max_forward_tokens),
            Self::CpuLlama(_) => BackgroundSupport::Unsupported(CPU_BACKGROUND.into()),
            Self::Gemma(b) => b.background_support(max_forward_tokens),
            Self::Qwen35(b) => b.background_support(max_forward_tokens),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.background_support(max_forward_tokens),
        }
    }
    fn set_stop_token(&mut self, stop: Option<StopToken>) -> std::result::Result<(), ExecutionError> {
        match self {
            Self::Llama(b) => b.set_stop_token(stop),
            Self::CpuLlama(_) if stop.is_some() => Err(ExecutionError::Unsupported { reason: CPU_BACKGROUND.into() }),
            Self::CpuLlama(b) => b.set_stop_token(None),
            Self::Gemma(b) => b.set_stop_token(stop),
            Self::Qwen35(b) => b.set_stop_token(stop),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.set_stop_token(stop),
        }
    }
    fn configure_background(&mut self) -> std::result::Result<(), ExecutionError> {
        match self {
            Self::Llama(b) => b.configure_background(),
            Self::CpuLlama(_) => Err(ExecutionError::Unsupported { reason: CPU_BACKGROUND.into() }),
            Self::Gemma(b) => b.configure_background(),
            Self::Qwen35(b) => b.configure_background(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.configure_background(),
        }
    }
    fn work_health(&self) -> Option<DeviceHealth> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.work_health(),
            Self::Gemma(b) => b.work_health(),
            Self::Qwen35(b) => b.work_health(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.work_health(),
        }
    }
    fn release_work(&mut self) -> std::result::Result<(), ExecutionError> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.release_work(),
            Self::Gemma(b) => b.release_work(),
            Self::Qwen35(b) => b.release_work(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.release_work(),
        }
    }
    fn device_work_bound(&self, shape: &basal_core::planner::DeviceWorkShape) -> basal_core::planner::DeviceWorkBound {
        match self {
            Self::Llama(b) => b.device_work_bound(shape),
            Self::CpuLlama(_) => basal_core::planner::DeviceWorkBound::Unknown(
                "the CPU backend bounds each call by its frozen cpu work ceiling".into(),
            ),
            Self::Gemma(b) => b.device_work_bound(shape),
            Self::Qwen35(b) => b.device_work_bound(shape),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.device_work_bound(shape),
        }
    }
    fn media_processor(&self) -> Option<Arc<basal_core::media::Processor>> {
        match self {
            Self::Gemma(b) => b.media_processor(),
            _ => None,
        }
    }
    fn host_staging(&self) -> Option<basal_core::planner::HostStaging> {
        match self {
            Self::Llama(b) | Self::CpuLlama(b) => b.host_staging(),
            Self::Gemma(b) => b.host_staging(),
            Self::Qwen35(b) => b.host_staging(),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Self::Intel(b) => b.host_staging(),
        }
    }
}
