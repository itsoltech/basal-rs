//! The typed execution plan of one model instance.
//!
//! A plan is resolved in two phases. The caller (basal-cli) first selects the versioned profile of the model and
//! acquires and validates its GEMM artifact (I/O); [`ResolvedExecutionPlan::freeze`] then evaluates the profile's
//! versioned [`DispatchRule`] against the GPUs of the instance and the requested options. The frozen plan is
//! immutable: the backend is loaded with its precision, weight format and kernels, applies its execution policy
//! before any template or state K/V exists and before execution lanes are forked, and copies it to every fork.
//! The backend receives decisions, never model names.
//!
//! Every input of a decision is part of the plan, including the diagnostic `BASAL_*` overrides, which are captured
//! once per process ([`DiagnosticOverrides::process`]). [`ResolvedExecutionPlan::identity`] is the canonical JSON of
//! the plan: equal identities select the same kernels, GEMM artifact and fallbacks.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use basal_core::architecture::Architecture;
use serde_json::{json, Map, Value};

use crate::{InferenceConfigError, InferenceMode, Kernels, Precision, Quantization, Readout};

/// Schema of [`ResolvedExecutionPlan::identity`].
pub const PLAN_SCHEMA: &str = "basal-execution-plan.2";

/// Whether fused Llama computes the last projection and MLP only at readout positions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadoutLayer {
    /// Keep the backend's historical behavior.
    #[default]
    Auto,
    /// Gather rows before the last output projection and MLP (fused CUDA/Metal Llama).
    On,
    /// Compute the full last layer, then gather normalized rows (fused CUDA/Metal Llama).
    Off,
}

impl std::str::FromStr for ReadoutLayer {
    type Err = PlanError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            _ => Err(PlanError::Option(format!("unknown readout-layer mode {value:?} (auto, on, off)"))),
        }
    }
}

impl ReadoutLayer {
    /// Validate without querying a device; returns the effective last-layer pruning policy.
    pub fn resolve(self, arch: &Architecture, kernels: Kernels) -> Result<bool, PlanError> {
        let supported = matches!(arch, Architecture::Llama(_))
            && kernels == Kernels::Fused
            && matches!(BackendKind::llama(), BackendKind::Cuda | BackendKind::Metal);
        if self != Self::Auto && !supported {
            return Err(PlanError::Option("explicit readout-layer on/off requires fused CUDA or Metal Llama".into()));
        }
        Ok(self != Self::Off)
    }
}

/// Placement requested for Gemma's per-layer token table. Absence keeps the historical device path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PleMode {
    #[default]
    Device,
    Host,
}

impl std::str::FromStr for PleMode {
    type Err = PlanError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "device" => Ok(Self::Device),
            "host" => Ok(Self::Host),
            _ => Err(PlanError::Option(format!("unknown PLE placement {value:?} (device, host)"))),
        }
    }
}

impl PleMode {
    /// CPU-only validation, before GPU discovery or GEMM resolution. The host ceiling covers every
    /// supported short bucket (8192 padded rows) and non-unified long/media chunk (at most 2048).
    pub fn resolve(self, arch: &Architecture, split: bool) -> Result<crate::PlePlacement, PlanError> {
        if self == Self::Device {
            return Ok(crate::PlePlacement::Device);
        }
        if split {
            return Err(PlanError::Option("host PLE does not support layer split".into()));
        }
        let Architecture::Gemma4(cfg) = arch else {
            return Err(PlanError::Option("host PLE requires Gemma; Llama and Qwen are unsupported".into()));
        };
        crate::gemma::GemmaBackend::host_ple_placement(cfg).map_err(|e| PlanError::Option(format!("{e:#}")))
    }
}

/// Where a model instance executes. Absence keeps this build's GPU backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackend {
    /// The GPU backend of this build (CUDA, Metal or Intel Vulkan).
    #[default]
    Gpu,
    /// The host CPU: dense BF16 Llama computed in F32 by the stock Candle padded decoder, text only.
    Cpu,
}

impl std::str::FromStr for ExecutionBackend {
    type Err = PlanError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "gpu" => Ok(Self::Gpu),
            "cpu" => Ok(Self::Cpu),
            _ => Err(PlanError::Option(format!("unknown backend {value:?} (gpu, cpu)"))),
        }
    }
}

/// A versioned set of kernel variants with fixed arithmetic. A profile names the set it was measured with; a
/// change of arithmetic is a new variant, and existing profiles keep their set until they are moved explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelSet {
    /// Reference Candle operations for text-only Qwen3.5, with sequential FP32 DeltaNet recurrence.
    Qwen35V1,
    /// CUDA half-precision prefill in chunks up to 512, merged projections, FP32 online attention (D=256, context
    /// <=2048), otherwise bounded batched FP32 attention. Qualified invariant GEMMs permit arbitrary short prefix
    /// forks and shared projections; uncovered shapes use separate Candle GEMMs. FP32/Metal retain reference shapes.
    Qwen35V2,
    /// CUDA half-precision long prefill in chunks up to 2048, bounded batched FP32 attention and GPU causal
    /// softmax. Contexts up to 2048 retain v2's 512-token chunks and online attention; FP32/Metal retain v1.
    Qwen35V3,
    /// v3 with half-input Tensor Core long attention in auto, FP32 softmax with a fused half store, and two
    /// independent recurrent value columns per warp. Accurate keeps v3 attention; recurrent arithmetic is unchanged.
    Qwen35V4,
    /// The kernels of kernels.cu, kernels.metal, gemma_ops and the cuBLASLt GEMM path of this crate as measured
    /// before the execution plans were introduced (attention hi/lo, QK1, WGMMA, fused Gemma FP16 operations).
    V1,
    /// Opt-in CUDA Llama BF16 thinking: BF16 RoPE table and q/k stores (C1/C2), an f32 interior residual with
    /// layer-end rounding (C3'), BF16 readout (C4), f32 SiLU through multiply (K3), and GEMM bias epilogues (K4-E).
    /// Not a parity mode: GEMM/FA2/norm/readout reductions, transcendentals and other backends remain deferred.
    LlamaThinkBf16V1,
}

impl KernelSet {
    /// The sets this build implements.
    pub const SUPPORTED: &'static [KernelSet] = &[
        KernelSet::V1,
        KernelSet::Qwen35V1,
        KernelSet::Qwen35V2,
        KernelSet::Qwen35V3,
        KernelSet::Qwen35V4,
        KernelSet::LlamaThinkBf16V1,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::V1 => "basal-kernels.1",
            Self::LlamaThinkBf16V1 => "basal-llama-think-bf16-pointwise.1",
            Self::Qwen35V1 => "basal-qwen35-reference.1",
            Self::Qwen35V2 => "basal-qwen35.2",
            Self::Qwen35V3 => "basal-qwen35.3",
            Self::Qwen35V4 => "basal-qwen35.4",
        }
    }
}

/// Process-wide diagnostic overrides from `BASAL_*` environment variables, read once at the first use and never
/// again. They exist for A/B measurements; a plan records the snapshot it was frozen with, so a run with an
/// override has another identity than one without it. The values keep the semantics of the earlier direct reads:
/// a variable that is not valid UTF-8 counts as absent, except for the presence checks noted below.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagnosticOverrides {
    /// `BASAL_ATT` is set (any value, also non-UTF-8): `auto` keeps the device default instead of a measured policy.
    att_present: bool,
    /// `BASAL_ATT`: attention kernel for A/B (CUDA `tc`, `wgp`, `simt`, ...; Metal `sdpa`).
    att: Option<String>,
    /// `BASAL_ATT_SG`: Metal simdgroups per threadgroup of the tree attention (the result does not depend on it).
    att_sg: Option<String>,
    /// `BASAL_GEMM=cublas`: candle cuBLAS instead of cuBLASLt.
    gemm: Option<String>,
    /// `BASAL_GEMM_EQUIV`: log the equivalence groups of the batch-invariant search.
    gemm_equiv: bool,
    /// `BASAL_THINK_GEMM=heuristic`: upstream-bf16-pointwise bias GEMMs use cuBLASLt's first heuristic algorithm of
    /// each exact shape instead of run-time timing (diagnostic; other kernel sets ignore it).
    think_gemm: Option<String>,
    /// `BASAL_WGP_LOAD`: Hopper WGP K/V load layout.
    wgp_load: Option<String>,
    /// `BASAL_SILU_BLOCK`: block size of the SiLU kernel.
    silu_block: Option<String>,
    /// `BASAL_NORM=register`: residual RMSNorm keeping the row in registers.
    norm: Option<String>,
    /// `BASAL_CUDA_SYNC`: how host threads wait for the GPU.
    cuda_sync: Option<String>,
    /// `BASAL_HOST_TIMING` is set (any value): CUDA readout timings on stderr.
    host_timing: bool,
    /// Unsynchronized wall timings of CPU planning and Pico enqueue/readout phases.
    phase_timing: bool,
    /// CUDA profiler ranges around Pico prompt forwards, excluding loading and CPU planning.
    cuda_profile: bool,
    /// Experimental resident FP32 Qwen weights instead of widening BF16 weights for each projection.
    qwen_f32_resident: bool,
    /// Avoid restacking the already-packed output of Qwen DeltaNet preparation (same arithmetic).
    qwen_packed_recurrence: bool,
}

impl DiagnosticOverrides {
    /// The snapshot of this process, taken at the first call.
    pub fn process() -> &'static Self {
        static SNAPSHOT: std::sync::OnceLock<DiagnosticOverrides> = std::sync::OnceLock::new();
        SNAPSHOT.get_or_init(Self::capture)
    }

    fn capture() -> Self {
        let text = |name: &str| std::env::var(name).ok();
        Self {
            att_present: std::env::var_os("BASAL_ATT").is_some(),
            att: text("BASAL_ATT"),
            att_sg: text("BASAL_ATT_SG"),
            gemm: text("BASAL_GEMM"),
            gemm_equiv: std::env::var("BASAL_GEMM_EQUIV").is_ok(),
            think_gemm: text("BASAL_THINK_GEMM"),
            wgp_load: text("BASAL_WGP_LOAD"),
            silu_block: text("BASAL_SILU_BLOCK"),
            norm: text("BASAL_NORM"),
            cuda_sync: text("BASAL_CUDA_SYNC"),
            host_timing: std::env::var_os("BASAL_HOST_TIMING").is_some(),
            phase_timing: std::env::var_os("BASAL_PHASE_TIMING").is_some(),
            cuda_profile: std::env::var_os("BASAL_CUDA_PROFILE").is_some(),
            qwen_f32_resident: std::env::var_os("BASAL_QWEN_F32_RESIDENT").is_some(),
            qwen_packed_recurrence: std::env::var_os("BASAL_QWEN_PACKED_RECURRENCE").is_some(),
        }
    }

    /// `BASAL_ATT` is set, whatever its value.
    pub fn att_present(&self) -> bool {
        self.att_present
    }
    /// The value of `BASAL_ATT`.
    pub fn att(&self) -> Option<&str> {
        self.att.as_deref()
    }
    pub fn att_sg(&self) -> Option<&str> {
        self.att_sg.as_deref()
    }
    pub fn gemm(&self) -> Option<&str> {
        self.gemm.as_deref()
    }
    pub fn gemm_equiv(&self) -> bool {
        self.gemm_equiv
    }
    pub fn think_gemm(&self) -> Option<&str> {
        self.think_gemm.as_deref()
    }
    pub fn wgp_load(&self) -> Option<&str> {
        self.wgp_load.as_deref()
    }
    pub fn silu_block(&self) -> Option<&str> {
        self.silu_block.as_deref()
    }
    pub fn norm(&self) -> Option<&str> {
        self.norm.as_deref()
    }
    pub fn cuda_sync(&self) -> Option<&str> {
        self.cuda_sync.as_deref()
    }
    pub fn host_timing(&self) -> bool {
        self.host_timing
    }
    pub fn phase_timing(&self) -> bool {
        self.phase_timing
    }
    pub fn cuda_profile(&self) -> bool {
        self.cuda_profile
    }
    pub fn qwen_f32_resident(&self) -> bool {
        self.qwen_f32_resident
    }
    pub fn qwen_packed_recurrence(&self) -> bool {
        self.qwen_packed_recurrence
    }

    /// The overrides that are set, by variable name.
    fn identity(&self) -> Value {
        let mut m = Map::new();
        let mut put = |name: &str, v: Option<&str>| {
            if let Some(v) = v {
                m.insert(name.into(), Value::from(v));
            }
        };
        put("BASAL_ATT", self.att.as_deref().or(self.att_present.then_some("<not UTF-8>")));
        put("BASAL_ATT_SG", self.att_sg.as_deref());
        put("BASAL_GEMM", self.gemm.as_deref());
        put("BASAL_GEMM_EQUIV", self.gemm_equiv.then_some("set"));
        put("BASAL_THINK_GEMM", self.think_gemm.as_deref());
        put("BASAL_WGP_LOAD", self.wgp_load.as_deref());
        put("BASAL_SILU_BLOCK", self.silu_block.as_deref());
        put("BASAL_NORM", self.norm.as_deref());
        put("BASAL_CUDA_SYNC", self.cuda_sync.as_deref());
        put("BASAL_HOST_TIMING", self.host_timing.then_some("set"));
        put("BASAL_PHASE_TIMING", self.phase_timing.then_some("set"));
        put("BASAL_CUDA_PROFILE", self.cuda_profile.then_some("set"));
        put("BASAL_QWEN_F32_RESIDENT", self.qwen_f32_resident.then_some("set"));
        put("BASAL_QWEN_PACKED_RECURRENCE", self.qwen_packed_recurrence.then_some("set"));
        Value::Object(m)
    }
}

/// The opt-in strict deterministic profile `llama-candle-f32.1` of a frozen plan: single-device CUDA Llama, Candle
/// F32 kernels and readout, the cuBLAS handle of the stage configured before publication, one request per engine
/// call. Absent keeps every plan identity byte unchanged. The profile fixes these controls only; whether cuBLAS
/// and the GPU then repeat results bit for bit is not established by this crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterministicProfile {
    /// `CUBLAS_WORKSPACE_CONFIG` as the operator set it, read once before any device and compared again when the
    /// strict handle is configured; never written by this process.
    workspace_config: String,
}

impl DeterministicProfile {
    pub const ID: &'static str = "llama-candle-f32.1";

    /// Read-only process checks before any device: a CUDA Llama build, `CUBLAS_WORKSPACE_CONFIG` present as
    /// `(:<KiB>:<count>)+` of positive decimal integers (no sign, no whitespace), no arithmetic `BASAL_*`
    /// override (allowed: `BASAL_CUDA_SYNC`, `BASAL_HOST_TIMING`, `BASAL_PHASE_TIMING`, `BASAL_CUDA_PROFILE`,
    /// `BASAL_GEMM_EQUIV`) and Candle's F32 reduced-precision GEMM flag off. The syntax check validates the
    /// configuration only; it says nothing about how cuBLAS uses the value.
    pub fn from_process() -> Result<Self, PlanError> {
        let refuse = |what: String| PlanError::Option(format!("deterministic profile {}: {what}", Self::ID));
        if BackendKind::llama() != BackendKind::Cuda {
            return Err(refuse(format!(
                "needs a CUDA build, this build runs Llama on {}",
                BackendKind::llama().as_str()
            )));
        }
        let workspace_config = match std::env::var("CUBLAS_WORKSPACE_CONFIG") {
            Ok(v) if valid_workspace_config(&v) => v,
            Ok(v) => {
                return Err(refuse(format!(
                    "CUBLAS_WORKSPACE_CONFIG={v:?} is not (:<KiB>:<count>)+ with positive decimal integers"
                )))
            }
            Err(std::env::VarError::NotPresent) => return Err(refuse(
                "set CUBLAS_WORKSPACE_CONFIG (for example :4096:8) before starting basal; the process never sets it"
                    .into(),
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(refuse("CUBLAS_WORKSPACE_CONFIG is not UTF-8".into()))
            }
        };
        if let Some(name) = arithmetic_override(DiagnosticOverrides::process()) {
            return Err(refuse(format!("the diagnostic override {name} changes kernels or arithmetic; unset it")));
        }
        if candle_core::cuda::gemm_reduced_precision_f32() {
            return Err(refuse("Candle's F32 reduced-precision GEMM is enabled".into()));
        }
        Ok(Self { workspace_config })
    }

    /// `CUBLAS_WORKSPACE_CONFIG` frozen by [`DeterministicProfile::from_process`].
    pub fn workspace_config(&self) -> &str {
        &self.workspace_config
    }

    /// The cuBLAS handle settings the profile applies or checks (plan identity and backend description).
    pub(crate) fn cublas_identity(&self) -> Value {
        json!({"workspace_config": self.workspace_config, "atomics": "not_allowed", "math_mode": "default",
            "f32_compute": "CUBLAS_COMPUTE_32F"})
    }
}

/// `(:<KiB>:<count>)+` with positive decimal integers.
fn valid_workspace_config(v: &str) -> bool {
    let Some(rest) = v.strip_prefix(':') else { return false };
    let parts: Vec<&str> = rest.split(':').collect();
    parts.len().is_multiple_of(2)
        && parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u64>().is_ok_and(|n| n > 0))
}

/// The first set `BASAL_*` override that selects kernels or changes arithmetic.
fn arithmetic_override(o: &DiagnosticOverrides) -> Option<&'static str> {
    [
        ("BASAL_ATT", o.att_present),
        ("BASAL_ATT_SG", o.att_sg.is_some()),
        ("BASAL_GEMM", o.gemm.is_some()),
        ("BASAL_THINK_GEMM", o.think_gemm.is_some()),
        ("BASAL_WGP_LOAD", o.wgp_load.is_some()),
        ("BASAL_SILU_BLOCK", o.silu_block.is_some()),
        ("BASAL_NORM", o.norm.is_some()),
        ("BASAL_QWEN_F32_RESIDENT", o.qwen_f32_resident),
        ("BASAL_QWEN_PACKED_RECURRENCE", o.qwen_packed_recurrence),
    ]
    .into_iter()
    .find_map(|(name, set)| set.then_some(name))
}

/// The GPU backend of this build that runs a model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cuda,
    Metal,
    Intel,
    /// No GPU backend in this build.
    None,
    /// The host CPU of an explicit `backend: cpu` instance; never returned by [`BackendKind::llama`] or
    /// [`BackendKind::gemma`].
    Cpu,
}

impl BackendKind {
    /// The backend of Llama models in this build (the Intel Vulkan backend only without CUDA, never on macOS).
    pub const fn llama() -> Self {
        if cfg!(feature = "cuda") {
            Self::Cuda
        } else if cfg!(target_os = "macos") {
            Self::Metal
        } else if cfg!(feature = "intel") {
            Self::Intel
        } else {
            Self::None
        }
    }

    /// The backend of Gemma (Omni) models: CUDA only.
    pub const fn gemma() -> Self {
        if cfg!(feature = "cuda") {
            Self::Cuda
        } else {
            Self::None
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Intel => "intel",
            Self::None => "none",
            Self::Cpu => "cpu",
        }
    }
}

/// The GPUs of one model instance, in stage order, as the driver reports them (CUDA builds; no context is
/// created). They are part of the plan: every device-dependent choice the plan makes or leaves to a versioned
/// device rule ([`DeviceDefaultRule`]) reads them, so plans of different GPU types have different identities.
pub struct InstanceGpus {
    devices: Vec<usize>,
    /// None outside CUDA builds (Metal, Intel): no rule reads a GPU there.
    identities: Vec<Option<crate::GpuIdentity>>,
}

impl InstanceGpus {
    /// The GPUs `devices` (one per stage), queried now on CUDA builds.
    fn query(devices: &[usize]) -> Result<Self, PlanError> {
        let identities = if cfg!(feature = "cuda") {
            devices
                .iter()
                .map(|&d| crate::gpu_identity(d).map(Some).map_err(|e| PlanError::Device(format!("{e:#}"))))
                .collect::<Result<_, _>>()?
        } else {
            vec![None; devices.len()]
        };
        Ok(Self { devices: devices.to_vec(), identities })
    }

    /// Name of the GPU of stage `stage` (CUDA only).
    fn name(&self, stage: usize) -> Option<&str> {
        self.identities.get(stage)?.as_ref().map(|g| g.name.as_str())
    }

    fn stages(&self) -> usize {
        self.devices.len()
    }

    fn identity(&self) -> Value {
        Value::Array(
            self.devices
                .iter()
                .zip(&self.identities)
                .map(|(d, g)| match g {
                    None => json!({"device": d, "gpu": Value::Null}),
                    Some(g) => json!({"device": d, "gpu": g.name,
                        "compute_capability": format!("{}.{}", g.compute_capability.0, g.compute_capability.1),
                        "multiprocessors": g.multiprocessors}),
                })
                .collect(),
        )
    }
}

/// A versioned rule for the CUDA attention kernel a Llama plan with [`LlamaAttention::DeviceDefault`] uses on a
/// stage, evaluated when the plan is frozen from the stage's GPU and this build. A new default is a new variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceDefaultRule {
    /// `attn_tree_wgp` (WGMMA) on a GPU of compute capability 9.0 when the build has the compute_90a PTX, else
    /// `attn_tree_tc` (hi/lo tensor cores). `BASAL_ATT` still selects another kernel for A/B, and a bf16 forward
    /// keeps `attn_tree_tc` unless `BASAL_ATT` asks for a Hopper kernel (fused `attention_tree`).
    V1,
}

impl DeviceDefaultRule {
    pub fn id(self) -> &'static str {
        match self {
            Self::V1 => "attention-default.1",
        }
    }

    /// The kernel of this rule for the GPU of each of `devices` (one per stage), queried now: what a plan with
    /// [`LlamaAttention::DeviceDefault`] fixes for those GPUs. Empty when this build runs Llama on another backend
    /// than CUDA, as in a plan's [`ExecutionPolicy::Llama`].
    pub(crate) fn stage_kernels(self, devices: &[usize]) -> Result<Vec<&'static str>, PlanError> {
        Ok(self.kernels_of(&InstanceGpus::query(devices)?))
    }

    fn kernels_of(self, gpus: &InstanceGpus) -> Vec<&'static str> {
        if BackendKind::llama() == BackendKind::Cuda {
            gpus.identities.iter().map(|g| self.kernel(g.as_ref())).collect()
        } else {
            Vec::new()
        }
    }

    fn kernel(self, gpu: Option<&crate::GpuIdentity>) -> &'static str {
        match self {
            Self::V1 => {
                #[cfg(feature = "cuda")]
                let wgmma = crate::fused::cu::PTX_90A.is_some();
                #[cfg(not(feature = "cuda"))]
                let wgmma = false;
                if wgmma && gpu.is_some_and(|g| g.compute_capability == (9, 0)) {
                    "attn_tree_wgp"
                } else {
                    "attn_tree_tc"
                }
            }
        }
    }
}

/// A versioned dispatch rule: from the requested options and the instance's GPUs to an [`ExecutionPolicy`] and the
/// precision of `dtype: auto`. The GPU lists are data of the model profile; the rule itself knows no model names.
/// A new rule is a new variant; profiles keep the version they were measured with.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum DispatchRule {
    /// Reference Qwen3.5: separate complete prompts and FP32 recurrent state, no tuned GEMM policies.
    Qwen35V1,
    /// Qwen3.5 v2: same precision defaults and FP32 attention, with optional qualified invariant cuBLASLt projections.
    Qwen35V2,
    /// Qwen3.5 v3: v2 options, selecting the long-prefill CUDA kernel set explicitly.
    Qwen35V3,
    /// Qwen3.5 v4: v3 chunking, Tensor Core long attention in auto, unchanged FP32 recurrence arithmetic.
    Qwen35V4,
    /// Llama v1. `auto`: f16. Attention for every stage alike (a layer split computes as on one GPU):
    /// `accurate` the portable hi/lo kernel; `balanced` the QK1 specialization (CUDA f16, fused kernels, f32
    /// readout, else refused); `auto` QK1 when that is available, `BASAL_ATT` is not set and the GPU of every stage
    /// is one of `balanced_auto_gpus`, otherwise the device default of [`DeviceDefaultRule::V1`] per stage.
    LlamaV1 { balanced_auto_gpus: &'static [&'static str] },
    /// Gemma (Omni) v1. `auto`: f16 when the GPU of the first stage is one of `f16_decoder_auto_gpus`, else bf16.
    /// `balanced` is refused. Per stage, with that stage's GPU: `auto` uses FP16 attention products on a CUDA f16
    /// forward on `f16_attention_auto_gpus` and the fused FP16 operations on an f16 forward with f32 readout on
    /// `fused_ops_auto_gpus`; otherwise, and with `accurate`, FP32 products and the reference operations.
    GemmaV1 {
        f16_decoder_auto_gpus: &'static [&'static str],
        f16_attention_auto_gpus: &'static [&'static str],
        fused_ops_auto_gpus: &'static [&'static str],
    },
}

impl DispatchRule {
    pub fn id(&self) -> &'static str {
        match self {
            Self::LlamaV1 { .. } => "llama-dispatch.1",
            Self::GemmaV1 { .. } => "gemma-dispatch.1",
            Self::Qwen35V1 => "qwen35-dispatch.1",
            Self::Qwen35V2 => "qwen35-dispatch.2",
            Self::Qwen35V3 => "qwen35-dispatch.3",
            Self::Qwen35V4 => "qwen35-dispatch.4",
        }
    }

    /// The precision of `dtype` (`auto`, f16, bf16, f32) for `arch`; `first_gpu` is asked only when the rule needs
    /// the GPU (`auto` of a Gemma model).
    pub fn precision(
        &self,
        dtype: &str,
        arch: &Architecture,
        first_gpu: impl FnOnce() -> Option<String>,
    ) -> Result<Precision, PlanError> {
        self.check_family(arch)?;
        if dtype != "auto" {
            return Precision::parse(dtype).map_err(|e| PlanError::Option(format!("{e:#}")));
        }
        Ok(match self {
            Self::Qwen35V1 | Self::Qwen35V2 | Self::Qwen35V3 | Self::Qwen35V4 => {
                if cfg!(target_os = "macos") {
                    Precision::F32
                } else {
                    Precision::Bf16
                }
            }
            Self::LlamaV1 { .. } => Precision::F16,
            Self::GemmaV1 { f16_decoder_auto_gpus, .. } => {
                if first_gpu().is_some_and(|name| f16_decoder_auto_gpus.contains(&name.as_str())) {
                    Precision::F16
                } else {
                    Precision::Bf16
                }
            }
        })
    }

    fn check_family(&self, arch: &Architecture) -> Result<(), PlanError> {
        match (self, arch) {
            (Self::Qwen35V1 | Self::Qwen35V2 | Self::Qwen35V3 | Self::Qwen35V4, Architecture::Qwen35(_)) => Ok(()),
            (Self::LlamaV1 { .. }, Architecture::Llama(_)) | (Self::GemmaV1 { .. }, Architecture::Gemma4(_)) => Ok(()),
            _ => Err(PlanError::Family(self.id())),
        }
    }

    fn policy(&self, o: &PlanOptions, gpus: &InstanceGpus) -> Result<ExecutionPolicy, PlanError> {
        let overrides = DiagnosticOverrides::process();
        match self {
            Self::Qwen35V1 | Self::Qwen35V2 | Self::Qwen35V3 | Self::Qwen35V4 => {
                if o.mode == InferenceMode::Balanced {
                    return Err(PlanError::Inference(InferenceConfigError::UnsupportedBalanced));
                }
                if o.quantization != Quantization::None || gpus.stages() != 1 {
                    return Err(PlanError::Option("Qwen3.5 requires dense weights on a single device".into()));
                }
                Ok(ExecutionPolicy::Qwen35 {
                    tensor_attention: matches!(self, Self::Qwen35V4)
                        && o.mode == InferenceMode::Auto
                        && o.precision != Precision::F32
                        && BackendKind::llama() == BackendKind::Cuda,
                })
            }
            Self::LlamaV1 { balanced_auto_gpus } => {
                let backend = BackendKind::llama();
                let capable = backend == BackendKind::Cuda
                    && o.precision == Precision::F16
                    && o.kernels == Kernels::Fused
                    && o.readout == Readout::F32;
                let attention = match o.mode {
                    InferenceMode::Auto
                        if capable
                            && !overrides.att_present()
                            && !balanced_auto_gpus.is_empty()
                            && (0..gpus.stages())
                                .all(|s| gpus.name(s).is_some_and(|name| balanced_auto_gpus.contains(&name))) =>
                    {
                        LlamaAttention::BalancedQk1
                    }
                    InferenceMode::Auto => LlamaAttention::DeviceDefault,
                    InferenceMode::Accurate => LlamaAttention::PortableHiLo,
                    InferenceMode::Balanced if capable => LlamaAttention::BalancedQk1,
                    InferenceMode::Balanced => {
                        return Err(PlanError::Inference(InferenceConfigError::UnsupportedBalanced))
                    }
                };
                let device_default = DeviceDefaultRule::V1;
                let default_kernels = device_default.kernels_of(gpus);
                Ok(ExecutionPolicy::Llama { backend, attention, device_default, default_kernels })
            }
            Self::GemmaV1 { f16_attention_auto_gpus, fused_ops_auto_gpus, .. } => {
                if o.mode == InferenceMode::Balanced {
                    return Err(PlanError::Inference(InferenceConfigError::UnsupportedBalanced));
                }
                let cuda = BackendKind::gemma() == BackendKind::Cuda;
                let auto = o.mode == InferenceMode::Auto && cuda && o.precision == Precision::F16;
                let mut stages = Vec::with_capacity(gpus.stages());
                for s in 0..gpus.stages() {
                    let attention_products = if auto
                        && !f16_attention_auto_gpus.is_empty()
                        && gpus.name(s).is_some_and(|name| f16_attention_auto_gpus.contains(&name))
                    {
                        AttentionProducts::F16
                    } else {
                        AttentionProducts::F32
                    };
                    let ops = if auto
                        && o.readout == Readout::F32
                        && !fused_ops_auto_gpus.is_empty()
                        && gpus.name(s).is_some_and(|name| fused_ops_auto_gpus.contains(&name))
                    {
                        GemmaOps::FusedF16
                    } else {
                        GemmaOps::Reference
                    };
                    stages.push(GemmaStagePolicy { attention_products, ops });
                }
                Ok(ExecutionPolicy::Gemma { stages })
            }
        }
    }
}

/// Attention of a Llama model, the same on every stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlamaAttention {
    /// The kernel the plan's [`DeviceDefaultRule`] fixed for each stage's GPU, f32 SIMT for f32 forwards
    /// (`BASAL_ATT` may select another one for A/B).
    DeviceDefault,
    /// The portable hi/lo tensor-core kernel `attn_tree_tc`, also on Hopper.
    PortableHiLo,
    /// The QK1 specialization `attn_tree_tc_qk1_static`: f16 Q·K, hi/lo probabilities, f32 softmax and accumulation.
    BalancedQk1,
}

impl LlamaAttention {
    /// The CUDA kernel this policy fixes; `None` keeps the device default.
    pub(crate) fn kernel_override(self) -> Option<&'static str> {
        match self {
            Self::DeviceDefault => None,
            Self::PortableHiLo => Some("attn_tree_tc"),
            Self::BalancedQk1 => Some("attn_tree_tc_qk1_static"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DeviceDefault => "device_default",
            Self::PortableHiLo => "portable_hilo",
            Self::BalancedQk1 => "balanced_qk1",
        }
    }
}

/// dtype of the Gemma attention products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionProducts {
    F32,
    F16,
}

/// Operations of a Gemma decoder stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmaOps {
    /// Separate reference operations, the attention of every prompt on its own.
    Reference,
    /// Direct RMSNorm, fused RoPE, GELU·up and layer scale in FP16 and batched attention matrices (CUDA f16).
    FusedF16,
}

/// The policy of one Gemma stage, decided with that stage's GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmaStagePolicy {
    pub attention_products: AttentionProducts,
    pub ops: GemmaOps,
}

/// The execution policy of a model family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Qwen3.5 with fresh recurrent state; optional half-input Tensor Core products for long-context attention.
    Qwen35 { tensor_attention: bool },
    /// Llama: one attention choice for every stage; `default_kernels` is the kernel of `device_default` for the GPU
    /// of each stage (CUDA; empty for other backends), used by [`LlamaAttention::DeviceDefault`].
    Llama {
        backend: BackendKind,
        attention: LlamaAttention,
        device_default: DeviceDefaultRule,
        default_kernels: Vec<&'static str>,
    },
    /// Gemma: one policy per stage, in stage order.
    Gemma { stages: Vec<GemmaStagePolicy> },
}

impl ExecutionPolicy {
    fn identity(&self) -> Value {
        match self {
            Self::Qwen35 { tensor_attention } => {
                json!({"family":"qwen3_5_text", "execution":"immutable_prefix_forks", "recurrent_state":"f32", "attention_products":if *tensor_attention { "half_input_f32_accumulation_long_context" } else { "f32" }})
            }
            Self::Llama { backend, attention, device_default, default_kernels } => json!({
                "family": "llama", "backend": backend.as_str(), "attention": attention.as_str(),
                "device_default": {"rule": device_default.id(), "kernels": default_kernels},
            }),
            Self::Gemma { stages } => json!({"family": "gemma", "stages": stages.iter().map(|s| json!({
                "attention_products": match s.attention_products { AttentionProducts::F32 => "f32", AttentionProducts::F16 => "f16" },
                "ops": match s.ops { GemmaOps::Reference => "reference", GemmaOps::FusedF16 => "fused_f16" },
            })).collect::<Vec<_>>()}),
        }
    }
}

/// Where a GEMM artifact comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmProvenance {
    /// A table compiled into this build and pinned by the model profile.
    Bundled,
    /// A table pinned by its digest (`gemm_table: sha256:...`), from the build or the local artifact store.
    Pinned,
    /// A table file given by path.
    Explicit,
    /// Made or imported on this machine (`gemm_table: auto`): a search or an earlier cache with local timings.
    AdaptiveLocal,
}

impl GemmProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bundled => "bundled",
            Self::Pinned => "pinned",
            Self::Explicit => "explicit",
            Self::AdaptiveLocal => "adaptive/local",
        }
    }
}

/// A validated GEMM table: the parsed table that was validated (the backend loads exactly this snapshot, never
/// the file again), its file in the immutable artifact store if it could be stored, the SHA-256 of its bytes, the
/// digest of its normalized runtime content and where it comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemmArtifact {
    path: Option<PathBuf>,
    bytes_sha256: [u8; 32],
    digest: String,
    provenance: GemmProvenance,
    table: Arc<Value>,
}

impl GemmArtifact {
    /// `digest`: lowercase hex SHA-256 of the normalized `table` (basal-cli `gemm_artifact`); `path`: the store
    /// file of the table, None when the store could not be written (the snapshot then exists only in memory);
    /// `bytes_sha256`: SHA-256 of the bytes of that store file, or without one of the bytes `table` was parsed from.
    pub fn new(
        path: Option<PathBuf>,
        bytes_sha256: [u8; 32],
        digest: String,
        provenance: GemmProvenance,
        table: Arc<Value>,
    ) -> Self {
        Self { path, bytes_sha256, digest, provenance, table }
    }
    /// The store file of the table; None when it was not stored.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
    /// SHA-256 of the table's bytes, captured when it was read or stored (never read again from a path).
    pub fn bytes_sha256(&self) -> &[u8; 32] {
        &self.bytes_sha256
    }
    /// The validated table.
    pub fn table(&self) -> &Value {
        &self.table
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn provenance(&self) -> GemmProvenance {
        self.provenance
    }
    /// `provenance:sha256-prefix`, for messages.
    pub fn label(&self) -> String {
        format!("{}:{}", self.provenance.as_str(), &self.digest[..self.digest.len().min(16)])
    }
}

/// What a GEMM shape without a table entry runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmFallback {
    /// Llama: the fastest heuristic cuBLASLt algorithm, timed at first use (may depend on the batch).
    RuntimeTiming,
    /// Gemma: one fixed GEMM per prompt (independent of the batch).
    FixedPerPrompt,
    /// A reproducible Llama plan: a GEMM without a pinned algorithm is an error, never timed or searched.
    Refused,
    /// No cuBLASLt tables apply: an f32 forward (reference matmul), Metal (MLX GEMM) or Intel (WGSL GEMM).
    NotApplicable,
}

/// GEMM part of a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemmPolicy {
    pub table: Option<GemmArtifact>,
    pub fallback: GemmFallback,
}

/// How strictly the plan's artifacts are fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reproducibility {
    /// Default: local caches, searches and run-time GEMM timing are allowed; the plan identity names what was used.
    Adaptive,
    /// Every GEMM artifact is pinned by digest (profile or configuration) and covers the model's weight shapes in
    /// the plan's dtype; a missing one is an error, never a search, and a Llama GEMM without a pinned algorithm is
    /// an error, never timed ([`GemmFallback::Refused`]).
    Pinned,
}

/// The requested options of a plan, before the dispatch rule.
#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub mode: InferenceMode,
    pub precision: Precision,
    pub readout: Readout,
    pub kernels: Kernels,
    pub quantization: Quantization,
    pub reproducibility: Reproducibility,
    pub ple: PleMode,
}

/// Why a plan cannot be frozen.
#[derive(Debug)]
pub enum PlanError {
    /// The dispatch rule is for another model family.
    Family(&'static str),
    /// The profile's kernel set is not implemented by this build.
    KernelSet(&'static str),
    /// An option value is invalid.
    Option(String),
    /// The inference mode is not available for this backend.
    Inference(InferenceConfigError),
    /// A pinned plan lacks a pinned GEMM artifact.
    Unpinned(String),
    /// A GPU of the instance cannot be queried.
    Device(String),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Family(rule) => write!(f, "dispatch rule {rule} does not match the model architecture"),
            Self::KernelSet(id) => write!(f, "kernel set {id} is not implemented by this build"),
            Self::Option(e) => f.write_str(e),
            Self::Inference(e) => write!(f, "{e}"),
            Self::Unpinned(e) => write!(f, "reproducible plan: {e}"),
            Self::Device(e) => write!(f, "GPU of the plan: {e}"),
        }
    }
}

impl std::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inference(e) => Some(e),
            _ => None,
        }
    }
}

/// The frozen plan of one model instance. Invariants: the policy matches the family and stage count of the
/// instance it was frozen for; with [`Reproducibility::Pinned`] the GEMM table is bundled or pinned by digest and
/// a Llama GEMM fallback is [`GemmFallback::Refused`].
#[derive(Clone, Debug)]
pub struct ResolvedExecutionPlan {
    profile: String,
    kernel_set: KernelSet,
    rule: &'static str,
    options: PlanOptions,
    policy: ExecutionPolicy,
    gemm: GemmPolicy,
    overrides: DiagnosticOverrides,
    gpus: Value,
    ple: crate::PlePlacement,
    prune_readout: bool,
    /// Some only for a plan frozen by [`ResolvedExecutionPlan::freeze_cpu`].
    cpu: Option<CpuPlan>,
    /// Some only after [`ResolvedExecutionPlan::with_deterministic`].
    deterministic: Option<DeterministicProfile>,
}

/// The decision inputs of a CPU instance: the per-forward host working-set ceiling (it selects the sub-batch
/// grouping, so GEMM shapes) and the GEMM thread count Candle uses on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CpuPlan {
    work_bytes: std::num::NonZeroU64,
    gemm_threads: usize,
}

impl ResolvedExecutionPlan {
    /// Phase two: evaluate `rule` of profile `profile` for the instance on `devices` (one per stage) and freeze the
    /// result with the already acquired GEMM `table`; `tables_apply` is false where no cuBLASLt table is used (f32,
    /// Metal, Intel). The plan must exist before the weights are loaded.
    #[expect(clippy::too_many_arguments, reason = "the profile's parts, the instance and its acquired artifact")]
    pub fn freeze(
        profile: &str,
        kernel_set: KernelSet,
        rule: &DispatchRule,
        arch: &Architecture,
        devices: &[usize],
        options: PlanOptions,
        table: Option<GemmArtifact>,
        tables_apply: bool,
    ) -> Result<Self, PlanError> {
        Self::freeze_with_readout_layer(
            profile,
            kernel_set,
            rule,
            arch,
            devices,
            options,
            table,
            tables_apply,
            ReadoutLayer::Auto,
        )
    }

    /// Freeze a plan with explicit last-layer row selection. `Auto` is identical to [`Self::freeze`].
    #[expect(clippy::too_many_arguments, reason = "the existing freeze inputs and an opt-in row-selection policy")]
    pub fn freeze_with_readout_layer(
        profile: &str,
        kernel_set: KernelSet,
        rule: &DispatchRule,
        arch: &Architecture,
        devices: &[usize],
        options: PlanOptions,
        table: Option<GemmArtifact>,
        tables_apply: bool,
        readout_layer: ReadoutLayer,
    ) -> Result<Self, PlanError> {
        let prune_readout = readout_layer.resolve(arch, options.kernels)?;
        let ple = options.ple.resolve(arch, devices.len() != 1)?;
        if options.ple == PleMode::Host && options.mode == InferenceMode::Balanced {
            return Err(PlanError::Inference(InferenceConfigError::UnsupportedBalanced));
        }
        if !KernelSet::SUPPORTED.contains(&kernel_set) {
            return Err(PlanError::KernelSet(kernel_set.id()));
        }
        if kernel_set == KernelSet::LlamaThinkBf16V1 {
            if !matches!((arch, rule), (Architecture::Llama(_), DispatchRule::LlamaV1 { .. }))
                || BackendKind::llama() != BackendKind::Cuda
                || devices.len() != 1
                || options.precision != Precision::Bf16
                || options.readout != Readout::Bf16Rounded
                || options.kernels != Kernels::Fused
                || options.quantization != Quantization::None
                || options.mode == InferenceMode::Balanced
                || options.reproducibility == Reproducibility::Pinned
                || !tables_apply
                || table.is_some()
            {
                return Err(PlanError::Option("upstream-bf16-pointwise requires single-device CUDA Llama, BF16, BF16-rounded readout, fused dense kernels and adaptive RuntimeTiming without a GEMM table".into()));
            }
            if DiagnosticOverrides::process().norm() == Some("register") {
                return Err(PlanError::Option("upstream-bf16-pointwise does not support BASAL_NORM=register".into()));
            }
            if DiagnosticOverrides::process().gemm() == Some("cublas") {
                return Err(PlanError::Option(
                    "upstream-bf16-pointwise requires cuBLASLt bias epilogues; BASAL_GEMM=cublas is unsupported".into(),
                ));
            }
            if !matches!(DiagnosticOverrides::process().think_gemm(), None | Some("heuristic")) {
                return Err(PlanError::Option("BASAL_THINK_GEMM accepts only `heuristic`".into()));
            }
        }
        rule.check_family(arch)?;
        if matches!(arch, Architecture::Qwen35(_))
            != matches!(
                kernel_set,
                KernelSet::Qwen35V1 | KernelSet::Qwen35V2 | KernelSet::Qwen35V3 | KernelSet::Qwen35V4
            )
        {
            return Err(PlanError::Option("kernel set does not match model architecture".into()));
        }
        if matches!(arch, Architecture::Qwen35(_))
            && !matches!(
                (rule, kernel_set),
                (DispatchRule::Qwen35V1, KernelSet::Qwen35V1)
                    | (DispatchRule::Qwen35V2, KernelSet::Qwen35V2)
                    | (DispatchRule::Qwen35V3, KernelSet::Qwen35V3)
                    | (DispatchRule::Qwen35V4, KernelSet::Qwen35V4)
            )
        {
            return Err(PlanError::Option("Qwen3.5 dispatch and kernel set versions must match".into()));
        }
        let pinned = options.reproducibility == Reproducibility::Pinned;
        let fallback = match arch {
            _ if !tables_apply => GemmFallback::NotApplicable,
            Architecture::Llama(_) if pinned => GemmFallback::Refused,
            Architecture::Llama(_) => GemmFallback::RuntimeTiming,
            Architecture::Gemma4(_) => GemmFallback::FixedPerPrompt,
            Architecture::Qwen35(_) => GemmFallback::FixedPerPrompt,
        };
        if pinned && tables_apply {
            match &table {
                Some(t) if matches!(t.provenance, GemmProvenance::Bundled | GemmProvenance::Pinned) => {}
                Some(t) => {
                    return Err(PlanError::Unpinned(format!("GEMM table {} is not pinned by digest", t.label())))
                }
                None => return Err(PlanError::Unpinned("no pinned GEMM table for this GPU and model".into())),
            }
        }
        let gpus = InstanceGpus::query(devices)?;
        let policy = rule.policy(&options, &gpus)?;
        Ok(Self {
            profile: profile.to_string(),
            kernel_set,
            rule: rule.id(),
            options,
            policy,
            gemm: GemmPolicy { table, fallback },
            overrides: DiagnosticOverrides::process().clone(),
            gpus: gpus.identity(),
            ple,
            prune_readout,
            cpu: None,
            deterministic: None,
        })
    }

    /// Freeze the plan of a host CPU instance without querying any GPU: Llama with dispatch rule `llama-dispatch.1`
    /// and kernel set `basal-kernels.1` only, with the fixed CPU options: auto mode, F32 (BF16 resident), F32
    /// readout, Candle kernels, no quantization, adaptive, device PLE, no GEMM table, readout-layer auto.
    /// `work_bytes` is the host working-set ceiling of one CPU forward.
    pub fn freeze_cpu(
        profile: &str,
        kernel_set: KernelSet,
        rule: &DispatchRule,
        arch: &Architecture,
        work_bytes: std::num::NonZeroU64,
    ) -> Result<Self, PlanError> {
        rule.check_family(arch)?;
        if !matches!((arch, rule), (Architecture::Llama(_), DispatchRule::LlamaV1 { .. }))
            || kernel_set != KernelSet::V1
        {
            return Err(PlanError::Option("the CPU backend runs Llama with kernel set basal-kernels.1 only".into()));
        }
        let options = PlanOptions {
            mode: InferenceMode::Auto,
            precision: Precision::F32,
            readout: Readout::F32,
            kernels: Kernels::Candle,
            quantization: Quantization::None,
            reproducibility: Reproducibility::Adaptive,
            ple: PleMode::Device,
        };
        let prune_readout = ReadoutLayer::Auto.resolve(arch, options.kernels)?;
        let ple = options.ple.resolve(arch, false)?;
        Ok(Self {
            profile: profile.to_string(),
            kernel_set,
            rule: rule.id(),
            options,
            policy: ExecutionPolicy::Llama {
                backend: BackendKind::Cpu,
                attention: LlamaAttention::DeviceDefault,
                device_default: DeviceDefaultRule::V1,
                default_kernels: Vec::new(),
            },
            gemm: GemmPolicy { table: None, fallback: GemmFallback::NotApplicable },
            overrides: DiagnosticOverrides::process().clone(),
            gpus: Value::Array(Vec::new()),
            ple,
            prune_readout,
            cpu: Some(CpuPlan { work_bytes, gemm_threads: candle_core::utils::get_num_threads() }),
            deterministic: None,
        })
    }

    /// Attach the strict profile to a frozen GPU plan. Refuses every plan outside it: a CPU plan, another family
    /// or backend, more than one stage, a kernel set other than `basal-kernels.1`, a precision, readout or kernels
    /// other than F32/F32/Candle, quantization, balanced mode, host PLE, a GEMM table or a GEMM fallback.
    pub fn with_deterministic(mut self, profile: DeterministicProfile) -> Result<Self, PlanError> {
        let refuse = |what: &str| {
            Err(PlanError::Option(format!("deterministic profile {} requires {what}", DeterministicProfile::ID)))
        };
        if self.cpu.is_some() {
            return refuse("a GPU instance; the CPU backend has no strict profile");
        }
        if !matches!(&self.policy, ExecutionPolicy::Llama { backend: BackendKind::Cuda, default_kernels, .. }
            if default_kernels.len() == 1)
        {
            return refuse("single-device CUDA Llama");
        }
        if self.kernel_set != KernelSet::V1 {
            return refuse("kernel set basal-kernels.1");
        }
        let o = &self.options;
        if o.precision != Precision::F32 {
            return refuse("dtype f32");
        }
        if o.readout != Readout::F32 {
            return refuse("readout f32");
        }
        if o.kernels != Kernels::Candle {
            return refuse("kernels candle");
        }
        if o.quantization != Quantization::None {
            return refuse("quantization none");
        }
        if o.mode == InferenceMode::Balanced {
            return refuse("inference_mode auto or accurate");
        }
        if o.ple != PleMode::Device || self.ple != crate::PlePlacement::Device {
            return refuse("ple_placement device");
        }
        if self.gemm.table.is_some() || self.gemm.fallback != GemmFallback::NotApplicable {
            return refuse("no GEMM table and no timed GEMM (f32)");
        }
        self.deterministic = Some(profile);
        Ok(self)
    }

    /// The strict profile; None for every default plan.
    pub fn deterministic(&self) -> Option<&DeterministicProfile> {
        self.deterministic.as_ref()
    }

    /// The per-forward host working-set ceiling of a CPU plan; None for a GPU plan.
    pub fn cpu_work_bytes(&self) -> Option<std::num::NonZeroU64> {
        self.cpu.map(|c| c.work_bytes)
    }

    /// The Candle host GEMM thread count captured when a CPU plan was frozen; None for a GPU plan.
    pub fn cpu_gemm_threads(&self) -> Option<usize> {
        self.cpu.map(|c| c.gemm_threads)
    }

    pub fn profile(&self) -> &str {
        &self.profile
    }
    /// Actual placement and checked per-call ceiling, frozen before any weight upload.
    pub fn ple_placement(&self) -> crate::PlePlacement {
        self.ple
    }
    pub fn kernel_set(&self) -> KernelSet {
        self.kernel_set
    }
    pub fn mode(&self) -> InferenceMode {
        self.options.mode
    }
    pub fn precision(&self) -> Precision {
        self.options.precision
    }
    pub fn readout(&self) -> Readout {
        self.options.readout
    }
    pub fn kernels(&self) -> Kernels {
        self.options.kernels
    }
    /// Whether the final output projection and MLP operate only on readout rows.
    pub fn prunes_readout_layer(&self) -> bool {
        self.prune_readout
    }
    pub fn quantization(&self) -> Quantization {
        self.options.quantization
    }
    pub fn reproducibility(&self) -> Reproducibility {
        self.options.reproducibility
    }
    pub fn policy(&self) -> &ExecutionPolicy {
        &self.policy
    }
    pub fn gemm(&self) -> &GemmPolicy {
        &self.gemm
    }
    pub fn overrides(&self) -> &DiagnosticOverrides {
        &self.overrides
    }

    /// Attention products of the first stage of a Gemma plan (None for Llama).
    pub fn first_stage_attention_products(&self) -> Option<AttentionProducts> {
        match &self.policy {
            ExecutionPolicy::Gemma { stages } => stages.first().map(|s| s.attention_products),
            ExecutionPolicy::Llama { .. } => None,
            ExecutionPolicy::Qwen35 { .. } => None,
        }
    }

    /// Canonical JSON of every input of the plan's decisions (no paths: the GEMM table is named by digest).
    pub fn identity(&self) -> Value {
        let o = &self.options;
        let mut identity = json!({
            "schema": PLAN_SCHEMA,
            "profile": self.profile,
            "kernel_set": self.kernel_set.id(),
            "dispatch_rule": self.rule,
            "inference_mode": match o.mode { InferenceMode::Auto => "auto", InferenceMode::Accurate => "accurate", InferenceMode::Balanced => "balanced" },
            "precision": match o.precision { Precision::F16 => "f16", Precision::Bf16 => "bf16", Precision::F32 => "f32" },
            "readout": match o.readout { Readout::F32 => "f32", Readout::Bf16Rounded => "bf16" },
            "kernels": match o.kernels { Kernels::Fused => "fused", Kernels::Candle => "candle" },
            "quantization": o.quantization.as_str(),
            "reproducibility": match o.reproducibility { Reproducibility::Adaptive => "adaptive", Reproducibility::Pinned => "pinned" },
            "gpus": self.gpus,
            "policy": self.policy.identity(),
            "gemm": {
                "table": self.gemm.table.as_ref().map(|t| json!({"sha256": t.digest, "provenance": t.provenance.as_str()})),
                "fallback": match self.gemm.fallback { GemmFallback::RuntimeTiming => "runtime_timing", GemmFallback::FixedPerPrompt => "fixed_per_prompt", GemmFallback::Refused => "refused", GemmFallback::NotApplicable => "not_applicable" },
            },
            "overrides": self.overrides.identity(),
        });
        // Keep default plan JSON and all existing fingerprint inputs byte-for-byte unchanged.
        if let crate::PlePlacement::Host { max_staging_bytes } = self.ple {
            identity["ple"] = json!({"placement":"host", "max_staging_bytes_per_call":max_staging_bytes});
        }
        if matches!(self.policy, ExecutionPolicy::Llama { .. }) && o.kernels == Kernels::Fused && !self.prune_readout {
            identity["last_layer"] =
                json!({"schema":"basal-llama-last-layer.1", "rows":"full", "gather":"after_final_norm"});
        }
        if let Some(cpu) = self.cpu {
            identity["cpu"] = json!({"schema":"basal-cpu-execution.1", "work_bytes":cpu.work_bytes.get(),
                "gemm_threads":cpu.gemm_threads});
        }
        if let Some(d) = &self.deterministic {
            identity["deterministic"] = json!({"schema":"basal-deterministic.1", "profile":DeterministicProfile::ID,
                "request_isolation":"whole_request", "cublas":d.cublas_identity(), "cublaslt":"unused_for_f32"});
        }
        identity
    }
}
