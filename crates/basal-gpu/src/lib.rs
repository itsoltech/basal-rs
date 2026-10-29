//! Basal inference on Metal, CUDA or Intel Vulkan. The optional `intel` module implements an independent
//! FP16 Vulkan forward; the following describes the Candle-based Metal/CUDA implementation.
//!
//! Forward on Apple Metal (matmul = candle's port of the MLX steel GEMM kernels,
//! MLX attention) or, with the `cuda` feature, NVIDIA CUDA (cuBLAS GEMM, fused kernels from kernels.cu, attention from
//! candle ops). The forward code is shared; only the kernels behind the fused ops, GEMM and attention differ.
//!
//! Llama decoder with attention and MLP biases, GQA, RMSNorm and rotate-half RoPE with explicit positions, run on
//! packed rows `[prefix | order 1 | order 2]` with a block mask (each option block sees the prefix and itself).
//! Weights stay in the checkpoint precision (bf16) unless `--dtype f16` is chosen as an experiment; `--dtype f32` keeps
//! bf16 weights resident and widens each layer to f32 just before use (exact FP32 forward of the checkpoint, a
//! numerical reference that fits in memory, not a serving mode). Attention
//! (scores, mask, softmax, weighted sum) and RoPE run in f32. Only the lm_head rows of the option letters are
//! evaluated, in f32, at the readout positions.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use basal_core::execution::{BackgroundSupport, ExecutionError, Interrupted, StopToken, WorkClass};
use basal_core::gate::{Gate, GateGuard};
use basal_core::manifest::LlamaConfig;
use basal_core::pack::Packed;
use basal_core::{Backend, DeviceHealth};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
#[cfg(target_os = "macos")]
use candle_nn::ops::sdpa;
use candle_nn::ops::{rms_norm, softmax_last_dim};
use candle_nn::rotary_emb::rope;
use serde_json::{json, Value};

#[cfg(feature = "cuda")]
mod cublaslt;
mod early_exit;
mod fused;
mod gemma;
pub use gemma::PlePlacement;
mod gemma_encoders;
#[cfg(feature = "cuda")]
mod gemma_ops;
pub mod gguf;
mod inference;
#[cfg(all(feature = "intel", not(target_os = "macos")))]
pub mod intel;
pub use inference::{InferenceConfigError, InferenceMode};
mod memory;
pub mod plan;
pub use plan::{
    AttentionProducts, DeviceDefaultRule, DiagnosticOverrides, DispatchRule, ExecutionPolicy, GemmArtifact,
    GemmFallback, GemmPolicy, GemmProvenance, KernelSet, PlanError, PlanOptions, PleMode, Reproducibility,
    ResolvedExecutionPlan,
};
mod quant;
mod qwen35;
mod runtime;
mod split;
pub use memory::{gpu_memory, pool_probe, GpuMemory, GpuMemoryProbe, PoolProbe, PoolUsage};
pub use quant::{estimated_stage_weight_bytes, estimated_weight_bytes, ParseQuantizationError, Quantization};
use quant::{ProjWeight, Projection};
pub use runtime::{HostPleMemory, PlannedPlacement, RuntimeBackend};
pub use split::{LayerSplit, LayerSplitError, StageSpec};
#[cfg(target_os = "macos")]
pub mod gemm;
use fused::HeadDims;

/// GPU backend of this build (`basal --version`).
#[cfg(feature = "cuda")]
pub const BUILD_BACKEND: &str = concat!("CUDA, kernels for compute capability ", env!("BASAL_CUDA_COMPUTE_CAPS"));
#[cfg(all(not(feature = "cuda"), target_os = "macos"))]
pub const BUILD_BACKEND: &str = "Metal";
#[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
pub const BUILD_BACKEND: &str = "Intel Arc (Vulkan)";
#[cfg(all(not(feature = "intel"), not(feature = "cuda"), not(target_os = "macos")))]
pub const BUILD_BACKEND: &str = "no GPU backend";
/// This build runs the Intel Vulkan backend (`intel` on Linux without `cuda`). [`BUILD_BACKEND`] is for display only.
pub const INTEL_BACKEND: bool = cfg!(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")));

/// Apple GPU of this machine: name and the memory Metal recommends as one process's working set (bytes).
#[cfg(target_os = "macos")]
pub fn metal_info() -> Option<(String, u64)> {
    use objc2_metal::MTLDevice;
    let d = objc2_metal::MTLCreateSystemDefaultDevice()?;
    Some((d.name().to_string(), d.recommendedMaxWorkingSetSize()))
}

/// Token id written into padding columns (never attended to by real tokens). Matches the upstream pad token `</s>`.
const PAD_ID: u32 = 2;
/// Additive mask value of a blocked (query, key) pair. Finite so that a key block that is fully masked for a query in
/// a tiled attention kernel cannot produce `-inf - -inf`; every query sees at least itself.
const MASKED: f32 = -1e30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Bf16,
    F16,
    F32,
}

/// Preserve the Llama default while using the published BF16 serving dtype for a Gemma checkpoint.
pub fn model_precision(value: &str, cfg: &basal_core::architecture::Architecture) -> Result<Precision> {
    if value == "auto" {
        Ok(match cfg {
            basal_core::architecture::Architecture::Llama(_) => Precision::F16,
            basal_core::architecture::Architecture::Gemma4(_) => Precision::Bf16,
            basal_core::architecture::Architecture::Qwen35(_) => {
                if cfg!(target_os = "macos") {
                    Precision::F32
                } else {
                    Precision::Bf16
                }
            }
        })
    } else {
        Precision::parse(value)
    }
}

impl Precision {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "bf16" | "bfloat16" => Ok(Precision::Bf16),
            "f16" | "float16" => Ok(Precision::F16),
            "f32" | "float32" => Ok(Precision::F32),
            _ => bail!("unknown precision {s:?} (bf16, f16, f32)"),
        }
    }
    /// dtype of resident weights
    fn storage(self) -> DType {
        match self {
            Precision::Bf16 | Precision::F32 => DType::BF16,
            Precision::F16 => DType::F16,
        }
    }
    /// dtype of the residual stream, norms and linear layers
    fn compute(self) -> DType {
        match self {
            Precision::Bf16 => DType::BF16,
            Precision::F16 => DType::F16,
            Precision::F32 => DType::F32,
        }
    }
}

/// How the letter logits leave the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readout {
    /// f32 product of the normalised hidden state and the selected lm_head rows.
    F32,
    /// The same, rounded to bf16 on the host: what a bf16 lm_head (MLX reference) returns. Diagnostic.
    Bf16Rounded,
}

/// Kernel set of the decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernels {
    /// Fused bias/RoPE/head-split, bias+SiLU*up and bias+residual kernels (kernels.metal) and the MLX-derived SDPA
    /// kernel of candle with an additive f32 mask.
    Fused,
    /// Only stock candle ops (first port): separate bias adds, strided copies, f32 matmul-softmax-matmul attention.
    Candle,
}

impl Kernels {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "fused" => Ok(Kernels::Fused),
            "candle" => Ok(Kernels::Candle),
            _ => bail!("unknown kernel set {s:?} (fused, candle)"),
        }
    }
}

#[derive(Clone)]
struct Layer {
    ln1: Tensor,
    /// [q; k; v] projection [(nh + 2 nkv) hd, h] and its bias
    wqkv: ProjWeight,
    bqkv: Tensor,
    wo: ProjWeight,
    bo: Tensor,
    ln2: Tensor,
    /// [gate; up] projection [2 i, h] and its bias
    wgu: ProjWeight,
    bgu: Tensor,
    wdown: ProjWeight,
    bdown: Tensor,
}

/// One attention segment of a ragged forward: `len` tokens at `off` in the compact token list, optionally after a
/// precomputed prefix. Tree attention (CUDA; Metal unless `BASAL_ATT=sdpa`): the blocks of the row as attention
/// units (`units`, offsets relative to `off`). Metal SDPA: additive mask `[len, np + len]` (np = prefix length).
struct Segment<'a> {
    off: usize,
    len: usize,
    past: Option<&'a PrefixKv>,
    mask: Vec<f32>,
    #[cfg_attr(
        not(any(feature = "cuda", target_os = "macos")),
        allow(dead_code, reason = "Packed attention units are read only by GPU kernels")
    )]
    units: Vec<Unit>,
}

/// Queries `q_off..q_off + q_len` (one block of a packed row) and their key ranges: ancestor blocks from the root
/// down, then the own block. Offsets relative to the segment's first token (after its precomputed prefix).
#[derive(Clone, Debug)]
#[cfg_attr(
    not(any(feature = "cuda", target_os = "macos")),
    allow(dead_code, reason = "Packed attention ranges are read only by GPU kernels")
)]
struct Unit {
    q_off: usize,
    q_len: usize,
    ranges: Vec<(usize, usize)>,
}

/// Attention units of a packed row whose first `np` tokens come from a precomputed prefix (they lie in the root
/// block). Every block must be one contiguous range of the row.
fn row_units(p: &Packed, np: usize) -> Result<Vec<Unit>> {
    let nodes = p.parent.len();
    let mut span = vec![(usize::MAX, 0usize, 0usize); nodes]; // first, last + 1, count
    for (t, &b) in p.node.iter().enumerate() {
        let s = &mut span[b as usize];
        s.0 = s.0.min(t);
        s.1 = t + 1;
        s.2 += 1;
    }
    let range = |b: usize| -> Result<(usize, usize)> {
        let (f, e, c) = span[b];
        ensure!(c == 0 || e - f == c, "block {b} of a packed row is not contiguous");
        let f = f.max(np);
        Ok(if c == 0 || e <= f { (0, 0) } else { (f - np, e - f) })
    };
    let mut units = Vec::new();
    for b in 0..nodes {
        let own = range(b)?;
        if own.1 == 0 {
            continue;
        }
        let mut chain = Vec::new();
        let mut a = p.parent[b];
        while a >= 0 {
            chain.push(a as usize);
            a = p.parent[a as usize];
        }
        let mut ranges = Vec::with_capacity(chain.len() + 1);
        for &a in chain.iter().rev() {
            let r = range(a)?;
            if r.1 > 0 {
                ranges.push(r);
            }
        }
        ranges.push(own);
        units.push(Unit { q_off: own.0, q_len: own.1, ranges });
    }
    Ok(units)
}

impl Segment<'_> {
    fn np(&self) -> usize {
        self.past.map_or(0, |p| p.ids.len())
    }
}

/// K/V tensors of every layer.
type LayerKv = Vec<(Tensor, Tensor)>;

/// K/V of every layer for a fixed token prefix at positions 0..len (f32, `[1, nkv, len, hd]`).
fn prefix_bytes(cfg: &LlamaConfig, tokens: usize, hilo: bool) -> usize {
    tokens * cfg.num_layers * 2 * cfg.kv_heads * cfg.head_dim * 4 * if hilo { 2 } else { 1 }
}

/// K/V of a precomputed prefix for the layers of one stage, on the stage's GPU.
#[derive(Clone)]
struct StageKv {
    /// The stage's layers in order.
    kv: LayerKv,
    /// CUDA tensor-core attention: per layer the K and V split into f16 hi / lo planes (`fused::split_hilo`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code, reason = "Hi/lo KV planes are used only by CUDA attention"))]
    hl: Vec<(Tensor, Tensor)>,
}

#[derive(Clone)]
struct PrefixKv {
    ids: Vec<u32>,
    /// One part per stage, in stage order: the K/V of a layer lies on the GPU of its stage.
    parts: Vec<OnStage<StageKv>>,
    /// Template prefixes are permanent; state prefixes live in the bounded LRU.
    pinned: bool,
    last_use: u64,
}

impl PrefixKv {
    /// Device memory of all stages: f32 K/V, plus the same again for the f16 hi / lo planes when present.
    fn bytes(&self, cfg: &LlamaConfig) -> usize {
        prefix_bytes(cfg, self.ids.len(), self.parts.iter().any(|p| !p.get().hl.is_empty()))
    }

    /// The part of stage `s`.
    fn part(&self, s: usize) -> Result<&StageKv> {
        Ok(self.parts.get(s).with_context(|| format!("prefix has no part for stage {s}"))?.get())
    }
}

/// Weight shapes (N, K) of the projection GEMMs of a model: qkv, o, gate|up, down.
pub fn projection_shapes(c: &LlamaConfig) -> Vec<(usize, usize)> {
    let (h, qd, kvd, i) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim, c.intermediate);
    vec![(qd + 2 * kvd, h), (h, qd), (2 * i, h), (h, i)]
}

/// Decoder projection shapes (N, K), including Gemma's per-layer embeddings and separate Q/K/V.
pub fn architecture_projection_shapes(cfg: &basal_core::architecture::Architecture) -> Result<Vec<(usize, usize)>> {
    use basal_core::architecture::Architecture;
    if let Architecture::Qwen35(c) = cfg {
        c.validate()?;
        let kd = c.linear_key_heads * c.linear_key_dim;
        let vd = c.linear_value_heads * c.linear_value_dim;
        let shapes = vec![
            (2 * c.heads * c.head_dim, c.hidden),
            (c.kv_heads * c.head_dim, c.hidden),
            (c.hidden, c.heads * c.head_dim),
            (2 * kd + 2 * vd + 2 * c.linear_value_heads, c.hidden),
            (c.hidden, vd),
            (2 * c.intermediate, c.hidden),
            (c.hidden, c.intermediate),
        ];
        return Ok(shapes.into_iter().collect::<std::collections::BTreeSet<_>>().into_iter().collect());
    }
    let Architecture::Gemma4(c) = cfg else {
        return Ok(projection_shapes(cfg.llama()?));
    };
    c.validate()?;
    let h = c.hidden;
    let mut shapes = std::collections::BTreeSet::new();
    for layer in &c.layers {
        let q = layer.heads.checked_mul(layer.head_dim).context("Gemma query shape overflow")?;
        let kv = layer.kv_heads.checked_mul(layer.head_dim).context("Gemma KV shape overflow")?;
        shapes.extend([(q, h), (h, q), (layer.intermediate, h), (h, layer.intermediate)]);
        if !layer.shared_kv {
            shapes.insert((kv, h));
        }
    }
    if c.per_layer_input > 0 {
        let packed = c.layers.len().checked_mul(c.per_layer_input).context("Gemma per-layer shape overflow")?;
        shapes.extend([(packed, h), (c.per_layer_input, h), (h, c.per_layer_input)]);
    }
    Ok(shapes.into_iter().collect())
}

/// Name of CUDA device `device` ([`gpu_device`]), as recorded in GEMM tables.
pub fn gpu_name(device: usize) -> Result<String> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = gpu_device(device)? {
        return Ok(d.cuda_stream().context().name()?);
    }
    bail!("gpu_name: CUDA only (device {device})")
}

/// Type of a CUDA device as the driver reports it: the name GEMM tables record, compute capability and SM count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuIdentity {
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (u32, u32),
    pub multiprocessors: u32,
}

impl std::fmt::Display for GpuIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (major, minor) = self.compute_capability;
        write!(f, "{}, compute capability {major}.{minor}, {} SMs", self.name, self.multiprocessors)
    }
}

/// Type of CUDA device `device` (an ordinal visible to this process) from the driver API, without creating a context
/// or applying `BASAL_CUDA_SYNC`: `serve` compares the GPUs of a replica group before downloading or loading anything.
pub fn gpu_identity(device: usize) -> Result<GpuIdentity> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::{result, sys::CUdevice_attribute as A};
        result::init().context("initialising the CUDA driver")?;
        let count = result::device::get_count().context("counting CUDA devices")?;
        ensure!(
            device < usize::try_from(count).unwrap_or(0),
            "CUDA device {device} does not exist: this process sees {count} CUDA device(s), numbered from 0 \
             (CUDA_VISIBLE_DEVICES selects and renumbers them)"
        );
        let dev = result::device::get(i32::try_from(device)?).with_context(|| format!("CUDA device {device}"))?;
        let name = result::device::get_name(dev).with_context(|| format!("name of CUDA device {device}"))?;
        let attribute = |a: A| -> Result<u32> {
            // SAFETY: `dev` was returned by cuDeviceGet for an ordinal below the device count (checked above), as
            // get_attribute requires; the call only reads a device property and needs no context.
            let v = unsafe { result::device::get_attribute(dev, a) }
                .with_context(|| format!("{a:?} of CUDA device {device}"))?;
            u32::try_from(v).with_context(|| format!("{a:?} of CUDA device {device}: {v}"))
        };
        Ok(GpuIdentity {
            name,
            compute_capability: (
                attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            multiprocessors: attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        })
    }
    #[cfg(not(feature = "cuda"))]
    bail!("GPU identity needs the CUDA backend (device {device})")
}

/// Version of the batch-invariant GEMM search of this build (CUDA; 0 without cuBLASLt), as recorded in GEMM tables.
pub fn gemm_search_version() -> u64 {
    #[cfg(feature = "cuda")]
    return cublaslt::INVARIANT_SEARCH_VERSION;
    #[cfg(not(feature = "cuda"))]
    0
}

/// Version of the cuBLASLt library this build runs with (CUDA), as recorded in GEMM tables.
pub fn cublaslt_version() -> Option<usize> {
    #[cfg(feature = "cuda")]
    return Some(cublaslt::version());
    #[cfg(not(feature = "cuda"))]
    None
}

/// How a host thread waits for the GPU (`BASAL_CUDA_SYNC`): `auto` (driver default: spins while there are more cores
/// than GPUs, a full core busy during every wait), `spin`, `yield` or `blocking` (sleeps until the GPU signals; frees
/// the core, a wake-up per synchronisation). Set on the primary context of `device` before it is created.
#[cfg(feature = "cuda")]
fn set_cuda_sync(device: usize) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};
    use sys::CUctx_flags_enum as F;
    let flag = match crate::plan::DiagnosticOverrides::process().cuda_sync() {
        None | Some("") | Some("auto") => return Ok(()),
        Some("spin") => F::CU_CTX_SCHED_SPIN,
        Some("yield") => F::CU_CTX_SCHED_YIELD,
        Some("blocking") => F::CU_CTX_SCHED_BLOCKING_SYNC,
        Some(v) => anyhow::bail!("BASAL_CUDA_SYNC={v:?}: auto, spin, yield or blocking"),
    };
    result::init()?;
    let dev = result::device::get(i32::try_from(device)?)?;
    // SAFETY: a valid device (the caller checked the ordinal against the device count); the flags apply to its
    // primary context, retained afterwards by candle
    unsafe { sys::cuDevicePrimaryCtxSetFlags_v2(dev, flag as u32).result()? };
    Ok(())
}

/// The GPU `device` of this build: with the `cuda` feature the CUDA device of that ordinal, an index among the GPUs
/// visible to this process (the driver orders all GPUs, by default fastest first or by PCI bus with
/// `CUDA_DEVICE_ORDER=PCI_BUS_ID`; `CUDA_VISIBLE_DEVICES` then selects, reorders and renumbers them from 0), otherwise
/// the Metal device (macOS), which is only device 0. Every call creates a new candle device (its own handles and
/// kernel caches) on the GPU's primary context.
pub fn gpu_device(device: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::CudaContext;
        let count = CudaContext::device_count().context("counting CUDA devices")?;
        ensure!(
            device < usize::try_from(count).unwrap_or(0),
            "CUDA device {device} does not exist: this process sees {count} CUDA device(s), numbered from 0 \
             (CUDA_VISIBLE_DEVICES selects and renumbers them)"
        );
        set_cuda_sync(device).context("BASAL_CUDA_SYNC")?;
        let dev = Device::new_cuda(device).with_context(|| format!("CUDA device {device}"))?;
        if let Device::Cuda(d) = &dev {
            // cudarc creates two CUDA events for every allocation, which it records and waits on only when the context
            // has streams of its own (`is_in_multi_stream_mode`); candle and this crate use only the per-thread default
            // stream, so the events are created and destroyed unused, two driver calls each per tensor.
            // SAFETY: set before the first allocation of this device. Without tracking, cudarc requires that no
            // allocation is used on a stream before its allocation there completed, freed before its last use on
            // another stream completed, or written by two streams at once. Every allocation here lives on the
            // per-thread stream of the thread that made it (candle and this crate create no streams; a handle of
            // that stream resolves to the context current on the calling thread), and the engines guarantee:
            // (a) weights, the evidence head, immutable exit-head tensors, cuBLASLt workspaces and pinned prefix
            //     K/V written at start-up are read by other threads' streams only after the publication barriers
            //     that synchronise every stage (end of `load_stages`, exit-head `upload`, end of `add_prefix`);
            //     exit-head upload synchronises before any fork or lane exists, and these tensors are only read
            //     afterwards;
            // (b) a layer-split boundary copy is queued only after an `Ok` drain of its source stage and of the
            //     destination's initialisation; the destination is used and the source freed only after `Ok`
            //     synchronisations of this thread's per-thread stream in both contexts (the copy lands on one of
            //     them); boundary buffers are freed with their owner's context bound only when no copy is or may be
            //     in flight, and are otherwise retained forever (`StageCopy`);
            // (c) a stage's work, including its cuBLASLt workspace, is drained with `Ok` before its gate is released
            //     (or its last host readout drained it), so lanes sharing the workspace never overlap; an
            //     unconfirmed drain marks the lineage lost and nothing of it runs again;
            // (d) no allocation is written by two streams at once: weights and pinned prefixes are only read after
            //     publication, and every lane writes only its own temporaries and state-cache entries;
            // (e) memory of a stage is freed only with that stage's context current, on the stream of the thread that
            //     queued its uses (stream order) or after their end was confirmed: stage temporaries as locals of the
            //     stage body, the final readout in the last stage's scope, groups through `OnStage` (binding first);
            //     an unconfirmed bind or a lost lineage leaks instead. Being on the same thread alone is not relied on.
            unsafe { d.disable_event_tracking() };
            use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as A;
            let ctx = d.cuda_stream().context().clone();
            let major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let cap = (major * 10 + minor) as u32;
            // the PTX of the highest compiled architecture not above the GPU's (build.rs, CUDA_COMPUTE_CAPS)
            ensure!(
                fused::cu::select(cap).is_some(),
                "this build has kernels for CUDA compute capability {} and newer, but {} (CUDA device {device}) is \
                 {major}.{minor} (a build with CUDA_COMPUTE_CAPS={major}{minor} may run if the kernels support it; \
                 the release packages and images need 8.0 or newer)",
                env!("BASAL_CUDA_COMPUTE_CAPS"),
                ctx.name()?
            );
        }
        Ok(dev)
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    {
        ensure!(device == 0, "device {device}: the Metal backend runs on device 0 (the system GPU) only");
        Device::new_metal(0).context("Metal device")
    }
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    bail!("no GPU backend in this build for device {device} (macOS: Metal; Linux: build with --features cuda)")
}

/// Fixed pointwise arithmetic, selected before any cache or lane exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum LlamaNumerics {
    #[default]
    Ordinary,
    ThinkBf16,
}

pub struct GpuBackend {
    /// Consecutive layer ranges in layer order, each on its own GPU; one stage holds the whole model on one GPU (the
    /// path without a layer split). Never empty.
    stages: Vec<Stage>,
    /// Health (and on CUDA the quarantine) of this loaded model, shared with its forks.
    lineage: Lineage,
    /// Layer split: transfer counters per boundary, shared with the forks; None on one GPU.
    transfers: Option<Arc<TransferStats>>,
    /// Precomputed prefixes (prompt template up to the state, and with a state cache the shared prefix of recent
    /// single-row forwards); a row that starts with one of them skips its tokens.
    prefixes: Vec<PrefixKv>,
    /// One causal generation only; never searched by ordinary forwards or copied to a fork.
    generation: Option<PrefixKv>,
    /// Byte budget of non-pinned prefixes (0 = no state cache).
    pub state_cache_bytes: usize,
    clock: u64,
    /// Forwards that used a cached state prefix / that inserted one.
    pub state_cache_hits: u64,
    pub state_cache_inserts: u64,
    cfg: LlamaConfig,
    precision: Precision,
    readout: Readout,
    kernels: Kernels,
    numerics: LlamaNumerics,
    /// Fixed before prefix warmup and copied to every execution lane.
    attention_override: Option<&'static str>,
    /// CUDA: the attention kernel the device rule fixed for the GPU of each stage (by stage): set by the public
    /// loaders ([`DeviceDefaultRule::V1`]) or by the applied plan, before any K/V or lane exists, and copied with
    /// `attention_override`. Empty only between an unplanned load and `apply_plan`.
    attention_defaults: Vec<&'static str>,
    inv_freq: Vec<f64>,
    /// RoPE cos and sin of every position below `cfg.max_positions` (`rope_row`), `[pos][cos: hd/2 | sin: hd/2]`:
    /// computed once instead of per token and forward; host memory shared by the stages
    rope: std::sync::Arc<[f32]>,
    pub load_s: f64,
    /// When set, every forward section is synchronised and timed (slows the forward; diagnostic only).
    pub profile: Option<std::cell::RefCell<BTreeMap<&'static str, f64>>>,
    /// Format of the projection weights.
    quantization: Quantization,
    /// `q8_0` / `q4_k`: load diagnostics of the projection matrices (the `quantization` object of
    /// [`Backend::describe`]).
    #[cfg(feature = "cuda")]
    quant_stats: Option<quant::QuantStats>,
    /// `q8_0` / `q4_k` while profiling: synchronised time of the dequantizations (ms, a part of the projection
    /// sections) and the weight values they produced.
    dequant: std::cell::Cell<(f64, u64)>,
    /// When set, the largest |value| of the residual stream after every layer is recorded (diagnostic: f16 range).
    pub residual_max: Option<std::cell::RefCell<Vec<f32>>>,
    /// Timing of the last call in milliseconds: host preparation, GPU forward + readout (synchronised).
    pub last_timing: (f64, f64),
    /// Stop token of background work ([`Backend::set_stop_token`]); never inherited by forks.
    stop: Option<StopToken>,
}

/// One consecutive range of decoder layers on one GPU.
struct Stage {
    /// `gpu_device(device)` of this stage: its own candle device (handles, kernel caches) on the GPU's primary context.
    dev: Device,
    /// The ordinal `dev` was opened with ([`gpu_device`]).
    device: usize,
    /// Global indices of the stage's layers.
    layers: Range<usize>,
    /// Device memory of the stage (weights, its cuBLASLt handle), freed with the stage's context bound.
    w: OnStage<StageWeights>,
    /// Device shared with other engines ([`Backend::set_device_work_class`]): the gate and this engine's work class.
    gate: Option<(Arc<Gate>, WorkClass)>,
}

/// Weights of one stage, all on its GPU.
#[derive(Clone, Default)]
struct StageWeights {
    /// The stage's layers in order.
    layers: Vec<Layer>,
    /// Before a boundary: a copy of the `input_layernorm` of the next stage's first layer (the same bits), so that
    /// the residual add of the stage's last layer fuses the next block's RMSNorm as on one GPU.
    next_ln1: Option<Tensor>,
    /// Stage 0: the embedding.
    embed: Option<Tensor>,
    /// Last stage: final norm, lm_head and the evidence head.
    head: Option<Head>,
    /// CUDA: f16/bf16 projections through cuBLASLt with measured algorithms (None = candle's cuBLAS matmul).
    #[cfg(feature = "cuda")]
    lt: Option<Arc<cublaslt::Lt>>,
}

/// Readout weights of the last stage.
#[derive(Clone)]
struct Head {
    /// Immutable trained exits on this head's device, protected by the stage's bind/loss ownership rules.
    exits: Option<Arc<early_exit::ExitHeads>>,
    norm: Tensor,
    /// bf16 (checkpoint dtype); only the letter rows are gathered and converted to f32 per call.
    lm_head: Tensor,
    /// basal-1.5 evidence head (`evidence_head.pt`, f32): ws, us, we, ue `[k, d]`, bs, be `[1, d]`.
    evidence: Option<EvidenceHead>,
}

/// What `transfer`'s boundary time covers (`describe` of a layer split): fixed text, no measured numbers.
const STAGE_TRANSFER: &str = "Sequential device-to-device API copy after source drain and destination initialization; destination and source PTDS completion are confirmed. Boundary time includes allocation, initialization, synchronization, source release and context binding; physical copy route is not established.";

/// Error of a call to a model whose lineage was marked lost.
const LOST: &str = "the GPU work of this model could not be confirmed after an earlier failure: model disabled until it is loaded again";

/// Health of one loaded model and its forks (lanes): once marked lost, entry points refuse work before any GPU call.
#[derive(Clone, Default)]
struct Lineage {
    health: DeviceHealth,
    /// Buffers whose last GPU use could not be confirmed (layer split boundaries), never freed.
    #[cfg(feature = "cuda")]
    quarantine: Arc<std::sync::Mutex<Quarantine>>,
}

#[cfg(feature = "cuda")]
impl Lineage {
    /// Keep `value` alive until the end of the process instead of freeing memory a GPU may still use.
    fn retain<T: Send + 'static>(&self, value: T) {
        self.quarantine.lock().unwrap_or_else(std::sync::PoisonError::into_inner).held.push(Box::new(value));
    }
}

/// Retained buffers of a lineage. A lane holds at most one boundary copy and stops at its next health check once the
/// lineage is lost, so it adds at most its two buffers (source and destination, each at most `2 * M * H` f16 of its
/// largest forward); later calls are refused before any allocation.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct Quarantine {
    held: Vec<Box<dyn std::any::Any + Send>>,
}

#[cfg(feature = "cuda")]
impl Drop for Quarantine {
    fn drop(&mut self) {
        // deliberately leaked: freeing could release memory a copy still reads or writes
        std::mem::forget(std::mem::take(&mut self.held));
    }
}

/// Device memory of one stage. Drop binds the stage context first (two attempts), so cudarc's stream-ordered free
/// lands on the owning GPU; a lost lineage or an unconfirmed bind leaks the value instead of freeing it. Ownership is
/// per group (a stage's weights, one stage's part of a prefix), never per tensor.
struct OnStage<T> {
    /// The stage's context; None outside CUDA (the wrapper is then transparent).
    #[cfg(feature = "cuda")]
    ctx: Option<Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>>,
    /// Bind before freeing; false only for state-cache parts on one GPU (their thread only uses that context).
    bind: bool,
    health: DeviceHealth,
    /// Invariant: `Some` from construction until `release` or `drop` takes it.
    value: Option<T>,
}

impl<T> OnStage<T> {
    fn new(dev: &Device, bind: bool, health: &DeviceHealth, value: T) -> Self {
        #[cfg(feature = "cuda")]
        let ctx = match dev {
            Device::Cuda(d) => Some(d.cuda_stream().context().clone()),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        Self {
            #[cfg(feature = "cuda")]
            ctx,
            bind,
            health: health.clone(),
            value: Some(value),
        }
    }

    fn get(&self) -> &T {
        self.value.as_ref().expect("OnStage holds its value until release or drop")
    }

    fn get_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("OnStage holds its value until release or drop")
    }

    /// Free the value now (error paths), returning a cleanup error instead of printing it.
    fn release(mut self) -> Result<()> {
        match self.value.take() {
            Some(v) => self.free(v),
            None => Ok(()),
        }
    }

    fn free(&self, value: T) -> Result<()> {
        if self.health.is_lost() {
            // work of this lineage may still run: never free its memory
            std::mem::forget(value);
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (true, Some(ctx)) = (self.bind, &self.ctx) {
            if let Twice::Neither(a, b) = twice(|| ctx.bind_to_thread()) {
                self.health.mark_lost();
                std::mem::forget(value);
                bail!(
                    "binding CUDA device {} to free its memory failed twice ({a}; {b}): memory retained, model \
                     disabled until it is loaded again",
                    ctx.ordinal()
                );
            }
        }
        drop(value);
        Ok(())
    }
}

impl<T> Drop for OnStage<T> {
    fn drop(&mut self) {
        if let Some(v) = self.value.take() {
            if let Err(e) = self.free(v) {
                cleanup_note(format_args!("{e:#}"));
            }
        }
    }
}

/// Transient device allocations of one device for [`Backend::device_work_bound`]: bytes and the number of
/// allocations (the caller adds its allocator's per-allocation granularity). Saturating arithmetic; sizes come from
/// the shapes the source allocates, never from measurements.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Tally {
    pub(crate) bytes: u64,
    pub(crate) allocations: u64,
}

impl Tally {
    /// One allocation of `bytes`.
    pub(crate) fn add(&mut self, bytes: u64) {
        self.add_n(1, bytes);
    }

    /// `count` allocations of `bytes` each.
    pub(crate) fn add_n(&mut self, count: u64, bytes: u64) {
        self.bytes = self.bytes.saturating_add(count.saturating_mul(bytes));
        self.allocations = self.allocations.saturating_add(count);
    }

    /// Both sets live at once.
    pub(crate) fn plus(mut self, other: Tally) -> Tally {
        self.bytes = self.bytes.saturating_add(other.bytes);
        self.allocations = self.allocations.saturating_add(other.allocations);
        self
    }

    /// One of two sequential peaks (the larger bytes; the larger allocation count, conservatively).
    pub(crate) fn peak(self, other: Tally) -> Tally {
        Tally { bytes: self.bytes.max(other.bytes), allocations: self.allocations.max(other.allocations) }
    }
}

/// `a * b * c ...`, saturating.
pub(crate) fn prod(values: &[u64]) -> u64 {
    values.iter().fold(1u64, |n, v| n.saturating_mul(*v))
}

/// Last-resort diagnostic of a destructor, written after its cleanup (free, retention, lost marking) is done. A failed
/// write to stderr is deliberately ignored: `eprintln!` would panic, and a second panic while unwinding aborts.
fn cleanup_note(message: std::fmt::Arguments<'_>) {
    use std::io::Write as _;
    let _ = writeln!(std::io::stderr().lock(), "basal: {message}");
}

impl<T: Clone> Clone for OnStage<T> {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "cuda")]
            ctx: self.ctx.clone(),
            bind: self.bind,
            health: self.health.clone(),
            value: self.value.clone(),
        }
    }
}

/// Outcome of a CUDA call tried at most twice. cudarc first returns (and clears) an error that a `Drop` stored on the
/// context object (`check_err`) and then does not make the call, so a first `Err` does not show that the call ran;
/// only `Ok` proves it. Repeated errors prove nothing more, so there is no third attempt.
enum Twice<E> {
    First,
    Second(E),
    Neither(E, E),
}

fn twice<E>(mut f: impl FnMut() -> std::result::Result<(), E>) -> Twice<E> {
    match f() {
        Ok(()) => Twice::First,
        Err(a) => match f() {
            Ok(()) => Twice::Second(a),
            Err(b) => Twice::Neither(a, b),
        },
    }
}

/// Bind the context of CUDA device `dev` to this thread (two attempts); a no-op outside CUDA.
fn bind_twice(dev: &Device) -> Twice<String> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = dev {
        let ctx = d.cuda_stream().context().clone();
        return match twice(|| ctx.bind_to_thread()) {
            Twice::First => Twice::First,
            Twice::Second(e) => Twice::Second(e.to_string()),
            Twice::Neither(a, b) => Twice::Neither(a.to_string(), b.to_string()),
        };
    }
    let _ = dev;
    Twice::First
}

/// Free `value` (CUDA memory of `ctx`) with `ctx` bound, two attempts; otherwise retain it and mark the lineage lost.
#[cfg(feature = "cuda")]
fn free_on<T: Send + 'static>(
    ctx: &candle_core::cuda_backend::cudarc::driver::CudaContext,
    value: T,
    lineage: &Lineage,
) -> std::result::Result<(), String> {
    if let Twice::Neither(a, b) = twice(|| ctx.bind_to_thread()) {
        lineage.retain(value);
        lineage.health.mark_lost();
        return Err(format!("binding CUDA device {} failed twice ({a}; {b}): buffer retained", ctx.ordinal()));
    }
    drop(value);
    Ok(())
}

/// Free the stages of a failed load in stage order, each with its own context bound; the load error stays the cause
/// and cleanup errors become its context. A stage whose context cannot be bound (two attempts) is leaked and marks the
/// not yet published lineage lost, after which the later stages are leaked as well (`OnStage::free`): a conservative
/// end of a start-up that fails anyway.
fn release_built(built: Vec<OnStage<StageWeights>>, mut err: anyhow::Error) -> anyhow::Error {
    for w in built {
        if let Err(c) = w.release() {
            err = err.context(format!("cleanup after the failed load: {c:#}"));
        }
    }
    err
}

/// What is known about a boundary copy ([`GpuBackend::transfer`]); it decides how its buffers may end.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum CopyState {
    /// The source is written completely and nothing reads it; no destination yet.
    Drained,
    /// The destination is allocated; its memset may still be pending on this thread's stream of its GPU; no copy.
    Allocated,
    /// The memset completed; no copy queued.
    Initialized,
    /// The copy may have been queued on either GPU (set before the call, kept after `Ok` and `Err`).
    MaybeInFlight,
    /// Both GPUs confirmed: the copy finished or never ran; nothing reads the source or writes the destination.
    Completed,
    /// The buffers left the copy (freed, retained or handed on).
    Done,
}

/// The two buffers of one boundary copy, owned from before the destination is allocated until a final state. Buffers
/// known idle are freed with their owner's context bound (two attempts, otherwise retained); in `MaybeInFlight` both
/// are retained and the lineage marked lost, without GPU calls. `Drop` (a panic) applies the same rules.
#[cfg(feature = "cuda")]
struct StageCopy<'a> {
    state: CopyState,
    /// The source buffer (the whole storage: the views of the stage body are gone).
    src: Option<Tensor>,
    dst: Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<half::f16>>,
    src_ctx: Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
    dst_ctx: Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
    lineage: &'a Lineage,
    /// "stage transfer {s}→{s+1}, CUDA device {a}→{b}, {bytes} B"
    what: String,
}

#[cfg(feature = "cuda")]
impl StageCopy<'_> {
    /// The conditions cudarc leaves to the caller (it asserts the lengths): one contiguous f16 storage of `n` values
    /// from offset 0. Returns the shape; each architecture validates its activation dimensions.
    fn check(&self, n: usize) -> Result<Vec<usize>> {
        let src = self.src.as_ref().context("the source buffer")?;
        ensure!(src.dtype() == DType::F16, "boundary buffer of dtype {:?}, f16 expected", src.dtype());
        let dims = src.dims().to_vec();
        let (st, l) = src.storage_and_layout();
        ensure!(l.is_contiguous() && l.start_offset() == 0, "boundary buffer is not contiguous from offset 0");
        let candle_core::Storage::Cuda(cs) = &*st else { bail!("boundary buffer is not on a CUDA device") };
        let len = cs.as_cuda_slice::<half::f16>()?.len();
        ensure!(len == n, "boundary buffer storage holds {len} values, {n} expected");
        Ok(dims)
    }

    /// Queue the device-to-device copy of the whole source storage into the destination, on the destination's stream.
    fn copy_on(&mut self, dst_dev: &candle_core::CudaDevice) -> Result<()> {
        let (Some(src), Some(dst)) = (&self.src, &mut self.dst) else { bail!("the boundary buffers are missing") };
        let (st, _) = src.storage_and_layout();
        let candle_core::Storage::Cuda(cs) = &*st else { bail!("boundary buffer is not on a CUDA device") };
        dst_dev.cuda_stream().memcpy_dtod(cs.as_cuda_slice::<half::f16>()?, dst)?;
        Ok(())
    }

    /// Free the buffers (none in flight), each with its owner's context bound; returns the cleanup errors.
    fn release(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        if let Some(src) = self.src.take() {
            errors.extend(free_on(&self.src_ctx, src, self.lineage).err());
        }
        if let Some(dst) = self.dst.take() {
            errors.extend(free_on(&self.dst_ctx, dst, self.lineage).err());
        }
        errors
    }

    fn quarantine(&mut self) {
        if let Some(src) = self.src.take() {
            self.lineage.retain(src);
        }
        if let Some(dst) = self.dst.take() {
            self.lineage.retain(dst);
        }
        self.lineage.health.mark_lost();
    }

    /// The request fails with nothing of this copy in flight: free both buffers; `err` stays the cause, cleanup
    /// errors become its context.
    fn fail(&mut self, err: anyhow::Error) -> anyhow::Error {
        let cleanup = self.release();
        self.state = CopyState::Done;
        let mut err = err.context(self.what.clone());
        for c in cleanup {
            err = err.context(format!("cleanup: {c}"));
        }
        err
    }

    /// The request fails and the end of GPU work on the buffers is unknown: retain both, mark the lineage lost.
    fn retain(&mut self, reason: String) -> anyhow::Error {
        self.quarantine();
        self.state = CopyState::Done;
        anyhow::anyhow!("{}: {reason}: buffers retained, model disabled until it is loaded again", self.what)
    }
}

#[cfg(feature = "cuda")]
impl Drop for StageCopy<'_> {
    fn drop(&mut self) {
        match self.state {
            CopyState::Done => {}
            CopyState::MaybeInFlight => {
                self.quarantine();
                cleanup_note(format_args!(
                    "{}: interrupted with the copy possibly in flight: buffers retained, model disabled until it is \
                     loaded again",
                    self.what
                ));
            }
            CopyState::Drained | CopyState::Allocated | CopyState::Initialized | CopyState::Completed => {
                for e in self.release() {
                    cleanup_note(format_args!("{}: cleanup: {e}", self.what));
                }
            }
        }
    }
}

/// Copy one complete FP16 activation between adjacent stages, with the same drain, binding and quarantine rules
/// for every architecture. Gates and devices are borrowed from the same call's owners.
#[cfg(feature = "cuda")]
fn transfer_activation(
    gates: &mut StageGates<'_>,
    s: usize,
    devices: (usize, usize),
    buf: Tensor,
    lineage: &Lineage,
    stats: Option<&TransferStats>,
) -> Result<Tensor> {
    let from = gates.owners.devs[s].clone();
    let to = gates.owners.devs[s + 1].clone();
    use candle_core::cuda_backend::CudaStorage;
    let n = buf.elem_count();
    let bytes = n * DType::F16.size_in_bytes();
    let what = format!("stage transfer {s}→{}, CUDA device {}→{}, {bytes} B", s + 1, devices.0, devices.1);
    let (Device::Cuda(src_dev), Device::Cuda(dst_dev)) = (&from, &to) else {
        bail!("{what}: both stages must be CUDA devices");
    };
    // step 1
    match twice(|| from.synchronize()) {
        Twice::First => {}
        Twice::Second(e) => {
            // drained after an earlier stored error surfaced: free the source (its context is current), fail
            drop(buf);
            gates.mark_drained();
            return Err(anyhow::Error::new(e).context(format!("{what}: draining stage {s}: first attempt failed")));
        }
        Twice::Neither(a, b) => {
            lineage.retain(buf);
            lineage.health.mark_lost();
            bail!(
                "{what}: draining stage {s} failed twice ({a}; {b}): buffers retained, model disabled until it is \
                     loaded again"
            );
        }
    }
    let mut copy = StageCopy {
        state: CopyState::Drained,
        src: Some(buf),
        dst: None,
        src_ctx: src_dev.cuda_stream().context().clone(),
        dst_ctx: dst_dev.cuda_stream().context().clone(),
        lineage,
        what,
    };
    let shape = match copy.check(n) {
        Ok(shape) => shape,
        Err(e) => return Err(copy.fail(e)),
    };
    // steps 2–3
    let waited = Instant::now();
    gates.mark_drained();
    if let Err(e) = gates.switch_to(s + 1) {
        return Err(copy.fail(e));
    }
    let gate_wait = waited.elapsed();
    let started = Instant::now();
    // step 4
    match dst_dev.alloc_zeros::<half::f16>(n) {
        Ok(d) => {
            copy.dst = Some(d);
            copy.state = CopyState::Allocated;
        }
        Err(e) => return Err(copy.fail(e.into())),
    }
    // step 5
    match twice(|| to.synchronize()) {
        Twice::First => copy.state = CopyState::Initialized,
        Twice::Second(e) => {
            return Err(copy.fail(anyhow::Error::new(e).context("draining the destination's initialisation")))
        }
        Twice::Neither(a, b) => {
            return Err(copy.retain(format!("initialisation of the destination not confirmed ({a}; {b})")))
        }
    }
    // step 6 (from here until both confirmations the copy may be queued on either GPU)
    copy.state = CopyState::MaybeInFlight;
    let copied = copy.copy_on(dst_dev);
    // step 7
    let on_dst = twice(|| to.synchronize());
    let on_src = twice(|| from.synchronize());
    let report = |t: &Twice<candle_core::Error>, stage: usize| match t {
        Twice::First => None,
        Twice::Second(e) => Some(format!("confirmation on stage {stage}: first attempt failed ({e})")),
        Twice::Neither(a, b) => Some(format!("confirmation on stage {stage} failed twice ({a}; {b})")),
    };
    let problems: Vec<String> = copied
        .as_ref()
        .err()
        .map(|e| format!("copy: {e:#}"))
        .into_iter()
        .chain(report(&on_dst, s + 1))
        .chain(report(&on_src, s))
        .collect();
    if matches!(on_dst, Twice::Neither(..)) || matches!(on_src, Twice::Neither(..)) {
        return Err(copy.retain(format!("end of the copy not confirmed ({})", problems.join("; "))));
    }
    copy.state = CopyState::Completed;
    if !problems.is_empty() {
        return Err(copy.fail(anyhow::anyhow!("{}", problems.join("; "))));
    }
    // step 8: stage s is current: its confirmation bound it
    drop(copy.src.take());
    // step 9
    match bind_twice(&to) {
        Twice::First => {}
        Twice::Second(e) => {
            return Err(copy.fail(anyhow::anyhow!("binding stage {} after the copy: first attempt failed ({e})", s + 1)))
        }
        Twice::Neither(a, b) => {
            return Err(copy.retain(format!("binding stage {} after the copy failed twice ({a}; {b})", s + 1)))
        }
    }
    let dst = copy.dst.take().context("the destination of the transfer")?;
    copy.state = CopyState::Done;
    let storage = candle_core::Storage::Cuda(CudaStorage::wrap_cuda_slice(dst, dst_dev.clone()));
    let out = Tensor::from_storage(storage, shape, candle_core::op::BackpropOp::none(), false);
    if let Some(stats) = stats {
        stats.record(s, bytes, gate_wait, started.elapsed());
    }
    Ok(out)
}

/// Counters of one stage boundary (host clock, relaxed atomics: independent counters read for reports).
struct BoundaryStats {
    from_device: usize,
    to_device: usize,
    copies: AtomicU64,
    bytes: AtomicU64,
    /// Steps 4–9 of `transfer` after the context binding of step 4: allocation, initialisation drain, copy, both
    /// confirmations, source release, final binding.
    transfer_ns: AtomicU64,
    /// Steps 2–3 of `transfer` and the binding of step 4: releasing this stage, taking the next stage's gate and
    /// binding its context.
    gate_wait_ns: AtomicU64,
}

/// Transfer counters of a layer split, per boundary (only with a split).
struct TransferStats {
    boundaries: Vec<BoundaryStats>,
}

impl TransferStats {
    fn new(stages: &[Stage]) -> Self {
        Self::for_devices(&stages.iter().map(|s| s.device).collect::<Vec<_>>())
    }

    fn for_devices(devices: &[usize]) -> Self {
        Self {
            boundaries: devices
                .windows(2)
                .map(|p| BoundaryStats {
                    from_device: p[0],
                    to_device: p[1],
                    copies: AtomicU64::new(0),
                    bytes: AtomicU64::new(0),
                    transfer_ns: AtomicU64::new(0),
                    gate_wait_ns: AtomicU64::new(0),
                })
                .collect(),
        }
    }

    #[cfg(feature = "cuda")]
    fn record(&self, boundary: usize, bytes: usize, gate_wait: std::time::Duration, transfer: std::time::Duration) {
        if let Some(b) = self.boundaries.get(boundary) {
            let ns = |d: std::time::Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
            b.copies.fetch_add(1, Ordering::Relaxed);
            b.bytes.fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
            b.transfer_ns.fetch_add(ns(transfer), Ordering::Relaxed);
            b.gate_wait_ns.fetch_add(ns(gate_wait), Ordering::Relaxed);
        }
    }
}

/// Transfer counters of a layer-split model (shared with its forks) for reports: per boundary the device-to-device
/// API calls, bytes and host time. They observe the API calls; they do not establish the physical copy route.
#[derive(Clone)]
pub struct StageTransfers(Arc<TransferStats>);

impl StageTransfers {
    /// The counters as JSON, one object per boundary in stage order.
    pub fn report(&self) -> Value {
        let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e6;
        Value::Array(
            self.0
                .boundaries
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    json!({"from_stage": i, "to_stage": i + 1, "from_device": b.from_device, "to_device": b.to_device,
                        "copies": b.copies.load(Ordering::Relaxed), "bytes": b.bytes.load(Ordering::Relaxed),
                        "transfer_ms": ms(&b.transfer_ns), "gate_wait_ms": ms(&b.gate_wait_ns),
                        "scope": "host time of successful boundary transfers: transfer_ms covers allocation and initialisation of the destination, its drain, the device-to-device API copy, confirmation on both GPUs, source release and the final context binding; gate_wait_ms covers taking the next GPU's gate and binding its context; the physical copy route is not established"})
                })
                .collect(),
        )
    }

    /// Start counting from zero (outside measured intervals).
    pub fn reset(&self) {
        for b in &self.0.boundaries {
            for a in [&b.copies, &b.bytes, &b.transfer_ns, &b.gate_wait_ns] {
                a.store(0, Ordering::Relaxed);
            }
        }
    }
}

/// Owners a call borrows its gates from, cloned at the call's start, so that the guards do not borrow the backend
/// (helpers taking `&mut self` can carry them).
struct GateOwners {
    gates: Vec<Option<(Arc<Gate>, WorkClass)>>,
    devs: Vec<Device>,
    health: DeviceHealth,
    split: bool,
    /// Stop token of background work: observed at every stage entry, resume and checkpoint.
    stop: Option<StopToken>,
}

/// The stage gates of one call: at most one held at a time (no hold-and-wait between GPUs). Leaving a stage releases
/// its gate only after this thread's work on that GPU was drained with `Ok` (a synchronisation, or a synchronous host
/// readout marked with [`StageGates::mark_drained`]). With a split, entering a stage binds its context.
///
/// Background work ([`WorkClass::Background`]) drains only this thread's work on the current stage before it parks or
/// stops at a checkpoint, marks that state, and after an [`Interrupted`] following the release it leaves no current
/// stage, so the error path never synchronises a GPU that online work may already use. An unconfirmed drain marks
/// the lineage lost ([`DeviceHealth`]); an interruption alone does not.
struct StageGates<'g> {
    owners: &'g GateOwners,
    /// Stage entered last (its gate held when it has one).
    current: Option<usize>,
    guard: Option<GateGuard<'g>>,
    /// The work of this thread on `current` is known to be complete.
    drained: bool,
    /// A preemptible lane has started its task: later entries continue it (`Gate::resume`).
    started: bool,
}

impl<'g> StageGates<'g> {
    fn new(owners: &'g GateOwners) -> Self {
        Self { owners, current: None, guard: None, drained: false, started: false }
    }

    /// Enter stage `s` (nothing when it is already the current one): leave the current stage, take the gate of `s`,
    /// check the lineage's health and, with a split, bind the stage's context.
    fn switch_to(&mut self, s: usize) -> Result<()> {
        if self.current == Some(s) {
            return Ok(());
        }
        self.leave()?;
        let o = self.owners;
        if let Some(stop) = &o.stop {
            stop.check()?;
        }
        if let Some((g, class)) = &o.gates[s] {
            // online main: always a new entry; online long and background continue a started task in FIFO order
            let taken = if self.started {
                g.resume_class(*class, o.stop.as_ref())
            } else {
                g.enter_class(*class, o.stop.as_ref())
            };
            self.guard = Some(taken?);
        }
        // A lane may have marked the model lost while this call waited for the next gate.
        // No GPU work was started here: release it without assigning a stage that error cleanup would drain.
        if o.health.is_lost() {
            drop(self.guard.take());
            bail!(LOST);
        }
        self.started = true;
        self.current = Some(s);
        self.drained = false;
        if o.split {
            match bind_twice(&o.devs[s]) {
                Twice::Neither(a, b) => bail!("binding stage {s} failed twice ({a}; {b})"),
                Twice::Second(e) => bail!("binding stage {s}: first attempt failed ({e})"),
                Twice::First => {}
            }
        }
        Ok(())
    }

    /// The last work of this call on the current stage was a synchronous host readout: nothing of it is pending.
    fn mark_drained(&mut self) {
        self.drained = true;
    }

    /// Preemptible lane between two layers: drain the current stage, then let an online caller run if one waits.
    /// After it yielded, the lineage's health is checked again (the other caller may have marked it lost before
    /// releasing the gate), as on every gate entry.
    ///
    /// Online long lane: as before (one synchronisation, then [`Gate::checkpoint`] under `min_slice`). Background:
    /// a confirmed drain (two attempts), marked before parking; a stop releases the gate and returns
    /// [`Interrupted`] with no current stage. A stage without a gate observes only the background stop token.
    fn checkpoint(&mut self) -> Result<()> {
        let Some(c) = self.current else { return Ok(()) };
        let o = self.owners;
        match &o.gates[c] {
            Some((_, WorkClass::OnlineMain)) => Ok(()),
            Some((g, WorkClass::OnlineLong)) => {
                o.devs[c].synchronize()?;
                let Some(guard) = self.guard.take() else { return Ok(()) };
                let cp = g.checkpoint(guard, None)?;
                self.guard = Some(cp.guard);
                if cp.yielded {
                    ensure!(!o.health.is_lost(), LOST);
                }
                Ok(())
            }
            Some((g, WorkClass::Background)) => {
                self.drain_current(c)?;
                let Some(guard) = self.guard.take() else { return Ok(()) };
                match g.checkpoint(guard, o.stop.as_ref()) {
                    Ok(cp) => {
                        self.guard = Some(cp.guard);
                        // the caller queues more work on this stage
                        self.drained = false;
                        if cp.yielded {
                            ensure!(!o.health.is_lost(), LOST);
                        }
                        Ok(())
                    }
                    Err(stopped) => {
                        // drained and released: nothing of this thread is pending on a GPU online work may use now
                        self.current = None;
                        Err(stopped.into())
                    }
                }
            }
            None => match o.stop.as_ref().and_then(StopToken::reason) {
                Some(reason) => {
                    self.drain_current(c)?;
                    self.current = None;
                    Err(Interrupted { reason }.into())
                }
                None => Ok(()),
            },
        }
    }

    /// Background checkpoint inside an encoder or between media items: a no-op for online classes (their existing
    /// checkpoints are unchanged), [`StageGates::checkpoint`] for background work.
    fn background_checkpoint(&mut self) -> Result<()> {
        let Some(c) = self.current else { return Ok(()) };
        match &self.owners.gates[c] {
            Some((_, WorkClass::Background)) => self.checkpoint(),
            None if self.owners.stop.is_some() => self.checkpoint(),
            _ => Ok(()),
        }
    }

    /// Confirm the end of this thread's work on stage `c` (two attempts) and mark it drained; unconfirmed, the
    /// lineage is marked lost and the error path keeps its rules.
    fn drain_current(&mut self, c: usize) -> Result<()> {
        match twice(|| self.owners.devs[c].synchronize()) {
            Twice::First => {
                self.drained = true;
                Ok(())
            }
            Twice::Second(e) => bail!("draining stage {c} at a checkpoint: first attempt failed ({e})"),
            Twice::Neither(a, b) => {
                self.owners.health.mark_lost();
                bail!("draining stage {c} at a checkpoint failed twice ({a}; {b}): {LOST}")
            }
        }
    }

    /// Success-path exit from the current stage: drain unless drained (`Ok` required), then release the gate. Two
    /// failed drains mark the lineage lost before the gate is released.
    ///
    /// A stage without a gate is left without a drain, also with a split: no other lane of this model uses that GPU
    /// (serve gates every GPU of a model with a long lane), and this thread's later work there (the next request's
    /// stage body after binding the context, LRU frees after binding it) is queued on the same per-thread stream of
    /// that context, after the pending work. A boundary copy confirms both GPUs itself (`transfer`), and the error
    /// path confirms the current stage ([`StageGates::release_on_error`]).
    fn leave(&mut self) -> Result<()> {
        let Some(c) = self.current.take() else { return Ok(()) };
        let Some(guard) = self.guard.take() else { return Ok(()) };
        if self.drained {
            drop(guard);
            return Ok(());
        }
        match twice(|| self.owners.devs[c].synchronize()) {
            Twice::First => {
                drop(guard);
                Ok(())
            }
            Twice::Second(e) => {
                drop(guard);
                bail!("draining stage {c} before releasing its gate: first attempt failed ({e})")
            }
            Twice::Neither(a, b) => {
                self.owners.health.mark_lost();
                drop(guard);
                bail!("draining stage {c} before releasing its gate failed twice ({a}; {b}): {LOST}")
            }
        }
    }

    /// Error-path exit: always synchronise the current stage (two attempts) when its gate is held or the model is
    /// split (also a stage without a gate); unconfirmed, mark the lineage lost before a held gate is released. One GPU
    /// without a gate keeps today's error path: no synchronisation. Returns the cleanup error, if any.
    fn release_on_error(&mut self) -> Option<String> {
        let c = self.current.take()?;
        let guard = self.guard.take();
        if guard.is_none() && !self.owners.split {
            return None;
        }
        let cleanup = match twice(|| self.owners.devs[c].synchronize()) {
            Twice::First => None,
            Twice::Second(e) => Some(format!("draining stage {c} after the error: first attempt failed ({e})")),
            Twice::Neither(a, b) => {
                self.owners.health.mark_lost();
                Some(format!("draining stage {c} after the error failed twice ({a}; {b}): {LOST}"))
            }
        };
        drop(guard);
        cleanup
    }

    /// End of the call: the result on success (after a drained release), else the original error with the cleanup
    /// errors as context.
    fn end<T>(mut self, result: Result<T>) -> Result<T> {
        match result {
            Ok(v) => {
                self.leave()?;
                Ok(v)
            }
            Err(e) => Err(match self.release_on_error() {
                Some(cleanup) => e.context(cleanup),
                None => e,
            }),
        }
    }
}

impl Drop for StageGates<'_> {
    /// Panic safeguard: the error-path release, with the cleanup error on stderr.
    fn drop(&mut self) {
        if let Some(cleanup) = self.release_on_error() {
            cleanup_note(format_args!("{cleanup}"));
        }
    }
}

/// Host inputs of one ragged forward, shared by its stages.
struct Ragged<'a, 'p> {
    ids: &'a [u32],
    cos_sin: &'a [f32],
    segs: &'a [Segment<'p>],
    read: &'a [u32],
    /// Capture mode (one segment, optional readout): `Some(bind)` with the `OnStage::bind` of the captured parts.
    capture: Option<bool>,
    dims: HeadDims,
    /// CUDA: the blocks of all rows as tree-attention units (compact offsets; prefix addresses set per layer).
    #[cfg(feature = "cuda")]
    units: Vec<(fused::TreeUnit, Option<&'p PrefixKv>)>,
    #[cfg(target_os = "macos")]
    metal_units: Vec<fused::MetalTreeUnit>,
    #[cfg(target_os = "macos")]
    metal_pasts: Vec<&'p PrefixKv>,
}

/// What leaves a stage body: the boundary buffer `[2, M, H]` for the next stage, or the last stage's final normalised
/// hidden states (None only for capture without readout).
enum StageOut {
    Boundary(Tensor),
    Final(Option<Tensor>),
}

#[derive(Clone)]
struct EvidenceHead {
    ws: Tensor,
    us: Tensor,
    we: Tensor,
    ue: Tensor,
    bs: Tensor,
    be: Tensor,
    k: usize,
}

impl EvidenceHead {
    fn load(path: &Path, d: usize, dev: &Device) -> Result<Self> {
        let all = candle_core::pickle::read_all_with_key(path, Some("state_dict"))
            .with_context(|| format!("reading {}", path.display()))?;
        let get = |n: &str| -> Result<Tensor> {
            let t = all.iter().find(|(k, _)| k == n).with_context(|| format!("{}: {n} missing", path.display()))?;
            Ok(t.1.to_dtype(DType::F32)?.to_device(dev)?)
        };
        let ws = get("ws.weight")?;
        let k = ws.dim(0)?;
        let h = Self {
            us: get("us.weight")?,
            we: get("we.weight")?,
            ue: get("ue.weight")?,
            bs: get("bs.weight")?,
            be: get("be.weight")?,
            ws,
            k,
        };
        for t in [&h.ws, &h.us, &h.we, &h.ue] {
            ensure!(t.dims() == [k, d], "{}: projection shape {:?}, expected [{k}, {d}]", path.display(), t.dims());
        }
        for t in [&h.bs, &h.be] {
            ensure!(t.dims() == [1, d], "{}: bias shape {:?}, expected [1, {d}]", path.display(), t.dims());
        }
        ensure!(all.len() == 6, "{}: unexpected tensors in the evidence head", path.display());
        Ok(h)
    }

    /// Non-affine LayerNorm over the last dimension (torch default eps 1e-5), f32.
    fn norm(x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(1)?;
        let c = x.broadcast_sub(&mean)?;
        let var = c.sqr()?.mean_keepdim(1)?;
        Ok(c.broadcast_div(&(var + 1e-5)?.sqrt()?)?)
    }

    /// `EvidenceHead.forward`: h `[n, d]` state tokens, q `[1, d]` answer position -> start / end log-probs.
    fn scores(&self, h: &Tensor, q: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        let (h, q) = (Self::norm(h)?, Self::norm(q)?);
        let r = (self.k as f64).sqrt();
        let side = |w: &Tensor, u: &Tensor, b: &Tensor| -> Result<Vec<f32>> {
            let x = ((h.matmul(&w.t()?)?.matmul(&q.matmul(&u.t()?)?.t()?)? / r)? + h.matmul(&b.t()?)?)?.squeeze(1)?;
            Ok(candle_nn::ops::log_softmax(&x, 0)?.to_vec1()?)
        };
        Ok((side(&self.ws, &self.us, &self.bs)?, side(&self.we, &self.ue, &self.be)?))
    }
}

/// Tensor data of a safetensors file read with plain reads. With the file mapped, page faults on the 22 GB of
/// basal-1.5-max on a 32 GB Mac, under memory pressure from the weights on the GPU, made loading 2-3 times slower
/// (M2 Max: 80-123 s mapped, 37-39 s read; reads past the page cache, F_NOCACHE: 59-65 s).
struct TensorReader {
    file: std::fs::File,
    /// first byte of the data section
    base: u64,
    /// byte range of each tensor in the data section
    ranges: std::collections::HashMap<String, (u64, u64)>,
}

impl TensorReader {
    fn open(path: &Path) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mut n = [0u8; 8];
        file.read_exact(&mut n)?;
        let n = u64::from_le_bytes(n);
        let file_bytes = file.metadata()?.len();
        ensure!(n <= 100_000_000 && n <= file_bytes.saturating_sub(8), "invalid safetensors header length {n}");
        let mut header = vec![0u8; usize::try_from(n)?];
        file.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header).context("safetensors header")?;
        let mut ranges = std::collections::HashMap::new();
        for (name, t) in header.as_object().context("safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            let r = t["data_offsets"].as_array().context("data_offsets")?;
            ensure!(r.len() == 2, "tensor {name}: expected two data_offsets");
            let (s, e) = (r[0].as_u64().context("data_offsets")?, r[1].as_u64().context("data_offsets")?);
            ensure!(s <= e && e <= file_bytes - 8 - n, "tensor {name}: data_offsets outside the checkpoint");
            ranges.insert(name.clone(), (s, e));
        }
        Ok(Self { file, base: 8 + n, ranges })
    }

    /// Append the bytes of tensor `name` to `out`.
    fn read(&self, name: &str, out: &mut Vec<u8>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let &(s, e) = self.ranges.get(name).with_context(|| format!("tensor {name} not in the checkpoint"))?;
        let at = out.len();
        out.resize(at + usize::try_from(e - s)?, 0);
        self.file.read_exact_at(&mut out[at..], self.base + s)?;
        Ok(())
    }
}

/// Little-endian bf16 bytes -> f16 (through f32, round to nearest even), split over the available cores.
fn bf16_to_f16(src: &[u8], dst: &mut [half::f16]) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = dst.len().div_ceil(threads).max(1 << 16);
    std::thread::scope(|s| {
        for (d, b) in dst.chunks_mut(chunk).zip(src.chunks(2 * chunk)) {
            s.spawn(move || {
                for (o, c) in d.iter_mut().zip(b.chunks_exact(2)) {
                    *o = half::f16::from_f32(half::bf16::from_le_bytes([c[0], c[1]]).to_f32());
                }
            });
        }
    });
}

/// Read the bf16 parts of a projection weight (concatenated along N, as the dense loader does), quantize them on the
/// host with `quantize` (`q8_0`: [`quant::quantize_q8_0`]; `q4_k`: [`quant::quantize_q4_k`]) and upload the blocks
/// once onto `dev` (Candle's `load_quantized`: one zeroed allocation with padding, then a copy on the loading thread's
/// stream). Checks the invariants Candle leaves to the caller: the block type and bytes of the shape, and that the
/// blocks live on `dev` itself (its `DeviceId`), so dequantizing them onto `q.device()` never transfers. The host
/// blocks are dropped here unless `keep_blocks` (the q4_k warm-up check).
#[cfg(feature = "cuda")]
fn load_blocks<B: candle_core::quantized::GgmlType + Send + Sync + 'static, S>(
    st: &MmapedSafetensors,
    reader: &TensorReader,
    names: &[String],
    dev: &Device,
    quantization: Quantization,
    keep_blocks: bool,
    quantize: impl FnOnce(&[u8], usize, usize) -> Result<(Vec<B>, S)>,
) -> Result<(candle_core::quantized::QTensor, Option<Vec<B>>, S)> {
    use candle_core::quantized::QTensor;
    let Device::Cuda(cuda) = dev else { bail!("{quantization} weights need a CUDA device") };
    let (mut rows, mut cols) = (0usize, None);
    for n in names {
        let v = st.get(n)?;
        let &[r, c] = v.shape() else { bail!("{n}: shape {:?}, expected a matrix", v.shape()) };
        ensure!(cols.is_none_or(|k| k == c), "{n}: K = {c} differs from the other parts of the projection");
        cols = Some(c);
        rows = rows.checked_add(r).context("row count overflows")?;
    }
    let cols = cols.context("a projection without tensors")?;
    let mut bytes = Vec::new();
    for n in names {
        reader.read(n, &mut bytes)?;
    }
    let (blocks, stats) = quantize(&bytes, rows, cols)?;
    drop(bytes);
    let storage = candle_core::quantized::cuda::load_quantized(cuda, &blocks)?;
    let q = QTensor::new(storage, (rows, cols))?;
    ensure!(q.dtype() == B::DTYPE, "uploaded blocks of type {:?}", q.dtype());
    let expected = blocks.len().checked_mul(std::mem::size_of::<B>()).context("block bytes overflow")?;
    ensure!(
        q.storage_size_in_bytes() == expected,
        "{} bytes of blocks on the device, expected {expected}",
        q.storage_size_in_bytes()
    );
    ensure!(q.device().same_device(dev), "the {quantization} blocks are not on the backend's candle device");
    Ok((q, keep_blocks.then_some(blocks), stats))
}

/// The bf16 tensors of a projection, concatenated along N in this order: name and rows (q4_k positions).
#[cfg(feature = "cuda")]
fn source_parts(st: &MmapedSafetensors, names: &[String]) -> Result<Vec<(String, usize)>> {
    names
        .iter()
        .map(|n| {
            let v = st.get(n)?;
            let &[r, _] = v.shape() else { bail!("{n}: shape {:?}, expected a matrix", v.shape()) };
            Ok((n.clone(), r))
        })
        .collect()
}

fn check(st: &MmapedSafetensors, name: &str, shape: &[usize]) -> Result<()> {
    let v = st.get(name).with_context(|| format!("checkpoint tensor {name} missing"))?;
    ensure!(v.shape() == shape, "{name}: shape {:?}, expected {:?}", v.shape(), shape);
    ensure!(v.dtype() == safetensors::Dtype::BF16, "{name}: dtype {:?}, expected BF16", v.dtype());
    Ok(())
}

impl GpuBackend {
    /// Only called by the validated planned loader, before prefixes, gates or execution lanes exist.
    fn enable_early_exit(&mut self, artifacts: &basal_core::early_exit::ExitArtifacts) -> Result<()> {
        ensure!(
            self.is_cuda()
                && self.stages.len() == 1
                && self.kernels == Kernels::Candle
                && self.quantization == Quantization::None
                && self.readout == Readout::F32,
            "trained exits require single CUDA Llama, Candle kernels, no quantization"
        );
        ensure!(self.prefixes.is_empty(), "enable trained exits before prefix warmup");
        let heads = early_exit::ExitHeads::load(artifacts, self.cfg.hidden, self.cfg.num_layers)?;
        let stage = &mut self.stages[0];
        let head = stage.w.get_mut().head.as_mut().context("single stage holds the LM head")?;
        head.exits = Some(Arc::new(heads));
        // No fork exists at this load boundary; the stage already owns every partially uploaded tensor.
        Arc::get_mut(head.exits.as_mut().context("exit heads just installed")?)
            .context("exit heads cannot be shared before upload")?
            .upload(&stage.dev)?;
        Ok(())
    }

    /// Load a model onto GPU `device`. `quantization` sets the format of the projection weights; a quantized format is
    /// checked against `precision`, `kernels` and this build's backend before anything is opened.
    ///
    /// Without an execution plan the backend runs the generic attention policy: no measured model profile, the
    /// CUDA kernel that [`DeviceDefaultRule::V1`] fixes for the GPU, decided before the weights are read
    /// (`BASAL_ATT` still selects another kernel for A/B). A model instance with a profile, inference mode or
    /// GEMM table is loaded by [`RuntimeBackend::load_planned`].
    pub fn load(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        device: usize,
    ) -> Result<Self> {
        quantization.check(precision, kernels, cfg!(feature = "cuda"))?;
        let defaults = DeviceDefaultRule::V1.stage_kernels(&[device])?;
        let mut backend = Self::load_stages(
            model_dir,
            cfg,
            precision,
            readout,
            kernels,
            quantization,
            &[(device, 0..cfg.num_layers)],
        )?;
        backend.attention_defaults = defaults;
        Ok(backend)
    }

    /// Load a model split by consecutive layer ranges over several CUDA devices ([`LayerSplit`]): every stage holds its
    /// layers, cuBLASLt handle and prefix K/V on its GPU; stage 0 also the embedding, the last stage the final norm,
    /// lm_head and evidence head. The split is checked against this build's backend, `precision`, `kernels` and the
    /// model's layer count before anything is opened. Computations are those of one GPU (same kernels, GEMM tables
    /// and bits of every weight); the activations move between GPUs at each boundary.
    ///
    /// The attention policy is the generic one of [`GpuBackend::load`], decided for the GPU of every stage.
    pub fn load_split(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
    ) -> Result<Self> {
        let stages = Self::split_stages(cfg, precision, kernels, quantization, split)?;
        let devices: Vec<usize> = stages.iter().map(|(d, _)| *d).collect();
        let defaults = DeviceDefaultRule::V1.stage_kernels(&devices)?;
        let mut backend = Self::load_stages(model_dir, cfg, precision, readout, kernels, quantization, &stages)?;
        backend.attention_defaults = defaults;
        Ok(backend)
    }

    /// [`GpuBackend::load`] without an attention policy, for [`RuntimeBackend::load_planned`], which applies the
    /// plan's policy before any K/V or lane exists. Not usable for a forward before that.
    #[cfg_attr(
        all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")),
        expect(dead_code, reason = "an Intel-only build loads Llama through IntelBackend")
    )]
    pub(crate) fn load_unplanned(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        device: usize,
    ) -> Result<Self> {
        quantization.check(precision, kernels, cfg!(feature = "cuda"))?;
        Self::load_stages(model_dir, cfg, precision, readout, kernels, quantization, &[(device, 0..cfg.num_layers)])
    }

    /// [`GpuBackend::load_split`] without an attention policy ([`GpuBackend::load_unplanned`]).
    #[cfg_attr(
        all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")),
        expect(dead_code, reason = "an Intel-only build loads Llama through IntelBackend")
    )]
    pub(crate) fn load_split_unplanned(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
    ) -> Result<Self> {
        let stages = Self::split_stages(cfg, precision, kernels, quantization, split)?;
        Self::load_stages(model_dir, cfg, precision, readout, kernels, quantization, &stages)
    }

    /// The checked device and layer range of every stage of `split`.
    fn split_stages(
        cfg: &LlamaConfig,
        precision: Precision,
        kernels: Kernels,
        quantization: Quantization,
        split: &LayerSplit,
    ) -> Result<Vec<(usize, Range<usize>)>> {
        quantization.check(precision, kernels, cfg!(feature = "cuda"))?;
        split.check(precision, kernels, cfg!(feature = "cuda"))?;
        let ranges = split.ranges(cfg.num_layers)?;
        Ok(split.devices().into_iter().zip(ranges).collect())
    }

    /// Load the stages `plan` (device ordinal and global layer range of each, in layer order; one stage is the whole
    /// model on one GPU). Stages are filled one after another; each stage's memory belongs to its entry of a local
    /// `built` list from before its first allocation, so a failed load frees everything already loaded, each stage
    /// with its own context bound.
    fn load_stages(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        plan: &[(usize, Range<usize>)],
    ) -> Result<Self> {
        let t0 = Instant::now();
        ensure!(!plan.is_empty(), "a model needs at least one stage");
        let split = plan.len() > 1;
        let devs = plan
            .iter()
            .enumerate()
            .map(|(s, (device, _))| {
                let dev = gpu_device(*device);
                if split {
                    dev.with_context(|| format!("stage {s} (CUDA device {device})"))
                } else {
                    dev
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let kind = if devs[0].is_cuda() { "CUDA" } else { "GPU" };
        let path = model_dir.join("model.safetensors");
        // SAFETY: the checkpoint file is not modified while mapped.
        let st = unsafe { MmapedSafetensors::new(&path) }.with_context(|| format!("mapping {}", path.display()))?;
        let (h, i, v) = (cfg.hidden, cfg.intermediate, cfg.vocab);
        let kvd = cfg.kv_heads * cfg.head_dim;
        // The loader accepts exactly the tensors of this architecture: every expected tensor with its shape, and no
        // others (an unknown tensor would mean a model this code does not implement).
        let mut expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        expected.insert("model.embed_tokens.weight".into(), vec![v, h]);
        expected.insert("model.norm.weight".into(), vec![h]);
        expected.insert("lm_head.weight".into(), vec![v, h]);
        for l in 0..cfg.num_layers {
            let p = format!("model.layers.{l}");
            let qd = cfg.heads * cfg.head_dim;
            for (n, s) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![qd, h]),
                ("self_attn.q_proj.bias", vec![qd]),
                ("self_attn.k_proj.weight", vec![kvd, h]),
                ("self_attn.k_proj.bias", vec![kvd]),
                ("self_attn.v_proj.weight", vec![kvd, h]),
                ("self_attn.v_proj.bias", vec![kvd]),
                ("self_attn.o_proj.weight", vec![h, qd]),
                ("self_attn.o_proj.bias", vec![h]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.gate_proj.bias", vec![i]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("mlp.up_proj.bias", vec![i]),
                ("mlp.down_proj.weight", vec![h, i]),
                ("mlp.down_proj.bias", vec![h]),
            ] {
                if n.ends_with(".bias") && !cfg.bias {
                    continue;
                }
                expected.insert(format!("{p}.{n}"), s);
            }
        }
        for (name, shape) in &expected {
            check(&st, name, shape)?;
        }
        let present: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
        let extra: Vec<&String> = present.iter().filter(|n| !expected.contains_key(*n)).collect();
        ensure!(extra.is_empty(), "checkpoint has tensors this runtime does not implement: {extra:?}");

        // Tensors are assembled (concatenated, converted) on the host from the mapped file and uploaded once, so the
        // GPU never holds a second copy of a weight (concatenating on the device kept the parts in candle's buffer
        // pool and raised the peak footprint by the size of all gate/up weights).
        let dt = precision.storage();
        // the mapping above gives names and shapes; the data is read (see TensorReader)
        let reader = TensorReader::open(&path)?;
        let load_cat = |names: &[String], dev: &Device| -> Result<Tensor> {
            let views = names.iter().map(|n| st.get(n)).collect::<candle_core::Result<Vec<_>>>()?;
            let mut shape = views[0].shape().to_vec();
            shape[0] = views.iter().map(|v| v.shape()[0]).sum();
            let mut bytes = Vec::new();
            for n in names {
                reader.read(n, &mut bytes)?;
            }
            Ok(match dt {
                DType::BF16 => Tensor::from_raw_buffer(&bytes, DType::BF16, &shape, dev)?,
                DType::F16 => {
                    // on every core (the 11B checkpoint: ~11e9 values)
                    let mut h = vec![half::f16::ZERO; bytes.len() / 2];
                    bf16_to_f16(&bytes, &mut h);
                    Tensor::from_vec(h, shape, dev)?
                }
                d => bail!("unsupported storage dtype {d:?}"),
            })
        };
        let load = |name: &str, dev: &Device| load_cat(&[name.to_string()], dev);
        // Projection weights: dense as above, or q8_0 / q4_k blocks quantized on the host from the bf16 values and
        // uploaded once (one allocation per matrix; no dense copy on the GPU). Statistics accumulate in layer order
        // 0..L over all stages, as on one GPU.
        #[cfg(feature = "cuda")]
        let q8_stats = std::cell::RefCell::new(quant::Q8Stats::default());
        #[cfg(feature = "cuda")]
        let q4k_stats = std::cell::RefCell::new(quant::Q4KStats::default());
        // q4_k: the host blocks of one matrix are kept for the warm-up check: the projection of layer 0 with the
        // fewest values in this architecture's shapes (the first of equals, as the q8_0 warm-up picks)
        #[cfg(feature = "cuda")]
        let q4k_warm_projection = Projection::ALL
            .into_iter()
            .zip(projection_shapes(cfg))
            .min_by_key(|&(_, (n, k))| n.saturating_mul(k))
            .map(|(p, _)| p);
        // the kept matrix: its tensor, host blocks, source tensors and projection
        #[cfg(feature = "cuda")]
        let q4k_warm = std::cell::Cell::new(None);
        #[cfg_attr(
            not(feature = "cuda"),
            allow(unused_variables, reason = "only CUDA loads quantized weights and accumulates their diagnostics")
        )]
        let load_proj = |layer: usize,
                         projection: Projection,
                         names: &[String],
                         dev: &Device,
                         device: usize|
         -> Result<ProjWeight> {
            match quantization {
                Quantization::None => Ok(ProjWeight::Dense(load_cat(names, dev)?)),
                #[cfg(feature = "cuda")]
                Quantization::Q8_0 => {
                    let what = || format!("{} as q8_0 on CUDA device {device}", names.join(" + "));
                    let (q, _, stats) =
                        load_blocks(&st, &reader, names, dev, quantization, false, quant::quantize_q8_0)
                            .with_context(what)?;
                    q8_stats.borrow_mut().add(&stats);
                    Ok(ProjWeight::Quantized(Arc::new(q)))
                }
                #[cfg(feature = "cuda")]
                Quantization::Q4K => {
                    let what = || format!("{} as q4_k on CUDA device {device}", names.join(" + "));
                    let parts = source_parts(&st, names).with_context(what)?;
                    let source = quant::Source { layer, projection, parts: &parts };
                    let keep = layer == 0 && q4k_warm_projection == Some(projection);
                    let (q, kept, matrix) = load_blocks(&st, &reader, names, dev, quantization, keep, |b, r, c| {
                        quant::quantize_q4_k(b, r, c, &source)
                    })
                    .with_context(what)?;
                    quant::check_q4k_resident(&q, matrix.blocks()).with_context(what)?;
                    q4k_stats.borrow_mut().add(&matrix).with_context(what)?;
                    let q = Arc::new(q);
                    if let Some(blocks) = kept {
                        q4k_warm.set(Some((Arc::clone(&q), blocks, parts, projection)));
                    }
                    Ok(ProjWeight::Quantized(q))
                }
                #[cfg(not(feature = "cuda"))]
                Quantization::Q8_0 | Quantization::Q4K => bail!("quantization {quantization} needs the CUDA backend"),
            }
        };
        // Models without biases (basal-1.5) get zero biases: the fused kernels then add exactly 0, which leaves every
        // rounding point as without a bias (T(T(y) + 0) = T(y)).
        let biases = |names: &[String], len: usize, dev: &Device| -> Result<Tensor> {
            if cfg.bias {
                load_cat(names, dev)
            } else {
                Ok(Tensor::zeros(len, dt, dev)?)
            }
        };
        let qd = cfg.heads * cfg.head_dim;
        let load_layer = |l: usize, dev: &Device, device: usize| -> Result<Layer> {
            let p = |n: &str| format!("model.layers.{l}.{n}");
            let ps = |ns: &[&str]| ns.iter().map(|n| p(n)).collect::<Vec<_>>();
            Ok(Layer {
                ln1: load(&p("input_layernorm.weight"), dev)?,
                wqkv: load_proj(
                    l,
                    Projection::Qkv,
                    &ps(&["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]),
                    dev,
                    device,
                )?,
                bqkv: biases(
                    &ps(&["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"]),
                    qd + 2 * kvd,
                    dev,
                )?,
                wo: load_proj(l, Projection::O, &ps(&["self_attn.o_proj.weight"]), dev, device)?,
                bo: biases(&ps(&["self_attn.o_proj.bias"]), h, dev)?,
                ln2: load(&p("post_attention_layernorm.weight"), dev)?,
                wgu: load_proj(
                    l,
                    Projection::GateUp,
                    &ps(&["mlp.gate_proj.weight", "mlp.up_proj.weight"]),
                    dev,
                    device,
                )?,
                bgu: biases(&ps(&["mlp.gate_proj.bias", "mlp.up_proj.bias"]), 2 * i, dev)?,
                wdown: load_proj(l, Projection::Down, &ps(&["mlp.down_proj.weight"]), dev, device)?,
                bdown: biases(&ps(&["mlp.down_proj.bias"]), h, dev)?,
            })
        };
        let lineage = Lineage::default();
        let last = plan.len() - 1;
        let mut built: Vec<OnStage<StageWeights>> = Vec::with_capacity(plan.len());
        // Fill the stages in order. Filling stage `s` binds only its context; its temporaries die inside, with it
        // current. On one GPU this is today's order: layers, embedding, final norm, lm_head, warm-ups.
        let filled =
            (|| -> Result<()> {
                for (s, ((device, range), dev)) in plan.iter().zip(&devs).enumerate() {
                    let device = *device;
                    (|| -> Result<()> {
                        if split {
                            match bind_twice(dev) {
                                Twice::First => {}
                                Twice::Second(e) => bail!("binding the context: first attempt failed ({e})"),
                                Twice::Neither(a, b) => bail!("binding the context failed twice ({a}; {b})"),
                            }
                        }
                        built.push(OnStage::new(dev, true, &lineage.health, StageWeights::default()));
                        let w = built.last_mut().context("the stage was just added")?.get_mut();
                        for l in range.clone() {
                            let layer = load_layer(l, dev, device)
                                .with_context(|| format!("layer {l} of stage {s} on {kind} device {device}"))?;
                            w.layers.push(layer);
                        }
                        if s == 0 {
                            w.embed = Some(load("model.embed_tokens.weight", dev)?);
                        }
                        if s < last {
                            // the next stage's first input_layernorm, converted by the same code: the same bits
                            w.next_ln1 =
                                Some(load(&format!("model.layers.{}.input_layernorm.weight", range.end), dev)?);
                        }
                        if s == last {
                            let norm = load("model.norm.weight", dev)?;
                            // lm_head stays in the checkpoint dtype; only the letter rows are gathered and converted to
                            // f32 per call.
                            let lm_head = {
                                let v = st.get("lm_head.weight")?;
                                let mut bytes = Vec::new();
                                reader.read("lm_head.weight", &mut bytes)?;
                                Tensor::from_raw_buffer(&bytes, DType::BF16, v.shape(), dev)?
                            };
                            w.head = Some(Head { norm, lm_head, evidence: None, exits: None });
                        }
                        // q8_0: load Candle's QUANTIZED module and its dequantization kernel now (a first use may
                        // compile the module's PTX), by dequantizing the smallest projection matrix of the stage's first
                        // layer once and discarding the result. Queued before the synchronisation below, like every
                        // upload: the engines run on the per-thread streams of other threads and event tracking is off
                        // (gpu_device), so that synchronisation is what makes the blocks visible to them. Stages after
                        // the first do this for q4_k too (the q4_k check below runs on stage 0 only).
                        #[cfg(feature = "cuda")]
                        if quantization == Quantization::Q8_0 || (s > 0 && quantization == Quantization::Q4K) {
                            if let Some(ProjWeight::Quantized(q)) = w.layers.first().and_then(|l| {
                                [&l.wqkv, &l.wo, &l.wgu, &l.wdown].into_iter().min_by_key(|w| w.elem_count())
                            }) {
                                drop(q.dequantize_f16(&q.device()).with_context(|| {
                                    format!("{quantization}: warming up the dequantization kernel")
                                })?);
                            }
                        }
                        // q4_k: the same warm-up on the kept matrix (layer 0), plus a control: its f16 result copied back
                        // to the host and compared bitwise (signed zeros included) with the host reconstruction of the
                        // uploaded blocks. A difference is a load error. The kept host blocks are released afterwards;
                        // the other matrices kept none.
                        #[cfg(feature = "cuda")]
                        if s == 0 && quantization == Quantization::Q4K {
                            let (q, blocks, parts, projection) =
                                q4k_warm.take().context("q4_k: no matrix was kept for the warm-up check")?;
                            let source = quant::Source { layer: 0, projection, parts: &parts };
                            let values = quant::check_q4k_warmup(&q, &blocks, &source)
                                .with_context(|| format!("q4_k warm-up check on CUDA device {device}"))?;
                            q4k_stats.borrow_mut().set_warmup(0, projection, values);
                        }
                        Ok(())
                    })()
                    .with_context(|| format!("stage {s} (layers {}–{})", range.start, range.end.saturating_sub(1)))?;
                }
                for (s, dev) in devs.iter().enumerate() {
                    dev.synchronize().with_context(|| format!("synchronising stage {s} after the upload"))?;
                }
                Ok(())
            })();
        if let Err(e) = filled {
            // q4_k: a matrix of stage 0 kept for its warm-up check (the fill failed in stage 0, still current)
            #[cfg(feature = "cuda")]
            drop(q4k_warm.take());
            return Err(release_built(built, e));
        }
        let d = cfg.head_dim;
        let inv_freq: Vec<f64> = (0..d / 2).map(|j| 1.0 / cfg.rope_theta.powf((2 * j) as f64 / d as f64)).collect();
        let mut rope = vec![0f32; cfg.max_positions * d];
        for (pos, row) in rope.chunks_exact_mut(d).enumerate() {
            let (cos, sin) = row.split_at_mut(d / 2);
            rope_angles(&inv_freq, pos as f64, cos, sin);
        }
        let load_s = t0.elapsed().as_secs_f64();
        // The evidence head (last stage) and a cuBLASLt handle with its workspace per stage, each after binding its
        // stage, then the publication barrier: these allocations were queued on this thread's streams after the
        // synchronisation above, and the engines of other threads read them only after this synchronisation.
        let published = (|| -> Result<()> {
            let p = model_dir.join("evidence_head.pt");
            if p.exists() {
                if split {
                    match bind_twice(&devs[last]) {
                        Twice::First => {}
                        Twice::Second(e) => {
                            bail!("binding stage {last} for the evidence head: first attempt failed ({e})")
                        }
                        Twice::Neither(a, b) => {
                            bail!("binding stage {last} for the evidence head failed twice ({a}; {b})")
                        }
                    }
                }
                let head = EvidenceHead::load(&p, cfg.hidden, &devs[last])?;
                built[last].get_mut().head.as_mut().context("the last stage holds the head")?.evidence = Some(head);
            }
            #[cfg(feature = "cuda")]
            for (s, dev) in devs.iter().enumerate() {
                if let (Device::Cuda(d), false) =
                    (dev, crate::plan::DiagnosticOverrides::process().gemm() == Some("cublas"))
                {
                    if split {
                        match bind_twice(dev) {
                            Twice::First => {}
                            Twice::Second(e) => {
                                bail!("binding stage {s} for its cuBLASLt handle: first attempt failed ({e})")
                            }
                            Twice::Neither(a, b) => {
                                bail!("binding stage {s} for its cuBLASLt handle failed twice ({a}; {b})")
                            }
                        }
                    }
                    let lt = cublaslt::Lt::new(d).with_context(|| format!("cuBLASLt handle of stage {s}"))?;
                    built[s].get_mut().lt = Some(Arc::new(lt));
                }
            }
            for (s, dev) in devs.iter().enumerate() {
                dev.synchronize().with_context(|| format!("publishing stage {s} (synchronising after the load)"))?;
            }
            Ok(())
        })();
        if let Err(e) = published {
            return Err(release_built(built, e));
        }
        let stages: Vec<Stage> = plan
            .iter()
            .zip(devs)
            .zip(built)
            .map(|(((device, range), dev), w)| Stage { dev, device: *device, layers: range.clone(), w, gate: None })
            .collect();
        let transfers = split.then(|| Arc::new(TransferStats::new(&stages)));
        Ok(Self {
            stages,
            lineage,
            transfers,
            prefixes: Vec::new(),
            generation: None,
            state_cache_bytes: 0,
            clock: 0,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            cfg: cfg.clone(),
            precision,
            readout,
            kernels,
            attention_override: None,
            attention_defaults: Vec::new(),
            numerics: LlamaNumerics::Ordinary,
            inv_freq,
            rope: rope.into(),
            load_s,
            profile: None,
            quantization,
            #[cfg(feature = "cuda")]
            quant_stats: match quantization {
                Quantization::None => None,
                Quantization::Q8_0 => Some(quant::QuantStats::Q8(q8_stats.into_inner())),
                Quantization::Q4K => Some(quant::QuantStats::Q4K(Box::new(q4k_stats.into_inner()))),
            },
            dequant: std::cell::Cell::new((0.0, 0)),
            residual_max: None,
            last_timing: (0.0, 0.0),
            stop: None,
        })
    }

    /// CUDA attention kernel of this forward precision.
    fn attention_kernel(&self) -> Result<&'static str> {
        #[cfg(feature = "cuda")]
        if self.stages[0].dev.is_cuda() {
            return if self.tc_attention() { self.tc_kernel_of(0) } else { Ok("attn_tree_f32") };
        }
        Ok(if self.tree_attention() { "attn_tree_f32" } else { "sdpa" })
    }

    /// The model runs on CUDA (every stage: a split is CUDA only).
    fn is_cuda(&self) -> bool {
        self.stages[0].dev.is_cuda()
    }

    /// The model is split over several GPUs.
    fn split(&self) -> bool {
        self.stages.len() > 1
    }

    /// Readout weights (final norm, lm_head, evidence head) of the last stage.
    fn head(&self) -> Result<&Head> {
        self.stages.last().and_then(|s| s.w.get().head.as_ref()).context("the last stage holds the head")
    }

    /// Owners of this call's stage gates (see [`StageGates`]).
    fn gate_owners(&self) -> GateOwners {
        GateOwners {
            stop: self.stop.clone(),
            gates: self.stages.iter().map(|s| s.gate.clone()).collect(),
            devs: self.stages.iter().map(|s| s.dev.clone()).collect(),
            health: self.lineage.health.clone(),
            split: self.split(),
        }
    }

    /// Attention by tree units, invariant to packing: always on CUDA; on Metal unless `BASAL_ATT=sdpa` (the MLX SDPA
    /// kernel over each packed row with an additive mask, the earlier path, kept for comparison).
    fn tree_attention(&self) -> bool {
        self.is_cuda() || crate::plan::DiagnosticOverrides::process().att() != Some("sdpa")
    }

    /// Profiling: synchronise `dev` (the GPU of the running stage) and attribute the time since `*t` to `section`.
    fn mark(&self, dev: &Device, t: &mut Instant, section: &'static str) -> Result<()> {
        if let Some(p) = &self.profile {
            dev.synchronize()?;
            let now = Instant::now();
            *p.borrow_mut().entry(section).or_insert(0.0) += (now - *t).as_secs_f64() * 1e3;
            *t = now;
        }
        Ok(())
    }

    /// Weight in the compute dtype (a no-op unless weights are widened to f32 per use).
    fn w(&self, t: &Tensor) -> Result<Tensor> {
        Ok(t.to_dtype(self.precision.compute())?)
    }

    fn linear(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
        Ok(x.matmul(&self.w(w)?.t()?)?.broadcast_add(&self.w(b)?)?)
    }

    /// `x @ w^T` with the weights of stage `sw`. candle tiles M by 64 rows; when the last row tile would be at most
    /// half used (M mod 64 in 1..=32, small M) 32-row tiles of the same MLX kernel family skip that padding (measured
    /// with `basal gemm-tune`; results are bitwise identical, the K reduction order does not change).
    /// On CUDA: the stage's cuBLASLt handle with measured algorithms (f16/bf16), otherwise cuBLAS through candle.
    /// `q8_0` / `q4_k`: the weight is dequantized into a temporary f16 tensor of this call (freed on this thread's
    /// stream) and multiplied like a dense f16 weight.
    fn matmul_t(&self, sw: &StageWeights, x: &Tensor, w: &ProjWeight) -> Result<Tensor> {
        let w = match w {
            ProjWeight::Dense(w) => self.w(w)?,
            #[cfg(feature = "cuda")]
            ProjWeight::Quantized(q) => self.dequantize_blocks(q)?,
        };
        #[cfg(feature = "cuda")]
        if let (Some(lt), DType::F16 | DType::BF16) = (&sw.lt, w.dtype()) {
            if x.is_contiguous() {
                return Ok(cublaslt::matmul_nt(lt, x, &w)?);
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = sw;
        #[cfg(target_os = "macos")]
        {
            let m = x.dim(0)?;
            if self.stages[0].dev.is_metal() && m <= 600 && (1..=32).contains(&(m % 64)) && x.is_contiguous() {
                return Ok(gemm::gemm_nt(x, &w, gemm::GemmConfig::new((32, 64, 16, 1, 2), 0))?);
            }
        }
        Ok(x.matmul(&w.t()?)?)
    }

    /// Ordinary leaves the bias for the fused consumer. Thinking uses the stage's existing, bias-only Lt;
    /// the bias epilogue precedes the final BF16 store (heuristic split-K reduction remains deferred).
    fn project<'b>(
        &self,
        sw: &StageWeights,
        x: &Tensor,
        w: &ProjWeight,
        b: &'b Tensor,
    ) -> Result<(Tensor, Option<&'b Tensor>)> {
        if self.numerics == LlamaNumerics::Ordinary {
            return Ok((self.matmul_t(sw, x, w)?, Some(b)));
        }
        #[cfg(feature = "cuda")]
        {
            ensure!(x.is_contiguous(), "thinking projection requires contiguous input");
            let ProjWeight::Dense(w) = w else { bail!("thinking projection requires dense weights") };
            let lt = sw.lt.as_ref().context("thinking projection requires the stage's cuBLASLt handle")?;
            Ok((cublaslt::matmul_nt_bias(lt, x, w, b)?, None))
        }
        #[cfg(not(feature = "cuda"))]
        bail!("thinking BF16 projections require CUDA")
    }

    /// f16 values of resident Q8_0 or Q4_K blocks, on the blocks' own candle device (their stage's, checked at load).
    /// While profiling, that device is synchronised before and after and the time added to `dequant` (a part of the
    /// projection sections).
    #[cfg(feature = "cuda")]
    fn dequantize_blocks(&self, q: &candle_core::quantized::QTensor) -> Result<Tensor> {
        if self.profile.is_none() {
            return Ok(q.dequantize_f16(&q.device())?);
        }
        let dev = q.device();
        dev.synchronize()?;
        let t = Instant::now();
        let w = q.dequantize_f16(&dev)?;
        dev.synchronize()?;
        let (ms, values) = self.dequant.get();
        self.dequant.set((ms + t.elapsed().as_secs_f64() * 1e3, values + q.shape().elem_count() as u64));
        Ok(w)
    }

    /// Quantized projection weights: their format, the synchronised time (ms) of the projection dequantizations since
    /// profiling started (already included in the projection sections of `profile`) and the number of weight values
    /// dequantized; `None` without quantized weights.
    pub fn dequant_profile(&self) -> Option<(Quantization, f64, u64)> {
        (self.quantization != Quantization::None).then(|| {
            let (ms, values) = self.dequant.get();
            (self.quantization, ms, values)
        })
    }

    /// f32 attention of one segment: `q [1, NH, L, HD]`, `k, v [1, NKV, LK, HD]`, additive mask `[1, 1, L, LK]`
    /// shared by the heads. Metal: candle's MLX SDPA kernel (GQA in the kernel). CUDA: scores GEMM, one fused
    /// scale + mask + softmax kernel and the weighted-sum GEMM, with the query heads of one KV head stacked as rows
    /// (no K/V repeat).
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f32) -> Result<Tensor> {
        let (_, nh, l, hd) = q.dims4()?;
        #[cfg(target_os = "macos")]
        if self.stages[0].dev.is_metal() {
            let m = mask.broadcast_as((1, nh, l, k.dim(2)?))?;
            return Ok(sdpa(q, k, v, Some(&m), false, scale, 1.0)?);
        }
        let rep = nh / k.dim(1)?;
        let nkv = nh / rep;
        let q = q.reshape((1, nkv, rep * l, hd))?;
        let att = fused::masked_softmax(&q.matmul(&k.t()?)?, mask, l, scale)?;
        Ok(att.matmul(v)?.reshape((1, nh, l, hd))?)
    }

    /// Stock-candle forward (first port, `--kernels candle`, one GPU only) on rows padded to `l`: final normalised
    /// hidden states at `read` (flat indices into `[b * l]`). `cos_sin` = `[2, b, l, hd/2]`, `mask` = additive
    /// `[b, 1, l, l]` f32.
    fn forward_padded(
        &self,
        ids: &[u32],
        cos_sin: &[f32],
        mask: &[f32],
        b: usize,
        l: usize,
        read: &[u32],
    ) -> Result<Tensor> {
        match self.forward_padded_exit(ids, cos_sin, mask, b, l, read, None)? {
            early_exit::PaddedOutput::Hidden(hidden) => Ok(hidden),
            early_exit::PaddedOutput::Exit(_) => bail!("unexpected trained exit on ordinary forward"),
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "existing padded input shape plus an optional request-local exit callback"
    )]
    fn forward_padded_exit(
        &self,
        ids: &[u32],
        cos_sin: &[f32],
        mask: &[f32],
        b: usize,
        l: usize,
        read: &[u32],
        mut exit: Option<&mut early_exit::ExitCheck<'_>>,
    ) -> Result<early_exit::PaddedOutput> {
        ensure!(self.stages.len() == 1, "the candle kernel set runs a model on one GPU only");
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
        let eps = c.rms_eps as f32;
        let dt = self.precision.compute();
        let st = &self.stages[0];
        let sw = st.w.get();
        let dev = &st.dev;
        let embed = sw.embed.as_ref().context("stage 0 holds the embedding")?;
        let head = self.head()?;
        let ids = Tensor::from_slice(ids, (b * l,), dev)?;
        let cs = Tensor::from_slice(cos_sin, (2 * b * l * (hd / 2),), dev)?;
        let mask = Tensor::from_slice(mask, (b, 1, l, l), dev)?;
        let read_t = Tensor::from_slice(read, (read.len(),), dev)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut t = Instant::now();
        let mut x = embed.index_select(&ids, 0)?.to_dtype(dt)?; // [b*l, h]
        self.mark(dev, &mut t, "embed+inputs")?;
        {
            {
                let rep = nh / nkv;
                let kvd = nkv * hd;
                let half = b * l * (hd / 2);
                let cos = cs.narrow(0, 0, half)?.reshape((b, l, hd / 2))?;
                let sin = cs.narrow(0, half, half)?.reshape((b, l, hd / 2))?;
                let inter = c.intermediate;
                for (layer, ly) in sw.layers.iter().enumerate() {
                    let a = rms_norm(&x, &self.w(&ly.ln1)?, eps)?;
                    self.mark(dev, &mut t, "rms_norm")?;
                    let qkv = self.linear(&a, ly.wqkv.dense()?, &ly.bqkv)?;
                    self.mark(dev, &mut t, "qkv_proj")?;
                    let heads = |off: usize, n: usize| -> Result<Tensor> {
                        Ok(qkv
                            .narrow(1, off, n * hd)?
                            .contiguous()?
                            .reshape((b, l, n, hd))?
                            .transpose(1, 2)?
                            .contiguous()?
                            .to_dtype(DType::F32)?)
                    };
                    let q = rope(&heads(0, nh)?, &cos, &sin)?;
                    let k = rope(&heads(h, nkv)?, &cos, &sin)?;
                    let v = heads(h + kvd, nkv)?;
                    self.mark(dev, &mut t, "bias+rope+heads")?;
                    // GQA without repeating K/V: query head j uses kv head j / rep (as HF repeat_kv).
                    let q = (q * scale)?.reshape((b, nkv, rep * l, hd))?;
                    let att = q.matmul(&k.t()?)?.reshape((b, nh, l, l))?.broadcast_add(&mask)?;
                    let att = softmax_last_dim(&att)?.reshape((b, nkv, rep * l, l))?;
                    let o = att
                        .matmul(&v)?
                        .reshape((b, nh, l, hd))?
                        .transpose(1, 2)?
                        .contiguous()?
                        .reshape((b * l, h))?
                        .to_dtype(dt)?;
                    self.mark(dev, &mut t, "attention")?;
                    x = (x + self.linear(&o, ly.wo.dense()?, &ly.bo)?)?;
                    self.mark(dev, &mut t, "o_proj+bias+residual")?;
                    let a = rms_norm(&x, &self.w(&ly.ln2)?, eps)?;
                    self.mark(dev, &mut t, "rms_norm")?;
                    let wgu = ly.wgu.dense()?;
                    let g = self.linear(&a, &wgu.narrow(0, 0, inter)?, &ly.bgu.narrow(0, 0, inter)?)?;
                    let u = self.linear(&a, &wgu.narrow(0, inter, inter)?, &ly.bgu.narrow(0, inter, inter)?)?;
                    self.mark(dev, &mut t, "gate_up_proj")?;
                    let act = (g.silu()? * u)?;
                    self.mark(dev, &mut t, "bias+silu*up")?;
                    x = (x + self.linear(&act, ly.wdown.dense()?, &ly.bdown)?)?;
                    self.mark(dev, &mut t, "down_proj+bias+residual")?;
                    if let Some(check) = exit.as_mut() {
                        if let Some(logits) = check(layer + 1, &x.index_select(&read_t, 0)?)? {
                            return Ok(early_exit::PaddedOutput::Exit(logits));
                        }
                    }
                }
                let out = rms_norm(&x.index_select(&read_t, 0)?, &self.w(&head.norm)?, eps)?;
                self.mark(dev, &mut t, "final_norm")?;
                Ok(early_exit::PaddedOutput::Hidden(out))
            }
        }
    }

    /// Fused forward on a compact token list: the tokens of all segments back to back (no padding), so every GEMM,
    /// norm and MLP runs on exactly the real tokens; attention runs per segment over its own keys (optionally after
    /// a precomputed prefix). `cos_sin` = `[2, total, hd/2]`. The stages run in order, each entered through `gates`;
    /// a split moves the activations at every boundary (`GpuBackend::transfer`, CUDA). Returns the final normalised
    /// hidden states at `read` (indices into the compact list) on the last stage's GPU, that stage entered and its
    /// context current; with `capture` (`Some(bind)`: one segment), also the K/V of every layer as one part per stage.
    /// Capture without readout skips the last layer's output as before. Nothing binds another stage's context after
    /// the last stage's body returned.
    fn forward_ragged(
        &self,
        gates: &mut StageGates<'_>,
        ids: &[u32],
        cos_sin: &[f32],
        segs: &[Segment],
        read: &[u32],
        capture: Option<bool>,
    ) -> Result<(Option<Tensor>, Vec<OnStage<StageKv>>)> {
        let c = &self.cfg;
        let dims = HeadDims { b: 1, l: ids.len(), nh: c.heads, nkv: c.kv_heads, hd: c.head_dim };
        ensure!(capture.is_none() || segs.len() == 1, "prefix capture takes one segment");
        // CUDA: the blocks of all rows as tree-attention units (compact offsets; prefix addresses set per layer).
        #[cfg(feature = "cuda")]
        let units: Vec<(fused::TreeUnit, Option<&PrefixKv>)> = if self.is_cuda() {
            segs.iter()
                .flat_map(|sg| {
                    sg.units.iter().map(move |u| {
                        let tu = fused::TreeUnit {
                            np: sg.np(),
                            q_off: sg.off + u.q_off,
                            q_len: u.q_len,
                            ranges: u.ranges.iter().map(|&(o, l)| (sg.off + o, l)).collect(),
                            ..Default::default()
                        };
                        (tu, sg.past)
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        #[cfg(target_os = "macos")]
        let (metal_units, metal_pasts) = if self.stages[0].dev.is_metal() && self.tree_attention() {
            let mut pasts: Vec<&PrefixKv> = Vec::new();
            let mut units = Vec::new();
            for sg in segs {
                let past = sg.past.map(|p| match pasts.iter().position(|q| std::ptr::eq(*q, p)) {
                    Some(i) => i,
                    None => {
                        pasts.push(p);
                        pasts.len() - 1
                    }
                });
                units.extend(sg.units.iter().map(|u| fused::MetalTreeUnit {
                    past,
                    q_off: sg.off + u.q_off,
                    q_len: u.q_len,
                    ranges: u.ranges.iter().map(|&(o, l)| (sg.off + o, l)).collect(),
                }));
            }
            (units, pasts)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut f = Ragged {
            ids,
            cos_sin,
            segs,
            read,
            capture,
            dims,
            #[cfg(feature = "cuda")]
            units,
            #[cfg(target_os = "macos")]
            metal_units,
            #[cfg(target_os = "macos")]
            metal_pasts,
        };
        let mut parts = Vec::with_capacity(self.stages.len());
        let mut input = None;
        let mut t = Instant::now();
        for s in 0..self.stages.len() {
            gates.switch_to(s)?;
            match self.run_stage(gates, s, input.take(), &mut f, &mut t, &mut parts)? {
                StageOut::Final(out) => return Ok((out, parts)),
                StageOut::Boundary(buf) => input = Some(self.cross(gates, s, buf, &mut t)?),
            }
        }
        bail!("the last stage returned no output")
    }

    /// The boundary after stage `s`: `GpuBackend::transfer` (CUDA; profiled as `stage_transfer`). Returns the
    /// input of stage `s + 1`, entered.
    fn cross(&self, gates: &mut StageGates<'_>, s: usize, buf: Tensor, t: &mut Instant) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let input = self.transfer(gates, s, buf)?;
            self.mark(&self.stages[s + 1].dev, t, "stage_transfer")?;
            Ok(input)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (gates, s, buf, t);
            bail!("a layer split needs the CUDA backend")
        }
    }

    /// The body of stage `s`, entered with its gate held and (split) its context current: the stage's uploads, its
    /// layers and, in capture mode, the hi/lo planes of its K/V. Every temporary tensor is a local of this call and
    /// dies in it (also on `Err` and while unwinding) with the stage's context current, since the body binds no other
    /// context and a failed bind leaves the current one unchanged. Only the boundary buffer, the captured part (owned
    /// by its `OnStage`) and the last stage's final hidden states leave it.
    fn run_stage(
        &self,
        gates: &mut StageGates<'_>,
        s: usize,
        input: Option<Tensor>,
        f: &mut Ragged<'_, '_>,
        t: &mut Instant,
        parts: &mut Vec<OnStage<StageKv>>,
    ) -> Result<StageOut> {
        let (out, captured) = self.stage_layers(gates, s, input, f, t)?;
        if let Some(bind) = f.capture {
            // under this stage's gate, after its layers' temporaries were freed (as on one GPU, where the planes were
            // split after the forward)
            let hl = self.split_kv(&captured)?;
            parts.push(OnStage::new(&self.stages[s].dev, bind, &self.lineage.health, StageKv { kv: captured, hl }));
        }
        Ok(out)
    }

    /// Uploads and layers of stage `s` ([`GpuBackend::run_stage`]); in capture mode also the stage's raw K/V.
    fn stage_layers(
        &self,
        gates: &mut StageGates<'_>,
        s: usize,
        input: Option<Tensor>,
        f: &mut Ragged<'_, '_>,
        t: &mut Instant,
    ) -> Result<(StageOut, LayerKv)> {
        let st = &self.stages[s];
        let sw = st.w.get();
        let dev = &st.dev;
        let c = &self.cfg;
        let hd = c.head_dim;
        let total = f.ids.len();
        let eps = c.rms_eps as f32;
        let dt = self.precision.compute();
        let last_stage = s + 1 == self.stages.len();
        let capture = f.capture.is_some();
        let n_layers = c.num_layers;
        let ids = if s == 0 { Some(Tensor::from_slice(f.ids, (total,), dev)?) } else { None };
        let cs = Tensor::from_slice(f.cos_sin, (2 * total * (hd / 2),), dev)?;
        // CUDA: one attention kernel per layer for all segments, masks in one buffer; also for a prefix capture, so
        // that a cached prefix has bitwise the K/V the same tokens get when computed inside a row.
        let ragged_cuda = cfg!(feature = "cuda") && dev.is_cuda();
        // Metal: the same tree units through attn_tree_f32 of kernels.metal (one dispatch per prefix)
        let ragged_metal = cfg!(target_os = "macos") && dev.is_metal() && self.tree_attention();
        let masks: Vec<Tensor> = if ragged_cuda || ragged_metal {
            Vec::new()
        } else {
            f.segs
                .iter()
                .map(|sg| {
                    let lk = sg.np() + sg.len;
                    Ok(Tensor::from_slice(&sg.mask, (1, 1, sg.len, lk), dev)?)
                })
                .collect::<Result<_>>()?
        };
        // A zero-length device buffer cannot be created on Metal; capture-only forwards have no readout.
        let read_t = if last_stage && !f.read.is_empty() {
            Some(Tensor::from_slice(f.read, (f.read.len(),), dev)?)
        } else {
            None
        };
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let mut captured = Vec::with_capacity(if capture { sw.layers.len() } else { 0 });
        // CUDA: the RMSNorm of the next block (after the last block: the final norm) comes out of the residual add of
        // the previous one (one kernel); after a boundary it arrives with the residual stream
        let (mut x, mut normed) = match input {
            None => {
                *t = Instant::now();
                let ids = ids.as_ref().context("stage 0 reads the token ids")?;
                let embed = sw.embed.as_ref().context("stage 0 holds the embedding")?;
                let x = embed.index_select(ids, 0)?.to_dtype(dt)?; // [total, h]
                self.mark(dev, t, "embed+inputs")?;
                (x, None)
            }
            Some(buf) => (buf.get(0)?, Some(buf.get(1)?)),
        };
        let mut boundary = None;
        for (i, ly) in sw.layers.iter().enumerate() {
            let li = st.layers.start + i;
            let a = match normed.take() {
                Some(a) => a,
                None => rms_norm(&x, &self.w(&ly.ln1)?, eps)?,
            };
            self.mark(dev, t, "rms_norm")?;
            let (qkv, bias) = self.project(sw, &a, &ly.wqkv, &ly.bqkv)?;
            self.mark(dev, t, "qkv_proj")?;
            let bias = bias.map(|b| self.w(b)).transpose()?;
            // CUDA tensor-core attention: K and V come split into hi / lo planes from the same kernel
            #[cfg(feature = "cuda")]
            let (_flat, q, k, v, hl) = if ragged_cuda && self.tc_attention() {
                let v_lo = self.precision != Precision::F16;
                let (fl, q, k, v, hl) =
                    fused::qkv_rope_hilo(&qkv, bias.as_ref(), &cs, f.dims, v_lo, capture, self.numerics)?;
                (fl, q, k, v, Some(hl))
            } else {
                let (fl, q, k, v) = fused::qkv_rope(&qkv, bias.as_ref(), &cs, f.dims, self.numerics)?;
                (fl, q, k, v, None)
            };
            #[cfg(not(feature = "cuda"))]
            let (_flat, q, k, v) = fused::qkv_rope(&qkv, bias.as_ref(), &cs, f.dims, self.numerics)?;
            drop(bias);
            self.mark(dev, t, "bias+rope+heads")?;
            #[cfg(feature = "cuda")]
            let ragged_o = if ragged_cuda {
                // f16/bf16 forwards: tensor cores (BASAL_ATT=simt keeps the f32 SIMT kernel, for A/B); the f32
                // reference forward always uses the f32 kernel.
                let tc = self.tc_attention();
                let split = self.split();
                // the kernel reads prefix K/V by raw address: with a split, they must lie on this stage's GPU
                let here = |p: &Tensor| -> Result<()> {
                    if split {
                        ensure!(
                            p.device().location() == dev.location(),
                            "prefix K/V of layer {li} on {:?}, stage {s} runs on {:?}",
                            p.device().location(),
                            dev.location()
                        );
                    }
                    Ok(())
                };
                for (u, past) in f.units.iter_mut() {
                    if let Some(p) = past {
                        let pp = p.part(s)?;
                        let (pk, pv) = &pp.kv[i];
                        here(pk)?;
                        here(pv)?;
                        u.pk = fused::device_addr_f32(pk)?;
                        u.pv = fused::device_addr_f32(pv)?;
                        if tc {
                            let (pkh, pvh) = &pp.hl[i];
                            here(pkh)?;
                            here(pvh)?;
                            u.pkh = fused::device_addr_f16(pkh)?;
                            u.pvh = fused::device_addr_f16(pvh)?;
                        }
                    }
                }
                let us: Vec<fused::TreeUnit> = f.units.iter().map(|(u, _)| u.clone()).collect();
                if let Some(hl) = &hl {
                    let nk = c.kv_heads * total * hd;
                    let (khl, vhl) = (hl.narrow(0, 0, 2 * nk)?, hl.narrow(0, 2 * nk, 2 * nk)?);
                    let v_exact = self.precision == Precision::F16;
                    Some(fused::attention_tree(
                        &_flat,
                        &us,
                        f.dims,
                        scale,
                        Some((&khl, &vhl, v_exact)),
                        Some(dt),
                        self.tc_kernel_of(s)?,
                    )?)
                } else {
                    Some(fused::attention_tree(&_flat, &us, f.dims, scale, None, Some(dt), self.tc_kernel_of(s)?)?)
                }
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let ragged_o: Option<Tensor> = None;
            #[cfg(target_os = "macos")]
            let ragged_o = if ragged_metal {
                let pasts: Vec<(&Tensor, &Tensor)> = f
                    .metal_pasts
                    .iter()
                    .map(|p| p.part(s).map(|pp| (&pp.kv[i].0, &pp.kv[i].1)))
                    .collect::<Result<_>>()?;
                Some(fused::attention_tree_metal(&_flat, &f.metal_units, &pasts, f.dims, scale)?)
            } else {
                ragged_o
            };
            let mut outs = Vec::with_capacity(f.segs.len());
            for (si, sg) in f.segs.iter().enumerate() {
                if ragged_o.is_some() && !capture {
                    break;
                }
                let one = f.segs.len() == 1;
                let part = |x: &Tensor| -> Result<Tensor> {
                    Ok(if one { x.clone() } else { x.narrow(2, sg.off, sg.len)?.contiguous()? })
                };
                let (qs, mut ks, mut vs) = (part(&q)?, part(&k)?, part(&v)?);
                if let Some(p) = sg.past {
                    let (pk, pv) = &p.part(s)?.kv[i];
                    ks = Tensor::cat(&[pk, &ks], 2)?;
                    vs = Tensor::cat(&[pv, &vs], 2)?;
                }
                if capture {
                    captured.push((ks.contiguous()?, vs.contiguous()?));
                    if li + 1 == n_layers && f.read.is_empty() {
                        // a captured prefix needs no output of the last layer
                        return Ok((StageOut::Final(None), captured));
                    }
                }
                if ragged_o.is_none() {
                    outs.push(self.attention(&qs, &ks, &vs, &masks[si], scale)?);
                }
            }
            let o = match ragged_o {
                Some(o) => o,
                None if outs.len() == 1 => outs.pop().unwrap(),
                None => Tensor::cat(&outs, 2)?,
            };
            // the CUDA tree attention returns merged heads [total, NH * HD] in the forward's dtype
            let mut o = if ragged_cuda { o } else { fused::merge_heads(&o, f.dims, dt)? };
            self.mark(dev, t, "attention")?;
            if let (true, Some(rt)) = (li + 1 == n_layers, &read_t) {
                // Only the readout positions are needed after the last attention.
                o = o.index_select(rt, 0)?;
                x = x.index_select(rt, 0)?;
            }
            let (y, bias) = self.project(sw, &o, &ly.wo, &ly.bo)?;
            let a;
            #[cfg(feature = "cuda")]
            if self.numerics == LlamaNumerics::ThinkBf16 {
                (x, a) = fused::residual_rmsnorm_mid(&x, &y, &self.w(&ly.ln2)?, eps)?;
            } else {
                (x, a) = self.residual_norm(dev, &x, &y, bias, &ly.ln2, eps)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                (x, a) = self.residual_norm(dev, &x, &y, bias, &ly.ln2, eps)?;
            }
            self.mark(dev, t, "o_proj+bias+residual+rms_norm")?;
            let (gu, bias) = self.project(sw, &a, &ly.wgu, &ly.bgu)?;
            self.mark(dev, t, "gate_up_proj")?;
            let act = {
                let bias = bias.map(|b| self.w(b)).transpose()?;
                fused::bias_silu_mul(&gu, bias.as_ref(), self.numerics)?
            };
            self.mark(dev, t, "bias+silu*up")?;
            let (y, bias) = self.project(sw, &act, &ly.wdown, &ly.bdown)?;
            if dev.is_cuda() {
                // the next block's RMSNorm (at a boundary: of the next stage's first block, through its copy
                // `next_ln1`, the same values), after the last block the final norm
                let ln = if li + 1 == n_layers {
                    &self.head()?.norm
                } else if i + 1 < sw.layers.len() {
                    &sw.layers[i + 1].ln1
                } else {
                    sw.next_ln1.as_ref().context("a stage before a boundary holds the next input_layernorm")?
                };
                if i + 1 == sw.layers.len() && !last_stage {
                    // the residual stream and its normalised copy as the one buffer the next stage receives
                    let buf = self.boundary_norm(&x, &y, bias.context("ordinary boundary bias")?, ln, eps)?;
                    x = buf.get(0)?;
                    boundary = Some(buf);
                } else {
                    let a;
                    (x, a) = self.residual_norm(dev, &x, &y, bias, ln, eps)?;
                    normed = Some(a);
                }
            } else {
                ensure!(self.numerics == LlamaNumerics::Ordinary, "thinking residual requires CUDA");
                x = fused::bias_residual(&x, &y, &self.w(bias.context("ordinary residual bias")?)?)?;
            }
            self.mark(dev, t, "down_proj+bias+residual")?;
            if let Some(rm) = &self.residual_max {
                let m = x.abs()?.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?;
                rm.borrow_mut().push(m);
                *t = Instant::now();
            }
            if li + 1 < n_layers {
                gates.checkpoint()?;
            }
        }
        if let Some(buf) = boundary {
            return Ok((StageOut::Boundary(buf), captured));
        }
        ensure!(last_stage, "stage {s} ended without its boundary buffer");
        let out = match normed {
            Some(a) => a,
            None => rms_norm(&x, &self.w(&self.head()?.norm)?, eps)?,
        };
        self.mark(dev, t, "final_norm")?;
        Ok((StageOut::Final(Some(out)), captured))
    }

    /// Move the boundary buffer of stage `s` (`[2, M, H]` f16: the residual stream and its normalised copy) to stage
    /// `s + 1` and enter that stage; split only. Protocol:
    /// - step 1: drain stage `s`: the buffer is written completely and nothing of this thread runs on GPU `s`;
    /// - steps 2–3: release gate `s`, take gate `s + 1` (health check), bind context `s + 1`;
    /// - step 4: allocate and zero the destination on GPU `s + 1`;
    /// - step 5: drain stage `s + 1`, so the memset completed before any copy is queued;
    /// - step 6: `memcpy_dtod` on the destination stream. cudarc destroys its temporary source event inside that
    ///   call, which binds the source context, so the copy lands on this thread's per-thread stream of either context;
    /// - step 7: confirm the end of the copy: drain `s + 1`, then `s` (both always);
    /// - step 8: free the source with `s` current;
    /// - step 9: bind `s + 1` and wrap the destination as the next stage's input.
    ///
    /// Synchronisations and binds have two attempts each (`Twice`); every first-attempt error fails the request. A
    /// buffer whose last use cannot be confirmed is quarantined, never freed, and marks the lineage lost; buffers
    /// known to be idle are freed with their owner's context bound ([`StageCopy`]).
    #[cfg(feature = "cuda")]
    fn transfer(&self, gates: &mut StageGates<'_>, s: usize, buf: Tensor) -> Result<Tensor> {
        let dims = buf.dims();
        ensure!(dims.len() == 3 && dims[0] == 2, "boundary buffer of shape {dims:?}, [2, M, H] expected");
        transfer_activation(
            gates,
            s,
            (self.stages[s].device, self.stages[s + 1].device),
            buf,
            &self.lineage,
            self.transfers.as_deref(),
        )
    }

    /// `x + (y + b)` and its RMSNorm with `ln`: one fused kernel on CUDA, the two separate ops elsewhere (bitwise the
    /// same results).
    fn residual_norm(
        &self,
        dev: &Device,
        x: &Tensor,
        y: &Tensor,
        b: Option<&Tensor>,
        ln: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if dev.is_cuda() {
            let b = b.map(|b| self.w(b)).transpose()?;
            return Ok(fused::residual_rmsnorm(x, y, b.as_ref(), &self.w(ln)?, eps, self.numerics)?);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = dev;
        ensure!(self.numerics == LlamaNumerics::Ordinary, "thinking residual requires CUDA");
        let x = fused::bias_residual(x, y, &self.w(b.context("ordinary residual bias")?)?)?;
        let a = rms_norm(&x, &self.w(ln)?, eps)?;
        Ok((x, a))
    }

    /// [`GpuBackend::residual_norm`] of a stage's last layer before a boundary (CUDA): the same kernel and values, as
    /// its one `[2, M, H]` buffer.
    fn boundary_norm(&self, x: &Tensor, y: &Tensor, b: &Tensor, ln: &Tensor, eps: f32) -> Result<Tensor> {
        ensure!(self.numerics == LlamaNumerics::Ordinary, "thinking numerics do not support stage boundaries");
        #[cfg(feature = "cuda")]
        return Ok(fused::residual_rmsnorm_buf(x, y, &self.w(b)?, &self.w(ln)?, eps)?);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (x, y, b, ln, eps);
            bail!("a layer split needs the CUDA backend")
        }
    }

    fn rope_row(&self, pos: f64, cos: &mut [f32], sin: &mut [f32]) {
        let d = 2 * self.inv_freq.len();
        match self.rope.get(pos as usize * d..(pos as usize + 1) * d).filter(|_| pos.fract() == 0.0 && pos >= 0.0) {
            Some(row) => {
                let (c, s) = row.split_at(d / 2);
                cos.copy_from_slice(c);
                sin.copy_from_slice(s);
            }
            None if self.numerics == LlamaNumerics::ThinkBf16 => {
                think_rope_angles(self.cfg.rope_theta, pos as f32, cos, sin)
            }
            None => rope_angles(&self.inv_freq, pos, cos, sin),
        }
    }

    /// The tensor-core attention kernel of stage `s`: the plan's attention policy, else `BASAL_ATT`, else the
    /// plan's device default for the stage's GPU.
    #[cfg(feature = "cuda")]
    fn tc_kernel_of(&self, s: usize) -> Result<&'static str> {
        if let Some(k) = self.attention_override {
            return Ok(k);
        }
        let default = self
            .attention_defaults
            .get(s)
            .copied()
            .with_context(|| format!("no planned attention kernel for stage {s}"))?;
        Ok(fused::tc_kernel(default))
    }

    /// CUDA attention on tensor cores (f16 / bf16 forwards; `BASAL_ATT=simt` selects the f32 kernel for A/B).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code, reason = "Tensor-core attention is a CUDA-only kernel"))]
    fn tc_attention(&self) -> bool {
        self.is_cuda()
            && self.precision != Precision::F32
            && (self.attention_override.is_some() || crate::plan::DiagnosticOverrides::process().att() != Some("simt"))
    }

    /// The prefix K / V of every layer (of one stage) split into f16 hi / lo planes, when the tensor-core attention
    /// reads them.
    fn split_kv(&self, kv: &LayerKv) -> Result<Vec<(Tensor, Tensor)>> {
        #[cfg(feature = "cuda")]
        if self.tc_attention() {
            return kv
                .iter()
                .map(|(k, v)| Ok((fused::split_hilo(&k.flatten_all()?)?, fused::split_hilo(&v.flatten_all()?)?)))
                .collect();
        }
        let _ = kv;
        Ok(Vec::new())
    }

    /// K/V of all layers for `ids` at positions 0..len, continuing a precomputed `past` prefix of it: one part per
    /// stage, each with `OnStage::bind = bind`. Ends with the last stage entered.
    fn capture_prefix(
        &self,
        gates: &mut StageGates<'_>,
        ids: &[u32],
        past: Option<usize>,
        bind: bool,
    ) -> Result<Vec<OnStage<StageKv>>> {
        let np = past.map_or(0, |i| self.prefixes[i].ids.len());
        let l = ids.len() - np;
        let lk = ids.len();
        let half = self.cfg.head_dim / 2;
        let mut cs = vec![0f32; 2 * l * half];
        for t in 0..l {
            let (cos, sin) = cs.split_at_mut(l * half);
            self.rope_row((np + t) as f64, &mut cos[t * half..(t + 1) * half], &mut sin[t * half..(t + 1) * half]);
        }
        let mut mask = Vec::new();
        if !self.tree_attention() {
            mask = vec![MASKED; l * lk];
            for t in 0..l {
                mask[t * lk..t * lk + np + t + 1].fill(0.0); // the past prefix and causal
            }
        }
        let units = vec![Unit { q_off: 0, q_len: l, ranges: vec![(0, l)] }];
        let seg = Segment { off: 0, len: l, past: past.map(|i| &self.prefixes[i]), mask, units };
        let (_, parts) = self.forward_ragged(gates, &ids[np..], &cs, &[seg], &[], Some(bind))?;
        Ok(parts)
    }

    /// One generation step, with only the new causal suffix sent through the decoder. The ordinary prefix cache
    /// is deliberately not consulted. All device readout temporaries die on the last stage before gates end.
    fn generation_step(&self, prefix: &[u32], codes: &[u32]) -> Result<(Vec<f32>, PrefixKv)> {
        ensure!(!self.lineage.health.is_lost(), LOST);
        ensure!(
            !prefix.is_empty() && prefix.len() <= self.cfg.max_positions && u32::try_from(prefix.len()).is_ok(),
            "generation prefix is empty or exceeds model context"
        );
        ensure!(prefix.iter().all(|&id| (id as usize) < self.cfg.vocab), "generation token outside vocabulary");
        ensure!(
            !codes.is_empty() && codes.iter().all(|&id| (id as usize) < self.cfg.vocab),
            "generation readout outside vocabulary"
        );
        let past = self.generation.as_ref();
        let np = past.map_or(0, |p| p.ids.len());
        ensure!(
            np < prefix.len() && past.is_none_or(|p| prefix.starts_with(&p.ids)),
            "generation must strictly extend the exact cached token prefix; close the session before a new prompt"
        );
        let ids = &prefix[np..];
        let n = ids.len();
        let half = self.cfg.head_dim / 2;
        let mut cs = vec![0f32; 2 * n * half];
        for t in 0..n {
            let (cos, sin) = cs.split_at_mut(n * half);
            self.rope_row((np + t) as f64, &mut cos[t * half..(t + 1) * half], &mut sin[t * half..(t + 1) * half]);
        }
        let seg = Segment {
            off: 0,
            len: n,
            past,
            mask: Vec::new(),
            units: vec![Unit { q_off: 0, q_len: n, ranges: vec![(0, n)] }],
        };
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let result = (|| {
            let (hidden, parts) = self.forward_ragged(&mut gates, ids, &cs, &[seg], &[(n - 1) as u32], Some(true))?;
            let hidden = hidden.context("generation final hidden state")?;
            let last = self.stages.last().context("generation model has a stage")?;
            let codes = Tensor::from_slice(codes, (codes.len(),), &last.dev)?;
            #[cfg(feature = "cuda")]
            let logits =
                fused::letter_logits(&hidden.contiguous()?, &self.head()?.lm_head, &codes)?.to_vec2::<f32>()?;
            #[cfg(not(feature = "cuda"))]
            let logits = hidden
                .to_dtype(DType::F32)?
                .matmul(&self.head()?.lm_head.index_select(&codes, 0)?.to_dtype(DType::F32)?.t()?)?
                .to_vec2::<f32>()?;
            gates.mark_drained();
            let mut logits = logits.into_iter().next().context("generation readout has no row")?;
            ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite generation logits");
            if self.readout == Readout::Bf16Rounded {
                logits.iter_mut().for_each(|v| *v = bf16_round(*v));
            }
            Ok((logits, PrefixKv { ids: prefix.to_vec(), parts, pinned: false, last_use: 0 }))
        })();
        gates.end(result)
    }

    /// State cache: for a single row whose template + state prefix (`Packed::state_len`) extends a known prefix by at least `MIN_STATE_TOKENS`,
    /// compute the root block once, keep its K/V (bounded LRU) and return its index. Exact: the root is visible to
    /// every token of the row, and later rows use it only on an exact token-prefix match. With a split, the capture
    /// ends on the last stage, and an evicted entry frees each part with its stage's context bound (no computation:
    /// no gate needed).
    fn cache_state(&mut self, gates: &mut StageGates<'_>, row: &Packed, past: Option<usize>) -> Result<Option<usize>> {
        const MIN_STATE_TOKENS: usize = 32;
        let np = past.map_or(0, |i| self.prefixes[i].ids.len());
        let root = row.state_len.min(row.ids.len() - 1);
        if self.state_cache_bytes == 0 || self.kernels != Kernels::Fused || root < np + MIN_STATE_TOKENS {
            return Ok(past);
        }
        #[cfg(feature = "cuda")]
        let hilo = self.tc_attention();
        #[cfg(not(feature = "cuda"))]
        let hilo = false;
        let need = prefix_bytes(&self.cfg, root, hilo);
        if need > self.state_cache_bytes {
            return Ok(past);
        }
        let ids = row.ids[..root].to_vec();
        // one GPU: the parts are freed without binding, as before (the lane's thread uses only that context)
        let parts = self.capture_prefix(gates, &ids, past, self.split())?;
        let entry = PrefixKv { ids, parts, pinned: false, last_use: self.clock };
        loop {
            let used: usize = self.prefixes.iter().filter(|p| !p.pinned).map(|p| p.bytes(&self.cfg)).sum();
            if used + need <= self.state_cache_bytes {
                break;
            }
            let lru = (0..self.prefixes.len())
                .filter(|&i| !self.prefixes[i].pinned)
                .min_by_key(|&i| self.prefixes[i].last_use);
            match lru {
                Some(i) => {
                    self.prefixes.remove(i);
                }
                None => break,
            }
        }
        self.prefixes.push(entry);
        self.state_cache_inserts += 1;
        Ok(Some(self.prefixes.len() - 1))
    }

    /// Longest precomputed prefix the row starts with and that every real token of the row may see.
    fn best_prefix(&self, r: &Packed) -> Option<usize> {
        self.prefixes
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                let n = e.ids.len();
                n < r.ids.len() && r.ids.starts_with(&e.ids) && (n <= r.prefix_len || r.last.len() == 1)
            })
            .max_by_key(|(_, e)| e.ids.len())
            .map(|(i, _)| i)
    }

    /// Host inputs of the padded (stock candle) forward: ids, cos/sin, additive mask, readout indices.
    fn padded_inputs(&self, rows: &[&Packed], l: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
        let b = rows.len();
        let half = self.cfg.head_dim / 2;
        let mut ids = vec![PAD_ID; b * l];
        let mut cs = vec![0f32; 2 * b * l * half];
        let mut mask = vec![MASKED; b * l * l];
        let mut read = Vec::new();
        for (r, p) in rows.iter().enumerate() {
            let n = p.ids.len();
            ids[r * l..r * l + n].copy_from_slice(&p.ids);
            let nodes = p.parent.len();
            let visible: Vec<bool> =
                (0..nodes * nodes).map(|k| p.is_ancestor_or_self((k / nodes) as u32, (k % nodes) as u32)).collect();
            for t in 0..l {
                let pos = if t < n { p.pos[t] as f64 } else { t as f64 };
                let (cos, sin) = cs.split_at_mut(b * l * half);
                let c0 = (r * l + t) * half;
                self.rope_row(pos, &mut cos[c0..c0 + half], &mut sin[c0..c0 + half]);
                let row = &mut mask[(r * l + t) * l..(r * l + t + 1) * l];
                row[t] = 0.0; // padding tokens see only themselves
                if t < n {
                    let ni = p.node[t] as usize;
                    for (s, m) in row.iter_mut().enumerate().take(t) {
                        if visible[p.node[s] as usize * nodes + ni] {
                            *m = 0.0;
                        }
                    }
                }
            }
            read.extend(p.last.iter().map(|&c| (r * l + c) as u32));
        }
        (ids, cs, mask, read)
    }
}

impl GpuBackend {
    /// Projection weight shapes (N, K) that the loaded GEMM table does not cover on some stage (CUDA with cuBLASLt;
    /// empty otherwise). Such shapes use algorithms timed at first use, whose results may depend on the batch.
    /// Projection shapes for which a background forward of up to `max_m` tokens could reach an M class the
    /// cuBLASLt table does not hold (and so tune at run time). Empty without cuBLASLt (no run-time tuning) and for a
    /// batch-invariant table covering the shape.
    pub fn background_gemm_missing(&self, max_m: usize) -> Vec<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            let dt = self.precision.compute();
            let mut missing = Vec::new();
            for lt in self.stages.iter().filter_map(|s| s.w.get().lt.as_ref()) {
                for (n, k) in projection_shapes(&self.cfg) {
                    if !lt.covers_classes(n, k, dt, max_m) && !missing.contains(&(n, k)) {
                        missing.push((n, k));
                    }
                }
            }
            missing
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_m;
            Vec::new()
        }
    }

    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        #[cfg(feature = "cuda")]
        {
            let dt = self.precision.compute();
            let mut missing = Vec::new();
            for lt in self.stages.iter().filter_map(|s| s.w.get().lt.as_ref()) {
                for (n, k) in projection_shapes(&self.cfg) {
                    if !lt.covers(n, k, dt) && !missing.contains(&(n, k)) {
                        missing.push((n, k));
                    }
                }
            }
            missing
        }
        #[cfg(not(feature = "cuda"))]
        Vec::new()
    }

    fn gemm_desc(&self) -> &'static str {
        if self.numerics == LlamaNumerics::ThinkBf16 {
            if crate::plan::DiagnosticOverrides::process().think_gemm().is_some() {
                return "cuBLASLt bias epilogues, f32 accumulation, first heuristic algorithm per exact M (BASAL_THINK_GEMM, untimed); split-K may round partial sums to BF16";
            }
            return "cuBLASLt bias epilogues, f32 accumulation, heuristic RuntimeTiming; split-K may round partial sums to BF16";
        }
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.stages[0].w.get().lt {
            // every stage loads the same table in the same mode (load_gemm_table)
            if lt.is_invariant() {
                return "cuBLASLt, f32 accumulation, batch-invariant: one algorithm per weight shape for every M, no split-K (--gemm-table from gemm-search --invariant); a row's result does not depend on its batch";
            }
            return "cuBLASLt, f32 accumulation; algorithm per (N, K, M class: 16 up to 512, then 128) from the --gemm-table search, else the fastest of up to 16 heuristic algorithms timed at first use";
        }
        "cuBLAS (candle), f32 accumulation"
    }

    /// cuBLASLt: classes taken from a search table and classes tuned at run time (heuristic timing), as JSON; with a
    /// split one object per stage.
    pub fn gemm_tuning(&self) -> Value {
        #[cfg(feature = "cuda")]
        {
            // the chosen configuration (opaque bytes decoded by cuBLASLt), to compare choices between processes
            let algo = |data: &[u64; 8]| {
                cublaslt::algo_config(data)
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), Value::from(v)))
                    .collect::<serde_json::Map<_, _>>()
            };
            let tuning = |lt: &cublaslt::Lt| {
                let t = lt.tuned.lock().unwrap();
                let mut v = json!({"from_table": *lt.from_table.lock().unwrap(), "tuned_at_run_time": t.iter().map(|e| json!({"m_class": e.m_class, "n": e.n, "k": e.k, "ms": e.ms, "heuristic_first_ms": e.heuristic_ms, "tried": e.tried, "algo": algo(&e.algo), "algo_bytes": e.algo})).collect::<Vec<_>>()});
                if self.numerics == LlamaNumerics::ThinkBf16 {
                    v["kind"] = if crate::plan::DiagnosticOverrides::process().think_gemm().is_some() {
                        "bias_epilogue_first_heuristic_exact_m_untimed"
                    } else {
                        "bias_epilogue_timings_not_table_entries"
                    }
                    .into();
                }
                v
            };
            if !self.split() {
                return self.stages[0].w.get().lt.as_deref().map_or(Value::Null, tuning);
            }
            Value::Array(
                self.stages
                    .iter()
                    .enumerate()
                    .map(|(s, st)| {
                        let mut v = st.w.get().lt.as_deref().map_or(Value::Null, tuning);
                        if let Some(o) = v.as_object_mut() {
                            o.insert("stage".into(), json!(s));
                            o.insert("device".into(), json!(st.device));
                        }
                        v
                    })
                    .collect(),
            )
        }
        #[cfg(not(feature = "cuda"))]
        Value::Null
    }

    /// Use the cuBLASLt algorithm table `v` written by `gemm_search` (CUDA; rejected if the cuBLASLt version
    /// differs), on every stage: the same table, checked against the GPU name of each stage, so all stages run the
    /// same mode. `name` names the table in messages. With `table_only` (a reproducible plan) only a
    /// batch-invariant table of the forward's dtype is accepted and a GEMM without its algorithm is an error.
    pub fn load_gemm_table(&mut self, v: &Value, name: &str, table_only: bool) -> Result<usize> {
        ensure!(self.numerics == LlamaNumerics::Ordinary, "thinking bias-only cuBLASLt does not accept GEMM tables");
        #[cfg(feature = "cuda")]
        if self.stages.iter().all(|s| s.w.get().lt.is_some()) {
            ensure!(
                v["cublaslt_version"].as_u64() == Some(cublaslt::version() as u64),
                "{name}: table for cuBLASLt {}, this process uses {}",
                v["cublaslt_version"],
                cublaslt::version()
            );
            for (s, st) in self.stages.iter().enumerate() {
                if let (Some(t), Device::Cuda(d)) = (v["gpu"].as_str(), &st.dev) {
                    let here = d.cuda_stream().context().name()?;
                    if self.stages.len() > 1 {
                        ensure!(t == here, "{name}: table for {t}, the GPU of stage {s} is {here}");
                    } else {
                        ensure!(t == here, "{name}: table for {t}, this GPU is {here}");
                    }
                }
            }
            let entries = table_entries(v)?;
            if table_only {
                let dt = self.precision.compute();
                ensure!(
                    v["invariant"].as_bool() == Some(true) && entries.iter().all(|e| e.dtype == dt),
                    "{name}: a reproducible plan needs a batch-invariant table of {dt:?} only"
                );
            }
            for lt in self.stages.iter().filter_map(|s| s.w.get().lt.as_ref()) {
                if v["invariant"].as_bool() == Some(true) {
                    lt.load_invariant(&entries)?;
                } else {
                    lt.load(&entries);
                }
                if table_only {
                    lt.set_table_only();
                }
            }
            return Ok(entries.len());
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (v, table_only);
        bail!("--gemm-table needs the CUDA backend with cuBLASLt ({name})")
    }

    /// The device of the first stage (the only one on one GPU).
    fn first_device(&self) -> &Device {
        &self.stages[0].dev
    }

    /// Ordinal and device of every stage, in stage order.
    fn stage_devices(&self) -> impl Iterator<Item = (usize, &Device)> {
        self.stages.iter().map(|s| (s.device, &s.dev))
    }

    /// Transfer counters of a layer split (shared with the forks); None on one GPU.
    pub fn stage_transfers(&self) -> Option<StageTransfers> {
        self.transfers.as_ref().map(|t| StageTransfers(Arc::clone(t)))
    }
}

/// The entries of a GEMM table written by [`gemm_search`].
#[cfg(feature = "cuda")]
fn table_entries(v: &Value) -> Result<Vec<cublaslt::Tuned>> {
    let mut entries = Vec::new();
    for e in v["entries"].as_array().context("entries")? {
        let algo: Vec<u64> = e["algo"].as_array().context("algo")?.iter().filter_map(|x| x.as_u64()).collect();
        ensure!(algo.len() == 8, "algo must have 8 words");
        entries.push(cublaslt::Tuned {
            m_class: e["m_class"].as_u64().context("m_class")? as usize,
            n: e["n"].as_u64().context("n")? as usize,
            k: e["k"].as_u64().context("k")? as usize,
            dtype: match e["dtype"].as_str() {
                Some("f16") => DType::F16,
                Some("bf16") => DType::BF16,
                d => bail!("dtype {d:?}"),
            },
            algo: algo.try_into().map_err(|_| anyhow::anyhow!("algo must have 8 words"))?,
            ms: 0.0,
            heuristic_ms: 0.0,
            tried: 0,
        });
    }
    Ok(entries)
}

/// A batch-invariant GEMM table (e.g. one shipped with the build) is usable on GPU `device`: same GPU name and
/// cuBLASLt version, and cuBLASLt accepts every algorithm at its M class there.
pub fn check_gemm_table(v: &Value, device: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        ensure!(
            v["cublaslt_version"].as_u64() == Some(cublaslt::version() as u64),
            "table for cuBLASLt {}, this process uses {}",
            v["cublaslt_version"],
            cublaslt::version()
        );
        ensure!(v["invariant"].as_bool() == Some(true), "not a batch-invariant table");
        let dev = gpu_device(device)?;
        let Device::Cuda(d) = &dev else { bail!("not a CUDA device") };
        let here = d.cuda_stream().context().name()?;
        ensure!(v["gpu"].as_str() == Some(here.as_str()), "table for {}, this GPU is {here}", v["gpu"]);
        let lt = cublaslt::Lt::new(d)?;
        lt.load_invariant(&table_entries(v)?)?;
        lt.validate_invariant()?;
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    bail!("GEMM tables need the CUDA backend ({}, device {device})", v["gpu"])
}

/// `BASAL_HOST_TIMING` is set: every CUDA readout reports its host preparation, enqueue and wait times (stderr).
#[cfg(feature = "cuda")]
fn host_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::plan::DiagnosticOverrides::process().host_timing())
}

/// RoPE cos and sin of position `pos` for every frequency.
fn rope_angles(inv_freq: &[f64], pos: f64, cos: &mut [f32], sin: &mut [f32]) {
    for (j, f) in inv_freq.iter().enumerate() {
        let a = pos * f;
        cos[j] = a.cos() as f32;
        sin[j] = a.sin() as f32;
    }
}

/// C1: f32 inverse-frequency/power and angle; host f64 trig rounded through f32 to BF16. Device-libm ulps are deferred.
fn think_rope_angles(base: f64, pos: f32, cos: &mut [f32], sin: &mut [f32]) {
    let d = 2 * cos.len();
    for j in 0..cos.len() {
        let inv = 1.0f32 / (base as f32).powf((2 * j) as f32 / d as f32);
        let angle = pos * inv;
        cos[j] = bf16_round((angle as f64).cos() as f32);
        sin[j] = bf16_round((angle as f64).sin() as f32);
    }
}

fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

impl GpuBackend {
    /// [`Backend::letter_logits`] inside the stage gates of the call: stage 0 is entered before anything else (as the
    /// gate of one GPU was), and the readout runs in a scope of the last stage only.
    fn letter_logits_gated(
        &mut self,
        gates: &mut StageGates<'_>,
        rows: &[&Packed],
        letters: &[&[Vec<u32>]],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        gates.switch_to(0)?;
        let t0 = Instant::now();
        let b = rows.len();
        let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        ensure!(l > 0, "empty batch");
        // positions restart in every branch of a tree row: the limit applies to the longest prompt, not the row
        let max_pos = rows.iter().flat_map(|r| r.pos.iter()).copied().max().unwrap_or(0) as usize;
        ensure!(
            max_pos < self.cfg.max_positions,
            "a prompt of {} tokens exceeds the model limit {}",
            max_pos + 1,
            self.cfg.max_positions
        );
        let half = self.cfg.head_dim / 2;
        // Union of letter ids of the batch -> one small f32 product.
        let mut uniq: Vec<u32> = letters.iter().flat_map(|g| g.iter().flatten().copied()).collect();
        uniq.sort_unstable();
        uniq.dedup();
        // Readout scope of the last stage: the final hidden states and every tensor below live on its GPU, with its
        // context current and (with a gate) its gate held. Nothing enters or binds another stage until they are gone
        // at the end of this block, also on `Err` and while unwinding, so before the gates are finished or aborted.
        let (logits, t1) = {
            let (hidden, t1) = if self.kernels == Kernels::Candle {
                let (ids, cs, mask, read) = self.padded_inputs(rows, l);
                let t1 = Instant::now();
                (self.forward_padded(&ids, &cs, &mask, b, l, &read)?, t1)
            } else {
                // Every row uses its own longest precomputed prefix; a single-row forward may add its state to the
                // cache.
                self.clock += 1;
                let mut past: Vec<Option<usize>> = rows.iter().map(|r| self.best_prefix(r)).collect();
                for &i in past.iter().flatten() {
                    self.prefixes[i].last_use = self.clock;
                    if !self.prefixes[i].pinned {
                        self.state_cache_hits += 1;
                    }
                }
                if b == 1 && past[0].is_none_or(|i| self.prefixes[i].pinned) {
                    past[0] = self.cache_state(gates, rows[0], past[0])?;
                }
                let total: usize = rows
                    .iter()
                    .zip(&past)
                    .map(|(r, p)| r.ids.len() - p.map_or(0, |i| self.prefixes[i].ids.len()))
                    .sum();
                let mut ids = Vec::with_capacity(total);
                let mut cs = vec![0f32; 2 * total * half];
                let mut read = Vec::new();
                let mut segs = Vec::with_capacity(b);
                for (p, pi) in rows.iter().zip(&past) {
                    let np = pi.map_or(0, |i| self.prefixes[i].ids.len());
                    let off = ids.len();
                    let n = p.ids.len() - np;
                    ids.extend_from_slice(&p.ids[np..]);
                    let nodes = p.parent.len();
                    // visible[a * nodes + b]: block a is block b or one of its ancestors
                    let visible: Vec<bool> = (0..nodes * nodes)
                        .map(|k| p.is_ancestor_or_self((k / nodes) as u32, (k % nodes) as u32))
                        .collect();
                    let lk = np + n;
                    let tree = self.tree_attention();
                    let mut mask = if tree { Vec::new() } else { vec![MASKED; n * lk] };
                    for t in 0..n {
                        let (cos, sin) = cs.split_at_mut(total * half);
                        let c0 = (off + t) * half;
                        self.rope_row(p.pos[np + t] as f64, &mut cos[c0..c0 + half], &mut sin[c0..c0 + half]);
                        if tree {
                            continue;
                        }
                        let row = &mut mask[t * lk..(t + 1) * lk];
                        row[..np].fill(0.0); // the precomputed prefix is visible to every token
                        row[np + t] = 0.0;
                        let ni = p.node[np + t] as usize;
                        for (s, m) in row[np..].iter_mut().enumerate().take(t) {
                            if visible[p.node[np + s] as usize * nodes + ni] {
                                *m = 0.0;
                            }
                        }
                    }
                    read.extend(p.last.iter().map(|&c| (off + c - np) as u32));
                    let units = if tree { row_units(p, np)? } else { Vec::new() };
                    segs.push(Segment { off, len: n, past: pi.map(|i| &self.prefixes[i]), mask, units });
                }
                let t1 = Instant::now();
                (self.forward_ragged(gates, &ids, &cs, &segs, &read, None)?.0.context("readout")?, t1)
            };
            let last = self.stages.last().context("a model has a stage")?;
            let head = self.head()?;
            let ids_t = Tensor::from_slice(&uniq, (uniq.len(),), &last.dev)?;
            #[cfg(feature = "cuda")]
            let cuda_readout = last.dev.is_cuda();
            #[cfg(not(feature = "cuda"))]
            let cuda_readout = false;
            let logits: Vec<Vec<f32>> = if cuda_readout {
                #[cfg(feature = "cuda")]
                {
                    let lt = fused::letter_logits(&hidden.contiguous()?, &head.lm_head, &ids_t)?;
                    let enqueued = Instant::now();
                    let v = lt.to_vec2()?;
                    if host_timing() {
                        // BASAL_HOST_TIMING: host preparation, enqueueing of the forward, wait for the GPU and the copy
                        eprintln!(
                            "host-timing rows {b} prepare {:.3} ms enqueue {:.3} ms wait {:.3} ms",
                            (t1 - t0).as_secs_f64() * 1e3,
                            (enqueued - t1).as_secs_f64() * 1e3,
                            enqueued.elapsed().as_secs_f64() * 1e3
                        );
                    }
                    v
                }
                #[cfg(not(feature = "cuda"))]
                unreachable!()
            } else {
                let w = head.lm_head.index_select(&ids_t, 0)?.to_dtype(DType::F32)?;
                hidden.to_dtype(DType::F32)?.matmul(&w.t()?)?.to_vec2()?
            };
            // the synchronous copy to the host was the last GPU work of this call on the last stage (only the
            // stream-ordered frees of this scope's tensors follow, on that stage's stream)
            gates.mark_drained();
            (logits, t1)
        };
        let t2 = Instant::now();
        let col = |id: u32| uniq.binary_search(&id).expect("letter in union");
        let mut out = Vec::with_capacity(b);
        let mut ri = 0;
        for g in letters {
            let mut orders = Vec::with_capacity(g.len());
            for ids in g.iter() {
                let row = &logits[ri];
                orders.push(
                    ids.iter()
                        .map(|&id| {
                            let v = row[col(id)];
                            if self.readout == Readout::Bf16Rounded {
                                bf16_round(v)
                            } else {
                                v
                            }
                        })
                        .collect(),
                );
                ri += 1;
            }
            out.push(orders);
        }
        ensure!(ri == logits.len(), "readout count mismatch");
        // A non-finite logit (e.g. an f16 overflow) is an error, never a decision.
        ensure!(
            logits.iter().flatten().all(|v| v.is_finite()),
            "non-finite letter logits ({:?} forward)",
            self.precision
        );
        self.last_timing = ((t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3);
        Ok(out)
    }

    /// [`Backend::evidence_scores`] inside the stage gates of the call (readout scope as in `letter_logits_gated`).
    fn evidence_gated(
        &self,
        gates: &mut StageGates<'_>,
        ids: &[u32],
        t0: usize,
        t1: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        gates.switch_to(0)?;
        let n = ids.len();
        ensure!(t0 < t1 && t1 <= n, "evidence: token range {t0}..{t1} outside {n} tokens");
        ensure!(n <= self.cfg.max_positions, "evidence: {n} tokens exceed the model limit");
        let half = self.cfg.head_dim / 2;
        let mut cs = vec![0f32; 2 * n * half];
        for t in 0..n {
            let (cos, sin) = cs.split_at_mut(n * half);
            self.rope_row(t as f64, &mut cos[t * half..(t + 1) * half], &mut sin[t * half..(t + 1) * half]);
        }
        // plain causal forward of the whole prompt (upstream: model.model(input_ids))
        let mut mask = Vec::new();
        if !self.tree_attention() {
            mask = vec![MASKED; n * n];
            for t in 0..n {
                mask[t * n..t * n + t + 1].fill(0.0);
            }
        }
        let units = vec![Unit { q_off: 0, q_len: n, ranges: vec![(0, n)] }];
        let seg = Segment { off: 0, len: n, past: None, mask, units };
        let read: Vec<u32> = (t0 as u32..t1 as u32).chain(std::iter::once(n as u32 - 1)).collect();
        // readout scope of the last stage: `h`, its f32 copy and the head's intermediates are on its GPU and die
        // in this call, before the gates end
        let (h, _) = self.forward_ragged(gates, ids, &cs, &[seg], &read, None)?;
        let h = h.context("evidence readout")?.to_dtype(DType::F32)?;
        let m = t1 - t0;
        let head = self.head()?.evidence.as_ref().context("this model has no evidence head")?;
        let scores = head.scores(&h.narrow(0, 0, m)?, &h.narrow(0, m, 1)?)?;
        // the second side's synchronous copy to the host was the last GPU work of this call
        gates.mark_drained();
        Ok(scores)
    }
}

/// One forward of [`GpuBackend::stage_work`]: tokens, readouts, distinct letter ids, and the evidence head's
/// dimension when the readout is the evidence head instead of the letter rows.
#[derive(Clone, Copy)]
struct ForwardWork {
    tokens: u64,
    readouts: u64,
    letters: u64,
    evidence: Option<u64>,
}

impl GpuBackend {
    /// Transient device bytes of one forward on stage `s` (`stage_layers` and the readout scope of
    /// `letter_logits_gated` / `evidence_gated`), read from the allocations of the source. Every tensor of one layer
    /// iteration is counted as live until the end of the iteration (the bindings of the loop body), with the residual
    /// stream and its normalised copy carried from the previous layer.
    fn stage_work(&self, s: usize, f: ForwardWork) -> Tally {
        let c = &self.cfg;
        let u = |n: usize| n as u64;
        let (m, r, l) = (f.tokens, f.readouts, f.letters);
        let (h, hd, nh, nkv, i) = (u(c.hidden), u(c.head_dim), u(c.heads), u(c.kv_heads), u(c.intermediate));
        let q = (nh + 2 * nkv) * hd;
        let e = u(self.precision.compute().size_in_bytes());
        let w = u(self.precision.storage().size_in_bytes());
        let cuda = self.is_cuda();
        let first = s == 0;
        let last = s + 1 == self.stages.len();
        let mut t = Tally::default();
        if first {
            // token ids; embedding rows (`index_select` in the storage dtype) and their compute-dtype copy
            t.add(prod(&[m, 4]));
            t.add(prod(&[m, h, w]));
            t.add(prod(&[m, h, e]));
        } else {
            // the boundary buffer [2, M, H] f16 received from the previous stage (`transfer_activation`)
            t.add(prod(&[2, m, h, 2]));
        }
        // RoPE cos/sin (every stage uploads them) and the readout indices (last stage)
        t.add(prod(&[m, hd, 4]));
        if last && r > 0 {
            t.add(prod(&[r, 4]));
        }
        // one layer: carried residual stream and norm, the input norm, qkv projection
        t.add_n(3, prod(&[m, h, e]));
        t.add(prod(&[m, q, e]));
        // qkv_rope: the f32 q/k/v buffer; CUDA tensor-core attention also K/V hi/lo f16 planes (4 * NKV*M*HD)
        t.add(prod(&[m, q, 4]));
        if cuda && self.tc_attention() {
            t.add(prod(&[4, nkv, m, hd, 2]));
        }
        // attention: f32 output (Metal, SIMT, tc_pipe) and the merged output in the compute dtype
        t.add(prod(&[m, nh, hd, 4]));
        t.add(prod(&[m, nh, hd, e]));
        if last {
            // only the readout rows after the last attention: attention output and residual stream
            t.add(prod(&[r, nh, hd, e]));
            t.add(prod(&[r, h, e]));
        }
        // o projection, residual + norm, gate/up, SiLU*up, down projection
        t.add(prod(&[m, h, e]));
        t.add_n(2, prod(&[m, h, e]));
        if self.numerics == LlamaNumerics::ThinkBf16 {
            // C3': unrounded s0 through the MLP; add counts bytes and one allocation slot.
            t.add(prod(&[m, h, 4]));
        }
        t.add(prod(&[m, 2, i, e]));
        t.add(prod(&[m, i, e]));
        t.add(prod(&[m, h, e]));
        if last {
            // residual + next norm (or the final norm)
            t.add_n(2, prod(&[m, h, e]));
        } else {
            // boundary_norm: the [2, M, H] f16 buffer sent to the next stage
            t.add(prod(&[2, m, h, 2]));
        }
        let largest_projection = [q * h, h * nh * hd, 2 * i * h, h * i].into_iter().max().unwrap_or(0);
        if self.precision == Precision::F32 {
            // bf16 weights widened to f32 per use: one projection at a time, plus the layer's biases and norms
            t.add(prod(&[largest_projection, 4]));
            t.add(prod(&[q + h + 2 * i + 4 * h, 4]));
        }
        if self.quantization != Quantization::None {
            // one projection dequantized to f16 at a time
            t.add(prod(&[largest_projection, 2]));
        }
        if last {
            // final norm when it was not carried
            t.add(prod(&[r, h, e]));
            match f.evidence {
                Some(k) => {
                    // evidence head: f32 hidden rows, both normalisations, and per side three products and the
                    // log-softmax
                    t.add(prod(&[r, h, 4]));
                    t.add_n(4, prod(&[r + 1, h, 4]));
                    t.add_n(2, prod(&[r + 1, k, 4]));
                    t.add_n(8, prod(&[r, 4]));
                }
                None if cuda => {
                    // letter ids and the fused readout [R, U] f32
                    t.add(prod(&[l, 4]));
                    t.add(prod(&[r, l, 4]));
                }
                None => {
                    // letter ids, gathered lm_head rows and their f32 copy, f32 hidden, [R, U] product
                    t.add(prod(&[l, 4]));
                    t.add(prod(&[l, h, w]));
                    t.add(prod(&[l, h, 4]));
                    t.add(prod(&[r, h, 4]));
                    t.add(prod(&[r, l, 4]));
                }
            }
        }
        t
    }
}

impl Backend for GpuBackend {
    fn generation_logits(&mut self, prefix: &[u32], codes: &[u32]) -> Result<Vec<f32>> {
        if self.kernels != Kernels::Fused || !self.is_cuda() {
            ensure!(!prefix.is_empty() && !codes.is_empty(), "empty generation prefix/readout");
            let row = basal_core::pack::pack(&[prefix.to_vec()]);
            let codes = [codes.to_vec()];
            let mut out = self.letter_logits(&[&row], &[&codes])?;
            ensure!(out.len() == 1 && out[0].len() == 1, "generation readout shape mismatch");
            return Ok(out.remove(0).remove(0));
        }
        match self.generation_step(prefix, codes) {
            Ok((logits, next)) => {
                self.end_generation().context("releasing previous generation prefix")?;
                self.generation = Some(next);
                Ok(logits)
            }
            Err(error) => match self.end_generation() {
                Ok(()) => Err(error),
                Err(cleanup) => Err(error.context(format!("releasing failed generation prefix: {cleanup:#}"))),
            },
        }
    }

    fn end_generation(&mut self) -> Result<()> {
        let Some(prefix) = self.generation.take() else { return Ok(()) };
        let mut failure: Option<anyhow::Error> = None;
        // Steps end with a synchronous readout and drained gate releases. Each part binds its owning context;
        // a lost lineage retains allocations under the existing OnStage safety contract.
        for (stage, part) in prefix.parts.into_iter().enumerate() {
            if let Err(error) = part.release() {
                failure = Some(match failure {
                    Some(prior) => prior.context(format!("releasing generation stage {stage}: {error:#}")),
                    None => error.context(format!("releasing generation stage {stage}")),
                });
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn generation_work_bound(
        &self,
        shape: &basal_core::planner::DeviceWorkShape,
    ) -> basal_core::planner::DeviceWorkBound {
        use basal_core::planner::DeviceWorkBound;
        let bound = self.device_work_bound(shape);
        if self.kernels != Kernels::Fused || !self.is_cuda() {
            return bound;
        }
        let DeviceWorkBound::Estimate(mut estimate) = bound else { return bound };
        let c = &self.cfg;
        let context = shape.forward_tokens.max(shape.longest_prompt);
        let kv_width = prod(&[c.kv_heads as u64, c.head_dim as u64]);
        let qkv_width = prod(&[c.heads as u64, c.head_dim as u64]).saturating_add(kv_width.saturating_mul(2));
        for (s, device) in estimate.devices.iter_mut().enumerate() {
            let layers = self.stages[s].layers.len() as u64;
            // Old + replacement raw storage, each charged a whole f32 QKV backing allocation per layer.
            // Prefill K/V views may alias Q too; contiguous() alone does not detach that storage. Later steps
            // concatenate K/V into two separate allocations whose total is smaller than QKV. Count four
            // allocations per layer for both old/new K/V. stage_work already covers current-layer scratch.
            let mut retained =
                Tally { bytes: prod(&[2, layers, context, qkv_width, 4]), allocations: layers.saturating_mul(4) };
            if self.tc_attention() {
                // Old + replacement K/V hi-lo: two f16 planes per tensor, including split_kv outputs.
                retained.add_n(layers.saturating_mul(4), prod(&[context, kv_width, 4]));
            }
            device.bytes = device.bytes.saturating_add(retained.bytes);
            device.allocations = device.allocations.saturating_add(retained.allocations);
        }
        estimate.basis.push_str("; generation: full-context old + replacement per-layer QKV backing and KV hi/lo, plus forward scratch; cleared before P2");
        DeviceWorkBound::Estimate(estimate)
    }

    fn generation_host_staging(&self) -> Option<basal_core::planner::HostStaging> {
        let mut staging = self.host_staging()?;
        if self.kernels == Kernels::Fused && self.is_cuda() {
            // A retained tensor has an Arc<Tensor_>, Arc<RwLock<Storage>>, shape/stride Vecs (rank <= 4),
            // and two Arc headers (four usize counters). Count independent storage even for aliased views.
            let tensor_metadata = (std::mem::size_of::<Tensor>()
                + std::mem::size_of::<<Tensor as std::ops::Deref>::Target>()
                + std::mem::size_of::<std::sync::RwLock<candle_core::Storage>>()
                + 12 * std::mem::size_of::<usize>()) as u64;
            // Old + replacement, raw K/V + hi/lo: eight tensors per layer; double for Vec capacity.
            staging.fixed = staging
                .fixed
                .saturating_add(prod(&[16, self.cfg.num_layers as u64, tensor_metadata]))
                .saturating_add(prod(&[4, self.stages.len() as u64, std::mem::size_of::<OnStage<StageKv>>() as u64]));
        }
        Some(staging)
    }

    fn early_exit_caps(&self) -> Option<&basal_core::early_exit::ExitCaps> {
        self.head().ok()?.exits.as_ref().map(|h| &h.caps)
    }

    fn letter_logits_with_policy(
        &mut self,
        rows: &[&Packed],
        letters: &[&[Vec<u32>]],
        policy: &str,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let heads = self.head()?.exits.as_ref().context("trained exits were not enabled at startup")?;
        heads.caps.thresholds(policy)?;
        if policy == "off" {
            return self.letter_logits(rows, letters);
        }
        ensure!(!self.lineage.health.is_lost(), LOST);
        ensure!(!rows.is_empty() && rows.len() == letters.len(), "invalid exit batch");
        for (row, codes) in rows.iter().zip(letters) {
            ensure!(!row.ids.is_empty() && row.last.len() == codes.len(), "invalid exit readout shape");
            ensure!(
                row.pos.iter().all(|&p| (p as usize) < self.cfg.max_positions),
                "exit prompt exceeds context length"
            );
            ensure!(codes.iter().flatten().all(|&id| (id as usize) < self.cfg.vocab), "exit code outside vocabulary");
        }
        let longest = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let short = longest <= 3072;
        let (b, l) = if short {
            let l = basal_core::early_exit::length_bucket(longest).context("exit length bucket")?;
            let b = basal_core::early_exit::batch_bucket(rows.len()).context("exit batch bucket")?;
            ensure!(b * l <= basal_core::pack::TOKEN_BUDGET, "exit chunk exceeds padded token budget");
            (b, l)
        } else {
            ensure!(rows.len() == 1, "oversize exit units must run individually at full depth");
            (1, longest)
        };
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let result = (|| {
            gates.switch_to(0)?;
            // Dummy rows create only padding, never readouts or gates. Actual rows retain their mapping.
            let empty = Packed {
                ids: vec![],
                pos: vec![],
                node: vec![],
                parent: vec![],
                last: vec![],
                prefix_len: 0,
                state_len: 0,
            };
            let mut padded = rows.to_vec();
            padded.resize(b, &empty);
            let (ids, cs, mask, read) = self.padded_inputs(&padded, l);
            let lm = &self.head()?.lm_head;
            let mut check = |layer, raw: &Tensor| {
                gates.checkpoint()?;
                if short {
                    heads.readout(layer, raw, self.cfg.rms_eps, lm, letters, policy)
                } else {
                    Ok(None)
                }
            };
            let output = self.forward_padded_exit(&ids, &cs, &mask, b, l, &read, Some(&mut check))?;
            let logits = match output {
                early_exit::PaddedOutput::Exit(logits) => logits,
                early_exit::PaddedOutput::Hidden(hidden) => early_exit::project(&hidden, lm, letters)?,
            };
            gates.mark_drained();
            Ok(logits)
        })();
        gates.end(result)
    }

    /// Per stage, the larger of the letter forward and (with an evidence head) the evidence forward, from
    /// `GpuBackend::stage_work`. Unknown for the paths without a source-derived bound: the candle reference
    /// kernels (padded rows with dense masks), the dense-mask SDPA attention (`BASAL_ATT=sdpa`), the state cache
    /// (single-row captures) and diagnostics.
    fn device_work_bound(&self, shape: &basal_core::planner::DeviceWorkShape) -> basal_core::planner::DeviceWorkBound {
        use basal_core::planner::{DeviceBytes, DeviceWorkBound, DeviceWorkEstimate};
        if self.kernels != Kernels::Fused {
            return DeviceWorkBound::Unknown(
                "the candle reference kernels (padded rows, dense masks) have no device work bound".into(),
            );
        }
        if !self.tree_attention() {
            return DeviceWorkBound::Unknown(
                "dense-mask SDPA attention (BASAL_ATT=sdpa) has no device work bound".into(),
            );
        }
        if self.state_cache_bytes > 0 {
            return DeviceWorkBound::Unknown(
                "state-cache captures are not modelled; run background without one".into(),
            );
        }
        if self.profile.is_some() || self.residual_max.is_some() {
            return DeviceWorkBound::Unknown("diagnostics allocate per layer; not modelled".into());
        }
        if !shape.media.is_empty() {
            return DeviceWorkBound::Unknown("this backend takes no media".into());
        }
        let letter = ForwardWork {
            tokens: shape.forward_tokens,
            readouts: shape.forward_readouts,
            letters: shape.forward_letters,
            evidence: None,
        };
        let evidence = match (shape.evidence, self.head().ok().and_then(|h| h.evidence.as_ref())) {
            (Some(e), Some(head)) => {
                Some(ForwardWork { tokens: e.tokens, readouts: e.readouts, letters: 0, evidence: Some(head.k as u64) })
            }
            _ => None,
        };
        let devices = (0..self.stages.len())
            .map(|s| {
                let mut t = self.stage_work(s, letter);
                if let Some(e) = evidence {
                    t = t.peak(self.stage_work(s, e));
                }
                DeviceBytes { device: self.stages[s].device, bytes: t.bytes, allocations: t.allocations }
            })
            .collect();
        DeviceWorkBound::Estimate(DeviceWorkEstimate {
            devices,
            basis: format!(
                "Llama fused {:?} ({}), quantization {:?}, {} stage(s); forward <= {} tokens, {} readouts, {} letters; \
                 evidence {:?}; later rounds worst case: {}",
                self.precision,
                if self.is_cuda() { self.attention_kernel().unwrap_or("cuda") } else { "metal tree attention" },
                self.quantization,
                self.stages.len(),
                shape.forward_tokens,
                shape.forward_readouts,
                shape.forward_letters,
                shape.evidence.map(|e| e.tokens),
                shape.worst_case_rounds
            ),
        })
    }

    fn has_evidence(&self) -> bool {
        self.head().is_ok_and(|h| h.evidence.is_some())
    }

    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(!self.lineage.health.is_lost(), LOST);
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let result = self.evidence_gated(&mut gates, ids, t0, t1);
        gates.end(result)
    }

    fn fork(&self) -> Result<Self> {
        ensure!(self.profile.is_none() && self.residual_max.is_none(), "fork: diagnostics are per backend");
        ensure!(!self.lineage.health.is_lost(), "fork: {LOST}");
        Ok(Self {
            stages: self
                .stages
                .iter()
                .map(|s| Stage {
                    dev: s.dev.clone(),
                    device: s.device,
                    layers: s.layers.clone(),
                    w: s.w.clone(),
                    gate: None,
                })
                .collect(),
            lineage: self.lineage.clone(),
            transfers: self.transfers.clone(),
            prefixes: self.prefixes.iter().filter(|p| p.pinned).cloned().collect(),
            generation: None,
            state_cache_bytes: self.state_cache_bytes,
            clock: 0,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            cfg: self.cfg.clone(),
            precision: self.precision,
            readout: self.readout,
            kernels: self.kernels,
            attention_override: self.attention_override,
            attention_defaults: self.attention_defaults.clone(),
            numerics: self.numerics,
            inv_freq: self.inv_freq.clone(),
            rope: self.rope.clone(),
            load_s: 0.0,
            profile: None,
            quantization: self.quantization,
            #[cfg(feature = "cuda")]
            quant_stats: self.quant_stats.clone(),
            dequant: std::cell::Cell::new((0.0, 0)),
            residual_max: None,
            last_timing: (0.0, 0.0),
            stop: None,
        })
    }

    /// One GPU: the gate of its stage. A split takes one gate per device ([`Backend::set_device_gate`]); given one gate
    /// here, every stage uses it (one at a time, as with separate gates), which serialises its GPUs.
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        for s in &mut self.stages {
            s.gate = Some((gate.clone(), WorkClass::from_urgent(urgent)));
        }
    }

    fn set_device_gate(&mut self, device: usize, gate: Arc<Gate>, urgent: bool) -> Result<()> {
        Ok(self.set_device_work_class(device, gate, WorkClass::from_urgent(urgent))?)
    }

    fn set_device_work_class(
        &mut self,
        device: usize,
        gate: Arc<Gate>,
        class: WorkClass,
    ) -> std::result::Result<(), ExecutionError> {
        let devices: Vec<usize> = self.stages.iter().map(|s| s.device).collect();
        let stage = self
            .stages
            .iter_mut()
            .find(|s| s.device == device)
            .ok_or(ExecutionError::NoSuchDevice { device, devices })?;
        stage.gate = Some((gate, class));
        Ok(())
    }

    /// CUDA and Metal: drained gate releases and stop checks at every stage entry and layer checkpoint. Refused
    /// with diagnostics enabled, with a state cache (background forks run without one), and on CUDA when the
    /// cuBLASLt table does not cover every M class up to `max_forward_tokens` (background never tunes GEMMs).
    fn background_support(&self, max_forward_tokens: usize) -> BackgroundSupport {
        if self.profile.is_some() || self.residual_max.is_some() {
            return BackgroundSupport::Unsupported("diagnostics are enabled on this backend".into());
        }
        if self.state_cache_bytes > 0 {
            return BackgroundSupport::Unsupported("the background engine must run without a state cache".into());
        }
        let missing = self.background_gemm_missing(max_forward_tokens);
        if !missing.is_empty() {
            return BackgroundSupport::Unsupported(format!(
                "the cuBLASLt table does not cover every M class up to {max_forward_tokens} tokens for projection \
                 shapes {missing:?}; background would tune GEMMs at run time"
            ));
        }
        BackgroundSupport::Supported
    }

    /// Background forks: the state cache is disabled and its (unpinned) prefixes are dropped, each freed with its
    /// stage's context bound (`OnStage`). Pinned template prefixes stay: resident K/V shared with the online engine.
    /// The lineage's health on CUDA and Metal alike (`device_health` stays CUDA-only for the server's routing).
    fn work_health(&self) -> Option<DeviceHealth> {
        Some(self.lineage.health.clone())
    }

    fn configure_background(&mut self) -> std::result::Result<(), ExecutionError> {
        if self.profile.is_some() || self.residual_max.is_some() {
            return Err(ExecutionError::Unsupported { reason: "diagnostics are enabled on this backend".into() });
        }
        self.end_generation().map_err(|e| ExecutionError::Device(e.into()))?;
        self.state_cache_bytes = 0;
        self.prefixes.retain(|p| p.pinned);
        Ok(())
    }

    /// A token is accepted only when every stage is gated as background work, so that no stage of a background call
    /// runs without its device's gate.
    fn set_stop_token(&mut self, stop: Option<StopToken>) -> std::result::Result<(), ExecutionError> {
        if stop.is_some() && !self.stages.iter().all(|s| matches!(s.gate, Some((_, WorkClass::Background)))) {
            return Err(ExecutionError::InvalidState {
                reason: "background work needs the background gate of every GPU of the model".into(),
            });
        }
        self.stop = stop;
        Ok(())
    }

    fn host_staging(&self) -> Option<basal_core::planner::HostStaging> {
        let c = &self.cfg;
        // host inputs per token (ids, positions, RoPE cos/sin, segment and tree-unit metadata), letter logits and
        // readout indices, evidence hidden rows (f32) read back for evidence questions; doubled for growth
        Some(basal_core::planner::HostStaging {
            fixed: 1 << 20,
            per_token: ((c.head_dim * 4 + 128) * 2) as u64,
            per_row: 1024,
            per_readout: 64,
            per_letter: 32,
            per_media_value: 0,
            per_evidence_token: (c.hidden * 4 * 2) as u64,
        })
    }

    /// CUDA: the lineage's health, marked lost when the end of this model's own GPU work could not be confirmed (two
    /// failed synchronisations or context binds); not a hardware check.
    fn device_health(&self) -> Option<DeviceHealth> {
        self.is_cuda().then(|| self.lineage.health.clone())
    }

    fn release_work(&mut self) -> std::result::Result<(), ExecutionError> {
        self.end_generation().map_err(|e| ExecutionError::Device(e.into()))
    }

    fn add_prefix(&mut self, ids: &[u32]) -> Result<()> {
        if self.kernels != Kernels::Fused || ids.is_empty() || self.prefixes.iter().any(|p| p.ids == ids) {
            return Ok(());
        }
        ensure!(!self.lineage.health.is_lost(), LOST);
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let captured = self.capture_prefix(&mut gates, ids, None, true);
        let parts = gates.end(captured)?;
        // Publication barrier: the lanes on other threads read the pinned K/V and their hi/lo planes only after this.
        for (s, st) in self.stages.iter().enumerate() {
            st.dev
                .synchronize()
                .with_context(|| format!("publishing the precomputed prefix: synchronising stage {s}"))?;
        }
        self.prefixes.push(PrefixKv { ids: ids.to_vec(), parts, pinned: true, last_use: 0 });
        Ok(())
    }

    fn describe(&self) -> Value {
        let cuda = self.is_cuda();
        let device = self.stages[0].device;
        let mut d = json!({
            "backend": if cuda { "cuda" } else { "metal" },
            "device": if cuda { format!("NVIDIA GPU through candle Device::new_cuda({device})") } else { format!("Apple GPU through candle Device::new_metal({device})") },
            "tensor_library": if cuda { "candle-core 0.11.0 (CUDA; matmul = cuBLAS, no TF32)" } else { "candle-core 0.11.0 (Metal; matmul = MLX steel GEMM port)" },
            "weights": match self.precision {
                Precision::Bf16 => "bf16 (checkpoint)",
                Precision::F16 => "f16 (converted from bf16, experiment)",
                Precision::F32 => "bf16 resident, widened to f32 per layer and use (exact)",
            },
            "activations": match self.precision {
                Precision::Bf16 => "bf16 residual stream, norms and MLP",
                Precision::F16 => "f16 residual stream, norms and MLP",
                Precision::F32 => "f32 everywhere (numerical reference)",
            },
            "attention": match self.kernels {
                Kernels::Fused if cuda => "f32 q/k/v (bias + rotate-half RoPE fused kernel, cos/sin from f64 on host); one f32 flash-style kernel for all segments of a forward (online softmax, additive block mask, prefix K/V read from the cache, fully masked key tiles skipped), GQA as stacked query rows",
                Kernels::Fused => "f32 q/k/v (bias + rotate-half RoPE fused kernel, cos/sin from f64 on host); candle sdpa (MLX steel attention) in f32 with additive block mask; GQA in-kernel",
                Kernels::Candle => "f32: RoPE (candle rope), scores matmul, additive block mask, softmax, weighted sum",
            },
            "kernels": match self.kernels {
                Kernels::Fused => "fused: q/k/v and gate/up projections as single GEMMs without bias; bias added in fused kernels (bias+RoPE+head split, bias+SiLU*up, bias+residual)",
                Kernels::Candle => "candle ops only: GEMM + broadcast bias add, separate SiLU, mul and residual add",
            },
            "readout": match self.readout { Readout::F32 => "letter rows of lm_head only, f32", Readout::Bf16Rounded => "letter rows of lm_head only, f32 rounded to bf16 on host" },
            "padding": match self.kernels {
                Kernels::Fused => "none: compact token list for GEMM/norm/MLP, attention per packed row (segment)",
                Kernels::Candle => "rows padded on the right to the longest packed row of a chunk",
            },
            "attention_kernel": self.attention_kernel().unwrap_or("unknown"),
            "gemm": if cuda { self.gemm_desc() } else { "MLX steel GEMM (candle); 32-row tiles when M mod 64 is in 1..=32 and M <= 600" },
            "prefix_kv": self.prefixes.iter().filter(|p| p.pinned).map(|p| p.ids.len()).collect::<Vec<_>>(),
            "state_cache_bytes": self.state_cache_bytes,
            "last_layer": match self.kernels {
                Kernels::Fused => "o_proj and MLP of the last layer only at the readout positions",
                Kernels::Candle => "full",
            },
        });
        if self.numerics == LlamaNumerics::ThinkBf16 {
            d["think_numerics"] = "upstream-bf16-pointwise".into();
            d["activations"] = "BF16 layer boundaries and norm/MLP outputs; f32 interior residual through MLP".into();
            d["attention"] =
                "BF16-valued q/k/v in f32 backing, BF16 RoPE table from host f32 recipe; existing attention kernels"
                    .into();
            d["kernels"] = "basal-llama-think-bf16-pointwise.1: bias epilogue before final BF16 store, BF16 q/k stores, f32 SiLU through up multiply, unrounded interior residual; no parity guarantee".into();
        }
        // `none` keeps the description above unchanged
        if let Some(heads) = self.head().ok().and_then(|h| h.exits.as_ref()) {
            d["early_exit"] = basal_core::early_exit::profile(&heads.caps);
            d["early_exit"]["resident_head_bytes"] = heads.resident_bytes.into();
            d["early_exit"]["readout"] =
                "FP32 trained RMSNorm + residual adapter + frozen LM code rows; FP32 code softmax".into();
            d["early_exit"]["execution_bounds"] = "UNBOUNDED_FEATURE: active Candle grid/dense masks/FP32 buffers; bounded/background execution unavailable".into();
        }
        #[cfg(feature = "cuda")]
        if let Some(stats) = &self.quant_stats {
            d["weights"] = match stats {
                quant::QuantStats::Q8(_) => json!("projections q8_0 (GGML blocks, dequantized to f16 per projection); embedding, norms, biases f16 (converted from bf16); lm_head bf16"),
                quant::QuantStats::Q4K(_) => json!("projections q4_k (GGML blocks, dequantized to f16 per projection); embedding, norms, biases f16 (converted from bf16); lm_head bf16"),
            };
            d["quantization"] = stats.describe();
        }
        // a layer split changes only `device` and adds its placement and the transfer protocol (no measured values);
        // one GPU keeps the description above unchanged
        if self.split() {
            let calls: Vec<String> = self.stages.iter().map(|s| format!("Device::new_cuda({})", s.device)).collect();
            let list = match calls.split_last() {
                Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
                _ => calls.concat(),
            };
            d["device"] = json!(format!("NVIDIA GPUs through candle {list}, consecutive layer ranges"));
            let evidence = self.has_evidence();
            let last = self.stages.len() - 1;
            d["layer_split"] = Value::Array(
                self.stages
                    .iter()
                    .enumerate()
                    .map(|(s, st)| {
                        let mut holds = Vec::new();
                        if s == 0 {
                            holds.push("embed");
                        }
                        if s == last {
                            holds.extend(["norm", "lm_head"]);
                            if evidence {
                                holds.push("evidence_head");
                            }
                        }
                        json!({"stage": s, "device": st.device, "first_layer": st.layers.start,
                            "last_layer": st.layers.end.saturating_sub(1), "holds": holds})
                    })
                    .collect(),
            );
            d["stage_transfer"] = json!(STAGE_TRANSFER);
        }
        d
    }

    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        ensure!(!self.lineage.health.is_lost(), LOST);
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let result = self.letter_logits_gated(&mut gates, rows, letters);
        gates.end(result)
    }
}

/// Time `x[m,k] @ w[n,k]^T` on GPU `device` for the projection shapes of the model (synchronised, after warm-up).
/// Returns `(m, k, n, dtype, ms, TFLOP/s)` rows. Diagnostic for comparing GEMM kernels with MLX / cuBLAS.
pub fn gemm_microbench(
    ms: &[usize],
    shapes: &[(usize, usize)],
    dtype: &str,
    reps: usize,
    device: usize,
) -> Result<Vec<Value>> {
    let dev = gpu_device(device)?;
    let dt = match dtype {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        _ => bail!("dtype {dtype}"),
    };
    let mut out = Vec::new();
    #[cfg(feature = "cuda")]
    let lt = match &dev {
        Device::Cuda(d) if dt != DType::F32 => Some(cublaslt::Lt::new(d)?),
        _ => None,
    };
    for &m in ms {
        for &(k, n) in shapes {
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(dt)?;
            let w = Tensor::randn(0f32, 1.0, (n, k), &dev)?.to_dtype(dt)?;
            let wt = w.t()?.contiguous()?; // [k, n]
            #[cfg(feature = "cuda")]
            if let (Device::Cuda(d), Some(lt)) = (&dev, &lt) {
                let _ = d;
                let f = || cublaslt::matmul_nt(lt, &x, &w);
                f()?; // tunes the class of m
                dev.synchronize()?;
                let t = Instant::now();
                for _ in 0..reps {
                    f()?;
                }
                dev.synchronize()?;
                let s = t.elapsed().as_secs_f64() / reps as f64;
                out.push(json!({"m": m, "k": k, "n": n, "dtype": dtype, "layout": "lt", "ms": s * 1e3, "tflops": 2.0 * (m * k * n) as f64 / s / 1e12}));
            }
            for (layout, rhs) in [("nt", w.t()?), ("nn", wt.clone())] {
                for _ in 0..3 {
                    x.matmul(&rhs)?;
                }
                dev.synchronize()?;
                // reps forwards queued back to back, one synchronisation (as inside a forward)
                let t = Instant::now();
                for _ in 0..reps {
                    x.matmul(&rhs)?;
                }
                dev.synchronize()?;
                let s = t.elapsed().as_secs_f64() / reps as f64;
                out.push(json!({"m": m, "k": k, "n": n, "dtype": dtype, "layout": layout, "ms": s * 1e3, "tflops": 2.0 * (m * k * n) as f64 / s / 1e12}));
            }
        }
    }
    Ok(out)
}

/// Time `gemm::gemm_nt` for every instantiated tile and swizzle 0..=3 against candle's matmul on the projection
/// shapes, with the max |difference| to candle's result (diagnostic for choosing the GEMM configuration). Metal only.
#[cfg(target_os = "macos")]
pub fn gemm_tune(ms: &[usize], shapes: &[(usize, usize)], dtype: &str, reps: usize) -> Result<Vec<Value>> {
    let dev = Device::new_metal(0)?;
    let dt = match dtype {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "f32" => DType::F32,
        _ => bail!("dtype {dtype}"),
    };
    let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
        for _ in 0..3 {
            f()?;
        }
        dev.synchronize()?;
        let t = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        dev.synchronize()?;
        Ok(t.elapsed().as_secs_f64() / reps as f64 * 1e3)
    };
    let mut out = Vec::new();
    for &m in ms {
        for &(k, n) in shapes {
            let x = (Tensor::randn(0f32, 1.0, (m, k), &dev)? * 0.1)?.to_dtype(dt)?;
            let w = (Tensor::randn(0f32, 1.0, (n, k), &dev)? * 0.1)?.to_dtype(dt)?;
            let reference = x.matmul(&w.t()?)?.to_dtype(DType::F32)?;
            let base = time(&|| Ok(x.matmul(&w.t()?)?))?;
            out.push(json!({"m": m, "k": k, "n": n, "kernel": "candle", "ms": base}));
            for tile in gemm::TILES {
                for sw in 0..=3 {
                    let cfg = gemm::GemmConfig::new(tile, sw);
                    let ms_ = time(&|| Ok(gemm::gemm_nt(&x, &w, cfg)?))?;
                    let diff = (gemm::gemm_nt(&x, &w, cfg)?.to_dtype(DType::F32)? - &reference)?
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    out.push(json!({"m": m, "k": k, "n": n, "kernel": format!("{tile:?}/s{sw}"), "ms": ms_,
                                    "speedup": base / ms_, "max_abs_diff_vs_candle": diff}));
                }
            }
        }
    }
    Ok(out)
}

/// Exhaustive cuBLASLt configuration search for the projection shapes `(n, k)` at every M class in `m_classes`
/// (CUDA, f16 or bf16), on random operands on GPU `device`. Returns the table (JSON) for `--gemm-table`.
#[cfg(feature = "cuda")]
pub fn gemm_search(
    m_classes: &[usize],
    shapes: &[(usize, usize)],
    dtype: &str,
    invariant: bool,
    device: usize,
) -> Result<Value> {
    let dev = gpu_device(device)?;
    let dt = match dtype {
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        _ => bail!("dtype {dtype} (f16, bf16)"),
    };
    let Device::Cuda(d) = &dev else { bail!("not a CUDA device") };
    let lt = cublaslt::Lt::new(d)?;
    for &(n, k) in shapes {
        // copies of the weight, together > 256 MiB (L2 of the RTX 6000 Ada: 96 MiB), read in turn by the timing runs
        let copies = (256usize << 20).div_ceil(n * k * dt.size_in_bytes()).max(2);
        let w = (Tensor::randn(0f32, 1.0, (copies * n, k), &dev)? * 0.05)?.to_dtype(dt)?;
        if invariant {
            let m = m_classes.iter().copied().max().unwrap_or(16);
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(dt)?;
            cublaslt::invariant_nt(&lt, &x, &w, copies, m_classes)?;
            dev.synchronize()?;
            let t = lt.tuned.lock().unwrap();
            let rows: Vec<&cublaslt::Tuned> = t.iter().filter(|e| e.n == n && e.k == k).collect();
            let slow: Vec<f64> = rows.iter().map(|e| e.ms / e.heuristic_ms).collect();
            eprintln!(
                "gemm-search invariant n={n} k={k}: {} candidates, slowdown vs best per class: mean {:.3}, max {:.3}",
                rows.first().map_or(0, |e| e.tried),
                slow.iter().sum::<f64>() / slow.len().max(1) as f64,
                slow.iter().copied().fold(0.0, f64::max)
            );
            continue;
        }
        for &m in m_classes {
            let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?.to_dtype(dt)?;
            cublaslt::search_nt(&lt, &x, &w, copies)?;
            dev.synchronize()?;
            let t = lt.tuned.lock().unwrap().last().cloned().context("search result")?;
            eprintln!(
                "gemm-search m={m} n={n} k={k}: {:.4} ms (heuristic {:.4} ms, {} configurations, {:.0} TFLOP/s)",
                t.ms,
                t.heuristic_ms,
                t.tried,
                2.0 * (m * n * k) as f64 / t.ms / 1e9
            );
        }
    }
    let entries: Vec<Value> = lt
        .tuned
        .lock()
        .unwrap()
        .iter()
        .map(|e| {
            json!({"m_class": e.m_class, "n": e.n, "k": e.k, "dtype": dtype, "algo": e.algo.to_vec(),
                   "ms": e.ms, if invariant { "best_candidate_ms" } else { "heuristic_first_ms" }: e.heuristic_ms,
                   "configurations": e.tried, "tflops": 2.0 * (e.m_class * e.n * e.k) as f64 / e.ms / 1e9})
        })
        .collect();
    Ok(json!({"cublaslt_version": cublaslt::version(), "gpu": d.cuda_stream().context().name()?, "dtype": dtype,
              "invariant": invariant, "gemm_search_version": cublaslt::INVARIANT_SEARCH_VERSION, "shapes": shapes,
              "entries": entries,
              "timing": "each configuration timed on rotating copies of the weight larger than L2 (cold weights, as in a forward)",
              "note": "algorithms are opaque cuBLASLt configurations, valid for this GPU and cuBLASLt version"}))
}
