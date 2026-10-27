//! Text-only Qwen3.5 decoder, following Transformers 5.17.0 (Apache-2.0).
//! Readouts may reuse immutable full-chunk prefix states within a call; each branch writes fresh state storage.
//!
//! Submodules: `load` (checkpoint upload and merged projections), `prefill` (chunked multi-prompt layer
//! execution), `attention` and `delta` (the two mixers) and `prefix` (packed rows, prefix forks and readouts).

mod attention;
mod delta;
mod load;
mod prefill;
mod prefix;

use std::sync::Arc;

use anyhow::{ensure, Result};
use basal_core::architecture::Qwen35Config;
use basal_core::execution::WorkClass;
use basal_core::gate::Gate;
use basal_core::pack::Packed;
use basal_core::{Backend, DeviceHealth};
use candle_core::{DType, Device, Tensor, D};
use serde_json::{json, Value};

use crate::{GateOwners, OnStage, Precision, Readout, StageGates};

struct Attention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}
struct DeltaNet {
    input: Option<Tensor>,
    qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: Tensor,
    #[cfg_attr(
        not(feature = "cuda"),
        expect(dead_code, reason = "packed FP32 parameters of the CUDA preparation kernel")
    )]
    params: Tensor,
}
enum Mixer {
    Attention(Attention),
    DeltaNet(DeltaNet),
}
impl Mixer {
    /// Output projection back to the hidden size.
    fn output(&self) -> &Tensor {
        match self {
            Mixer::DeltaNet(w) => &w.out,
            Mixer::Attention(w) => &w.o,
        }
    }
}
struct Layer {
    gate_up: Option<Tensor>,
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}
struct Weights {
    #[cfg(feature = "cuda")]
    lt: Option<Arc<crate::cublaslt::Lt>>,
    embed: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    rope_inv: Tensor,
    head_map: Tensor,
}
#[derive(Clone, Default)]
struct State {
    kv: Option<(Tensor, Tensor)>,
    conv: Option<Tensor>,
    /// CUDA fused 128x128 heads use `[head,value,key]`; the Candle fallback uses `[head,key,value]`. A loaded
    /// backend never switches paths for a layer, and every branch's state starts at zero in its path's layout.
    recurrent: Option<Tensor>,
}

/// One prompt's chunk inside a prefill group.
struct PrefillRow {
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    mask: Tensor,
    #[cfg_attr(not(feature = "cuda"), expect(dead_code, reason = "selects the CUDA short-context attention"))]
    short_context: bool,
}

pub struct Qwen35Backend {
    pub(crate) dev: Device,
    pub(crate) device: usize,
    cfg: Qwen35Config,
    weights: Arc<OnStage<Weights>>,
    precision: Precision,
    readout: Readout,
    health: DeviceHealth,
    gate: Option<(Arc<Gate>, WorkClass)>,
    chunk: usize,
    kernel_set: crate::KernelSet,
    tensor_attention: bool,
    arbitrary_prefix: bool,
    pub load_s: f64,
}

// Fixed chunking bounds convolution/projection temporaries independently of the request batch.
const CHUNK: usize = 64;
/// Prompts up to this length keep the short-context prefill stride and attention path.
const SHORT_CONTEXT: usize = 2048;
/// Prompts prefilled together in one group.
const MAX_GROUP: usize = 8;

fn is_short(len: usize) -> bool {
    len <= SHORT_CONTEXT
}

/// Kernel sets with merged CUDA projections, cuBLASLt GEMMs and longer prefill chunks.
fn optimized_kernels(kernel_set: crate::KernelSet) -> bool {
    matches!(kernel_set, crate::KernelSet::Qwen35V2 | crate::KernelSet::Qwen35V3 | crate::KernelSet::Qwen35V4)
}

fn normalized(x: &Tensor, eps: f64, mean: bool) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let sum = if mean { f.sqr()?.mean_keepdim(D::Minus1)? } else { f.sqr()?.sum_keepdim(D::Minus1)? };
    Ok(f.broadcast_mul(&(sum + eps)?.powf(-0.5)?)?)
}

/// Qwen's ordinary RMSNorm has a zero-centered affine weight, applied in FP32 before rounding.
fn norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return Ok(crate::fused::qwen35_norm(x, w, eps)?);
    }
    Ok(normalized(x, eps, true)?.broadcast_mul(&(w.to_dtype(DType::F32)? + 1.0)?)?.to_dtype(x.dtype())?)
}

impl Qwen35Backend {
    fn linear(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.weights.get().lt {
            if lt.is_invariant() && lt.covers(w.dim(0)?, x.dim(1)?, x.dtype()) {
                return Ok(crate::cublaslt::matmul_nt(lt, &x.contiguous()?, w)?);
            }
        }
        super::gemma::linear(x, w)
    }

    /// CUDA v3/v4 half-precision attention derives causal positions on the GPU and never reads a dense mask.
    fn positional_attention(&self) -> bool {
        matches!(self.kernel_set, crate::KernelSet::Qwen35V3 | crate::KernelSet::Qwen35V4)
            && self.precision != Precision::F32
            && self.dev.is_cuda()
    }

    fn fresh_states(&self) -> Vec<State> {
        vec![State::default(); self.cfg.layers.len()]
    }

    pub fn apply_plan(&mut self, plan: &crate::ResolvedExecutionPlan) -> Result<(), crate::InferenceConfigError> {
        crate::inference::check_plan(plan)?;
        let crate::ExecutionPolicy::Qwen35 { tensor_attention } = plan.policy() else {
            return Err(crate::InferenceConfigError::PlanMismatch("Qwen3.5 execution policy"));
        };
        if plan.kernel_set() != self.kernel_set
            || plan.quantization() != crate::Quantization::None
            || (plan.gemm().table.is_some() && !optimized_kernels(self.kernel_set))
            || plan.precision() != self.precision
            || plan.readout() != self.readout
            || self.gate.is_some()
        {
            return Err(crate::InferenceConfigError::PlanMismatch("Qwen3.5 load options or lifecycle"));
        }
        self.tensor_attention = *tensor_attention;
        Ok(())
    }

    pub fn load_gemm_table(&mut self, value: &Value, name: &str) -> Result<usize> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.weights.get().lt {
            ensure!(self.gate.is_none(), "Qwen3.5 GEMM table must be loaded before execution lanes");
            crate::check_gemm_table(value, self.device)?;
            let entries = crate::table_entries(value)?;
            ensure!(
                !entries.is_empty() && entries.iter().all(|e| e.dtype == self.precision.compute()),
                "Qwen3.5 GEMM table dtype mismatch"
            );
            lt.load_invariant(&entries)?;
            lt.validate_invariant()?;
            let shapes = crate::architecture_projection_shapes(&basal_core::architecture::Architecture::Qwen35(
                self.cfg.clone(),
            ))?;
            self.arbitrary_prefix = self.chunk > CHUNK
                && self.cfg.head_dim == 256
                && shapes.iter().all(|&(n, k)| lt.covers(n, k, self.precision.compute()));
            return Ok(entries.len());
        }
        let _ = value;
        anyhow::bail!("Qwen3.5 GEMM tables require CUDA half precision and an optimized kernel set ({name})")
    }
}

impl Backend for Qwen35Backend {
    fn describe(&self) -> Value {
        json!({"backend":if self.dev.is_cuda() {"cuda"} else {"metal"},"device":self.device,
            "architecture":"qwen3_5_text","precision":format!("{:?}",self.precision),
            "implementation":"fused CUDA operations and immutable prefix forks; Candle fallback","recurrent_state":"F32",
            "attention_products":if self.tensor_attention { "half_input_f32_accumulation_long_context" } else { "F32" },"prefill_chunk":self.chunk,"qualified_short_prefix_forks":self.arbitrary_prefix,"prefix_cache":false,"state_cache":false})
    }
    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        ensure!(!self.health.is_lost(), crate::LOST);
        ensure!(rows.len() == letters.len(), "Qwen3.5 row/readout count mismatch");
        let owners = GateOwners {
            gates: vec![self.gate.clone()],
            devs: vec![self.dev.clone()],
            health: self.health.clone(),
            split: false,
            stop: None,
        };
        let mut gates = StageGates::new(&owners);
        #[cfg(feature = "cuda")]
        let profile = self.dev.is_cuda() && crate::plan::DiagnosticOverrides::process().cuda_profile();
        #[cfg(feature = "cuda")]
        if profile {
            gates.switch_to(0)?;
            if let Device::Cuda(dev) = &self.dev {
                dev.cuda_stream().context().bind_to_thread()?;
            }
            // SAFETY: this model's context is current and its GPU gate is owned. Profiling retains no borrowed
            // memory and is stopped after gates.end drains the forward on both the success and error paths.
            unsafe { candle_core::cuda_backend::cudarc::driver::sys::cuProfilerStart() }.result()?;
        }
        let result = rows.iter().zip(letters).map(|(row, codes)| self.read_row(row, codes, &mut gates)).collect();
        let result = gates.end(result);
        #[cfg(feature = "cuda")]
        if profile {
            use anyhow::Context as _;
            // SAFETY: matches the start on this context; the normal gate drain completed or marked the device
            // lost before this tracing-only API call. It never reads or mutates model buffers.
            let stopped = unsafe { candle_core::cuda_backend::cudarc::driver::sys::cuProfilerStop() }
                .result()
                .context("stopping CUDA profiler");
            // Always stop capture, but preserve the original inference failure if both operations failed.
            return result.and_then(|output| stopped.map(|()| output));
        }
        result
    }
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            dev: self.dev.clone(),
            device: self.device,
            cfg: self.cfg.clone(),
            weights: self.weights.clone(),
            precision: self.precision,
            readout: self.readout,
            health: self.health.clone(),
            gate: None,
            chunk: self.chunk,
            kernel_set: self.kernel_set,
            tensor_attention: self.tensor_attention,
            arbitrary_prefix: self.arbitrary_prefix,
            load_s: 0.0,
        })
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        self.gate = Some((gate, WorkClass::from_urgent(urgent)));
    }
    fn set_device_gate(&mut self, device: usize, gate: Arc<Gate>, urgent: bool) -> Result<()> {
        ensure!(device == self.device, "Qwen3.5 backend does not run on device {device}");
        self.set_gate(gate, urgent);
        Ok(())
    }
    fn device_health(&self) -> Option<DeviceHealth> {
        Some(self.health.clone())
    }
}
