//! Intel Arc inference through Vulkan (Mesa ANV), with FP16 storage and FP32 reductions.
//! The protocol, packing and calibration remain in `basal-core`. No Python or oneAPI runtime is used.
mod checkpoint;
mod runtime;
pub(crate) mod telemetry;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use basal_core::execution::{BackgroundSupport, ExecutionError, Interrupted, StopToken, WorkClass};
use basal_core::gate::{Gate, GateGuard};
use basal_core::{manifest::LlamaConfig, pack::Packed, Backend, DeviceHealth};
use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use wgpu::Buffer;

use crate::{EvidenceHead, GpuMemory, GpuMemoryProbe, Kernels, Precision, Quantization, Readout};
use checkpoint::{Bf16, Checkpoint};
use runtime::Gpu;

/// The backend selects its Vulkan adapter itself ([`info`]); there is no choice of another one.
const SINGLE_DEVICE: &str = "the Intel backend runs on device 0 only (the Intel GPU it selects)";

/// Errors while configuring or loading the Intel backend. Inference uses the common [`Backend`] error contract.
#[derive(Debug, thiserror::Error)]
pub enum IntelError {
    #[error("unsupported Intel backend configuration: {0}")]
    Configuration(&'static str),
    #[error("initializing Intel Vulkan")]
    Device(#[source] anyhow::Error),
    #[error("loading Intel model")]
    Model(#[source] anyhow::Error),
}

struct Layer {
    norm1: Buffer,
    qkv: Buffer,
    qkv_bias: Buffer,
    output: Buffer,
    output_bias: Buffer,
    norm2: Buffer,
    gate_up: Buffer,
    gate_up_bias: Buffer,
    down: Buffer,
    down_bias: Buffer,
}
struct Weights {
    layers: Vec<Layer>,
    norm: Buffer,
    /// Embeddings and the letter head are read from the checkpoint at requested rows only.
    checkpoint: Checkpoint,
    embed: Bf16,
    head: Bf16,
    evidence: Option<EvidenceHead>,
}

#[derive(Clone)]
struct Prefix {
    ids: Vec<u32>,
    kv: Vec<Buffer>,
}

/// The gate of one Intel call, held from entry to the end of the call. Between layers, after `Gpu::sync` confirmed
/// that layer's commands complete, the online long lane may hand the GPU to a main-lane caller and background work
/// may yield or stop ([`CallGate::checkpoint`]); a stop leaves the GPU released and nothing pending.
struct CallGate<'g> {
    gate: Option<(&'g Gate, WorkClass)>,
    /// Invariant: `Some` whenever `gate` is `Some` and this call holds the GPU.
    guard: Option<GateGuard<'g>>,
    stop: Option<&'g StopToken>,
    health: &'g DeviceHealth,
}

impl<'g> CallGate<'g> {
    fn enter(
        gate: Option<&'g (Arc<Gate>, WorkClass)>,
        stop: Option<&'g StopToken>,
        health: &'g DeviceHealth,
    ) -> Result<Self> {
        if let Some(stop) = stop {
            stop.check()?;
        }
        let gate = gate.map(|(g, class)| (&**g, *class));
        let guard = match gate {
            Some((g, class)) => Some(g.enter_class(class, stop)?),
            None => None,
        };
        Ok(Self { gate, guard, stop, health })
    }

    /// A wait for this call's own GPU work. Background work whose end cannot be confirmed (a failed wait,
    /// including a timeout) marks the device's health lost before the gate is released, so a healthy device after
    /// a failed background call means its work completed; online lanes keep their previous error path.
    fn confirm<T>(&self, waited: Result<T>) -> Result<T> {
        if waited.is_err() && matches!(self.gate, Some((_, WorkClass::Background))) {
            self.health.mark_lost();
            return waited.context("the end of background GPU work could not be confirmed: device marked lost");
        }
        waited
    }

    /// After a layer whose commands were confirmed complete: propagate the checkpoint's result and re-check the
    /// device's health after the GPU was handed over and taken back.
    fn checkpoint(&mut self) -> Result<()> {
        let Some((gate, class)) = self.gate else {
            return match self.stop.and_then(StopToken::reason) {
                Some(reason) => Err(Interrupted { reason }.into()),
                None => Ok(()),
            };
        };
        if class == WorkClass::OnlineMain {
            return Ok(());
        }
        let Some(guard) = self.guard.take() else { return Ok(()) };
        let stop = if class == WorkClass::Background { self.stop } else { None };
        let cp = gate.checkpoint(guard, stop)?;
        self.guard = Some(cp.guard);
        if cp.yielded {
            ensure!(!self.health.is_lost(), "Intel Vulkan device lost; restart the model");
        }
        Ok(())
    }
}

/// An Intel model. Forks share immutable weights/device/pipelines, with independent forward scratch buffers.
pub struct IntelBackend {
    prefixes: Vec<Prefix>,
    gpu: Arc<Gpu>,
    weights: Arc<Weights>,
    cfg: LlamaConfig,
    readout: Readout,
    gate: Option<(Arc<Gate>, WorkClass)>,
    /// Stop token of background work ([`Backend::set_stop_token`]); never inherited by forks.
    stop: Option<StopToken>,
    /// Load duration, excluding request work.
    pub load_s: f64,
    /// Cross-request state cache is not yet supported; a nonzero value is rejected before inference.
    pub state_cache_bytes: usize,
    pub state_cache_hits: u64,
    pub state_cache_inserts: u64,
    /// Optional synchronised timings in milliseconds; enables diagnostic synchronisation per layer.
    pub profile: Option<RefCell<BTreeMap<&'static str, f64>>>,
    pub residual_max: Option<RefCell<Vec<f32>>>,
    pub last_timing: (f64, f64),
}

impl IntelBackend {
    /// Estimate resident GPU weight buffers from the validated checkpoint header, without loading tensors.
    /// Embedding/head rows, prefix caches, forward scratch and driver allocations need additional headroom.
    pub fn estimated_weight_bytes(dir: &Path, cfg: &LlamaConfig) -> std::result::Result<u64, IntelError> {
        Checkpoint::open(&dir.join("model.safetensors"))
            .and_then(|checkpoint| checkpoint.resident_weight_bytes(cfg))
            .map_err(IntelError::Model)
    }

    pub fn load(
        dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
        quantization: Quantization,
        device: usize,
    ) -> std::result::Result<Self, IntelError> {
        if device != 0 {
            return Err(IntelError::Configuration(SINGLE_DEVICE));
        }
        match quantization {
            Quantization::None => {}
            Quantization::Q8_0 => {
                return Err(IntelError::Configuration("use quantization none; q8_0 weights need the CUDA backend"))
            }
            Quantization::Q4K => {
                return Err(IntelError::Configuration("use quantization none; q4_k weights need the CUDA backend"))
            }
        }
        if precision != Precision::F16 {
            return Err(IntelError::Configuration("use --dtype f16; bf16/f32 forwards are not implemented"));
        }
        if kernels != Kernels::Fused {
            return Err(IntelError::Configuration("use --kernels fused"));
        }
        let start = Instant::now();
        let mut gpu = Gpu::new().map_err(IntelError::Device)?;
        let shaders = [
            ("gemm", include_str!("gemm.wgsl")),
            ("norm", include_str!("norm.wgsl")),
            ("rope", include_str!("rope.wgsl")),
            ("attention", include_str!("attention.wgsl")),
            ("elementwise", include_str!("elementwise.wgsl")),
            ("gather", include_str!("gather.wgsl")),
            ("readout", include_str!("readout.wgsl")),
        ];
        for (name, source) in shaders {
            gpu.compile(name, source).map_err(IntelError::Device)?;
        }
        let weights = Self::load_weights(&gpu, dir, cfg).map_err(IntelError::Model)?;
        Ok(Self {
            prefixes: Vec::new(),
            gpu: Arc::new(gpu),
            weights: Arc::new(weights),
            cfg: cfg.clone(),
            readout,
            gate: None,
            stop: None,
            load_s: start.elapsed().as_secs_f64(),
            state_cache_bytes: 0,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            profile: None,
            residual_max: None,
            last_timing: (0.0, 0.0),
        })
    }

    fn load_weights(gpu: &Gpu, dir: &Path, c: &LlamaConfig) -> Result<Weights> {
        ensure!(
            c.num_layers > 0
                && c.num_layers <= 256
                && c.hidden > 0
                && c.hidden <= 65536
                && c.intermediate > 0
                && c.intermediate <= 131072
                && c.kv_heads > 0
                && c.kv_heads <= c.heads
                && c.heads > 0
                && c.heads <= 2048
                && c.vocab > 0
                && c.vocab <= 1_000_000,
            "unsupported Intel model dimensions"
        );
        ensure!(
            c.head_dim >= 32
                && c.head_dim <= 256
                && c.head_dim.is_multiple_of(32)
                && c.head_dim.is_multiple_of(gpu.info.subgroup_max_size as usize)
                && c.heads.is_multiple_of(c.kv_heads)
                && c.heads.checked_mul(c.head_dim) == Some(c.hidden),
            "unsupported Intel attention dimensions"
        );
        ensure!(
            c.max_positions > 0
                && c.max_positions <= 32768
                && c.rms_eps.is_finite()
                && c.rms_eps > 0.0
                && c.rope_theta.is_finite()
                && c.rope_theta > 0.0,
            "invalid Intel model positions/norm/RoPE"
        );
        let checkpoint = Checkpoint::open(&dir.join("model.safetensors"))?;
        let mut used = std::collections::BTreeSet::new();
        let mut chunk = Vec::new();
        let h = c.hidden;
        let q = (c.heads + 2 * c.kv_heads) * c.head_dim;
        let i = c.intermediate;
        // Concatenate the tensors in order, converted chunk by chunk into the mapped buffer.
        let mut load = |names: &[String], shapes: &[Vec<usize>]| -> Result<Buffer> {
            let mut tensors = Vec::with_capacity(names.len());
            for (name, shape) in names.iter().zip(shapes) {
                tensors.push(checkpoint.bf16(name, shape)?);
                used.insert(name.clone());
            }
            // F16 has the size of BF16, so the converted tensors fill the buffer exactly.
            let bytes = tensors.iter().map(|t| t.bytes()).sum();
            gpu.upload_with(&names[0], bytes, |mut mapped| {
                let mut offset = 0;
                for &tensor in &tensors {
                    checkpoint.read_chunks(tensor, &mut chunk, |data| {
                        for value in data.chunks_exact_mut(2) {
                            let bits = f16_bits(bf16(value));
                            value.copy_from_slice(&bits.to_le_bytes());
                        }
                        mapped.slice(offset..offset + data.len()).copy_from_slice(data);
                        offset += data.len();
                    })?;
                }
                Ok(())
            })
        };
        // Shaders index a bias by output column, so one zero buffer of the widest projection serves every layer.
        let zero_bias = if c.bias { None } else { Some(gpu.upload("zero bias", &vec![0u16; q.max(2 * i)])?) };
        let mut layers = Vec::with_capacity(c.num_layers);
        for l in 0..c.num_layers {
            let name = |s: &str| format!("model.layers.{l}.{s}");
            let norm1 = load(&[name("input_layernorm.weight")], &[vec![h]])?;
            let norm2 = load(&[name("post_attention_layernorm.weight")], &[vec![h]])?;
            let qkv = load(
                &[name("self_attn.q_proj.weight"), name("self_attn.k_proj.weight"), name("self_attn.v_proj.weight")],
                &[vec![h, h], vec![c.kv_heads * c.head_dim, h], vec![c.kv_heads * c.head_dim, h]],
            )?;
            let output = load(&[name("self_attn.o_proj.weight")], &[vec![h, h]])?;
            let gate_up = load(&[name("mlp.gate_proj.weight"), name("mlp.up_proj.weight")], &[vec![i, h], vec![i, h]])?;
            let down = load(&[name("mlp.down_proj.weight")], &[vec![h, i]])?;
            let (qkv_bias, output_bias, gate_up_bias, down_bias) = match &zero_bias {
                Some(zero) => (zero.clone(), zero.clone(), zero.clone(), zero.clone()),
                None => (
                    load(
                        &[name("self_attn.q_proj.bias"), name("self_attn.k_proj.bias"), name("self_attn.v_proj.bias")],
                        &[vec![h], vec![c.kv_heads * c.head_dim], vec![c.kv_heads * c.head_dim]],
                    )?,
                    load(&[name("self_attn.o_proj.bias")], &[vec![h]])?,
                    load(&[name("mlp.gate_proj.bias"), name("mlp.up_proj.bias")], &[vec![i], vec![i]])?,
                    load(&[name("mlp.down_proj.bias")], &[vec![h]])?,
                ),
            };
            layers.push(Layer {
                norm1,
                norm2,
                qkv,
                qkv_bias,
                output,
                output_bias,
                gate_up,
                gate_up_bias,
                down,
                down_bias,
            });
            if !gpu.direct_upload {
                // wgpu retains mapped-at-creation staging copies until submission completes.
                // Flush each layer so loading a large model does not require twice its GPU memory.
                gpu.flush_uploads()?;
            }
        }
        let norm = load(&["model.norm.weight".into()], &[vec![h]])?;
        let embed = checkpoint.bf16("model.embed_tokens.weight", &[c.vocab, h])?;
        let head = checkpoint.bf16("lm_head.weight", &[c.vocab, h])?;
        used.extend(["model.embed_tokens.weight".to_string(), "lm_head.weight".to_string()]);
        ensure!(checkpoint.names().iter().all(|n| used.contains(n)), "checkpoint has unsupported tensors");
        gpu.flush_uploads()?;
        let evidence_path = dir.join("evidence_head.pt");
        let evidence =
            if evidence_path.exists() { Some(EvidenceHead::load(&evidence_path, h, &Device::Cpu)?) } else { None };
        Ok(Weights { layers, norm, checkpoint, embed, head, evidence })
    }

    /// Rows `ids` of a `[vocab, hidden]` BF16 matrix, each value converted by `convert`.
    fn rows<T>(&self, matrix: Bf16, ids: &[u32], convert: impl Fn(f32) -> T) -> Result<Vec<T>> {
        let bytes = self.weights.checkpoint.rows(matrix, self.cfg.hidden * 2, ids)?;
        Ok(bytes.chunks_exact(2).map(|b| convert(bf16(b))).collect())
    }

    fn gemm(&self, e: &mut wgpu::CommandEncoder, x: &Buffer, w: &Buffer, y: &Buffer, shape: [usize; 3]) -> Result<()> {
        let [m, n, k] = shape;
        self.gpu.dispatch(
            e,
            "gemm",
            &[x, w, y],
            [m as u32, n as u32, k as u32, 0, 0, 0, 0, 0],
            [n.div_ceil(32) as u32, m.div_ceil(32) as u32, 1],
        )
    }
    fn norm(&self, e: &mut wgpu::CommandEncoder, x: &Buffer, w: &Buffer, y: &Buffer, n: usize) -> Result<()> {
        self.gpu.dispatch(
            e,
            "norm",
            &[x, w, y],
            [n as u32, self.cfg.hidden as u32, (self.cfg.rms_eps as f32).to_bits(), 0, 0, 0, 0, 0],
            [n as u32, 1, 1],
        )
    }
    fn gather(&self, e: &mut wgpu::CommandEncoder, x: &Buffer, read: &Buffer, y: &Buffer, n: usize) -> Result<()> {
        let h = self.cfg.hidden;
        self.gpu.dispatch(e, "gather", &[x, read, y], [n as u32, h as u32, 0, 0, 0, 0, 0, 0], linear_groups(n * h))
    }
    fn elementwise(
        &self,
        e: &mut wgpu::CommandEncoder,
        x: &Buffer,
        bias: &Buffer,
        y: &Buffer,
        shape: [usize; 2],
        mode: u32,
    ) -> Result<()> {
        let [n, h] = shape;
        self.gpu.dispatch(
            e,
            "elementwise",
            &[x, bias, y],
            [n as u32, h as u32, mode, 0, 0, 0, 0, 0],
            linear_groups(n * h),
        )
    }

    fn mark(&self, encoder: &mut wgpu::CommandEncoder, start: &mut Instant, section: &'static str) -> Result<()> {
        if let Some(profile) = &self.profile {
            self.gpu.submit(std::mem::replace(encoder, self.gpu.encoder()));
            self.gpu.sync()?;
            *profile.borrow_mut().entry(section).or_default() += start.elapsed().as_secs_f64() * 1e3;
            *start = Instant::now();
        }
        Ok(())
    }

    /// One packed tree. Scratch is bounded by the actual token count and reused across layers.
    fn forward(
        &self,
        p: &Packed,
        read: &[usize],
        capture: bool,
        gate: &mut CallGate<'_>,
    ) -> Result<(Option<Buffer>, Vec<Buffer>)> {
        let c = &self.cfg;
        let gpu = &self.gpu;
        ensure!(self.state_cache_bytes == 0, "Intel state cache is not implemented; use --state-cache-mb 0");
        let ranges = tree_ranges(p, c)?;
        ensure!(!read.is_empty() && read.iter().all(|&t| t < p.ids.len()), "invalid Intel readout positions");
        let past = (!capture)
            .then(|| {
                self.prefixes
                    .iter()
                    .filter(|prefix| {
                        p.prefix_len >= prefix.ids.len()
                            && p.ids.starts_with(&prefix.ids)
                            && read.iter().all(|&t| t >= prefix.ids.len())
                    })
                    .max_by_key(|prefix| prefix.ids.len())
            })
            .flatten();
        let np = past.map_or(0, |prefix| prefix.ids.len());
        let n = p.ids.len() - np;
        let h = c.hidden;
        let i = c.intermediate;
        let q = (c.heads + 2 * c.kv_heads) * c.head_dim;
        let mut x = gpu.upload("embeddings", &self.rows(self.weights.embed, &p.ids[np..], f16_bits)?)?;
        let mut cs = vec![0f32; n * c.head_dim];
        let inv: Vec<f64> =
            (0..c.head_dim / 2).map(|j| 1.0 / c.rope_theta.powf((2 * j) as f64 / c.head_dim as f64)).collect();
        for (&pos, row) in p.pos[np..].iter().zip(cs.chunks_exact_mut(c.head_dim)) {
            let (cos, sin) = row.split_at_mut(c.head_dim / 2);
            crate::rope_angles(&inv, pos as f64, cos, sin);
        }
        let cs = gpu.upload("RoPE", &cs)?;
        let ranges = gpu.upload("tree ranges", &ranges[np * 17..])?;
        let nr = read.len();
        let read = gpu.upload("readout indices", &read.iter().map(|&v| (v - np) as u32).collect::<Vec<_>>())?;
        let scratch_rows = n.max(nr);
        let a = gpu.buffer("normalised", scratch_rows * h * 2)?;
        let projected = gpu.buffer("qkv projection", n * q * 2)?;
        let qkv = gpu.buffer("qkv rotated", n * q * 4)?;
        let attention = gpu.buffer("attention output", n * h * 2)?;
        let y = gpu.buffer("projection", scratch_rows * h * 2)?;
        let gu = gpu.buffer("gate/up", scratch_rows * i * 4)?;
        let activated = gpu.buffer("activated", scratch_rows * i * 2)?;
        let final_x = gpu.buffer("last residual", nr * h * 2)?;
        let final_att = gpu.buffer("last attention", nr * h * 2)?;
        gpu.checked(|| {
            let mut captured = Vec::new();
            for (li, layer) in self.weights.layers.iter().enumerate() {
                let mut start = Instant::now();
                let mut e = gpu.encoder();
                self.norm(&mut e, &x, &layer.norm1, &a, n)?;
                self.mark(&mut e, &mut start, "rms_norm")?;
                self.gemm(&mut e, &a, &layer.qkv, &projected, [n, q, h])?;
                self.mark(&mut e, &mut start, "qkv_proj")?;
                gpu.dispatch(
                    &mut e,
                    "rope",
                    &[&projected, &layer.qkv_bias, &cs, &qkv],
                    [n as u32, q as u32, c.head_dim as u32, ((c.heads + c.kv_heads) * c.head_dim) as u32, 0, 0, 0, 0],
                    linear_groups(n * q),
                )?;
                self.mark(&mut e, &mut start, "bias+rope")?;
                if capture {
                    let saved = gpu.buffer("prefix qkv", n * q * 4)?;
                    e.copy_buffer_to_buffer(&qkv, 0, &saved, 0, (n * q * 4) as u64);
                    captured.push(saved);
                    if li + 1 == c.num_layers {
                        gpu.submit(e);
                        break;
                    }
                }
                gpu.dispatch(
                    &mut e,
                    "attention",
                    // Without a cached prefix (np = 0) the kernel never reads binding 2.
                    &[&qkv, &ranges, past.map_or(&qkv, |prefix| &prefix.kv[li]), &attention],
                    [n as u32, c.heads as u32, c.kv_heads as u32, c.head_dim as u32, np as u32, 0, 0, 0],
                    [n.div_ceil(64 / gpu.info.subgroup_max_size as usize) as u32, c.heads as u32, 1],
                )?;
                self.mark(&mut e, &mut start, "attention")?;
                let last = li + 1 == c.num_layers;
                let (m, att) = if last {
                    self.gather(&mut e, &x, &read, &final_x, nr)?;
                    self.gather(&mut e, &attention, &read, &final_att, nr)?;
                    x = final_x.clone();
                    (nr, &final_att)
                } else {
                    (n, &attention)
                };
                self.gemm(&mut e, att, &layer.output, &y, [m, h, h])?;
                self.elementwise(&mut e, &y, &layer.output_bias, &x, [m, h], 0)?;
                self.mark(&mut e, &mut start, "o_proj+residual")?;
                self.norm(&mut e, &x, &layer.norm2, &a, m)?;
                self.mark(&mut e, &mut start, "rms_norm")?;
                self.gemm(&mut e, &a, &layer.gate_up, &gu, [m, 2 * i, h])?;
                self.mark(&mut e, &mut start, "gate_up_proj")?;
                self.elementwise(&mut e, &gu, &layer.gate_up_bias, &activated, [m, i], 1)?;
                self.mark(&mut e, &mut start, "bias+silu*up")?;
                self.gemm(&mut e, &activated, &layer.down, &y, [m, h, i])?;
                self.elementwise(&mut e, &y, &layer.down_bias, &x, [m, h], 0)?;
                self.mark(&mut e, &mut start, "down_proj+residual")?;
                gpu.submit(e);
                // Bound queued work to one layer. On i915, queuing an entire max forward caused
                // fence expiration and a device reset even for fewer than 1024 tokens.
                // This also completes GPU work before a long-lane handover; arithmetic is unchanged.
                gate.confirm(gpu.sync())?;
                if let Some(max) = &self.residual_max {
                    let data = gpu.read(&x, m * h * 2)?;
                    max.borrow_mut().push(
                        data.chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32().abs())
                            .fold(0f32, f32::max),
                    );
                }
                gate.checkpoint()?;
            }
            if capture {
                gate.confirm(gpu.sync())?;
                return Ok((None, captured));
            }
            let output = gpu.buffer("normalised readout", nr * h * 2)?;
            let mut e = gpu.encoder();
            self.norm(&mut e, &x, &self.weights.norm, &output, nr)?;
            gpu.submit(e);
            Ok((Some(output), captured))
        })
    }

    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// Device buffers of one [`IntelBackend::forward`] of `n` tokens with `nr` readouts (exactly the buffers it
    /// creates, each rounded up to 4 bytes as `Gpu::buffer` does). Uploads (`Gpu::upload`, mapped at creation) also
    /// hold a wgpu staging copy of the same size until the upload's copy completes, unless the device maps its
    /// buffers directly (integrated GPUs). Prefix K/V captures (`add_prefix`) are resident, not request work.
    fn forward_work(&self, n: u64, nr: u64) -> crate::Tally {
        let c = &self.cfg;
        let u = |v: usize| v as u64;
        let (h, hd, i) = (u(c.hidden), u(c.head_dim), u(c.intermediate));
        let q = (u(c.heads) + 2 * u(c.kv_heads)) * hd;
        let uploads = if self.gpu.direct_upload { 1 } else { 2 };
        let s = n.max(nr);
        let buf = |t: &mut crate::Tally, bytes: u64| t.add(bytes.saturating_add(3) / 4 * 4);
        let mut t = crate::Tally::default();
        for bytes in
            [crate::prod(&[n, h, 2]), crate::prod(&[n, hd, 4]), crate::prod(&[n, 17, 4]), crate::prod(&[nr, 4])]
        {
            for _ in 0..uploads {
                buf(&mut t, bytes);
            }
        }
        for bytes in [
            crate::prod(&[s, h, 2]),  // normalised
            crate::prod(&[n, q, 2]),  // qkv projection
            crate::prod(&[n, q, 4]),  // qkv rotated
            crate::prod(&[n, h, 2]),  // attention output
            crate::prod(&[s, h, 2]),  // projection
            crate::prod(&[s, i, 4]),  // gate/up
            crate::prod(&[s, i, 2]),  // activated
            crate::prod(&[nr, h, 2]), // last residual
            crate::prod(&[nr, h, 2]), // last attention
            crate::prod(&[nr, h, 2]), // normalised readout
        ] {
            buf(&mut t, bytes);
        }
        t
    }
    pub fn load_gemm_table(&mut self, _: &serde_json::Value, _: &str) -> Result<usize> {
        bail!("CUDA GEMM tables cannot be used by Intel Vulkan")
    }
    /// Quantized projection weights are CUDA-only: never any dequantization format, time or values.
    pub fn dequant_profile(&self) -> Option<(Quantization, f64, u64)> {
        None
    }
    pub fn memory(&self) -> Result<GpuMemory> {
        telemetry::Probe::new(&self.gpu.device, &self.gpu.info).snapshot()
    }
    pub fn memory_probe(&self) -> GpuMemoryProbe {
        GpuMemoryProbe::intel(telemetry::Probe::new(&self.gpu.device, &self.gpu.info))
    }
}

impl Backend for IntelBackend {
    fn add_prefix(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || self.prefixes.iter().any(|p| p.ids == ids) {
            return Ok(());
        }
        ensure!(
            self.prefixes.len() < 2 && ids.len() <= 256,
            "Intel template cache supports at most two prefixes of at most 256 tokens"
        );
        let (gate, stop, health) = (self.gate.clone(), self.stop.clone(), self.gpu.health.clone());
        let mut call = CallGate::enter(gate.as_ref(), stop.as_ref(), &health)?;
        let packed = basal_core::pack::pack(&[ids.to_vec()]);
        let (_, kv) = self.forward(&packed, &[ids.len() - 1], true, &mut call)?;
        self.prefixes.push(Prefix { ids: ids.to_vec(), kv });
        Ok(())
    }

    fn describe(&self) -> Value {
        json!({"backend":"intel-vulkan","device":self.gpu.info.name,
        "driver":self.gpu.info.driver_info,"tensor_library":"wgpu 30.0.1, WGSL",
        "upload":if self.gpu.direct_upload { "direct mapped unified memory" } else { "device local with per-layer staging flush" },
        "weights":"f16 from checkpoint bf16; embedding/head rows read from the checkpoint on demand", "activations":"f16",
        "gemm":"32x32x32 tiled, f32 FMA accumulation", "attention":"tree ancestor ranges, online softmax and RoPE in f32",
        "readout":"selected lm_head rows in f32", "last_layer":"o_proj and MLP at readout positions only",
        "prefix_kv":self.prefixes.iter().map(|p|p.ids.len()).collect::<Vec<_>>(),"state_cache_bytes":0})
    }
    fn fork(&self) -> Result<Self> {
        ensure!(self.profile.is_none() && self.residual_max.is_none(), "fork: diagnostics are per backend");
        Ok(Self {
            prefixes: self.prefixes.clone(),
            gpu: Arc::clone(&self.gpu),
            weights: Arc::clone(&self.weights),
            cfg: self.cfg.clone(),
            readout: self.readout,
            gate: None,
            stop: None,
            load_s: 0.0,
            state_cache_bytes: self.state_cache_bytes,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            profile: None,
            residual_max: None,
            last_timing: (0.0, 0.0),
        })
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        self.gate = Some((gate, WorkClass::from_urgent(urgent)));
    }
    fn set_device_work_class(
        &mut self,
        device: usize,
        gate: Arc<Gate>,
        class: WorkClass,
    ) -> std::result::Result<(), ExecutionError> {
        if device != 0 {
            return Err(ExecutionError::NoSuchDevice { device, devices: vec![0] });
        }
        self.gate = Some((gate, class));
        Ok(())
    }
    /// Every layer's commands are confirmed complete before a checkpoint; stop checks at entry and every layer.
    fn background_support(&self, _max_forward_tokens: usize) -> BackgroundSupport {
        if self.profile.is_some() || self.residual_max.is_some() {
            return BackgroundSupport::Unsupported("diagnostics are enabled on this backend".into());
        }
        if self.state_cache_bytes > 0 {
            return BackgroundSupport::Unsupported("the Intel state cache is not implemented".into());
        }
        BackgroundSupport::Supported
    }
    /// Letter forward: `IntelBackend::forward_work` of the largest packed row (each row is its own forward, its
    /// buffers are released at the next device poll, which the readback performs) plus the readout's letter rows
    /// (uploaded), logits and readback staging; evidence forward: the forward plus the hidden-row readback.
    fn device_work_bound(&self, shape: &basal_core::planner::DeviceWorkShape) -> basal_core::planner::DeviceWorkBound {
        use basal_core::planner::{DeviceBytes, DeviceWorkBound, DeviceWorkEstimate};
        if self.state_cache_bytes > 0 {
            return DeviceWorkBound::Unknown("the Intel state cache is not implemented".into());
        }
        if !shape.media.is_empty() {
            return DeviceWorkBound::Unknown("this backend takes no media".into());
        }
        let h = self.cfg.hidden as u64;
        let (n, nr, l) = (shape.forward_tokens, shape.forward_readouts, shape.forward_letters);
        let uploads = if self.gpu.direct_upload { 1 } else { 2 };
        let mut letter = self.forward_work(n, nr);
        letter.add_n(uploads, crate::prod(&[l, h, 2]));
        letter.add_n(2, crate::prod(&[nr, l, 4]));
        if self.residual_max.is_some() {
            letter.add(crate::prod(&[n.max(nr), h, 2]));
        }
        let mut t = letter;
        if let (Some(e), true) = (shape.evidence, self.weights.evidence.is_some()) {
            let mut ev = self.forward_work(e.tokens, e.readouts);
            ev.add(crate::prod(&[e.readouts, h, 2]));
            t = t.peak(ev);
        }
        DeviceWorkBound::Estimate(DeviceWorkEstimate {
            devices: vec![DeviceBytes { device: 0, bytes: t.bytes, allocations: t.allocations }],
            basis: format!(
                "Intel Vulkan f16; row <= {n} tokens, {nr} readouts, {l} letters; uploads {}; evidence {:?}; later \
                 rounds worst case: {}",
                if self.gpu.direct_upload { "mapped directly" } else { "with wgpu staging copies" },
                shape.evidence.map(|e| e.tokens),
                shape.worst_case_rounds
            ),
        })
    }
    /// Background forks: no state cache (Intel has none; a nonzero setting would be refused per call). Template
    /// prefixes stay: they are resident pinned K/V shared with the online engine.
    fn configure_background(&mut self) -> std::result::Result<(), ExecutionError> {
        if self.profile.is_some() || self.residual_max.is_some() {
            return Err(ExecutionError::Unsupported { reason: "diagnostics are enabled on this backend".into() });
        }
        self.state_cache_bytes = 0;
        Ok(())
    }
    /// Free the buffers of finished or failed calls now: one non-blocking device poll (wgpu releases dropped
    /// buffers only at device maintenance).
    fn release_work(&mut self) -> std::result::Result<(), ExecutionError> {
        self.gpu.maintain().map_err(|e| ExecutionError::Device(e.into()))
    }
    /// A token is accepted only with the device's background gate.
    fn set_stop_token(&mut self, stop: Option<StopToken>) -> std::result::Result<(), ExecutionError> {
        if stop.is_some() && !matches!(self.gate, Some((_, WorkClass::Background))) {
            return Err(ExecutionError::InvalidState {
                reason: "background work needs the background gate of the Intel GPU".into(),
            });
        }
        self.stop = stop;
        Ok(())
    }
    fn host_staging(&self) -> Option<basal_core::planner::HostStaging> {
        let (h, d) = (self.cfg.hidden, self.cfg.head_dim);
        // per token: embedding rows read as bf16 and converted to f16, RoPE cos/sin, tree ranges (17 u32, built and
        // sliced), readout indices; per letter: gathered head rows (bf16 read, converted) and logits; evidence
        // questions read the hidden rows back (f16 bytes, f32 values and a CPU tensor)
        Some(basal_core::planner::HostStaging {
            fixed: 1 << 20,
            per_token: (h * 2 * 2 + d * 4 + 17 * 4 * 2 + 64) as u64,
            per_row: 1024,
            per_readout: 64,
            per_letter: (h * 2 * 2 + 16) as u64,
            per_media_value: 0,
            per_evidence_token: (h * 2 + h * 4 * 2) as u64,
        })
    }
    fn device_health(&self) -> Option<DeviceHealth> {
        Some(self.gpu.health.clone())
    }
    fn has_evidence(&self) -> bool {
        self.weights.evidence.is_some()
    }
    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(t0 < t1 && t1 <= ids.len(), "invalid evidence token range");
        let (gate, stop, health) = (self.gate.clone(), self.stop.clone(), self.gpu.health.clone());
        let mut call = CallGate::enter(gate.as_ref(), stop.as_ref(), &health)?;
        let p = basal_core::pack::pack(&[ids.to_vec()]);
        let read: Vec<usize> = (t0..t1).chain(std::iter::once(ids.len() - 1)).collect();
        let hidden = self.forward(&p, &read, false, &mut call)?.0.context("missing evidence readout")?;
        let bytes = call.confirm(self.gpu.read(&hidden, read.len() * self.cfg.hidden * 2))?;
        let values: Vec<f32> = bytes.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        ensure!(values.iter().all(|v| v.is_finite()), "non-finite Intel evidence hidden states");
        let h = Tensor::from_vec(values, (read.len(), self.cfg.hidden), &Device::Cpu)?;
        self.weights
            .evidence
            .as_ref()
            .context("no evidence head")?
            .scores(&h.narrow(0, 0, t1 - t0)?, &h.narrow(0, t1 - t0, 1)?)
    }
    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        ensure!(!rows.is_empty() && rows.len() == letters.len(), "invalid Intel row/letter count");
        let (gate, stop, health) = (self.gate.clone(), self.stop.clone(), self.gpu.health.clone());
        let mut call = CallGate::enter(gate.as_ref(), stop.as_ref(), &health)?;
        let start = Instant::now();
        let mut out = Vec::with_capacity(rows.len());
        for (row, groups) in rows.iter().zip(letters) {
            ensure!(
                row.last.len() == groups.len() && !groups.is_empty() && groups.iter().all(|g| !g.is_empty()),
                "invalid Intel readouts"
            );
            let hidden = self.forward(row, &row.last, false, &mut call)?.0.context("missing letter readout")?;
            let mut uniq: Vec<u32> = groups.iter().flatten().copied().collect();
            uniq.sort_unstable();
            uniq.dedup();
            let weights = self.gpu.upload("letter weights", &self.rows(self.weights.head, &uniq, |v| v)?)?;
            let logits = self.gpu.buffer("letter logits", groups.len() * uniq.len() * 4)?;
            self.gpu.checked(|| {
                let mut e = self.gpu.encoder();
                self.gpu.dispatch(
                    &mut e,
                    "readout",
                    &[&hidden, &weights, &logits],
                    [groups.len() as u32, uniq.len() as u32, self.cfg.hidden as u32, 0, 0, 0, 0, 0],
                    [groups.len() as u32, uniq.len() as u32, 1],
                )?;
                self.gpu.submit(e);
                Ok(())
            })?;
            let bytes = call.confirm(self.gpu.read(&logits, groups.len() * uniq.len() * 4))?;
            let values: Vec<f32> =
                bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite Intel letter logits");
            out.push(
                groups
                    .iter()
                    .enumerate()
                    .map(|(r, ids)| {
                        ids.iter()
                            .map(|id| {
                                let idx =
                                    uniq.binary_search(id).expect("every requested letter was inserted into the union");
                                let v = values[r * uniq.len() + idx];
                                if self.readout == Readout::Bf16Rounded {
                                    crate::bf16_round(v)
                                } else {
                                    v
                                }
                            })
                            .collect()
                    })
                    .collect(),
            );
        }
        self.last_timing = (0.0, start.elapsed().as_secs_f64() * 1e3);
        Ok(out)
    }
}

/// Name, driver and memory budget of the Intel Vulkan device the backend would select.
pub struct DeviceInfo {
    pub name: String,
    pub driver: String,
    /// An error when the driver does not provide the optional memory telemetry.
    pub memory: Result<GpuMemory>,
}

/// Inspect the selected Intel Vulkan device before loading a model, through one device.
pub fn info() -> std::result::Result<DeviceInfo, IntelError> {
    let gpu = Gpu::new().map_err(IntelError::Device)?;
    let memory = telemetry::Probe::new(&gpu.device, &gpu.info).snapshot();
    Ok(DeviceInfo { name: gpu.info.name.clone(), driver: gpu.info.driver_info.clone(), memory })
}

#[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
pub(crate) fn memory(device: usize) -> Result<GpuMemory> {
    ensure!(device == 0, "device {device}: {SINGLE_DEVICE}");
    let gpu = Gpu::new()?;
    telemetry::Probe::new(&gpu.device, &gpu.info).snapshot()
}

fn bf16(bytes: &[u8]) -> f32 {
    half::bf16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

fn f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// Workgroups for the 1D kernels (128 invocations, index `x + y * 65535 * 128`), split into 2D past 65535 groups.
fn linear_groups(items: usize) -> [u32; 3] {
    let groups = items.div_ceil(128) as u32;
    if groups > 65535 {
        [65535, groups.div_ceil(65535), 1]
    } else {
        [groups, 1, 1]
    }
}

/// Validate the public Packed boundary, including backward-only parents, contiguous nodes and logical positions.
fn tree_ranges(p: &Packed, c: &LlamaConfig) -> Result<Vec<u32>> {
    let n = p.ids.len();
    ensure!(n > 0 && n <= 32768 && p.pos.len() == n && p.node.len() == n, "invalid Intel packed dimensions");
    ensure!(
        p.prefix_len == p.node.iter().take_while(|&&node| node == 0).count(),
        "Intel prefix length disagrees with the root block"
    );
    ensure!(
        p.ids.iter().all(|&i| (i as usize) < c.vocab) && p.pos.iter().all(|&i| (i as usize) < c.max_positions),
        "token/position exceeds model limits"
    );
    ensure!(!p.parent.is_empty() && p.parent.len() <= n + 1, "invalid Intel tree nodes");
    for (i, &parent) in p.parent.iter().enumerate() {
        ensure!(parent >= -1 && parent < (i as i32), "invalid Intel tree parent");
    }
    let mut spans = vec![(n, 0usize, 0usize); p.parent.len()];
    for (t, &b) in p.node.iter().enumerate() {
        let s = spans.get_mut(b as usize).context("invalid Intel node index")?;
        s.0 = s.0.min(t);
        s.1 = t + 1;
        s.2 += 1;
    }
    for &(start, end, count) in &spans {
        ensure!(count == 0 || end - start == count, "non-contiguous Intel tree node");
    }
    let mut out = vec![0u32; n * 17];
    for t in 0..n {
        let mut chain = Vec::new();
        let mut node = p.node[t] as i32;
        let mut depth = 0;
        while node >= 0 {
            depth += 1;
            ensure!(depth <= basal_core::pack::MAX_TREE_DEPTH, "Intel tree depth exceeds eight");
            let (start, end, count) = spans[node as usize];
            if count > 0 {
                chain.push((start, end.min(t + 1)));
            }
            node = p.parent[node as usize];
        }
        chain.reverse();
        let dst = &mut out[t * 17..(t + 1) * 17];
        dst[0] = chain.len() as u32;
        let mut visible = 0;
        for (j, (start, end)) in chain.into_iter().enumerate() {
            ensure!(start < end && end <= t + 1, "Intel tree ancestor follows query");
            dst[1 + 2 * j] = start as u32;
            dst[2 + 2 * j] = end as u32;
            visible += end - start;
        }
        ensure!(p.pos[t] as usize + 1 == visible, "Intel tree position disagrees with ancestry");
    }
    Ok(out)
}
