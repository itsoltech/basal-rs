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
use basal_core::{gate::Gate, manifest::LlamaConfig, pack::Packed, Backend};
use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use wgpu::Buffer;

use crate::{EvidenceHead, GpuMemory, GpuMemoryProbe, Kernels, Precision, Readout};
use checkpoint::{Bf16, Checkpoint};
use runtime::Gpu;

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

/// An Intel model. Forks share immutable weights/device/pipelines, with independent forward scratch buffers.
pub struct IntelBackend {
    prefixes: Vec<Prefix>,
    gpu: Arc<Gpu>,
    weights: Arc<Weights>,
    cfg: LlamaConfig,
    readout: Readout,
    gate: Option<(Arc<Gate>, bool)>,
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
    pub fn load(
        dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
    ) -> std::result::Result<Self, IntelError> {
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
        let mut load = |names: &[String], shapes: &[Vec<usize>]| -> Result<Buffer> {
            let mut data = Vec::<u16>::new();
            for (name, shape) in names.iter().zip(shapes) {
                let tensor = checkpoint.bf16(name, shape)?;
                used.insert(name.clone());
                let elements = shape.iter().product::<usize>();
                ensure!(
                    ((data.len() + elements) as u64) * 2 <= gpu.limits.max_storage_buffer_binding_size,
                    "weight tensor exceeds the Vulkan storage binding limit"
                );
                data.try_reserve(elements)?;
                checkpoint.read_chunks(tensor, &mut chunk, |bytes| {
                    data.extend(bytes.chunks_exact(2).map(|b| f16_bits(bf16(b))));
                })?;
            }
            gpu.upload(&names[0], &data)
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
    fn forward(&self, p: &Packed, read: &[usize], capture: bool) -> Result<(Option<Buffer>, Vec<Buffer>)> {
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
                gpu.sync()?;
                if let Some(max) = &self.residual_max {
                    let data = gpu.read(&x, m * h * 2)?;
                    max.borrow_mut().push(
                        data.chunks_exact(2)
                            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32().abs())
                            .fold(0f32, f32::max),
                    );
                }
                if let Some((gate, false)) = &self.gate {
                    gate.checkpoint();
                }
            }
            if capture {
                gpu.sync()?;
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
    pub fn load_gemm_table(&mut self, _: &Path) -> Result<usize> {
        bail!("CUDA GEMM tables cannot be used by Intel Vulkan")
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
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, u)| g.enter(*u));
        let packed = basal_core::pack::pack(&[ids.to_vec()]);
        let (_, kv) = self.forward(&packed, &[ids.len() - 1], true)?;
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
        self.gate = Some((gate, urgent));
    }
    fn has_evidence(&self) -> bool {
        self.weights.evidence.is_some()
    }
    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(t0 < t1 && t1 <= ids.len(), "invalid evidence token range");
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, u)| g.enter(*u));
        let p = basal_core::pack::pack(&[ids.to_vec()]);
        let read: Vec<usize> = (t0..t1).chain(std::iter::once(ids.len() - 1)).collect();
        let hidden = self.forward(&p, &read, false)?.0.context("missing evidence readout")?;
        let bytes = self.gpu.read(&hidden, read.len() * self.cfg.hidden * 2)?;
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
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, u)| g.enter(*u));
        let start = Instant::now();
        let mut out = Vec::with_capacity(rows.len());
        for (row, groups) in rows.iter().zip(letters) {
            ensure!(
                row.last.len() == groups.len() && !groups.is_empty() && groups.iter().all(|g| !g.is_empty()),
                "invalid Intel readouts"
            );
            let hidden = self.forward(row, &row.last, false)?.0.context("missing letter readout")?;
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
            let bytes = self.gpu.read(&logits, groups.len() * uniq.len() * 4)?;
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

/// Name and driver of the Intel Vulkan device the backend would select.
pub struct DeviceInfo {
    pub name: String,
    pub driver: String,
}

/// Inspect the selected Intel Vulkan device before loading a model.
pub fn info() -> std::result::Result<DeviceInfo, IntelError> {
    let gpu = Gpu::new().map_err(IntelError::Device)?;
    Ok(DeviceInfo { name: gpu.info.name.clone(), driver: gpu.info.driver_info.clone() })
}

#[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
pub(crate) fn memory() -> Result<GpuMemory> {
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
