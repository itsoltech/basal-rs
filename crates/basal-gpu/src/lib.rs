//! basal-1.0 forward on a GPU through candle: Apple Metal (matmul = candle's port of the MLX steel GEMM kernels,
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
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use basal_core::gate::Gate;
use basal_core::manifest::LlamaConfig;
use basal_core::pack::Packed;
use basal_core::Backend;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
#[cfg(target_os = "macos")]
use candle_nn::ops::sdpa;
use candle_nn::ops::{rms_norm, softmax_last_dim};
use candle_nn::rotary_emb::rope;
use serde_json::{json, Value};

#[cfg(feature = "cuda")]
mod cublaslt;
mod fused;
#[cfg(target_os = "macos")]
pub mod gemm;
use fused::HeadDims;

/// GPU backend of this build (`basal --version`).
#[cfg(feature = "cuda")]
pub const BUILD_BACKEND: &str = concat!("CUDA, kernels for compute capability ", env!("BASAL_CUDA_COMPUTE_CAP"));
#[cfg(all(not(feature = "cuda"), target_os = "macos"))]
pub const BUILD_BACKEND: &str = "Metal";
#[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
pub const BUILD_BACKEND: &str = "no GPU backend";

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
    wqkv: Tensor,
    bqkv: Tensor,
    wo: Tensor,
    bo: Tensor,
    ln2: Tensor,
    /// [gate; up] projection [2 i, h] and its bias
    wgu: Tensor,
    bgu: Tensor,
    wdown: Tensor,
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
    #[cfg_attr(not(any(feature = "cuda", target_os = "macos")), allow(dead_code))]
    units: Vec<Unit>,
}

/// Queries `q_off..q_off + q_len` (one block of a packed row) and their key ranges: ancestor blocks from the root
/// down, then the own block. Offsets relative to the segment's first token (after its precomputed prefix).
#[derive(Clone, Debug)]
#[cfg_attr(not(any(feature = "cuda", target_os = "macos")), allow(dead_code))]
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

#[derive(Clone)]
struct PrefixKv {
    ids: Vec<u32>,
    kv: LayerKv,
    /// CUDA tensor-core attention: per layer the K and V split into f16 hi / lo planes (`fused::split_hilo`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    hl: Vec<(Tensor, Tensor)>,
    /// Template prefixes are permanent; state prefixes live in the bounded LRU.
    pinned: bool,
    last_use: u64,
}

impl PrefixKv {
    /// Device memory: f32 K/V, plus the same again for the f16 hi / lo planes when present.
    fn bytes(&self, cfg: &LlamaConfig) -> usize {
        prefix_bytes(cfg, self.ids.len(), !self.hl.is_empty())
    }
}

/// The GPU of this build: CUDA device 0 with the `cuda` feature, otherwise the Metal device (macOS).
/// Weight shapes (N, K) of the projection GEMMs of a model: qkv, o, gate|up, down.
pub fn projection_shapes(c: &LlamaConfig) -> Vec<(usize, usize)> {
    let (h, qd, kvd, i) = (c.hidden, c.heads * c.head_dim, c.kv_heads * c.head_dim, c.intermediate);
    vec![(qd + 2 * kvd, h), (h, qd), (2 * i, h), (h, i)]
}

/// Name of the GPU (CUDA device 0), as recorded in GEMM tables.
pub fn gpu_name() -> Result<String> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(d) = gpu_device()? {
        return Ok(d.cuda_stream().context().name()?);
    }
    bail!("gpu_name: CUDA only")
}

/// Version of the cuBLASLt library this build runs with (CUDA), as recorded in GEMM tables.
pub fn cublaslt_version() -> Option<usize> {
    #[cfg(feature = "cuda")]
    return Some(cublaslt::version());
    #[cfg(not(feature = "cuda"))]
    None
}

pub fn gpu_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        let dev = Device::new_cuda(0).context("CUDA device 0")?;
        if let Device::Cuda(d) = &dev {
            use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as A;
            let ctx = d.cuda_stream().context().clone();
            let major = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
            let minor = ctx.attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            let built: i32 = env!("BASAL_CUDA_COMPUTE_CAP").parse().unwrap_or(0);
            ensure!(
                major * 10 + minor >= built,
                "this build targets CUDA compute capability {}.{} and newer, but {} is {major}.{minor}: use a build \
                 for {major}{minor} or lower (CUDA_COMPUTE_CAP; container image tags: latest = 8.9, -sm80 = 8.0, \
                 -sm90 = 9.0)",
                built / 10,
                built % 10,
                ctx.name()?
            );
        }
        Ok(dev)
    }
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    return Device::new_metal(0).context("Metal device");
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    bail!("no GPU backend in this build (macOS: Metal; Linux: build with --features cuda)")
}

pub struct GpuBackend {
    dev: Device,
    /// Precomputed prefixes (prompt template up to the state, and with a state cache the shared prefix of recent
    /// single-row forwards); a row that starts with one of them skips its tokens.
    prefixes: Vec<PrefixKv>,
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
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    lm_head: Tensor,
    inv_freq: Vec<f64>,
    pub load_s: f64,
    /// When set, every forward section is synchronised and timed (slows the forward; diagnostic only).
    pub profile: Option<std::cell::RefCell<BTreeMap<&'static str, f64>>>,
    /// When set, the largest |value| of the residual stream after every layer is recorded (diagnostic: f16 range).
    pub residual_max: Option<std::cell::RefCell<Vec<f32>>>,
    /// Timing of the last call in milliseconds: host preparation, GPU forward + readout (synchronised).
    pub last_timing: (f64, f64),
    /// CUDA: f16/bf16 projections through cuBLASLt with measured algorithms (None = candle's cuBLAS matmul).
    #[cfg(feature = "cuda")]
    lt: Option<Arc<cublaslt::Lt>>,
    /// basal-1.5 evidence head (`evidence_head.pt`, f32): ws, us, we, ue `[k, d]`, bs, be `[1, d]`.
    evidence: Option<EvidenceHead>,
    /// Device shared with another engine ([`Backend::set_gate`]): the gate and whether this one is the urgent lane.
    gate: Option<(Arc<Gate>, bool)>,
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
        let mut header = vec![0u8; usize::try_from(n)?];
        file.read_exact(&mut header)?;
        let header: Value = serde_json::from_slice(&header).context("safetensors header")?;
        let mut ranges = std::collections::HashMap::new();
        for (name, t) in header.as_object().context("safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            let r = t["data_offsets"].as_array().context("data_offsets")?;
            let (s, e) = (r[0].as_u64().context("data_offsets")?, r[1].as_u64().context("data_offsets")?);
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

fn check(st: &MmapedSafetensors, name: &str, shape: &[usize]) -> Result<()> {
    let v = st.get(name).with_context(|| format!("checkpoint tensor {name} missing"))?;
    ensure!(v.shape() == shape, "{name}: shape {:?}, expected {:?}", v.shape(), shape);
    ensure!(v.dtype() == safetensors::Dtype::BF16, "{name}: dtype {:?}, expected BF16", v.dtype());
    Ok(())
}

impl GpuBackend {
    pub fn load(
        model_dir: &Path,
        cfg: &LlamaConfig,
        precision: Precision,
        readout: Readout,
        kernels: Kernels,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let dev = gpu_device()?;
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
        let load_cat = |names: &[String]| -> Result<Tensor> {
            let views = names.iter().map(|n| st.get(n)).collect::<candle_core::Result<Vec<_>>>()?;
            let mut shape = views[0].shape().to_vec();
            shape[0] = views.iter().map(|v| v.shape()[0]).sum();
            let mut bytes = Vec::new();
            for n in names {
                reader.read(n, &mut bytes)?;
            }
            Ok(match dt {
                DType::BF16 => Tensor::from_raw_buffer(&bytes, DType::BF16, &shape, &dev)?,
                DType::F16 => {
                    // on every core (the 11B checkpoint: ~11e9 values)
                    let mut h = vec![half::f16::ZERO; bytes.len() / 2];
                    bf16_to_f16(&bytes, &mut h);
                    Tensor::from_vec(h, shape, &dev)?
                }
                d => bail!("unsupported storage dtype {d:?}"),
            })
        };
        let load = |name: &str| load_cat(&[name.to_string()]);
        // Models without biases (basal-1.5) get zero biases: the fused kernels then add exactly 0, which leaves every
        // rounding point as without a bias (T(T(y) + 0) = T(y)).
        let biases = |names: &[String], len: usize| -> Result<Tensor> {
            if cfg.bias {
                load_cat(names)
            } else {
                Ok(Tensor::zeros(len, dt, &dev)?)
            }
        };
        let qd = cfg.heads * cfg.head_dim;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            let p = |n: &str| format!("model.layers.{l}.{n}");
            let ps = |ns: &[&str]| ns.iter().map(|n| p(n)).collect::<Vec<_>>();
            layers.push(Layer {
                ln1: load(&p("input_layernorm.weight"))?,
                wqkv: load_cat(&ps(&[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ]))?,
                bqkv: biases(
                    &ps(&["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"]),
                    qd + 2 * kvd,
                )?,
                wo: load(&p("self_attn.o_proj.weight"))?,
                bo: biases(&ps(&["self_attn.o_proj.bias"]), h)?,
                ln2: load(&p("post_attention_layernorm.weight"))?,
                wgu: load_cat(&ps(&["mlp.gate_proj.weight", "mlp.up_proj.weight"]))?,
                bgu: biases(&ps(&["mlp.gate_proj.bias", "mlp.up_proj.bias"]), 2 * i)?,
                wdown: load(&p("mlp.down_proj.weight"))?,
                bdown: biases(&ps(&["mlp.down_proj.bias"]), h)?,
            });
        }
        let embed = load("model.embed_tokens.weight")?;
        let norm = load("model.norm.weight")?;
        // lm_head stays in the checkpoint dtype; only the letter rows are gathered and converted to f32 per call.
        let lm_head = {
            let v = st.get("lm_head.weight")?;
            let mut bytes = Vec::new();
            reader.read("lm_head.weight", &mut bytes)?;
            Tensor::from_raw_buffer(&bytes, DType::BF16, v.shape(), &dev)?
        };
        dev.synchronize()?;
        let d = cfg.head_dim;
        let inv_freq = (0..d / 2).map(|j| 1.0 / cfg.rope_theta.powf((2 * j) as f64 / d as f64)).collect();
        Ok(Self {
            prefixes: Vec::new(),
            state_cache_bytes: 0,
            clock: 0,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            cfg: cfg.clone(),
            precision,
            readout,
            kernels,
            embed,
            layers,
            norm,
            lm_head,
            inv_freq,
            load_s: t0.elapsed().as_secs_f64(),
            profile: None,
            residual_max: None,
            last_timing: (0.0, 0.0),
            evidence: {
                let p = model_dir.join("evidence_head.pt");
                if p.exists() {
                    Some(EvidenceHead::load(&p, cfg.hidden, &dev)?)
                } else {
                    None
                }
            },
            #[cfg(feature = "cuda")]
            lt: match &dev {
                Device::Cuda(d) if std::env::var("BASAL_GEMM").as_deref() != Ok("cublas") => {
                    Some(Arc::new(cublaslt::Lt::new(d)?))
                }
                _ => None,
            },
            gate: None,
            dev,
        })
    }

    /// CUDA attention kernel of this forward precision.
    fn attention_kernel(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.dev.is_cuda() {
            return if self.tc_attention() { fused::tc_kernel() } else { "attn_tree_f32" };
        }
        if self.tree_attention() {
            "attn_tree_f32"
        } else {
            "sdpa"
        }
    }

    /// Attention by tree units, invariant to packing: always on CUDA; on Metal unless `BASAL_ATT=sdpa` (the MLX SDPA
    /// kernel over each packed row with an additive mask, the earlier path, kept for comparison).
    fn tree_attention(&self) -> bool {
        self.dev.is_cuda() || std::env::var("BASAL_ATT").as_deref() != Ok("sdpa")
    }

    /// The device for one call (a no-op without a gate).
    fn enter_gate(&self) -> Option<basal_core::gate::GateGuard<'_>> {
        self.gate.as_ref().map(|(g, urgent)| g.enter(*urgent))
    }

    /// Preemptible lane between two layers: wait for the queued work, then let an urgent caller run if one waits.
    fn layer_checkpoint(&self) -> Result<()> {
        if let Some((g, false)) = &self.gate {
            self.dev.synchronize()?;
            g.checkpoint();
        }
        Ok(())
    }

    /// Profiling: synchronise and attribute the time since `*t` to `section`.
    fn mark(&self, t: &mut Instant, section: &'static str) -> Result<()> {
        if let Some(p) = &self.profile {
            self.dev.synchronize()?;
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

    /// `x @ w^T`. candle tiles M by 64 rows; when the last row tile would be at most half used (M mod 64 in 1..=32,
    /// small M) 32-row tiles of the same MLX kernel family skip that padding (measured with `basal gemm-tune`;
    /// results are bitwise identical, the K reduction order does not change).
    /// On CUDA: cuBLASLt with measured algorithms (f16/bf16), otherwise cuBLAS through candle.
    fn matmul_t(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let w = self.w(w)?;
        #[cfg(feature = "cuda")]
        if let (Some(lt), DType::F16 | DType::BF16) = (&self.lt, w.dtype()) {
            if x.is_contiguous() {
                return Ok(cublaslt::matmul_nt(lt, x, &w)?);
            }
        }
        #[cfg(target_os = "macos")]
        {
            let m = x.dim(0)?;
            if self.dev.is_metal() && m <= 600 && (1..=32).contains(&(m % 64)) && x.is_contiguous() {
                return Ok(gemm::gemm_nt(x, &w, gemm::GemmConfig::new((32, 64, 16, 1, 2), 0))?);
            }
        }
        Ok(x.matmul(&w.t()?)?)
    }

    /// f32 attention of one segment: `q [1, NH, L, HD]`, `k, v [1, NKV, LK, HD]`, additive mask `[1, 1, L, LK]`
    /// shared by the heads. Metal: candle's MLX SDPA kernel (GQA in the kernel). CUDA: scores GEMM, one fused
    /// scale + mask + softmax kernel and the weighted-sum GEMM, with the query heads of one KV head stacked as rows
    /// (no K/V repeat).
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, scale: f32) -> Result<Tensor> {
        let (_, nh, l, hd) = q.dims4()?;
        #[cfg(target_os = "macos")]
        if self.dev.is_metal() {
            let m = mask.broadcast_as((1, nh, l, k.dim(2)?))?;
            return Ok(sdpa(q, k, v, Some(&m), false, scale, 1.0)?);
        }
        let rep = nh / k.dim(1)?;
        let nkv = nh / rep;
        let q = q.reshape((1, nkv, rep * l, hd))?;
        let att = fused::masked_softmax(&q.matmul(&k.t()?)?, mask, l, scale)?;
        Ok(att.matmul(v)?.reshape((1, nh, l, hd))?)
    }

    /// Stock-candle forward (first port, `--kernels candle`) on rows padded to `l`: final normalised hidden states
    /// at `read` (flat indices into `[b * l]`). `cos_sin` = `[2, b, l, hd/2]`, `mask` = additive `[b, 1, l, l]` f32.
    #[allow(clippy::too_many_arguments)]
    fn forward_padded(
        &self,
        ids: &[u32],
        cos_sin: &[f32],
        mask: &[f32],
        b: usize,
        l: usize,
        read: &[u32],
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.head_dim);
        let eps = c.rms_eps as f32;
        let dt = self.precision.compute();
        let dev = &self.dev;
        let ids = Tensor::from_slice(ids, (b * l,), dev)?;
        let cs = Tensor::from_slice(cos_sin, (2 * b * l * (hd / 2),), dev)?;
        let mask = Tensor::from_slice(mask, (b, 1, l, l), dev)?;
        let read_t = Tensor::from_slice(read, (read.len(),), dev)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let mut t = Instant::now();
        let mut x = self.embed.index_select(&ids, 0)?.to_dtype(dt)?; // [b*l, h]
        self.mark(&mut t, "embed+inputs")?;
        {
            {
                let rep = nh / nkv;
                let kvd = nkv * hd;
                let half = b * l * (hd / 2);
                let cos = cs.narrow(0, 0, half)?.reshape((b, l, hd / 2))?;
                let sin = cs.narrow(0, half, half)?.reshape((b, l, hd / 2))?;
                let inter = c.intermediate;
                for ly in &self.layers {
                    let a = rms_norm(&x, &self.w(&ly.ln1)?, eps)?;
                    self.mark(&mut t, "rms_norm")?;
                    let qkv = self.linear(&a, &ly.wqkv, &ly.bqkv)?;
                    self.mark(&mut t, "qkv_proj")?;
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
                    self.mark(&mut t, "bias+rope+heads")?;
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
                    self.mark(&mut t, "attention")?;
                    x = (x + self.linear(&o, &ly.wo, &ly.bo)?)?;
                    self.mark(&mut t, "o_proj+bias+residual")?;
                    let a = rms_norm(&x, &self.w(&ly.ln2)?, eps)?;
                    self.mark(&mut t, "rms_norm")?;
                    let g = self.linear(&a, &ly.wgu.narrow(0, 0, inter)?, &ly.bgu.narrow(0, 0, inter)?)?;
                    let u = self.linear(&a, &ly.wgu.narrow(0, inter, inter)?, &ly.bgu.narrow(0, inter, inter)?)?;
                    self.mark(&mut t, "gate_up_proj")?;
                    let act = (g.silu()? * u)?;
                    self.mark(&mut t, "bias+silu*up")?;
                    x = (x + self.linear(&act, &ly.wdown, &ly.bdown)?)?;
                    self.mark(&mut t, "down_proj+bias+residual")?;
                }
                let out = rms_norm(&x.index_select(&read_t, 0)?, &self.w(&self.norm)?, eps)?;
                self.mark(&mut t, "final_norm")?;
                Ok(out)
            }
        }
    }

    /// Fused forward on a compact token list: the tokens of all segments back to back (no padding), so every GEMM,
    /// norm and MLP runs on exactly the real tokens; attention runs per segment over its own keys (optionally after
    /// a precomputed prefix). `cos_sin` = `[2, total, hd/2]`. Returns the final normalised hidden states at `read`
    /// (indices into the compact list) or, with `capture` (one segment, no readout), the K/V of every layer.
    fn forward_ragged(
        &self,
        ids: &[u32],
        cos_sin: &[f32],
        segs: &[Segment],
        read: &[u32],
        capture: bool,
    ) -> Result<(Option<Tensor>, LayerKv)> {
        let c = &self.cfg;
        let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
        let total = ids.len();
        let dims = HeadDims { b: 1, l: total, nh, nkv, hd };
        let eps = c.rms_eps as f32;
        let dt = self.precision.compute();
        let dev = &self.dev;
        ensure!(!capture || segs.len() == 1, "prefix capture takes one segment");
        let ids = Tensor::from_slice(ids, (total,), dev)?;
        let cs = Tensor::from_slice(cos_sin, (2 * total * (hd / 2),), dev)?;
        // CUDA: one attention kernel per layer for all segments, masks in one buffer; also for a prefix capture, so
        // that a cached prefix has bitwise the K/V the same tokens get when computed inside a row.
        let ragged_cuda = cfg!(feature = "cuda") && dev.is_cuda();
        // Metal: the same tree units through attn_tree_f32 of kernels.metal (one dispatch per prefix)
        let ragged_metal = cfg!(target_os = "macos") && dev.is_metal() && self.tree_attention();
        let masks: Vec<Tensor> = if ragged_cuda || ragged_metal {
            Vec::new()
        } else {
            segs.iter()
                .map(|sg| {
                    let lk = sg.np() + sg.len;
                    Ok(Tensor::from_slice(&sg.mask, (1, 1, sg.len, lk), dev)?)
                })
                .collect::<Result<_>>()?
        };
        // CUDA: the blocks of all rows as tree-attention units (compact offsets; prefix addresses set per layer).
        #[cfg(feature = "cuda")]
        let mut units: Vec<(fused::TreeUnit, Option<&PrefixKv>)> = if ragged_cuda {
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
        let (metal_units, metal_pasts) = if ragged_metal {
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
        // (a zero-length device buffer cannot be created on Metal; a captured prefix has no readout)
        let read_t = if read.is_empty() { None } else { Some(Tensor::from_slice(read, (read.len(),), dev)?) };
        let scale = (1.0 / (hd as f64).sqrt()) as f32;
        let mut captured = Vec::new();
        let mut t = Instant::now();
        let mut x = self.embed.index_select(&ids, 0)?.to_dtype(dt)?; // [total, h]
        self.mark(&mut t, "embed+inputs")?;
        let n_layers = self.layers.len();
        for (li, ly) in self.layers.iter().enumerate() {
            let a = rms_norm(&x, &self.w(&ly.ln1)?, eps)?;
            self.mark(&mut t, "rms_norm")?;
            let qkv = self.matmul_t(&a, &ly.wqkv)?;
            self.mark(&mut t, "qkv_proj")?;
            let (_flat, q, k, v) = fused::qkv_rope(&qkv, &self.w(&ly.bqkv)?, &cs, dims)?;
            self.mark(&mut t, "bias+rope+heads")?;
            #[cfg(feature = "cuda")]
            let ragged_o = if ragged_cuda {
                // f16/bf16 forwards: tensor cores (BASAL_ATT=simt keeps the f32 SIMT kernel, for A/B); the f32
                // reference forward always uses the f32 kernel.
                let tc = self.tc_attention();
                for (u, past) in units.iter_mut() {
                    if let Some(p) = past {
                        u.pk = fused::device_addr_f32(&p.kv[li].0)?;
                        u.pv = fused::device_addr_f32(&p.kv[li].1)?;
                        if tc {
                            u.pkh = fused::device_addr_f16(&p.hl[li].0)?;
                            u.pvh = fused::device_addr_f16(&p.hl[li].1)?;
                        }
                    }
                }
                let us: Vec<fused::TreeUnit> = units.iter().map(|(u, _)| u.clone()).collect();
                if tc {
                    let (nq, nk) = (nh * total * hd, nkv * total * hd);
                    let khl = fused::split_hilo(&_flat.narrow(0, nq, nk)?)?;
                    let vhl = fused::split_hilo(&_flat.narrow(0, nq + nk, nk)?)?;
                    let v_exact = self.precision == Precision::F16;
                    Some(fused::attention_tree(&_flat, &us, dims, scale, Some((&khl, &vhl, v_exact)))?)
                } else {
                    Some(fused::attention_tree(&_flat, &us, dims, scale, None)?)
                }
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let ragged_o: Option<Tensor> = None;
            #[cfg(target_os = "macos")]
            let ragged_o = if ragged_metal {
                let pasts: Vec<(&Tensor, &Tensor)> = metal_pasts.iter().map(|p| (&p.kv[li].0, &p.kv[li].1)).collect();
                Some(fused::attention_tree_metal(&_flat, &metal_units, &pasts, dims, scale)?)
            } else {
                ragged_o
            };
            let mut outs = Vec::with_capacity(segs.len());
            for (si, sg) in segs.iter().enumerate() {
                if ragged_o.is_some() && !capture {
                    break;
                }
                let one = segs.len() == 1;
                let part = |x: &Tensor| -> Result<Tensor> {
                    Ok(if one { x.clone() } else { x.narrow(2, sg.off, sg.len)?.contiguous()? })
                };
                let (qs, mut ks, mut vs) = (part(&q)?, part(&k)?, part(&v)?);
                if let Some(p) = sg.past {
                    let (pk, pv) = &p.kv[li];
                    ks = Tensor::cat(&[pk, &ks], 2)?;
                    vs = Tensor::cat(&[pv, &vs], 2)?;
                }
                if capture {
                    captured.push((ks.contiguous()?, vs.contiguous()?));
                    if li + 1 == n_layers {
                        return Ok((None, captured)); // a captured prefix needs no output of the last layer
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
            let mut o = fused::merge_heads(&o, dims, dt)?;
            self.mark(&mut t, "attention")?;
            if let (true, Some(rt)) = (li + 1 == n_layers, &read_t) {
                // Only the readout positions are needed after the last attention.
                o = o.index_select(rt, 0)?;
                x = x.index_select(rt, 0)?;
            }
            x = fused::bias_residual(&x, &self.matmul_t(&o, &ly.wo)?, &self.w(&ly.bo)?)?;
            self.mark(&mut t, "o_proj+bias+residual")?;
            let a = rms_norm(&x, &self.w(&ly.ln2)?, eps)?;
            self.mark(&mut t, "rms_norm")?;
            let gu = self.matmul_t(&a, &ly.wgu)?;
            self.mark(&mut t, "gate_up_proj")?;
            let act = fused::bias_silu_mul(&gu, &self.w(&ly.bgu)?)?;
            self.mark(&mut t, "bias+silu*up")?;
            x = fused::bias_residual(&x, &self.matmul_t(&act, &ly.wdown)?, &self.w(&ly.bdown)?)?;
            self.mark(&mut t, "down_proj+bias+residual")?;
            if let Some(rm) = &self.residual_max {
                let m = x.abs()?.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?;
                rm.borrow_mut().push(m);
                t = Instant::now();
            }
            if li + 1 < n_layers {
                self.layer_checkpoint()?;
            }
        }
        let out = rms_norm(&x, &self.w(&self.norm)?, eps)?;
        self.mark(&mut t, "final_norm")?;
        Ok((Some(out), captured))
    }

    fn rope_row(&self, pos: f64, cos: &mut [f32], sin: &mut [f32]) {
        for (j, f) in self.inv_freq.iter().enumerate() {
            let a = pos * f;
            cos[j] = a.cos() as f32;
            sin[j] = a.sin() as f32;
        }
    }

    /// CUDA attention on tensor cores (f16 / bf16 forwards; `BASAL_ATT=simt` selects the f32 kernel for A/B).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn tc_attention(&self) -> bool {
        self.dev.is_cuda() && self.precision != Precision::F32 && std::env::var("BASAL_ATT").as_deref() != Ok("simt")
    }

    /// The prefix K / V of every layer split into f16 hi / lo planes, when the tensor-core attention reads them.
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

    /// K/V of all layers for `ids` at positions 0..len, continuing a precomputed `past` prefix of it.
    fn capture_prefix(&self, ids: &[u32], past: Option<usize>) -> Result<LayerKv> {
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
        let (_, kv) = self.forward_ragged(&ids[np..], &cs, &[seg], &[], true)?;
        Ok(kv)
    }

    /// State cache: for a single row whose template + state prefix (`Packed::state_len`) extends a known prefix by at least `MIN_STATE_TOKENS`,
    /// compute the root block once, keep its K/V (bounded LRU) and return its index. Exact: the root is visible to
    /// every token of the row, and later rows use it only on an exact token-prefix match.
    fn cache_state(&mut self, row: &Packed, past: Option<usize>) -> Result<Option<usize>> {
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
        let kv = self.capture_prefix(&ids, past)?;
        let hl = self.split_kv(&kv)?;
        let entry = PrefixKv { ids, kv, hl, pinned: false, last_use: self.clock };
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
    /// Projection weight shapes (N, K) that the loaded GEMM table does not cover (CUDA with cuBLASLt; empty
    /// otherwise). Such shapes use algorithms timed at first use, whose results may depend on the batch.
    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            let dt = self.precision.compute();
            return projection_shapes(&self.cfg).into_iter().filter(|&(n, k)| !lt.covers(n, k, dt)).collect();
        }
        Vec::new()
    }

    fn gemm_desc(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if self.lt.as_ref().is_some_and(|lt| lt.is_invariant()) {
            return "cuBLASLt, f32 accumulation, batch-invariant: one algorithm per weight shape for every M, no split-K (--gemm-table from gemm-search --invariant); a row's result does not depend on its batch";
        }
        #[cfg(feature = "cuda")]
        if self.lt.is_some() {
            return "cuBLASLt, f32 accumulation; algorithm per (N, K, M class: 16 up to 512, then 128) from the --gemm-table search, else the fastest of up to 16 heuristic algorithms timed at first use";
        }
        "cuBLAS (candle), f32 accumulation"
    }

    /// cuBLASLt: classes taken from a search table and classes tuned at run time (heuristic timing), as JSON.
    pub fn gemm_tuning(&self) -> Value {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            let t = lt.tuned.lock().unwrap();
            return json!({"from_table": *lt.from_table.lock().unwrap(), "tuned_at_run_time": t.iter().map(|e| json!({"m_class": e.m_class, "n": e.n, "k": e.k, "ms": e.ms, "heuristic_first_ms": e.heuristic_ms})).collect::<Vec<_>>()});
        }
        Value::Null
    }

    /// Use a cuBLASLt algorithm table written by [`gemm_search`] (CUDA; rejected if the cuBLASLt version differs).
    pub fn load_gemm_table(&mut self, path: &Path) -> Result<usize> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            ensure!(
                v["cublaslt_version"].as_u64() == Some(cublaslt::version() as u64),
                "{}: table for cuBLASLt {}, this process uses {}",
                path.display(),
                v["cublaslt_version"],
                cublaslt::version()
            );
            if let (Some(t), Device::Cuda(d)) = (v["gpu"].as_str(), &self.dev) {
                let here = d.cuda_stream().context().name()?;
                ensure!(t == here, "{}: table for {t}, this GPU is {here}", path.display());
            }
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
                    algo: algo.try_into().unwrap(),
                    ms: 0.0,
                    heuristic_ms: 0.0,
                    tried: 0,
                });
            }
            if v["invariant"].as_bool() == Some(true) {
                lt.load_invariant(&entries)?;
            } else {
                lt.load(&entries);
            }
            return Ok(entries.len());
        }
        bail!("--gemm-table needs the CUDA backend with cuBLASLt ({})", path.display())
    }
}

fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

impl Backend for GpuBackend {
    fn has_evidence(&self) -> bool {
        self.evidence.is_some()
    }

    fn evidence_scores(&mut self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let _gpu = self.enter_gate();
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
        let (h, _) = self.forward_ragged(ids, &cs, &[seg], &read, false)?;
        let h = h.context("evidence readout")?.to_dtype(DType::F32)?;
        let m = t1 - t0;
        let head = self.evidence.as_ref().context("this model has no evidence head")?;
        head.scores(&h.narrow(0, 0, m)?, &h.narrow(0, m, 1)?)
    }

    fn fork(&self) -> Result<Self> {
        ensure!(self.profile.is_none() && self.residual_max.is_none(), "fork: diagnostics are per backend");
        Ok(Self {
            dev: self.dev.clone(),
            prefixes: self.prefixes.iter().filter(|p| p.pinned).cloned().collect(),
            state_cache_bytes: self.state_cache_bytes,
            clock: 0,
            state_cache_hits: 0,
            state_cache_inserts: 0,
            cfg: self.cfg.clone(),
            precision: self.precision,
            readout: self.readout,
            kernels: self.kernels,
            embed: self.embed.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            inv_freq: self.inv_freq.clone(),
            load_s: 0.0,
            profile: None,
            residual_max: None,
            last_timing: (0.0, 0.0),
            #[cfg(feature = "cuda")]
            lt: self.lt.clone(),
            evidence: self.evidence.clone(),
            gate: None,
        })
    }

    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        self.gate = Some((gate, urgent));
    }

    fn add_prefix(&mut self, ids: &[u32]) -> Result<()> {
        if self.kernels != Kernels::Fused || ids.is_empty() || self.prefixes.iter().any(|p| p.ids == ids) {
            return Ok(());
        }
        let kv = self.capture_prefix(ids, None)?;
        let hl = self.split_kv(&kv)?;
        self.prefixes.push(PrefixKv { ids: ids.to_vec(), kv, hl, pinned: true, last_use: 0 });
        Ok(())
    }

    fn describe(&self) -> Value {
        let cuda = self.dev.is_cuda();
        json!({
            "backend": if cuda { "cuda" } else { "metal" },
            "device": if cuda { "NVIDIA GPU through candle Device::new_cuda(0)" } else { "Apple GPU through candle Device::new_metal(0)" },
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
            "attention_kernel": self.attention_kernel(),
            "gemm": if cuda { self.gemm_desc() } else { "MLX steel GEMM (candle); 32-row tiles when M mod 64 is in 1..=32 and M <= 600" },
            "prefix_kv": self.prefixes.iter().filter(|p| p.pinned).map(|p| p.ids.len()).collect::<Vec<_>>(),
            "state_cache_bytes": self.state_cache_bytes,
            "last_layer": match self.kernels {
                Kernels::Fused => "o_proj and MLP of the last layer only at the readout positions",
                Kernels::Candle => "full",
            },
        })
    }

    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        let gate = self.gate.clone();
        let _gpu = gate.as_ref().map(|(g, urgent)| g.enter(*urgent));
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
        let (hidden, t1) = if self.kernels == Kernels::Candle {
            let (ids, cs, mask, read) = self.padded_inputs(rows, l);
            let t1 = Instant::now();
            (self.forward_padded(&ids, &cs, &mask, b, l, &read)?, t1)
        } else {
            // Every row uses its own longest precomputed prefix; a single-row forward may add its state to the cache.
            self.clock += 1;
            let mut past: Vec<Option<usize>> = rows.iter().map(|r| self.best_prefix(r)).collect();
            for &i in past.iter().flatten() {
                self.prefixes[i].last_use = self.clock;
                if !self.prefixes[i].pinned {
                    self.state_cache_hits += 1;
                }
            }
            if b == 1 && past[0].is_none_or(|i| self.prefixes[i].pinned) {
                past[0] = self.cache_state(rows[0], past[0])?;
            }
            let total: usize =
                rows.iter().zip(&past).map(|(r, p)| r.ids.len() - p.map_or(0, |i| self.prefixes[i].ids.len())).sum();
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
                let visible: Vec<bool> =
                    (0..nodes * nodes).map(|k| p.is_ancestor_or_self((k / nodes) as u32, (k % nodes) as u32)).collect();
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
            (self.forward_ragged(&ids, &cs, &segs, &read, false)?.0.context("readout")?, t1)
        };
        let ids_t = Tensor::from_slice(&uniq, (uniq.len(),), &self.dev)?;
        #[cfg(feature = "cuda")]
        let cuda_readout = self.dev.is_cuda();
        #[cfg(not(feature = "cuda"))]
        let cuda_readout = false;
        let logits: Vec<Vec<f32>> = if cuda_readout {
            #[cfg(feature = "cuda")]
            {
                fused::letter_logits(&hidden.contiguous()?, &self.lm_head, &ids_t)?.to_vec2()?
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!()
        } else {
            let w = self.lm_head.index_select(&ids_t, 0)?.to_dtype(DType::F32)?;
            hidden.to_dtype(DType::F32)?.matmul(&w.t()?)?.to_vec2()?
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
}

/// Time `x[m,k] @ w[n,k]^T` on the GPU for the projection shapes of the model (synchronised, after warm-up).
/// Returns `(m, k, n, dtype, ms, TFLOP/s)` rows. Diagnostic for comparing GEMM kernels with MLX / cuBLAS.
pub fn gemm_microbench(ms: &[usize], shapes: &[(usize, usize)], dtype: &str, reps: usize) -> Result<Vec<Value>> {
    let dev = gpu_device()?;
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
/// (CUDA, f16 or bf16), on random operands. Returns the table (JSON) for `--gemm-table`.
#[cfg(feature = "cuda")]
pub fn gemm_search(m_classes: &[usize], shapes: &[(usize, usize)], dtype: &str, invariant: bool) -> Result<Value> {
    let dev = gpu_device()?;
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
              "invariant": invariant, "shapes": shapes, "entries": entries,
              "timing": "each configuration timed on rotating copies of the weight larger than L2 (cold weights, as in a forward)",
              "note": "algorithms are opaque cuBLASLt configurations, valid for this GPU and cuBLASLt version"}))
}
