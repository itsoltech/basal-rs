//! Gemma 4 / Gemma 4 Unified typed readout, ported from Transformers 5.17.0 (Apache-2.0).
//! Separate causal prompts and bounded query tiles avoid relying on Llama trie kernels or a batch-dependent mask.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use basal_core::architecture::{GemmaConfig, GemmaLayerConfig};
use basal_core::gate::Gate;
use basal_core::pack::Packed;
use basal_core::Backend;
use candle_core::{DType, Device, Tensor, D};
use serde_json::{json, Value};

use crate::{Precision, Readout};

#[derive(Clone)]
struct Layer {
    input_norm: Tensor,
    post_attn_norm: Tensor,
    pre_mlp_norm: Tensor,
    post_mlp_norm: Tensor,
    scalar: Tensor,
    q: Tensor,
    q_norm: Tensor,
    k: Option<Tensor>,
    k_norm: Option<Tensor>,
    v: Option<Tensor>,
    o: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    ple_gate: Option<Tensor>,
    ple_projection: Option<Tensor>,
    ple_norm: Option<Tensor>,
    head_map: Tensor,
}

struct Weights {
    #[cfg(feature = "cuda")]
    empty_mask_group: Tensor,
    embed: Tensor,
    norm: Tensor,
    layers: Vec<Layer>,
    ple_embed: Option<Tensor>,
    ple_projection: Option<Tensor>,
    ple_norm: Option<Tensor>,
    encoders: Option<crate::gemma_encoders::Encoders>,
}

#[derive(Clone)]
struct Kv {
    k: Tensor,
    v: Tensor,
    start: usize,
}

pub struct GemmaBackend {
    pub(crate) dev: Device,
    /// The CUDA ordinal `dev` was opened with ([`crate::gpu_device`]).
    pub(crate) device: usize,
    cfg: GemmaConfig,
    weights: Arc<Weights>,
    precision: Precision,
    attention_dtype: DType,
    readout: Readout,
    gate: Option<(Arc<Gate>, bool)>,
    pub load_s: f64,
    processor: Option<Arc<basal_core::media::Processor>>,
    media_slot: Option<Vec<(basal_core::media::Kind, Tensor)>>,
    #[cfg(feature = "cuda")]
    lt: Option<Arc<crate::cublaslt::Lt>>,
}

struct MediaSpan {
    start: usize,
    features: Tensor,
}
struct MediaPrompt {
    #[cfg(feature = "cuda")]
    groups_device: Tensor,
    #[cfg(feature = "cuda")]
    has_image_groups: bool,
    spans: Vec<MediaSpan>,
    groups: Vec<Option<usize>>,
}

struct AttentionMask<'a> {
    groups: &'a [Option<usize>],
    #[cfg(feature = "cuda")]
    device_groups: &'a Tensor,
    #[cfg(feature = "cuda")]
    has_image_groups: bool,
}

struct Rotary {
    head_dim: usize,
    theta: f64,
    fraction: f64,
    cos: Tensor,
    sin: Tensor,
}

const QUERY_TILE: usize = 512;

/// Gemma RMSNorm has direct weights, with multiplication in FP32 before the cast (no +1 offset).
pub(crate) fn norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let f = x.to_dtype(DType::F32)?;
    // Apply the affine weight in FP32 before the model-dtype cast. The fused RMS reduction
    // changes rounding, so this numerical candidate requires upstream FP32 quality validation.
    if x.device().is_cuda() {
        if let Some(w) = weight {
            return Ok(
                candle_nn::ops::rms_norm(&f.contiguous()?, &w.to_dtype(DType::F32)?, eps as f32)?.to_dtype(dtype)?
            );
        }
    }
    let scale = (f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.powf(-0.5)?;
    let mut out = f.broadcast_mul(&scale)?;
    if let Some(w) = weight {
        out = out.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    }
    Ok(out.to_dtype(dtype)?)
}

pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let mut shape = x.dims().to_vec();
    let rows = x.elem_count() / d;
    let out = x.reshape((rows, d))?.matmul(&w.to_dtype(x.dtype())?.t()?)?;
    *shape.last_mut().context("linear input shape")? = w.dim(0)?;
    Ok(out.reshape(shape)?)
}

/// Scalar embedding scales are buffers in the model dtype, unlike ordinary affine coefficients.
fn rounded(value: f64, dtype: DType) -> f64 {
    match dtype {
        DType::BF16 => f64::from(half::bf16::from_f64(value).to_f32()),
        DType::F16 => f64::from(half::f16::from_f64(value).to_f32()),
        _ => f64::from(value as f32),
    }
}

impl GemmaBackend {
    pub fn load(
        dir: &std::path::Path,
        cfg: &GemmaConfig,
        precision: Precision,
        readout: Readout,
        device: usize,
        media: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        ensure!(cfg!(feature = "cuda"), "basal-omni currently requires the CUDA backend");
        let dev = crate::gpu_device(device)?;
        ensure!(dev.is_cuda(), "basal-omni currently requires the CUDA backend");
        Self::load_on(dir, cfg, precision, readout, dev, device, media)
    }

    fn load_on(
        dir: &std::path::Path,
        cfg: &GemmaConfig,
        precision: Precision,
        readout: Readout,
        dev: Device,
        device: usize,
        media: bool,
    ) -> Result<Self> {
        let start = Instant::now();
        let processor =
            if media { Some(Arc::new(basal_core::media::Processor::load(dir, cfg.unified)?)) } else { None };
        let path = dir.join("model.safetensors");
        // SAFETY: the checkpoint stays immutable for the mapping's lifetime; tensors are copied to owned device storage.
        let st =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.context("mapping Gemma checkpoint")?;
        let reader = crate::TensorReader::open(&path)?;
        let load_as = |name: &str, shape: &[usize], dtype: DType| -> Result<Tensor> {
            crate::check(&st, name, shape)?;
            let mut data = Vec::new();
            reader.read(name, &mut data)?;
            let tensor = Tensor::from_raw_buffer(&data, DType::BF16, shape, &Device::Cpu)?;
            Ok(tensor.to_dtype(dtype)?.to_device(&dev)?)
        };
        let load = |name: &str, shape: &[usize]| load_as(name, shape, precision.storage());
        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (i, c) in cfg.layers.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{i}");
            let get = |name: &str, shape: &[usize]| load(&format!("{prefix}.{name}"), shape);
            let qd = c.heads.checked_mul(c.head_dim).context("query dimension overflow")?;
            let kvd = c.kv_heads.checked_mul(c.head_dim).context("KV dimension overflow")?;
            layers.push(Layer {
                input_norm: get("input_layernorm.weight", &[h])?,
                post_attn_norm: get("post_attention_layernorm.weight", &[h])?,
                pre_mlp_norm: get("pre_feedforward_layernorm.weight", &[h])?,
                post_mlp_norm: get("post_feedforward_layernorm.weight", &[h])?,
                scalar: get("layer_scalar", &[1])?,
                q: get("self_attn.q_proj.weight", &[qd, h])?,
                q_norm: get("self_attn.q_norm.weight", &[c.head_dim])?,
                k: if c.shared_kv { None } else { Some(get("self_attn.k_proj.weight", &[kvd, h])?) },
                k_norm: if c.shared_kv { None } else { Some(get("self_attn.k_norm.weight", &[c.head_dim])?) },
                v: if c.shared_kv || c.key_equals_value {
                    None
                } else {
                    Some(get("self_attn.v_proj.weight", &[kvd, h])?)
                },
                o: get("self_attn.o_proj.weight", &[h, qd])?,
                gate: get("mlp.gate_proj.weight", &[c.intermediate, h])?,
                up: get("mlp.up_proj.weight", &[c.intermediate, h])?,
                down: get("mlp.down_proj.weight", &[h, c.intermediate])?,
                ple_gate: if cfg.per_layer_input == 0 {
                    None
                } else {
                    Some(get("per_layer_input_gate.weight", &[cfg.per_layer_input, h])?)
                },
                ple_projection: if cfg.per_layer_input == 0 {
                    None
                } else {
                    Some(get("per_layer_projection.weight", &[h, cfg.per_layer_input])?)
                },
                ple_norm: if cfg.per_layer_input == 0 {
                    None
                } else {
                    Some(get("post_per_layer_input_norm.weight", &[h])?)
                },
                // The GQA head-to-KV mapping is immutable. Avoid allocating and uploading
                // the same small index tensor for every prompt in every decoder layer.
                head_map: {
                    let indexes = (0..c.heads)
                        .map(|head| u32::try_from(head / (c.heads / c.kv_heads)))
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .context("GQA head index exceeds u32")?;
                    Tensor::from_vec(indexes, c.heads, &dev)?
                },
            });
        }
        let packed_ple = cfg.layers.len().checked_mul(cfg.per_layer_input).context("PLE dimension overflow")?;
        let encoders = if media {
            // FP16 is a decoder precision; media towers retain their original BF16 weights and bounds.
            Some(crate::gemma_encoders::Encoders::load(cfg, &|name, shape| load_as(name, shape, DType::BF16))?)
        } else {
            None
        };
        let weights = Weights {
            #[cfg(feature = "cuda")]
            empty_mask_group: Tensor::from_vec(vec![u32::MAX], 1, &dev)?,
            encoders,
            embed: load("model.language_model.embed_tokens.weight", &[cfg.vocab, h])?,
            norm: load("model.language_model.norm.weight", &[h])?,
            layers,
            ple_embed: if packed_ple == 0 {
                None
            } else {
                Some(load("model.language_model.embed_tokens_per_layer.weight", &[cfg.per_layer_vocab, packed_ple])?)
            },
            ple_projection: if packed_ple == 0 {
                None
            } else {
                Some(load("model.language_model.per_layer_model_projection.weight", &[packed_ple, h])?)
            },
            ple_norm: if packed_ple == 0 {
                None
            } else {
                Some(load("model.language_model.per_layer_projection_norm.weight", &[cfg.per_layer_input])?)
            },
        };
        #[cfg(feature = "cuda")]
        let lt = match (&dev, precision.compute()) {
            (Device::Cuda(d), DType::F16 | DType::BF16) => Some(Arc::new(crate::cublaslt::Lt::new(d)?)),
            _ => None,
        };
        Ok(Self {
            dev,
            device,
            cfg: cfg.clone(),
            weights: Arc::new(weights),
            precision,
            attention_dtype: DType::F32,
            readout,
            gate: None,
            load_s: start.elapsed().as_secs_f64(),
            processor,
            media_slot: None,
            #[cfg(feature = "cuda")]
            lt,
        })
    }

    pub fn configure_inference(
        &mut self,
        mode: crate::InferenceMode,
        model_name: &str,
    ) -> Result<(), crate::InferenceConfigError> {
        if self.gate.is_some() || self.media_slot.is_some() {
            return Err(crate::InferenceConfigError::CacheInitialized);
        }
        if mode == crate::InferenceMode::Balanced {
            return Err(crate::InferenceConfigError::UnsupportedBalanced);
        }
        #[cfg(feature = "cuda")]
        let dtype = match (mode, self.precision, &self.dev) {
            (crate::InferenceMode::Auto, Precision::F16, Device::Cuda(dev))
                if matches!(model_name, "basal-omni-1.0-mini" | "basal-omni-1.0-max")
                    && dev
                        .cuda_stream()
                        .context()
                        .name()
                        .is_ok_and(|name| name == "NVIDIA RTX 6000 Ada Generation") =>
            {
                DType::F16
            }
            _ => DType::F32,
        };
        #[cfg(not(feature = "cuda"))]
        let dtype = {
            let _ = model_name;
            DType::F32
        };
        self.attention_dtype = dtype;
        Ok(())
    }

    /// A table accelerates shared GEMMs only for covered shapes with an invariant reduction.
    /// All other shapes keep one fixed GEMM per prompt.
    fn linear(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let (batch, _, _) = x.dims3()?;
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            let (_, n, k) = x.dims3()?;
            let out = w.dim(0)?;
            if lt.is_invariant() && lt.covers(out, k, x.dtype()) {
                let w = w.to_dtype(x.dtype())?;
                return Ok(crate::cublaslt::matmul_nt(lt, &x.contiguous()?.reshape((batch * n, k))?, &w)?
                    .reshape((batch, n, out))?);
            }
        }
        let rows: Vec<Tensor> =
            (0..batch).map(|i| Ok(linear(&x.narrow(0, i, 1)?.squeeze(0)?, w)?.unsqueeze(0)?)).collect::<Result<_>>()?;
        Ok(Tensor::cat(&rows, 0)?)
    }

    pub fn gemm_table_missing(&self) -> Vec<(usize, usize)> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            if lt.is_invariant() {
                // The configuration was validated at load time; enumerate the actual resident projections.
                let mut shapes = std::collections::BTreeSet::new();
                for layer in &self.weights.layers {
                    for weight in [&layer.q, &layer.o, &layer.gate, &layer.up, &layer.down].into_iter().chain(
                        [&layer.k, &layer.v, &layer.ple_gate, &layer.ple_projection]
                            .into_iter()
                            .filter_map(Option::as_ref),
                    ) {
                        if let Ok((n, k)) = weight.dims2() {
                            shapes.insert((n, k));
                        }
                    }
                }
                if let Some(weight) = &self.weights.ple_projection {
                    if let Ok(shape) = weight.dims2() {
                        shapes.insert(shape);
                    }
                }
                return shapes.into_iter().filter(|(n, k)| !lt.covers(*n, *k, self.precision.compute())).collect();
            }
        }
        Vec::new()
    }

    /// Accept only an invariant table for this GPU/library/dtype; uncovered shapes use fixed GEMMs.
    pub fn load_gemm_table(&mut self, path: &std::path::Path) -> Result<usize> {
        #[cfg(feature = "cuda")]
        if let Some(lt) = &self.lt {
            let value: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            crate::check_gemm_table(&value, self.device)?;
            let entries = crate::table_entries(&value)?;
            ensure!(
                !entries.is_empty() && entries.iter().all(|e| e.dtype == self.precision.compute()),
                "Gemma GEMM table dtype must match the selected compute dtype"
            );
            lt.load_invariant(&entries)?;
            lt.validate_invariant()?;
            return Ok(entries.len());
        }
        anyhow::bail!("Gemma GEMM tables require CUDA half precision ({})", path.display())
    }

    fn checkpoint(&self) -> Result<()> {
        if let Some((gate, false)) = &self.gate {
            self.dev.synchronize()?;
            gate.checkpoint();
        }
        Ok(())
    }

    // This cache belongs to one step: every rotation uses the same offset, token count and dtype.
    fn rotate(&self, x: &Tensor, cfg: &GemmaLayerConfig, offset: usize, cache: &mut Vec<Rotary>) -> Result<Tensor> {
        let (_, n, _, d) = x.dims4()?;
        let half = d / 2;
        let index = match cache.iter().position(|r| {
            r.head_dim == d
                && r.theta.to_bits() == cfg.rope_theta.to_bits()
                && r.fraction.to_bits() == cfg.rotary_fraction.to_bits()
        }) {
            Some(index) => index,
            None => {
                let rotary = (cfg.rotary_fraction * d as f64 / 2.0) as usize;
                let theta = cfg.rope_theta as f32;
                let inv: Vec<f32> = (0..half)
                    .map(|j| if j < rotary { 1.0 / theta.powf((2 * j) as f32 / d as f32) } else { 0.0 })
                    .collect();
                let mut cos = Vec::with_capacity(n * d);
                let mut sin = Vec::with_capacity(n * d);
                for pos in offset..offset + n {
                    let angles: Vec<f32> = inv.iter().map(|f| *f * pos as f32).collect();
                    for _ in 0..2 {
                        cos.extend(angles.iter().map(|a| a.cos()));
                        sin.extend(angles.iter().map(|a| a.sin()));
                    }
                }
                let index = cache.len();
                cache.push(Rotary {
                    head_dim: d,
                    theta: cfg.rope_theta,
                    fraction: cfg.rotary_fraction,
                    cos: Tensor::from_vec(cos, (1, n, 1, d), &self.dev)?.to_dtype(x.dtype())?,
                    sin: Tensor::from_vec(sin, (1, n, 1, d), &self.dev)?.to_dtype(x.dtype())?,
                });
                index
            }
        };
        let r = &cache[index];
        let rotated = Tensor::cat(&[&x.narrow(3, half, half)?.neg()?, &x.narrow(3, 0, half)?], 3)?;
        Ok((x.broadcast_mul(&r.cos)? + rotated.broadcast_mul(&r.sin)?)?)
    }

    #[cfg(not(feature = "cuda"))]
    fn attention_mask(
        &self,
        kv: &Kv,
        c: &GemmaLayerConfig,
        offset: usize,
        groups: &[Option<usize>],
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        let k_len = kv.k.dim(1)?;
        let mut mask = vec![f32::NEG_INFINITY; len.checked_mul(k_len).context("attention mask overflow")?];
        for i in 0..len {
            let pos = offset + start + i;
            for j in 0..k_len {
                let key_pos = kv.start + j;
                let vision = c.sliding
                    && groups.get(pos).copied().flatten().is_some_and(|g| groups.get(key_pos) == Some(&Some(g)));
                if (key_pos <= pos || vision)
                    && (!c.sliding || pos < self.cfg.sliding_window || key_pos > pos - self.cfg.sliding_window)
                {
                    mask[i * k_len + j] = 0.0;
                }
            }
        }
        Ok(Tensor::from_vec(mask, (len, k_len), &self.dev)?)
    }

    fn attention_probs(
        &self,
        scores: &Tensor,
        kv: &Kv,
        c: &GemmaLayerConfig,
        offset: usize,
        mask: &AttentionMask<'_>,
        start: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            Ok(crate::fused::gemma_masked_softmax(
                scores,
                mask.device_groups,
                crate::fused::GemmaMask {
                    query_start: offset + start,
                    key_start: kv.start,
                    window: if c.sliding { self.cfg.sliding_window } else { 0 },
                    use_groups: c.sliding && mask.has_image_groups && !mask.groups.is_empty(),
                },
            )?)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let dense = self.attention_mask(kv, c, offset, mask.groups, start, scores.dim(1)?)?;
            Ok(candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&dense)?)?)
        }
    }

    // Long prompts keep unique GQA K/V and bound the score workspace, including explicit FP32.
    fn attention_long(
        &self,
        q: &Tensor,
        kv: &Kv,
        c: &GemmaLayerConfig,
        offset: usize,
        mask: &AttentionMask<'_>,
        dtype: DType,
    ) -> Result<Tensor> {
        let (_, n, _, _) = q.dims4()?;
        let k_len = kv.k.dim(1)?;
        let group = c.heads / c.kv_heads;
        let query = q.squeeze(0)?.transpose(0, 1)?.to_dtype(dtype)?.contiguous()?;
        let tile = QUERY_TILE.min((128 << 20) / (k_len * group * 4)).max(1);
        let mut heads = Vec::with_capacity(c.kv_heads);
        for head in 0..c.kv_heads {
            let key = kv.k.squeeze(0)?.narrow(1, head, 1)?.squeeze(1)?.to_dtype(dtype)?.t()?.contiguous()?;
            let key = key.unsqueeze(0)?.broadcast_as((group, c.head_dim, k_len))?;
            let value = kv.v.squeeze(0)?.narrow(1, head, 1)?.squeeze(1)?.to_dtype(dtype)?.contiguous()?;
            let value = value.unsqueeze(0)?.broadcast_as((group, k_len, c.head_dim))?;
            let query = query.narrow(0, head * group, group)?;
            let mut tiles = Vec::new();
            for start in (0..n).step_by(tile) {
                let len = tile.min(n - start);
                let scores = query.narrow(1, start, len)?.contiguous()?.matmul(&key)?.to_dtype(DType::F32)?;
                let probs = self.attention_probs(&scores, kv, c, offset, mask, start)?;
                tiles.push(probs.to_dtype(dtype)?.matmul(&value)?.to_dtype(q.dtype())?);
            }
            heads.push(Tensor::cat(&tiles, 1)?);
        }
        Ok(Tensor::cat(&heads, 0)?.transpose(0, 1)?.contiguous()?.reshape((1, n, c.heads * c.head_dim))?)
    }

    fn attention(
        &self,
        q: &Tensor,
        kv: &Kv,
        c: &GemmaLayerConfig,
        offset: usize,
        mask: &AttentionMask<'_>,
        map: &Tensor,
    ) -> Result<Tensor> {
        let attention_dtype = self.attention_dtype;
        let (batch, n, _, _) = q.dims4()?;
        if batch > 1 {
            let mut parts = Vec::with_capacity(batch);
            for i in 0..batch {
                let kv = Kv { k: kv.k.narrow(0, i, 1)?, v: kv.v.narrow(0, i, 1)?, start: kv.start };
                parts.push(self.attention(&q.narrow(0, i, 1)?, &kv, c, offset, mask, map)?);
            }
            return Ok(Tensor::cat(&parts, 0)?);
        }
        let k_len = kv.k.dim(1)?;
        if batch == 1 && k_len > 4096 {
            return self.attention_long(q, kv, c, offset, mask, attention_dtype);
        }
        let query_tile = QUERY_TILE.min((128 << 20) / (k_len * c.heads * batch * 4)).max(1);
        // FP32 attention batches heads into one matrix product. GQA heads select their shared K/V.
        let key =
            kv.k.index_select(map, 2)?
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .reshape((batch * c.heads, k_len, c.head_dim))?
                .to_dtype(attention_dtype)?;
        let key = key.transpose(1, 2)?.contiguous()?;
        let value =
            kv.v.index_select(map, 2)?
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .reshape((batch * c.heads, k_len, c.head_dim))?
                .to_dtype(attention_dtype)?;
        let query = q
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch * c.heads, n, c.head_dim))?
            .to_dtype(attention_dtype)?;
        let mut tiles = Vec::new();
        // Bound score tiles; CUDA derives the positional mask inside the softmax kernel.
        for start in (0..n).step_by(query_tile) {
            let len = (n - start).min(query_tile);
            let scores = query.narrow(1, start, len)?.contiguous()?.matmul(&key)?.to_dtype(DType::F32)?;
            let probabilities = self.attention_probs(&scores, kv, c, offset, mask, start)?;
            tiles.push(probabilities.to_dtype(attention_dtype)?.matmul(&value)?.to_dtype(q.dtype())?);
        }
        Ok(Tensor::cat(&tiles, 1)?
            .reshape((batch, c.heads, n, c.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, n, c.heads * c.head_dim))?)
    }

    fn step(
        &self,
        ids: &[u32],
        offset: usize,
        cache: &mut [Option<Kv>],
        media: Option<&MediaPrompt>,
        readouts: &[usize],
    ) -> Result<Tensor> {
        let batch = readouts.len();
        ensure!(batch > 0 && !ids.is_empty() && ids.len().is_multiple_of(batch), "invalid Gemma batch");
        ensure!(media.is_none() || batch == 1, "media batches are separate");
        let n = ids.len() / batch;
        ensure!(readouts.iter().all(|pos| *pos < n), "Gemma batch readout bounds");
        let dt = self.precision.compute();
        let indices = Tensor::from_slice(ids, ids.len(), &self.dev)?;
        let mut x = self.weights.embed.index_select(&indices, 0)?.to_dtype(dt)?.reshape((batch, n, self.cfg.hidden))?;
        x = (x * rounded((self.cfg.hidden as f64).sqrt(), dt))?;
        let mut ple_ids = ids.to_vec();
        if let Some(media) = media {
            let mut parts = Vec::new();
            let mut cursor = 0;
            for span in &media.spans {
                let end = span.start + span.features.dim(0)?;
                if end <= offset || span.start >= offset + n {
                    continue;
                }
                let start = span.start.max(offset) - offset;
                let stop = end.min(offset + n) - offset;
                if cursor < start {
                    parts.push(x.narrow(1, cursor, start - cursor)?);
                }
                parts.push(
                    span.features.narrow(0, offset + start - span.start, stop - start)?.to_dtype(dt)?.unsqueeze(0)?,
                );
                ple_ids[start..stop].fill(self.cfg.pad_token);
                cursor = stop;
            }
            if cursor < n {
                parts.push(x.narrow(1, cursor, n - cursor)?);
            }
            if !parts.is_empty() {
                x = Tensor::cat(&parts, 1)?;
            }
        }
        let ple_indices = Tensor::from_vec(ple_ids, ids.len(), &self.dev)?;

        let ple = if let (Some(embed), Some(proj), Some(w)) =
            (&self.weights.ple_embed, &self.weights.ple_projection, &self.weights.ple_norm)
        {
            let identity = (embed.index_select(&ple_indices, 0)?.to_dtype(dt)?
                * rounded((self.cfg.per_layer_input as f64).sqrt(), dt))?
            .reshape((batch, n, self.cfg.layers.len(), self.cfg.per_layer_input))?;
            let context = (self.linear(&x, proj)? / (self.cfg.hidden as f64).sqrt())?.reshape(identity.shape())?;
            Some(((norm(&context, Some(w), self.cfg.rms_eps)? + identity)? * std::f64::consts::FRAC_1_SQRT_2)?)
        } else {
            None
        };
        let mask = AttentionMask {
            groups: media.map_or(&[], |m| m.groups.as_slice()),
            #[cfg(feature = "cuda")]
            device_groups: media.map_or(&self.weights.empty_mask_group, |m| &m.groups_device),
            #[cfg(feature = "cuda")]
            has_image_groups: media.is_some_and(|m| m.has_image_groups),
        };
        let mut rotations = Vec::new();
        let mut active: Vec<Option<Kv>> = vec![None; self.cfg.layers.len()];
        for (i, (c, l)) in self.cfg.layers.iter().zip(&self.weights.layers).enumerate() {
            let input = norm(&x, Some(&l.input_norm), self.cfg.rms_eps)?;
            let query = self.linear(&input, &l.q)?.reshape((batch, n, c.heads, c.head_dim))?;
            let query = self.rotate(&norm(&query, Some(&l.q_norm), self.cfg.rms_eps)?, c, offset, &mut rotations)?;
            let kv = if c.shared_kv {
                active[c.kv_source].as_ref().context("Gemma KV producer not run")?.clone()
            } else {
                let key = self
                    .linear(&input, l.k.as_ref().context("Gemma key weight")?)?
                    .reshape((batch, n, c.kv_heads, c.head_dim))?;
                let value = match &l.v {
                    Some(w) => self.linear(&input, w)?.reshape(key.shape())?,
                    None => key.clone(),
                };
                let key = self.rotate(&norm(&key, l.k_norm.as_ref(), self.cfg.rms_eps)?, c, offset, &mut rotations)?;
                let value = norm(&value, None, self.cfg.rms_eps)?;
                match &cache[i] {
                    Some(past) => Kv {
                        k: Tensor::cat(&[&past.k, &key], 1)?,
                        v: Tensor::cat(&[&past.v, &value], 1)?,
                        start: past.start,
                    },
                    None => Kv { k: key, v: value, start: offset },
                }
            };
            let attn = self.linear(&self.attention(&query, &kv, c, offset, &mask, &l.head_map)?, &l.o)?;
            x = (x + norm(&attn, Some(&l.post_attn_norm), self.cfg.rms_eps)?)?;
            let input = norm(&x, Some(&l.pre_mlp_norm), self.cfg.rms_eps)?;
            let gate = self.linear(&input, &l.gate)?.gelu()?;
            let mlp = self.linear(&(gate * self.linear(&input, &l.up)?)?, &l.down)?;
            x = (x + norm(&mlp, Some(&l.post_mlp_norm), self.cfg.rms_eps)?)?;
            if let (Some(ple), Some(gate), Some(proj), Some(w)) = (&ple, &l.ple_gate, &l.ple_projection, &l.ple_norm) {
                let signal = (self.linear(&x, gate)?.gelu()? * ple.narrow(2, i, 1)?.squeeze(2)?)?;
                x = (x + norm(&self.linear(&signal, proj)?, Some(w), self.cfg.rms_eps)?)?;
            }
            x = x.broadcast_mul(&l.scalar.to_dtype(dt)?)?;
            if !c.shared_kv {
                let keep =
                    if c.sliding { kv.k.dim(1)?.min(self.cfg.sliding_window.saturating_sub(1)) } else { kv.k.dim(1)? };
                let skip = kv.k.dim(1)? - keep;
                cache[i] = Some(Kv {
                    k: kv.k.narrow(1, skip, keep)?.contiguous()?,
                    v: kv.v.narrow(1, skip, keep)?.contiguous()?,
                    start: kv.start + skip,
                });
            }
            active[i] = Some(kv);
            self.checkpoint()?;
        }
        let states: Vec<Tensor> = readouts
            .iter()
            .enumerate()
            .map(|(i, pos)| x.narrow(0, i, 1)?.narrow(1, *pos, 1)?.squeeze(1))
            .collect::<candle_core::Result<_>>()?;
        norm(&Tensor::cat(&states, 0)?, Some(&self.weights.norm), self.cfg.rms_eps)
    }

    fn read_prompt(&self, ids: &[u32], codes: &[u32]) -> Result<Vec<f32>> {
        self.read_prompt_with_media(ids, codes, None)
    }

    fn read_prompt_with_media(&self, ids: &[u32], codes: &[u32], media: Option<&MediaPrompt>) -> Result<Vec<f32>> {
        ensure!(
            !ids.is_empty()
                && ids.len() <= self.cfg.max_positions
                && ids.iter().chain(codes).all(|id| (*id as usize) < self.cfg.vocab),
            "Gemma token/context bounds"
        );
        let mut cache = vec![None; self.cfg.layers.len()];
        let mut hidden = None;
        let mut offset = 0;
        while offset < ids.len() {
            let mut end = (offset + 2048).min(ids.len());
            if self.cfg.unified {
                if let Some(media) = media {
                    for span in &media.spans {
                        if media.groups[span.start].is_some()
                            && span.start < end
                            && span.start + span.features.dim(0)? > end
                        {
                            end = span.start + span.features.dim(0)?;
                        }
                    }
                }
            }
            hidden = Some(self.step(&ids[offset..end], offset, &mut cache, media, &[end - offset - 1])?);
            offset = end;
        }
        self.readout_hidden(&hidden.context("Gemma empty prompt")?, codes)
    }

    fn readout_hidden(&self, h: &Tensor, codes: &[u32]) -> Result<Vec<f32>> {
        let h = h.to_dtype(DType::F32)?;
        let code_ids = Tensor::from_slice(codes, codes.len(), &self.dev)?;
        let head = self.weights.embed.index_select(&code_ids, 0)?.to_dtype(DType::F32)?;
        let logits = ((h.matmul(&head.t()?)? / self.cfg.softcap)?.tanh()? * self.cfg.softcap)?;
        let mut logits: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        ensure!(logits.iter().all(|v| v.is_finite()), "non-finite Gemma code logits");
        if self.readout == Readout::Bf16Rounded {
            for v in &mut logits {
                *v = half::bf16::from_f32(*v).to_f32();
            }
        }
        Ok(logits)
    }

    /// Reconstruct one prompt from a row whose dimensions, nodes and parents were validated by `letter_logits`.
    fn prompt_ids(row: &Packed, last: usize) -> Result<Vec<u32>> {
        ensure!(last < row.ids.len(), "invalid Gemma readout position");
        let node = row.node[last];
        let mut ids = Vec::new();
        for i in 0..=last {
            if row.is_ancestor_or_self(row.node[i], node) {
                ensure!(row.pos[i] as usize == ids.len(), "non-contiguous Gemma prompt positions");
                ids.push(row.ids[i]);
            }
        }
        Ok(ids)
    }

    fn read_batch(&self, prompts: &[(&[u32], &[u32])]) -> Result<Vec<Vec<f32>>> {
        ensure!(!prompts.is_empty() && prompts.len() <= 32, "Gemma batch size");
        let n = prompts.iter().map(|(ids, _)| ids.len()).max().context("empty Gemma batch")?.div_ceil(32) * 32;
        ensure!(
            n > 0
                && n <= 512
                && prompts.len() * n <= 8192
                && prompts.iter().all(|(ids, codes)| !ids.is_empty()
                    && ids.iter().chain(*codes).all(|id| (*id as usize) < self.cfg.vocab)),
            "Gemma bucket batch bounds"
        );
        let mut ids = Vec::with_capacity(prompts.len() * n);
        let mut readouts = Vec::with_capacity(prompts.len());
        for (tokens, _) in prompts {
            ids.extend_from_slice(tokens);
            readouts.push(tokens.len() - 1);
            ids.resize(readouts.len() * n, self.cfg.pad_token);
        }
        let mut cache = vec![None; self.cfg.layers.len()];
        let hidden = self.step(&ids, 0, &mut cache, None, &readouts)?;
        prompts.iter().enumerate().map(|(i, (_, codes))| self.readout_hidden(&hidden.narrow(0, i, 1)?, codes)).collect()
    }
}

impl Backend for GemmaBackend {
    fn media_kinds(&self) -> Vec<basal_core::media::Kind> {
        if self.processor.is_some() {
            vec![basal_core::media::Kind::Image, basal_core::media::Kind::Audio]
        } else {
            vec![]
        }
    }
    fn prepare_media(&self, entries: &Value, detail: &str) -> Result<Arc<basal_core::media::ProcessedMedia>> {
        Ok(Arc::new(self.processor.as_ref().context("media not enabled")?.process(entries, detail)?))
    }
    fn begin_media(&mut self, media: &basal_core::media::ProcessedMedia) -> Result<()> {
        self.media_slot = None;
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, urgent)| g.enter(*urgent));
        let encoders = self.weights.encoders.as_ref().context("media encoders not enabled")?;
        let mut encoded = Vec::with_capacity(media.items.len());
        for item in &media.items {
            let values = match item {
                basal_core::media::Input::Image(x) => &x.patches,
                basal_core::media::Input::Audio(x) => &x.features,
            };
            ensure!(values.iter().all(|v| v.is_finite()), "non-finite media features");
            encoded.push((
                item.kind(),
                encoders.encode(
                    item,
                    if self.precision == Precision::F16 { DType::BF16 } else { self.precision.compute() },
                    &self.dev,
                    self.attention_dtype,
                )?,
            ));
            self.checkpoint()?;
        }
        self.media_slot = Some(encoded);
        Ok(())
    }
    fn end_media(&mut self) {
        self.media_slot = None;
    }
    fn media_logits(&mut self, prompts: &[&[u32]], codes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, urgent)| g.enter(*urgent));
        ensure!(prompts.len() == codes.len(), "media prompt/code counts differ");
        let encoded = self.media_slot.as_ref().context("media session not started")?;
        let mut output = Vec::with_capacity(prompts.len());
        for (ids, codes) in prompts.iter().zip(codes) {
            let mut spans = Vec::new();
            let mut groups = vec![None; ids.len()];
            let mut cursor = 0;
            for (group, (kind, features)) in encoded.iter().enumerate() {
                let id = match kind {
                    basal_core::media::Kind::Image => self.cfg.image_token,
                    basal_core::media::Kind::Audio => self.cfg.audio_token,
                };
                let start = ids[cursor..]
                    .iter()
                    .position(|token| *token == id)
                    .map(|i| i + cursor)
                    .context("media token span missing")?;
                let len = features.dim(0)?;
                ensure!(
                    len > 0 && start + len <= ids.len() && ids[start..start + len].iter().all(|token| *token == id),
                    "media token/embedding counts differ"
                );
                if self.cfg.unified && *kind == basal_core::media::Kind::Image {
                    groups[start..start + len].fill(Some(group));
                }
                spans.push(MediaSpan { start, features: features.clone() });
                cursor = start + len;
            }
            let expected: usize =
                encoded.iter().map(|(_, x)| x.dim(0)).collect::<candle_core::Result<Vec<_>>>()?.iter().sum();
            ensure!(
                ids.iter().filter(|id| **id == self.cfg.image_token || **id == self.cfg.audio_token).count()
                    == expected,
                "extra media placeholders in the state/question"
            );
            #[cfg(feature = "cuda")]
            let has_image_groups = groups.iter().any(Option::is_some);
            #[cfg(feature = "cuda")]
            let groups_device = if has_image_groups {
                let ids = groups
                    .iter()
                    .map(|group| match group {
                        Some(id) => u32::try_from(*id).context("image group index exceeds u32"),
                        None => Ok(u32::MAX),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::from_vec(ids, groups.len(), &self.dev)?
            } else {
                self.weights.empty_mask_group.clone()
            };
            output.push(self.read_prompt_with_media(
                ids,
                codes,
                Some(&MediaPrompt {
                    spans,
                    groups,
                    #[cfg(feature = "cuda")]
                    groups_device,
                    #[cfg(feature = "cuda")]
                    has_image_groups,
                }),
            )?);
        }
        Ok(output)
    }
    fn describe(&self) -> Value {
        json!({"backend":"cuda", "device": self.device, "architecture":if self.cfg.unified {"gemma4_unified"} else {"gemma4"},
            "precision":format!("{:?}",self.precision), "attention_products":format!("{:?}",self.attention_dtype),
            "attention_accumulation":"F32", "softmax":"F32 with positional masks on CUDA", "query_tile_max":QUERY_TILE,
            "score_workspace_mib":128, "long_attention":"unique GQA K/V", "prefill_chunk":2048,
            "short_attention_batch":1, "text_bucket_tokens":32, "text_batch_max":32, "text_batch_tokens_max":8192,
            "gemm":"covered invariant table shapes share a GEMM; otherwise fixed per-prompt GEMMs",
            "readout":"FP32 code rows, final logit softcap", "state_cache":false,
            "encoder_weights":"BF16", "encoder_compute":if self.precision==Precision::F32 {"F32"} else {"BF16"}})
    }
    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, urgent)| g.enter(*urgent));
        ensure!(rows.len() == letters.len(), "Gemma row/readout count mismatch");
        let mut out: Vec<Vec<Vec<f32>>> = letters.iter().map(|row| vec![Vec::new(); row.len()]).collect();
        let mut jobs = Vec::new();
        let mut long = std::collections::BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for (row_index, (row, codes)) in rows.iter().zip(letters).enumerate() {
            ensure!(
                row.ids.len() == row.node.len() && row.ids.len() == row.pos.len() && row.last.len() == codes.len(),
                "invalid packed Gemma row"
            );
            ensure!(
                !row.parent.is_empty()
                    && row.parent[0] == -1
                    && row.parent.iter().enumerate().skip(1).all(|(i, p)| *p >= 0 && (*p as usize) < i),
                "invalid Gemma tree parents"
            );
            ensure!(row.node.iter().all(|n| (*n as usize) < row.parent.len()), "invalid Gemma block id");
            for (code_index, (&last, codes)) in row.last.iter().zip(*codes).enumerate() {
                let ids = Self::prompt_ids(row, last)?;
                if ids.len() > 512 {
                    // Keep coordinates into the borrowed packed rows, rather than retaining
                    // one full copy of a potentially large state for every question.
                    long.entry(ids.len()).or_default().push((row_index, code_index));
                } else {
                    jobs.push((row_index, code_index, ids, codes.as_slice()));
                }
            }
        }
        let mut lengths = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for (index, (_, _, ids, _)) in jobs.iter().enumerate() {
            lengths.entry(ids.len().div_ceil(32) * 32).or_default().push(index);
        }
        for (len, indexes) in lengths {
            for indexes in indexes.chunks(32.min(8192 / len.max(1))) {
                let prompts: Vec<(&[u32], &[u32])> = indexes
                    .iter()
                    .map(|index| {
                        let (_, _, ids, codes) = &jobs[*index];
                        (ids.as_slice(), *codes)
                    })
                    .collect();
                for (&index, logits) in indexes.iter().zip(self.read_batch(&prompts)?) {
                    let (row, code, _, _) = &jobs[index];
                    out[*row][*code] = logits;
                }
            }
        }
        // As before, short buckets run first, then long prompts by length and input order.
        // Only the currently executed long prompt is materialized.
        for indexes in long.values() {
            for &(row, code) in indexes {
                let ids = Self::prompt_ids(rows[row], rows[row].last[code])?;
                out[row][code] = self.read_prompt(&ids, &letters[row][code])?;
            }
        }
        Ok(out)
    }
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            dev: self.dev.clone(),
            device: self.device,
            cfg: self.cfg.clone(),
            weights: self.weights.clone(),
            precision: self.precision,
            attention_dtype: self.attention_dtype,
            readout: self.readout,
            gate: None,
            load_s: 0.0,
            processor: self.processor.clone(),
            media_slot: None,
            #[cfg(feature = "cuda")]
            lt: self.lt.clone(),
        })
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        self.gate = Some((gate, urgent));
    }
}
