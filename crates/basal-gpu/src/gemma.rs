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
}

struct Weights {
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
    cfg: GemmaConfig,
    weights: Arc<Weights>,
    precision: Precision,
    readout: Readout,
    gate: Option<(Arc<Gate>, bool)>,
    pub load_s: f64,
    processor: Option<Arc<basal_core::media::Processor>>,
    media_slot: Option<Vec<(basal_core::media::Kind, Tensor)>>,
}

struct MediaSpan {
    start: usize,
    features: Tensor,
}
struct MediaPrompt {
    spans: Vec<MediaSpan>,
    groups: Vec<Option<usize>>,
}

/// Gemma RMSNorm has direct weights, with multiplication in FP32 before the cast (no +1 offset).
pub(crate) fn norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let f = x.to_dtype(DType::F32)?;
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
        media: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        ensure!(cfg!(feature = "cuda"), "basal-omni currently requires the CUDA backend");
        let dev = crate::gpu_device()?;
        ensure!(dev.is_cuda(), "basal-omni currently requires the CUDA backend");
        Self::load_on(dir, cfg, precision, readout, dev, media)
    }

    fn load_on(
        dir: &std::path::Path,
        cfg: &GemmaConfig,
        precision: Precision,
        readout: Readout,
        dev: Device,
        media: bool,
    ) -> Result<Self> {
        let start = Instant::now();
        let processor =
            if media { Some(Arc::new(basal_core::media::Processor::load(dir, cfg.unified)?)) } else { None };
        ensure!(!media || precision != Precision::F16, "omni media needs bf16 or f32 precision");
        let path = dir.join("model.safetensors");
        // SAFETY: the checkpoint stays immutable for the mapping's lifetime; tensors are copied to owned device storage.
        let st =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.context("mapping Gemma checkpoint")?;
        let reader = crate::TensorReader::open(&path)?;
        let load = |name: &str, shape: &[usize]| -> Result<Tensor> {
            crate::check(&st, name, shape)?;
            let mut data = Vec::new();
            reader.read(name, &mut data)?;
            let tensor = Tensor::from_raw_buffer(&data, DType::BF16, shape, &Device::Cpu)?;
            Ok(tensor.to_dtype(precision.storage())?.to_device(&dev)?)
        };
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
            });
        }
        let packed_ple = cfg.layers.len().checked_mul(cfg.per_layer_input).context("PLE dimension overflow")?;
        let encoders = if media { Some(crate::gemma_encoders::Encoders::load(cfg, &load)?) } else { None };
        let weights = Weights {
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
        Ok(Self {
            dev,
            cfg: cfg.clone(),
            weights: Arc::new(weights),
            precision,
            readout,
            gate: None,
            load_s: start.elapsed().as_secs_f64(),
            processor,
            media_slot: None,
        })
    }

    fn checkpoint(&self) -> Result<()> {
        if let Some((gate, false)) = &self.gate {
            self.dev.synchronize()?;
            gate.checkpoint();
        }
        Ok(())
    }

    fn rotate(&self, x: &Tensor, cfg: &GemmaLayerConfig, offset: usize) -> Result<Tensor> {
        let (n, _, d) = x.dims3()?;
        let half = d / 2;
        let rotary = (cfg.rotary_fraction * d as f64 / 2.0) as usize;
        let theta = cfg.rope_theta as f32;
        let inv: Vec<f32> =
            (0..half).map(|j| if j < rotary { 1.0 / theta.powf((2 * j) as f32 / d as f32) } else { 0.0 }).collect();
        let mut cos = Vec::with_capacity(n * d);
        let mut sin = Vec::with_capacity(n * d);
        for pos in offset..offset + n {
            let angles: Vec<f32> = inv.iter().map(|f| *f * pos as f32).collect();
            for _ in 0..2 {
                cos.extend(angles.iter().map(|a| a.cos()));
                sin.extend(angles.iter().map(|a| a.sin()));
            }
        }
        let cos = Tensor::from_vec(cos, (n, 1, d), &self.dev)?.to_dtype(x.dtype())?;
        let sin = Tensor::from_vec(sin, (n, 1, d), &self.dev)?.to_dtype(x.dtype())?;
        let rotated = Tensor::cat(&[&x.narrow(2, half, half)?.neg()?, &x.narrow(2, 0, half)?], 2)?;
        Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
    }

    fn attention(
        &self,
        q: &Tensor,
        kv: &Kv,
        c: &GemmaLayerConfig,
        offset: usize,
        groups: &[Option<usize>],
    ) -> Result<Tensor> {
        let n = q.dim(0)?;
        let k_len = kv.k.dim(0)?;
        let head_groups = c.heads / c.kv_heads;
        let mut heads = Vec::with_capacity(c.heads);
        for head in 0..c.heads {
            let key = kv.k.narrow(1, head / head_groups, 1)?.squeeze(1)?.to_dtype(DType::F32)?.contiguous()?;
            let value = kv.v.narrow(1, head / head_groups, 1)?.squeeze(1)?.to_dtype(DType::F32)?.contiguous()?;
            let query = q.narrow(1, head, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
            let mut tiles = Vec::new();
            for start in (0..n).step_by(64) {
                let len = (n - start).min(64);
                let mut mask = vec![f32::NEG_INFINITY; len.checked_mul(k_len).context("attention mask overflow")?];
                for i in 0..len {
                    let pos = offset + start + i;
                    for j in 0..k_len {
                        let key_pos = kv.start + j;
                        let vision = c.sliding
                            && groups
                                .get(pos)
                                .copied()
                                .flatten()
                                .is_some_and(|g| groups.get(key_pos) == Some(&Some(g)));
                        if (key_pos <= pos || vision)
                            && (!c.sliding || pos < self.cfg.sliding_window || key_pos > pos - self.cfg.sliding_window)
                        {
                            mask[i * k_len + j] = 0.0;
                        }
                    }
                }
                let scores = query.narrow(0, start, len)?.contiguous()?.matmul(&key.t()?.contiguous()?)?;
                let mask = Tensor::from_vec(mask, (len, k_len), &self.dev)?;
                let probabilities = candle_nn::ops::softmax_last_dim(&(scores + mask)?)?;
                tiles.push(probabilities.matmul(&value)?.to_dtype(q.dtype())?);
            }
            heads.push(Tensor::cat(&tiles, 0)?.unsqueeze(1)?);
        }
        Ok(Tensor::cat(&heads, 1)?.reshape((n, c.heads * c.head_dim))?)
    }

    fn step(
        &self,
        ids: &[u32],
        offset: usize,
        cache: &mut [Option<Kv>],
        media: Option<&MediaPrompt>,
    ) -> Result<Tensor> {
        let dt = self.precision.compute();
        let indices = Tensor::from_slice(ids, ids.len(), &self.dev)?;
        let mut x = self.weights.embed.index_select(&indices, 0)?.to_dtype(dt)?;
        x = (x * rounded((self.cfg.hidden as f64).sqrt(), dt))?;
        let mut ple_ids = ids.to_vec();
        if let Some(media) = media {
            let mut parts = Vec::new();
            let mut cursor = 0;
            for span in &media.spans {
                let end = span.start + span.features.dim(0)?;
                if end <= offset || span.start >= offset + ids.len() {
                    continue;
                }
                let start = span.start.max(offset) - offset;
                let stop = end.min(offset + ids.len()) - offset;
                if cursor < start {
                    parts.push(x.narrow(0, cursor, start - cursor)?);
                }
                parts.push(span.features.narrow(0, offset + start - span.start, stop - start)?.to_dtype(dt)?);
                ple_ids[start..stop].fill(self.cfg.pad_token);
                cursor = stop;
            }
            if cursor < ids.len() {
                parts.push(x.narrow(0, cursor, ids.len() - cursor)?);
            }
            if !parts.is_empty() {
                x = Tensor::cat(&parts, 0)?;
            }
        }
        let ple_indices = Tensor::from_vec(ple_ids, ids.len(), &self.dev)?;

        let ple = if let (Some(embed), Some(proj), Some(w)) =
            (&self.weights.ple_embed, &self.weights.ple_projection, &self.weights.ple_norm)
        {
            let identity = (embed.index_select(&ple_indices, 0)?.to_dtype(dt)?
                * rounded((self.cfg.per_layer_input as f64).sqrt(), dt))?
            .reshape((ids.len(), self.cfg.layers.len(), self.cfg.per_layer_input))?;
            let context = (linear(&x, proj)? / (self.cfg.hidden as f64).sqrt())?.reshape(identity.shape())?;
            Some(((norm(&context, Some(w), self.cfg.rms_eps)? + identity)? * std::f64::consts::FRAC_1_SQRT_2)?)
        } else {
            None
        };
        let mut active: Vec<Option<Kv>> = vec![None; self.cfg.layers.len()];
        for (i, (c, l)) in self.cfg.layers.iter().zip(&self.weights.layers).enumerate() {
            let input = norm(&x, Some(&l.input_norm), self.cfg.rms_eps)?;
            let query = linear(&input, &l.q)?.reshape((ids.len(), c.heads, c.head_dim))?;
            let query = self.rotate(&norm(&query, Some(&l.q_norm), self.cfg.rms_eps)?, c, offset)?;
            let kv = if c.shared_kv {
                active[c.kv_source].as_ref().context("Gemma KV producer not run")?.clone()
            } else {
                let key = linear(&input, l.k.as_ref().context("Gemma key weight")?)?.reshape((
                    ids.len(),
                    c.kv_heads,
                    c.head_dim,
                ))?;
                let value = match &l.v {
                    Some(w) => linear(&input, w)?.reshape(key.shape())?,
                    None => key.clone(),
                };
                let key = self.rotate(&norm(&key, l.k_norm.as_ref(), self.cfg.rms_eps)?, c, offset)?;
                let value = norm(&value, None, self.cfg.rms_eps)?;
                match &cache[i] {
                    Some(past) => Kv {
                        k: Tensor::cat(&[&past.k, &key], 0)?,
                        v: Tensor::cat(&[&past.v, &value], 0)?,
                        start: past.start,
                    },
                    None => Kv { k: key, v: value, start: offset },
                }
            };
            let attn =
                linear(&self.attention(&query, &kv, c, offset, media.map_or(&[], |m| m.groups.as_slice()))?, &l.o)?;
            x = (x + norm(&attn, Some(&l.post_attn_norm), self.cfg.rms_eps)?)?;
            let input = norm(&x, Some(&l.pre_mlp_norm), self.cfg.rms_eps)?;
            let gate = linear(&input, &l.gate)?.gelu()?;
            let mlp = linear(&(gate * linear(&input, &l.up)?)?, &l.down)?;
            x = (x + norm(&mlp, Some(&l.post_mlp_norm), self.cfg.rms_eps)?)?;
            if let (Some(ple), Some(gate), Some(proj), Some(w)) = (&ple, &l.ple_gate, &l.ple_projection, &l.ple_norm) {
                let signal = (linear(&x, gate)?.gelu()? * ple.narrow(1, i, 1)?.squeeze(1)?)?;
                x = (x + norm(&linear(&signal, proj)?, Some(w), self.cfg.rms_eps)?)?;
            }
            x = x.broadcast_mul(&l.scalar.to_dtype(dt)?)?;
            if !c.shared_kv {
                let keep =
                    if c.sliding { kv.k.dim(0)?.min(self.cfg.sliding_window.saturating_sub(1)) } else { kv.k.dim(0)? };
                let skip = kv.k.dim(0)? - keep;
                cache[i] = Some(Kv {
                    k: kv.k.narrow(0, skip, keep)?.contiguous()?,
                    v: kv.v.narrow(0, skip, keep)?.contiguous()?,
                    start: kv.start + skip,
                });
            }
            active[i] = Some(kv);
            self.checkpoint()?;
        }
        norm(&x.narrow(0, ids.len() - 1, 1)?, Some(&self.weights.norm), self.cfg.rms_eps)
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
            let mut end = (offset + 512).min(ids.len());
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
            hidden = Some(self.step(&ids[offset..end], offset, &mut cache, media)?);
            offset = end;
        }
        let h = hidden.context("Gemma empty prompt")?.to_dtype(DType::F32)?;
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
            encoded.push((item.kind(), encoders.encode(item, self.precision.compute(), &self.dev)?));
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
            output.push(self.read_prompt_with_media(ids, codes, Some(&MediaPrompt { spans, groups }))?);
        }
        Ok(output)
    }
    fn describe(&self) -> Value {
        json!({"backend":"cuda", "architecture":if self.cfg.unified {"gemma4_unified"} else {"gemma4"}, "precision":format!("{:?}",self.precision), "attention":"FP32 math, separate prompts, query tiles of 64", "prefill_chunk":512, "readout":"FP32 code rows, final logit softcap", "state_cache":false})
    }
    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
        let gate = self.gate.clone();
        let _guard = gate.as_ref().map(|(g, urgent)| g.enter(*urgent));
        ensure!(rows.len() == letters.len(), "Gemma row/readout count mismatch");
        let mut out = Vec::with_capacity(rows.len());
        for (row, codes) in rows.iter().zip(letters) {
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
            let mut readouts = Vec::with_capacity(codes.len());
            for (&last, codes) in row.last.iter().zip(*codes) {
                ensure!(last < row.ids.len(), "invalid Gemma readout position");
                let node = row.node[last];
                let mut ids = Vec::new();
                for i in 0..=last {
                    if row.is_ancestor_or_self(row.node[i], node) {
                        ensure!(row.pos[i] as usize == ids.len(), "non-contiguous Gemma prompt positions");
                        ids.push(row.ids[i]);
                    }
                }
                readouts.push(self.read_prompt(&ids, codes)?);
            }
            out.push(readouts);
        }
        Ok(out)
    }
    fn fork(&self) -> Result<Self> {
        Ok(Self {
            dev: self.dev.clone(),
            cfg: self.cfg.clone(),
            weights: self.weights.clone(),
            precision: self.precision,
            readout: self.readout,
            gate: None,
            load_s: 0.0,
            processor: self.processor.clone(),
            media_slot: None,
        })
    }
    fn set_gate(&mut self, gate: Arc<Gate>, urgent: bool) {
        self.gate = Some((gate, urgent));
    }
}
