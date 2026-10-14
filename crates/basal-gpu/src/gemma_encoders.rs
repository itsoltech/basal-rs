//! Native media projections and encoders ported from Transformers 5.17.0 (Apache-2.0).

use anyhow::{ensure, Context, Result};
use basal_core::architecture::GemmaConfig;
use basal_core::media::{AudioInput, ImageInput, Input};
use candle_core::{DType, Device, Tensor, D};

use crate::gemma::{linear, norm};

type Load<'a> = dyn Fn(&str, &[usize]) -> Result<Tensor> + 'a;

struct Clip {
    weight: Tensor,
    input_min: f64,
    input_max: f64,
    output_min: f64,
    output_max: f64,
}
impl Clip {
    fn load(load: &Load<'_>, prefix: &str, input: usize, output: usize) -> Result<Self> {
        let scalar = |name| -> Result<f64> {
            let value = f64::from(load(&format!("{prefix}.{name}"), &[])?.to_dtype(DType::F32)?.to_scalar::<f32>()?);
            ensure!(!value.is_nan(), "NaN clipping bound");
            Ok(value)
        };
        let c = Self {
            weight: load(&format!("{prefix}.linear.weight"), &[output, input])?,
            input_min: scalar("input_min")?,
            input_max: scalar("input_max")?,
            output_min: scalar("output_min")?,
            output_max: scalar("output_max")?,
        };
        ensure!(c.input_min <= c.input_max && c.output_min <= c.output_max, "invalid clipping bounds");
        Ok(c)
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(linear(&x.clamp(self.input_min, self.input_max)?, &self.weight)?.clamp(self.output_min, self.output_max)?)
    }
}

fn layer_norm(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let inv = (centered.sqr()?.mean_keepdim(D::Minus1)? + eps)?.powf(-0.5)?;
    let mut value = centered.broadcast_mul(&inv)?.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    if let Some(b) = bias {
        value = value.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    Ok(value.to_dtype(dtype)?)
}

pub(crate) struct UnifiedVision {
    ln1: Tensor,
    ln1_bias: Tensor,
    dense: Tensor,
    dense_bias: Tensor,
    ln2: Tensor,
    ln2_bias: Tensor,
    positions: Tensor,
    pos_norm: Tensor,
    pos_bias: Tensor,
    projection: Tensor,
}
impl UnifiedVision {
    fn load(load: &Load<'_>, cfg: &GemmaConfig) -> Result<Self> {
        let h = cfg.hidden;
        let p = "model.vision_embedder";
        Ok(Self {
            ln1: load(&format!("{p}.patch_ln1.weight"), &[6912])?,
            ln1_bias: load(&format!("{p}.patch_ln1.bias"), &[6912])?,
            dense: load(&format!("{p}.patch_dense.weight"), &[h, 6912])?,
            dense_bias: load(&format!("{p}.patch_dense.bias"), &[h])?,
            ln2: load(&format!("{p}.patch_ln2.weight"), &[h])?,
            ln2_bias: load(&format!("{p}.patch_ln2.bias"), &[h])?,
            positions: load(&format!("{p}.pos_embedding"), &[1120, 2, h])?,
            pos_norm: load(&format!("{p}.pos_norm.weight"), &[h])?,
            pos_bias: load(&format!("{p}.pos_norm.bias"), &[h])?,
            projection: load("model.embed_vision.embedding_projection.weight", &[h, h])?,
        })
    }
    fn forward(&self, image: &ImageInput, dt: DType, dev: &Device) -> Result<Tensor> {
        let n = image.soft_tokens;
        ensure!(
            image.patch_dim == 6912
                && n > 0
                && n <= image.positions.len()
                && n <= 1120
                && image.patches.len() >= n * 6912,
            "unified image feature shape"
        );
        let patches = Tensor::from_slice(&image.patches[..n * 6912], (n, 6912), dev)?.to_dtype(dt)?;
        let x = layer_norm(&patches, &self.ln1, Some(&self.ln1_bias), 1e-5)?;
        let x = linear(&x, &self.dense)?.broadcast_add(&self.dense_bias.to_dtype(dt)?)?;
        let mut x = layer_norm(&x, &self.ln2, Some(&self.ln2_bias), 1e-5)?;
        let pos_x: Vec<u32> = image.positions[..n].iter().map(|(x, _)| *x as u32).collect();
        let pos_y: Vec<u32> = image.positions[..n].iter().map(|(_, y)| *y as u32).collect();
        ensure!(pos_x.iter().chain(&pos_y).all(|v| *v < 1120), "unified image position bounds");
        let pos = (self
            .positions
            .narrow(1, 0, 1)?
            .squeeze(1)?
            .contiguous()?
            .index_select(&Tensor::from_vec(pos_x, n, dev)?, 0)?
            + self
                .positions
                .narrow(1, 1, 1)?
                .squeeze(1)?
                .contiguous()?
                .index_select(&Tensor::from_vec(pos_y, n, dev)?, 0)?)?;
        x = (x + pos.to_dtype(dt)?)?;
        x = layer_norm(&x, &self.pos_norm, Some(&self.pos_bias), 1e-5)?;
        linear(&norm(&x, None, 1e-6)?, &self.projection)
    }
}

struct VisionLayer {
    input: Tensor,
    post_attn: Tensor,
    pre_mlp: Tensor,
    post_mlp: Tensor,
    q: Clip,
    k: Clip,
    v: Clip,
    o: Clip,
    q_norm: Tensor,
    k_norm: Tensor,
    gate: Clip,
    up: Clip,
    down: Clip,
}
pub(crate) struct Vision {
    patch: Tensor,
    positions: Tensor,
    layers: Vec<VisionLayer>,
    projection: Tensor,
}

impl Vision {
    fn load(load: &Load<'_>) -> Result<Self> {
        let mut layers = Vec::with_capacity(16);
        for i in 0..16 {
            let p = format!("model.vision_tower.encoder.layers.{i}");
            let get = |name, shape: &[usize]| load(&format!("{p}.{name}"), shape);
            layers.push(VisionLayer {
                input: get("input_layernorm.weight", &[768])?,
                post_attn: get("post_attention_layernorm.weight", &[768])?,
                pre_mlp: get("pre_feedforward_layernorm.weight", &[768])?,
                post_mlp: get("post_feedforward_layernorm.weight", &[768])?,
                q: Clip::load(load, &format!("{p}.self_attn.q_proj"), 768, 768)?,
                k: Clip::load(load, &format!("{p}.self_attn.k_proj"), 768, 768)?,
                v: Clip::load(load, &format!("{p}.self_attn.v_proj"), 768, 768)?,
                o: Clip::load(load, &format!("{p}.self_attn.o_proj"), 768, 768)?,
                q_norm: get("self_attn.q_norm.weight", &[64])?,
                k_norm: get("self_attn.k_norm.weight", &[64])?,
                gate: Clip::load(load, &format!("{p}.mlp.gate_proj"), 768, 3072)?,
                up: Clip::load(load, &format!("{p}.mlp.up_proj"), 768, 3072)?,
                down: Clip::load(load, &format!("{p}.mlp.down_proj"), 3072, 768)?,
            });
        }
        Ok(Self {
            patch: load("model.vision_tower.patch_embedder.input_proj.weight", &[768, 768])?,
            positions: load("model.vision_tower.patch_embedder.position_embedding_table", &[2, 10240, 768])?,
            layers,
            projection: load("model.embed_vision.embedding_projection.weight", &[2560, 768])?,
        })
    }
    fn rotate(x: &Tensor, positions: &[(i64, i64)], dev: &Device) -> Result<Tensor> {
        let n = x.dim(0)?;
        let mut cos = Vec::with_capacity(n * 64);
        let mut sin = Vec::with_capacity(n * 64);
        for &(a, b) in positions {
            for position in [a, b] {
                let angles: Vec<f32> =
                    (0..16).map(|i| position as f32 / 100.0f32.powf((2 * i) as f32 / 32.0)).collect();
                for _ in 0..2 {
                    cos.extend(angles.iter().map(|x| x.cos()));
                    sin.extend(angles.iter().map(|x| x.sin()));
                }
            }
        }
        let cos = Tensor::from_vec(cos, (n, 1, 64), dev)?.to_dtype(x.dtype())?;
        let sin = Tensor::from_vec(sin, (n, 1, 64), dev)?.to_dtype(x.dtype())?;
        let mut parts = Vec::new();
        for offset in [0, 32] {
            parts.push(Tensor::cat(&[&x.narrow(2, offset + 16, 16)?.neg()?, &x.narrow(2, offset, 16)?], 2)?);
        }
        Ok((x.broadcast_mul(&cos)? + Tensor::cat(&parts, 2)?.broadcast_mul(&sin)?)?)
    }
    fn forward(&self, image: &ImageInput, dt: DType, dev: &Device) -> Result<Tensor> {
        let n = image.patch_width.checked_mul(image.patch_height).context("image dimensions overflow")?;
        ensure!(
            image.patch_dim == 768
                && n > 0
                && n <= 10080
                && n <= image.positions.len()
                && image.patches.len() >= n * 768,
            "vision patch shape"
        );
        let pixels = Tensor::from_slice(&image.patches[..n * 768], (n, 768), dev)?.to_dtype(dt)?;
        let patches = ((pixels - 0.5)? * 2.0)?;
        let px: Vec<u32> = image.positions[..n].iter().map(|(x, _)| *x as u32).collect();
        let py: Vec<u32> = image.positions[..n].iter().map(|(_, y)| *y as u32).collect();
        ensure!(px.iter().chain(&py).all(|v| *v < 10240), "vision position bounds");
        let positions = (self.positions.get(0)?.index_select(&Tensor::from_vec(px, n, dev)?, 0)?
            + self.positions.get(1)?.index_select(&Tensor::from_vec(py, n, dev)?, 0)?)?
        .to_dtype(dt)?;
        let mut x = (linear(&patches, &self.patch)? + positions)?;
        for layer in &self.layers {
            let input = norm(&x, Some(&layer.input), 1e-6)?;
            let q = Self::rotate(
                &norm(&layer.q.forward(&input)?.reshape((n, 12, 64))?, Some(&layer.q_norm), 1e-6)?,
                &image.positions[..n],
                dev,
            )?;
            let k = Self::rotate(
                &norm(&layer.k.forward(&input)?.reshape((n, 12, 64))?, Some(&layer.k_norm), 1e-6)?,
                &image.positions[..n],
                dev,
            )?;
            let v = norm(&layer.v.forward(&input)?.reshape((n, 12, 64))?, None, 1e-6)?;
            let mut heads = Vec::with_capacity(12);
            for h in 0..12 {
                let q = q.narrow(1, h, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
                let k = k.narrow(1, h, 1)?.squeeze(1)?.to_dtype(DType::F32)?.contiguous()?;
                let v = v.narrow(1, h, 1)?.squeeze(1)?.to_dtype(DType::F32)?.contiguous()?;
                let mut tiles = Vec::new();
                for i in (0..n).step_by(64) {
                    let len = (n - i).min(64);
                    let scores = q.narrow(0, i, len)?.contiguous()?.matmul(&k.t()?.contiguous()?)?;
                    tiles.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?.to_dtype(dt)?);
                }
                heads.push(Tensor::cat(&tiles, 0)?.unsqueeze(1)?);
            }
            let attn = layer.o.forward(&Tensor::cat(&heads, 1)?.reshape((n, 768))?)?;
            x = (x + norm(&attn, Some(&layer.post_attn), 1e-6)?)?;
            let input = norm(&x, Some(&layer.pre_mlp), 1e-6)?;
            let mlp = layer.down.forward(&(layer.gate.forward(&input)?.gelu()? * layer.up.forward(&input)?)?)?;
            x = (x + norm(&mlp, Some(&layer.post_mlp), 1e-6)?)?;
        }
        let out = image.soft_tokens;
        ensure!(
            out <= 1120 && image.patch_width.is_multiple_of(3) && image.patch_height.is_multiple_of(3) && n == out * 9,
            "vision pooling dimensions"
        );
        // Pool in FP32 as upstream, cast to the model dtype before sqrt(hidden) scaling in FP32.
        let mut pooled = Vec::with_capacity(out);
        for y in 0..image.patch_height / 3 {
            for col in 0..image.patch_width / 3 {
                let mut rows = Vec::with_capacity(9);
                for dy in 0..3 {
                    for dx in 0..3 {
                        rows.push(
                            x.narrow(0, (y * 3 + dy) * image.patch_width + col * 3 + dx, 1)?.to_dtype(DType::F32)?,
                        );
                    }
                }
                pooled.push((Tensor::cat(&rows, 0)?.sum_keepdim(0)? / 9.0)?.to_dtype(dt)?);
            }
        }
        let pooled = (Tensor::cat(&pooled, 0)?.to_dtype(DType::F32)? * (768.0f64).sqrt())?.to_dtype(dt)?;
        linear(&norm(&pooled, None, 1e-6)?, &self.projection)
    }
}

struct FeedForward {
    pre: Tensor,
    post: Tensor,
    first: Clip,
    second: Clip,
}
impl FeedForward {
    fn load(load: &Load<'_>, p: &str) -> Result<Self> {
        Ok(Self {
            pre: load(&format!("{p}.pre_layer_norm.weight"), &[1024])?,
            post: load(&format!("{p}.post_layer_norm.weight"), &[1024])?,
            first: Clip::load(load, &format!("{p}.ffw_layer_1"), 1024, 4096)?,
            second: Clip::load(load, &format!("{p}.ffw_layer_2"), 4096, 1024)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = norm(&x.clamp(-1e10, 1e10)?, Some(&self.pre), 1e-6)?;
        let hidden = self.second.forward(&candle_nn::ops::silu(&self.first.forward(&hidden)?)?)?;
        Ok((x + (norm(&hidden.clamp(-1e10, 1e10)?, Some(&self.post), 1e-6)? * 0.5)?)?)
    }
}
struct AudioLayer {
    ff1: FeedForward,
    ff2: FeedForward,
    pre_attn: Tensor,
    post_attn: Tensor,
    out_norm: Tensor,
    q: Clip,
    k: Clip,
    v: Clip,
    post: Clip,
    relative: Tensor,
    scale: Tensor,
    conv_pre: Tensor,
    conv_norm: Tensor,
    conv_start: Clip,
    conv_end: Clip,
    conv: Tensor,
}
pub(crate) struct Audio {
    conv0: Tensor,
    norm0: Tensor,
    conv1: Tensor,
    norm1: Tensor,
    input: Tensor,
    layers: Vec<AudioLayer>,
    output: Tensor,
    output_bias: Tensor,
    projection: Tensor,
}

impl Audio {
    fn load(load: &Load<'_>) -> Result<Self> {
        let mut layers = Vec::with_capacity(12);
        for i in 0..12 {
            let p = format!("model.audio_tower.layers.{i}");
            let get = |name, shape: &[usize]| load(&format!("{p}.{name}"), shape);
            layers.push(AudioLayer {
                ff1: FeedForward::load(load, &format!("{p}.feed_forward1"))?,
                ff2: FeedForward::load(load, &format!("{p}.feed_forward2"))?,
                pre_attn: get("norm_pre_attn.weight", &[1024])?,
                post_attn: get("norm_post_attn.weight", &[1024])?,
                out_norm: get("norm_out.weight", &[1024])?,
                q: Clip::load(load, &format!("{p}.self_attn.q_proj"), 1024, 1024)?,
                k: Clip::load(load, &format!("{p}.self_attn.k_proj"), 1024, 1024)?,
                v: Clip::load(load, &format!("{p}.self_attn.v_proj"), 1024, 1024)?,
                post: Clip::load(load, &format!("{p}.self_attn.post"), 1024, 1024)?,
                relative: get("self_attn.relative_k_proj.weight", &[1024, 1024])?,
                scale: get("self_attn.per_dim_scale", &[128])?,
                conv_pre: get("lconv1d.pre_layer_norm.weight", &[1024])?,
                conv_norm: get("lconv1d.conv_norm.weight", &[1024])?,
                conv_start: Clip::load(load, &format!("{p}.lconv1d.linear_start"), 1024, 2048)?,
                conv_end: Clip::load(load, &format!("{p}.lconv1d.linear_end"), 1024, 1024)?,
                conv: get("lconv1d.depthwise_conv1d.weight", &[1024, 1, 5])?,
            });
        }
        let p = "model.audio_tower.subsample_conv_projection";
        Ok(Self {
            conv0: load(&format!("{p}.layer0.conv.weight"), &[128, 1, 3, 3])?,
            norm0: load(&format!("{p}.layer0.norm.weight"), &[128])?,
            conv1: load(&format!("{p}.layer1.conv.weight"), &[32, 128, 3, 3])?,
            norm1: load(&format!("{p}.layer1.norm.weight"), &[32])?,
            input: load(&format!("{p}.input_proj_linear.weight"), &[1024, 1024])?,
            layers,
            output: load("model.audio_tower.output_proj.weight", &[1536, 1024])?,
            output_bias: load("model.audio_tower.output_proj.bias", &[1536])?,
            projection: load("model.embed_audio.embedding_projection.weight", &[2560, 1536])?,
        })
    }
    fn forward(&self, audio: &AudioInput, dt: DType, dev: &Device) -> Result<Tensor> {
        let frames = audio.valid.len();
        ensure!(
            audio.feature_dim == 128 && frames > 0 && frames <= 3001 && audio.features.len() == frames * 128,
            "mel audio shape"
        );
        let mut valid = audio.valid.clone();
        let mut x = Tensor::from_slice(&audio.features, (1, 1, frames, 128), dev)?.to_dtype(dt)?;
        for (conv, w) in [(&self.conv0, &self.norm0), (&self.conv1, &self.norm1)] {
            let mask: Vec<f32> = valid.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect();
            x = x.broadcast_mul(&Tensor::from_vec(mask, (1, 1, valid.len(), 1), dev)?.to_dtype(dt)?)?;
            x = x.conv2d(&conv.to_dtype(dt)?, 1, 2, 1, 1)?;
            x = layer_norm(&x.permute((0, 2, 3, 1))?, w, None, 1e-6)?.relu()?.permute((0, 3, 1, 2))?.contiguous()?;
            valid = valid.into_iter().step_by(2).take(x.dim(2)?).collect();
        }
        let n = x.dim(2)?;
        x = linear(&x.permute((0, 2, 3, 1))?.reshape((n, 1024))?, &self.input)?;
        let times = (Tensor::arange(0u32, 512, dev)?.to_dtype(dt)? * -((10000.0f64).ln() / 511.0))?.exp()?;
        let positions: Vec<u32> = (0..=12).rev().collect();
        let positions = Tensor::from_vec(positions, (13, 1), dev)?.to_dtype(dt)?;
        let angles = positions.broadcast_mul(&times.reshape((1, 512))?)?;
        let pos = Tensor::cat(&[&angles.sin()?, &angles.cos()?], 1)?;
        for layer in &self.layers {
            x = layer.ff1.forward(&x)?;
            let input = norm(&x.clamp(-1e10, 1e10)?, Some(&layer.pre_attn), 1e-6)?;
            let scale = layer.scale.to_dtype(DType::F32)?;
            let smooth = (scale.clamp(f64::NEG_INFINITY, 20.0)?.exp()? + 1.0)?.log()?;
            let softplus = scale.gt(20.0)?.where_cond(&scale, &smooth)?.to_dtype(dt)?.to_dtype(DType::F32)?;
            let q = (layer.q.forward(&input)?.to_dtype(DType::F32)?.reshape((n, 8, 128))?
                * ((128.0f64).powf(-0.5) / std::f64::consts::LN_2))?
                .broadcast_mul(&softplus.reshape((1, 1, 128))?)?;
            let k = (layer.k.forward(&input)?.to_dtype(DType::F32)?
                * ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2))?
                .reshape((n, 8, 128))?;
            let v = layer.v.forward(&input)?.to_dtype(DType::F32)?.reshape((n, 8, 128))?;
            let relative = linear(&pos, &layer.relative)?.to_dtype(DType::F32)?.reshape((13, 8, 128))?;
            let mut rows = Vec::with_capacity(n);
            // Upstream mask uses dist < (context_left - 1): 11 past frames and the current frame.
            // Relative positions still use the 13-vector blocked shift (index 12 - lag).
            for i in 0..n {
                let first = i.saturating_sub(11);
                let len = i - first + 1;
                let mut heads = Vec::with_capacity(8);
                for head in 0..8 {
                    let query = q.narrow(0, i, 1)?.narrow(1, head, 1)?.reshape((1, 128))?;
                    let keys = k.narrow(0, first, len)?.narrow(1, head, 1)?.reshape((len, 128))?;
                    let values = v.narrow(0, first, len)?.narrow(1, head, 1)?.reshape((len, 128))?;
                    let indexes: Vec<u32> = (first..=i).map(|j| (12 - (i - j)) as u32).collect();
                    let rel = relative
                        .narrow(1, head, 1)?
                        .squeeze(1)?
                        .contiguous()?
                        .index_select(&Tensor::from_vec(indexes, len, dev)?, 0)?;
                    let scores = query.matmul(&keys.t()?)? + query.matmul(&rel.t()?)?;
                    let mut scores = ((scores? / 50.0)?.tanh()? * 50.0)?;
                    let mask: Vec<f32> = valid[first..=i].iter().map(|b| if *b { 0.0 } else { -1e9 }).collect();
                    scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, len), dev)?)?;
                    heads.push(candle_nn::ops::softmax_last_dim(&scores)?.matmul(&values)?);
                }
                rows.push(Tensor::cat(&heads, 1)?.to_dtype(dt)?);
            }
            let attn = layer.post.forward(&Tensor::cat(&rows, 0)?)?;
            x = (x + norm(&attn.clamp(-1e10, 1e10)?, Some(&layer.post_attn), 1e-6)?)?;
            let input = layer.conv_start.forward(&norm(&x, Some(&layer.conv_pre), 1e-6)?)?;
            let gated = (input.narrow(1, 0, 1024)? * candle_nn::ops::sigmoid(&input.narrow(1, 1024, 1024)?)?)?;
            let padded = gated.t()?.unsqueeze(0)?.pad_with_zeros(2, 4, 0)?;
            let conv = padded.conv1d(&layer.conv.to_dtype(dt)?, 0, 1, 1, 1024)?.squeeze(0)?.t()?;
            let conv = candle_nn::ops::silu(&norm(&conv.clamp(-1e10, 1e10)?, Some(&layer.conv_norm), 1e-6)?)?;
            x = (x + layer.conv_end.forward(&conv)?)?;
            x = layer.ff2.forward(&x)?;
            x = norm(&x.clamp(-1e10, 1e10)?, Some(&layer.out_norm), 1e-6)?;
        }
        let output = linear(&x, &self.output)?.broadcast_add(&self.output_bias.to_dtype(dt)?)?;
        let indices: Vec<u32> = valid.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| i as u32).collect();
        ensure!(indices.len() == audio.soft_tokens, "audio soft token count differs");
        let output = output.index_select(&Tensor::from_vec(indices, audio.soft_tokens, dev)?, 0)?;
        linear(&norm(&output, None, 1e-6)?, &self.projection)
    }
}

pub(crate) enum Encoders {
    Unified { vision: UnifiedVision, audio: Tensor },
    Towers { vision: Vision, audio: Audio },
}
impl Encoders {
    pub fn load(cfg: &GemmaConfig, load: &Load<'_>) -> Result<Self> {
        let vision = cfg.vision.as_ref().context("Gemma vision config missing")?;
        let audio = cfg.audio.as_ref().context("Gemma audio config missing")?;
        if cfg.unified {
            ensure!(
                cfg.hidden == 3840
                    && vision["patch_size"] == 16
                    && vision["pooling_kernel_size"] == 3
                    && vision["mm_embed_dim"] == 3840
                    && vision["mm_posemb_size"] == 1120
                    && audio["audio_embed_dim"] == 640,
                "unsupported unified media architecture"
            );
            Ok(Self::Unified {
                vision: UnifiedVision::load(load, cfg)?,
                audio: load("model.embed_audio.embedding_projection.weight", &[3840, 640])?,
            })
        } else {
            ensure!(
                cfg.hidden == 2560
                    && vision["hidden_size"] == 768
                    && vision["num_hidden_layers"] == 16
                    && vision["num_attention_heads"] == 12
                    && vision["num_key_value_heads"] == 12
                    && vision["head_dim"] == 64
                    && vision["intermediate_size"] == 3072
                    && vision["standardize"] == false
                    && vision["hidden_activation"] == "gelu_pytorch_tanh"
                    && vision["use_clipped_linears"] == true
                    && audio["hidden_size"] == 1024
                    && audio["num_hidden_layers"] == 12
                    && audio["num_attention_heads"] == 8
                    && audio["conv_kernel_size"] == 5
                    && audio["attention_context_left"] == 13
                    && audio["attention_context_right"] == 0
                    && audio["attention_chunk_size"] == 12
                    && audio["attention_logit_cap"] == 50.0
                    && audio["use_clipped_linears"] == true,
                "unsupported tower media architecture"
            );
            Ok(Self::Towers { vision: Vision::load(load)?, audio: Audio::load(load)? })
        }
    }
    pub fn encode(&self, input: &Input, dt: DType, dev: &Device) -> Result<Tensor> {
        match (self, input) {
            (Self::Unified { vision, .. }, Input::Image(image)) => vision.forward(image, dt, dev),
            (Self::Unified { audio, .. }, Input::Audio(input)) => {
                ensure!(
                    input.feature_dim == 640
                        && input.soft_tokens > 0
                        && input.soft_tokens <= 750
                        && input.features.len() == input.soft_tokens * 640,
                    "unified audio frame shape"
                );
                let x = Tensor::from_slice(&input.features, (input.soft_tokens, 640), dev)?.to_dtype(dt)?;
                linear(&norm(&x, None, 1e-6)?, audio)
            }
            (Self::Towers { vision, .. }, Input::Image(image)) => vision.forward(image, dt, dev),
            (Self::Towers { audio, .. }, Input::Audio(input)) => audio.forward(input, dt, dev),
        }
    }
}
