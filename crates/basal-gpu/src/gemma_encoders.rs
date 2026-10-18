//! Native media projections and encoders ported from Transformers 5.17.0 (Apache-2.0).

use anyhow::{ensure, Context, Result};
use basal_core::architecture::GemmaConfig;
use basal_core::media::{AudioInput, ImageInput, Input};
use candle_core::{DType, Device, Tensor, D};

use crate::gemma::{linear, norm};

type Load<'a> = dyn Fn(&str, &[usize]) -> Result<Tensor> + 'a;

struct Clip {
    weight: Tensor,
    bounds: Option<[f64; 4]>,
}
impl Clip {
    fn load(load: &Load<'_>, prefix: &str, input: usize, output: usize) -> Result<Self> {
        let scalar = |name| -> Result<f64> {
            let value = f64::from(load(&format!("{prefix}.{name}"), &[])?.to_dtype(DType::F32)?.to_scalar::<f32>()?);
            ensure!(!value.is_nan(), "NaN clipping bound");
            Ok(value)
        };
        let bounds = [scalar("input_min")?, scalar("input_max")?, scalar("output_min")?, scalar("output_max")?];
        ensure!(bounds[0] <= bounds[1] && bounds[2] <= bounds[3], "invalid clipping bounds");
        Ok(Self { weight: load(&format!("{prefix}.linear.weight"), &[output, input])?, bounds: Some(bounds) })
    }
    fn vision(load: &Load<'_>, prefix: &str, input: usize, output: usize, clipped: bool) -> Result<Self> {
        if clipped {
            Self::load(load, prefix, input, output)
        } else {
            Ok(Self { weight: load(&format!("{prefix}.linear.weight"), &[output, input])?, bounds: None })
        }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self.bounds {
            Some([input_min, input_max, output_min, output_max]) => {
                Ok(linear(&x.clamp(input_min, input_max)?, &self.weight)?.clamp(output_min, output_max)?)
            }
            None => linear(x, &self.weight),
        }
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
    hidden: usize,
    heads: usize,
    head_dim: usize,
    standardization: Option<(Tensor, Tensor)>,
}

impl Vision {
    fn load(load: &Load<'_>, cfg: &GemmaConfig) -> Result<Self> {
        let v = cfg.vision.as_ref().context("vision config missing")?;
        // Both published tower architectures use full axial RoPE and ordinary multi-head attention.
        // Validate the supported layouts before allocating any encoder weights.
        let mini = cfg.hidden == 2560
            && v["hidden_size"] == 768
            && v["num_hidden_layers"] == 16
            && v["num_attention_heads"] == 12
            && v["head_dim"] == 64
            && v["intermediate_size"] == 3072
            && v["standardize"] == false
            && v["use_clipped_linears"] == true;
        let ultra = cfg.hidden == 5376
            && v["hidden_size"] == 1152
            && v["num_hidden_layers"] == 27
            && v["num_attention_heads"] == 16
            && v["head_dim"] == 72
            && v["intermediate_size"] == 4304
            && v["standardize"] == true
            && v["use_clipped_linears"] == false;
        ensure!(
            (mini || ultra)
                && v["num_key_value_heads"] == v["num_attention_heads"]
                && v["patch_size"] == 16
                && v["pooling_kernel_size"] == 3
                && v["position_embedding_size"] == 10240
                && v["rms_norm_eps"] == 1e-6
                && v["hidden_activation"] == "gelu_pytorch_tanh"
                && v["rope_parameters"]["rope_type"] == "axial"
                && v["rope_parameters"]["rope_theta"] == 100.0,
            "unsupported vision tower architecture"
        );
        let (h, count, heads, d, mlp) = if mini { (768, 16, 12, 64, 3072) } else { (1152, 27, 16, 72, 4304) };
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let p = format!("model.vision_tower.encoder.layers.{i}");
            let get = |name, shape: &[usize]| load(&format!("{p}.{name}"), shape);
            let linear = |name: &str, input, output| Clip::vision(load, &format!("{p}.{name}"), input, output, mini);
            layers.push(VisionLayer {
                input: get("input_layernorm.weight", &[h])?,
                post_attn: get("post_attention_layernorm.weight", &[h])?,
                pre_mlp: get("pre_feedforward_layernorm.weight", &[h])?,
                post_mlp: get("post_feedforward_layernorm.weight", &[h])?,
                q: linear("self_attn.q_proj", h, h)?,
                k: linear("self_attn.k_proj", h, h)?,
                v: linear("self_attn.v_proj", h, h)?,
                o: linear("self_attn.o_proj", h, h)?,
                q_norm: get("self_attn.q_norm.weight", &[d])?,
                k_norm: get("self_attn.k_norm.weight", &[d])?,
                gate: linear("mlp.gate_proj", h, mlp)?,
                up: linear("mlp.up_proj", h, mlp)?,
                down: linear("mlp.down_proj", mlp, h)?,
            });
        }
        Ok(Self {
            patch: load("model.vision_tower.patch_embedder.input_proj.weight", &[h, 768])?,
            positions: load("model.vision_tower.patch_embedder.position_embedding_table", &[2, 10240, h])?,
            layers,
            projection: load("model.embed_vision.embedding_projection.weight", &[cfg.hidden, h])?,
            hidden: h,
            heads,
            head_dim: d,
            standardization: if ultra {
                Some((load("model.vision_tower.std_bias", &[h])?, load("model.vision_tower.std_scale", &[h])?))
            } else {
                None
            },
        })
    }
    fn rotation(positions: &[(i64, i64)], dim: usize, dtype: DType, dev: &Device) -> Result<(Tensor, Tensor)> {
        let n = positions.len();
        let mut cos = Vec::with_capacity(n * dim);
        let mut sin = Vec::with_capacity(n * dim);
        for &(a, b) in positions {
            for position in [a, b] {
                let angles: Vec<f32> =
                    (0..dim / 4).map(|i| position as f32 / 100.0f32.powf((2 * i) as f32 / (dim / 2) as f32)).collect();
                for _ in 0..2 {
                    cos.extend(angles.iter().map(|x| x.cos()));
                    sin.extend(angles.iter().map(|x| x.sin()));
                }
            }
        }
        Ok((
            Tensor::from_vec(cos, (n, 1, dim), dev)?.to_dtype(dtype)?,
            Tensor::from_vec(sin, (n, 1, dim), dev)?.to_dtype(dtype)?,
        ))
    }

    fn rotate(x: &Tensor, rotation: &(Tensor, Tensor)) -> Result<Tensor> {
        let (cos, sin) = rotation;
        let mut parts = Vec::new();
        let quarter = x.dim(2)? / 4;
        for offset in [0, quarter * 2] {
            parts.push(Tensor::cat(
                &[&x.narrow(2, offset + quarter, quarter)?.neg()?, &x.narrow(2, offset, quarter)?],
                2,
            )?);
        }
        Ok((x.broadcast_mul(cos)? + Tensor::cat(&parts, 2)?.broadcast_mul(sin)?)?)
    }
    fn forward(&self, image: &ImageInput, dt: DType, dev: &Device, attention_dtype: DType) -> Result<Tensor> {
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
        // Position/dtype are fixed for this image. Reuse exactly the same rotary tensors
        // for Q and K in all layers.
        let rotation = Self::rotation(&image.positions[..n], self.head_dim, dt, dev)?;
        for layer in &self.layers {
            let input = norm(&x, Some(&layer.input), 1e-6)?;
            let q = Self::rotate(
                &norm(&layer.q.forward(&input)?.reshape((n, self.heads, self.head_dim))?, Some(&layer.q_norm), 1e-6)?,
                &rotation,
            )?;
            let k = Self::rotate(
                &norm(&layer.k.forward(&input)?.reshape((n, self.heads, self.head_dim))?, Some(&layer.k_norm), 1e-6)?,
                &rotation,
            )?;
            let v = norm(&layer.v.forward(&input)?.reshape((n, self.heads, self.head_dim))?, None, 1e-6)?;
            // Batch heads while keeping softmax and accumulation in FP32; QK/PV outputs use FP16 for half models. Query tiling bounds the
            // score workspace for high-detail images; changed cuBLAS geometry needs FP32 validation.
            let q = q.transpose(0, 1)?.to_dtype(attention_dtype)?.contiguous()?;
            let k = k.transpose(0, 1)?.to_dtype(attention_dtype)?.contiguous()?;
            let v = v.transpose(0, 1)?.to_dtype(attention_dtype)?.contiguous()?;
            let kt = k.transpose(1, 2)?;
            let mut tiles = Vec::new();
            for i in (0..n).step_by(128) {
                let len = (n - i).min(128);
                let scores = q.narrow(1, i, len)?.matmul(&kt)?.to_dtype(DType::F32)?;
                let values =
                    candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(attention_dtype)?.matmul(&v)?.to_dtype(dt)?;
                tiles.push(values.transpose(0, 1)?.contiguous()?);
            }
            let attn = layer.o.forward(&Tensor::cat(&tiles, 0)?.reshape((n, self.hidden))?)?;
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
        // Preserve dy-major/dx-minor 3x3 patch order and the FP32 reduction axis.
        // Batched pooling avoids nine tiny copies and one reduction per output token.
        let pooled = x
            .reshape((image.patch_height / 3, 3, image.patch_width / 3, 3, self.hidden))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((out, 9, self.hidden))?
            .to_dtype(DType::F32)?;
        let pooled = (pooled.sum_keepdim(1)?.squeeze(1)? / 9.0)?.to_dtype(dt)?;
        let scaled = (pooled.to_dtype(DType::F32)? * (self.hidden as f64).sqrt())?;
        // Ultra standardizes before the final cast: scaling alone can overflow FP16.
        let pooled = if let Some((bias, scale)) = &self.standardization {
            scaled.broadcast_sub(&bias.to_dtype(DType::F32)?)?.broadcast_mul(&scale.to_dtype(DType::F32)?)?
        } else {
            scaled
        }
        .to_dtype(dt)?;
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
        #[cfg(feature = "cuda")]
        let valid_tensor = Tensor::from_vec(valid.iter().map(|v| u32::from(*v)).collect::<Vec<_>>(), n, dev)?;
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
            #[cfg(feature = "cuda")]
            let attended = crate::fused::gemma_audio_attention(&q, &k, &v, &relative, &valid_tensor)?
                .reshape((n, 1024))?
                .to_dtype(dt)?;
            #[cfg(not(feature = "cuda"))]
            let attended = {
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
                Tensor::cat(&rows, 0)?
            };
            let attn = layer.post.forward(&attended)?;
            x = (x + norm(&attn.clamp(-1e10, 1e10)?, Some(&layer.post_attn), 1e-6)?)?;
            let input = layer.conv_start.forward(&norm(&x, Some(&layer.conv_pre), 1e-6)?)?;
            let gated = (input.narrow(1, 0, 1024)? * candle_nn::ops::sigmoid(&input.narrow(1, 1024, 1024)?)?)?;
            // Candle splits groups=1024 into separate convolution calls. Compute the five taps
            // over all channels together, accumulating in FP32 before the model-dtype cast.
            // This changes reduction rounding and requires upstream FP32 quality validation.
            let padded = gated.to_dtype(DType::F32)?.pad_with_zeros(0, 4, 0)?;
            let weight = layer.conv.squeeze(1)?.to_dtype(DType::F32)?;
            let mut conv = Tensor::zeros((n, 1024), DType::F32, dev)?;
            for tap in 0..5 {
                let w = weight.narrow(1, tap, 1)?.squeeze(1)?.unsqueeze(0)?;
                conv = (conv + padded.narrow(0, tap, n)?.broadcast_mul(&w)?)?;
            }
            let conv = conv.to_dtype(dt)?;
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
    Towers { vision: Vision, audio: Option<Audio> },
}
impl Encoders {
    pub fn load(cfg: &GemmaConfig, load: &Load<'_>) -> Result<Self> {
        let vision = cfg.vision.as_ref().context("Gemma vision config missing")?;
        if cfg.unified {
            let audio = cfg.audio.as_ref().context("Gemma audio config missing")?;
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
            let audio = if let Some(audio) = &cfg.audio {
                ensure!(
                    cfg.hidden == 2560
                        && audio["hidden_size"] == 1024
                        && audio["num_hidden_layers"] == 12
                        && audio["num_attention_heads"] == 8
                        && audio["conv_kernel_size"] == 5
                        && audio["attention_context_left"] == 13
                        && audio["attention_context_right"] == 0
                        && audio["attention_chunk_size"] == 12
                        && audio["attention_logit_cap"] == 50.0
                        && audio["use_clipped_linears"] == true,
                    "unsupported audio tower architecture"
                );
                Some(Audio::load(load)?)
            } else {
                None
            };
            Ok(Self::Towers { vision: Vision::load(load, cfg)?, audio })
        }
    }
    pub fn encode(&self, input: &Input, dt: DType, dev: &Device, attention_dtype: DType) -> Result<Tensor> {
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
            (Self::Towers { vision, .. }, Input::Image(image)) => vision.forward(image, dt, dev, attention_dtype),
            (Self::Towers { audio, .. }, Input::Audio(input)) => {
                audio.as_ref().context("this model does not support audio")?.forward(input, dt, dev)
            }
        }
    }
}
