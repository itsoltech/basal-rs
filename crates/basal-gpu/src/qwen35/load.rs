//! Checkpoint loading: shape-checked uploads, merged CUDA projections and fixed position/head tables.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use basal_core::architecture::{Qwen35Config, Qwen35Layer};
use basal_core::DeviceHealth;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};

use super::{optimized_kernels, Attention, DeltaNet, Layer, Mixer, Qwen35Backend, Weights, CHUNK};
use crate::{OnStage, Precision, Readout};

/// Uploads checkpoint tensors from the one namespace the checkpoint uses and records every name it reads.
struct Loader<'a> {
    st: &'a MmapedSafetensors,
    prefix: &'static str,
    dev: &'a Device,
    precision: Precision,
    expected: HashSet<String>,
}

impl Loader<'_> {
    /// `fp32` keeps the tensor in FP32 whatever the load precision.
    fn tensor(&mut self, name: &str, shape: &[usize], fp32: bool) -> Result<Tensor> {
        let name = format!("{}{}", self.prefix, name.strip_prefix("model.").context("Qwen3.5 tensor namespace")?);
        let view = self.st.get(&name).with_context(|| format!("Qwen3.5 tensor {name}"))?;
        ensure!(view.shape() == shape, "Qwen3.5 tensor {name}: expected {shape:?}, got {:?}", view.shape());
        ensure!(
            matches!(view.dtype(), safetensors::Dtype::BF16 | safetensors::Dtype::F32),
            "Qwen3.5 tensor {name}: unsupported dtype"
        );
        self.expected.insert(name.clone());
        // Widen BF16 projections once without changing their values or GEMM shapes. Embeddings stay
        // compact: only selected rows are widened at use sites. This opt-in trades VRAM for fewer casts.
        let dtype = if fp32
            || (self.precision == Precision::F32
                && (view.dtype() == safetensors::Dtype::F32
                    || (self.dev.is_cuda()
                        && !name.ends_with("embed_tokens.weight")
                        && crate::plan::DiagnosticOverrides::process().qwen_f32_resident())))
        {
            DType::F32
        } else {
            self.precision.storage()
        };
        Ok(self.st.load(&name, &Device::Cpu)?.to_dtype(dtype)?.to_device(self.dev)?)
    }

    /// Decoder layer `i`, uploaded in the checkpoint's reference order.
    fn layer(&mut self, c: &Qwen35Config, i: usize, kind: &Qwen35Layer) -> Result<Layer> {
        let p = format!("model.layers.{i}");
        let mut w = |suffix: &str, shape: &[usize], fp32: bool| self.tensor(&format!("{p}.{suffix}"), shape, fp32);
        let mixer = match kind {
            Qwen35Layer::Attention => Mixer::Attention(Attention {
                q: w("self_attn.q_proj.weight", &[2 * c.heads * c.head_dim, c.hidden], false)?,
                k: w("self_attn.k_proj.weight", &[c.kv_heads * c.head_dim, c.hidden], false)?,
                v: w("self_attn.v_proj.weight", &[c.kv_heads * c.head_dim, c.hidden], false)?,
                o: w("self_attn.o_proj.weight", &[c.hidden, c.heads * c.head_dim], false)?,
                q_norm: w("self_attn.q_norm.weight", &[c.head_dim], true)?,
                k_norm: w("self_attn.k_norm.weight", &[c.head_dim], true)?,
            }),
            Qwen35Layer::DeltaNet => {
                let kd = c.linear_key_heads * c.linear_key_dim;
                let vd = c.linear_value_heads * c.linear_value_dim;
                let a_log = w("linear_attn.A_log", &[c.linear_value_heads], true)?;
                let dt_bias = w("linear_attn.dt_bias", &[c.linear_value_heads], false)?;
                let params = Tensor::stack(&[&a_log, &dt_bias.to_dtype(DType::F32)?], 0)?;
                Mixer::DeltaNet(DeltaNet {
                    input: None,
                    qkv: w("linear_attn.in_proj_qkv.weight", &[2 * kd + vd, c.hidden], false)?,
                    z: w("linear_attn.in_proj_z.weight", &[vd, c.hidden], false)?,
                    a: w("linear_attn.in_proj_a.weight", &[c.linear_value_heads, c.hidden], false)?,
                    b: w("linear_attn.in_proj_b.weight", &[c.linear_value_heads, c.hidden], false)?,
                    conv: w("linear_attn.conv1d.weight", &[2 * kd + vd, 1, c.conv_kernel], false)?
                        .to_dtype(DType::F32)?,
                    a_log,
                    dt_bias,
                    params,
                    norm: w("linear_attn.norm.weight", &[c.linear_value_dim], true)?,
                    out: w("linear_attn.out_proj.weight", &[c.hidden, vd], false)?,
                })
            }
        };
        Ok(Layer {
            gate_up: None,
            mixer,
            input_norm: w("input_layernorm.weight", &[c.hidden], true)?,
            post_norm: w("post_attention_layernorm.weight", &[c.hidden], true)?,
            gate: w("mlp.gate_proj.weight", &[c.intermediate, c.hidden], false)?,
            up: w("mlp.up_proj.weight", &[c.intermediate, c.hidden], false)?,
            down: w("mlp.down_proj.weight", &[c.hidden, c.intermediate], false)?,
        })
    }
}

/// Optimized kernel sets: one gate/up GEMM and one DeltaNet input GEMM per layer. The separate projections become
/// row views of the merged weights, so both paths read identical values.
fn merge_projections(layer: &mut Layer, c: &Qwen35Config) -> Result<()> {
    let gate_up = Tensor::cat(&[&layer.gate, &layer.up], 0)?;
    layer.gate = gate_up.narrow(0, 0, c.intermediate)?;
    layer.up = gate_up.narrow(0, c.intermediate, c.intermediate)?;
    layer.gate_up = Some(gate_up);
    if let Mixer::DeltaNet(w) = &mut layer.mixer {
        let qkv_rows = w.qkv.dim(0)?;
        let z_rows = w.z.dim(0)?;
        let h = c.linear_value_heads;
        let input = Tensor::cat(&[&w.qkv, &w.z, &w.a, &w.b], 0)?;
        w.qkv = input.narrow(0, 0, qkv_rows)?;
        w.z = input.narrow(0, qkv_rows, z_rows)?;
        w.a = input.narrow(0, qkv_rows + z_rows, h)?;
        w.b = input.narrow(0, qkv_rows + z_rows + h, h)?;
        w.input = Some(input);
    }
    Ok(())
}

/// Prefill chunk: the reference 64 rows, or the fixed larger chunks of the optimized kernel sets.
fn prefill_chunk(kernel_set: crate::KernelSet, optimized: bool) -> usize {
    if !optimized {
        CHUNK
    } else if matches!(kernel_set, crate::KernelSet::Qwen35V3 | crate::KernelSet::Qwen35V4) {
        2048
    } else {
        512
    }
}

impl Qwen35Backend {
    pub fn load(
        dir: &Path,
        cfg: &Qwen35Config,
        precision: Precision,
        readout: Readout,
        device: usize,
        kernel_set: crate::KernelSet,
    ) -> Result<Self> {
        cfg.validate()?;
        ensure!(cfg!(feature = "cuda") || cfg!(target_os = "macos"), "Qwen3.5 requires a CUDA or Metal build");
        ensure!(!cfg!(target_os = "macos") || precision != Precision::Bf16, "Qwen3.5 on Metal needs dtype f32 or f16");
        let started = Instant::now();
        let dev = crate::gpu_device(device)?;
        let optimized = optimized_kernels(kernel_set) && dev.is_cuda() && precision != Precision::F32;
        let paths = basal_core::checkpoint::files(dir)?;
        // SAFETY: checkpoint shards remain immutable while mapped. Loading is synchronous; every returned tensor
        // owns its uploaded storage, and no mapping or borrowed checkpoint data escapes this loader.
        let st = unsafe { MmapedSafetensors::multi(&paths) }.context("mapping Qwen3.5 weights")?;
        // Published Pico retains the conditional model's language_model namespace after stripping vision.
        // Transformers also accepts the plain text-model namespace; select exactly one for all weights.
        let prefix =
            if st.get("model.language_model.embed_tokens.weight").is_ok() { "model.language_model." } else { "model." };
        let mut loader = Loader { st: &st, prefix, dev: &dev, precision, expected: HashSet::new() };
        let c = cfg;
        let embed = loader.tensor("model.embed_tokens.weight", &[c.vocab, c.hidden], false)?;
        let final_norm = loader.tensor("model.norm.weight", &[c.hidden], true)?;
        let mut layers = Vec::with_capacity(c.layers.len());
        for (i, kind) in c.layers.iter().enumerate() {
            let mut layer = loader.layer(c, i, kind)?;
            if optimized {
                merge_projections(&mut layer, c)?;
            }
            layers.push(layer);
        }
        ensure!(st.tensors().len() == loader.expected.len(), "unexpected or duplicate Qwen3.5 checkpoint tensors");
        dev.synchronize()?;
        let health = DeviceHealth::default();
        let rope_inv = Tensor::from_vec(
            (0..c.rotary_dim / 2)
                .map(|i| 1.0 / (c.rope_theta as f32).powf((2 * i) as f32 / c.rotary_dim as f32))
                .collect::<Vec<_>>(),
            (1, c.rotary_dim / 2),
            &dev,
        )?;
        let head_map = Tensor::from_vec(
            (0..c.heads).map(|h| (h / (c.heads / c.kv_heads)) as u32).collect::<Vec<_>>(),
            c.heads,
            &dev,
        )?;
        dev.synchronize()?;
        let weights = Arc::new(OnStage::new(
            &dev,
            true,
            &health,
            Weights {
                #[cfg(feature = "cuda")]
                lt: match &dev {
                    Device::Cuda(d) if optimized => Some(Arc::new(crate::cublaslt::Lt::new(d)?)),
                    _ => None,
                },
                embed,
                norm: final_norm,
                layers,
                rope_inv,
                head_map,
            },
        ));
        Ok(Self {
            dev,
            device,
            cfg: cfg.clone(),
            weights,
            precision,
            readout,
            health,
            gate: None,
            chunk: prefill_chunk(kernel_set, optimized),
            kernel_set,
            tensor_attention: false,
            arbitrary_prefix: false,
            load_s: started.elapsed().as_secs_f64(),
        })
    }
}
