//! Text-only Qwen3.5 contract, independent of Gemma Omni and of Llama KV trees.

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use super::{integer, number};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35Layer {
    DeltaNet,
    Attention,
}

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub layers: Vec<Qwen35Layer>,
}

impl Qwen35Config {
    pub(super) fn parse(v: &Value) -> Result<Self> {
        ensure!(
            v.get("architectures") == Some(&serde_json::json!(["Qwen3_5ForCausalLM"])),
            "only text-only Qwen3_5ForCausalLM checkpoints are supported"
        );
        ensure!(
            v.get("hidden_act").and_then(Value::as_str) == Some("silu")
                && v.get("attention_bias").and_then(Value::as_bool) == Some(false)
                && v.get("attn_output_gate").and_then(Value::as_bool) == Some(true)
                && v.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true),
            "Qwen3.5 requires SiLU, bias-free gated attention and tied embeddings"
        );
        ensure!(
            v.get("mlp_only_layers").is_none_or(|x| x.as_array().is_some_and(Vec::is_empty))
                && v.get("rope_scaling").is_none_or(Value::is_null),
            "unsupported Qwen3.5 MLP-only layers or RoPE scaling"
        );
        ensure!(
            v.get("mamba_ssm_dtype").is_none_or(|x| x.as_str() == Some("float32")),
            "Qwen3.5 recurrent state must use FP32"
        );
        let rope = v.get("rope_parameters").context("config.json: Qwen3.5 rope_parameters")?;
        ensure!(rope.get("rope_type").and_then(Value::as_str) == Some("default"), "unsupported Qwen3.5 RoPE");
        let dim = integer(v, "head_dim")?;
        let fraction = number(rope, "partial_rotary_factor")?;
        let rotary = dim as f64 * fraction;
        ensure!(fraction <= 1.0 && rotary.fract() == 0.0, "invalid Qwen3.5 rotary fraction");
        let n = integer(v, "num_hidden_layers")?;
        ensure!(n <= 128, "unsupported Qwen3.5 layer count");
        let layers = v
            .get("layer_types")
            .and_then(Value::as_array)
            .filter(|x| x.len() == n)
            .context("config.json: Qwen3.5 layer_types")?
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(Qwen35Layer::DeltaNet),
                Some("full_attention") => Ok(Qwen35Layer::Attention),
                _ => anyhow::bail!("unsupported Qwen3.5 layer type {x}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let c = Self {
            hidden: integer(v, "hidden_size")?,
            intermediate: integer(v, "intermediate_size")?,
            vocab: integer(v, "vocab_size")?,
            max_positions: integer(v, "max_position_embeddings")?,
            heads: integer(v, "num_attention_heads")?,
            kv_heads: integer(v, "num_key_value_heads")?,
            head_dim: dim,
            linear_key_heads: integer(v, "linear_num_key_heads")?,
            linear_value_heads: integer(v, "linear_num_value_heads")?,
            linear_key_dim: integer(v, "linear_key_head_dim")?,
            linear_value_dim: integer(v, "linear_value_head_dim")?,
            conv_kernel: integer(v, "linear_conv_kernel_dim")?,
            rms_eps: number(v, "rms_norm_eps")?,
            rope_theta: number(rope, "rope_theta")?,
            rotary_dim: rotary as usize,
            layers,
        };
        c.validate()?;
        Ok(c)
    }

    /// Bound all products before checkpoint shapes and recurrent buffers are allocated.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=128).contains(&self.layers.len())
                && (1..=32768).contains(&self.hidden)
                && (1..=262144).contains(&self.intermediate)
                && (1..=4_000_000).contains(&self.vocab)
                && (1..=1_048_576).contains(&self.max_positions),
            "unsupported Qwen3.5 dimensions"
        );
        ensure!(
            (1..=256).contains(&self.heads)
                && self.kv_heads > 0
                && self.heads.is_multiple_of(self.kv_heads)
                && (2..=512).contains(&self.head_dim)
                && self.rotary_dim >= 2
                && self.rotary_dim <= self.head_dim
                && self.rotary_dim.is_multiple_of(2),
            "invalid Qwen3.5 attention dimensions"
        );
        ensure!(
            (1..=128).contains(&self.linear_key_heads)
                && (1..=128).contains(&self.linear_value_heads)
                && self.linear_value_heads.is_multiple_of(self.linear_key_heads)
                && (1..=256).contains(&self.linear_key_dim)
                && (1..=256).contains(&self.linear_value_dim)
                && (2..=16).contains(&self.conv_kernel),
            "unsupported Qwen3.5 DeltaNet dimensions"
        );
        ensure!(
            self.rms_eps.is_finite() && self.rms_eps > 0.0 && self.rope_theta.is_finite() && self.rope_theta > 0.0,
            "invalid Qwen3.5 normalization or RoPE"
        );
        Ok(())
    }
}
