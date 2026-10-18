//! Validated architecture contracts. Gemma 4 is distinct from Llama and from Gemma 4 Unified.

use anyhow::{ensure, Context, Result};
use serde_json::Value;

use crate::manifest::{parse_config, LlamaConfig};

#[derive(Clone, Debug)]
pub enum Architecture {
    Llama(LlamaConfig),
    Gemma4(GemmaConfig),
}

#[derive(Clone, Debug)]
pub struct GemmaLayerConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub sliding: bool,
    pub shared_kv: bool,
    pub kv_source: usize,
    pub key_equals_value: bool,
    pub intermediate: usize,
    pub rope_theta: f64,
    pub rotary_fraction: f64,
}

#[derive(Clone, Debug)]
pub struct GemmaConfig {
    pub unified: bool,
    pub hidden: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub rms_eps: f64,
    pub sliding_window: usize,
    pub softcap: f64,
    pub per_layer_input: usize,
    pub per_layer_vocab: usize,
    pub layers: Vec<GemmaLayerConfig>,
    pub vision: Option<Value>,
    pub audio: Option<Value>,
    pub image_token: u32,
    pub audio_token: u32,
    pub pad_token: u32,
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    let n = value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .with_context(|| format!("config.json: integer {key}"))?;
    ensure!(n > 0, "config.json: {key} must be positive");
    Ok(n)
}

fn number(value: &Value, key: &str) -> Result<f64> {
    let n = value.get(key).and_then(Value::as_f64).with_context(|| format!("config.json: number {key}"))?;
    ensure!(n.is_finite() && n > 0.0, "config.json: {key} must be positive and finite");
    Ok(n)
}

impl Architecture {
    pub fn parse(config: &Value) -> Result<Self> {
        ensure!(config.get("quantization_config").is_none_or(Value::is_null),
            "serialized checkpoint quantization (including ModelOpt FP8) is not supported; use the base BF16 checkpoint");
        let unified = match config.get("model_type").and_then(Value::as_str) {
            Some("llama") => return Ok(Self::Llama(parse_config(config)?)),
            Some("gemma4") => false,
            Some("gemma4_unified") => true,
            _ => anyhow::bail!("config.json: unsupported architecture"),
        };
        let expected = if unified { "Gemma4UnifiedForConditionalGeneration" } else { "Gemma4ForConditionalGeneration" };
        ensure!(
            config.get("architectures") == Some(&serde_json::json!([expected])),
            "config.json: unsupported Gemma architecture"
        );
        let t = config.get("text_config").context("config.json: text_config")?;
        ensure!(
            t.get("model_type").and_then(Value::as_str)
                == Some(if unified { "gemma4_unified_text" } else { "gemma4_text" }),
            "config.json: text architecture mismatch"
        );
        ensure!(
            t.get("attention_bias").and_then(Value::as_bool) == Some(false),
            "Gemma attention biases not supported"
        );
        ensure!(
            t.get("hidden_activation").and_then(Value::as_str) == Some("gelu_pytorch_tanh"),
            "Gemma requires GELU tanh"
        );
        ensure!(t.get("tie_word_embeddings").and_then(Value::as_bool) == Some(true), "Gemma requires tied embeddings");
        ensure!(t.get("enable_moe_block").and_then(Value::as_bool) == Some(false), "Gemma MoE is not supported");
        ensure!(
            t.get("use_bidirectional_attention").is_none_or(|v| v.is_null() || v.as_str() == Some("vision")),
            "unsupported Gemma bidirectional attention"
        );
        let n = integer(t, "num_hidden_layers")?;
        ensure!(n <= 128, "unsupported Gemma layer count");
        let shared = t
            .get("num_kv_shared_layers")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .context("config.json: num_kv_shared_layers")?;
        ensure!(shared < n, "Gemma shared KV needs preceding producer layers");
        let kinds = t
            .get("layer_types")
            .and_then(Value::as_array)
            .filter(|v| v.len() == n)
            .context("config.json: layer_types")?;
        let base_heads = integer(t, "num_attention_heads")?;
        let base_kv = integer(t, "num_key_value_heads")?;
        let base_dim = integer(t, "head_dim")?;
        let overrides =
            t.get("per_layer_config").and_then(Value::as_object).context("config.json: per_layer_config")?;
        ensure!(
            overrides.keys().all(|k| k.parse::<usize>().is_ok_and(|i| i < n)),
            "unsupported per-layer configuration keys"
        );
        let mut layers = Vec::with_capacity(n);
        for (i, kind) in kinds.iter().enumerate() {
            let kind = kind.as_str().context("config.json: layer type must be string")?;
            ensure!(matches!(kind, "full_attention" | "sliding_attention"), "unsupported Gemma layer type {kind}");
            let sliding = kind == "sliding_attention";
            let entry = overrides.get(&format!("{i:02}")).or_else(|| overrides.get(&i.to_string()));
            if let Some(e) = entry {
                ensure!(
                    e.as_object()
                        .is_some_and(|m| m.keys().all(|k| matches!(k.as_str(), "head_dim" | "num_key_value_heads"))),
                    "unsupported Gemma layer override"
                );
            }
            let dim = entry.filter(|e| e.get("head_dim").is_some()).map_or(Ok(base_dim), |e| integer(e, "head_dim"))?;
            let kv = entry
                .filter(|e| e.get("num_key_value_heads").is_some())
                .map_or(Ok(base_kv), |e| integer(e, "num_key_value_heads"))?;
            ensure!(dim.is_multiple_of(2) && base_heads.is_multiple_of(kv), "invalid Gemma head dimensions");
            let rope = t.pointer(&format!("/rope_parameters/{kind}")).context("config.json: Gemma RoPE")?;
            let rope_type = rope.get("rope_type").and_then(Value::as_str).context("config.json: rope_type")?;
            ensure!(matches!(rope_type, "default" | "proportional"), "unsupported Gemma RoPE");
            let fraction = if rope_type == "proportional" { number(rope, "partial_rotary_factor")? } else { 1.0 };
            ensure!(
                fraction <= 1.0 && rope.get("factor").is_none_or(|v| v.as_f64() == Some(1.0)),
                "unsupported Gemma rotary scaling"
            );
            let shared_kv = i >= n - shared;
            let source = if shared_kv {
                (0..n - shared).rev().find(|&j| kinds[j] == kinds[i]).context("Gemma KV producer missing")?
            } else {
                i
            };
            let intermediate = integer(t, "intermediate_size")?;
            let intermediate = if shared_kv && t.get("use_double_wide_mlp").and_then(Value::as_bool) == Some(true) {
                intermediate.checked_mul(2).context("MLP dimension overflow")?
            } else {
                intermediate
            };
            layers.push(GemmaLayerConfig {
                heads: base_heads,
                kv_heads: kv,
                head_dim: dim,
                sliding,
                shared_kv,
                kv_source: source,
                key_equals_value: t.get("attention_k_eq_v").and_then(Value::as_bool) == Some(true) && !sliding,
                intermediate,
                rope_theta: number(rope, "rope_theta")?,
                rotary_fraction: fraction,
            });
        }
        for layer in layers.iter().filter(|l| l.shared_kv) {
            let source = &layers[layer.kv_source];
            ensure!(
                source.head_dim == layer.head_dim && source.kv_heads == layer.kv_heads,
                "shared Gemma KV dimensions differ"
            );
        }
        let token = |value: &Value, key| {
            value
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .with_context(|| format!("config.json: token {key}"))
        };
        let per_layer_input = t
            .get("hidden_size_per_layer_input")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .context("config.json: hidden_size_per_layer_input")?;
        ensure!(!unified || per_layer_input == 0, "unified per-layer embeddings not supported");
        let cfg = GemmaConfig {
            unified,
            hidden: integer(t, "hidden_size")?,
            vocab: integer(t, "vocab_size")?,
            max_positions: integer(t, "max_position_embeddings")?,
            rms_eps: number(t, "rms_norm_eps")?,
            sliding_window: integer(t, "sliding_window")?,
            softcap: number(t, "final_logit_softcapping")?,
            per_layer_input,
            per_layer_vocab: integer(t, "vocab_size_per_layer_input")?,
            layers,
            vision: config.get("vision_config").filter(|v| !v.is_null()).cloned(),
            audio: config.get("audio_config").filter(|v| !v.is_null()).cloned(),
            image_token: token(config, "image_token_id")?,
            audio_token: token(config, "audio_token_id")?,
            pad_token: token(t, "pad_token_id")?,
        };
        cfg.validate()?;
        Ok(Self::Gemma4(cfg))
    }

    pub fn llama(&self) -> Result<&LlamaConfig> {
        match self {
            Self::Llama(c) => Ok(c),
            _ => anyhow::bail!("this operation only supports Llama models"),
        }
    }
    pub fn max_positions(&self) -> usize {
        match self {
            Self::Llama(c) => c.max_positions,
            Self::Gemma4(c) => c.max_positions,
        }
    }
    pub fn num_layers(&self) -> usize {
        match self {
            Self::Llama(c) => c.num_layers,
            Self::Gemma4(c) => c.layers.len(),
        }
    }
}

impl GemmaConfig {
    /// Validate dimensions and sharing before allocation, including callers constructing a config directly.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=128).contains(&self.layers.len())
                && (1..=32768).contains(&self.hidden)
                && (1..=4_000_000).contains(&self.vocab)
                && (1..=1_048_576).contains(&self.max_positions),
            "unsupported Gemma dimensions"
        );
        ensure!(
            self.sliding_window > 0
                && self.sliding_window <= self.max_positions
                && self.rms_eps.is_finite()
                && self.rms_eps > 0.0
                && self.softcap.is_finite()
                && self.softcap > 0.0,
            "invalid Gemma normalization, context or softcap"
        );
        ensure!(
            self.image_token != self.audio_token
                && self.image_token < self.vocab as u32
                && self.audio_token < self.vocab as u32
                && self.pad_token < self.vocab as u32,
            "invalid Gemma special tokens"
        );
        ensure!(
            self.per_layer_input <= 4096
                && (!self.unified || self.per_layer_input == 0)
                && self.layers.len().checked_mul(self.per_layer_input).is_some_and(|n| n <= 32768)
                && (self.per_layer_input == 0
                    || (self.per_layer_vocab > self.pad_token as usize && self.per_layer_vocab <= 4_000_000)),
            "unsupported Gemma per-layer embeddings"
        );
        for (i, c) in self.layers.iter().enumerate() {
            ensure!(
                (1..=256).contains(&c.heads)
                    && c.kv_heads > 0
                    && c.heads.is_multiple_of(c.kv_heads)
                    && (2..=4096).contains(&c.head_dim)
                    && c.head_dim.is_multiple_of(2)
                    && (1..=262144).contains(&c.intermediate)
                    && c.rotary_fraction.is_finite()
                    && c.rotary_fraction > 0.0
                    && c.rotary_fraction <= 1.0
                    && c.rope_theta.is_finite()
                    && c.rope_theta > 0.0,
                "invalid Gemma layer {i} dimensions or RoPE"
            );
            ensure!(
                (!c.shared_kv && c.kv_source == i) || (c.shared_kv && c.kv_source < i),
                "invalid KV source at layer {i}"
            );
            let source = &self.layers[c.kv_source];
            ensure!(
                source.head_dim == c.head_dim && source.kv_heads == c.kv_heads && !source.shared_kv,
                "invalid KV producer at layer {i}"
            );
        }
        Ok(())
    }
}
