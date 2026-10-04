//! Model directory manifest: architecture checks of `config.json`, the readout protocol of `basal.json`, the chat
//! template and per-type calibration. Anything this runtime does not implement is rejected instead of ignored.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::prompt::{Lang, CHAT_TEMPLATE, LETTERS, PREFILL, TEMPLATE_EN, TEMPLATE_PL};
use crate::request::QType;

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub num_layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub max_positions: usize,
    /// Attention (q, k, v, o) and MLP projections carry biases (basal-1.0); basal-1.5 has none.
    pub bias: bool,
}

#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub dir: PathBuf,
    pub name: String,
    pub version: String,
    pub config: LlamaConfig,
    pub temperatures: HashMap<QType, f64>,
    /// CALIBRATION.json `thresholds` (certified confidence per target error, keyed by Python `str()` of the target).
    pub thresholds: serde_json::Map<String, Value>,
    pub bos_token: String,
}

fn read_json(path: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}

fn usize_of(cfg: &Value, key: &str) -> Result<usize> {
    cfg.get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .with_context(|| format!("config.json: missing integer {key}"))
}

fn f64_of(v: Option<&Value>, what: &str) -> Result<f64> {
    v.and_then(|x| x.as_f64()).with_context(|| format!("missing number {what}"))
}

pub fn parse_config(cfg: &Value) -> Result<LlamaConfig> {
    let expect = |key: &str, want: Value| -> Result<()> {
        let got = cfg.get(key).cloned().unwrap_or(Value::Null);
        ensure!(got == want, "config.json: {key} = {got}, this runtime implements {want}");
        Ok(())
    };
    expect("model_type", "llama".into())?;
    expect("architectures", serde_json::json!(["LlamaForCausalLM"]))?;
    let bias = cfg.get("attention_bias").and_then(Value::as_bool).context("config.json: attention_bias")?;
    ensure!(
        cfg.get("mlp_bias").and_then(Value::as_bool) == Some(bias),
        "config.json: attention_bias and mlp_bias differ (not implemented)"
    );
    expect("hidden_act", "silu".into())?;
    expect("tie_word_embeddings", false.into())?;
    if let Some(tp) = cfg.get("pretraining_tp") {
        ensure!(tp.as_u64() == Some(1), "config.json: pretraining_tp {tp} not supported");
    }
    if cfg.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        bail!("config.json: rope_scaling is not supported");
    }
    // transformers 5 keeps rope_theta inside rope_parameters; a top-level value or a default of 10000 would be wrong.
    let rope = cfg.get("rope_parameters").context("config.json: rope_parameters missing")?;
    ensure!(
        rope.get("rope_type").and_then(Value::as_str).unwrap_or("default") == "default",
        "config.json: only default RoPE is supported, got {rope}"
    );
    let rope_theta = f64_of(rope.get("rope_theta"), "rope_parameters.rope_theta")?;
    let c = LlamaConfig {
        num_layers: usize_of(cfg, "num_hidden_layers")?,
        hidden: usize_of(cfg, "hidden_size")?,
        intermediate: usize_of(cfg, "intermediate_size")?,
        heads: usize_of(cfg, "num_attention_heads")?,
        kv_heads: usize_of(cfg, "num_key_value_heads")?,
        head_dim: usize_of(cfg, "head_dim")?,
        vocab: usize_of(cfg, "vocab_size")?,
        rms_eps: f64_of(cfg.get("rms_norm_eps"), "rms_norm_eps")?,
        rope_theta,
        max_positions: usize_of(cfg, "max_position_embeddings")?,
        bias,
    };
    ensure!(c.heads * c.head_dim == c.hidden, "heads * head_dim != hidden");
    ensure!(c.heads.is_multiple_of(c.kv_heads), "heads not divisible by kv heads");
    ensure!(c.head_dim.is_multiple_of(2), "odd head_dim");
    Ok(c)
}

impl ModelManifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let config = parse_config(&read_json(&dir.join("config.json"))?)?;
        let basal = read_json(&dir.join("basal.json"))?;
        let ro = basal.get("readout").context("basal.json: readout missing")?;
        ensure!(
            ro.get("kind").and_then(Value::as_str) == Some("letter logits"),
            "basal.json: unsupported readout kind"
        );
        ensure!(
            ro.get("letters").and_then(Value::as_str) == Some(LETTERS),
            "basal.json: letters differ from {LETTERS}"
        );
        ensure!(ro.get("prefill").and_then(Value::as_str) == Some(PREFILL), "basal.json: prefill differs");
        // basal-1.0 lists the templates; basal-1.5 declares that the 1.0 prompt contract is unchanged.
        match ro.get("templates") {
            Some(Value::String(s)) => ensure!(
                s == "basal-1.0 (unchanged prompt contract)",
                "basal.json: templates {s:?} (only the basal-1.0 prompt contract is implemented)"
            ),
            _ => {
                for (lang, t) in [(Lang::Pl, &TEMPLATE_PL), (Lang::En, &TEMPLATE_EN)] {
                    let j = ro
                        .pointer(&format!("/templates/{lang}"))
                        .with_context(|| format!("basal.json: template {lang}"))?;
                    ensure!(
                        j.get("system").and_then(Value::as_str) == Some(t.system)
                            && j.get("user").and_then(Value::as_str) == Some(t.user),
                        "basal.json: {lang} template differs from the implemented prompt"
                    );
                }
            }
        }
        let tmpl = std::fs::read_to_string(dir.join("chat_template.jinja")).context("chat_template.jinja")?;
        ensure!(tmpl == CHAT_TEMPLATE, "chat_template.jinja differs from the implemented template");
        let tcfg = read_json(&dir.join("tokenizer_config.json"))?;
        let bos_token =
            tcfg.get("bos_token").and_then(Value::as_str).context("tokenizer_config.json: bos_token")?.to_string();
        let mut temperatures = HashMap::new();
        let mut thresholds = serde_json::Map::new();
        let cal = dir.join("CALIBRATION.json");
        if cal.exists() {
            let c = read_json(&cal)?;
            if let Some(t) = c.get("thresholds").and_then(Value::as_object) {
                thresholds = t.clone();
            }
            if let Some(m) = c.get("temperature_per_prim").and_then(Value::as_object) {
                for (k, v) in m {
                    let t = QType::parse(k).with_context(|| format!("CALIBRATION.json: unknown type {k}"))?;
                    temperatures.insert(t, v.as_f64().context("CALIBRATION.json: temperature")?);
                }
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            name: basal.get("name").and_then(Value::as_str).context("basal.json: name")?.to_string(),
            version: basal.get("version").and_then(Value::as_str).unwrap_or("").to_string(),
            config,
            temperatures,
            thresholds,
            bos_token,
        })
    }

    pub fn temperature(&self, t: QType) -> f64 {
        self.temperatures.get(&t).copied().unwrap_or(1.0)
    }
}
