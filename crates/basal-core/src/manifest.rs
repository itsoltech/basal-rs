//! Model directory manifest: architecture checks of `config.json`, the readout protocol of `basal.json`, the chat
//! template and per-type calibration. Anything this runtime does not implement is rejected instead of ignored.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::calibration::{CalibrationProvenance, CalibrationSource, ServedIdentity};
use crate::prompt::{ChatStyle, Lang, OptionOrders, PromptFormat, PromptLayout, CHAT_TEMPLATE, LETTERS, PREFILL};
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
    /// Whether attention/MLP projections carry biases (mini/4.5B do; max does not).
    pub bias: bool,
}

#[derive(Clone, Debug)]
pub struct ModelManifest {
    pub dir: PathBuf,
    pub name: String,
    pub version: String,
    pub config: crate::architecture::Architecture,
    pub chat_style: ChatStyle,
    pub temperatures: HashMap<QType, f64>,
    /// CALIBRATION.json `thresholds` (certified confidence per target error, keyed by Python `str()` of the target).
    pub thresholds: serde_json::Map<String, Value>,
    /// Certified confidence thresholds fitted for each experimental task, independent of primitive thresholds.
    pub task_thresholds: serde_json::Map<String, Value>,
    pub bos_token: String,
    pub prompt_format: PromptFormat,
    pub layout: PromptLayout,
    pub calibration_provenance: CalibrationProvenance,
    pub orders: OptionOrders,
    pub temperature_buckets: HashMap<QType, Vec<TemperatureBucket>>,
    /// Calibration fallback rules keyed by upstream label_set_id.
    pub multi_calibration: serde_json::Map<String, Value>,
    /// Why temperatures are usable but certificates and fitted fallbacks were discarded.
    pub calibration_unvalidated: Option<String>,
    calibration_prompt: PromptFormat,
}

#[derive(Clone, Debug)]
pub struct TemperatureBucket {
    pub lo: usize,
    pub hi: usize,
    pub temperature: f64,
}

/// Additional serving settings. Existing loaders use these defaults and honor basal.json's layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelLoadOptions<'a> {
    pub orders: Option<&'a str>,
    pub layout: Option<PromptLayout>,
    pub calibration: CalibrationSource<'a>,
    pub strict_calibration: bool,
    /// When unavailable here, the application must call `calibration_provenance.check_identity` later.
    pub served_identity: Option<ServedIdentity<'a>>,
}

fn read_json(path: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
}

fn usize_of(cfg: &Value, key: &str) -> Result<usize> {
    cfg.get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
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
    ensure!(
        [c.num_layers, c.hidden, c.intermediate, c.heads, c.kv_heads, c.head_dim, c.vocab, c.max_positions]
            .iter()
            .all(|n| *n > 0),
        "config.json: model dimensions must be positive"
    );
    ensure!(
        c.rms_eps.is_finite() && c.rms_eps > 0.0 && c.rope_theta.is_finite() && c.rope_theta > 0.0,
        "config.json: RMS epsilon and RoPE theta must be positive and finite"
    );
    ensure!(c.heads.checked_mul(c.head_dim) == Some(c.hidden), "heads * head_dim != hidden");
    ensure!(c.heads.is_multiple_of(c.kv_heads), "heads not divisible by kv heads");
    ensure!(c.head_dim.is_multiple_of(2), "odd head_dim");
    Ok(c)
}

/// Shared semantic stages; filesystem reads remain at the legacy call sites.
pub(crate) struct ManifestProtocol {
    config: crate::architecture::Architecture,
    chat_style: ChatStyle,
    prompt_format: PromptFormat,
    layout: PromptLayout,
    orders: OptionOrders,
}

impl ManifestProtocol {
    pub(crate) fn parse(
        config: crate::architecture::Architecture,
        basal: &Value,
        options: ModelLoadOptions<'_>,
    ) -> Result<Self> {
        let chat_style = match &config {
            crate::architecture::Architecture::Llama(_) => ChatStyle::Llama,
            crate::architecture::Architecture::Gemma4(_) => ChatStyle::Gemma4,
            crate::architecture::Architecture::Qwen35(_) => ChatStyle::Qwen35,
        };
        let prompt_format = match basal.get("prompt") {
            None => PromptFormat::One,
            Some(v) => PromptFormat::parse(v.as_u64().context("basal.json: prompt must be an integer")?)?,
        };
        let orders = options
            .orders
            .map(OptionOrders::parse)
            .transpose()?
            .unwrap_or_else(|| OptionOrders::default_for(prompt_format));
        let model_layout = basal
            .get("layout")
            .map(|v| {
                PromptLayout::parse(v.as_str().context("basal.json: layout must be a string")?)
                    .map_err(anyhow::Error::from)
            })
            .transpose()?
            .unwrap_or_default();
        let layout = options.layout.unwrap_or(model_layout);
        layout.validate(prompt_format)?;
        let ro = basal.get("readout").context("basal.json: readout missing")?;
        ensure!(ro.get("prefill").and_then(Value::as_str) == Some(PREFILL), "basal.json: prefill differs");
        match prompt_format {
            PromptFormat::One => {
                ensure!(
                    ro.get("kind").and_then(Value::as_str) == Some("letter logits"),
                    "basal.json: unsupported readout kind"
                );
                ensure!(
                    ro.get("letters").and_then(Value::as_str) == Some(LETTERS),
                    "basal.json: letters differ from {LETTERS}"
                );
                match ro.get("templates") {
                    Some(Value::String(s)) => {
                        ensure!(s == "basal-1.0 (unchanged prompt contract)", "basal.json: unsupported templates {s:?}")
                    }
                    _ => {
                        for lang in [Lang::Pl, Lang::En] {
                            let t = prompt_format.template(lang);
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
            }
            PromptFormat::Two => {
                ensure!(
                    ro.get("kind").and_then(Value::as_str) == Some("code logits"),
                    "basal.json: format 2 requires code logits"
                );
                ensure!(
                    ro.get("codes").and_then(Value::as_u64) == Some(255),
                    "basal.json: format 2 requires 255 codes"
                );
            }
        }
        Ok(Self { config, chat_style, prompt_format, layout, orders })
    }

    pub(crate) fn chat_style(&self) -> ChatStyle {
        self.chat_style
    }

    pub(crate) fn layout(&self) -> PromptLayout {
        self.layout
    }

    pub(crate) fn orders(&self) -> OptionOrders {
        self.orders
    }

    pub(crate) fn prompt_format(&self) -> PromptFormat {
        self.prompt_format
    }

    pub(crate) fn validate_template(&mut self, tmpl: &str) -> Result<()> {
        let Self { chat_style, prompt_format, .. } = *self;
        match chat_style {
            ChatStyle::Qwen35 => {
                use sha2::{Digest, Sha256};
                ensure!(
                    format!("{:x}", Sha256::digest(tmpl.as_bytes()))
                        == "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
                    "unsupported Qwen3.5 chat template"
                );
                ensure!(prompt_format == PromptFormat::Two, "Qwen3.5 Basal needs prompt format 2");
            }
            ChatStyle::Llama => {
                ensure!(tmpl == CHAT_TEMPLATE, "chat_template.jinja differs from the implemented template")
            }
            ChatStyle::Gemma4 | ChatStyle::Gemma4Unified => {
                use sha2::{Digest, Sha256};
                let hash = format!("{:x}", Sha256::digest(tmpl.as_bytes()));
                ensure!(
                    matches!(
                        hash.as_str(),
                        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
                            | "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
                    ),
                    "unsupported Gemma chat template"
                );
                // Renderer grammar follows the validated template, independently of the decoder architecture.
                // Ultra uses the same closed thought prefill as Unified; the older E4B template does not.
                self.chat_style = if hash == "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4" {
                    ChatStyle::Gemma4Unified
                } else {
                    ChatStyle::Gemma4
                };
                ensure!(prompt_format == PromptFormat::Two, "Gemma Basal needs prompt format 2");
            }
        }
        Ok(())
    }

    pub(crate) fn parse_bos(&self, tcfg: &Value) -> Result<String> {
        Ok(if self.chat_style == ChatStyle::Qwen35 {
            ensure!(tcfg.get("bos_token").is_none_or(Value::is_null), "Qwen3.5 expects no BOS token");
            String::new()
        } else {
            tcfg.get("bos_token").and_then(Value::as_str).context("tokenizer_config.json: bos_token")?.to_string()
        })
    }

    pub(crate) fn finish(
        self,
        dir: &Path,
        basal: &Value,
        bos_token: String,
        loaded: crate::calibration::LoadedCalibration,
    ) -> Result<ModelManifest> {
        let Self { config, chat_style, prompt_format, layout, orders } = self;
        let crate::calibration::LoadedCalibration {
            data: calibration,
            provenance: calibration_provenance,
            unvalidated: calibration_unvalidated,
        } = loaded;
        let calibration_prompt = match calibration.get("prompt") {
            None => prompt_format,
            Some(v) => PromptFormat::parse(v.as_u64().context("CALIBRATION.json: prompt must be an integer")?)?,
        };
        ensure!(calibration_prompt == prompt_format, "calibration was fitted for another prompt format");
        let mut temperatures = HashMap::new();
        let mut temperature_buckets = HashMap::new();
        let temperature = |v: &Value| -> Result<f64> {
            let t = v.as_f64().context("calibration temperature must be a number")?;
            ensure!(t.is_finite() && t > 0.0, "calibration temperature must be positive and finite");
            Ok(t)
        };
        if let Some(value) = calibration.get("temperature_per_prim") {
            for (k, v) in value.as_object().context("temperature_per_prim must be an object")? {
                let kind = QType::parse(k).with_context(|| format!("calibration: unknown type {k}"))?;
                temperatures.insert(kind, temperature(v)?);
            }
        }
        if let Some(value) = calibration.get("temperature_per_bucket") {
            for (k, buckets) in value.as_object().context("temperature_per_bucket must be an object")? {
                let kind = QType::parse(k).with_context(|| format!("calibration: unknown type {k}"))?;
                let mut parsed = Vec::new();
                for (key, v) in buckets.as_object().context("calibration buckets must be an object")? {
                    let (lo, hi) = key.split_once('-').with_context(|| format!("bad calibration bucket {key:?}"))?;
                    ensure!(
                        !lo.is_empty() && !hi.is_empty() && lo.bytes().chain(hi.bytes()).all(|x| x.is_ascii_digit()),
                        "bad calibration bucket {key:?}"
                    );
                    let (lo, hi) = (lo.parse::<usize>()?, hi.parse::<usize>()?);
                    ensure!(lo <= hi, "bad calibration bucket {key:?}");
                    parsed.push(TemperatureBucket { lo, hi, temperature: temperature(v)? });
                }
                temperature_buckets.insert(kind, parsed);
            }
        }
        let map = |key| -> Result<serde_json::Map<String, Value>> {
            match calibration.get(key) {
                None => Ok(Default::default()),
                Some(v) => Ok(v.as_object().with_context(|| format!("calibration {key} must be an object"))?.clone()),
            }
        };
        let thresholds = if calibration_unvalidated.is_none() { map("thresholds")? } else { Default::default() };
        let task_thresholds =
            if calibration_unvalidated.is_none() { map("task_thresholds")? } else { Default::default() };
        for (task, rates) in &task_thresholds {
            let rates = rates.as_object().with_context(|| format!("task_thresholds.{task} must be an object"))?;
            for (rate, value) in rates {
                let risk: f64 = rate.parse().with_context(|| format!("invalid task error rate {rate:?}"))?;
                ensure!(risk.is_finite() && risk > 0.0 && risk < 1.0, "task error rate must be in (0, 1)");
                let threshold = value.as_f64().context("task threshold must be a number")?;
                ensure!(threshold.is_finite() && (0.0..=1.0).contains(&threshold), "task threshold must be in [0, 1]");
            }
        }
        let multi_calibration = if calibration_unvalidated.is_none() { map("multi")? } else { Default::default() };
        Ok(ModelManifest {
            dir: dir.to_path_buf(),
            name: basal.get("name").and_then(Value::as_str).context("basal.json: name")?.to_string(),
            version: basal.get("version").and_then(Value::as_str).unwrap_or("").to_string(),
            config,
            chat_style,
            temperatures,
            thresholds,
            task_thresholds,
            bos_token,
            prompt_format,
            layout,
            calibration_provenance,
            orders,
            temperature_buckets,
            multi_calibration,
            calibration_unvalidated,
            calibration_prompt,
        })
    }
}

impl ModelManifest {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with_orders(dir, None)
    }

    /// Select a trained protocol and calibration matching the served option-order policy.
    pub fn load_with_orders(dir: &Path, orders: Option<&str>) -> Result<Self> {
        Self::load_with_options(dir, ModelLoadOptions { orders, ..Default::default() })
    }

    pub fn load_with_options(dir: &Path, options: ModelLoadOptions<'_>) -> Result<Self> {
        let config = crate::architecture::Architecture::parse(&read_json(&dir.join("config.json"))?)?;
        let basal = read_json(&dir.join("basal.json"))?;
        let mut protocol = ManifestProtocol::parse(config, &basal, options)?;
        let tmpl = std::fs::read_to_string(dir.join("chat_template.jinja")).context("chat_template.jinja")?;
        protocol.validate_template(&tmpl)?;
        let tcfg = read_json(&dir.join("tokenizer_config.json"))?;
        let bos = protocol.parse_bos(&tcfg)?;
        let (data, provenance, unvalidated) = crate::calibration::load(
            dir,
            options.calibration,
            protocol.layout(),
            protocol.orders(),
            options.strict_calibration,
            options.served_identity,
        )?;
        protocol.finish(dir, &basal, bos, crate::calibration::LoadedCalibration { data, provenance, unvalidated })
    }

    pub fn temperature(&self, t: QType) -> f64 {
        self.temperatures.get(&t).copied().unwrap_or(1.0)
    }

    pub fn temperature_for(&self, t: QType, options: usize) -> f64 {
        self.temperature_buckets
            .get(&t)
            .and_then(|xs| xs.iter().find(|b| b.lo <= options && options <= b.hi))
            .map_or_else(|| self.temperature(t), |b| b.temperature)
    }

    pub fn has_calibration(&self) -> bool {
        !self.temperatures.is_empty() || !self.temperature_buckets.is_empty()
    }

    /// Format-1 files predating per-type certificates retain their pooled certificate; format 2 never does.
    pub fn certified(&self, target: &str, kind: QType) -> Option<f64> {
        let entry = self.thresholds.get(target)?;
        let value = if entry.get("per_type").is_none() && self.calibration_prompt == PromptFormat::One {
            entry.get("confidence")
        } else {
            entry.pointer(&format!("/per_type/{}/confidence", kind.as_str()))
        };
        value.and_then(Value::as_f64).filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
    }
}
