//! Opt-in native thinking: a separate full checkpoint generates notes and performs the second code readout.
//! Prefixes are recomputed on every greedy step; no KV session or speed/token-parity claim is implied.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::decision::{calibrate32, softmax32};
use crate::execution::StopToken;
use crate::manifest::ModelManifest;
use crate::pack::pack;
use crate::prompt::{render_model_with_layout, ChatStyle, OptionOrders, PromptLayout, PREFILL};
use crate::request::{DecideError, Ext, FieldError, Item, QType};
use crate::{tokenize::BasalTokenizer, Backend};

pub const THINK_REPO: &str = "Remek/basal-2.0-4.5B";
pub const THINK_REVISION: &str = "e5ba9149760d99b73b1963d8e0ccf948b58afacb";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkMode {
    #[default]
    Off,
    Auto,
    Force,
}

/// Validated immutable generation limits. The real checkpoint context further caps `max_len` at load time.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ThinkSettings {
    pub tokens: usize,
    pub max_len: usize,
}

impl ThinkSettings {
    pub fn new(tokens: usize, max_len: usize) -> Result<Self> {
        ensure!(
            tokens > 0 && max_len > 0 && max_len <= u32::MAX as usize,
            "thinking limits must be positive and fit u32"
        );
        ensure!(tokens.checked_add(16).is_some_and(|n| n < max_len), "think_tokens + 16 must be below think_max_len");
        Ok(Self { tokens, max_len })
    }
}

/// CPU-only capability snapshot: no backend or model cache crosses into an offline planner.
#[derive(Clone, Debug)]
pub struct ThinkCaps {
    pub settings: ThinkSettings,
    pub vocab: usize,
    pub decoded_token_bytes: usize,
    pub host_staging: Option<crate::planner::HostStaging>,
    pub device_unavailable: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ThinkRequest {
    pub mode: ThinkMode,
    pub text: bool,
    /// Enabled server emits metadata even for explicit off and ineligible questions.
    pub enabled: bool,
}

impl ThinkRequest {
    pub(crate) fn parse(body: &Value, caps: Option<&ThinkCaps>) -> Result<Self, DecideError> {
        let invalid = |field: &str, msg| {
            DecideError::Validation(vec![FieldError::new(vec!["body".into(), field.into()], msg, "value_error")])
        };
        let mode = match body.get("think") {
            None => {
                if caps.is_some() {
                    ThinkMode::Auto
                } else {
                    ThinkMode::Off
                }
            }
            Some(Value::String(s)) => match s.as_str() {
                "off" => ThinkMode::Off,
                "auto" => ThinkMode::Auto,
                "force" => ThinkMode::Force,
                _ => return Err(invalid("think", "think must be off, auto or force")),
            },
            Some(_) => return Err(invalid("think", "think must be off, auto or force")),
        };
        let text = match body.get("think_text") {
            None => false,
            Some(Value::Bool(v)) => *v,
            Some(_) => return Err(invalid("think_text", "think_text must be a boolean")),
        };
        let unsupported = |msg| {
            DecideError::Unsupported(vec![FieldError::new(
                vec!["body".into(), "think".into()],
                msg,
                "unsupported_feature",
            )])
        };
        if mode != ThinkMode::Off {
            if caps.is_none() {
                return Err(unsupported(
                    "thinking requires a server enabled with the published base + think checkpoint",
                ));
            }
            if body.get("questions").and_then(Value::as_object).is_some_and(|qs| {
                qs.values().any(|q| {
                    matches!(
                        q.get("type").and_then(Value::as_str),
                        Some("classify" | "verify" | "compress" | "extract")
                    )
                })
            }) {
                return Err(unsupported("active thinking is not supported with tasks; use think off"));
            }
        }
        Ok(Self { mode, text, enabled: caps.is_some() })
    }
}

/// Concrete optional second backend. Its manifest is canonical/order1 with independent calibration.
pub struct Thinking<B> {
    pub(crate) manifest: ModelManifest,
    pub(crate) tok: BasalTokenizer,
    pub(crate) backend: B,
    pub(crate) caps: ThinkCaps,
    pub(crate) calibration_label: String,
    pub(crate) sources: Value,
    eos: Vec<u32>,
}

pub(crate) struct Thought {
    pub probabilities: Option<Vec<f64>>,
    pub details: Value,
    pub tokens: usize,
}

impl<B: Backend> Thinking<B> {
    /// Construct only after the application validates the published source identity and independent weights.
    pub fn new(
        manifest: ModelManifest,
        backend: B,
        settings: ThinkSettings,
        eos: Vec<u32>,
        calibration_label: String,
        sources: Value,
    ) -> Result<Self> {
        ensure!(
            manifest.chat_style == ChatStyle::Llama
                && manifest.layout == PromptLayout::Canonical
                && manifest.orders == OptionOrders::One,
            "thinking requires canonical Llama order1"
        );
        let settings = ThinkSettings::new(settings.tokens, settings.max_len.min(manifest.config.max_positions()))?;
        let tok = BasalTokenizer::load_for(&manifest.dir.join("tokenizer.json"), manifest.chat_style)?;
        let vocab = tok.vocab_size();
        ensure!(
            vocab > 0
                && u32::try_from(vocab).is_ok()
                && tok.has_dense_vocabulary()
                && eos.iter().all(|id| (*id as usize) < vocab),
            "invalid thinking vocabulary/EOS"
        );
        let shape = crate::planner::DeviceWorkShape {
            forward_tokens: settings.max_len as u64,
            longest_prompt: settings.max_len as u64,
            forward_readouts: 1,
            forward_letters: vocab as u64,
            prompt_letters: vocab as u64,
            short_prompts: false,
            evidence: None,
            media: Vec::new(),
            worst_case_rounds: false,
        };
        let device_unavailable = match backend.device_work_bound(&shape) {
            crate::planner::DeviceWorkBound::Unknown(reason) => Some(reason),
            crate::planner::DeviceWorkBound::Estimate(_) => None,
        };
        let caps = ThinkCaps {
            settings,
            vocab,
            decoded_token_bytes: tok.llama_decoded_token_bytes()?,
            host_staging: backend.host_staging(),
            device_unavailable,
        };
        Ok(Self { manifest, tok, backend, caps, calibration_label, sources, eos })
    }

    pub(crate) fn fork(&self) -> Result<Self> {
        Self::new(
            self.manifest.clone(),
            self.backend.fork()?,
            self.caps.settings,
            self.eos.clone(),
            self.calibration_label.clone(),
            self.sources.clone(),
        )
    }

    pub(crate) fn readout(&mut self, ids: &[u32], codes: Vec<u32>) -> Result<Vec<f32>> {
        ensure!(!ids.is_empty() && ids.len() <= self.caps.settings.max_len, "thinking context exceeded");
        let packed = pack(&[ids.to_vec()]);
        let codes = [codes];
        let mut result = self.backend.letter_logits(&[&packed], &[&codes])?;
        ensure!(
            result.len() == 1 && result[0].len() == 1 && result[0][0].len() == codes[0].len(),
            "thinking readout shape mismatch"
        );
        let logits = result.remove(0).remove(0);
        ensure!(logits.iter().all(|x| x.is_finite()), "nonfinite thinking logits");
        Ok(logits)
    }

    pub(crate) fn think(&mut self, item: &Item, stop: Option<&StopToken>) -> Result<Thought> {
        let check = || -> Result<()> {
            if let Some(stop) = stop {
                stop.check()?;
            }
            Ok(())
        };
        check()?;
        let start = Instant::now();
        let prompt = render_model_with_layout(
            self.manifest.chat_style,
            self.manifest.prompt_format,
            PromptLayout::Canonical,
            &self.manifest.bos_token,
            &item.state,
            &item.question,
            &item.options.iter().map(String::as_str).collect::<Vec<_>>(),
            item.lang,
        )?;
        let head = format!("{}<think>\n", prompt.strip_suffix(PREFILL).context("thinking prefill invariant")?);
        let mut prefix = self.tok.encode(&head)?;
        if prefix
            .len()
            .checked_add(self.caps.settings.tokens)
            .and_then(|n| n.checked_add(16))
            .is_none_or(|n| n > self.caps.settings.max_len)
        {
            return Ok(Thought { probabilities: None, tokens: 0, details: json!({"skipped":"context"}) });
        }
        let vocabulary: Vec<u32> = (0..u32::try_from(self.caps.vocab)?).collect();
        let mut generated = Vec::with_capacity(self.caps.settings.tokens);
        let mut notes = String::new();
        for _ in 0..self.caps.settings.tokens {
            check()?;
            let logits = self.readout(&prefix, vocabulary.clone())?;
            // Strict comparison preserves the lowest token ID on ties.
            let mut best = 0;
            for i in 1..logits.len() {
                if logits[i] > logits[best] {
                    best = i;
                }
            }
            let id = u32::try_from(best)?;
            generated.push(id);
            if self.eos.contains(&id) {
                break;
            }
            notes = self.tok.decode(&generated)?;
            if let Some(end) = notes.find("</think>") {
                notes.truncate(end);
                break;
            }
            prefix.push(id);
        }
        check()?;
        let second = format!("{head}{}\n</think>\n{PREFILL}", notes.trim());
        let ids = self.tok.encode(&second)?;
        // Re-encoding may expand the prefix. Never infer this length from generated token count.
        if ids.len() > self.caps.settings.max_len {
            return Ok(Thought { probabilities: None, tokens: generated.len(), details: json!({"skipped":"context"}) });
        }
        let codes = self
            .tok
            .context_code_ids(self.manifest.prompt_format, &second, &ids, item.options.len())?
            .context("thinking P2 codes are not single distinct tokens at the actual answer context")?;
        check()?;
        let probabilities = softmax32(&self.readout(&ids, codes)?);
        let temperature =
            if item.uncalibrated { 1.0 } else { self.manifest.temperature_for(item.temp_kind(), item.options.len()) };
        let probabilities = calibrate32(&probabilities, temperature);
        Ok(Thought {
            probabilities: Some(probabilities),
            tokens: generated.len(),
            details: json!({
                "tokens": generated.len(), "ms": start.elapsed().as_secs_f64()*1e3,
                "calibration": self.calibration_label, "text": notes,
            }),
        })
    }
}

/// Caps needed only when this plan can execute a thought. Off/truncated/ineligible plans keep the old budgets.
pub(crate) fn active_caps(plan: &crate::engine::RequestPlan) -> Option<&ThinkCaps> {
    (plan.thinking.mode != ThinkMode::Off
        && plan.truncated.is_none()
        && plan
            .questions
            .iter()
            .any(|q| matches!(q, crate::engine::QuestionPlan::Direct(i) if eligible(&plan.prepared[*i].item))))
    .then_some(plan.thinking_caps.as_ref())
    .flatten()
}

impl ThinkCaps {
    pub(crate) fn note_bytes(&self) -> u64 {
        (self.decoded_token_bytes as u64).saturating_mul(self.settings.tokens as u64)
    }

    /// A full context is the hard guard for every generation/P2 forward, including retokenization expansion.
    pub(crate) fn host_bound(&self, plan: &crate::engine::RequestPlan) -> Option<u64> {
        let staging = self.host_staging.as_ref()?;
        let context = self.settings.max_len as u64;
        let head_bytes: u64 = plan.prepared.iter().map(|p| crate::planner::item_bytes(&p.item)).max().unwrap_or(0);
        let text = head_bytes.saturating_add(self.note_bytes()).saturating_add(4096);
        let tokenizer = text.saturating_mul(crate::planner::TOKENIZER_BYTES_PER_BYTE);
        let retained =
            (plan.questions.len() as u64).saturating_mul(self.note_bytes().saturating_mul(8).saturating_add(65536));
        Some(
            tokenizer
                .saturating_add(text.saturating_mul(8))
                .saturating_add(retained)
                .saturating_add(context.saturating_mul(128))
                .saturating_add((self.vocab as u64).saturating_mul(64))
                .saturating_add(staging.bound(crate::planner::StagingShape {
                    tokens: context,
                    rows: 1,
                    readouts: 1,
                    letters: self.vocab as u64,
                    ..Default::default()
                })),
        )
    }
}

pub(crate) fn eligible(item: &Item) -> bool {
    matches!(item.kind, QType::Choice | QType::Noul | QType::Score | QType::Act | QType::Tool)
        && item.multi_spec().is_none()
        && item.options.len() >= 2
}

pub(crate) fn recalibrate_item(item: &mut Item, manifest: &ModelManifest) {
    match &mut item.ext {
        Ext::Act(a) => {
            a.calibrated = manifest.has_calibration() && manifest.calibration_unvalidated.is_none();
            a.threshold = a.max_error.as_deref().and_then(|k| manifest.certified(k, a.base));
        }
        Ext::Tool(t) => t.threshold = t.max_error.as_deref().and_then(|k| manifest.certified(k, QType::Choice)),
        _ => {}
    }
}
