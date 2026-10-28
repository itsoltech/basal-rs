//! Prompt, calibration and adaptive-order options shared by CLI loading and per-model YAML.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use basal_core::calibration::{CalibrationSource, ServedIdentity};
use basal_core::engine::AutoOrderSettings;
use basal_core::manifest::ModelLoadOptions;
use basal_core::overflow::{ContextOverflow, OverflowSettings};
use basal_core::prompt::PromptLayout;
use basal_core::ModelManifest;
use clap::Args;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Args, Clone, Default, Deserialize)]
pub(crate) struct ModelOptions {
    /// Opt in to trained exits: off (head-capable full depth) or a policy from thresholds.json
    #[arg(long)]
    #[serde(default)]
    pub early_exit: Option<String>,
    /// Local directory with published heads.pt and thresholds.json (alone defaults to off)
    #[arg(long)]
    #[serde(default)]
    pub exit_heads: Option<PathBuf>,
    /// Native second-checkpoint reasoning: off (default) or auto; only the published 2.0 4.5B pair
    #[arg(long, value_parser = ["off", "auto"])]
    #[serde(default)]
    pub think: Option<String>,
    /// Most generated thinking tokens (default 256)
    #[arg(long)]
    #[serde(default)]
    pub think_tokens: Option<usize>,
    /// Thinking context cap, also bounded by the checkpoint context (default 8192)
    #[arg(long)]
    #[serde(default)]
    pub think_max_len: Option<usize>,
    /// Independent canonical/order1 calibration: PATH, off, or automatic root CALIBRATION.think.json
    #[arg(long)]
    #[serde(default)]
    pub think_calibration: Option<PathBuf>,
    /// Prompt layout (otherwise the model declaration, default canonical)
    #[arg(long, value_parser = ["canonical", "qfirst", "state_twice"])]
    #[serde(default)]
    pub layout: Option<String>,
    /// Explicit calibration file; otherwise select the model's layout/order calibration automatically
    #[arg(long, conflicts_with = "no_calibration")]
    #[serde(default)]
    pub calibration: Option<PathBuf>,
    /// Disable calibration, including automatic calibration files
    #[arg(long, conflicts_with = "calibration")]
    #[serde(default)]
    pub no_calibration: bool,
    /// Refuse explicit calibration fitted on another model or known HF revision (otherwise warn)
    #[arg(long)]
    #[serde(default)]
    pub strict_calibration: bool,
    /// Adaptive orders: second pass when top probability is below this threshold (default 0.6)
    #[arg(long)]
    #[serde(default)]
    pub auto_tau: Option<f64>,
    /// Adaptive noul orders: inclusive uncertainty band (default 0.2 0.8)
    #[arg(long, num_args = 2, value_names = ["LO", "HI"])]
    #[serde(default)]
    pub auto_band: Option<Vec<f64>>,
    /// Long text context: refuse (default), or drop the state's middle before inference
    #[arg(long, value_parser = ["refuse", "truncate"])]
    #[serde(default)]
    pub context_overflow: Option<String>,
    /// Fraction of retained state tokens taken from its start, finite in `[0,1]` (default 1/3)
    #[arg(long)]
    #[serde(default)]
    pub truncate_head: Option<f64>,
}

impl ModelOptions {
    pub(crate) fn is_set(&self) -> bool {
        self.early_exit.is_some()
            || self.exit_heads.is_some()
            || self.think.is_some()
            || self.think_tokens.is_some()
            || self.think_max_len.is_some()
            || self.think_calibration.is_some()
            || self.layout.is_some()
            || self.calibration.is_some()
            || self.no_calibration
            || self.strict_calibration
            || self.auto_tau.is_some()
            || self.auto_band.is_some()
            || self.context_overflow.is_some()
            || self.truncate_head.is_some()
    }
    /// Explicit shared CLI values override per-model YAML; one calibration source replaces the other.
    pub(crate) fn apply_cli(&mut self, cli: &Self) {
        if cli.early_exit.is_some() {
            self.early_exit.clone_from(&cli.early_exit);
        }
        if cli.exit_heads.is_some() {
            self.exit_heads.clone_from(&cli.exit_heads);
        }
        if cli.think.is_some() {
            self.think.clone_from(&cli.think);
        }
        if cli.think_tokens.is_some() {
            self.think_tokens = cli.think_tokens;
        }
        if cli.think_max_len.is_some() {
            self.think_max_len = cli.think_max_len;
        }
        if cli.think_calibration.is_some() {
            self.think_calibration.clone_from(&cli.think_calibration);
        }
        if cli.layout.is_some() {
            self.layout.clone_from(&cli.layout);
        }
        if cli.calibration.is_some() || cli.no_calibration {
            self.calibration.clone_from(&cli.calibration);
            self.no_calibration = cli.no_calibration;
        }
        self.strict_calibration |= cli.strict_calibration;
        if cli.auto_tau.is_some() {
            self.auto_tau = cli.auto_tau;
        }
        if cli.auto_band.is_some() {
            self.auto_band.clone_from(&cli.auto_band);
        }
        if cli.context_overflow.is_some() {
            self.context_overflow.clone_from(&cli.context_overflow);
        }
        if cli.truncate_head.is_some() {
            self.truncate_head = cli.truncate_head;
        }
    }

    pub(crate) fn overflow(&self) -> Result<OverflowSettings> {
        let policy = match self.context_overflow.as_deref().unwrap_or("refuse") {
            "refuse" => ContextOverflow::Refuse,
            "truncate" => ContextOverflow::Truncate,
            other => anyhow::bail!("unknown context_overflow {other:?} (refuse, truncate)"),
        };
        Ok(OverflowSettings::new(policy, self.truncate_head.unwrap_or(1.0 / 3.0))?)
    }

    pub(crate) fn auto_orders(&self) -> Result<AutoOrderSettings> {
        let band = match self.auto_band.as_deref() {
            None => [0.2, 0.8],
            Some([lo, hi]) => [*lo, *hi],
            Some(_) => anyhow::bail!("auto_band requires exactly two numbers"),
        };
        Ok(AutoOrderSettings::new(self.auto_tau.unwrap_or(0.6), band)?)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            !self.exits_enabled() || !self.thinking_enabled()?,
            "trained exits cannot be combined with the distinct thinking checkpoint"
        );
        self.thinking_enabled()?;
        self.think_settings()?;
        ensure!(!(self.no_calibration && self.calibration.is_some()), "calibration conflicts with no_calibration");
        if let Some(layout) = &self.layout {
            PromptLayout::parse(layout)?;
        }
        self.auto_orders()?;
        self.overflow()?;
        Ok(())
    }

    pub(crate) fn thinking_enabled(&self) -> Result<bool> {
        match self.think.as_deref().unwrap_or("off") {
            "off" => Ok(false),
            "auto" => Ok(true),
            other => anyhow::bail!("unknown think {other:?} (off, auto)"),
        }
    }

    pub(crate) fn exits_enabled(&self) -> bool {
        self.early_exit.is_some() || self.exit_heads.is_some()
    }

    pub(crate) fn load_exits(&self, manifest: &ModelManifest) -> Result<Option<basal_core::early_exit::ExitArtifacts>> {
        if !self.exits_enabled() {
            return Ok(None);
        }
        Ok(Some(basal_core::early_exit::ExitArtifacts::load(
            manifest,
            self.exit_heads.as_deref(),
            self.early_exit.as_deref().unwrap_or("off"),
        )?))
    }

    fn think_settings(&self) -> Result<basal_core::thinking::ThinkSettings> {
        basal_core::thinking::ThinkSettings::new(self.think_tokens.unwrap_or(256), self.think_max_len.unwrap_or(8192))
    }

    /// Validate the separately published checkpoint, tokenizer, calibration and content identities before GPU or
    /// GEMM activity. No fallback to base weights, adapters or an untrained smaller model.
    pub(crate) fn load_thinking(&self, base: &ModelManifest) -> Result<Option<LoadedThinking>> {
        if !self.thinking_enabled()? {
            return Ok(None);
        }
        use basal_core::prompt::{ChatStyle, PromptFormat};
        use basal_core::thinking::{THINK_REPO, THINK_REVISION};
        ensure!(
            base.name == "basal-2.0-4.5B"
                && base.chat_style == ChatStyle::Llama
                && base.prompt_format == PromptFormat::Two,
            "native thinking supports only {THINK_REPO}@{THINK_REVISION} with its separate think/ checkpoint"
        );
        let dir = base.dir.join("think");
        let automatic = base.dir.join("CALIBRATION.think.json");
        let source = match self.think_calibration.as_deref() {
            Some(p) if p == Path::new("off") => CalibrationSource::Off,
            Some(p) => CalibrationSource::File(p),
            None if automatic.is_file() => CalibrationSource::File(&automatic),
            None => CalibrationSource::Off,
        };
        let calibration_label = match source {
            CalibrationSource::File(p) => {
                p.file_name().context("thinking calibration filename")?.to_string_lossy().into_owned()
            }
            _ if self.think_calibration.is_some() => "off".into(),
            _ => {
                crate::warn!("thinking: root CALIBRATION.think.json missing; using T=1");
                "missing".into()
            }
        };
        let manifest = ModelManifest::load_with_options(
            &dir,
            ModelLoadOptions {
                orders: Some("1"),
                layout: Some(PromptLayout::Canonical),
                calibration: source,
                strict_calibration: false,
                served_identity: None,
            },
        )?;
        ensure!(manifest.calibration_unvalidated.is_none(), "thinking calibration must fit canonical/order1");
        let settings = self.think_settings()?;
        basal_core::thinking::ThinkSettings::new(
            settings.tokens,
            settings.max_len.min(manifest.config.max_positions()),
        )?;
        // Config and tokenizer identity is checked before the expensive weight digests and before any load.
        let mut sources = json!({"repo":THINK_REPO,"revision":THINK_REVISION,"checkpoint":"think/","files":{}});
        for name in ["config.json", "basal.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            let base_bytes = std::fs::read(base.dir.join(name)).with_context(|| format!("reading base {name}"))?;
            let think_bytes = std::fs::read(dir.join(name)).with_context(|| format!("reading think {name}"))?;
            if name == "tokenizer.json" {
                ensure!(
                    content_hash(&think_bytes) == "20f059c5ba108cc6e4d59df55be30e73c30b86ff7aab314362bfa4eff1e21657",
                    "thinking requires the published tokenizer content identity"
                );
            }
            if matches!(name, "tokenizer.json" | "tokenizer_config.json" | "chat_template.jinja") {
                ensure!(base_bytes == think_bytes, "base/think {name} differs from the published compatible pair");
            }
            sources["files"][format!("base/{name}")] = content_hash(&base_bytes).into();
            sources["files"][format!("think/{name}")] = content_hash(&think_bytes).into();
        }
        let (basal_core::architecture::Architecture::Llama(b), basal_core::architecture::Architecture::Llama(t)) =
            (&base.config, &manifest.config)
        else {
            anyhow::bail!("thinking needs two Llama checkpoints");
        };
        ensure!(
            b.num_layers == t.num_layers
                && b.hidden == t.hidden
                && b.vocab == t.vocab
                && b.heads == t.heads
                && b.kv_heads == t.kv_heads
                && b.head_dim == t.head_dim
                && b.intermediate == t.intermediate
                && b.bias == t.bias
                && b.rms_eps == t.rms_eps
                && b.rope_theta == t.rope_theta
                && b.max_positions == t.max_positions,
            "base/think architecture mismatch"
        );
        ensure!(
            t.num_layers == 60
                && t.hidden == 2048
                && t.intermediate == 11008
                && t.heads == 16
                && t.kv_heads == 2
                && t.head_dim == 128
                && t.vocab == 32000
                && t.rms_eps == 1e-6
                && t.rope_theta == 1_000_000.0
                && t.max_positions == 32768
                && t.bias,
            "thinking requires the published 4.5B architecture"
        );
        let tokenizer = basal_core::tokenize::BasalTokenizer::load(&dir.join("tokenizer.json"))?;
        ensure!(tokenizer.vocab_size() == t.vocab, "thinking vocabulary size differs from config");
        let raw: Value = serde_json::from_slice(&std::fs::read(dir.join("tokenizer.json"))?)?;
        let decoder = json!({"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]});
        ensure!(
            raw.get("decoder") == Some(&decoder),
            "thinking decoded-byte bound requires the published Llama decoder"
        );
        let generation_bytes =
            std::fs::read(dir.join("generation_config.json")).context("thinking generation_config.json")?;
        let generation: Value = serde_json::from_slice(&generation_bytes)?;
        let eos: Vec<u32> = generation["eos_token_id"]
            .as_array()
            .context("thinking EOS must be a list")?
            .iter()
            .map(|v| v.as_u64().and_then(|x| u32::try_from(x).ok()).context("invalid thinking EOS ID"))
            .collect::<Result<_>>()?;
        ensure!(!eos.is_empty() && eos.iter().all(|v| (*v as usize) < t.vocab), "invalid thinking EOS vocabulary");
        ensure!(
            eos.len() == 2 && eos.contains(&4) && eos.contains(&2),
            "thinking requires the published effective EOS set [4, 2]"
        );
        sources["files"]["think/generation_config.json"] = content_hash(&generation_bytes).into();
        // These digests bind the supported full checkpoint pair even when supplied through local directories.
        // Shards/alternate checkpoints require a separately qualified identity, rather than matching shape alone.
        for (checkpoint, expected) in [
            (base.dir.as_path(), "481d8ea4972379a611f46784196e407bee239a37c3981c6e29ff1aaaeb0f19c2"),
            (dir.as_path(), "1e295ab2bcc6e812bc9e22f710cd0a6fc4d91371eefffcb271d77cb6ad97c8a5"),
        ] {
            let hash = file_hash(&checkpoint.join("model.safetensors"))?;
            ensure!(hash == expected, "native thinking requires the exact published base/think weight identities");
            let key = if checkpoint == base.dir { "base/model.safetensors" } else { "think/model.safetensors" };
            sources["files"][key] = hash.into();
        }
        Ok(Some(LoadedThinking { manifest, settings, eos, calibration_label, sources }))
    }

    /// Identity follows upstream: HF snapshot repo/commit, otherwise the actual configured source and no revision.
    /// Never infer identity from the routed name or basal.json. Runs after model-file resolution, before GEMM
    /// selection and GPU weight loading.
    pub(crate) fn load_manifest(&self, dir: &Path, orders: Option<&str>, source: &str) -> Result<ModelManifest> {
        self.validate()?;
        let snapshot_repo = dir
            .parent()
            .filter(|p| p.file_name().is_some_and(|n| n == "snapshots"))
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("models--"))
            .map(|n| n.replace("--", "/"));
        let revision = snapshot_repo.as_ref().and_then(|_| dir.file_name()).and_then(|n| n.to_str());
        let manifest = ModelManifest::load_with_options(
            dir,
            ModelLoadOptions {
                orders,
                layout: self.layout.as_deref().map(PromptLayout::parse).transpose()?,
                calibration: if self.no_calibration {
                    CalibrationSource::Off
                } else if let Some(path) = &self.calibration {
                    CalibrationSource::File(path)
                } else {
                    CalibrationSource::Automatic
                },
                strict_calibration: self.strict_calibration,
                served_identity: Some(ServedIdentity { model: snapshot_repo.as_deref().unwrap_or(source), revision }),
            },
        )
        .context("loading model prompt and calibration settings")?;
        // Local log: unlike the metadata, it may name the fitted and served identities.
        let provenance = &manifest.calibration_provenance;
        if let Some(warning) = &manifest.calibration_unvalidated {
            crate::warn!("{}: {warning}", manifest.name);
        }
        if let Some(identity) = &provenance.identity_warning {
            crate::warn!("{}: explicit calibration {identity}; calibration retained", manifest.name);
        }
        if !provenance.explicit && !provenance.off && provenance.content_sha256.is_none() {
            crate::warn!("{}: no calibration file; probabilities are uncalibrated", manifest.name);
        }
        Ok(manifest)
    }
}

#[derive(Clone)]
pub(crate) struct LoadedThinking {
    pub manifest: ModelManifest,
    pub settings: basal_core::thinking::ThinkSettings,
    pub eos: Vec<u32>,
    pub calibration_label: String,
    pub sources: Value,
}

fn content_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn file_hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// Effective prompt, calibration and adaptive-order settings that change results (batch fingerprints, benchmark
/// reports): the calibration by exact content hash, never by path or identity. Absent hash without `off`: no file.
pub(crate) fn record_effective_settings(
    target: &mut Value,
    manifest: &ModelManifest,
    auto: AutoOrderSettings,
    overflow: OverflowSettings,
) {
    let calibration = &manifest.calibration_provenance;
    target["orders"] = manifest.orders.metadata();
    target["layout"] = manifest.layout.as_str().into();
    target["calibration"] =
        json!({"content_sha256": calibration.content_sha256, "off": calibration.off, "strict": calibration.strict});
    target["auto_orders"] = json!(auto);
    target["context_overflow"] = overflow.policy().as_str().into();
    target["truncate_head"] = overflow.head_share().into();
}

/// API metadata warning without local source identities or the calibration's full path.
pub(crate) fn calibration_warning(manifest: &ModelManifest) -> Option<String> {
    let mut warnings = Vec::new();
    if let Some(warning) = &manifest.calibration_unvalidated {
        warnings.push(warning.as_str());
    }
    if manifest.calibration_provenance.identity_warning.is_some() {
        warnings.push("explicit calibration was fitted on another model or revision; calibration retained");
    }
    (!warnings.is_empty()).then(|| warnings.join("; "))
}
