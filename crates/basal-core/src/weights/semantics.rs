//! Captured companion semantics. No GGUF payload or runtime admission lives here.

use std::collections::TryReserveError;
use std::fmt;
use std::path::Path;

use anyhow::{bail, ensure, Context};
use serde_json::Value;

use super::{read_path, CompanionRole, GgufCompanionSnapshot, Sha256Digest, WeightSourceError};
use crate::architecture::Architecture;
use crate::calibration::{CalibrationIdentityError, CalibrationSource, ServedIdentity};
use crate::manifest::{LlamaConfig, ManifestProtocol, ModelLoadOptions, ModelManifest};
use crate::prompt::{OptionOrders, PromptFormat, PromptLayout};
use crate::tokenize::{BasalTokenizer, TokenizerInputError};

const MAX_CALIBRATION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CALIBRATION_TOTAL: u64 = 128 * 1024 * 1024;
const MAX_COMBINED_BYTES: u64 = 256 * 1024 * 1024;

/// The semantic stage that failed; detailed causes remain untrusted diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticStage {
    Config,
    Protocol,
    Template,
    TokenizerConfig,
    CalibrationSelection,
    CalibrationManifest,
    Tokenizer,
}

/// New-source errors preserve their original causes without printing raw inputs.
#[derive(thiserror::Error)]
pub enum GgufSemanticError {
    #[error("GGUF companion capture failed")]
    Capture {
        #[source]
        source: WeightSourceError,
    },
    #[error("G1 requires config.json; GGUF-only companion sets without config require G3")]
    MissingConfigG3 {
        #[source]
        source: WeightSourceError,
    },
    #[error("invalid UTF-8 in {role:?}")]
    Utf8 {
        role: CompanionRole,
        #[source]
        source: std::str::Utf8Error,
    },
    #[error("GGUF companion semantics failed at {stage:?}")]
    Semantic {
        stage: SemanticStage,
        #[source]
        source: anyhow::Error,
    },
    #[error("GGUF tokenizer backend failed at {stage:?}")]
    TokenizerBackend {
        stage: SemanticStage,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("unsupported G1 config field {field}: expected {expected}")]
    UnsupportedConfig { field: String, expected: &'static str },
    #[error("invalid token ID at {location}: {id:?}, configured vocabulary {vocab}")]
    TokenId { location: &'static str, id: Option<u64>, vocab: usize },
}

impl fmt::Debug for GgufSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl GgufSemanticError {
    fn semantic(stage: SemanticStage, source: anyhow::Error) -> Self {
        Self::Semantic { stage, source }
    }

    pub(crate) fn check_id(id: u64, location: &'static str, vocab: usize) -> Result<u32, Self> {
        let invalid = || Self::TokenId { location, id: Some(id), vocab };
        let token = u32::try_from(id).map_err(|_| invalid())?;
        if usize::try_from(token).ok().is_none_or(|id| id >= vocab) {
            return Err(invalid());
        }
        Ok(token)
    }

    pub(crate) fn parse_id(value: &Value, location: &'static str, vocab: usize) -> Result<u32, Self> {
        let id = value.as_u64().ok_or(Self::TokenId { location, id: None, vocab })?;
        Self::check_id(id, location, vocab)
    }
}

/// How the existing calibration selector was invoked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationMode {
    Automatic,
    Explicit,
    Off,
}

/// Logical role of an actually read calibration file; never a filesystem path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationReadRole {
    Candidate,
    OrdersVariant,
    Explicit,
}

/// Logical role of an actually executed `Path::exists` probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationProbeRole {
    LayoutPreference,
    Candidate,
    OrdersVariant,
}

/// One observed probe, which is not authoritative proof of absence.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationProbe {
    role: CalibrationProbeRole,
    exists: bool,
}

impl CalibrationProbe {
    pub fn role(&self) -> CalibrationProbeRole {
        self.role
    }
    pub fn exists(&self) -> bool {
        self.exists
    }
}

/// Only validations actually reached are recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutCheck {
    NotEvaluated,
    NoMismatch,
    Mismatch,
}

/// Absence of fitted orders is distinct from a matching fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrdersCheck {
    NotEvaluated,
    Absent,
    Match,
    Mismatch(OptionOrders),
}

/// The existing selector considers at most one orders replacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrdersVariantDecision {
    NotConsidered,
    ProbeFalse,
    Selected,
}

/// Read-only selection evidence with no operational path or warning text.
pub struct CalibrationDecision {
    mode: CalibrationMode,
    layout: PromptLayout,
    orders: OptionOrders,
    prompt: PromptFormat,
    strict: bool,
    initial_layout: LayoutCheck,
    selected_layout: LayoutCheck,
    initial_orders: OrdersCheck,
    selected_orders: OrdersCheck,
    variant: OrdersVariantDecision,
    probes: [Option<CalibrationProbe>; 3],
}

impl CalibrationDecision {
    pub fn mode(&self) -> CalibrationMode {
        self.mode
    }
    pub fn layout(&self) -> PromptLayout {
        self.layout
    }
    pub fn orders(&self) -> OptionOrders {
        self.orders
    }
    pub fn prompt(&self) -> PromptFormat {
        self.prompt
    }
    pub fn strict(&self) -> bool {
        self.strict
    }
    pub fn initial_layout(&self) -> LayoutCheck {
        self.initial_layout
    }
    pub fn selected_layout(&self) -> LayoutCheck {
        self.selected_layout
    }
    pub fn initial_orders(&self) -> OrdersCheck {
        self.initial_orders
    }
    pub fn selected_orders(&self) -> OrdersCheck {
        self.selected_orders
    }
    pub fn variant(&self) -> OrdersVariantDecision {
        self.variant
    }
    pub fn probes(&self) -> impl Iterator<Item = &CalibrationProbe> {
        self.probes.iter().flatten()
    }
}

/// Exact bytes of a consumed calibration file, including a replaced candidate.
pub struct CapturedCalibrationFile {
    bytes: Vec<u8>,
    sha256: Sha256Digest,
    role: CalibrationReadRole,
}

impl CapturedCalibrationFile {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn sha256(&self) -> Sha256Digest {
        self.sha256
    }
    pub fn role(&self) -> CalibrationReadRole {
        self.role
    }
}

/// Bounded consumed-file set, not an inventory of unread calibration variants.
pub struct CalibrationCapture {
    files: [Option<CapturedCalibrationFile>; 2],
    selected: Option<usize>,
    decision: CalibrationDecision,
    total: u64,
    total_limit: u64,
}

impl CalibrationCapture {
    fn new(
        snapshot: &GgufCompanionSnapshot,
        protocol: &ManifestProtocol,
        options: ModelLoadOptions<'_>,
    ) -> Result<Self, WeightSourceError> {
        let overflow = || WeightSourceError::ResourceLimit {
            role: "calibration",
            limit: MAX_COMBINED_BYTES,
            reason: "captured length sum overflows",
        };
        let mut total = 0_u64;
        for role in CompanionRole::ALL {
            let length = u64::try_from(snapshot.bytes(role).len()).map_err(|_| overflow())?;
            total = total.checked_add(length).ok_or_else(overflow)?;
        }
        let total_limit = total.checked_add(MAX_CALIBRATION_TOTAL).ok_or_else(overflow)?.min(MAX_COMBINED_BYTES);
        let mode = match options.calibration {
            CalibrationSource::Automatic => CalibrationMode::Automatic,
            CalibrationSource::File(_) => CalibrationMode::Explicit,
            CalibrationSource::Off => CalibrationMode::Off,
        };
        Ok(Self {
            files: [None, None],
            selected: None,
            total,
            total_limit,
            decision: CalibrationDecision {
                mode,
                layout: protocol.layout(),
                orders: protocol.orders(),
                prompt: protocol.prompt_format(),
                strict: options.strict_calibration,
                initial_layout: LayoutCheck::NotEvaluated,
                selected_layout: LayoutCheck::NotEvaluated,
                initial_orders: OrdersCheck::NotEvaluated,
                selected_orders: OrdersCheck::NotEvaluated,
                variant: OrdersVariantDecision::NotConsidered,
                probes: [None; 3],
            },
        })
    }

    pub fn files(&self) -> impl Iterator<Item = &CapturedCalibrationFile> {
        self.files.iter().flatten()
    }
    pub fn selected(&self) -> Option<&CapturedCalibrationFile> {
        self.selected.and_then(|slot| self.files[slot].as_ref())
    }
    pub fn decision(&self) -> &CalibrationDecision {
        &self.decision
    }

    pub(crate) fn record_probe(&mut self, role: CalibrationProbeRole, exists: bool) {
        self.decision.probes[role as usize] = Some(CalibrationProbe { role, exists });
    }

    pub(crate) fn record_layout(&mut self, initial: bool, mismatch: bool) {
        let check = if mismatch { LayoutCheck::Mismatch } else { LayoutCheck::NoMismatch };
        if initial {
            self.decision.initial_layout = check;
        }
        self.decision.selected_layout = check;
    }

    pub(crate) fn record_orders(&mut self, initial: bool, fitted: Option<OptionOrders>, served: OptionOrders) {
        let check = match fitted {
            None => OrdersCheck::Absent,
            Some(fitted) if fitted == served => OrdersCheck::Match,
            Some(fitted) => OrdersCheck::Mismatch(fitted),
        };
        if initial {
            self.decision.initial_orders = check;
        } else {
            self.decision.selected_orders = check;
        }
    }

    pub(crate) fn record_variant(&mut self, decision: OrdersVariantDecision) {
        self.decision.variant = decision;
    }

    pub(crate) fn read(&mut self, path: &Path, role: CalibrationReadRole) -> anyhow::Result<(Value, String)> {
        let slot = self.files.iter().position(Option::is_none).ok_or(WeightSourceError::ResourceLimit {
            role: "calibration",
            limit: 2,
            reason: "calibration read count exceeds allowance",
        })?;
        let bytes = read_path(path, "calibration", MAX_CALIBRATION_BYTES, &mut self.total, self.total_limit)
            .with_context(|| format!("reading {}", path.display()))?;
        let data = serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
        let sha256 = Sha256Digest::of(&bytes);
        self.files[slot] = Some(CapturedCalibrationFile { bytes, sha256, role });
        self.selected = Some(slot);
        Ok((data, sha256.to_string()))
    }
}

/// Explicit files without actual served identity must still be checked by a later runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationIdentityState {
    NotRequired,
    Deferred,
    Checked,
}

/// Result of checking the retained calibration against a supplied logical identity.
/// Missing fitted fields or an unknown revision can skip comparisons; this is not a match proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufCalibrationBindingOutcome {
    NotRequired,
    CheckedNoWarning,
    CheckedWarning,
}

/// Exact logical identity and check result retained by the actual semantic owner.
/// The warning, if any, belongs to that owner's manifest provenance, not this binding.
pub struct GgufCalibrationBinding {
    model: String,
    revision: Option<String>,
    outcome: GgufCalibrationBindingOutcome,
}

impl GgufCalibrationBinding {
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }
    pub fn outcome(&self) -> GgufCalibrationBindingOutcome {
        self.outcome
    }
}

/// One-shot binding refusals. Top-level text omits even externally supplied field/reason text.
/// Allocation and calibration causes remain available through `source()` and are untrusted.
#[derive(thiserror::Error)]
pub enum GgufIdentityBindingError {
    #[error("GGUF served identity is already bound")]
    AlreadyBound,
    #[error("invalid GGUF served identity field")]
    InvalidField { field: &'static str, reason: &'static str },
    #[error("GGUF served identity reservation failed")]
    Allocation {
        #[source]
        source: TryReserveError,
    },
    #[error("GGUF calibration identity check failed")]
    Calibration {
        #[source]
        source: CalibrationIdentityError,
    },
}

impl fmt::Debug for GgufIdentityBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

fn validate_identity_field(value: &str, field: &'static str) -> Result<(), GgufIdentityBindingError> {
    let reason = if value.is_empty() || value.len() > 4096 {
        Some("expected 1..=4096 UTF-8 bytes")
    } else if value.bytes().any(|byte| byte.is_ascii_control()) {
        Some("ASCII control byte is forbidden")
    } else if value.starts_with('/') || value.contains(['\\', ':']) {
        Some("expected a logical identity without a leading slash, backslash or colon")
    } else {
        None
    };
    match reason {
        Some(reason) => Err(GgufIdentityBindingError::InvalidField { field, reason }),
        None => Ok(()),
    }
}

fn copy_identity_field(value: &str) -> Result<String, GgufIdentityBindingError> {
    let mut owned = String::new();
    owned.try_reserve_exact(value.len()).map_err(|source| GgufIdentityBindingError::Allocation { source })?;
    owned.push_str(value);
    Ok(owned)
}

/// Owned, validated companions plus the actual shared manifest and tokenizer.
///
/// Requires trusted local inputs. Byte quotas do not bound parser amplification,
/// provide an immutable source lease or make GGUF device allocation ready.
pub struct GgufCompanionSemantics {
    snapshot: GgufCompanionSnapshot,
    manifest: ModelManifest,
    tokenizer: BasalTokenizer,
    calibration: CalibrationCapture,
    identity_state: CalibrationIdentityState,
    binding: Option<GgufCalibrationBinding>,
}

impl GgufCompanionSemantics {
    /// Capture once, then parse those bytes. Only selected calibration opens extra files.
    /// No GGUF file, device, model execution or companion path reload is involved.
    pub fn read(directory: &Path, options: ModelLoadOptions<'_>) -> Result<Self, GgufSemanticError> {
        let snapshot = GgufCompanionSnapshot::read(directory).map_err(|source| match &source {
            WeightSourceError::Io { role: "config.json", operation: "open", source: io }
                if io.kind() == std::io::ErrorKind::NotFound =>
            {
                GgufSemanticError::MissingConfigG3 { source }
            }
            _ => GgufSemanticError::Capture { source },
        })?;
        Self::from_snapshot(snapshot, directory, options)
    }

    fn from_snapshot(
        snapshot: GgufCompanionSnapshot,
        directory: &Path,
        options: ModelLoadOptions<'_>,
    ) -> Result<Self, GgufSemanticError> {
        use SemanticStage as Stage;
        let config = json(&snapshot, CompanionRole::Config, Stage::Config)?;
        let architecture = Architecture::parse(&config).map_err(|e| GgufSemanticError::semantic(Stage::Config, e))?;
        let Architecture::Llama(llama) = &architecture else {
            return unsupported("model_type", "ordinary Llama");
        };
        validate_config(&config, llama)?;
        let vocab = llama.vocab;
        let basal = json(&snapshot, CompanionRole::Basal, Stage::Protocol)?;
        let mut protocol = ManifestProtocol::parse(architecture, &basal, options)
            .map_err(|e| GgufSemanticError::semantic(Stage::Protocol, e))?;
        protocol
            .validate_template(text(&snapshot, CompanionRole::ChatTemplate)?)
            .map_err(|e| GgufSemanticError::semantic(Stage::Template, e))?;
        let tcfg = json(&snapshot, CompanionRole::TokenizerConfig, Stage::TokenizerConfig)?;
        let bos = protocol.parse_bos(&tcfg).map_err(|e| GgufSemanticError::semantic(Stage::TokenizerConfig, e))?;
        let style = protocol.chat_style();
        let mut calibration = CalibrationCapture::new(&snapshot, &protocol, options)
            .map_err(|source| GgufSemanticError::Capture { source })?;
        let loaded = crate::calibration::load_captured(
            directory,
            options,
            protocol.layout(),
            protocol.orders(),
            &mut calibration,
        )
        .map_err(|e| GgufSemanticError::semantic(Stage::CalibrationSelection, e))?;
        let manifest = protocol
            .finish(directory, &basal, bos, loaded)
            .map_err(|e| GgufSemanticError::semantic(Stage::CalibrationManifest, e))?;
        drop(basal);
        let original = json(&snapshot, CompanionRole::Tokenizer, Stage::Tokenizer)?;
        let added = BasalTokenizer::audit_raw_ids(&original, vocab)?;
        audit_bpe_merges(&original).map_err(|e| GgufSemanticError::semantic(Stage::Tokenizer, e))?;
        let tokenizer =
            BasalTokenizer::from_json_value(original, &directory.join("tokenizer.json"), style).map_err(|error| {
                match error {
                    TokenizerInputError::Semantic(source) => GgufSemanticError::semantic(Stage::Tokenizer, source),
                    TokenizerInputError::Backend { source, .. } => {
                        GgufSemanticError::TokenizerBackend { stage: Stage::Tokenizer, source }
                    }
                }
            })?;
        let bos = tokenizer.audit_effective_ids(&added, vocab, &manifest.bos_token)?;
        audit_specials(&config, vocab, bos, "config special ID")?;
        audit_specials(&tcfg, vocab, bos, "tokenizer_config special ID")?;
        let identity_state = match options.calibration {
            CalibrationSource::File(_) if options.served_identity.is_none() => CalibrationIdentityState::Deferred,
            CalibrationSource::File(_) => CalibrationIdentityState::Checked,
            _ => CalibrationIdentityState::NotRequired,
        };
        Ok(Self { snapshot, manifest, tokenizer, calibration, identity_state, binding: None })
    }

    /// Bind once to an exact application-supplied logical model/revision and recheck provenance.
    ///
    /// Each present field must contain 1..=4096 UTF-8 bytes, without ASCII controls,
    /// a leading slash, backslash or colon. Case and internal/trailing slashes are
    /// retained exactly; no normalization or identity inference occurs. Existing
    /// read-time identity checks do not create a binding or bypass this fresh check.
    /// Errors leave the binding, warning and identity state unchanged. Success
    /// preserves temperatures and certificates, including a non-strict warning.
    ///
    /// Input copies request at most 8192 bytes; allocator capacity/overhead and the
    /// unchanged matcher's warning/error allocations are not covered by that limit.
    /// This is neither semantic heap admission nor durable source/publication approval.
    pub fn bind_served_identity(&mut self, served: ServedIdentity<'_>) -> Result<(), GgufIdentityBindingError> {
        if self.binding.is_some() {
            return Err(GgufIdentityBindingError::AlreadyBound);
        }
        validate_identity_field(served.model, "model")?;
        if let Some(revision) = served.revision {
            validate_identity_field(revision, "revision")?;
        }
        let model = copy_identity_field(served.model)?;
        let revision = served.revision.map(copy_identity_field).transpose()?;
        let warning = self
            .manifest
            .calibration_provenance
            .check_identity(served)
            .map_err(|source| GgufIdentityBindingError::Calibration { source })?;
        let (outcome, identity_state) = match self.calibration.decision.mode {
            CalibrationMode::Explicit => (
                if warning.is_some() {
                    GgufCalibrationBindingOutcome::CheckedWarning
                } else {
                    GgufCalibrationBindingOutcome::CheckedNoWarning
                },
                CalibrationIdentityState::Checked,
            ),
            CalibrationMode::Automatic | CalibrationMode::Off => {
                (GgufCalibrationBindingOutcome::NotRequired, CalibrationIdentityState::NotRequired)
            }
        };
        // All fallible work is complete; only the actual provenance owns the warning.
        self.manifest.calibration_provenance.identity_warning = warning;
        self.binding = Some(GgufCalibrationBinding { model, revision, outcome });
        self.identity_state = identity_state;
        Ok(())
    }

    /// Borrow the one-shot binding; all existing read constructors start without one.
    pub fn calibration_binding(&self) -> Option<&GgufCalibrationBinding> {
        self.binding.as_ref()
    }

    pub fn snapshot(&self) -> &GgufCompanionSnapshot {
        &self.snapshot
    }
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
    pub fn tokenizer(&self) -> &BasalTokenizer {
        &self.tokenizer
    }
    pub fn calibration(&self) -> &CalibrationCapture {
        &self.calibration
    }
    pub fn identity_state(&self) -> CalibrationIdentityState {
        self.identity_state
    }
}

/// Guard the pinned tokenizers 0.23.2 BPE builder's unchecked scratch writes and UTF-8 suffix.
/// Only the captured G1 route calls this; the shared constructor and legacy route stay unchanged.
fn audit_bpe_merges(original: &Value) -> anyhow::Result<()> {
    let Some(model) = original.get("model").filter(|m| m.get("type").and_then(Value::as_str) == Some("BPE")) else {
        return Ok(());
    };
    let vocab = model.get("vocab").and_then(Value::as_object).context("BPE vocab must be an object")?;
    let merges = model.get("merges").and_then(Value::as_array).context("BPE merges must be an array")?;
    let prefix_len = match model.get("continuing_subword_prefix") {
        None | Some(Value::Null) => 0,
        Some(Value::String(prefix)) => prefix.len(),
        _ => bail!("BPE continuing_subword_prefix must be a string or null"),
    };
    let capacity = vocab.keys().map(String::len).max().unwrap_or(0);
    for merge in merges {
        // Deserialization accepts a whole array of pairs or legacy strings. Checking their
        // union here is sufficient: mixed forms fail deserialization before the builder runs.
        let (left, right) = match merge {
            Value::Array(pair) if pair.len() == 2 => (
                pair[0].as_str().context("BPE merge component must be a string")?,
                pair[1].as_str().context("BPE merge component must be a string")?,
            ),
            Value::String(line) => {
                if line.starts_with("#version") {
                    continue;
                }
                let (left, right) = line.split_once(' ').context("BPE legacy merge must contain two parts")?;
                ensure!(!right.contains(' '), "BPE legacy merge must contain two parts");
                (left, right)
            }
            _ => bail!("BPE merge must be a string pair or legacy string"),
        };
        if !vocab.contains_key(left) || !vocab.contains_key(right) {
            // The builder returns an error for missing components before touching its buffer.
            continue;
        }
        // The dependency cuts by prefix byte length, without checking the prefix contents.
        let suffix =
            right.get(prefix_len..).context("BPE merge prefix must end at a UTF-8 boundary within the token")?;
        let length = left.len().checked_add(suffix.len()).context("BPE merge byte length overflows")?;
        ensure!(length <= capacity, "BPE merge exceeds the tokenizer builder's scratch capacity");
    }
    Ok(())
}

fn text(snapshot: &GgufCompanionSnapshot, role: CompanionRole) -> Result<&str, GgufSemanticError> {
    std::str::from_utf8(snapshot.bytes(role)).map_err(|source| GgufSemanticError::Utf8 { role, source })
}

fn json(
    snapshot: &GgufCompanionSnapshot,
    role: CompanionRole,
    stage: SemanticStage,
) -> Result<Value, GgufSemanticError> {
    serde_json::from_str(text(snapshot, role)?)
        .with_context(|| format!("parsing {}", role.filename()))
        .map_err(|e| GgufSemanticError::semantic(stage, e))
}

fn unsupported<T>(field: &str, expected: &'static str) -> Result<T, GgufSemanticError> {
    Err(GgufSemanticError::UnsupportedConfig { field: field.to_owned(), expected })
}

/// Closed G1 policy, deliberately separate from the permissive legacy parser.
fn validate_config(raw: &Value, config: &LlamaConfig) -> Result<(), GgufSemanticError> {
    let Some(object) = raw.as_object() else {
        return unsupported("config", "object");
    };
    for (field, value) in object {
        let valid = match field.as_str() {
            // These fields have already been validated by the shared architecture parser.
            "model_type"
            | "architectures"
            | "hidden_act"
            | "tie_word_embeddings"
            | "attention_bias"
            | "mlp_bias"
            | "pretraining_tp"
            | "num_hidden_layers"
            | "hidden_size"
            | "intermediate_size"
            | "num_attention_heads"
            | "num_key_value_heads"
            | "head_dim"
            | "vocab_size"
            | "rms_norm_eps"
            | "max_position_embeddings" => continue,
            "rope_parameters" => {
                validate_rope(value)?;
                continue;
            }
            "rope_scaling"
            | "quantization_config"
            | "sliding_window"
            | "layer_types"
            | "per_layer_config"
            | "text_config"
            | "vision_config"
            | "audio_config" => value.is_null(),
            "rope_theta" => finite_number(value).is_some_and(|v| v > 0.0 && v == config.rope_theta),
            "partial_rotary_factor" => finite_number(value) == Some(1.0),
            "is_causal" | "is_decoder" => value.as_bool() == Some(true),
            "is_encoder_decoder" | "add_cross_attention" | "use_bidirectional_attention" => {
                value.as_bool() == Some(false)
            }
            "_attn_implementation" | "attn_implementation" => value.is_null() || value.as_str() == Some("eager"),
            "attention_dropout" => finite_number(value) == Some(0.0),
            // Bounds and BOS equality are checked against the constructed tokenizer below.
            "bos_token_id" | "eos_token_id" | "pad_token_id" => continue,
            "_name_or_path" | "transformers_version" => value.is_string(),
            "torch_dtype" | "dtype" => {
                value.is_null() || matches!(value.as_str(), Some("float16" | "bfloat16" | "float32"))
            }
            "initializer_range" => finite_number(value).is_some_and(|v| (0.0..=1.0).contains(&v)),
            "use_cache" => value.is_boolean(),
            _ => return unsupported(field, "a field in the closed ordinary-Llama G1 policy"),
        };
        if !valid {
            return unsupported(field, "the fixed ordinary-Llama G1 value policy (docs/GGUF.md)");
        }
    }
    Ok(())
}

fn finite_number(value: &Value) -> Option<f64> {
    value.as_f64().filter(|v| v.is_finite())
}

fn validate_rope(value: &Value) -> Result<(), GgufSemanticError> {
    let Some(object) = value.as_object() else {
        return unsupported("rope_parameters", "object");
    };
    for (key, value) in object {
        let valid = match key.as_str() {
            "rope_theta" => finite_number(value).is_some_and(|v| v > 0.0),
            "rope_type" | "type" => value.as_str() == Some("default"),
            "partial_rotary_factor" | "factor" | "attention_factor" => finite_number(value) == Some(1.0),
            _ => return unsupported(&format!("rope_parameters.{key}"), "a field in the closed default-RoPE G1 policy"),
        };
        if !valid {
            return unsupported(
                &format!("rope_parameters.{key}"),
                "positive theta, default type, or exactly-one factor",
            );
        }
    }
    Ok(())
}

fn audit_specials(raw: &Value, vocab: usize, bos: u32, location: &'static str) -> Result<(), GgufSemanticError> {
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
        let Some(value) = raw.get(key).filter(|v| !v.is_null()) else {
            continue;
        };
        match value.as_array() {
            Some(ids) if key == "eos_token_id" => {
                for id in ids {
                    GgufSemanticError::parse_id(id, location, vocab)?;
                }
            }
            _ => {
                let id = GgufSemanticError::parse_id(value, location, vocab)?;
                if key == "bos_token_id" && id != bos {
                    return Err(GgufSemanticError::TokenId {
                        location: "declared BOS disagreement",
                        id: Some(u64::from(id)),
                        vocab,
                    });
                }
            }
        }
    }
    Ok(())
}
