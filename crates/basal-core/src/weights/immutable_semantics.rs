//! Direct immutable companion/calibration consumers. Tokenizer storage is excluded,
//! except the BPE merge scratch buffer on `read_with_bpe_merge_scratch_budget`,
//! on `read_with_bpe_merge_table_budget` that buffer and the BPE merge table backing,
//! and on `read_with_decoder_regex_budget` both plus the decoder regexes' compiled backing.

use std::alloc::{alloc, Layout};
use std::fmt;
use std::path::{Path, PathBuf};

use super::calibration_inventory::{
    CallerCalibrationDirectoryLease, GgufCalibrationInventory, GgufCalibrationInventoryError, InventoryPhase,
};
use super::provider_construction::{
    build_captured_tokenizer_with_decoder_regex, build_captured_tokenizer_with_merge_scratch,
    build_captured_tokenizer_with_merge_table, GgufDecoderRegexReadError, GgufMergeScratchReadError,
    GgufMergeTableReadError,
};
use super::raw_source::RawCharge;
use super::semantic_json::{
    GgufSemanticMembers, GgufSemanticObject, NodeId, SemanticConstructionError, SemanticDocument, SemanticPlan,
    TextSpan, ValueRef,
};
use super::semantics::{build_captured_tokenizer, CalibrationBindingFacts};
use super::{
    CalibrationCapture, CalibrationIdentityState, CalibrationMode, CalibrationProbeRole, CalibrationReadRole,
    CompanionRole, GgufCalibrationBindingOutcome, GgufCompanionSnapshot, GgufDeclarationSourceBudget,
    GgufRawSourceBudget, GgufSemanticError, JsonEnvelopeError, JsonReadMode, OrdersVariantDecision, SemanticStage,
    Sha256Digest, SourceBudget, WeightSourceError,
};
use crate::calibration::{calibration_identity_mismatch, CalibrationIdentityError, CalibrationSource, ServedIdentity};
use crate::manifest::{LlamaConfig, ModelLoadOptions, TemperatureBucket};
use crate::prompt::{ChatStyle, Lang, OptionOrders, PromptFormat, PromptLayout, CHAT_TEMPLATE, LETTERS, PREFILL};
use crate::request::QType;
use crate::tokenize::BasalTokenizer;

/// Actual JSON role, separate from logical identity byte construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmutableJsonRole {
    Config,
    Basal,
    TokenizerConfig,
    Calibration(CalibrationReadRole),
}

/// Typed immutable-route errors. Top-level text omits untrusted source text.
/// I/O, inventory, tokenizer and identity diagnostic causes are unadmitted.
#[derive(thiserror::Error)]
pub enum GgufImmutableSemanticError {
    #[error("GGUF immutable companion capture failed")]
    Capture {
        #[source]
        source: WeightSourceError,
    },
    #[error("GGUF immutable calibration inventory failed")]
    Inventory {
        #[source]
        source: GgufCalibrationInventoryError,
    },
    #[error("GGUF immutable JSON envelope failed for {role:?}")]
    Envelope {
        role: ImmutableJsonRole,
        #[source]
        source: JsonEnvelopeError,
    },
    #[error("GGUF immutable JSON construction failed for {role:?}")]
    Construction {
        role: ImmutableJsonRole,
        #[source]
        source: SemanticConstructionError,
    },
    #[error("GGUF immutable semantic rule failed at {stage:?}, byte {offset}")]
    Rule { stage: SemanticStage, offset: usize, field: &'static str, reason: &'static str },
    #[error("GGUF immutable tokenizer construction failed")]
    Tokenizer {
        #[source]
        source: GgufSemanticError,
    },
    #[error("GGUF immutable calibration identity check failed")]
    Identity {
        #[source]
        source: CalibrationIdentityError,
    },
    #[error("GGUF immutable served identity construction failed")]
    IdentityConstruction {
        #[source]
        source: SemanticConstructionError,
    },
    #[error("GGUF immutable served identity is already bound")]
    AlreadyBound,
    #[error("invalid GGUF immutable served identity field")]
    InvalidIdentityField { field: &'static str, reason: &'static str },
}
impl fmt::Debug for GgufImmutableSemanticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}
use GgufImmutableSemanticError as Error;

fn rule(stage: SemanticStage, value: Option<ValueRef<'_>>, field: &'static str, reason: &'static str) -> Error {
    Error::Rule { stage, offset: value.map_or(0, ValueRef::offset), field, reason }
}
fn require(
    condition: bool,
    stage: SemanticStage,
    value: Option<ValueRef<'_>>,
    field: &'static str,
    reason: &'static str,
) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(rule(stage, value, field, reason))
    }
}
fn get<'a>(value: Option<ValueRef<'a>>, key: &str) -> Option<ValueRef<'a>> {
    value?.get(key)
}
fn string(value: Option<ValueRef<'_>>) -> Option<&str> {
    value?.as_str()
}
fn number(value: Option<ValueRef<'_>>) -> Option<f64> {
    value?.as_f64()
}

#[derive(Clone, Copy)]
enum DocSlot {
    Basal,
    TokenizerConfig,
    Calibration(usize),
}
#[derive(Clone, Copy)]
struct TextRef {
    doc: DocSlot,
    span: TextSpan,
}
fn text_ref(value: Option<ValueRef<'_>>, doc: DocSlot) -> Option<TextRef> {
    Some(TextRef { doc, span: value?.text_span()? })
}
#[derive(Clone, Copy)]
enum SelectedCalibration {
    Empty,
    Slot(usize),
}

struct ProtocolState {
    config: LlamaConfig,
    chat_style: ChatStyle,
    prompt_format: PromptFormat,
    layout: PromptLayout,
    orders: OptionOrders,
}
struct ImmutableManifestState {
    config: LlamaConfig,
    chat_style: ChatStyle,
    prompt_format: PromptFormat,
    layout: PromptLayout,
    orders: OptionOrders,
    calibration_prompt: PromptFormat,
    name: TextRef,
    version: Option<TextRef>,
    bos: TextRef,
    temperatures: [Option<f64>; 6],
    bucket_objects: [Option<NodeId>; 6],
    thresholds: Option<NodeId>,
    task_thresholds: Option<NodeId>,
    multi: Option<NodeId>,
    selected: SelectedCalibration,
}
struct ImmutableProvenance {
    path: Option<PathBuf>,
    digest: Option<Sha256Digest>,
    explicit: bool,
    off: bool,
    strict: bool,
    fitted_model: Option<TextRef>,
    fitted_revision: Option<TextRef>,
    // Diagnostic only; no selected semantic backing is copied into this field.
    identity_warning: Option<String>,
}
struct LoadedImmutableCalibration {
    documents: [Option<SemanticDocument>; 2],
    selected: SelectedCalibration,
    provenance: ImmutableProvenance,
    unvalidated: Option<String>,
}
impl LoadedImmutableCalibration {
    fn root(&self) -> Option<ValueRef<'_>> {
        match self.selected {
            SelectedCalibration::Empty => None,
            SelectedCalibration::Slot(slot) => self.documents.get(slot)?.as_ref().map(SemanticDocument::root),
        }
    }
}
struct AdmittedIdentity {
    model: Vec<u8>,
    revision: Option<Vec<u8>>,
    outcome: GgufCalibrationBindingOutcome,
    // Last: exact retained identity bytes deallocate before refund.
    _charge: RawCharge,
}

/// Actual immutable owner of directly consumed companion and calibration arenas.
///
/// Only named requested layouts share the declaration budget's existing ledger.
/// Tokenizer Value/provider/native/cache (except opt-in BPE merge scratch, held
/// only during the BPE build, and the opt-in BPE merge table and decoder-regex
/// compiled backing, held while the tokenizer or a clone holds it), diagnostics, paths/control/inventory
/// records, GGUF parsing, staging/device and allocator overhead remain unadmitted.
/// No Clone, mutable manifest or owned subtree conversion is provided.
pub struct GgufImmutableCompanionSemantics {
    snapshot: GgufCompanionSnapshot,
    documents: [SemanticDocument; 3],
    calibration_documents: [Option<SemanticDocument>; 2],
    manifest: ImmutableManifestState,
    provenance: ImmutableProvenance,
    unvalidated: Option<String>,
    tokenizer: BasalTokenizer,
    calibration: CalibrationCapture,
    identity_state: CalibrationIdentityState,
    binding: Option<AdmittedIdentity>,
    calibration_inventory: GgufCalibrationInventory,
    budget: GgufRawSourceBudget,
}
impl GgufImmutableCompanionSemantics {
    /// Capture, admit and directly consume immutable semantics on the same ledger.
    /// Requires the caller's trusted-local calibration lease through return.
    /// Retains both actual calibration slots, including a replaced candidate.
    /// This opt-in route is not a loader or a total-memory/RSS/OOM guarantee.
    pub fn read(
        directory: &Path,
        options: ModelLoadOptions<'_>,
        lease: CallerCalibrationDirectoryLease,
        budget: &GgufDeclarationSourceBudget,
    ) -> Result<Self, Error> {
        Self::read_with(directory, options, lease, budget, |snapshot, directory, vocab, style, bos| {
            build_captured_tokenizer(snapshot, directory, vocab, style, bos)
                .map_err(|source| Error::Tokenizer { source })
        })
    }

    /// [`read`](Self::read), but the tokenizer's BPE merge scratch buffer is
    /// admitted on the same ledger before allocation and released when the
    /// BPE build returns. Every other tokenizer heap remains unadmitted.
    pub fn read_with_bpe_merge_scratch_budget(
        directory: &Path,
        options: ModelLoadOptions<'_>,
        lease: CallerCalibrationDirectoryLease,
        budget: &GgufDeclarationSourceBudget,
    ) -> Result<Self, GgufMergeScratchReadError> {
        Self::read_with(directory, options, lease, budget, |snapshot, directory, vocab, style, bos| {
            build_captured_tokenizer_with_merge_scratch(snapshot, directory, vocab, style, bos, budget.raw())
        })
    }

    /// [`read_with_bpe_merge_scratch_budget`](Self::read_with_bpe_merge_scratch_budget),
    /// and every backing block of the built BPE's merge hash table is admitted on
    /// the same ledger before allocation and released after it is freed, which is
    /// when the returned owner's tokenizer and any clone of it are gone. Every
    /// other tokenizer heap, including the table's entire `Arc` allocation
    /// (control block, map header, allocator and hasher), remains unadmitted.
    pub fn read_with_bpe_merge_table_budget(
        directory: &Path,
        options: ModelLoadOptions<'_>,
        lease: CallerCalibrationDirectoryLease,
        budget: &GgufDeclarationSourceBudget,
    ) -> Result<Self, GgufMergeTableReadError> {
        Self::read_with(directory, options, lease, budget, |snapshot, directory, vocab, style, bos| {
            build_captured_tokenizer_with_merge_table(snapshot, directory, vocab, style, bos, budget.raw())
        })
    }

    /// [`read_with_bpe_merge_table_budget`](Self::read_with_bpe_merge_table_budget),
    /// and every retained compiled Oniguruma backing block of each `Replace`
    /// regex in the tokenizer's decoder tree (`regex_t`, program, operands,
    /// string pool, repeat ranges, optimizer string) is admitted on the same
    /// ledger before allocation and released after it is freed, which is when
    /// the returned owner's tokenizer and any clone of it are gone. Regex parse
    /// temporaries, name and callout tables, the process-wide Oniguruma
    /// registries, match-time state and every other tokenizer heap remain
    /// unadmitted.
    pub fn read_with_decoder_regex_budget(
        directory: &Path,
        options: ModelLoadOptions<'_>,
        lease: CallerCalibrationDirectoryLease,
        budget: &GgufDeclarationSourceBudget,
    ) -> Result<Self, GgufDecoderRegexReadError> {
        Self::read_with(directory, options, lease, budget, |snapshot, directory, vocab, style, bos| {
            build_captured_tokenizer_with_decoder_regex(snapshot, directory, vocab, style, bos, budget.raw())
        })
    }

    /// The single read chronology; `construct_tokenizer` runs at the original tokenizer position.
    fn read_with<E: From<Error>>(
        directory: &Path,
        options: ModelLoadOptions<'_>,
        lease: CallerCalibrationDirectoryLease,
        budget: &GgufDeclarationSourceBudget,
        construct_tokenizer: impl FnOnce(
            &GgufCompanionSnapshot,
            &Path,
            usize,
            ChatStyle,
            &str,
        ) -> Result<(BasalTokenizer, u32), E>,
    ) -> Result<Self, E> {
        let (mut inventory, handle) = GgufCalibrationInventory::capture(directory, lease, Some(budget.raw()))
            .map_err(|source| Error::Inventory { source })?;
        let snapshot =
            GgufCompanionSnapshot::read_mode(directory, JsonReadMode::Envelope, SourceBudget::Declaration(budget))
                .map_err(|source| Error::Capture { source })?;
        let config_doc = document(snapshot.bytes(CompanionRole::Config), ImmutableJsonRole::Config, budget)?;
        let config = read_g1_config(config_doc.root())?;
        let vocab = config.vocab;
        let basal_doc = document(snapshot.bytes(CompanionRole::Basal), ImmutableJsonRole::Basal, budget)?;
        let protocol = read_protocol(config, basal_doc.root(), options)?;
        let template = std::str::from_utf8(snapshot.bytes(CompanionRole::ChatTemplate))
            .map_err(|_| rule(SemanticStage::Template, None, "chat_template", "expected UTF-8"))?;
        require(template == CHAT_TEMPLATE, SemanticStage::Template, None, "chat_template", "unsupported template")?;
        let tcfg_doc =
            document(snapshot.bytes(CompanionRole::TokenizerConfig), ImmutableJsonRole::TokenizerConfig, budget)?;
        let bos = text_ref(tcfg_doc.root().get("bos_token"), DocSlot::TokenizerConfig).ok_or_else(|| {
            rule(SemanticStage::TokenizerConfig, Some(tcfg_doc.root()), "bos_token", "expected string")
        })?;
        let mut calibration = CalibrationCapture::new_for_protocol(
            &snapshot,
            protocol.layout,
            protocol.orders,
            protocol.prompt_format,
            options,
            JsonReadMode::Envelope,
            Some(budget.raw()),
        )
        .map_err(|source| Error::Capture { source })?;
        let loaded = load_immutable_calibration(directory, options, &protocol, &mut calibration, budget)?;
        let manifest = finish_manifest_ref(protocol, basal_doc.root(), bos, &loaded)?;
        let (tokenizer, effective_bos) =
            construct_tokenizer(&snapshot, directory, vocab, manifest.chat_style, tcfg_doc.text(bos.span))?;
        audit_specials_ref(config_doc.root(), vocab, effective_bos, "config special ID")?;
        audit_specials_ref(tcfg_doc.root(), vocab, effective_bos, "tokenizer_config special ID")?;
        let identity_state = match options.calibration {
            CalibrationSource::File(_) if options.served_identity.is_none() => CalibrationIdentityState::Deferred,
            CalibrationSource::File(_) => CalibrationIdentityState::Checked,
            _ => CalibrationIdentityState::NotRequired,
        };
        let facts = CalibrationBindingFacts {
            digest: loaded.provenance.digest,
            explicit: loaded.provenance.explicit,
            off: loaded.provenance.off,
            strict: loaded.provenance.strict,
        };
        calibration.bind_inventory_facts(&mut inventory, facts).map_err(|source| Error::Inventory { source })?;
        inventory
            .check_directory(&handle, InventoryPhase::BeforeInstall)
            .map_err(|source| Error::Inventory { source })?;
        Ok(Self {
            snapshot,
            documents: [config_doc, basal_doc, tcfg_doc],
            calibration_documents: loaded.documents,
            manifest,
            provenance: loaded.provenance,
            unvalidated: loaded.unvalidated,
            tokenizer,
            calibration,
            identity_state,
            binding: None,
            calibration_inventory: inventory,
            budget: budget.raw().clone(),
        })
    }
    pub fn snapshot(&self) -> &GgufCompanionSnapshot {
        &self.snapshot
    }
    pub fn manifest(&self) -> GgufImmutableManifest<'_> {
        GgufImmutableManifest { owner: self }
    }
    /// The tokenizer and its caches remain outside semantic admission, except the
    /// BPE merge table backing on `read_with_bpe_merge_table_budget` and, on
    /// `read_with_decoder_regex_budget`, also the decoder regexes' compiled backing.
    pub fn tokenizer(&self) -> &BasalTokenizer {
        &self.tokenizer
    }
    pub fn calibration(&self) -> &CalibrationCapture {
        &self.calibration
    }
    pub fn calibration_inventory(&self) -> &GgufCalibrationInventory {
        &self.calibration_inventory
    }
    pub fn identity_state(&self) -> CalibrationIdentityState {
        self.identity_state
    }
    pub fn calibration_binding(&self) -> Option<GgufImmutableIdentityBinding<'_>> {
        self.binding.as_ref().map(|binding| GgufImmutableIdentityBinding { binding })
    }
    fn text(&self, text: TextRef) -> &str {
        match text.doc {
            DocSlot::Basal => self.documents[1].text(text.span),
            DocSlot::TokenizerConfig => self.documents[2].text(text.span),
            DocSlot::Calibration(slot) => {
                self.calibration_documents.get(slot).and_then(Option::as_ref).map_or("", |doc| doc.text(text.span))
            }
        }
    }
    fn selected_document(&self) -> Option<&SemanticDocument> {
        match self.manifest.selected {
            SelectedCalibration::Empty => None,
            SelectedCalibration::Slot(slot) => self.calibration_documents.get(slot)?.as_ref(),
        }
    }
    fn selected_object(&self, id: Option<NodeId>) -> GgufSemanticObject<'_> {
        id.and_then(|id| self.selected_document()?.value(id).object()).unwrap_or_else(GgufSemanticObject::empty)
    }
    /// Bind once, admitting exact retained identity bytes before either copy.
    /// Consumes the owner: every failure destroys it, including strict mismatch.
    /// Matcher lowercase/warning/error allocations remain explicit diagnostics.
    pub fn bind_served_identity(mut self, served: ServedIdentity<'_>) -> Result<Self, Error> {
        if self.binding.is_some() {
            return Err(Error::AlreadyBound);
        }
        validate_identity_field(served.model, "model")?;
        if let Some(revision) = served.revision {
            validate_identity_field(revision, "revision")?;
        }
        let construct = || -> Result<AdmittedIdentity, SemanticConstructionError> {
            let model_layout = identity_layout(served.model.len())?;
            let revision_layout = identity_layout(served.revision.map_or(0, str::len))?;
            let requested =
                model_layout.size().checked_add(revision_layout.size()).ok_or(SemanticConstructionError::Arithmetic)?;
            let mut charge = self.budget.charge();
            charge.acquire(requested).map_err(|source| SemanticConstructionError::Admission { source })?;
            let model = copy_identity_bytes(served.model, model_layout)?;
            let revision = served.revision.map(|value| copy_identity_bytes(value, revision_layout)).transpose()?;
            Ok(AdmittedIdentity {
                model,
                revision,
                outcome: GgufCalibrationBindingOutcome::NotRequired,
                _charge: charge,
            })
        };
        let mut binding = construct().map_err(|source| Error::IdentityConstruction { source })?;
        let warning = check_identity(
            &self.provenance,
            self.provenance.fitted_model.map(|text| self.text(text)),
            self.provenance.fitted_revision.map(|text| self.text(text)),
            served,
        )?;
        let (outcome, state) = match self.calibration.decision().mode() {
            CalibrationMode::Explicit if warning.is_some() => {
                (GgufCalibrationBindingOutcome::CheckedWarning, CalibrationIdentityState::Checked)
            }
            CalibrationMode::Explicit => {
                (GgufCalibrationBindingOutcome::CheckedNoWarning, CalibrationIdentityState::Checked)
            }
            _ => (GgufCalibrationBindingOutcome::NotRequired, CalibrationIdentityState::NotRequired),
        };
        binding.outcome = outcome;
        // Atomic install after all fallible work. No arena/caller borrows escape.
        self.provenance.identity_warning = warning;
        self.identity_state = state;
        self.binding = Some(binding);
        Ok(self)
    }
}

/// Borrowed one-shot binding, with no owned identity/permit escape.
#[derive(Clone, Copy)]
pub struct GgufImmutableIdentityBinding<'a> {
    binding: &'a AdmittedIdentity,
}
impl<'a> GgufImmutableIdentityBinding<'a> {
    pub fn model(self) -> &'a str {
        identity_text(&self.binding.model)
    }
    pub fn revision(self) -> Option<&'a str> {
        self.binding.revision.as_deref().map(identity_text)
    }
    pub fn outcome(self) -> GgufCalibrationBindingOutcome {
        self.binding.outcome
    }
}
fn identity_text(bytes: &[u8]) -> &str {
    // Private bytes are exact copies of validated &str, never mutated afterwards.
    std::str::from_utf8(bytes).unwrap_or("")
}
fn validate_identity_field(value: &str, field: &'static str) -> Result<(), Error> {
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
        Some(reason) => Err(Error::InvalidIdentityField { field, reason }),
        None => Ok(()),
    }
}
fn identity_layout(count: usize) -> Result<Layout, SemanticConstructionError> {
    Layout::array::<u8>(count).map_err(|_| SemanticConstructionError::Layout { storage: "identity", count })
}
fn copy_identity_bytes(value: &str, layout: Layout) -> Result<Vec<u8>, SemanticConstructionError> {
    // Caller validated a nonempty identity and checked this exact u8 layout,
    // then acquired the aggregate identity charge before entering this helper.
    if value.is_empty() || layout.size() != value.len() {
        return Err(SemanticConstructionError::PlanMismatch { storage: "identity", offset: 0 });
    }
    // SAFETY: Exact checked array<u8> layout, nonzero, alignment 1, global
    // allocator matching Vec. Caller charge precedes/outlives both byte locals;
    // no lock is held and null is checked before adoption or dereference.
    let pointer = unsafe { alloc(layout) };
    if pointer.is_null() {
        return Err(SemanticConstructionError::Allocation { storage: "identity", bytes: layout.size() });
    }
    // SAFETY: Immediate unique adoption with length zero and the exact allocated
    // capacity/alignment. The bounded copy initializes only those bytes. Vec
    // destroys that prefix/deallocates before the caller's charge on error,
    // unwind or final binding drop. No raw pointer or mutable backing escapes.
    let mut bytes = unsafe { Vec::from_raw_parts(pointer, 0, value.len()) };
    if value.len() > bytes.capacity() {
        return Err(SemanticConstructionError::PlanMismatch { storage: "identity", offset: 0 });
    }
    bytes.extend_from_slice(value.as_bytes());
    Ok(bytes)
}

/// Immutable manifest view borrowing the actual admitted owner.
#[derive(Clone, Copy)]
pub struct GgufImmutableManifest<'a> {
    owner: &'a GgufImmutableCompanionSemantics,
}
impl<'a> GgufImmutableManifest<'a> {
    pub fn config(self) -> &'a LlamaConfig {
        &self.owner.manifest.config
    }
    pub fn name(self) -> &'a str {
        self.owner.text(self.owner.manifest.name)
    }
    pub fn version(self) -> &'a str {
        self.owner.manifest.version.map_or("", |text| self.owner.text(text))
    }
    pub fn bos_token(self) -> &'a str {
        self.owner.text(self.owner.manifest.bos)
    }
    pub fn chat_style(self) -> ChatStyle {
        self.owner.manifest.chat_style
    }
    pub fn prompt_format(self) -> PromptFormat {
        self.owner.manifest.prompt_format
    }
    pub fn layout(self) -> PromptLayout {
        self.owner.manifest.layout
    }
    pub fn orders(self) -> OptionOrders {
        self.owner.manifest.orders
    }
    pub fn provenance(self) -> GgufImmutableCalibrationProvenance<'a> {
        GgufImmutableCalibrationProvenance { owner: self.owner }
    }
    pub fn calibration_unvalidated(self) -> Option<&'a str> {
        self.owner.unvalidated.as_deref()
    }
    pub fn thresholds(self) -> GgufSemanticObject<'a> {
        self.owner.selected_object(self.owner.manifest.thresholds)
    }
    pub fn task_thresholds(self) -> GgufSemanticObject<'a> {
        self.owner.selected_object(self.owner.manifest.task_thresholds)
    }
    pub fn multi_calibration(self) -> GgufSemanticObject<'a> {
        self.owner.selected_object(self.owner.manifest.multi)
    }
    pub fn temperature(self, kind: QType) -> f64 {
        self.owner.manifest.temperatures[kind_slot(kind)].unwrap_or(1.0)
    }
    pub fn temperature_buckets(self, kind: QType) -> impl Iterator<Item = TemperatureBucket> + 'a {
        // Constructor validated every key/value; parsing again borrows the same
        // immutable object and uses exactly the original primitive conversions.
        self.owner
            .selected_object(self.owner.manifest.bucket_objects[kind_slot(kind)])
            .members()
            .filter_map(|(key, value)| bucket(key, value))
    }
    pub fn temperature_for(self, kind: QType, options: usize) -> f64 {
        self.temperature_buckets(kind)
            .find(|bucket| bucket.lo <= options && options <= bucket.hi)
            .map_or_else(|| self.temperature(kind), |bucket| bucket.temperature)
    }
    pub fn has_calibration(self) -> bool {
        self.owner.manifest.temperatures.iter().any(Option::is_some)
            || self.owner.manifest.bucket_objects.iter().any(Option::is_some)
    }
    pub fn certified(self, target: &str, kind: QType) -> Option<f64> {
        let entry = self.thresholds().get(target)?;
        let value = if entry.get("per_type").is_none() && self.owner.manifest.calibration_prompt == PromptFormat::One {
            entry.get("confidence")
        } else {
            entry.get("per_type")?.get(kind.as_str())?.get("confidence")
        };
        number(value).filter(|value| (0.0..=1.0).contains(value))
    }
    pub fn task_threshold(self, task: &str, rate_key: &str) -> Option<f64> {
        self.task_thresholds().get(task)?.get(rate_key)?.as_f64()
    }
    pub fn multi_fallback_threshold(self, label_set_id: &str) -> Result<Option<f64>, Error> {
        let value = self.multi_calibration().get(label_set_id).and_then(|entry| entry.get("fallback_threshold"));
        let threshold = number(value);
        require(
            threshold.is_none_or(|value| (0.0..=1.0).contains(&value)),
            SemanticStage::CalibrationManifest,
            value,
            "fallback_threshold",
            "expected threshold in [0, 1]",
        )?;
        Ok(threshold)
    }
}

/// Borrowed operational provenance; digest remains the original-byte digest.
#[derive(Clone, Copy)]
pub struct GgufImmutableCalibrationProvenance<'a> {
    owner: &'a GgufImmutableCompanionSemantics,
}
impl<'a> GgufImmutableCalibrationProvenance<'a> {
    pub fn path(self) -> Option<&'a Path> {
        self.owner.provenance.path.as_deref()
    }
    pub fn content_sha256(self) -> Option<Sha256Digest> {
        self.owner.provenance.digest
    }
    pub fn explicit(self) -> bool {
        self.owner.provenance.explicit
    }
    pub fn off(self) -> bool {
        self.owner.provenance.off
    }
    pub fn strict(self) -> bool {
        self.owner.provenance.strict
    }
    pub fn identity_warning(self) -> Option<&'a str> {
        self.owner.provenance.identity_warning.as_deref()
    }
    pub fn fitted_model(self) -> Option<&'a str> {
        self.owner.provenance.fitted_model.map(|text| self.owner.text(text))
    }
    pub fn fitted_revision(self) -> Option<&'a str> {
        self.owner.provenance.fitted_revision.map(|text| self.owner.text(text))
    }
}

fn document(
    bytes: &[u8],
    role: ImmutableJsonRole,
    budget: &GgufDeclarationSourceBudget,
) -> Result<SemanticDocument, Error> {
    SemanticPlan::scan(bytes)
        .and_then(|plan| plan.materialize(budget))
        .map_err(|source| Error::Construction { role, source })
}

fn integer(raw: ValueRef<'_>, key: &'static str) -> Result<usize, Error> {
    raw.get(key)
        .and_then(ValueRef::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| rule(SemanticStage::Config, raw.get(key), key, "expected integer fitting usize"))
}
fn read_g1_config(raw: ValueRef<'_>) -> Result<LlamaConfig, Error> {
    let stage = SemanticStage::Config;
    require(
        raw.get("quantization_config").is_none_or(ValueRef::is_null),
        stage,
        raw.get("quantization_config"),
        "quantization_config",
        "serialized quantization is unsupported",
    )?;
    require(
        string(raw.get("model_type")) == Some("llama"),
        stage,
        raw.get("model_type"),
        "model_type",
        "expected ordinary Llama",
    )?;
    let architectures = raw.get("architectures").and_then(ValueRef::array);
    require(
        architectures.is_some_and(|array| {
            array.len() == 1 && array.at(0).and_then(ValueRef::as_str) == Some("LlamaForCausalLM")
        }),
        stage,
        raw.get("architectures"),
        "architectures",
        "expected singleton Llama architecture",
    )?;
    let bias = raw
        .get("attention_bias")
        .and_then(ValueRef::as_bool)
        .ok_or_else(|| rule(stage, raw.get("attention_bias"), "attention_bias", "expected boolean"))?;
    require(
        raw.get("mlp_bias").and_then(ValueRef::as_bool) == Some(bias),
        stage,
        raw.get("mlp_bias"),
        "mlp_bias",
        "bias policies differ",
    )?;
    require(
        string(raw.get("hidden_act")) == Some("silu"),
        stage,
        raw.get("hidden_act"),
        "hidden_act",
        "expected silu",
    )?;
    require(
        raw.get("tie_word_embeddings").and_then(ValueRef::as_bool) == Some(false),
        stage,
        raw.get("tie_word_embeddings"),
        "tie_word_embeddings",
        "expected false",
    )?;
    require(
        raw.get("pretraining_tp").is_none_or(|value| value.as_u64() == Some(1)),
        stage,
        raw.get("pretraining_tp"),
        "pretraining_tp",
        "expected integer one",
    )?;
    require(
        raw.get("rope_scaling").is_none_or(ValueRef::is_null),
        stage,
        raw.get("rope_scaling"),
        "rope_scaling",
        "unsupported scaling",
    )?;
    let rope = raw
        .get("rope_parameters")
        .ok_or_else(|| rule(stage, Some(raw), "rope_parameters", "missing RoPE parameters"))?;
    require(
        string(rope.get("rope_type")).unwrap_or("default") == "default",
        stage,
        Some(rope),
        "rope_type",
        "expected default RoPE",
    )?;
    let rope_theta =
        number(rope.get("rope_theta")).ok_or_else(|| rule(stage, Some(rope), "rope_theta", "expected number"))?;
    let config = LlamaConfig {
        num_layers: integer(raw, "num_hidden_layers")?,
        hidden: integer(raw, "hidden_size")?,
        intermediate: integer(raw, "intermediate_size")?,
        heads: integer(raw, "num_attention_heads")?,
        kv_heads: integer(raw, "num_key_value_heads")?,
        head_dim: integer(raw, "head_dim")?,
        vocab: integer(raw, "vocab_size")?,
        rms_eps: number(raw.get("rms_norm_eps"))
            .ok_or_else(|| rule(stage, raw.get("rms_norm_eps"), "rms_norm_eps", "expected number"))?,
        rope_theta,
        max_positions: integer(raw, "max_position_embeddings")?,
        bias,
    };
    require(
        [
            config.num_layers,
            config.hidden,
            config.intermediate,
            config.heads,
            config.kv_heads,
            config.head_dim,
            config.vocab,
            config.max_positions,
        ]
        .iter()
        .all(|value| *value > 0),
        stage,
        Some(raw),
        "dimensions",
        "expected positive dimensions",
    )?;
    require(
        config.rms_eps > 0.0 && config.rope_theta > 0.0,
        stage,
        Some(raw),
        "rms/rope",
        "expected positive finite values",
    )?;
    require(
        config.heads.checked_mul(config.head_dim) == Some(config.hidden),
        stage,
        Some(raw),
        "hidden_size",
        "heads * head_dim differs",
    )?;
    require(
        config.heads.is_multiple_of(config.kv_heads) && config.head_dim.is_multiple_of(2),
        stage,
        Some(raw),
        "heads",
        "invalid GQA or odd head dimension",
    )?;
    validate_g1_fields(raw, &config)?;
    Ok(config)
}

fn validate_g1_fields(raw: ValueRef<'_>, config: &LlamaConfig) -> Result<(), Error> {
    let object = raw.object().ok_or_else(|| rule(SemanticStage::Config, Some(raw), "config", "expected object"))?;
    let mut members = object.members();
    while let Some((field, value, offset)) = members.next_with_offset() {
        let valid = match field {
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
                validate_g1_rope(value)?;
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
            "rope_theta" => value.as_f64().is_some_and(|value| value > 0.0 && value == config.rope_theta),
            "partial_rotary_factor" => value.as_f64() == Some(1.0),
            "is_causal" | "is_decoder" => value.as_bool() == Some(true),
            "is_encoder_decoder" | "add_cross_attention" | "use_bidirectional_attention" => {
                value.as_bool() == Some(false)
            }
            "_attn_implementation" | "attn_implementation" => value.is_null() || value.as_str() == Some("eager"),
            "attention_dropout" => value.as_f64() == Some(0.0),
            "bos_token_id" | "eos_token_id" | "pad_token_id" => continue,
            "_name_or_path" | "transformers_version" => value.as_str().is_some(),
            "torch_dtype" | "dtype" => {
                value.is_null() || matches!(value.as_str(), Some("float16" | "bfloat16" | "float32"))
            }
            "initializer_range" => value.as_f64().is_some_and(|value| (0.0..=1.0).contains(&value)),
            "use_cache" => value.as_bool().is_some(),
            _ => false,
        };
        if !valid {
            return Err(Error::Rule {
                stage: SemanticStage::Config,
                offset,
                field: "config field",
                reason: "outside closed ordinary-Llama value policy",
            });
        }
    }
    Ok(())
}
fn validate_g1_rope(raw: ValueRef<'_>) -> Result<(), Error> {
    let object =
        raw.object().ok_or_else(|| rule(SemanticStage::Config, Some(raw), "rope_parameters", "expected object"))?;
    let mut members = object.members();
    while let Some((field, value, offset)) = members.next_with_offset() {
        let valid = match field {
            "rope_theta" => value.as_f64().is_some_and(|value| value > 0.0),
            "rope_type" | "type" => value.as_str() == Some("default"),
            "partial_rotary_factor" | "factor" | "attention_factor" => value.as_f64() == Some(1.0),
            _ => false,
        };
        if !valid {
            return Err(Error::Rule {
                stage: SemanticStage::Config,
                offset,
                field: "rope_parameters field",
                reason: "outside closed default-RoPE value policy",
            });
        }
    }
    Ok(())
}

fn read_protocol(
    config: LlamaConfig,
    basal: ValueRef<'_>,
    options: ModelLoadOptions<'_>,
) -> Result<ProtocolState, Error> {
    let stage = SemanticStage::Protocol;
    let prompt_format = parse_prompt(basal.get("prompt"), PromptFormat::One, stage)?;
    let orders = options
        .orders
        .map(|value| {
            OptionOrders::parse(value).map_err(|_| rule(stage, Some(basal), "orders", "invalid option orders"))
        })
        .transpose()?
        .unwrap_or_else(|| OptionOrders::default_for(prompt_format));
    let model_layout = basal
        .get("layout")
        .map(|value| {
            let text = value.as_str().ok_or_else(|| rule(stage, Some(value), "layout", "expected string"))?;
            PromptLayout::parse(text).map_err(|_| rule(stage, Some(value), "layout", "unknown layout"))
        })
        .transpose()?
        .unwrap_or_default();
    let layout = options.layout.unwrap_or(model_layout);
    layout.validate(prompt_format).map_err(|_| rule(stage, Some(basal), "layout", "layout requires format two"))?;
    let readout = basal.get("readout").ok_or_else(|| rule(stage, Some(basal), "readout", "missing readout"))?;
    require(string(readout.get("prefill")) == Some(PREFILL), stage, Some(readout), "prefill", "prefill differs")?;
    match prompt_format {
        PromptFormat::One => {
            require(
                string(readout.get("kind")) == Some("letter logits"),
                stage,
                Some(readout),
                "kind",
                "unsupported readout kind",
            )?;
            require(
                string(readout.get("letters")) == Some(LETTERS),
                stage,
                Some(readout),
                "letters",
                "letters differ",
            )?;
            match readout.get("templates").and_then(ValueRef::as_str) {
                Some(text) => require(
                    text == "basal-1.0 (unchanged prompt contract)",
                    stage,
                    Some(readout),
                    "templates",
                    "templates differ",
                )?,
                None => {
                    for (lang, key) in [(Lang::Pl, "pl"), (Lang::En, "en")] {
                        let template = prompt_format.template(lang);
                        let value = readout.get("templates").and_then(|value| value.get(key));
                        require(
                            string(get(value, "system")) == Some(template.system)
                                && string(get(value, "user")) == Some(template.user),
                            stage,
                            value,
                            "templates",
                            "language template differs",
                        )?;
                    }
                }
            }
        }
        PromptFormat::Two => {
            require(
                string(readout.get("kind")) == Some("code logits"),
                stage,
                Some(readout),
                "kind",
                "expected code logits",
            )?;
            require(
                readout.get("codes").and_then(ValueRef::as_u64) == Some(255),
                stage,
                Some(readout),
                "codes",
                "expected 255 codes",
            )?;
        }
    }
    Ok(ProtocolState { config, chat_style: ChatStyle::Llama, prompt_format, layout, orders })
}
fn parse_prompt(
    value: Option<ValueRef<'_>>,
    default: PromptFormat,
    stage: SemanticStage,
) -> Result<PromptFormat, Error> {
    let Some(value) = value else {
        return Ok(default);
    };
    let integer = value.as_u64().ok_or_else(|| rule(stage, Some(value), "prompt", "expected integer"))?;
    PromptFormat::parse(integer).map_err(|_| rule(stage, Some(value), "prompt", "unsupported format"))
}

fn audit_specials_ref(raw: ValueRef<'_>, vocab: usize, bos: u32, location: &'static str) -> Result<(), Error> {
    let parse = |value: ValueRef<'_>| -> Result<u32, Error> {
        let id = value
            .as_u64()
            .ok_or_else(|| rule(SemanticStage::Tokenizer, Some(value), location, "expected integer token ID"))?;
        GgufSemanticError::check_id(id, location, vocab).map_err(|source| Error::Tokenizer { source })
    };
    for key in ["bos_token_id", "eos_token_id", "pad_token_id"] {
        let Some(value) = raw.get(key).filter(|value| !value.is_null()) else {
            continue;
        };
        match value.array() {
            Some(array) if key == "eos_token_id" => {
                for id in array.elements() {
                    parse(id)?;
                }
            }
            _ => {
                let id = parse(value)?;
                require(
                    key != "bos_token_id" || id == bos,
                    SemanticStage::Tokenizer,
                    Some(value),
                    location,
                    "declared BOS disagreement",
                )?;
            }
        }
    }
    Ok(())
}

fn probe(capture: &mut CalibrationCapture, path: &Path, role: CalibrationProbeRole) -> bool {
    let exists = path.exists();
    capture.record_probe(role, exists);
    exists
}
fn layout_mismatch_ref(data: Option<ValueRef<'_>>, path: &Path, layout: PromptLayout) -> Result<Option<String>, Error> {
    if data.and_then(ValueRef::object).is_none_or(GgufSemanticObject::is_empty) {
        return Ok(None);
    }
    let set = |value: &ValueRef<'_>| !value.is_null() && value.as_str() != Some("");
    let value = get(get(data, "fitted_on"), "layout").filter(set).or_else(|| get(data, "layout").filter(set));
    let fitted = match value {
        None => "canonical",
        Some(value) => value
            .as_str()
            .ok_or_else(|| rule(SemanticStage::CalibrationSelection, Some(value), "layout", "expected string"))?,
    };
    Ok((fitted != layout.as_str()).then(|| {
        let name = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy();
        format!("{name} was fitted on the {fitted} layout, served with --layout {layout}")
    }))
}
fn fitted_orders_ref(data: Option<ValueRef<'_>>) -> Result<Option<OptionOrders>, Error> {
    let value = get(get(data, "fitted_on"), "orders")
        .filter(|value| !value.is_null())
        .or_else(|| get(data, "orders").filter(|value| !value.is_null()));
    value
        .map(|value| {
            let text = value.as_str().or_else(|| value.number_text()).ok_or_else(|| {
                rule(SemanticStage::CalibrationSelection, Some(value), "orders", "expected orders string or number")
            })?;
            OptionOrders::parse(text)
                .map_err(|_| rule(SemanticStage::CalibrationSelection, Some(value), "orders", "invalid fitted orders"))
        })
        .transpose()
}
fn load_immutable_calibration(
    directory: &Path,
    options: ModelLoadOptions<'_>,
    protocol: &ProtocolState,
    capture: &mut CalibrationCapture,
    budget: &GgufDeclarationSourceBudget,
) -> Result<LoadedImmutableCalibration, Error> {
    let explicit = matches!(options.calibration, CalibrationSource::File(_));
    let off = matches!(options.calibration, CalibrationSource::Off);
    let layout = protocol.layout;
    let orders = protocol.orders;
    let mut path = match options.calibration {
        CalibrationSource::File(path) => path.to_path_buf(),
        _ => {
            let preferred = directory.join(format!("CALIBRATION.{layout}.json"));
            if layout != PromptLayout::Canonical && probe(capture, &preferred, CalibrationProbeRole::LayoutPreference) {
                preferred
            } else {
                directory.join("CALIBRATION.json")
            }
        }
    };
    let mut loaded = LoadedImmutableCalibration {
        documents: [None, None],
        selected: SelectedCalibration::Empty,
        provenance: ImmutableProvenance {
            path: None,
            digest: None,
            explicit,
            off,
            strict: options.strict_calibration,
            fitted_model: None,
            fitted_revision: None,
            identity_warning: None,
        },
        unvalidated: None,
    };
    if !off && (explicit || probe(capture, &path, CalibrationProbeRole::Candidate)) {
        let role = if explicit { CalibrationReadRole::Explicit } else { CalibrationReadRole::Candidate };
        let (slot, document) = capture.read_immutable(&path, role, budget)?;
        loaded.documents[slot] = Some(document);
        loaded.selected = SelectedCalibration::Slot(slot);
    }
    loaded.unvalidated = layout_mismatch_ref(loaded.root(), &path, layout)?;
    capture.record_layout(true, loaded.unvalidated.is_some());
    if loaded.unvalidated.is_none() {
        let replace = !explicit && !off && {
            let fitted = fitted_orders_ref(loaded.root())?;
            capture.record_orders(true, fitted, orders);
            fitted.is_some_and(|value| value != orders)
        };
        if replace {
            let alternate = path.with_extension(format!("o{orders}.json"));
            if probe(capture, &alternate, CalibrationProbeRole::OrdersVariant) {
                let (slot, document) =
                    capture.read_immutable(&alternate, CalibrationReadRole::OrdersVariant, budget)?;
                // Candidate remains owned/charged while alternate layout is checked.
                loaded.unvalidated = layout_mismatch_ref(Some(document.root()), &alternate, layout)?;
                capture.record_layout(false, loaded.unvalidated.is_some());
                capture.record_variant(OrdersVariantDecision::Selected);
                loaded.documents[slot] = Some(document);
                loaded.selected = SelectedCalibration::Slot(slot);
                path = alternate;
            } else {
                capture.record_variant(OrdersVariantDecision::ProbeFalse);
            }
        }
        if loaded.unvalidated.is_none() {
            let fitted = fitted_orders_ref(loaded.root())?;
            capture.record_orders(false, fitted, orders);
            loaded.unvalidated = fitted
                .filter(|value| *value != orders)
                .map(|fitted| format!("calibration was fitted on {fitted} option order(s), served with {orders}"));
        }
    }
    let doc_slot = match loaded.selected {
        SelectedCalibration::Empty => None,
        SelectedCalibration::Slot(slot) => Some(DocSlot::Calibration(slot)),
    };
    let fitted = get(loaded.root(), "fitted_on");
    let model = string(get(fitted, "model"));
    let revision = string(get(fitted, "revision"));
    let fitted_model = doc_slot.and_then(|slot| text_ref(get(fitted, "model"), slot));
    let fitted_revision = doc_slot.and_then(|slot| text_ref(get(fitted, "revision"), slot));
    let warning = options
        .served_identity
        .map(|served| check_identity(&loaded.provenance, model, revision, served))
        .transpose()?
        .flatten();
    let digest = capture.selected().map(|file| file.sha256());
    loaded.provenance.path = (!off).then_some(path);
    loaded.provenance.digest = digest;
    loaded.provenance.fitted_model = fitted_model;
    loaded.provenance.fitted_revision = fitted_revision;
    loaded.provenance.identity_warning = warning;
    Ok(loaded)
}
fn check_identity(
    provenance: &ImmutableProvenance,
    model: Option<&str>,
    revision: Option<&str>,
    served: ServedIdentity<'_>,
) -> Result<Option<String>, Error> {
    let warning = if provenance.explicit && !provenance.off {
        calibration_identity_mismatch(model, revision, served)
    } else {
        None
    };
    if provenance.strict {
        if let Some(why) = &warning {
            return Err(Error::Identity { source: CalibrationIdentityError(why.clone()) });
        }
    }
    Ok(warning)
}

fn kind_slot(kind: QType) -> usize {
    match kind {
        QType::Choice => 0,
        QType::Noul => 1,
        QType::Score => 2,
        QType::Multi => 3,
        QType::Act => 4,
        QType::Tool => 5,
    }
}
fn object_members<'a>(value: ValueRef<'a>, field: &'static str) -> Result<GgufSemanticMembers<'a>, Error> {
    value
        .object()
        .map(GgufSemanticObject::members)
        .ok_or_else(|| rule(SemanticStage::CalibrationManifest, Some(value), field, "expected object"))
}
fn bucket(key: &str, value: ValueRef<'_>) -> Option<TemperatureBucket> {
    let (lo, hi) = key.split_once('-')?;
    if lo.is_empty() || hi.is_empty() || !lo.bytes().chain(hi.bytes()).all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let (lo, hi) = (lo.parse::<usize>().ok()?, hi.parse::<usize>().ok()?);
    let temperature = value.as_f64()?;
    (lo <= hi && temperature > 0.0).then_some(TemperatureBucket { lo, hi, temperature })
}
fn selected_map(data: Option<ValueRef<'_>>, key: &'static str, validated: bool) -> Result<Option<NodeId>, Error> {
    if !validated {
        return Ok(None);
    }
    get(data, key)
        .map(|value| {
            object_members(value, key)?;
            Ok(value.id())
        })
        .transpose()
}
fn finish_manifest_ref(
    protocol: ProtocolState,
    basal: ValueRef<'_>,
    bos: TextRef,
    loaded: &LoadedImmutableCalibration,
) -> Result<ImmutableManifestState, Error> {
    let stage = SemanticStage::CalibrationManifest;
    let data = loaded.root();
    let calibration_prompt = parse_prompt(get(data, "prompt"), protocol.prompt_format, stage)?;
    require(
        calibration_prompt == protocol.prompt_format,
        stage,
        get(data, "prompt"),
        "prompt",
        "calibration fitted for another prompt format",
    )?;
    let mut temperatures = [None; 6];
    let mut bucket_objects = [None; 6];
    if let Some(value) = get(data, "temperature_per_prim") {
        for (key, value) in object_members(value, "temperature_per_prim")? {
            let kind = QType::parse(key)
                .ok_or_else(|| rule(stage, Some(value), "temperature_per_prim kind", "unknown primitive"))?;
            let temperature = value
                .as_f64()
                .filter(|value| *value > 0.0)
                .ok_or_else(|| rule(stage, Some(value), "temperature", "expected positive finite number"))?;
            temperatures[kind_slot(kind)] = Some(temperature);
        }
    }
    if let Some(value) = get(data, "temperature_per_bucket") {
        for (key, buckets) in object_members(value, "temperature_per_bucket")? {
            let kind = QType::parse(key)
                .ok_or_else(|| rule(stage, Some(buckets), "temperature_per_bucket kind", "unknown primitive"))?;
            for (key, value) in object_members(buckets, "buckets")? {
                require(bucket(key, value).is_some(), stage, Some(value), "bucket", "invalid range or temperature")?;
            }
            bucket_objects[kind_slot(kind)] = Some(buckets.id());
        }
    }
    let validated = loaded.unvalidated.is_none();
    let thresholds = selected_map(data, "thresholds", validated)?;
    let task_thresholds = selected_map(data, "task_thresholds", validated)?;
    if task_thresholds.is_some() {
        // The node is the selected task object, validated above. Borrow through
        // data rather than copying a map or constructing an owned rate table.
        if let Some(tasks) = get(data, "task_thresholds") {
            for (_, rates) in object_members(tasks, "task_thresholds")? {
                for (rate, value) in object_members(rates, "task rates")? {
                    let risk = rate.parse::<f64>().ok().filter(|risk| risk.is_finite() && *risk > 0.0 && *risk < 1.0);
                    require(risk.is_some(), stage, Some(value), "task rate", "expected finite rate in (0, 1)")?;
                    require(
                        value.as_f64().is_some_and(|value| (0.0..=1.0).contains(&value)),
                        stage,
                        Some(value),
                        "task threshold",
                        "expected threshold in [0, 1]",
                    )?;
                }
            }
        }
    }
    let multi = selected_map(data, "multi", validated)?;
    let name = text_ref(basal.get("name"), DocSlot::Basal)
        .ok_or_else(|| rule(stage, Some(basal), "name", "expected string"))?;
    let version = text_ref(basal.get("version"), DocSlot::Basal);
    Ok(ImmutableManifestState {
        config: protocol.config,
        chat_style: protocol.chat_style,
        prompt_format: protocol.prompt_format,
        layout: protocol.layout,
        orders: protocol.orders,
        calibration_prompt,
        name,
        version,
        bos,
        temperatures,
        bucket_objects,
        thresholds,
        task_thresholds,
        multi,
        selected: loaded.selected,
    })
}
