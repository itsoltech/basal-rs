//! CPU request planning shared by the online engine and offline (batch) validation: parse and validate a request,
//! prepare the prompts and token ids of its option orders and build its [`RequestPlan`], without a GPU backend.
//!
//! [`PlanContext`] borrows the model manifest and a tokenizer. The online [`Engine`](crate::Engine) builds one per
//! call over its own tokenizer and backend capabilities with [`PlanBudget::unbounded`], which skips every bound
//! computation, so online operation order, allocations and results stay as before the extraction. An offline
//! [`RequestPlanner`] owns its tokenizer and a [`PlanCaps`] snapshot of the backend capabilities (no backend, no
//! `Backend: Sync` requirement) and plans under a bounded [`PlanBudget`]: host memory is reserved before each
//! allocation whose size is known from preflight, and a reservation that does not fit is a typed
//! [`BudgetExceeded`] naming its [`PlanStage`].
//!
//! The pure estimators [`result_bound`], [`retained_bound`] and [`exec_bound`] give the host memory a plan holds
//! after planning, needs while it runs and the size of its native response. They are conservative estimates for
//! admission; their constants are not measurements and are compared with measured peaks in the qualification runs.

use std::cell::Cell;
use std::sync::Arc;

use serde_json::Value;

use crate::engine::{prepare_protocol, Prepared, QuestionPlan, RequestPlan};
use crate::execution::{Interrupted, StopToken};
use crate::manifest::ModelManifest;
use crate::pack::{pack, TrieSize};
use crate::prompt::{render_model_with_layout, ChatStyle, OptionOrders, PromptFormat, PromptLayout};
use crate::request::{check_capability_with_limit, DecideError, Item};
use crate::tokenize::BasalTokenizer;

/// Transient bytes the tokenizer may allocate per byte of an encoded prompt (normalised copy with per-byte
/// alignments, pre-token splits and the encoding's per-token fields, at most one token per byte). An estimator
/// constant, not a measurement; the code-id probes encode the prompt with one more letter again.
pub(crate) const TOKENIZER_BYTES_PER_BYTE: u64 = 160;

/// Largest native result line of one record accepted by the batch contract (and the bound used for features
/// without a verified estimate).
pub const MAX_RESULT_BYTES: u64 = 64 * 1024 * 1024;

/// Planning options of an engine (copied from [`crate::Engine`]).
#[derive(Clone, Copy, Debug)]
pub struct PlanOptions {
    pub overflow: crate::overflow::OverflowSettings,
    pub mark_state: bool,
    pub large_choice: bool,
    pub large_choice_strategy: crate::large_choice::Strategy,
}

/// Snapshot of the planning-relevant capabilities of a backend, sendable to CPU planning workers. Built once from a
/// loaded backend ([`PlanCaps::from_backend`]); media are planned through the backend's shared media processor.
#[derive(Clone)]
pub struct PlanCaps {
    pub has_evidence: bool,
    pub media_kinds: Vec<crate::media::Kind>,
    pub processor: Option<Arc<crate::media::Processor>>,
    /// Why media cannot be planned offline although the backend takes media (no shared processor).
    pub media_unavailable: Option<String>,
}

impl std::fmt::Debug for PlanCaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanCaps")
            .field("has_evidence", &self.has_evidence)
            .field("media_kinds", &self.media_kinds)
            .field("processor", &self.processor.is_some())
            .field("media_unavailable", &self.media_unavailable)
            .finish()
    }
}

impl PlanCaps {
    pub fn from_backend<B: crate::Backend + ?Sized>(backend: &B) -> Self {
        let kinds = backend.media_kinds();
        let processor = backend.media_processor();
        let media_unavailable = (!kinds.is_empty() && processor.is_none())
            .then(|| "this backend takes media but exposes no shared media processor for offline planning".into());
        Self { has_evidence: backend.has_evidence(), media_kinds: kinds, processor, media_unavailable }
    }

    /// Text-only capabilities (no media, no evidence head).
    pub fn text_only() -> Self {
        Self { has_evidence: false, media_kinds: Vec::new(), processor: None, media_unavailable: None }
    }
}

/// Capabilities a [`PlanContext`] asks for lazily (the online path queries its backend only where it did before).
pub(crate) trait PlanCapabilities {
    fn has_evidence(&self) -> bool;
    fn media_kinds(&self) -> Vec<crate::media::Kind>;
    fn prepare_media(
        &self,
        entries: &Value,
        detail: &str,
        budget: &PlanBudget,
        stop: Option<&StopToken>,
    ) -> anyhow::Result<Arc<crate::media::ProcessedMedia>>;
}

/// The online path: the backend's own media preparation, unchanged. It has no host-budget accounting, so it refuses
/// a bounded budget instead of ignoring it.
pub(crate) struct BackendCaps<'b, B: ?Sized>(pub(crate) &'b B);

impl<B: crate::Backend + ?Sized> PlanCapabilities for BackendCaps<'_, B> {
    fn has_evidence(&self) -> bool {
        self.0.has_evidence()
    }
    fn media_kinds(&self) -> Vec<crate::media::Kind> {
        self.0.media_kinds()
    }
    fn prepare_media(
        &self,
        entries: &Value,
        detail: &str,
        budget: &PlanBudget,
        _stop: Option<&StopToken>,
    ) -> anyhow::Result<Arc<crate::media::ProcessedMedia>> {
        if budget.is_bounded() {
            return Err(UnboundedFeature("backend media preparation has no host budget accounting".into()).into());
        }
        self.0.prepare_media(entries, detail)
    }
}

impl PlanCapabilities for PlanCaps {
    fn has_evidence(&self) -> bool {
        self.has_evidence
    }
    fn media_kinds(&self) -> Vec<crate::media::Kind> {
        if self.processor.is_some() {
            self.media_kinds.clone()
        } else {
            Vec::new()
        }
    }
    fn prepare_media(
        &self,
        entries: &Value,
        detail: &str,
        budget: &PlanBudget,
        stop: Option<&StopToken>,
    ) -> anyhow::Result<Arc<crate::media::ProcessedMedia>> {
        let Some(processor) = &self.processor else {
            let reason = self.media_unavailable.clone().unwrap_or_else(|| "this model does not take media".into());
            return Err(UnboundedFeature(reason).into());
        };
        let budgeted = if budget.is_bounded() { Some((budget, stop)) } else { None };
        Ok(Arc::new(processor.process_with(entries, detail, budgeted)?))
    }
}

/// A feature of the record has no verified host-memory bound (or is unavailable offline); bounded planning refuses
/// it with this reason.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{0}")]
pub struct UnboundedFeature(pub String);

/// Planning stage of a host-memory reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanStage {
    /// Bytes of the record line.
    Line,
    /// The parsed JSON value of the record.
    Json,
    /// A decoded base64 media payload.
    Base64,
    /// Transient allocations of an image decoder (its `image::Limits::max_alloc`).
    ImageDecoder,
    /// The decoded image buffer and its RGB conversion.
    ImageBuffer,
    /// Resampling scratch and the resized image.
    Bicubic,
    /// Padded image patch features (retained until the end of the plan).
    ImageFeatures,
    /// Audio container/codec buffers and decoded packets.
    AudioDecoder,
    /// Mono PCM and its resampled copy.
    AudioPcm,
    /// Audio features (retained until the end of the plan).
    AudioFeatures,
    /// Spectral analysis scratch (FFT, filters).
    AudioAnalysis,
    /// Item copies for branches and groups.
    Items,
    /// Rendered prompts and their token ids.
    Tokens,
    /// Packed rows of prepared questions.
    Packed,
}

impl PlanStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Json => "json",
            Self::Base64 => "base64",
            Self::ImageDecoder => "image_decoder",
            Self::ImageBuffer => "image_buffer",
            Self::Bicubic => "bicubic",
            Self::ImageFeatures => "image_features",
            Self::AudioDecoder => "audio_decoder",
            Self::AudioPcm => "audio_pcm",
            Self::AudioFeatures => "audio_features",
            Self::AudioAnalysis => "audio_analysis",
            Self::Items => "items",
            Self::Tokens => "tokens",
            Self::Packed => "packed",
        }
    }
}

impl std::fmt::Display for PlanStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A host-memory reservation that does not fit the record's planning budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("host planning budget exceeded at stage {stage}: {requested} B requested, {used} of {limit} B in use")]
pub struct BudgetExceeded {
    pub stage: PlanStage,
    pub requested: u64,
    pub used: u64,
    pub limit: u64,
}

/// Host-memory budget of planning one record (the worker's fixed slot `R`). Single-threaded accounting: no waiting
/// inside the slot (so no reservation cycle) and no `Sync` requirement. Reservations are RAII; bytes retained to the
/// end of the plan stay counted until [`PlanBudget::reset`].
///
/// [`PlanBudget::unbounded`] (the online path) accounts nothing; planning code skips bound computations for it.
pub struct PlanBudget {
    limit: Option<u64>,
    used: Cell<u64>,
    peak: Cell<u64>,
    retained: Cell<u64>,
}

impl std::fmt::Debug for PlanBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanBudget")
            .field("limit", &self.limit)
            .field("used", &self.used.get())
            .field("peak", &self.peak.get())
            .field("retained", &self.retained.get())
            .finish()
    }
}

impl PlanBudget {
    pub fn unbounded() -> Self {
        Self { limit: None, used: Cell::new(0), peak: Cell::new(0), retained: Cell::new(0) }
    }

    pub fn new(limit_bytes: u64) -> Self {
        Self { limit: Some(limit_bytes), used: Cell::new(0), peak: Cell::new(0), retained: Cell::new(0) }
    }

    pub fn is_bounded(&self) -> bool {
        self.limit.is_some()
    }

    pub fn limit(&self) -> Option<u64> {
        self.limit
    }

    /// Bytes currently reserved (live reservations and retained bytes).
    pub fn used(&self) -> u64 {
        self.used.get()
    }

    /// Highest `used` since the last reset: the planning peak of the record.
    pub fn peak(&self) -> u64 {
        self.peak.get()
    }

    /// Bytes retained to the end of the plan (part of `used`).
    pub fn retained(&self) -> u64 {
        self.retained.get()
    }

    /// Reserve `bytes` for `stage` before allocating them. Unbounded budgets return an empty reservation.
    pub fn reserve(&self, stage: PlanStage, bytes: u64) -> Result<Reservation<'_>, BudgetExceeded> {
        let Some(limit) = self.limit else {
            return Ok(Reservation { budget: self, bytes: 0 });
        };
        let used = self.used.get();
        match used.checked_add(bytes).filter(|n| *n <= limit) {
            Some(n) => {
                self.used.set(n);
                self.peak.set(self.peak.get().max(n));
                Ok(Reservation { budget: self, bytes })
            }
            None => Err(BudgetExceeded { stage, requested: bytes, used, limit }),
        }
    }

    /// Forget all reservations and statistics (between records). Live [`Reservation`]s must not outlive this call's
    /// record; their later drops saturate at zero.
    pub fn reset(&self) {
        self.used.set(0);
        self.peak.set(0);
        self.retained.set(0);
    }

    /// A failed CPU attempt has been dropped. Its retained prompt/branch buffers no longer consume the slot.
    /// The caller must keep earlier allocations and live transient reservations counted.
    fn release_attempt(&self, before: u64) {
        let freed = self.retained.get().saturating_sub(before);
        self.retained.set(before);
        self.used.set(self.used.get().saturating_sub(freed));
    }
}

/// Reserved bytes of a [`PlanBudget`], returned on drop unless retained.
#[must_use = "a reservation is returned when dropped"]
pub struct Reservation<'b> {
    budget: &'b PlanBudget,
    bytes: u64,
}

impl Reservation<'_> {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Keep the bytes counted until the end of the plan ([`PlanBudget::reset`]).
    pub fn retain(mut self) {
        let b = std::mem::take(&mut self.bytes);
        self.budget.retained.set(self.budget.retained.get().saturating_add(b));
    }

    /// Make the reservation exactly `bytes`: return the excess, or reserve the shortfall for `stage` when an
    /// allocation turned out larger than its bound (so the budget never under-counts).
    pub fn fit(&mut self, stage: PlanStage, bytes: u64) -> Result<(), BudgetExceeded> {
        if bytes > self.bytes {
            let extra = self.budget.reserve(stage, bytes - self.bytes)?;
            self.bytes += std::mem::ManuallyDrop::new(extra).bytes;
        } else {
            self.shrink_to(bytes);
        }
        Ok(())
    }

    /// Return the part above `bytes` (the allocation turned out smaller than its bound).
    pub fn shrink_to(&mut self, bytes: u64) {
        if bytes < self.bytes {
            let freed = self.bytes - bytes;
            self.budget.used.set(self.budget.used.get().saturating_sub(freed));
            self.bytes = bytes;
        }
    }
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.budget.used.set(self.budget.used.get().saturating_sub(self.bytes));
        }
    }
}

/// Why a request could not be planned.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The request is invalid or unsupported (the same errors and HTTP mapping as online).
    #[error(transparent)]
    Decide(DecideError),
    /// A reservation did not fit the planning budget.
    #[error(transparent)]
    Budget(#[from] BudgetExceeded),
    /// Planning was stopped.
    #[error(transparent)]
    Interrupted(#[from] Interrupted),
    /// A feature without a verified host-memory bound (refused by bounded planning).
    #[error("unsupported in bounded planning: {0}")]
    Unbounded(#[from] UnboundedFeature),
}

impl From<DecideError> for PlanError {
    fn from(e: DecideError) -> Self {
        Self::Decide(e)
    }
}

impl From<anyhow::Error> for PlanError {
    /// Typed planning conditions keep their type; any other error is internal, as online.
    fn from(e: anyhow::Error) -> Self {
        if let Some(b) = e.chain().find_map(|c| c.downcast_ref::<BudgetExceeded>()) {
            return Self::Budget(*b);
        }
        if let Some(i) = e.chain().find_map(|c| c.downcast_ref::<Interrupted>()) {
            return Self::Interrupted(*i);
        }
        if let Some(u) = e.chain().find_map(|c| c.downcast_ref::<UnboundedFeature>()) {
            return Self::Unbounded(u.clone());
        }
        Self::Decide(DecideError::Internal(e))
    }
}

impl PlanError {
    /// The online error of this planning error (online planning never produces the other kinds).
    pub fn into_decide(self) -> DecideError {
        match self {
            Self::Decide(e) => e,
            other => DecideError::Internal(anyhow::Error::new(other)),
        }
    }
}

/// Borrowed planning context: manifest, tokenizer, options, capabilities, budget and an optional stop token.
pub struct PlanContext<'a> {
    pub(crate) manifest: &'a ModelManifest,
    pub(crate) tok: &'a BasalTokenizer,
    pub(crate) options: PlanOptions,
    caps: &'a dyn PlanCapabilities,
    budget: &'a PlanBudget,
    stop: Option<&'a StopToken>,
}

impl<'a> PlanContext<'a> {
    /// Offline context over a capability snapshot.
    pub fn new(
        manifest: &'a ModelManifest,
        tok: &'a BasalTokenizer,
        options: PlanOptions,
        caps: &'a PlanCaps,
        budget: &'a PlanBudget,
        stop: Option<&'a StopToken>,
    ) -> Self {
        Self { manifest, tok, options, caps, budget, stop }
    }

    pub(crate) fn with_caps(
        manifest: &'a ModelManifest,
        tok: &'a BasalTokenizer,
        options: PlanOptions,
        caps: &'a dyn PlanCapabilities,
        budget: &'a PlanBudget,
        stop: Option<&'a StopToken>,
    ) -> Self {
        Self { manifest, tok, options, caps, budget, stop }
    }

    pub fn budget(&self) -> &PlanBudget {
        self.budget
    }

    fn check_stop(&self) -> Result<(), Interrupted> {
        self.stop.map_or(Ok(()), StopToken::check)
    }

    fn bounded(&self) -> Option<&'a PlanBudget> {
        self.budget.is_bounded().then_some(self.budget)
    }

    fn truncates(&self) -> bool {
        self.options.overflow.policy() == crate::overflow::ContextOverflow::Truncate
    }

    pub fn prepare(&self, item: Item) -> anyhow::Result<Prepared> {
        let orders = self.manifest.orders.count(item.options.len());
        self.prepare_orders(item, orders)
    }

    /// The first `orders` option orders only (1: as written; a screening group of a large choice).
    pub fn prepare_orders(&self, item: Item, orders: usize) -> anyhow::Result<Prepared> {
        self.prepare_orders_for(item, orders, None)
    }

    pub(crate) fn prepare_orders_for(
        &self,
        item: Item,
        orders: usize,
        media: Option<&crate::media::ProcessedMedia>,
    ) -> anyhow::Result<Prepared> {
        let budget = self.bounded();
        let mut p = prepare_protocol(
            self.tok,
            &self.manifest.bos_token,
            item,
            self.manifest.prompt_format,
            self.manifest.chat_style,
            self.manifest.layout,
            media,
            budget,
        )?;
        if orders < p.orders.len() {
            p.orders.truncate(orders);
            let packed_reservation = match budget {
                Some(b) => Some(b.reserve(PlanStage::Packed, packed_bound(&p.orders))?),
                None => None,
            };
            p.packed = pack(&p.orders.iter().map(|o| o.input_ids.clone()).collect::<Vec<_>>());
            if let Some(r) = packed_reservation {
                r.retain();
            }
        }
        if self.options.mark_state {
            p.state_len = self.state_len(&p)?;
            p.packed.state_len = p.state_len.min(p.packed.prefix_len);
        }
        Ok(p)
    }

    /// Tokens of the prompt up to and including the state: the common prefix of the item's prompt with two probe
    /// prompts that keep the state and change the question. Format 2 takes the exact rendered prefix through the
    /// first state copy; qfirst includes the actual question in that prefix and cache reuse requires a token match.
    pub(crate) fn state_len(&self, p: &Prepared) -> anyhow::Result<usize> {
        self.check_stop()?;
        // Prefix-cache probes also render/tokenize after a cut. Count both live token lists and tokenizer scratch
        // before creating the probe strings, just as for the full prompt.
        let _probe = match self.bounded() {
            Some(b) if self.truncates() => {
                let bytes = p.orders.iter().map(|o| o.prompt.len() as u64).max().unwrap_or(0);
                Some(b.reserve(
                    PlanStage::Tokens,
                    bytes.saturating_mul(TOKENIZER_BYTES_PER_BYTE + 16).saturating_add(4096),
                )?)
            }
            _ => None,
        };
        let it = &p.item;
        if self.manifest.prompt_format == PromptFormat::Two {
            if it.state.is_empty() {
                return Ok(0);
            }
            let label = match it.lang {
                crate::prompt::Lang::Pl => "Stan:\n",
                crate::prompt::Lang::En => "State:\n",
            };
            let needle = format!("{label}{}", it.state);
            let prompt = &p.orders[0].prompt;
            let start = prompt.find(&needle).ok_or_else(|| anyhow::anyhow!("state not found in rendered prompt"))?;
            let prefix = self.tok.encode(&prompt[..start + needle.len()])?;
            return Ok(prefix.iter().zip(&p.orders[0].input_ids).take_while(|(a, b)| a == b).count());
        }

        let probe = |q: &str| {
            self.tok.encode(&render_model_with_layout(
                self.manifest.chat_style,
                self.manifest.prompt_format,
                self.manifest.layout,
                &self.manifest.bos_token,
                &it.state,
                q,
                &["x", "y"],
                it.lang,
            )?)
        };
        let (a, b) = (probe("A")?, probe("B")?);
        let ids = &p.orders[0].input_ids;
        Ok(a.iter().zip(&b).zip(ids).take_while(|((x, y), z)| x == y && x == z).count())
    }

    /// Parse, validate and prepare a System One request without running the model.
    pub fn prepare_request(&self, body: &Value) -> Result<Vec<Prepared>, PlanError> {
        if self.truncates()
            || self.manifest.prompt_format == PromptFormat::Two
            || body.get("media").is_some_and(|v| !v.is_null())
            || body.get("assets").is_some()
        {
            return self.plan_request(body).map(|p| p.prepared);
        }
        crate::systemone::check_features(body)?;
        let req = crate::request::parse_request_for_format(
            body,
            crate::request::Dialect::TypeSafe,
            self.manifest.prompt_format,
        )?;
        if req.model != self.manifest.name {
            return Err(crate::request::api_usage_error(format!("Unknown model: {}", req.model)).into());
        }
        check_capability_with_limit(&req.items, self.manifest.prompt_format.max_options(), false, false)?;
        Ok(req.items.into_iter().map(|it| self.prepare(it)).collect::<anyhow::Result<Vec<_>>>()?)
    }

    /// Validate a request and plan its questions: one readout for 2..=10 options, the grouped strategy for choice
    /// questions with 11..=255 options (when `large_choice` is on).
    pub fn plan_request(&self, body: &Value) -> Result<RequestPlan, PlanError> {
        self.plan_request_as(body, crate::request::Dialect::TypeSafe)
    }

    /// [`PlanContext::plan_request`] for an endpoint dialect (upstream: no model check, 2..10 options only).
    pub fn plan_request_as(&self, body: &Value, dialect: crate::request::Dialect) -> Result<RequestPlan, PlanError> {
        self.check_stop()?;
        // Parsing and task expansion clone state, options and JSON before token preparation. Include the retained
        // task text/character indexes and transient expansion copies, not only the final rendered prompts.
        let items_bound = if self.budget.is_bounded() && self.truncates() {
            (crate::request::prepared_input_bytes(body)? as u64).saturating_mul(16).saturating_add(65536)
        } else {
            0
        };
        let items = self.budget.reserve(PlanStage::Items, items_bound)?;
        if dialect == crate::request::Dialect::TypeSafe {
            crate::systemone::check_features(body)?;
            // Asset questions are restricted to the TypeSafe primitives before native task expansion.
            if body.get("assets").is_some() {
                let plan = self.plan_decision_request_as(body, dialect)?;
                items.retain();
                return Ok(plan);
            }
        }
        let plan = if let Some(expansion) = crate::tasks::expand(body, self.manifest.prompt_format.max_options())? {
            let mut plan = self.plan_decision_request_as(&expansion.request, dialect)?;
            plan.tasks = Some(expansion.plan);
            plan.task_bytes = items_bound;
            plan
        } else {
            self.plan_decision_request_as(body, dialect)?
        };
        items.retain();
        Ok(plan)
    }

    /// Media preparation errors: input errors as online (422 on the media field); typed planning conditions of a
    /// bounded budget keep their type.
    fn media_error(&self, e: anyhow::Error, field: &str) -> PlanError {
        if self.budget.is_bounded() {
            let typed = PlanError::from(e);
            return match typed {
                PlanError::Decide(DecideError::Internal(e)) => {
                    PlanError::Decide(DecideError::Validation(vec![crate::request::FieldError::new(
                        vec!["body".into(), field.into()],
                        format!("{e:#}"),
                        "value_error",
                    )]))
                }
                other => other,
            };
        }
        PlanError::Decide(DecideError::Validation(vec![crate::request::FieldError::new(
            vec!["body".into(), field.into()],
            format!("{e:#}"),
            "value_error",
        )]))
    }

    pub(crate) fn plan_decision_request_as(
        &self,
        body: &Value,
        dialect: crate::request::Dialect,
    ) -> Result<RequestPlan, PlanError> {
        if self.manifest.layout != PromptLayout::Canonical {
            if let Some(field) =
                ["media", "assets"].into_iter().find(|field| body.get(*field).is_some_and(|v| !v.is_null()))
            {
                return Err(DecideError::Validation(vec![crate::request::FieldError::new(
                    vec!["body".into(), field.into()],
                    "images and audio require the canonical layout",
                    "value_error",
                )])
                .into());
            }
        }
        for field in ["early_exit", "thinking"] {
            if body.get(field).is_some_and(|v| !v.is_null()) {
                return Err(DecideError::Unsupported(vec![crate::request::FieldError::new(
                    vec!["body".into(), field.into()],
                    format!("{field} is not available in this runtime"),
                    "unsupported_feature",
                )])
                .into());
            }
        }
        let assets =
            if dialect == crate::request::Dialect::TypeSafe { crate::systemone::Assets::parse(body)? } else { None };
        let media = if let Some(assets) = &assets {
            if !self.caps.media_kinds().contains(&crate::media::Kind::Image) {
                return Err(crate::systemone::unsupported(
                    vec!["body".into(), "assets".into()],
                    "this model/backend does not support images; enable media on an omni CUDA model",
                )
                .into());
            }
            self.check_stop()?;
            let mut processed = self
                .caps
                .prepare_media(&assets.entries, "default", self.budget, self.stop)
                .map_err(|e| self.media_error(e, "assets"))?;
            Arc::make_mut(&mut processed).asset_markers = Some(assets.markers.clone());
            Some(processed)
        } else if let Some(entries) = body.get("media").filter(|v| !v.is_null()) {
            if self.caps.media_kinds().is_empty() {
                return Err(DecideError::Unsupported(vec![crate::request::FieldError::new(
                    vec!["body".into(), "media".into()],
                    "this model/backend does not take media; enable media on an omni CUDA model",
                    "unsupported_feature",
                )])
                .into());
            }
            let detail = body.get("detail").map_or(Some("default"), Value::as_str).ok_or_else(|| {
                DecideError::Validation(vec![crate::request::FieldError::new(
                    vec!["body".into(), "detail".into()],
                    "detail must be default or high",
                    "value_error",
                )])
            })?;
            self.check_stop()?;
            Some(
                self.caps
                    .prepare_media(entries, detail, self.budget, self.stop)
                    .map_err(|e| self.media_error(e, "media"))?,
            )
        } else {
            None
        };
        self.check_stop()?;
        let mut req = crate::request::parse_request_for_format(body, dialect, self.manifest.prompt_format)?;
        if let Some(assets) = &assets {
            assets.apply(body, &mut req.items, self.manifest.prompt_format)?;
            if let Some(media) = &media {
                assets.check_prompt_budget(body, &req.items, media, self.manifest.config.max_positions())?;
            }
            if self.manifest.prompt_format != PromptFormat::Two || self.manifest.chat_style == ChatStyle::Llama {
                return Err(crate::systemone::unsupported(
                    vec!["body".into(), "assets".into()],
                    "image assets require an omni prompt format",
                )
                .into());
            }
        }
        if dialect == crate::request::Dialect::TypeSafe && req.model != self.manifest.name {
            return Err(crate::request::api_usage_error(format!("Unknown model: {}", req.model)).into());
        }
        check_capability_with_limit(
            &req.items,
            self.manifest.prompt_format.max_options(),
            self.options.large_choice,
            true,
        )?;
        let ev: Vec<&Item> = req.items.iter().filter(|it| it.evidence).collect();
        if media.is_some() && !ev.is_empty() {
            return Err(DecideError::Unsupported(vec![crate::request::FieldError::new(
                vec!["body".into(), "evidence".into()],
                "evidence is not available with media",
                "unsupported_feature",
            )])
            .into());
        }
        if !ev.is_empty() {
            let mut errs = Vec::new();
            for it in ev {
                let loc = vec!["body".into(), "questions".into(), it.name.clone().into(), "evidence".into()];
                if !self.caps.has_evidence() {
                    errs.push(crate::request::FieldError::new(
                        loc,
                        "evidence is not available for this model (no evidence_head.pt)",
                        "unsupported_feature",
                    ));
                } else if !(2..=crate::prompt::MAX_OPTIONS).contains(&it.options.len()) {
                    errs.push(crate::request::FieldError::new(
                        loc,
                        "evidence needs 2 to 10 options",
                        "unsupported_feature",
                    ));
                }
            }
            if !errs.is_empty() {
                return Err(DecideError::Unsupported(errs).into());
            }
        }
        // Media never truncates. Validate its placeholders before context refusal, as before overflow support.
        let mut plan = if media.is_some() {
            self.plan_items_with_media(req.items, media)?
        } else {
            self.plan_with_overflow(req.items, None)?
        };
        if let (Some(media), crate::architecture::Architecture::Gemma4(cfg)) = (&plan.media, &self.manifest.config) {
            let image_tokens: usize = media
                .items
                .iter()
                .filter(|x| x.kind() == crate::media::Kind::Image)
                .map(crate::media::Input::soft_tokens)
                .sum();
            let audio_tokens: usize = media
                .items
                .iter()
                .filter(|x| x.kind() == crate::media::Kind::Audio)
                .map(crate::media::Input::soft_tokens)
                .sum();
            for p in &plan.prepared {
                for order in &p.orders {
                    let expected_images = order
                        .asset_order
                        .as_ref()
                        .map_or(image_tokens, |indices| indices.iter().map(|&i| media.items[i].soft_tokens()).sum());
                    let expected_audio = if order.asset_order.is_some() { 0 } else { audio_tokens };
                    if order.input_ids.iter().filter(|id| **id == cfg.image_token).count() != expected_images
                        || order.input_ids.iter().filter(|id| **id == cfg.audio_token).count() != expected_audio
                    {
                        return Err(DecideError::Validation(vec![crate::request::FieldError::new(
                            vec!["body".into(), "questions".into(), p.item.name.clone().into()],
                            "extra media placeholders in state, question or options",
                            "value_error",
                        )])
                        .into());
                    }
                }
            }
        }
        if plan.media.is_some() {
            let lengths = self.context_lengths(&plan)?;
            if !lengths.errors.is_empty() {
                return Err(DecideError::Unsupported(lengths.errors).into());
            }
            plan.auto_reverse = lengths.reverse;
        }
        plan.evidence_limit = req.facts_limit;
        plan.dialect = dialect;
        Ok(plan)
    }

    /// Prompts longer than the model's position limit are refused before the forward (422, "maximum context length",
    /// the marker benchmark runners read as a capacity limit), not left to fail inside the backend (500).
    /// With adaptive orders also returns the token lengths of the reversed-order prompts of the probe.
    fn context_lengths(&self, plan: &RequestPlan) -> Result<ContextLengths, PlanError> {
        let limit = self.manifest.config.max_positions();
        let mut errs = Vec::new();
        let mut request_longest = 0;
        let mut reverse = (self.manifest.orders == OptionOrders::Auto).then_some(ReverseTokens::default());
        for p in &plan.prepared {
            self.check_stop()?;
            // a transient probe of every option order (at most three times this question's buffers, plus the
            // tokenizer's transient per prompt byte), returned at the end of this iteration
            let _probe = match self.bounded() {
                Some(b) if self.manifest.orders == OptionOrders::Auto => {
                    let longest = p.orders.iter().map(|o| o.prompt.len() as u64).max().unwrap_or(0);
                    Some(
                        b.reserve(
                            PlanStage::Tokens,
                            prepared_bytes(p)
                                .saturating_mul(3)
                                .saturating_add(longest.saturating_mul(TOKENIZER_BYTES_PER_BYTE + 2)),
                        )?,
                    )
                }
                _ => None,
            };
            let possible = if self.manifest.orders == OptionOrders::Auto {
                Some(prepare_protocol(
                    self.tok,
                    &self.manifest.bos_token,
                    p.item.clone(),
                    self.manifest.prompt_format,
                    self.manifest.chat_style,
                    self.manifest.layout,
                    plan.media.as_deref(),
                    None,
                )?)
            } else {
                None
            };
            if let (Some(r), Some(probe)) = (reverse.as_mut(), possible.as_ref()) {
                // the adaptive round re-runs every order but the first of the full protocol
                for o in probe.orders.iter().skip(1) {
                    r.longest = r.longest.max(o.input_ids.len());
                    r.total = r.total.saturating_add(o.input_ids.len());
                    r.prompts += 1;
                }
            }
            let longest = possible.as_ref().unwrap_or(p).orders.iter().map(|o| o.input_ids.len()).max().unwrap_or(0);
            request_longest = request_longest.max(longest);
            if longest > limit {
                let name = p.item.name.split('#').next().unwrap_or(&p.item.name).to_string();
                if errs.iter().any(|e: &crate::request::FieldError| e.loc.get(2) == Some(&Value::from(name.clone()))) {
                    continue;
                }
                errs.push(context_error(&name, longest, limit, &self.manifest.name));
            }
        }
        if self.truncates() && plan.media.is_none() {
            for q in &plan.questions {
                let QuestionPlan::Groups { item, .. } = q else { continue };
                self.check_stop()?;
                // One bounded heuristic, not a search of score-dependent combinations. Reserve the item clone,
                // index/length pairs and single-option tokenizer scratch before making any of them.
                let scratch = item.options.iter().map(|s| s.len() as u64).max().unwrap_or(0);
                let _items = self.budget.reserve(
                    PlanStage::Items,
                    item_bytes(item)
                        .saturating_mul(2)
                        .saturating_add(item.options.len() as u64 * 32)
                        .saturating_add(scratch.saturating_mul(TOKENIZER_BYTES_PER_BYTE + 8)),
                )?;
                let mut lengths = Vec::with_capacity(item.options.len());
                for (i, option) in item.options.iter().enumerate() {
                    self.check_stop()?;
                    lengths.push((i, self.tok.encode(option)?.len()));
                }
                lengths.sort_unstable_by_key(|&(i, n)| (std::cmp::Reverse(n), i));
                lengths.truncate(crate::prompt::MAX_OPTIONS);
                lengths.sort_unstable_by_key(|&(i, _)| i);
                let mut sub = item.clone();
                sub.keys = lengths.iter().map(|&(i, _)| item.keys[i].clone()).collect();
                sub.options = lengths.iter().map(|&(i, _)| item.options[i].clone()).collect();
                self.check_stop()?;
                let before = self.budget.retained();
                let probe = self.prepare_orders_for(sub, 2, None)?;
                let longest = probe.orders.iter().map(|o| o.input_ids.len()).max().unwrap_or(0);
                request_longest = request_longest.max(longest);
                if longest > limit && !errs.iter().any(|e| e.loc.get(2) == Some(&Value::from(item.name.clone()))) {
                    errs.push(context_error(&item.name, longest, limit, &self.manifest.name));
                }
                drop(probe);
                self.budget.release_attempt(before);
            }
        }
        Ok(ContextLengths { reverse, longest: request_longest, errors: errs })
    }

    /// Only state text is replaced. Parsed items keep their original language, question/options, facts metadata
    /// and task-generated candidate labels. Retokenization has at most two CPU attempts, without GPU calls.
    fn plan_with_overflow(
        &self,
        items: Vec<Item>,
        media: Option<Arc<crate::media::ProcessedMedia>>,
    ) -> Result<RequestPlan, PlanError> {
        let before = self.budget.retained();
        let mut plan = self.plan_items_with_media(items, media)?;
        let mut lengths = self.context_lengths(&plan)?;
        if lengths.errors.is_empty() {
            plan.auto_reverse = lengths.reverse;
            return Ok(plan);
        }
        if !self.truncates() || plan.media.is_some() {
            return Err(DecideError::Unsupported(lengths.errors).into());
        }
        let original_state = &question_item(&plan.questions[0], &plan.prepared).state;
        if plan.questions.iter().any(|q| &question_item(q, &plan.prepared).state != original_state) {
            return Err(DecideError::Unsupported(vec![crate::request::FieldError::new(
                vec!["body".into(), "state".into()],
                "context truncation requires one shared state; plan different states as separate requests",
                "unsupported_feature",
            )])
            .into());
        }
        self.check_stop()?;
        let original_bytes = if self.budget.is_bounded() {
            plan.questions
                .iter()
                .map(|q| item_bytes(question_item(q, &plan.prepared)))
                .sum::<u64>()
                .saturating_add(plan.questions.len() as u64 * std::mem::size_of::<Item>() as u64)
        } else {
            0
        };
        let _originals = self.budget.reserve(PlanStage::Items, original_bytes)?;
        let originals: Vec<Item> = plan.questions.iter().map(|q| question_item(q, &plan.prepared).clone()).collect();
        drop(plan);
        self.budget.release_attempt(before);
        // An overflowing prompt necessarily belongs to a question.
        let state = &originals[0].state;
        // Tokenizer normalization/alignments, ids and offsets stay counted through both attempts. Decode may hold
        // a vocabulary string per token; the reservation also covers the UTF-8 cut buffers.
        let _cut = self
            .budget
            .reserve(PlanStage::Tokens, (state.len() as u64).saturating_mul(TOKENIZER_BYTES_PER_BYTE + 64))?;
        self.check_stop()?;
        let (ids, offsets) = self.tok.encode_char_offsets(state)?;
        let mut excess = 0usize;
        for _ in 0..2 {
            self.check_stop()?;
            excess = excess.saturating_add(lengths.longest - self.manifest.config.max_positions());
            let Some(keep) = ids.len().checked_sub(excess.saturating_add(32)).filter(|k| *k >= 32) else {
                break;
            };
            // Python round uses ties-to-even. A byte-fallback character can occupy several tokens with overlapping
            // offsets; move each boundary inward before decode so it never produces a partial Unicode character.
            let mut head = (keep as f64 * self.options.overflow.head_share()).round_ties_even() as usize;
            let mut tail = keep - head;
            while head > 0 && head < offsets.len() && offsets[head].0 < offsets[head - 1].1 {
                head -= 1;
            }
            while tail > 0
                && tail < offsets.len()
                && offsets[offsets.len() - tail].0 < offsets[offsets.len() - tail - 1].1
            {
                tail -= 1;
            }
            if head + tail < 32 {
                break;
            }
            let mut cut = self.tok.decode(&ids[..head])?;
            cut.push_str("\n[…]\n");
            if tail > 0 {
                cut.push_str(&self.tok.decode(&ids[ids.len() - tail..])?);
            }
            if originals.iter().any(|it| it.evidence) {
                return Err(DecideError::Unsupported(vec![crate::request::FieldError::new(
                    vec!["body".into(), "evidence".into()],
                    "evidence is not available for a state truncated to the context (shorten the state)",
                    "unsupported_feature",
                )])
                .into());
            }
            self.check_stop()?;
            // Reserve before cloning (including the transient original state inside each clone).
            let copies = self.budget.reserve(
                PlanStage::Items,
                original_bytes.saturating_add((cut.len() as u64).saturating_mul(originals.len() as u64)),
            )?;
            let items = originals
                .iter()
                .map(|it| {
                    let mut it = it.clone();
                    it.state.clone_from(&cut);
                    it
                })
                .collect();
            plan = self.plan_items_with_media(items, None)?;
            lengths = self.context_lengths(&plan)?;
            if lengths.errors.is_empty() {
                plan.auto_reverse = lengths.reverse;
                plan.truncated = Some(crate::overflow::Truncation {
                    state_tokens: ids.len(),
                    head_tokens: head,
                    tail_tokens: tail,
                    dropped_tokens: ids.len() - head - tail,
                    head_share: self.options.overflow.head_share(),
                });
                copies.retain();
                return Ok(plan);
            }
            drop(plan);
            self.budget.release_attempt(before);
        }
        Err(DecideError::Unsupported(lengths.errors).into())
    }

    /// Plan validated items (see [`PlanContext::plan_request`]). Truncation requires one shared request state.
    pub fn plan_items(&self, items: Vec<Item>) -> Result<RequestPlan, PlanError> {
        if self.truncates() {
            self.plan_with_overflow(items, None)
        } else {
            // Preserve the low-level wrapper: refusal/context checks belong to plan_request, without adding
            // reversed-order tokenization to existing callers of plan_items.
            self.plan_items_with_media(items, None)
        }
    }

    fn plan_items_with_media(
        &self,
        items: Vec<Item>,
        media: Option<Arc<crate::media::ProcessedMedia>>,
    ) -> Result<RequestPlan, PlanError> {
        let mut plan = RequestPlan {
            tasks: None,
            prepared: Vec::new(),
            questions: Vec::new(),
            evidence_limit: None,
            dialect: Default::default(),
            calibration_unvalidated: self.manifest.calibration_unvalidated.is_some(),
            media,
            auto_reverse: None,
            truncated: None,
            task_bytes: 0,
        };
        let budget = self.bounded();
        for mut it in items {
            self.check_stop()?;
            it.prompt_format = self.manifest.prompt_format;
            if let crate::request::Ext::Act(a) = &mut it.ext {
                a.calibrated = self.manifest.has_calibration() && self.manifest.calibration_unvalidated.is_none();
                a.threshold = a.max_error.as_ref().and_then(|k| self.manifest.certified(k, a.base));
            }
            if let crate::request::Ext::Tool(t) = &mut it.ext {
                t.threshold =
                    t.max_error.as_deref().and_then(|k| self.manifest.certified(k, crate::request::QType::Choice));
            }
            let keys = it.keys.clone();
            if let Some(m) = it.multi_spec_mut() {
                if let Some(fallback) = m.rules.fallback.filter(|_| m.rules.fallback_threshold.is_none()) {
                    let id = crate::sets::label_set_id(&keys, &m.labels, &keys[fallback])?;
                    if let Some(threshold) = self
                        .manifest
                        .multi_calibration
                        .get(&id)
                        .and_then(|v| v.get("fallback_threshold"))
                        .and_then(Value::as_f64)
                    {
                        if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
                            return Err(anyhow::anyhow!("calibration: invalid fallback threshold").into());
                        }
                        m.rules.fallback_threshold = Some(threshold);
                        m.rules.fallback_threshold_source = Some("calibration");
                    }
                }
            }
            if let Some(m) = it.multi_spec() {
                // one yes/no branch per label, each an ordinary noul readout
                let mut parts = Vec::with_capacity(m.labels.len());
                for (k, label) in m.labels.iter().enumerate() {
                    let copy_reservation = match budget {
                        Some(b) => Some(b.reserve(PlanStage::Items, item_bytes(&it))?),
                        None => None,
                    };
                    let mut sub = it.clone();
                    sub.name = format!("{}#{k}", it.name);
                    sub.kind = crate::request::QType::Noul;
                    sub.keys = vec!["true".into(), "false".into()];
                    sub.question = crate::request::multi_question(it.lang, &it.question, label);
                    sub.ext = crate::request::Ext::None;
                    parts.push(plan.prepared.len());
                    let orders = self.manifest.orders.count(sub.options.len());
                    plan.prepared.push(self.prepare_orders_for(sub, orders, plan.media.as_deref())?);
                    if let Some(r) = copy_reservation {
                        r.retain();
                    }
                }
                plan.questions.push(QuestionPlan::Multi { item: it, parts });
                continue;
            }
            if it.options.len() == 1 {
                // one option or level (choice, score): probability 1 without the model, as the TypeSafe API answers
                plan.questions.push(QuestionPlan::Fixed(it));
            } else if it.options.len() <= self.manifest.prompt_format.max_options() {
                plan.questions.push(QuestionPlan::Direct(plan.prepared.len()));
                let orders = self.manifest.orders.count(it.options.len());
                plan.prepared.push(self.prepare_orders_for(it, orders, plan.media.as_deref())?);
            } else {
                let groups =
                    crate::large_choice::groups(it.options.len(), self.options.large_choice_strategy.partitions);
                let mut parts = Vec::with_capacity(groups.len());
                for g in &groups {
                    let copy_reservation = match budget {
                        Some(b) => Some(b.reserve(PlanStage::Items, item_bytes(&it))?),
                        None => None,
                    };
                    let mut sub = it.clone();
                    sub.keys = g.iter().map(|&i| it.keys[i].clone()).collect();
                    sub.options = g.iter().map(|&i| it.options[i].clone()).collect();
                    parts.push(plan.prepared.len());
                    plan.prepared.push(self.prepare_orders(sub, self.options.large_choice_strategy.screen_orders)?);
                    if let Some(r) = copy_reservation {
                        r.retain();
                    }
                }
                plan.questions.push(QuestionPlan::Groups { item: it, groups, parts });
            }
        }
        Ok(plan)
    }
}

struct ContextLengths {
    reverse: Option<ReverseTokens>,
    longest: usize,
    errors: Vec<crate::request::FieldError>,
}

pub(crate) fn context_error(name: &str, longest: usize, limit: usize, model: &str) -> crate::request::FieldError {
    crate::request::FieldError::new(
        vec!["body".into(), "questions".into(), name.into()],
        format!(
            "the prompt has {longest} tokens, more than the maximum context length of {limit} tokens of model {model}"
        ),
        "context_length_exceeded",
    )
}

fn question_item<'a>(q: &'a QuestionPlan, prepared: &'a [Prepared]) -> &'a Item {
    match q {
        QuestionPlan::Direct(i) => &prepared[*i].item,
        QuestionPlan::Fixed(it) | QuestionPlan::Groups { item: it, .. } | QuestionPlan::Multi { item: it, .. } => it,
    }
}

/// Owned offline planner: its own tokenizer (loaded once per CPU worker or model), the model manifest, planning
/// options and a capability snapshot. No backend and no GPU.
pub struct RequestPlanner {
    manifest: ModelManifest,
    tok: BasalTokenizer,
    options: PlanOptions,
    caps: PlanCaps,
}

/// An offline planner could not be created.
#[derive(Debug, thiserror::Error)]
pub enum PlannerLoadError {
    #[error("loading the tokenizer of {model}")]
    Tokenizer {
        model: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("bos token {0} is not in the vocabulary")]
    MissingBos(String),
}

impl RequestPlanner {
    pub fn load(manifest: ModelManifest, options: PlanOptions, caps: PlanCaps) -> Result<Self, PlannerLoadError> {
        let tok = BasalTokenizer::load_for(&manifest.dir.join("tokenizer.json"), manifest.chat_style)
            .map_err(|e| PlannerLoadError::Tokenizer { model: manifest.name.clone(), source: e.into() })?;
        if tok.token_id(&manifest.bos_token).is_none() {
            return Err(PlannerLoadError::MissingBos(manifest.bos_token.clone()));
        }
        Ok(Self { manifest, tok, options, caps })
    }

    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    pub fn options(&self) -> PlanOptions {
        self.options
    }

    pub fn caps(&self) -> &PlanCaps {
        &self.caps
    }

    pub fn context<'a>(&'a self, budget: &'a PlanBudget, stop: Option<&'a StopToken>) -> PlanContext<'a> {
        PlanContext::new(&self.manifest, &self.tok, self.options, &self.caps, budget, stop)
    }

    /// Plan one record body under `budget` (the record's slot), observing `stop` between stages.
    pub fn plan(
        &self,
        body: &Value,
        dialect: crate::request::Dialect,
        budget: &PlanBudget,
        stop: Option<&StopToken>,
    ) -> Result<RequestPlan, PlanError> {
        self.context(budget, stop).plan_request_as(body, dialect)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Estimators

/// Conservative heap bytes of a `serde_json::Value` (with `preserve_order` and `arbitrary_precision`) holding `nodes`
/// values (each array element, object member and scalar counts once) and `text_bytes` bytes of strings, keys and
/// number text: every container may hold twice its length after growth (Vec, IndexMap entries, String), an object
/// member also costs its key `String`, its hash and an index slot.
pub fn json_value_bound(nodes: u64, text_bytes: u64) -> u64 {
    let value = std::mem::size_of::<Value>() as u64;
    let member = value + std::mem::size_of::<String>() as u64 + 8;
    nodes.saturating_mul(2 * member + 2 * 9).saturating_add(text_bytes.saturating_mul(2)).saturating_add(4096)
}

/// Upper bound of `s` as a JSON string (control characters as `\u00XX`, quotes).
fn escaped(s: &str) -> u64 {
    (s.len() as u64).saturating_mul(6).saturating_add(2)
}

/// Upper bound of the JSON serialisation of `v` (strings escaped, numbers as written).
fn json_bound(v: &Value) -> u64 {
    match v {
        Value::String(s) => escaped(s),
        Value::Array(a) => a.iter().fold(2u64, |n, v| n.saturating_add(json_bound(v)).saturating_add(1)),
        Value::Object(m) => m
            .iter()
            .fold(2u64, |n, (k, v)| n.saturating_add(escaped(k)).saturating_add(json_bound(v)).saturating_add(2)),
        Value::Number(n) => n.to_string().len() as u64,
        _ => 5,
    }
}

/// Width of one serialised probability or score (`f64` shortest representation, sign and exponent).
const NUMBER: u64 = 32;

/// Upper bound of the native (`plan_response`) JSON of `plan` for `model`, at most [`MAX_RESULT_BYTES`]. Plain
/// choice, noul and score answers (also grouped large choices) are bounded from the plan shape; tasks, `multi`,
/// `act`, `tool` and evidence answers have no verified estimate and get [`MAX_RESULT_BYTES`]. The record envelope of
/// a batch line (custom_id, ids, status) is added by the caller.
pub fn result_bound(model: &str, plan: &RequestPlan) -> u64 {
    if plan.tasks.is_some() {
        return MAX_RESULT_BYTES;
    }
    let mut bytes = 512u64.saturating_add(escaped(model));
    if plan.media.is_some() {
        bytes = bytes.saturating_add(256);
    }
    if plan.truncated.is_some() {
        bytes = bytes.saturating_add(256);
    }
    for q in &plan.questions {
        let item = match q {
            QuestionPlan::Direct(i) => &plan.prepared[*i].item,
            QuestionPlan::Fixed(item) | QuestionPlan::Groups { item, .. } => item,
            QuestionPlan::Multi { .. } => return MAX_RESULT_BYTES,
        };
        if item.evidence || !matches!(item.ext, crate::request::Ext::None) {
            return MAX_RESULT_BYTES;
        }
        let keys: u64 = item.keys.iter().map(|k| escaped(k).saturating_add(NUMBER + 2)).sum();
        // legend of a score: the levels (TypeSafe) or the option texts (upstream)
        let legend: u64 = item
            .keys
            .iter()
            .map(|k| escaped(k).saturating_add(2))
            .chain(item.legend.iter().map(json_bound))
            .chain(item.options.iter().map(|o| escaped(o)))
            .sum();
        let answer = 256u64
            .saturating_add(escaped(&item.name))
            .saturating_add(keys.saturating_mul(2))
            .saturating_add(legend)
            .saturating_add(NUMBER * 3);
        bytes = bytes.saturating_add(answer);
        if bytes >= MAX_RESULT_BYTES {
            return MAX_RESULT_BYTES;
        }
    }
    bytes.min(MAX_RESULT_BYTES)
}

/// Heap bytes of an item (strings, keys, options, legend) with a fixed allowance for its extension.
pub(crate) fn item_bytes(it: &Item) -> u64 {
    let strings: u64 =
        [it.name.capacity(), it.state.capacity(), it.question.capacity()].iter().map(|n| *n as u64).sum();
    let lists: u64 =
        it.keys.iter().chain(&it.options).map(|s| s.capacity() as u64 + std::mem::size_of::<String>() as u64).sum();
    let legend: u64 = it.legend.iter().map(|v| json_value_bound(1, json_bound(v))).sum();
    // `ext` (multi rules, act/tool specs) holds labels and keys of the same item: bounded by its strings again
    std::mem::size_of::<Item>() as u64 + strings.saturating_mul(2) + lists.saturating_mul(2) + legend + 4096
}

/// Upper bound of a packed row of `orders` (ids, positions, nodes, parents and readouts per token).
fn packed_bound(orders: &[crate::engine::OrderJob]) -> u64 {
    let tokens: u64 = orders.iter().map(|o| o.input_ids.len() as u64).sum();
    tokens.saturating_mul(4 * 4).saturating_mul(2).saturating_add(orders.len() as u64 * 16).saturating_add(256)
}

/// Heap bytes held by a prepared question, measured from the capacities of its buffers.
fn prepared_bytes(p: &Prepared) -> u64 {
    let orders: u64 = p
        .orders
        .iter()
        .map(|o| {
            (o.perm.capacity() * 8
                + o.prompt.capacity()
                + o.input_ids.capacity() * 4
                + o.letter_ids.capacity() * 4
                + o.asset_order.as_ref().map_or(0, |a| a.capacity() * 8)
                + std::mem::size_of::<crate::engine::OrderJob>()) as u64
        })
        .sum();
    let packed = (p.packed.ids.capacity() * 4
        + p.packed.pos.capacity() * 4
        + p.packed.node.capacity() * 4
        + p.packed.parent.capacity() * 4
        + p.packed.last.capacity() * 8) as u64;
    item_bytes(&p.item) + orders + packed + std::mem::size_of::<Prepared>() as u64
}

/// Heap bytes of processed media features.
fn media_bytes(m: &crate::media::ProcessedMedia) -> u64 {
    m.items
        .iter()
        .map(|x| match x {
            crate::media::Input::Image(i) => (i.patches.capacity() * 4 + i.positions.capacity() * 16) as u64,
            crate::media::Input::Audio(a) => (a.features.capacity() * 4 + a.valid.capacity()) as u64,
        })
        .sum::<u64>()
        .saturating_add(m.asset_markers.as_ref().map_or(0, |v| v.iter().map(|s| s.capacity() as u64 + 24).sum()))
}

/// Host bytes a built plan holds until it is answered: its prepared questions, question plans and media features
/// (counted once: the plan holds them in an `Arc`), measured from buffer capacities, plus a fixed allowance for the
/// task plan of native tasks.
pub fn retained_bound(plan: &RequestPlan) -> u64 {
    let prepared: u64 = plan.prepared.iter().map(prepared_bytes).sum();
    let questions: u64 = plan
        .questions
        .iter()
        .map(|q| match q {
            QuestionPlan::Direct(_) => 16,
            QuestionPlan::Fixed(it) => item_bytes(it),
            QuestionPlan::Multi { item, parts } => item_bytes(item) + parts.capacity() as u64 * 8,
            QuestionPlan::Groups { item, groups, parts } => {
                item_bytes(item)
                    + groups.iter().map(|g| g.capacity() as u64 * 8 + 24).sum::<u64>()
                    + parts.capacity() as u64 * 8
            }
        })
        .sum();
    let tasks = if plan.tasks.is_some() { plan.task_bytes.max(1 << 20) } else { 0 };
    prepared
        .saturating_add(questions)
        .saturating_add(plan.media.as_deref().map_or(0, media_bytes))
        .saturating_add(tasks)
        .saturating_add(std::mem::size_of::<RequestPlan>() as u64)
}

/// Per-call host staging of a backend (inputs it builds on the host and outputs it reads back), as linear
/// coefficients of the forward's shape. Stateless: derived from the model configuration once, sendable to planners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostStaging {
    pub fixed: u64,
    pub per_token: u64,
    pub per_row: u64,
    pub per_readout: u64,
    pub per_letter: u64,
    /// Per `f32` of media features (host copies before upload).
    pub per_media_value: u64,
    /// Per prompt token of an evidence question (hidden rows read back for the evidence head).
    pub per_evidence_token: u64,
}

/// Shape of the work of one plan for [`HostStaging::bound`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StagingShape {
    pub tokens: u64,
    pub rows: u64,
    pub readouts: u64,
    pub letters: u64,
    pub media_values: u64,
    pub evidence_tokens: u64,
}

impl HostStaging {
    pub fn bound(&self, s: StagingShape) -> u64 {
        self.fixed
            .saturating_add(self.per_token.saturating_mul(s.tokens))
            .saturating_add(self.per_row.saturating_mul(s.rows))
            .saturating_add(self.per_readout.saturating_mul(s.readouts))
            .saturating_add(self.per_letter.saturating_mul(s.letters))
            .saturating_add(self.per_media_value.saturating_mul(s.media_values))
            .saturating_add(self.per_evidence_token.saturating_mul(s.evidence_tokens))
    }
}

/// Token lengths of the reversed-order prompts of a plan with adaptive option orders.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReverseTokens {
    pub longest: usize,
    pub total: usize,
    pub prompts: usize,
}

/// Engine limits that bound the forwards of a plan ([`crate::Engine::shape_limits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeLimits {
    pub batching: crate::Batching,
    pub tree_max_tokens: usize,
    pub forward_max_tokens: usize,
    /// The model's position limit: planned prompts and actual dependent orders are checked before their forwards.
    pub max_positions: usize,
    /// Most options of one readout of the prompt format (later grouped rounds included).
    pub max_options: usize,
    /// The evidence forward tokenizes the same text as the as-written option order (`ChatStyle::Llama`, where
    /// `render_model` is `render_format`).
    pub evidence_matches_prompt: bool,
    /// Adaptive option orders: an uncertain first-round answer re-runs its reversed order.
    pub adaptive_orders: bool,
}

/// A plan whose device shape cannot be bounded before execution.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("device work shape not bounded: {0}")]
pub struct UnmodelledShape(pub String);

/// One media item of a media plan, as the encoders receive it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaItemShape {
    Image { patch_width: u64, patch_height: u64, patch_dim: u64, soft_tokens: u64, padded_patches: u64 },
    Audio { frames: u64, feature_dim: u64, soft_tokens: u64 },
}

impl MediaItemShape {
    pub fn soft_tokens(&self) -> u64 {
        match self {
            Self::Image { soft_tokens, .. } | Self::Audio { soft_tokens, .. } => *soft_tokens,
        }
    }
}

/// A media plan: its encoder inputs and its longest prompt (media plans run one at a time, one prompt per forward).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaPlanShape {
    pub items: Vec<MediaItemShape>,
    pub longest_prompt: u64,
}

/// Evidence forwards: one plain causal forward of `tokens` with `readouts` hidden rows read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvidenceShape {
    pub tokens: u64,
    pub readouts: u64,
}

/// Upper bounds of the device work of one `Engine::run_plans` micro-batch, for every round it can run.
///
/// - Packed text forwards (Llama and Intel run packed rows): `forward_tokens` bounds the tokens of any one forward
///   (one packed row on Intel) under the engine's batching rule: `Tree` cuts rows at `tree_max_tokens` (or one
///   longer question) and groups rows up to `forward_max_tokens`; `Budget` chunks stay within
///   [`crate::pack::TOKEN_BUDGET`] (or one longer question); `Single` runs one question per forward. A forward
///   never holds more than all questions of the micro-batch. `forward_readouts` and `forward_letters` bound the
///   readouts and distinct letter ids of one forward.
/// - Single prompts (Gemma runs every option-order prompt by itself or in buckets of short prompts):
///   `longest_prompt` bounds any executed prompt.
/// - Adaptive option orders: the reversed prompts measured by the context check.
/// - Rounds planned at run time (grouped large choices after the first round, native task continuations) are not
///   known in advance: their bound is `worst_case_rounds`, derived from the engine limits and `max_positions`
///   (a question has at most two option orders of at most `max_positions` tokens).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceWorkShape {
    pub forward_tokens: u64,
    pub forward_readouts: u64,
    pub forward_letters: u64,
    pub longest_prompt: u64,
    /// Most letter ids of one prompt's readout.
    pub prompt_letters: u64,
    /// Short prompts (at most 512 tokens) may run in buckets.
    pub short_prompts: bool,
    pub evidence: Option<EvidenceShape>,
    pub media: Vec<MediaPlanShape>,
    /// Later rounds are bounded by the engine limits, not by planned prompts.
    pub worst_case_rounds: bool,
}

impl DeviceWorkShape {
    /// Bounds of running `plans` together (one micro-batch) under `limits`.
    pub fn from_plans(plans: &[&RequestPlan], limits: &ShapeLimits) -> Result<Self, UnmodelledShape> {
        let as_u64 = |n: usize| n as u64;
        let mut total = 0u64;
        let mut largest_question = 0u64;
        let mut readouts = 0u64;
        let mut letters = 0u64;
        let mut longest_prompt = 0u64;
        let mut prompt_letters = 0u64;
        let mut short_prompts = false;
        let mut rev = ReverseTokens::default();
        let mut evidence: Option<EvidenceShape> = None;
        let mut media = Vec::new();
        let mut later_rounds = false;
        for plan in plans {
            later_rounds |=
                plan.tasks.is_some() || plan.questions.iter().any(|q| matches!(q, QuestionPlan::Groups { .. }));
            let mut plan_longest = 0u64;
            for p in &plan.prepared {
                let question: u64 = p.orders.iter().map(|o| as_u64(o.input_ids.len())).sum();
                total = total.saturating_add(question);
                largest_question = largest_question.max(question);
                readouts = readouts.saturating_add(as_u64(p.orders.len()));
                letters = letters.saturating_add(p.orders.iter().map(|o| as_u64(o.letter_ids.len())).sum());
                for o in &p.orders {
                    prompt_letters = prompt_letters.max(as_u64(o.letter_ids.len()));
                    plan_longest = plan_longest.max(as_u64(o.input_ids.len()));
                    short_prompts |= o.input_ids.len() <= 512;
                }
                if p.item.evidence {
                    if !limits.evidence_matches_prompt {
                        return Err(UnmodelledShape(
                            "evidence prompts of this chat style are tokenized only at run time".into(),
                        ));
                    }
                    // decision::orders: the first option order is the as-written one
                    let n = p.orders.first().map_or(0, |o| as_u64(o.input_ids.len()));
                    let e = evidence.get_or_insert(EvidenceShape { tokens: 0, readouts: 0 });
                    e.tokens = e.tokens.max(n);
                    e.readouts = e.readouts.max(n);
                }
            }
            if limits.adaptive_orders && plan.auto_reverse.is_none() && !plan.prepared.is_empty() {
                return Err(UnmodelledShape(
                    "adaptive option orders without the reversed-order lengths of the context check (a plan not \
                     built by plan_request)"
                        .into(),
                ));
            }
            if let Some(r) = plan.auto_reverse {
                rev.longest = rev.longest.max(r.longest);
                rev.total = rev.total.saturating_add(r.total);
                rev.prompts = rev.prompts.saturating_add(r.prompts);
                prompt_letters = prompt_letters.max(as_u64(limits.max_options));
                plan_longest = plan_longest.max(as_u64(r.longest));
                short_prompts |= r.prompts > 0 && r.longest <= 512;
            }
            if let Some(m) = plan.media.as_deref() {
                let items = m
                    .items
                    .iter()
                    .map(|x| match x {
                        crate::media::Input::Image(i) => MediaItemShape::Image {
                            patch_width: as_u64(i.patch_width),
                            patch_height: as_u64(i.patch_height),
                            patch_dim: as_u64(i.patch_dim),
                            soft_tokens: as_u64(i.soft_tokens),
                            padded_patches: as_u64(i.positions.len()),
                        },
                        crate::media::Input::Audio(a) => MediaItemShape::Audio {
                            frames: as_u64(a.valid.len()),
                            feature_dim: as_u64(a.feature_dim),
                            soft_tokens: as_u64(a.soft_tokens),
                        },
                    })
                    .collect();
                media.push(MediaPlanShape { items, longest_prompt: plan_longest });
            }
            longest_prompt = longest_prompt.max(plan_longest);
        }
        let rule = |total: u64, largest: u64| -> u64 {
            let cap = match limits.batching {
                crate::Batching::Tree => {
                    as_u64(limits.forward_max_tokens).max(as_u64(limits.tree_max_tokens)).max(largest)
                }
                crate::Batching::Budget => as_u64(crate::pack::TOKEN_BUDGET).max(largest),
                crate::Batching::Single => largest,
            };
            total.min(cap)
        };
        let mut forward_tokens = rule(total, largest_question).max(rule(as_u64(rev.total), as_u64(rev.longest)));
        let mut forward_readouts = readouts.max(as_u64(rev.prompts)).min(forward_tokens.max(1));
        let mut forward_letters = letters.saturating_add(as_u64(rev.prompts * limits.max_options));
        if later_rounds {
            let positions = as_u64(limits.max_positions);
            let question = positions.saturating_mul(2);
            let worst = match limits.batching {
                crate::Batching::Tree => {
                    as_u64(limits.forward_max_tokens).max(as_u64(limits.tree_max_tokens)).max(question)
                }
                crate::Batching::Budget => as_u64(crate::pack::TOKEN_BUDGET).max(question),
                crate::Batching::Single => question,
            };
            forward_tokens = forward_tokens.max(worst);
            forward_readouts = forward_readouts.max(forward_tokens);
            forward_letters = forward_letters.max(forward_readouts.saturating_mul(as_u64(limits.max_options)));
            longest_prompt = longest_prompt.max(positions);
            prompt_letters = prompt_letters.max(as_u64(limits.max_options));
            short_prompts = true;
            for m in &mut media {
                m.longest_prompt = m.longest_prompt.max(positions);
            }
            if plans.iter().any(|p| p.tasks.is_some()) {
                // a continuation may ask for evidence of a prompt up to the position limit
                evidence = Some(EvidenceShape { tokens: positions, readouts: positions });
            }
        }
        Ok(Self {
            forward_tokens,
            forward_readouts,
            forward_letters,
            longest_prompt,
            prompt_letters,
            short_prompts,
            evidence,
            media,
            worst_case_rounds: later_rounds,
        })
    }
}

/// Peak transient device bytes of one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBytes {
    /// The device ordinal of the stage.
    pub device: usize,
    pub bytes: u64,
    /// Distinct allocations counted in `bytes`: the caller adds its allocator's per-allocation granularity.
    pub allocations: u64,
}

/// A source-derived upper bound of the transient device memory of one micro-batch.
///
/// Included: every tensor or buffer a call of the backend allocates for that work (activations, attention
/// inputs and score tiles, K/V of the forward and of chunked prefill, per-use widened or dequantized weights,
/// boundary copies of a layer split, readout and evidence tensors, media encoder activations and the encoded
/// media session), counted as if no temporary of one layer (or encoder block) were freed before the next.
///
/// Not included, to be accounted separately by the caller: resident weights and pinned prefix K/V shared with the
/// online engine, cuBLASLt / cuBLAS workspaces (one per stage, allocated at load and shared by forks), allocator
/// retention and granularity (CUDA memory pools, the Metal buffer cache, Vulkan memory blocks), the online reserve,
/// and fragmentation. The constants are allocation sizes read from the source, not measurements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceWorkEstimate {
    pub devices: Vec<DeviceBytes>,
    /// What the estimate assumes about the backend configuration (logged with it).
    pub basis: String,
}

impl DeviceWorkEstimate {
    /// Largest bytes over the devices.
    pub fn max_bytes(&self) -> u64 {
        self.devices.iter().map(|d| d.bytes).max().unwrap_or(0)
    }
}

/// Transient device memory of a micro-batch on a backend ([`crate::Backend::device_work_bound`]). `Unknown`
/// keeps background disabled for that configuration or shape, with the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceWorkBound {
    Estimate(DeviceWorkEstimate),
    Unknown(String),
}

/// Neutral cost figures of a plan for admission and scheduling (no backend needed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCost {
    /// Packed tokens of the plan's prompts in one shared-prefix row (the server's admission size).
    pub packed_tokens: usize,
    /// Sum of the prompt tokens of every option order (`usage.input_tokens` of the first round).
    pub prompt_tokens: usize,
    pub longest_prompt: usize,
    /// Option-order prompts (readouts) of the first round.
    pub readouts: usize,
    /// Letter ids read over all readouts.
    pub letters: usize,
    pub prepared: usize,
    pub questions: usize,
    pub media: bool,
    pub media_values: usize,
    /// The plan has a dependent continuation round (native tasks) or later grouped rounds.
    pub multi_round: bool,
}

pub fn plan_cost(plan: &RequestPlan) -> PlanCost {
    let mut size = TrieSize::default();
    size.insert(
        &plan.prepared.iter().flat_map(|p| p.orders.iter().map(|o| o.input_ids.as_slice())).collect::<Vec<_>>(),
    );
    let orders = plan.prepared.iter().flat_map(|p| &p.orders);
    PlanCost {
        packed_tokens: size.tokens,
        prompt_tokens: orders.clone().map(|o| o.input_ids.len()).sum(),
        longest_prompt: orders.clone().map(|o| o.input_ids.len()).max().unwrap_or(0),
        readouts: orders.clone().count(),
        letters: orders.map(|o| o.letter_ids.len()).sum(),
        prepared: plan.prepared.len(),
        questions: plan.questions.len(),
        media: plan.media.is_some(),
        media_values: plan.media.as_deref().map_or(0, |m| {
            m.items
                .iter()
                .map(|x| match x {
                    crate::media::Input::Image(i) => i.patches.len(),
                    crate::media::Input::Audio(a) => a.features.len(),
                })
                .sum()
        }),
        multi_round: plan.tasks.is_some() || plan.questions.iter().any(|q| matches!(q, QuestionPlan::Groups { .. })),
    }
}

/// Host bytes the execution of `plan` may add on top of [`retained_bound`], or `None` when the backend reports no
/// host staging bound (background then stays disabled for it). Covers the clones `Engine::run_decision_plans` and
/// `Engine::run` make of prepared questions and packed rows (tree rows clone them again), letter id lists, logits
/// and item results, the adaptive reversed order, later grouped rounds, `plan_response`, the continuation body and
/// plan of native tasks (from `input_bytes` and `result_bound`) and the backend's host staging.
pub fn exec_bound(
    plan: &RequestPlan,
    input_bytes: u64,
    result_bound: u64,
    staging: Option<&HostStaging>,
) -> Option<u64> {
    let staging = staging?;
    let cost = plan_cost(plan);
    let prepared: u64 = plan.prepared.iter().map(prepared_bytes).sum();
    // run_decision_plans clones every prepared question; Tree batching clones them again for tree rows and packs
    // rows of at most all tokens; adaptive orders re-prepare at most every question once more.
    let clones = prepared.saturating_mul(3);
    let rows = (cost.prompt_tokens as u64).saturating_mul(16).saturating_mul(2);
    let letters = (cost.letters as u64).saturating_mul(4 + 4 + 8 * 3);
    let results = (cost.readouts as u64).saturating_mul(256).saturating_add(cost.prepared as u64 * 512);
    // grouped rounds: at most one more prepared copy per group of each later round
    let groups: u64 = plan
        .questions
        .iter()
        .map(|q| match q {
            QuestionPlan::Groups { item, .. } => item_bytes(item).saturating_mul(item.options.len() as u64 * 2),
            _ => 0,
        })
        .sum();
    let response = json_value_bound(1 + cost.questions as u64 * 64, result_bound);
    let continuation = if plan.tasks.is_some() {
        let body =
            json_value_bound(input_bytes.saturating_add(result_bound) / 2, input_bytes.saturating_add(result_bound));
        // A truncated first round can be much smaller than the continuation's ORIGINAL state. `task_bytes`
        // reserves 16 × the original expansion; another factor of 32 covers tokenizer/decoder scratch, the
        // two state copies of state_twice, auto probes and live retry buffers of continuation planning.
        body.saturating_add(retained_bound(plan))
            .saturating_add(response)
            .saturating_add(plan.task_bytes.saturating_mul(32))
    } else {
        0
    };
    let shape = StagingShape {
        tokens: cost.prompt_tokens as u64,
        rows: cost.readouts as u64,
        readouts: cost.readouts as u64,
        letters: cost.letters as u64,
        media_values: cost.media_values as u64,
        evidence_tokens: plan
            .prepared
            .iter()
            .filter(|p| p.item.evidence)
            .map(|p| p.orders.iter().map(|o| o.input_ids.len() as u64).max().unwrap_or(0))
            .sum(),
    };
    Some(
        clones
            .saturating_add(rows)
            .saturating_add(letters)
            .saturating_add(results)
            .saturating_add(groups)
            .saturating_add(response)
            .saturating_add(continuation)
            .saturating_add(staging.bound(shape)),
    )
}

/// The record's host bound (`retained_bound + exec_bound`) reserved in the execution pool before a claim.
pub fn host_bound(retained: u64, exec: Option<u64>) -> Option<u64> {
    exec.map(|e| retained.saturating_add(e))
}
