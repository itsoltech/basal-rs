//! Request pipeline independent of the GPU backend: items -> prompts and token ids of both option orders -> packed
//! rows -> backend letter logits -> averaged, calibrated distributions -> System One answers.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::decision::{self, answer, calibrate, canonical_average, softmax};
use crate::execution::{BackgroundSupport, ExecutionError, StopToken, WorkClass};
use crate::manifest::ModelManifest;
use crate::pack::{chunks, pack, pack_tree, Packed, TrieSize};
use crate::planner::{
    BackendCaps, HostStaging, PlanBudget, PlanCaps, PlanContext, PlanError, PlanOptions, PlanStage, PlannerLoadError,
    RequestPlanner, TOKENIZER_BYTES_PER_BYTE,
};
use crate::prompt::{render_model, ChatStyle, OptionOrders, PromptFormat};
use crate::request::{DecideError, Item};
use crate::tokenize::BasalTokenizer;

/// A forward implementation: packed rows in, letter logits (float32, before softmax) of every readout out.
pub trait Backend {
    /// Precision, device and kernel choices, recorded in reports.
    fn describe(&self) -> Value;
    /// `rows[r]` is one packed row, `letters[r][k]` the letter token ids of its readout `rows[r].last[k]`.
    /// Returns `[row][readout][letter]`.
    fn letter_logits(&mut self, rows: &[&Packed], letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>>;
    /// Precompute a token prefix that rows may start with (positions 0..len). Exact: a row uses it only when its
    /// token ids begin with exactly these ids. Default: not supported, ignored.
    fn add_prefix(&mut self, _ids: &[u32]) -> Result<()> {
        Ok(())
    }
    /// The model has an evidence head (basal-1.5 `evidence_head.pt`).
    fn has_evidence(&self) -> bool {
        false
    }
    /// Evidence head on the plain causal forward of `ids`: start / end log-probabilities over the tokens `t0..t1`,
    /// queried by the final state of the last token.
    fn evidence_scores(&mut self, _ids: &[u32], _t0: usize, _t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        anyhow::bail!("this backend has no evidence head")
    }
    /// A second backend on the same device sharing the weights and precomputed template prefixes (its own state
    /// cache), for a second engine thread. Default: not supported.
    fn fork(&self) -> Result<Self>
    where
        Self: Sized,
    {
        anyhow::bail!("this backend cannot be forked")
    }
    /// Share the device with another engine through `gate` (urgent lane, or preemptible between layers).
    /// Default: ignored.
    fn set_gate(&mut self, _gate: std::sync::Arc<crate::gate::Gate>, _urgent: bool) {}
    /// Share device `device` (an ordinal this backend runs on) with other engines through `gate`. A backend placed on
    /// several devices takes one gate per device and holds at most one of them at a time. Default: [`Backend::set_gate`]
    /// (a backend on one device).
    fn set_device_gate(&mut self, device: usize, gate: std::sync::Arc<crate::gate::Gate>, urgent: bool) -> Result<()> {
        let _ = device;
        self.set_gate(gate, urgent);
        Ok(())
    }
    /// Readiness of this backend's device, shared with its forks. Default: not reported.
    fn device_health(&self) -> Option<DeviceHealth> {
        None
    }
    fn media_kinds(&self) -> Vec<crate::media::Kind> {
        Vec::new()
    }
    /// Decode/prepare on the model worker, outside the Tokio executor; no GPU state mutation.
    fn prepare_media(&self, _entries: &Value, _detail: &str) -> Result<Arc<crate::media::ProcessedMedia>> {
        bail!("this backend does not take media")
    }
    /// Request-scoped encoder outputs, retained only until end_media. Forks must own independent slots.
    fn begin_media(&mut self, _media: &crate::media::ProcessedMedia) -> Result<()> {
        bail!("this backend does not take media")
    }
    fn media_logits(&mut self, _prompts: &[&[u32]], _codes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        bail!("this backend does not take media")
    }
    /// System One assets in prompt occurrence order; indices address the session's encoded images.
    /// An empty mapping is a text-only question in a request that also contains images.
    fn asset_logits(&mut self, _prompts: &[&[u32]], _codes: &[&[u32]], _order: &[&[usize]]) -> Result<Vec<Vec<f32>>> {
        bail!("this backend does not support positioned image assets")
    }
    fn end_media(&mut self) {}
    /// Share device `device` with other engines through `gate` in work class `class`. Default: the online classes
    /// through [`Backend::set_device_gate`]; [`WorkClass::Background`] is refused, because a backend without
    /// cooperative stop checks and drained releases cannot honour background controls.
    fn set_device_work_class(
        &mut self,
        device: usize,
        gate: Arc<crate::gate::Gate>,
        class: WorkClass,
    ) -> std::result::Result<(), ExecutionError> {
        match class {
            WorkClass::Background => Err(ExecutionError::Unsupported {
                reason: "this backend has no cooperative background execution".into(),
            }),
            online => self
                .set_device_gate(device, gate, online == WorkClass::OnlineMain)
                .map_err(|e| ExecutionError::Gate(e.into())),
        }
    }
    /// Whether this backend can run [`WorkClass::Background`] work with forwards of up to `max_forward_tokens`
    /// tokens. Default: unsupported.
    fn background_support(&self, max_forward_tokens: usize) -> BackgroundSupport {
        let _ = max_forward_tokens;
        BackgroundSupport::Unsupported("this backend has no cooperative background execution".into())
    }
    /// Stop token observed at gate entry, resume and every checkpoint of background work (`None` clears it).
    /// Default: refuses a token, since checkpoints of this backend would not observe it.
    fn set_stop_token(&mut self, stop: Option<StopToken>) -> std::result::Result<(), ExecutionError> {
        match stop {
            None => Ok(()),
            Some(_) => Err(ExecutionError::Unsupported { reason: "this backend does not observe stop tokens".into() }),
        }
    }
    /// Prepare this backend (a fork) for [`WorkClass::Background`] before its gate is set: no private state
    /// survives a call except resident weights and pinned prefixes shared with the online engine (a state cache is
    /// disabled and emptied). Default: refused, so a backend that keeps unknown private state is not used for
    /// background work.
    fn configure_background(&mut self) -> std::result::Result<(), ExecutionError> {
        Err(ExecutionError::Unsupported { reason: "this backend cannot be configured for background work".into() })
    }
    /// Health of this backend's own GPU work, also where [`Backend::device_health`] reports none: marked lost when
    /// the end of its GPU work could not be confirmed. Background work requires it. Default: `device_health`.
    fn work_health(&self) -> Option<DeviceHealth> {
        self.device_health()
    }
    /// Release device memory of finished or failed calls that the backend frees lazily, without waiting for the
    /// device. Default: nothing to do (allocations are released when a call returns).
    fn release_work(&mut self) -> std::result::Result<(), ExecutionError> {
        Ok(())
    }
    /// The media processor shared with forks, for offline planning without the backend ([`PlanCaps`]). Default: none.
    fn media_processor(&self) -> Option<Arc<crate::media::Processor>> {
        None
    }
    /// Host staging of this backend's calls as linear coefficients, for execution host bounds ([`exec_bound`]).
    /// Default: unknown (background admission then stays disabled for this backend).
    ///
    /// [`exec_bound`]: crate::planner::exec_bound
    fn host_staging(&self) -> Option<HostStaging> {
        None
    }
    /// Conservative transient device memory of running work of `shape` (one `Engine::run_plans` micro-batch),
    /// per device, excluding resident weights, pinned prefixes, GEMM workspaces, allocator retention and the online
    /// reserve ([`crate::planner::DeviceWorkEstimate`]). Default: unknown, so background stays disabled for a
    /// backend without a source-derived bound.
    fn device_work_bound(&self, shape: &crate::planner::DeviceWorkShape) -> crate::planner::DeviceWorkBound {
        let _ = shape;
        crate::planner::DeviceWorkBound::Unknown("this backend has no source-derived device work bound".into())
    }
}

/// Readiness flag shared by a backend and its observers. Once marked lost, the device cannot run requests until
/// the model is loaded again; request errors, a failed allocation or a timeout do not mark it.
#[derive(Clone, Debug, Default)]
pub struct DeviceHealth(Arc<AtomicBool>);

impl DeviceHealth {
    /// Record a permanent device loss. Lock-free, so it may be called from a driver callback.
    pub fn mark_lost(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_lost(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OrderJob {
    pub perm: Vec<usize>,
    pub prompt: String,
    pub input_ids: Vec<u32>,
    pub letter_ids: Vec<u32>,
    /// Present only for System One assets; one index per image occurrence in this option order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asset_order: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Prepared {
    pub item: Item,
    pub orders: Vec<OrderJob>,
    pub packed: Packed,
    /// Tokens up to and including the state (only with `Engine::mark_state`).
    pub state_len: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ItemResult {
    pub letter_logits: Vec<Vec<f32>>,
    pub probs: Vec<Vec<f64>>,
    pub p_avg: Vec<f64>,
    pub temperature: f64,
    pub p_cal: Vec<f64>,
}

/// How prepared questions are grouped into forwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Batching {
    /// Upstream token-budget chunks of one-question rows (what a server does with queued questions).
    Budget,
    /// One question per forward (reference export, latency at batch 1).
    Single,
    /// All questions in shared-prefix rows (state computed once for the questions of one request); rows are cut at
    /// `Engine::tree_max_tokens` and sent in one forward per prompt language.
    Tree,
}

/// Longest packed row built by [`Batching::Tree`] (the largest upstream bucket).
/// Longest shared-prefix (tree) row. With attention per tree block the row length changes neither the cost nor the
/// result of a question, so a request with a long state keeps all its questions over one copy of the state.
pub const TREE_MAX_TOKENS: usize = 32768;
/// Most tokens of one [`Batching::Tree`] forward (sum of its rows); more rows go to further forwards.
pub const FORWARD_MAX_TOKENS: usize = 8192;

pub struct Engine<B: Backend> {
    pub manifest: ModelManifest,
    pub tok: BasalTokenizer,
    pub backend: B,
    /// How the questions of one request are run by `decide` (default: shared-prefix tree).
    pub request_batching: Batching,
    /// Mark the token length of the template + state in packed rows (for a backend state cache).
    pub mark_state: bool,
    /// Longest row built by [`Batching::Tree`].
    pub tree_max_tokens: usize,
    /// Most tokens of one [`Batching::Tree`] forward.
    pub forward_max_tokens: usize,
    /// Answer choice questions with 11..=255 options by the grouped strategy (`crate::large_choice`).
    pub large_choice: bool,
    /// Rounds and answer of the grouped strategy.
    pub large_choice_strategy: crate::large_choice::Strategy,
    /// Keep the joint fit after the first round of grouped large choices in [`PlanOutput::first_round`] (evaluation).
    pub trace_large_choice: bool,
    /// Stop token of background work, checked between rounds ([`Engine::set_stop_token`]); `None` online.
    stop: Option<StopToken>,
}

impl<B: Backend> Engine<B> {
    pub fn new(manifest: ModelManifest, backend: B) -> Result<Self> {
        let tok = BasalTokenizer::load_for(&manifest.dir.join("tokenizer.json"), manifest.chat_style)?;
        if tok.token_id(&manifest.bos_token).is_none() {
            bail!("bos token {} is not in the vocabulary", manifest.bos_token);
        }
        Ok(Self {
            manifest,
            tok,
            backend,
            request_batching: Batching::Tree,
            mark_state: false,
            tree_max_tokens: TREE_MAX_TOKENS,
            forward_max_tokens: FORWARD_MAX_TOKENS,
            large_choice: true,
            large_choice_strategy: crate::large_choice::Strategy::DEFAULT,
            trace_large_choice: false,
            stop: None,
        })
    }

    /// A second engine with the same options on a [`Backend::fork`] of the backend.
    pub fn fork(&self) -> Result<Self> {
        let mut e = Self::new(self.manifest.clone(), self.backend.fork()?)?;
        e.request_batching = self.request_batching;
        e.mark_state = self.mark_state;
        e.tree_max_tokens = self.tree_max_tokens;
        e.forward_max_tokens = self.forward_max_tokens;
        e.large_choice = self.large_choice;
        e.large_choice_strategy = self.large_choice_strategy;
        e.trace_large_choice = self.trace_large_choice;
        Ok(e)
    }

    /// Precompute the prompt template up to the state for both languages: the longest common token prefix of two
    /// prompts that differ only in the state. Rows reuse it only on an exact token match.
    pub fn warm_static_prefixes(&mut self) -> Result<Vec<usize>> {
        let mut lens = Vec::new();
        for lang in [crate::prompt::Lang::Pl, crate::prompt::Lang::En] {
            let a = self.tok.encode(&render_model(
                self.manifest.chat_style,
                self.manifest.prompt_format,
                &self.manifest.bos_token,
                "a",
                "q",
                &["x", "y"],
                lang,
            )?)?;
            let b = self.tok.encode(&render_model(
                self.manifest.chat_style,
                self.manifest.prompt_format,
                &self.manifest.bos_token,
                "b",
                "q",
                &["x", "y"],
                lang,
            )?)?;
            let n = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            self.backend.add_prefix(&a[..n])?;
            lens.push(n);
        }
        Ok(lens)
    }

    pub fn run(&mut self, prepared: &[Prepared], batching: Batching) -> Result<Vec<ItemResult>> {
        // forwards -> rows -> (questions of the row, packed row)
        let calls: Vec<Vec<(Vec<usize>, Packed)>> = match batching {
            // Chunks never mix prompt languages: rows of one forward then start with the same template prefix,
            // which the backend can precompute.
            Batching::Budget => [crate::prompt::Lang::Pl, crate::prompt::Lang::En]
                .into_iter()
                .flat_map(|lang| {
                    let idx: Vec<usize> = (0..prepared.len()).filter(|&i| prepared[i].item.lang == lang).collect();
                    chunks(&idx.iter().map(|&i| prepared[i].packed.ids.len()).collect::<Vec<_>>())
                        .into_iter()
                        .map(|c| c.into_iter().map(|j| (vec![idx[j]], prepared[idx[j]].packed.clone())).collect())
                        .collect::<Vec<_>>()
                })
                .collect(),
            Batching::Single => (0..prepared.len()).map(|i| vec![(vec![i], prepared[i].packed.clone())]).collect(),
            Batching::Tree => [crate::prompt::Lang::Pl, crate::prompt::Lang::En]
                .into_iter()
                .filter_map(|lang| {
                    // questions about one state next to each other (first appearance order), so that a row shares
                    // the state across requests
                    let mut idx: Vec<usize> = (0..prepared.len()).filter(|&i| prepared[i].item.lang == lang).collect();
                    let mut first: HashMap<&str, usize> = HashMap::new();
                    for (k, &i) in idx.iter().enumerate() {
                        first.entry(prepared[i].item.state.as_str()).or_insert(k);
                    }
                    idx.sort_by_key(|&i| first[prepared[i].item.state.as_str()]);
                    (!idx.is_empty()).then(|| {
                        let rows: Vec<(Vec<usize>, Packed)> = tree_rows(
                            &idx.iter().map(|&i| prepared[i].clone()).collect::<Vec<_>>(),
                            self.tree_max_tokens,
                        )
                        .into_iter()
                        .map(|(r, p)| (r.into_iter().map(|j| idx[j]).collect(), p))
                        .collect();
                        // consecutive rows up to `forward_max_tokens` per forward
                        let mut calls: Vec<Vec<(Vec<usize>, Packed)>> = vec![];
                        let mut used = 0;
                        for row in rows {
                            let n = row.1.ids.len();
                            if calls.is_empty() || used + n > self.forward_max_tokens {
                                calls.push(vec![]);
                                used = 0;
                            }
                            used += n;
                            calls.last_mut().unwrap().push(row);
                        }
                        calls
                    })
                })
                .flatten()
                .collect(),
        };
        let mut out: Vec<Option<ItemResult>> = vec![None; prepared.len()];
        for rows in calls {
            let letters: Vec<Vec<Vec<u32>>> = rows
                .iter()
                .map(|(idx, _)| {
                    idx.iter().flat_map(|&i| prepared[i].orders.iter().map(|o| o.letter_ids.clone())).collect()
                })
                .collect();
            let lrefs: Vec<&[Vec<u32>]> = letters.iter().map(Vec::as_slice).collect();
            let prefs: Vec<&Packed> = rows.iter().map(|(_, p)| p).collect();
            let logits = self.backend.letter_logits(&prefs, &lrefs)?;
            for ((idx, _), row) in rows.iter().zip(logits) {
                let mut it = row.into_iter();
                for &i in idx {
                    let lg: Vec<Vec<f32>> = it.by_ref().take(prepared[i].orders.len()).collect();
                    out[i] = Some(self.finish(&prepared[i], lg));
                }
            }
        }
        Ok(out.into_iter().map(Option::unwrap).collect())
    }

    fn finish(&self, p: &Prepared, letter_logits: Vec<Vec<f32>>) -> ItemResult {
        let perms: Vec<Vec<usize>> = p.orders.iter().map(|o| o.perm.clone()).collect();
        let temperature = if p.item.uncalibrated {
            1.0
        } else {
            self.manifest.temperature_for(p.item.temp_kind(), p.item.options.len())
        };
        let (probs, p_avg, p_cal) = if self.manifest.prompt_format == PromptFormat::Two {
            let probs: Vec<Vec<f64>> = letter_logits.iter().map(|l| decision::softmax32(l)).collect();
            let p_avg = decision::canonical_average32(&perms, &probs);
            let p_cal = decision::calibrate32(&p_avg, temperature);
            (probs, p_avg, p_cal)
        } else {
            let probs: Vec<Vec<f64>> =
                letter_logits.iter().map(|l| softmax(&l.iter().map(|v| *v as f64).collect::<Vec<_>>())).collect();
            let p_avg = canonical_average(&perms, &probs);
            let p_cal = calibrate(&p_avg, temperature);
            (probs, p_avg, p_cal)
        };
        ItemResult { letter_logits, probs, p_avg, temperature, p_cal }
    }

    /// Planning options of this engine, for an offline [`RequestPlanner`].
    pub fn plan_options(&self) -> PlanOptions {
        PlanOptions {
            mark_state: self.mark_state,
            large_choice: self.large_choice,
            large_choice_strategy: self.large_choice_strategy,
        }
    }

    /// Capability snapshot of this engine's backend for offline planning.
    pub fn plan_caps(&self) -> PlanCaps {
        PlanCaps::from_backend(&self.backend)
    }

    /// An offline planner of this engine's model with its own tokenizer (loaded again), options and capabilities.
    pub fn planner(&self) -> std::result::Result<RequestPlanner, PlannerLoadError> {
        RequestPlanner::load(self.manifest.clone(), self.plan_options(), self.plan_caps())
    }

    /// The online planning context: this engine's tokenizer and backend, an unbounded budget, no stop token.
    fn with_planner<R>(&self, f: impl FnOnce(&PlanContext<'_>) -> R) -> R {
        let budget = PlanBudget::unbounded();
        let caps = BackendCaps(&self.backend);
        let ctx = PlanContext::with_caps(&self.manifest, &self.tok, self.plan_options(), &caps, &budget);
        f(&ctx)
    }

    pub fn prepare(&self, item: Item) -> Result<Prepared> {
        self.with_planner(|p| p.prepare(item))
    }

    /// [`Engine::prepare`] with the first `orders` option orders only (1: as written; a screening group of a large
    /// choice).
    pub fn prepare_orders(&self, item: Item, orders: usize) -> Result<Prepared> {
        self.with_planner(|p| p.prepare_orders(item, orders))
    }

    /// Tokens of the prompt up to and including the state (see `PlanContext::state_len`).
    fn state_len(&self, p: &Prepared) -> Result<usize> {
        self.with_planner(|c| c.state_len(p))
    }

    /// Parse, validate and prepare a System One request without running the model.
    pub fn prepare_request(&self, body: &Value) -> Result<Vec<Prepared>, DecideError> {
        self.with_planner(|p| p.prepare_request(body)).map_err(PlanError::into_decide)
    }

    /// Validate a request and plan its questions: one readout for 2..=10 options, the grouped strategy for choice
    /// questions with 11..=255 options (when `large_choice` is on).
    pub fn plan_request(&self, body: &Value) -> Result<RequestPlan, DecideError> {
        self.plan_request_as(body, crate::request::Dialect::TypeSafe)
    }

    /// [`Engine::plan_request`] for an endpoint dialect (upstream: no model check, 2..10 options only).
    pub fn plan_request_as(&self, body: &Value, dialect: crate::request::Dialect) -> Result<RequestPlan, DecideError> {
        self.with_planner(|p| p.plan_request_as(body, dialect)).map_err(PlanError::into_decide)
    }

    fn plan_decision_request_as(
        &self,
        body: &Value,
        dialect: crate::request::Dialect,
    ) -> Result<RequestPlan, DecideError> {
        self.with_planner(|p| p.plan_decision_request_as(body, dialect)).map_err(PlanError::into_decide)
    }

    /// Plan validated items (see [`Engine::plan_request`]).
    pub fn plan_items(&self, items: Vec<Item>) -> Result<RequestPlan> {
        self.with_planner(|p| p.plan_items(items)).map_err(|e| match e.into_decide() {
            DecideError::Internal(e) => e,
            other => anyhow::Error::new(other),
        })
    }

    /// Whether this engine can run background work (its backend's support for forwards of up to
    /// `forward_max_tokens`).
    pub fn background_support(&self) -> BackgroundSupport {
        self.backend.background_support(self.forward_max_tokens)
    }

    /// Limits of this engine's execution that bound the device shape of its forwards.
    pub fn shape_limits(&self) -> crate::planner::ShapeLimits {
        crate::planner::ShapeLimits {
            batching: self.request_batching,
            tree_max_tokens: self.tree_max_tokens,
            forward_max_tokens: self.forward_max_tokens,
            max_positions: self.manifest.config.max_positions(),
            max_options: self.manifest.prompt_format.max_options(),
            evidence_matches_prompt: self.manifest.chat_style == ChatStyle::Llama,
            adaptive_orders: self.manifest.orders == OptionOrders::Auto,
        }
    }

    /// Upper bounds of the device work of running `plans` together through [`Engine::run_plans`].
    pub fn device_work_shape(
        &self,
        plans: &[&RequestPlan],
    ) -> std::result::Result<crate::planner::DeviceWorkShape, crate::planner::UnmodelledShape> {
        crate::planner::DeviceWorkShape::from_plans(plans, &self.shape_limits())
    }

    /// Transient device memory of running `plans` together on this engine's backend, or `Unknown` with the reason.
    pub fn device_work_bound(&self, plans: &[&RequestPlan]) -> crate::planner::DeviceWorkBound {
        match self.device_work_shape(plans) {
            Ok(shape) => self.backend.device_work_bound(&shape),
            Err(e) => crate::planner::DeviceWorkBound::Unknown(e.to_string()),
        }
    }

    /// Stop token of the next background work: checked between engine rounds and passed to the backend, which
    /// observes it at gate entry, resume and checkpoints. `None` clears it (online work).
    pub fn set_stop_token(&mut self, stop: Option<StopToken>) -> std::result::Result<(), ExecutionError> {
        if stop.is_some() {
            if let BackgroundSupport::Unsupported(reason) = self.background_support() {
                return Err(ExecutionError::Unsupported { reason });
            }
        }
        self.backend.set_stop_token(stop.clone())?;
        self.stop = stop;
        Ok(())
    }

    /// Prepare this engine (a background fork) for background work: [`Backend::configure_background`]. Call it
    /// before setting the background work class of its devices.
    pub fn configure_background(&mut self) -> std::result::Result<(), ExecutionError> {
        if self.backend.work_health().is_none() {
            return Err(ExecutionError::Unsupported {
                reason: "this backend reports no health of its GPU work, needed after a failed background call".into(),
            });
        }
        self.backend.configure_background()
    }

    /// Reset execution state before the next background micro-batch, also after an error or interruption: end any
    /// media session (its private slot), clear the stop token and release lazily freed device memory
    /// ([`Backend::release_work`]).
    ///
    /// After `run_plans` returned (`Ok` or `Err`) and this succeeded, with the backend configured by
    /// [`Engine::configure_background`] and its [`Backend::work_health`] not lost: the call's own GPU work is confirmed
    /// complete and no private device allocation of the call remains held by the engine. Device memory may still
    /// be retained by the allocator (CUDA memory pool, Metal buffer cache) for reuse; that retention is separate
    /// from the work estimate. A lost `DeviceHealth` means the end was not confirmed: memory may be retained and
    /// the device must be treated as unusable (quarantine).
    pub fn reset_execution(&mut self) -> std::result::Result<(), ExecutionError> {
        self.backend.end_media();
        self.stop = None;
        self.backend.set_stop_token(None)?;
        self.backend.release_work()
    }

    /// `Err(Interrupted)` once the background stop token is stopped; always `Ok` online.
    fn check_stop(&self) -> Result<()> {
        if let Some(stop) = &self.stop {
            stop.check()?;
        }
        Ok(())
    }

    /// Run a plan: one round for direct questions; grouped questions get a second, final round with their strongest
    /// candidates. Returns the final (calibrated) distribution of every planned question.
    /// Evidence spans of one question (upstream `Evidence.spans`): a separate forward of its prompt in the original
    /// option order; empty when no token lies inside the state.
    fn evidence_for(&mut self, item: &Item, limit: Option<usize>) -> Result<Value> {
        let Some(inp) =
            crate::evidence::input_format(&self.tok, &self.manifest.bos_token, item, self.manifest.prompt_format)?
        else {
            return Ok(Value::Array(vec![]));
        };
        let (ls, le) = self.backend.evidence_scores(&inp.ids, inp.t0, inp.t1)?;
        Ok(crate::evidence::spans(&ls, &le, &inp, &item.state, limit))
    }

    pub fn run_plan(&mut self, plan: &RequestPlan) -> Result<PlanOutput> {
        Ok(self.run_plans(&[plan])?.remove(0))
    }

    /// Diagnostic/export readouts for the already prepared orders. Media encodings are request-scoped;
    /// adaptive orders are executed by `run_plan`, not added to this fixed-order export.
    pub fn run_prepared(
        &mut self,
        prepared: &[Prepared],
        media: Option<&crate::media::ProcessedMedia>,
        batching: Batching,
    ) -> Result<Vec<ItemResult>> {
        let Some(media) = media else {
            return self.run(prepared, batching);
        };
        self.backend.begin_media(media)?;
        let result = (|| {
            let prompts: Vec<_> =
                prepared.iter().flat_map(|p| p.orders.iter().map(|o| o.input_ids.as_slice())).collect();
            let codes: Vec<_> =
                prepared.iter().flat_map(|p| p.orders.iter().map(|o| o.letter_ids.as_slice())).collect();
            let orders: Vec<_> = prepared.iter().flat_map(|p| &p.orders).collect();
            let logits = media_readout(&mut self.backend, &orders, &prompts, &codes)?;
            anyhow::ensure!(logits.len() == prompts.len(), "media readout count mismatch");
            let mut logits = logits.into_iter();
            Ok(prepared.iter().map(|p| self.finish(p, logits.by_ref().take(p.orders.len()).collect())).collect())
        })();
        self.backend.end_media();
        result
    }

    /// Run the plans of several requests together: the questions of all plans share the forwards of each round
    /// (first round, then the final round of grouped large choices), as a server batches waiting requests.
    pub fn run_plans(&mut self, plans: &[&RequestPlan]) -> Result<Vec<PlanOutput>> {
        self.check_stop()?;
        let mut outputs = self.run_decision_plans(plans)?;
        if !plans.iter().any(|p| p.tasks.is_some()) {
            return Ok(outputs);
        }
        let mut continuations = Vec::new();
        let mut owners = Vec::new();
        for (i, plan) in plans.iter().enumerate() {
            let Some(mut task) = plan.tasks.clone() else { continue };
            let raw = plan_response(&self.manifest.name, plan, &outputs[i])?;
            if let Some(mut body) = task.second_round(&raw, self.manifest.prompt_format.max_options())? {
                body["model"] = self.manifest.name.clone().into();
                match self.plan_decision_request_as(&body, plan.dialect) {
                    Ok(next) => {
                        continuations.push(next);
                        owners.push((i, task, raw));
                    }
                    Err(error) => outputs[i].task_error = Some(error.to_string()),
                }
            } else {
                let empty = serde_json::Map::new();
                let thresholds = if plan.media.is_some() { &empty } else { &self.manifest.task_thresholds };
                outputs[i].task_response = Some(task.finish(raw, None, thresholds)?);
            }
        }
        if !continuations.is_empty() {
            self.check_stop()?;
            // These are primitive plans only: extraction has exactly one dependent continuation, on this engine.
            let refs: Vec<_> = continuations.iter().collect();
            let second = self.run_decision_plans(&refs)?;
            for ((next, output), (i, task, mut raw)) in continuations.iter().zip(second).zip(owners) {
                let next = plan_response(&self.manifest.name, next, &output)?;
                for key in ["input_tokens", "branches", "second_order_branches"] {
                    if let Some(n) = next["usage"][key].as_u64() {
                        let old = raw["usage"][key].as_u64().unwrap_or(0);
                        raw["usage"][key] = old.saturating_add(n).into();
                    }
                }
                if let Some(n) = next["usage"]["longest_prompt_tokens"].as_u64() {
                    raw["usage"]["longest_prompt_tokens"] =
                        n.max(raw["usage"]["longest_prompt_tokens"].as_u64().unwrap_or(0)).into();
                }
                let empty = serde_json::Map::new();
                let thresholds = if plans[i].media.is_some() { &empty } else { &self.manifest.task_thresholds };
                outputs[i].task_response = Some(task.finish(raw, Some(&next), thresholds)?);
            }
        }
        Ok(outputs)
    }

    fn run_decision_plans(&mut self, plans: &[&RequestPlan]) -> Result<Vec<PlanOutput>> {
        self.check_stop()?;
        if plans.iter().any(|p| p.media.is_some()) {
            let mut results = Vec::with_capacity(plans.len());
            for plan in plans {
                results.push(if plan.media.is_some() {
                    self.run_media_plan(plan)?
                } else {
                    self.run_decision_plans(&[plan])?.remove(0)
                });
            }
            return Ok(results);
        }
        let mut offs = Vec::with_capacity(plans.len());
        let mut all = Vec::new();
        for p in plans {
            offs.push(all.len());
            all.extend(p.prepared.iter().cloned());
        }
        let mut results = self.run(&all, self.request_batching)?;
        let mut second_tokens = vec![0usize; all.len()];
        if self.manifest.orders == OptionOrders::Auto {
            let mut reverse = Vec::new();
            let mut owners = Vec::new();
            for (i, p) in all.iter().enumerate() {
                if p.orders.len() != 1 {
                    continue;
                }
                let probabilities = &results[i].p_cal;
                let uncertain = if p.item.temp_kind() == crate::request::QType::Noul {
                    (0.2..=0.8).contains(&probabilities[0])
                } else {
                    probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max) < 0.6
                };
                if uncertain {
                    let mut redo = prepare_protocol(
                        &self.tok,
                        &self.manifest.bos_token,
                        p.item.clone(),
                        self.manifest.prompt_format,
                        self.manifest.chat_style,
                        None,
                        None,
                    )?;
                    redo.orders.remove(0);
                    redo.packed = pack(&redo.orders.iter().map(|o| o.input_ids.clone()).collect::<Vec<_>>());
                    if self.mark_state {
                        redo.state_len = self.state_len(&redo)?;
                        redo.packed.state_len = redo.state_len.min(redo.packed.prefix_len);
                    }
                    owners.push(i);
                    reverse.push(redo);
                }
            }
            self.check_stop()?;
            let rerun = self.run(&reverse, self.request_batching)?;
            for ((i, p2), r2) in owners.into_iter().zip(&reverse).zip(rerun) {
                second_tokens[i] = p2.orders[0].input_ids.len();
                let mut both = all[i].clone();
                both.orders.extend(p2.orders.clone());
                let mut logits = results[i].letter_logits.clone();
                logits.extend(r2.letter_logits);
                results[i] = self.finish(&both, logits);
            }
        }
        // grouped large choices: further rounds until each has asked its final group; the questions of all plans
        // share the forwards of a round
        let strategy = self.large_choice_strategy;
        let mut large: Vec<Large> = Vec::new();
        for (plan, &off) in plans.iter().zip(&offs) {
            for q in &plan.questions {
                if let QuestionPlan::Groups { item, groups, parts } = q {
                    large.push(Large {
                        item: item.clone(),
                        groups: groups.clone(),
                        targets: parts.iter().map(|&i| results[off + i].p_avg.clone()).collect(),
                        last: (0..groups.len()).collect(),
                        finalists: None,
                        final_p: vec![],
                        tokens: 0,
                        prompts: 0,
                        rounds: 1,
                        first_round: self.trace_large_choice.then(|| {
                            let targets: Vec<Vec<f64>> =
                                parts.iter().map(|&i| results[off + i].p_avg.clone()).collect();
                            crate::large_choice::fit_joint(item.options.len(), groups, &targets)
                        }),
                    });
                }
            }
        }
        loop {
            let mut prep = Vec::new();
            let mut owners = Vec::new();
            for (li, l) in large.iter().enumerate().filter(|(_, l)| l.finalists.is_none()) {
                let n = l.item.options.len();
                let (round, last) =
                    match crate::large_choice::next_round(&strategy, n, &l.groups, &l.targets, &l.last, l.rounds) {
                        crate::large_choice::Round::Final(f) => (vec![f], true),
                        crate::large_choice::Round::Groups(g) => (g, false),
                    };
                for g in round {
                    let mut sub = l.item.clone();
                    sub.keys = g.iter().map(|&i| l.item.keys[i].clone()).collect();
                    sub.options = g.iter().map(|&i| l.item.options[i].clone()).collect();
                    prep.push(self.prepare_orders(sub, if last { 2 } else { strategy.screen_orders })?);
                    owners.push((li, g, last));
                }
            }
            if prep.is_empty() {
                break;
            }
            self.check_stop()?;
            let res = self.run(&prep, self.request_batching)?;
            for l in large.iter_mut() {
                if l.finalists.is_none() {
                    l.last.clear();
                    l.rounds += 1;
                }
            }
            for (((li, g, last), r), p) in owners.into_iter().zip(res).zip(&prep) {
                let l = &mut large[li];
                l.tokens += p.orders.iter().map(|o| o.input_ids.len()).sum::<usize>();
                l.prompts += p.orders.len();
                l.last.push(l.groups.len());
                if last {
                    l.finalists = Some(g.clone());
                    l.final_p = r.p_cal.clone();
                }
                l.groups.push(g);
                l.targets.push(r.p_avg);
            }
        }
        let mut li = 0;
        let mut outs = Vec::with_capacity(plans.len());
        for (plan, &off) in plans.iter().zip(&offs) {
            let mut finals = Vec::with_capacity(plan.questions.len());
            let end = off + plan.prepared.len();
            let second_order_branches = (self.manifest.orders == OptionOrders::Auto)
                .then(|| second_tokens[off..end].iter().filter(|n| **n > 0).count());
            let mut final_round_tokens = second_tokens[off..end].iter().sum();
            let mut extra_prompts = 0;
            let mut finalists = Vec::with_capacity(plan.questions.len());
            let mut first_round = Vec::with_capacity(plan.questions.len());
            for q in &plan.questions {
                let mut fin = None;
                let mut first = None;
                match q {
                    QuestionPlan::Direct(i) => {
                        finals.push((plan.prepared[*i].item.clone(), results[off + *i].p_cal.clone()))
                    }
                    QuestionPlan::Fixed(item) => finals.push((item.clone(), vec![1.0])),
                    QuestionPlan::Multi { item, parts } => {
                        finals.push((item.clone(), parts.iter().map(|&i| results[off + i].p_cal[0]).collect()))
                    }
                    QuestionPlan::Groups { item, .. } => {
                        let l = &large[li];
                        li += 1;
                        final_round_tokens += l.tokens;
                        extra_prompts += l.prompts;
                        let f = l.finalists.clone().unwrap_or_default();
                        let t = self.manifest.temperature(item.temp_kind());
                        let p = crate::large_choice::answer(
                            &strategy,
                            item.options.len(),
                            &l.groups,
                            &l.targets,
                            &f,
                            &l.final_p,
                            |x| calibrate(x, t),
                        );
                        fin = Some(f);
                        first = l.first_round.clone();
                        finals.push((item.clone(), p));
                    }
                }
                finalists.push(fin);
                first_round.push(first);
            }
            let mut evidence = Vec::with_capacity(finals.len());
            if finals.iter().any(|(item, _)| item.evidence) {
                self.check_stop()?;
            }
            for (item, _) in &finals {
                evidence.push(if item.evidence { Some(self.evidence_for(item, plan.evidence_limit)?) } else { None });
            }
            let longest_prompt_tokens = plan
                .prepared
                .iter()
                .flat_map(|p| &p.orders)
                .map(|o| o.input_ids.len())
                .chain(second_tokens[off..end].iter().copied())
                .max()
                .unwrap_or(0);
            outs.push(PlanOutput {
                task_response: None,
                task_error: None,
                asset_media_tokens: None,
                longest_prompt_tokens,
                finals,
                final_round_tokens,
                extra_prompts,
                finalists,
                first_round,
                evidence,
                second_order_branches,
            });
        }
        Ok(outs)
    }

    fn run_media_plan(&mut self, plan: &RequestPlan) -> Result<PlanOutput> {
        let media = plan.media.as_deref().ok_or_else(|| anyhow::anyhow!("missing media plan"))?;
        self.check_stop()?;
        self.backend.begin_media(media)?;
        let result = (|| -> Result<PlanOutput> {
            self.check_stop()?;
            let prompts: Vec<&[u32]> =
                plan.prepared.iter().flat_map(|p| p.orders.iter().map(|o| o.input_ids.as_slice())).collect();
            let codes: Vec<&[u32]> =
                plan.prepared.iter().flat_map(|p| p.orders.iter().map(|o| o.letter_ids.as_slice())).collect();
            let orders: Vec<_> = plan.prepared.iter().flat_map(|p| &p.orders).collect();
            let mut asset_media_tokens = media.asset_markers.as_ref().map(|_| asset_tokens(&orders, media));
            let logits = media_readout(&mut self.backend, &orders, &prompts, &codes)?;
            anyhow::ensure!(logits.len() == prompts.len(), "media backend readout count mismatch");
            let mut logits = logits.into_iter();
            let mut results: Vec<ItemResult> =
                plan.prepared.iter().map(|p| self.finish(p, logits.by_ref().take(p.orders.len()).collect())).collect();
            let mut longest_prompt_tokens = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
            let mut extra_tokens = 0;
            let mut second = 0;
            if self.manifest.orders == OptionOrders::Auto {
                let mut redo = Vec::new();
                let mut owners = Vec::new();
                for (i, p) in plan.prepared.iter().enumerate() {
                    let values = &results[i].p_cal;
                    let uncertain = if p.item.temp_kind() == crate::request::QType::Noul {
                        (0.2..=0.8).contains(&values[0])
                    } else {
                        values.iter().copied().fold(f64::NEG_INFINITY, f64::max) < 0.6
                    };
                    if uncertain {
                        let mut reverse = prepare_protocol(
                            &self.tok,
                            &self.manifest.bos_token,
                            p.item.clone(),
                            self.manifest.prompt_format,
                            self.manifest.chat_style,
                            Some(media),
                            None,
                        )?;
                        reverse.orders.remove(0);
                        owners.push(i);
                        redo.push(reverse);
                    }
                }
                if !redo.is_empty() {
                    self.check_stop()?;
                }
                let prompts: Vec<&[u32]> = redo.iter().map(|p| p.orders[0].input_ids.as_slice()).collect();
                let codes: Vec<&[u32]> = redo.iter().map(|p| p.orders[0].letter_ids.as_slice()).collect();
                longest_prompt_tokens = longest_prompt_tokens.max(prompts.iter().map(|p| p.len()).max().unwrap_or(0));
                let orders: Vec<_> = redo.iter().map(|p| &p.orders[0]).collect();
                if let Some(tokens) = &mut asset_media_tokens {
                    *tokens += asset_tokens(&orders, media);
                }
                let logits = media_readout(&mut self.backend, &orders, &prompts, &codes)?;
                anyhow::ensure!(logits.len() == redo.len(), "media backend reverse readout count mismatch");
                for ((i, p2), logits) in owners.into_iter().zip(&redo).zip(logits) {
                    extra_tokens += p2.orders[0].input_ids.len();
                    second += 1;
                    let mut both = plan.prepared[i].clone();
                    both.orders.extend(p2.orders.clone());
                    let mut all = results[i].letter_logits.clone();
                    all.push(logits);
                    results[i] = self.finish(&both, all);
                }
            }
            let mut finals = Vec::with_capacity(plan.questions.len());
            for question in &plan.questions {
                match question {
                    QuestionPlan::Direct(i) => finals.push((plan.prepared[*i].item.clone(), results[*i].p_cal.clone())),
                    QuestionPlan::Multi { item, parts } => {
                        finals.push((item.clone(), parts.iter().map(|&i| results[i].p_cal[0]).collect()))
                    }
                    QuestionPlan::Fixed(item) => finals.push((item.clone(), vec![1.0])),
                    QuestionPlan::Groups { .. } => anyhow::bail!("grouped large choices are not supported with media"),
                }
            }
            let n = finals.len();
            Ok(PlanOutput {
                task_response: None,
                task_error: None,
                asset_media_tokens,
                longest_prompt_tokens,
                finals,
                final_round_tokens: extra_tokens,
                extra_prompts: second,
                finalists: vec![None; n],
                first_round: vec![None; n],
                evidence: vec![None; n],
                second_order_branches: (self.manifest.orders == OptionOrders::Auto).then_some(second),
            })
        })();
        self.backend.end_media();
        result
    }

    /// Prompts of the final rounds are not known before the first round; `usage.input_tokens` counts the first round.
    pub fn decide(&mut self, body: &Value) -> Result<Value, DecideError> {
        self.decide_as(body, crate::request::Dialect::TypeSafe)
    }

    /// [`Engine::decide`] with the conventions of an endpoint dialect.
    pub fn decide_as(&mut self, body: &Value, dialect: crate::request::Dialect) -> Result<Value, DecideError> {
        let plan = self.plan_request_as(body, dialect)?;
        let finals = self.run_plan(&plan)?;
        plan_response(&self.manifest.name, &plan, &finals)
    }

    /// Like [`Engine::decide`], with host/backend timing in milliseconds (not part of the System One response).
    pub fn decide_timed(&mut self, body: &Value) -> Result<(Value, Value), DecideError> {
        let t0 = Instant::now();
        let plan = self.plan_request(body)?;
        let t1 = Instant::now();
        let finals = self.run_plan(&plan)?;
        let t2 = Instant::now();
        let resp = plan_response(&self.manifest.name, &plan, &finals)?;
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        Ok((resp, json!({"prepare_ms": ms(t0, t1), "forward_ms": ms(t1, t2), "total_ms": ms(t0, Instant::now())})))
    }
}

/// Greedy shared-prefix rows in question order, each about `max_tokens` long at most (a single longer question gets
/// its own row; [`TrieSize`] ignores the trie depth limit).
fn tree_rows(prepared: &[Prepared], max_tokens: usize) -> Vec<(Vec<usize>, Packed)> {
    let row = |idx: &[usize]| -> Packed {
        let toks: Vec<Vec<Vec<u32>>> =
            idx.iter().map(|&i| prepared[i].orders.iter().map(|o| o.input_ids.clone()).collect()).collect();
        let mut p = pack_tree(&toks);
        p.state_len = idx.iter().map(|&i| prepared[i].state_len).min().unwrap_or(0).min(p.prefix_len);
        p
    };
    let mut rows = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut size = TrieSize::default();
    for (i, q) in prepared.iter().enumerate() {
        let prompts = q.prompts();
        if !cur.is_empty() && size.tokens + size.growth(&prompts) > max_tokens {
            let r = row(&cur);
            rows.push((std::mem::take(&mut cur), r));
            size = TrieSize::default();
        }
        cur.push(i);
        size.insert(&prompts);
    }
    if !cur.is_empty() {
        let r = row(&cur);
        rows.push((cur, r));
    }
    rows
}

impl Prepared {
    /// Token ids of the option-order prompts.
    pub fn prompts(&self) -> Vec<&[u32]> {
        self.orders.iter().map(|o| o.input_ids.as_slice()).collect()
    }
}

pub fn prepare(tok: &BasalTokenizer, bos: &str, item: Item) -> Result<Prepared> {
    prepare_format(tok, bos, item, PromptFormat::One)
}

pub fn prepare_format(tok: &BasalTokenizer, bos: &str, item: Item, fmt: PromptFormat) -> Result<Prepared> {
    prepare_protocol(tok, bos, item, fmt, ChatStyle::Llama, None, None)
}

/// Upper bound of a rendered prompt of `item` with `opts` (template text, the item's strings and media
/// placeholders), doubled for string growth.
fn render_bound(bos: &str, item: &Item, opts: &[&str], media: Option<&crate::media::ProcessedMedia>) -> u64 {
    let text = item.state.len() + item.question.len() + opts.iter().map(|o| o.len() + 8).sum::<usize>();
    let placeholders: usize = media.map_or(0, |m| {
        m.items
            .iter()
            .map(|x| match x {
                crate::media::Input::Image(x) => 16 + 9 * x.soft_tokens,
                crate::media::Input::Audio(x) => 16 + 9 * x.soft_tokens,
            })
            .sum()
    });
    // the prompt templates are smaller than 16 KiB; asset substitution may repeat placeholders per marker
    let markers = media.and_then(|m| m.asset_markers.as_ref()).map_or(1, |v| v.len().max(1));
    (16384 + bos.len() + 2 * text + placeholders * markers) as u64 * 2
}

/// Prepare the option-order prompts of `item`. With a bounded `budget` every allocation of the prompts, token ids
/// and packed row is reserved before it is made (retained until the end of the plan); the tokenizer's transient
/// allocations are reserved for the duration of each encode. `None` keeps the online path without accounting.
pub(crate) fn prepare_protocol(
    tok: &BasalTokenizer,
    bos: &str,
    mut item: Item,
    fmt: PromptFormat,
    chat: ChatStyle,
    media: Option<&crate::media::ProcessedMedia>,
    budget: Option<&PlanBudget>,
) -> Result<Prepared> {
    item.prompt_format = fmt;
    let k = item.options.len();
    let mut orders = Vec::new();
    for perm in decision::orders(k) {
        let opts: Vec<&str> = perm.iter().map(|&j| item.options[j].as_str()).collect();
        let mut rendered = match budget {
            Some(b) => Some(b.reserve(PlanStage::Tokens, render_bound(bos, &item, &opts, media))?),
            None => None,
        };
        let mut asset_order = None;
        let prompt = match media {
            Some(media) if media.asset_markers.is_some() => {
                let raw = render_model(chat, fmt, bos, &item.state, &item.question, &opts, item.lang)?;
                let (prompt, indices) = crate::systemone::substitute(
                    &raw,
                    media.asset_markers.as_deref().unwrap_or_default(),
                    &media.placeholders(),
                )?;
                asset_order = Some(indices);
                prompt
            }
            Some(media) => {
                crate::prompt::render_media(chat, fmt, bos, &item.state, &item.question, &opts, item.lang, media)?
            }
            None => render_model(chat, fmt, bos, &item.state, &item.question, &opts, item.lang)?,
        };
        let (ids, transient) = match (budget, rendered.as_mut()) {
            (Some(b), Some(r)) => {
                r.fit(PlanStage::Tokens, prompt.capacity() as u64)?;
                let len = prompt.len() as u64;
                // token ids (at most one per byte) and letter ids stay with the plan
                let ids = b.reserve(PlanStage::Tokens, len.saturating_mul(4).saturating_add(64))?;
                let transient = b.reserve(
                    PlanStage::Tokens,
                    len.saturating_mul(TOKENIZER_BYTES_PER_BYTE).saturating_add(len.saturating_mul(2)),
                )?;
                (Some(ids), Some(transient))
            }
            _ => (None, None),
        };
        let input_ids = tok.encode(&prompt)?;
        let Some(letter_ids) = tok.code_ids(fmt, &prompt, &input_ids, k)? else {
            bail!("option letters are not single tokens for this tokenizer ({k} options)");
        };
        drop(transient);
        if let (Some(mut ids), Some(r)) = (ids, rendered) {
            ids.fit(
                PlanStage::Tokens,
                ((input_ids.capacity() + letter_ids.capacity()) * 4 + perm.capacity() * 8) as u64,
            )?;
            ids.retain();
            r.retain();
        }
        orders.push(OrderJob { perm, prompt, input_ids, letter_ids, asset_order });
    }
    let packed_reservation = match budget {
        Some(b) => {
            let tokens: u64 = orders.iter().map(|o| o.input_ids.len() as u64).sum();
            Some(
                b.reserve(PlanStage::Packed, tokens.saturating_mul(32).saturating_add(orders.len() as u64 * 16 + 256))?,
            )
        }
        None => None,
    };
    let packed = pack(&orders.iter().map(|o| o.input_ids.clone()).collect::<Vec<_>>());
    if let Some(r) = packed_reservation {
        r.retain();
    }
    Ok(Prepared { item, orders, packed, state_len: 0 })
}

/// A grouped large choice between rounds: the groups asked so far with their distributions (`p_avg`), the groups of
/// the latest round, and the final group with its calibrated readout once asked.
struct Large {
    item: Item,
    groups: Vec<Vec<usize>>,
    targets: Vec<Vec<f64>>,
    last: Vec<usize>,
    finalists: Option<Vec<usize>>,
    final_p: Vec<f64>,
    tokens: usize,
    prompts: usize,
    rounds: usize,
    first_round: Option<Vec<f64>>,
}

/// Answers of a planned request: final distribution per question, and the prompt tokens of the rounds after the
/// first of grouped large choices (counted in `usage.input_tokens`).
pub struct PlanOutput {
    task_response: Option<Value>,
    task_error: Option<String>,
    asset_media_tokens: Option<usize>,
    pub finals: Vec<(Item, Vec<f64>)>,
    pub second_order_branches: Option<usize>,
    /// Longest executed prompt, including an adaptive reversed order.
    pub longest_prompt_tokens: usize,
    pub final_round_tokens: usize,
    /// Prompts of the rounds after the first (grouped large choices).
    pub extra_prompts: usize,
    /// Final group of each grouped large choice (same order as `finals`).
    pub finalists: Vec<Option<Vec<usize>>>,
    /// Joint fit after the first round of each grouped large choice, with [`Engine::trace_large_choice`].
    pub first_round: Vec<Option<Vec<f64>>>,
    /// Evidence spans per question (same order as `finals`), for questions that asked for them.
    pub evidence: Vec<Option<Value>>,
}

/// How one question of a request is answered.
#[derive(Clone, Debug)]
pub enum QuestionPlan {
    /// One letter readout (index into `RequestPlan::prepared`).
    Direct(usize),
    /// A choice or score question with a single option: probability 1, no readout.
    Fixed(Item),
    /// Grouped strategy for 11..=255 choice options: `parts[k]` answers `groups[k]`.
    Groups { item: Item, groups: Vec<Vec<usize>>, parts: Vec<usize> },
    /// `multi`: `parts[k]` is the yes/no branch of label k.
    Multi { item: Item, parts: Vec<usize> },
}

#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub(crate) tasks: Option<crate::tasks::TaskPlan>,
    pub prepared: Vec<Prepared>,
    pub questions: Vec<QuestionPlan>,
    /// Characters of the request's own state when facts were appended (evidence spans stay inside it).
    pub evidence_limit: Option<usize>,
    /// Answer conventions of the endpoint.
    pub dialect: crate::request::Dialect,
    pub calibration_unvalidated: bool,
    pub media: Option<Arc<crate::media::ProcessedMedia>>,
    /// With adaptive option orders: token lengths of the reversed-order prompts an uncertain first-round answer may
    /// run (computed by the context check, which tokenizes both orders anyway); bounds the device shape of that
    /// round ([`crate::planner::DeviceWorkShape`]).
    pub(crate) auto_reverse: Option<crate::planner::ReverseTokens>,
}

pub fn plan_response(model: &str, plan: &RequestPlan, out: &PlanOutput) -> Result<Value, DecideError> {
    if let Some(error) = &out.task_error {
        return Err(crate::tasks::invalid(error));
    }
    if let Some(response) = &out.task_response {
        return Ok(response.clone());
    }
    let mut answers = serde_json::Map::new();
    for ((it, p), ev) in out.finals.iter().zip(&out.evidence) {
        let a = crate::decision::answer_checked(it, p, plan.dialect).map_err(|msg| {
            DecideError::Validation(vec![crate::request::FieldError::new(
                vec!["body".into(), "questions".into(), it.name.clone().into()],
                msg,
                "value_error",
            )])
        })?;
        let mut a = a;
        if let (Some(ev), Some(obj)) = (ev, a.as_object_mut()) {
            obj.insert("evidence".into(), ev.clone());
        }
        if plan.calibration_unvalidated {
            a["calibration"] = "unvalidated".into();
        }
        answers.insert(it.name.clone(), a);
    }
    let branches: usize = plan
        .questions
        .iter()
        .map(|q| match q {
            QuestionPlan::Multi { parts, .. } => parts.len(),
            QuestionPlan::Fixed(_) => 0,
            _ => 1,
        })
        .sum();
    let input_tokens: usize =
        plan.prepared.iter().flat_map(|p| &p.orders).map(|o| o.input_ids.len()).sum::<usize>() + out.final_round_tokens;
    let mut response = json!({"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                                                             "questions": plan.questions.len(), "branches": branches}});
    if out.finals.first().is_some_and(|(it, _)| it.prompt_format == PromptFormat::Two) {
        response["usage"]["longest_prompt_tokens"] = out.longest_prompt_tokens.into();
    }
    if let Some(n) = out.second_order_branches {
        response["usage"]["second_order_branches"] = n.into();
    }
    if let Some(media) = &plan.media {
        for answer in response["answers"].as_object_mut().into_iter().flat_map(|m| m.values_mut()) {
            answer["calibration"] = if plan.calibration_unvalidated { "unvalidated" } else { "text" }.into();
        }
        let prompts = plan.prepared.iter().map(|p| p.orders.len()).sum::<usize>() + out.extra_prompts;
        response["usage"]["media"] = json!({"images":media.image_count(),"audio":media.audio_count(),"media_tokens":out.asset_media_tokens.unwrap_or_else(|| media.items.iter().map(crate::media::Input::soft_tokens).sum::<usize>()*prompts),"detail":if media.high_detail {"high"} else {"default"}});
    }
    Ok(response)
}

pub fn response(model: &str, prepared: &[Prepared], results: &[ItemResult]) -> Value {
    let answers: serde_json::Map<String, Value> =
        prepared.iter().zip(results).map(|(p, r)| (p.item.name.clone(), answer(&p.item, &r.p_cal))).collect();
    // Upstream counts the tokens of every option-order prompt; no text is generated.
    let input_tokens: usize = prepared.iter().flat_map(|p| &p.orders).map(|o| o.input_ids.len()).sum();
    json!({"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}})
}

fn media_readout<B: Backend>(
    backend: &mut B,
    orders: &[&OrderJob],
    prompts: &[&[u32]],
    codes: &[&[u32]],
) -> Result<Vec<Vec<f32>>> {
    if orders.is_empty() {
        return Ok(Vec::new());
    }
    if orders.iter().any(|o| o.asset_order.is_some()) {
        let mapping: Vec<&[usize]> = orders
            .iter()
            .map(|o| o.asset_order.as_deref())
            .collect::<Option<_>>()
            .ok_or_else(|| anyhow::anyhow!("cannot mix native and positioned media in one session"))?;
        backend.asset_logits(prompts, codes, &mapping)
    } else {
        backend.media_logits(prompts, codes)
    }
}

fn asset_tokens(orders: &[&OrderJob], media: &crate::media::ProcessedMedia) -> usize {
    orders.iter().flat_map(|o| o.asset_order.iter().flatten()).map(|&i| media.items[i].soft_tokens()).sum()
}
