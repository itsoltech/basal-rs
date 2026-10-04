//! Request pipeline independent of the GPU backend: items -> prompts and token ids of both option orders -> packed
//! rows -> backend letter logits -> averaged, calibrated distributions -> System One answers.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::decision::{self, answer, calibrate, canonical_average, softmax};
use crate::manifest::ModelManifest;
use crate::pack::{chunks, pack, pack_tree, Packed, TrieSize};
use crate::prompt::render;
use crate::request::{check_capability, parse_request, DecideError, Item};
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
}

#[derive(Clone, Debug, Serialize)]
pub struct OrderJob {
    pub perm: Vec<usize>,
    pub prompt: String,
    pub input_ids: Vec<u32>,
    pub letter_ids: Vec<u32>,
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
}

impl<B: Backend> Engine<B> {
    pub fn new(manifest: ModelManifest, backend: B) -> Result<Self> {
        let tok = BasalTokenizer::load(&manifest.dir.join("tokenizer.json"))?;
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
        Ok(e)
    }

    /// Precompute the prompt template up to the state for both languages: the longest common token prefix of two
    /// prompts that differ only in the state. Rows reuse it only on an exact token match.
    pub fn warm_static_prefixes(&mut self) -> Result<Vec<usize>> {
        let mut lens = Vec::new();
        for lang in [crate::prompt::Lang::Pl, crate::prompt::Lang::En] {
            let a = self.tok.encode(&render(&self.manifest.bos_token, "a", "q", &["x", "y"], lang))?;
            let b = self.tok.encode(&render(&self.manifest.bos_token, "b", "q", &["x", "y"], lang))?;
            let n = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            self.backend.add_prefix(&a[..n])?;
            lens.push(n);
        }
        Ok(lens)
    }

    pub fn prepare(&self, item: Item) -> Result<Prepared> {
        let mut p = prepare(&self.tok, &self.manifest.bos_token, item)?;
        if self.mark_state {
            p.state_len = self.state_len(&p)?;
            p.packed.state_len = p.state_len.min(p.packed.prefix_len);
        }
        Ok(p)
    }

    /// Tokens of the prompt up to and including the state: the common prefix of the item's prompt with two probe
    /// prompts that keep the state and change the question.
    fn state_len(&self, p: &Prepared) -> Result<usize> {
        let it = &p.item;
        let probe = |q: &str| self.tok.encode(&render(&self.manifest.bos_token, &it.state, q, &["x", "y"], it.lang));
        let (a, b) = (probe("A")?, probe("B")?);
        let ids = &p.orders[0].input_ids;
        Ok(a.iter().zip(&b).zip(ids).take_while(|((x, y), z)| x == y && x == z).count())
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
        let probs: Vec<Vec<f64>> =
            letter_logits.iter().map(|l| softmax(&l.iter().map(|v| *v as f64).collect::<Vec<_>>())).collect();
        let perms: Vec<Vec<usize>> = p.orders.iter().map(|o| o.perm.clone()).collect();
        let p_avg = canonical_average(&perms, &probs);
        let temperature = if p.item.uncalibrated { 1.0 } else { self.manifest.temperature(p.item.temp_kind()) };
        let p_cal = calibrate(&p_avg, temperature);
        ItemResult { letter_logits, probs, p_avg, temperature, p_cal }
    }

    /// Parse, validate and prepare a System One request without running the model.
    pub fn prepare_request(&self, body: &Value) -> Result<Vec<Prepared>, DecideError> {
        let req = parse_request(body)?;
        if req.model != self.manifest.name {
            return Err(DecideError::Validation(vec![crate::request::FieldError {
                loc: vec!["body".into(), "model".into()],
                msg: format!("unknown model {:?}; served: {:?}", req.model, self.manifest.name),
                kind: "unknown_model".into(),
            }]));
        }
        check_capability(&req.items, false)?;
        Ok(req.items.into_iter().map(|it| self.prepare(it)).collect::<Result<Vec<_>>>()?)
    }

    /// Validate a request and plan its questions: one readout for 2..=10 options, the grouped strategy for choice
    /// questions with 11..=255 options (when `large_choice` is on).
    pub fn plan_request(&self, body: &Value) -> Result<RequestPlan, DecideError> {
        self.plan_request_as(body, crate::request::Dialect::TypeSafe)
    }

    /// [`Engine::plan_request`] for an endpoint dialect (upstream: no model check, 2..10 options only).
    pub fn plan_request_as(&self, body: &Value, dialect: crate::request::Dialect) -> Result<RequestPlan, DecideError> {
        let req = crate::request::parse_request_as(body, dialect)?;
        if dialect == crate::request::Dialect::TypeSafe && req.model != self.manifest.name {
            return Err(DecideError::Validation(vec![crate::request::FieldError {
                loc: vec!["body".into(), "model".into()],
                msg: format!("unknown model {:?}; served: {:?}", req.model, self.manifest.name),
                kind: "unknown_model".into(),
            }]));
        }
        check_capability(&req.items, self.large_choice)?;
        let ev: Vec<&Item> = req.items.iter().filter(|it| it.evidence).collect();
        if !ev.is_empty() {
            let mut errs = Vec::new();
            for it in ev {
                let loc = vec!["body".into(), "questions".into(), it.name.clone().into(), "evidence".into()];
                if !self.backend.has_evidence() {
                    errs.push(crate::request::FieldError {
                        loc,
                        msg: "evidence is not available for this model (no evidence_head.pt)".into(),
                        kind: "unsupported_feature".into(),
                    });
                } else if it.options.len() > crate::prompt::MAX_OPTIONS {
                    errs.push(crate::request::FieldError {
                        loc,
                        msg: "evidence needs at most 10 options".into(),
                        kind: "unsupported_feature".into(),
                    });
                }
            }
            if !errs.is_empty() {
                return Err(DecideError::Unsupported(errs));
            }
        }
        let mut plan = self.plan_items(req.items)?;
        plan.evidence_limit = req.facts_limit;
        plan.dialect = dialect;
        Ok(plan)
    }

    /// Plan validated items (see [`Engine::plan_request`]).
    pub fn plan_items(&self, items: Vec<Item>) -> Result<RequestPlan> {
        let mut plan = RequestPlan {
            prepared: Vec::new(),
            questions: Vec::new(),
            evidence_limit: None,
            dialect: Default::default(),
        };
        for mut it in items {
            if let crate::request::Ext::Act(a) = &mut it.ext {
                a.calibrated = !self.manifest.temperatures.is_empty();
                a.threshold = a.max_error.as_ref().and_then(|k| {
                    self.manifest.thresholds.get(k).and_then(|t| t.get("confidence")).and_then(Value::as_f64)
                });
            }
            if let crate::request::Ext::Multi(m) = &it.ext {
                // one yes/no branch per label, each an ordinary noul readout
                let mut parts = Vec::with_capacity(m.labels.len());
                for (k, label) in m.labels.iter().enumerate() {
                    let mut sub = it.clone();
                    sub.name = format!("{}#{k}", it.name);
                    sub.kind = crate::request::QType::Noul;
                    sub.keys = vec!["true".into(), "false".into()];
                    sub.question = crate::request::multi_question(it.lang, &it.question, label);
                    sub.ext = crate::request::Ext::None;
                    parts.push(plan.prepared.len());
                    plan.prepared.push(self.prepare(sub)?);
                }
                plan.questions.push(QuestionPlan::Multi { item: it, parts });
                continue;
            }
            if it.options.len() <= crate::prompt::MAX_OPTIONS {
                plan.questions.push(QuestionPlan::Direct(plan.prepared.len()));
                plan.prepared.push(self.prepare(it)?);
            } else {
                let groups = crate::large_choice::groups(it.options.len());
                let mut parts = Vec::with_capacity(groups.len());
                for g in &groups {
                    let mut sub = it.clone();
                    sub.keys = g.iter().map(|&i| it.keys[i].clone()).collect();
                    sub.options = g.iter().map(|&i| it.options[i].clone()).collect();
                    parts.push(plan.prepared.len());
                    plan.prepared.push(self.prepare(sub)?);
                }
                plan.questions.push(QuestionPlan::Groups { item: it, groups, parts });
            }
        }
        Ok(plan)
    }

    /// Run a plan: one round for direct questions; grouped questions get a second, final round with their strongest
    /// candidates. Returns the final (calibrated) distribution of every planned question.
    /// Evidence spans of one question (upstream `Evidence.spans`): a separate forward of its prompt in the original
    /// option order; empty when no token lies inside the state.
    fn evidence_for(&mut self, item: &Item, limit: Option<usize>) -> Result<Value> {
        let Some(inp) = crate::evidence::input(&self.tok, &self.manifest.bos_token, item)? else {
            return Ok(Value::Array(vec![]));
        };
        let (ls, le) = self.backend.evidence_scores(&inp.ids, inp.t0, inp.t1)?;
        Ok(crate::evidence::spans(&ls, &le, &inp, &item.state, limit))
    }

    pub fn run_plan(&mut self, plan: &RequestPlan) -> Result<PlanOutput> {
        Ok(self.run_plans(&[plan])?.remove(0))
    }

    /// Run the plans of several requests together: the questions of all plans share the forwards of each round
    /// (first round, then the final round of grouped large choices), as a server batches waiting requests.
    pub fn run_plans(&mut self, plans: &[&RequestPlan]) -> Result<Vec<PlanOutput>> {
        let mut offs = Vec::with_capacity(plans.len());
        let mut all = Vec::new();
        for p in plans {
            offs.push(all.len());
            all.extend(p.prepared.iter().cloned());
        }
        let results = self.run(&all, self.request_batching)?;
        // first-round fits and final groups
        let mut finals_prep = Vec::new();
        let mut fits = Vec::new();
        for (plan, &off) in plans.iter().zip(&offs) {
            for q in &plan.questions {
                if let QuestionPlan::Groups { item, groups, parts } = q {
                    let targets: Vec<Vec<f64>> = parts.iter().map(|&i| results[off + i].p_avg.clone()).collect();
                    let first = crate::large_choice::fit_joint(item.options.len(), groups, &targets);
                    let fin = crate::large_choice::finalists(&first);
                    let mut sub = item.clone();
                    sub.keys = fin.iter().map(|&i| item.keys[i].clone()).collect();
                    sub.options = fin.iter().map(|&i| item.options[i].clone()).collect();
                    finals_prep.push(self.prepare(sub)?);
                    fits.push((targets, fin));
                }
            }
        }
        let final_results =
            if finals_prep.is_empty() { vec![] } else { self.run(&finals_prep, self.request_batching)? };
        let mut fi = 0;
        let mut outs = Vec::with_capacity(plans.len());
        for (plan, &off) in plans.iter().zip(&offs) {
            let mut finals = Vec::with_capacity(plan.questions.len());
            let mut final_round_tokens = 0;
            for q in &plan.questions {
                match q {
                    QuestionPlan::Direct(i) => {
                        finals.push((plan.prepared[*i].item.clone(), results[off + *i].p_cal.clone()))
                    }
                    QuestionPlan::Multi { item, parts } => {
                        finals.push((item.clone(), parts.iter().map(|&i| results[off + i].p_cal[0]).collect()))
                    }
                    QuestionPlan::Groups { item, groups, .. } => {
                        let (mut targets, fin) = fits[fi].clone();
                        let mut all_groups = groups.clone();
                        all_groups.push(fin);
                        targets.push(final_results[fi].p_avg.clone());
                        final_round_tokens += finals_prep[fi].orders.iter().map(|o| o.input_ids.len()).sum::<usize>();
                        fi += 1;
                        let joint = crate::large_choice::fit_joint(item.options.len(), &all_groups, &targets);
                        finals.push((item.clone(), calibrate(&joint, self.manifest.temperature(item.temp_kind()))));
                    }
                }
            }
            let mut evidence = Vec::with_capacity(finals.len());
            for (item, _) in &finals {
                evidence.push(if item.evidence { Some(self.evidence_for(item, plan.evidence_limit)?) } else { None });
            }
            outs.push(PlanOutput { finals, final_round_tokens, evidence });
        }
        Ok(outs)
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
    let k = item.options.len();
    let mut orders = Vec::new();
    for perm in decision::orders(k) {
        let opts: Vec<&str> = perm.iter().map(|&j| item.options[j].as_str()).collect();
        let prompt = render(bos, &item.state, &item.question, &opts, item.lang);
        let input_ids = tok.encode(&prompt)?;
        let Some(letter_ids) = tok.letter_ids(&prompt, &input_ids, k)? else {
            bail!("option letters are not single tokens for this tokenizer ({k} options)");
        };
        orders.push(OrderJob { perm, prompt, input_ids, letter_ids });
    }
    let packed = pack(&orders.iter().map(|o| o.input_ids.clone()).collect::<Vec<_>>());
    Ok(Prepared { item, orders, packed, state_len: 0 })
}

/// Answers of a planned request: final distribution per question, and the prompt tokens of the final round of
/// grouped large choices (counted in `usage.input_tokens`).
pub struct PlanOutput {
    pub finals: Vec<(Item, Vec<f64>)>,
    pub final_round_tokens: usize,
    /// Evidence spans per question (same order as `finals`), for questions that asked for them.
    pub evidence: Vec<Option<Value>>,
}

/// How one question of a request is answered.
#[derive(Clone, Debug)]
pub enum QuestionPlan {
    /// One letter readout (index into `RequestPlan::prepared`).
    Direct(usize),
    /// Grouped strategy for 11..=255 choice options: `parts[k]` answers `groups[k]`.
    Groups { item: Item, groups: Vec<Vec<usize>>, parts: Vec<usize> },
    /// `multi`: `parts[k]` is the yes/no branch of label k.
    Multi { item: Item, parts: Vec<usize> },
}

#[derive(Clone, Debug)]
pub struct RequestPlan {
    pub prepared: Vec<Prepared>,
    pub questions: Vec<QuestionPlan>,
    /// Characters of the request's own state when facts were appended (evidence spans stay inside it).
    pub evidence_limit: Option<usize>,
    /// Answer conventions of the endpoint.
    pub dialect: crate::request::Dialect,
}

pub fn plan_response(model: &str, plan: &RequestPlan, out: &PlanOutput) -> Result<Value, DecideError> {
    let mut answers = serde_json::Map::new();
    for ((it, p), ev) in out.finals.iter().zip(&out.evidence) {
        let a = crate::decision::answer_checked(it, p, plan.dialect).map_err(|msg| {
            DecideError::Validation(vec![crate::request::FieldError {
                loc: vec!["body".into(), "questions".into(), it.name.clone().into()],
                msg,
                kind: "value_error".into(),
            }])
        })?;
        let mut a = a;
        if let (Some(ev), Some(obj)) = (ev, a.as_object_mut()) {
            obj.insert("evidence".into(), ev.clone());
        }
        answers.insert(it.name.clone(), a);
    }
    let branches: usize = plan
        .questions
        .iter()
        .map(|q| match q {
            QuestionPlan::Multi { parts, .. } => parts.len(),
            _ => 1,
        })
        .sum();
    let input_tokens: usize =
        plan.prepared.iter().flat_map(|p| &p.orders).map(|o| o.input_ids.len()).sum::<usize>() + out.final_round_tokens;
    Ok(json!({"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0,
                                                             "questions": plan.questions.len(), "branches": branches}}))
}

pub fn response(model: &str, prepared: &[Prepared], results: &[ItemResult]) -> Value {
    let answers: serde_json::Map<String, Value> =
        prepared.iter().zip(results).map(|(p, r)| (p.item.name.clone(), answer(&p.item, &r.p_cal))).collect();
    // Upstream counts the tokens of every option-order prompt; no text is generated.
    let input_tokens: usize = prepared.iter().flat_map(|p| &p.orders).map(|o| o.input_ids.len()).sum();
    json!({"model": model, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}})
}
