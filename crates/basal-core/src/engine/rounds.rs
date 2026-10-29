//! The decision pipeline's existing phase boundaries, with scatter slots for partial admission.

use super::*;

/// Opaque identity of one primitive in the current dependency round of a request.
/// All option orders of that primitive are admitted together. Identities expire on round advancement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadyBranch {
    round: usize,
    index: usize,
}

/// Owned online text request. The original plan and completion state outlive every admitted branch.
/// Media and active early-exit requests must use [`Engine::run_plans`] instead.
pub struct RequestExecution {
    plan: RequestPlan,
    decision: DecisionRounds,
    continuation: Option<TaskContinuation>,
    first: Option<PlanOutput>,
    completed: bool,
    round: usize,
}

pub(super) struct TaskContinuation {
    pub(super) plan: RequestPlan,
    task: crate::tasks::TaskPlan,
    raw: Value,
}

impl RequestExecution {
    /// Original immutable plan, including dialect, task expansion and usage inputs.
    pub fn plan(&self) -> &RequestPlan {
        &self.plan
    }

    /// Currently unscheduled primitives, in original prepared order. This list is bounded by the validated plan
    /// and option limits; it contains no independent request, completion sender or permit per primitive.
    pub fn ready(&self) -> impl Iterator<Item = (ReadyBranch, &Prepared)> {
        let round = self.round;
        self.decision.prepared().iter().enumerate().filter_map(move |(index, p)| {
            (!self.completed && self.decision.slots[index].is_none()).then_some((ReadyBranch { round, index }, p))
        })
    }
}

impl<B: Backend> Engine<B> {
    /// Start partial admission of a text/off request. This operation does no backend work. The caller owns one
    /// request reservation throughout, calls [`Engine::run_ready`] then [`Engine::advance_request`], and drops the
    /// state only after admitted synchronous work has drained. Empty plans may be advanced without a readout.
    pub fn begin_request(&self, plan: RequestPlan) -> Result<RequestExecution> {
        ensure!(plan.media.is_none(), "media requests require atomic execution");
        ensure!(plan.early_exit.as_deref().is_none_or(|p| p == "off"), "active early-exit requires atomic execution");
        let decision = DecisionRounds::new(&[&plan]);
        Ok(RequestExecution { plan, decision, continuation: None, first: None, completed: false, round: 0 })
    }

    /// Execute only the selected ready primitives with Budget packing, then scatter their results. Each pair
    /// identifies an owner in `requests` and one of its current ready branches. Duplicate/stale identities are
    /// rejected before backend work; no phase advances until all of its primitive slots are complete.
    pub fn run_ready(
        &mut self,
        requests: &mut [&mut RequestExecution],
        selected: &[(usize, ReadyBranch)],
    ) -> Result<()> {
        self.check_stop()?;
        let mut prepared = Vec::with_capacity(selected.len());
        let mut seen = std::collections::HashSet::with_capacity(selected.len());
        let mut policy = None;
        for &(owner, branch) in selected {
            let request = requests.get(owner).context("invalid ready request owner")?;
            ensure!(!request.completed && request.round == branch.round, "stale ready branch");
            ensure!(seen.insert((owner, branch.index)), "duplicate ready branch");
            let p = request.decision.prepared().get(branch.index).context("invalid ready branch")?;
            ensure!(request.decision.slots[branch.index].is_none(), "ready branch already executed");
            let current = request.continuation.as_ref().map_or(&request.plan, |c| &c.plan);
            if prepared.is_empty() {
                policy = current.early_exit.as_deref();
            } else {
                ensure!(policy == current.early_exit.as_deref(), "incompatible ready exit policies");
            }
            prepared.push(p.clone());
        }
        let results = self.run_with_policy(&prepared, Batching::Budget, policy)?;
        ensure!(results.len() == selected.len(), "ready readout count mismatch");
        for (&(owner, branch), result) in selected.iter().zip(results) {
            requests[owner].decision.slots[branch.index] = Some(result);
        }
        Ok(())
    }

    /// Advance completed dependency rounds, preserving whole-request context refusal and task assembly. Thinking
    /// generation and evidence run synchronously only after P1 completes, using their existing isolated paths.
    /// Returns the final output once; an unfinished round returns `None` without executing missing branches.
    pub fn advance_request(&mut self, request: &mut RequestExecution) -> Result<Option<PlanOutput>> {
        ensure!(!request.completed, "request already completed");
        loop {
            let plan = request.continuation.as_ref().map_or(&request.plan, |c| &c.plan);
            let before = request.decision.phase_number;
            let output = request.decision.advance(self, &[plan])?;
            if request.decision.phase_number != before {
                request.round = request.round.checked_add(1).context("request round overflow")?;
            }
            let Some(mut outputs) = output else { return Ok(None) };
            let mut output = outputs.remove(0);
            if let Some(next) = request.continuation.take() {
                let mut first = request.first.take().context("missing first task output")?;
                self.finish_task(&request.plan, &mut first, next, output)?;
                request.completed = true;
                return Ok(Some(first));
            }
            self.apply_thinking(&[&request.plan], std::slice::from_mut(&mut output))?;
            if let Some(next) = self.task_continuation(&request.plan, &mut output)? {
                request.decision = DecisionRounds::new(&[&next.plan]);
                request.continuation = Some(next);
                request.first = Some(output);
                request.round = request.round.checked_add(1).context("request round overflow")?;
            } else {
                request.completed = true;
                return Ok(Some(output));
            }
        }
    }

    pub(super) fn task_continuation(
        &self,
        plan: &RequestPlan,
        output: &mut PlanOutput,
    ) -> Result<Option<TaskContinuation>> {
        if !output.context_errors.is_empty() || output.thinking_error.is_some() {
            return Ok(None);
        }
        let Some(mut task) = plan.tasks.clone() else { return Ok(None) };
        let raw = plan_response(&self.manifest.name, plan, output)?;
        if let Some(mut body) = task.second_round(&raw, self.manifest.prompt_format.max_options())? {
            body["model"] = self.manifest.name.clone().into();
            if let Some(policy) = &plan.early_exit {
                body["early_exit"] = policy.clone().into();
            }
            if self.thinking.is_some() {
                body["think"] = "off".into();
            }
            match self.plan_decision_request_as(&body, plan.dialect) {
                Ok(plan) => return Ok(Some(TaskContinuation { plan, task, raw })),
                Err(DecideError::Unsupported(errors)) if errors.iter().any(|e| e.kind == "context_length_exceeded") => {
                    output.context_errors = errors;
                }
                Err(error) => output.task_error = Some(error.to_string()),
            }
        } else {
            let empty = serde_json::Map::new();
            let thresholds = if plan.media.is_some() { &empty } else { &self.manifest.task_thresholds };
            output.task_response = Some(task.finish(raw, None, thresholds)?);
        }
        Ok(None)
    }

    pub(super) fn finish_task(
        &self,
        plan: &RequestPlan,
        first: &mut PlanOutput,
        next: TaskContinuation,
        output: PlanOutput,
    ) -> Result<()> {
        if !output.context_errors.is_empty() {
            first.context_errors = output.context_errors;
            return Ok(());
        }
        let response = plan_response(&self.manifest.name, &next.plan, &output)?;
        let mut raw = next.raw;
        for key in ["input_tokens", "branches", "second_order_branches"] {
            if let Some(n) = response["usage"][key].as_u64() {
                let old = raw["usage"][key].as_u64().unwrap_or(0);
                raw["usage"][key] = old.saturating_add(n).into();
            }
        }
        if let Some(n) = response["usage"]["longest_prompt_tokens"].as_u64() {
            raw["usage"]["longest_prompt_tokens"] =
                n.max(raw["usage"]["longest_prompt_tokens"].as_u64().unwrap_or(0)).into();
        }
        if response["usage"]["truncated"]["dropped_tokens"].as_u64().unwrap_or(0)
            > raw["usage"]["truncated"]["dropped_tokens"].as_u64().unwrap_or(0)
        {
            raw["usage"]["truncated"] = response["usage"]["truncated"].clone();
        }
        let empty = serde_json::Map::new();
        let thresholds = if plan.media.is_some() { &empty } else { &self.manifest.task_thresholds };
        first.task_response = Some(next.task.finish(raw, Some(&response), thresholds)?);
        Ok(())
    }
}

enum Phase {
    First,
    Reverse(Vec<usize>),
    Groups(Vec<(usize, Vec<usize>, bool)>),
    Done,
}

/// Also drives the whole-plan path: one state for all plans preserves flattening, phase order and arithmetic.
pub(super) struct DecisionRounds {
    offs: Vec<usize>,
    all: Vec<Prepared>,
    results: Vec<ItemResult>,
    context_errors: Vec<Vec<crate::request::FieldError>>,
    second_tokens: Vec<usize>,
    large: Vec<Large>,
    work: Vec<Prepared>,
    slots: Vec<Option<ItemResult>>,
    phase: Phase,
    phase_number: usize,
}

impl DecisionRounds {
    pub(super) fn new(plans: &[&RequestPlan]) -> Self {
        let mut offs = Vec::with_capacity(plans.len());
        let mut all = Vec::new();
        for p in plans {
            offs.push(all.len());
            all.extend(p.prepared.iter().cloned());
        }
        let n = all.len();
        Self {
            offs,
            all,
            results: Vec::new(),
            context_errors: vec![Vec::new(); plans.len()],
            second_tokens: vec![0; n],
            large: Vec::new(),
            work: Vec::new(),
            slots: vec![None; n],
            phase: Phase::First,
            phase_number: 0,
        }
    }

    pub(super) fn prepared(&self) -> &[Prepared] {
        if matches!(self.phase, Phase::First) {
            &self.all
        } else {
            &self.work
        }
    }

    pub(super) fn scatter_all(&mut self, results: Vec<ItemResult>) -> Result<()> {
        ensure!(results.len() == self.slots.len(), "decision readout count mismatch");
        self.slots = results.into_iter().map(Some).collect();
        Ok(())
    }

    pub(super) fn advance<B: Backend>(
        &mut self,
        engine: &mut Engine<B>,
        plans: &[&RequestPlan],
    ) -> Result<Option<Vec<PlanOutput>>> {
        loop {
            if self.slots.iter().any(Option::is_none) {
                return Ok(None);
            }
            engine.check_stop()?;
            let res = std::mem::take(&mut self.slots)
                .into_iter()
                .map(|r| r.context("missing round result"))
                .collect::<Result<Vec<_>>>()?;
            let policy = plans.first().and_then(|p| p.early_exit.as_deref());
            match std::mem::replace(&mut self.phase, Phase::Done) {
                Phase::First => {
                    self.results = res;
                    if engine.manifest.orders == OptionOrders::Auto {
                        let mut reverse = Vec::new();
                        let mut owners = Vec::new();
                        for (pi, (plan, &off)) in plans.iter().zip(&self.offs).enumerate() {
                            let queued = reverse.len();
                            for (i, p) in self.all.iter().enumerate().skip(off).take(plan.prepared.len()) {
                                if p.orders.len() != 1 {
                                    continue;
                                }
                                if engine.auto_orders.uncertain(p.item.temp_kind(), &self.results[i].p_cal) {
                                    let mut redo = prepare_protocol(
                                        &engine.tok,
                                        &engine.manifest.bos_token,
                                        p.item.clone(),
                                        engine.manifest.prompt_format,
                                        engine.manifest.chat_style,
                                        engine.manifest.layout,
                                        None,
                                        None,
                                    )?;
                                    redo.orders.remove(0);
                                    if let Some(error) = engine.prepared_context_error(&redo) {
                                        reverse.truncate(queued);
                                        owners.truncate(queued);
                                        self.context_errors[pi].push(error);
                                        break;
                                    }
                                    redo.packed =
                                        pack(&redo.orders.iter().map(|o| o.input_ids.clone()).collect::<Vec<_>>());
                                    if engine.mark_state {
                                        redo.state_len = engine.state_len(&redo)?;
                                        redo.packed.state_len = redo.state_len.min(redo.packed.prefix_len);
                                    }
                                    owners.push(i);
                                    reverse.push(redo);
                                }
                            }
                        }
                        self.work = reverse;
                        self.phase = Phase::Reverse(owners);
                    } else {
                        self.init_large(engine, plans);
                        self.next_groups(engine)?;
                    }
                }
                Phase::Reverse(owners) => {
                    for ((i, p2), r2) in owners.into_iter().zip(&self.work).zip(res) {
                        self.second_tokens[i] = p2.orders[0].input_ids.len();
                        let mut both = self.all[i].clone();
                        both.orders.extend(p2.orders.clone());
                        let mut logits = self.results[i].letter_logits.clone();
                        logits.extend(r2.letter_logits);
                        self.results[i] = engine.finish_readout(&both, logits, policy.is_some_and(|s| s != "off"));
                    }
                    self.init_large(engine, plans);
                    self.next_groups(engine)?;
                }
                Phase::Groups(owners) => {
                    for l in &mut self.large {
                        if l.finalists.is_none() && self.context_errors[l.owner].is_empty() {
                            l.last.clear();
                            l.rounds += 1;
                        }
                    }
                    for (((li, g, last), r), p) in owners.into_iter().zip(res).zip(&self.work) {
                        let l = &mut self.large[li];
                        l.tokens += p.orders.iter().map(|o| o.input_ids.len()).sum::<usize>();
                        l.longest = p.orders.iter().map(|o| o.input_ids.len()).fold(l.longest, usize::max);
                        l.prompts += p.orders.len();
                        l.last.push(l.groups.len());
                        if last {
                            l.finalists = Some(g.clone());
                            l.final_p = r.p_cal.clone();
                        }
                        l.groups.push(g);
                        l.targets.push(r.p_avg);
                    }
                    self.next_groups(engine)?;
                }
                Phase::Done => bail!("decision rounds already completed"),
            }
            self.phase_number = self.phase_number.checked_add(1).context("decision round overflow")?;
            if matches!(self.phase, Phase::Done) {
                return self.finish(engine, plans).map(Some);
            }
            self.slots = vec![None; self.work.len()];
        }
    }

    fn init_large<B: Backend>(&mut self, engine: &Engine<B>, plans: &[&RequestPlan]) {
        for (pi, (plan, &off)) in plans.iter().zip(&self.offs).enumerate() {
            if !self.context_errors[pi].is_empty() {
                continue;
            }
            for q in &plan.questions {
                if let QuestionPlan::Groups { item, groups, parts } = q {
                    self.large.push(Large {
                        owner: pi,
                        item: item.clone(),
                        groups: groups.clone(),
                        targets: parts.iter().map(|&i| self.results[off + i].p_avg.clone()).collect(),
                        last: (0..groups.len()).collect(),
                        finalists: None,
                        final_p: vec![],
                        tokens: 0,
                        longest: 0,
                        prompts: 0,
                        rounds: 1,
                        first_round: engine.trace_large_choice.then(|| {
                            let targets: Vec<Vec<f64>> =
                                parts.iter().map(|&i| self.results[off + i].p_avg.clone()).collect();
                            crate::large_choice::fit_joint(item.options.len(), groups, &targets)
                        }),
                    });
                }
            }
        }
    }

    fn next_groups<B: Backend>(&mut self, engine: &Engine<B>) -> Result<()> {
        let strategy = engine.large_choice_strategy;
        let mut prep = Vec::new();
        let mut owners: Vec<(usize, Vec<usize>, bool)> = Vec::new();
        for (li, l) in self.large.iter().enumerate() {
            if l.finalists.is_some() || !self.context_errors[l.owner].is_empty() {
                continue;
            }
            engine.check_stop()?;
            let (round, last) = match crate::large_choice::next_round(
                &strategy,
                l.item.options.len(),
                &l.groups,
                &l.targets,
                &l.last,
                l.rounds,
            ) {
                crate::large_choice::Round::Final(f) => (vec![f], true),
                crate::large_choice::Round::Groups(g) => (g, false),
            };
            for g in round {
                engine.check_stop()?;
                let mut sub = l.item.clone();
                sub.keys = g.iter().map(|&i| l.item.keys[i].clone()).collect();
                sub.options = g.iter().map(|&i| l.item.options[i].clone()).collect();
                let p = engine.prepare_orders(sub, if last { 2 } else { strategy.screen_orders })?;
                if let Some(error) = engine.prepared_context_error(&p) {
                    while owners.last().is_some_and(|&(queued, _, _)| self.large[queued].owner == l.owner) {
                        owners.pop();
                        prep.pop();
                    }
                    self.context_errors[l.owner].push(error);
                    break;
                }
                prep.push(p);
                owners.push((li, g, last));
            }
        }
        self.phase = if prep.is_empty() { Phase::Done } else { Phase::Groups(owners) };
        self.work = prep;
        Ok(())
    }

    fn finish<B: Backend>(&mut self, engine: &mut Engine<B>, plans: &[&RequestPlan]) -> Result<Vec<PlanOutput>> {
        let strategy = engine.large_choice_strategy;
        let mut li = 0;
        let mut outs = Vec::with_capacity(plans.len());
        for (pi, (plan, &off)) in plans.iter().zip(&self.offs).enumerate() {
            if !self.context_errors[pi].is_empty() {
                while self.large.get(li).is_some_and(|l| l.owner == pi) {
                    li += 1;
                }
                outs.push(PlanOutput {
                    context_errors: std::mem::take(&mut self.context_errors[pi]),
                    ..Default::default()
                });
                continue;
            }
            let mut finals = Vec::with_capacity(plan.questions.len());
            let end = off + plan.prepared.len();
            let second_order_branches = (engine.manifest.orders == OptionOrders::Auto)
                .then(|| self.second_tokens[off..end].iter().filter(|n| **n > 0).count());
            let mut final_round_tokens = self.second_tokens[off..end].iter().sum();
            let mut longest_prompt_tokens = plan
                .prepared
                .iter()
                .flat_map(|p| &p.orders)
                .map(|o| o.input_ids.len())
                .chain(self.second_tokens[off..end].iter().copied())
                .max()
                .unwrap_or(0);
            let mut extra_prompts = 0;
            let mut finalists = Vec::with_capacity(plan.questions.len());
            let mut first_round = Vec::with_capacity(plan.questions.len());
            for q in &plan.questions {
                let mut fin = None;
                let mut first = None;
                match q {
                    QuestionPlan::Direct(i) => {
                        finals.push((plan.prepared[*i].item.clone(), self.results[off + *i].p_cal.clone()))
                    }
                    QuestionPlan::Fixed(item) => finals.push((item.clone(), vec![1.0])),
                    QuestionPlan::Multi { item, parts } => {
                        finals.push((item.clone(), parts.iter().map(|&i| self.results[off + i].p_cal[0]).collect()))
                    }
                    QuestionPlan::Groups { item, .. } => {
                        let l = &self.large[li];
                        li += 1;
                        final_round_tokens += l.tokens;
                        longest_prompt_tokens = longest_prompt_tokens.max(l.longest);
                        extra_prompts += l.prompts;
                        let f = l.finalists.clone().unwrap_or_default();
                        let t = engine.manifest.temperature(item.temp_kind());
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
                engine.check_stop()?;
            }
            for (item, _) in &finals {
                evidence.push(if item.evidence { Some(engine.evidence_for(item, plan.evidence_limit)?) } else { None });
            }
            outs.push(PlanOutput {
                longest_prompt_tokens,
                finals,
                final_round_tokens,
                extra_prompts,
                finalists,
                first_round,
                evidence,
                second_order_branches,
                ..Default::default()
            });
        }
        Ok(outs)
    }
}
