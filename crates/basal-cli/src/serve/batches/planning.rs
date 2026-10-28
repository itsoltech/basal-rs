//! Dedicated CPU planning threads for offline records (validation and background execution).
//!
//! `planning_workers` std threads take jobs from one bounded queue; no Tokio worker and no blocking-pool thread
//! plans records, and no backend is touched. Each job runs under a fresh `PlanBudget::new(R)` (the slot of its
//! thread): the raw line, the JSON value and the planner's own reservations (media decoders, features, tokens)
//! must fit, with no waiting inside the slot. Queued jobs hold only their raw line (which the record's `host_bound`
//! covers in the execution pool `P`); `R` is never reserved for them. Planning applies the online rules of
//! `/v1/basal`: the record envelope of `validation.rs`, the online expanded-input limit and `Dialect::Upstream`.

use std::sync::{Arc, Mutex, PoisonError};

use basal_core::planner::{self, PlanBudget, PlanError, PlanStage};
use basal_core::request::Dialect;
use basal_core::{RequestPlanner, StopReason, StopToken};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot};

use super::dto::{codes, result_line_bound};
use super::profile::ResolvedProfile;
use super::validation::{json_memory_bound, parse_record, InputError, InputLimits};

/// One record line to plan.
pub(crate) struct PlanJob {
    pub(crate) profile: Arc<ResolvedProfile>,
    /// Raw bytes of the line in the original JSONL, without its separator.
    pub(crate) line: Vec<u8>,
    /// Keep the plan (execution) or drop it after measuring (validation).
    pub(crate) keep_plan: bool,
    pub(crate) stop: Option<StopToken>,
    /// Effective model of a record without a string `model` field: the pinned catalog's default model, when known.
    /// `None` refuses such records unless the expected model is known to be the default.
    pub(crate) default_model: Option<String>,
}

/// A planned record.
pub(crate) struct PlannedRecord {
    pub(crate) custom_id: String,
    /// Effective model (always the profile's model).
    pub(crate) model: String,
    pub(crate) plan: Option<basal_core::engine::RequestPlan>,
    /// DRR cost: ns of device-gate time estimated as packed tokens (at least 1) × `cost_ns_per_token`.
    pub(crate) cost: u64,
    /// Upper bound of the finished result line incl. LF.
    pub(crate) result_bound: u64,
    /// `line_len + retained_bound + exec_bound`, at most `P`.
    pub(crate) host_bound: u64,
}

/// Why a record was not planned.
#[derive(Debug)]
pub(crate) enum PlanFailure {
    /// The record's own fault, terminal (code of `errors.data` / record lines).
    Data { code: &'static str, message: String },
    /// Stopped (lease lost, cancel, shutdown); not the record's fault.
    Interrupted(StopReason),
    /// The profile cannot plan bounded records (no host staging) or the pool stopped.
    Unavailable(String),
}

impl PlanFailure {
    fn data(code: &'static str, message: impl Into<String>) -> Self {
        Self::Data { code, message: message.into() }
    }
}

/// Map an envelope/framing error of one line to its `errors.data` code.
pub(crate) fn input_issue(error: &InputError) -> Option<(&'static str, String)> {
    let code = match error {
        InputError::LineTooLarge => codes::LINE_TOO_LARGE,
        InputError::TooManyRecords => codes::TOO_MANY_RECORDS,
        InputError::EmptyLine => codes::EMPTY_LINE,
        InputError::JsonDepth | InputError::JsonNodes | InputError::JsonMemory => codes::BUDGET_EXCEEDED,
        InputError::Json(_) => codes::INVALID_JSON_LINE,
        InputError::CustomId => codes::INVALID_CUSTOM_ID,
        InputError::Method => codes::INVALID_METHOD,
        InputError::Endpoint => codes::INVALID_URL,
        InputError::Body => codes::INVALID_BODY,
        InputError::Streaming => codes::STREAM_UNSUPPORTED,
        // Storage or allocation failures are not the client's data.
        InputError::Io(_) | InputError::PositionOverflow | InputError::Allocation => return None,
    };
    let message = match error {
        InputError::Json(e) => format!("invalid JSON: {e}"),
        other => other.to_string(),
    };
    Some((code, message))
}

/// Limits of one planning job.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlanLimits {
    /// Slot `R` of a planning thread.
    pub(crate) slot_bytes: u64,
    /// Execution pool `P`: no record may need more.
    pub(crate) pool_bytes: u64,
    pub(crate) input: InputLimits,
}

impl PlanLimits {
    /// Line, JSON depth and node limits that fit the slot: a line holds at most a sixth of `R` (raw bytes plus
    /// the JSON value bound of `validation.rs`), nodes at most `R / 512`.
    pub(crate) fn new(slot_bytes: u64, pool_bytes: u64, max_records: u64, max_file_bytes: u64) -> Self {
        let max_line = (slot_bytes / 6).min(max_file_bytes);
        Self {
            slot_bytes,
            pool_bytes,
            input: InputLimits {
                max_line_bytes: usize::try_from(max_line).unwrap_or(usize::MAX),
                max_records,
                max_json_depth: 128,
                max_json_nodes: usize::try_from(slot_bytes / 512).unwrap_or(usize::MAX),
            },
        }
    }
}

enum Work {
    Plan(PlanJob, oneshot::Sender<Result<PlannedRecord, PlanFailure>>),
    /// Discovery: the `model` field of a line (after the same preflight).
    Model(Vec<u8>, oneshot::Sender<Result<Option<String>, PlanFailure>>),
    /// End one thread.
    Stop,
}

/// The planning threads. Dropping the pool closes the queue; threads finish their job and exit.
pub(crate) struct PlanningPool {
    tx: mpsc::Sender<Work>,
    limits: PlanLimits,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for PlanningPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanningPool").field("limits", &self.limits).finish_non_exhaustive()
    }
}

impl PlanningPool {
    /// Start `workers` threads with a queue of `2 × workers` jobs.
    pub(crate) fn start(workers: u32, limits: PlanLimits) -> std::io::Result<Self> {
        let workers = workers.max(1) as usize;
        let (tx, rx) = mpsc::channel::<Work>(2 * workers);
        let rx = Arc::new(Mutex::new(rx));
        let mut threads = Vec::with_capacity(workers);
        for i in 0..workers {
            let rx = Arc::clone(&rx);
            threads.push(std::thread::Builder::new().name(format!("batch-plan-{i}")).spawn(move || loop {
                // The lock is held only while waiting for the next job, never while planning.
                let next = rx.lock().unwrap_or_else(PoisonError::into_inner).blocking_recv();
                match next {
                    Some(Work::Plan(job, reply)) => {
                        let _ = reply.send(plan_line(&job, &limits));
                    }
                    Some(Work::Model(line, reply)) => {
                        let _ = reply.send(model_of(&line, &limits));
                    }
                    Some(Work::Stop) | None => break,
                }
            })?);
        }
        Ok(Self { tx, limits, threads: Mutex::new(threads) })
    }

    pub(crate) fn limits(&self) -> PlanLimits {
        self.limits
    }

    /// Plan one record line on a planning thread (waits for queue space; cancel-safe).
    pub(crate) async fn plan(&self, job: PlanJob) -> Result<PlannedRecord, PlanFailure> {
        self.submit(|reply| Work::Plan(job, reply)).await
    }

    /// The `model` field of a record line, for discovery; `None` when absent or not a string (the default model).
    pub(crate) async fn model_of(&self, line: Vec<u8>) -> Result<Option<String>, PlanFailure> {
        self.submit(|reply| Work::Model(line, reply)).await
    }

    /// Queue the work built around a fresh reply channel and wait for its answer.
    async fn submit<T>(
        &self,
        work: impl FnOnce(oneshot::Sender<Result<T, PlanFailure>>) -> Work,
    ) -> Result<T, PlanFailure> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(work(reply)).await.map_err(|_| PlanFailure::Unavailable("planning threads stopped".into()))?;
        rx.await.map_err(|_| PlanFailure::Unavailable("planning thread stopped".into()))?
    }

    /// Stop the threads after the jobs already queued and join them (on the blocking pool).
    pub(crate) async fn shutdown(&self) {
        let threads = std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for _ in 0..threads.len() {
            if self.tx.send(Work::Stop).await.is_err() {
                break;
            }
        }
        let _ = tokio::task::spawn_blocking(move || {
            for thread in threads {
                let _ = thread.join();
            }
        })
        .await;
    }
}

/// The record's envelope parsed under the slot: preflight, reservations for the line and its JSON value, parse.
fn parse_in_slot(
    line: &[u8],
    limits: &PlanLimits,
    budget: &PlanBudget,
) -> Result<super::validation::InputRecord, PlanFailure> {
    let issue = |e: InputError| match input_issue(&e) {
        Some((code, message)) => PlanFailure::Data { code, message },
        None => PlanFailure::Unavailable(e.to_string()),
    };
    budget.reserve(PlanStage::Line, line.len() as u64).map_err(budget_issue)?.retain();
    let json = json_memory_bound(line, limits.input).map_err(issue)?;
    budget.reserve(PlanStage::Json, json as u64).map_err(budget_issue)?.retain();
    parse_record(line).map_err(issue)
}

fn budget_issue(e: planner::BudgetExceeded) -> PlanFailure {
    PlanFailure::data(
        codes::BUDGET_EXCEEDED,
        format!(
            "planning stage {} needs {} bytes; the slot has {} of {}",
            e.stage.as_str(),
            e.requested,
            e.limit.saturating_sub(e.used),
            e.limit
        ),
    )
}

fn model_of(line: &[u8], limits: &PlanLimits) -> Result<Option<String>, PlanFailure> {
    let budget = PlanBudget::new(limits.slot_bytes);
    let record = parse_in_slot(line, limits, &budget)?;
    Ok(record.body.get("model").and_then(Value::as_str).map(str::to_string))
}

fn plan_line(job: &PlanJob, limits: &PlanLimits) -> Result<PlannedRecord, PlanFailure> {
    let profile = &job.profile;
    let staging = profile
        .host_staging
        .as_ref()
        .ok_or_else(|| PlanFailure::Unavailable(format!("{}: no host staging bound", profile.name)))?;
    if let Some(reason) = job.stop.as_ref().and_then(StopToken::reason) {
        return Err(PlanFailure::Interrupted(reason));
    }
    let budget = PlanBudget::new(limits.slot_bytes);
    let record = parse_in_slot(&job.line, limits, &budget)?;
    // The same model rule as online (`serve::resolve_model`): a missing or non-string model is the default one.
    let known = |name: &str| name == profile.name;
    let model = match super::super::resolve_model(&record.body, job.default_model.as_deref().unwrap_or(""), known) {
        Ok(m) => m.to_string(),
        Err("") => {
            return Err(PlanFailure::data(
                codes::MIXED_MODELS,
                format!("the record has no model; it must name {} like the first record", profile.name),
            ))
        }
        Err(other) => {
            return Err(PlanFailure::data(
                codes::MIXED_MODELS,
                format!(
                    "model {} differs from {}, the model of the batch",
                    super::dto::truncate_utf8(other, 128),
                    profile.name
                ),
            ))
        }
    };
    basal_core::request::prepared_input_bytes(&record.body)
        .map_err(|e| PlanFailure::data(codes::INVALID_BODY, e.to_string()))?;
    let plan = plan_body(&profile.planner, &record.body, &budget, job.stop.as_ref())?;
    let cost = planner::plan_cost(&plan);
    // Background forwards must stay within the M classes `background_support(forward_max_tokens)` checked, in every
    // round the plan can run (grouped rounds, task continuations and adaptive orders included).
    let shape = planner::DeviceWorkShape::from_plans(&[&plan], &profile.shape_limits)
        .map_err(|e| PlanFailure::data(codes::UNBOUNDED_FEATURE, e.0))?;
    let forward = shape.forward_tokens.max(shape.longest_prompt);
    if forward > profile.forward_max_tokens as u64 {
        return Err(PlanFailure::data(
            codes::BUDGET_EXCEEDED,
            format!(
                "the record needs forwards of {forward} tokens; background forwards have at most {} tokens",
                profile.forward_max_tokens
            ),
        ));
    }
    // DRR cost in ns of device-gate time: packed tokens × the profile's coefficient.
    let packed = (cost.packed_tokens as u64).saturating_add(cost.thinking_tokens).max(1);
    let cost_units = packed.checked_mul(profile.cost_ns_per_token).filter(|&c| c <= profile.max_record_cost);
    let Some(cost_units) = cost_units else {
        return Err(PlanFailure::data(
            codes::HOST_BUDGET_EXCEEDED,
            format!(
                "the record needs {packed} packed tokens; background admits at most {}",
                profile.max_record_cost / profile.cost_ns_per_token.max(1)
            ),
        ));
    };
    let body_bound = planner::result_bound(&profile.name, &plan);
    let result_bound = result_line_bound(&record.custom_id, body_bound);
    let line_len = job.line.len() as u64;
    let exec = planner::exec_bound(&plan, line_len, result_bound, Some(staging));
    let host_bound = planner::host_bound(planner::retained_bound(&plan), exec)
        .ok_or_else(|| PlanFailure::Unavailable(format!("{}: no host staging bound", profile.name)))?
        .saturating_add(line_len);
    if host_bound > limits.pool_bytes {
        return Err(PlanFailure::data(
            codes::HOST_BUDGET_EXCEEDED,
            format!("the record needs {host_bound} bytes of host memory; the execution pool has {}", limits.pool_bytes),
        ));
    }
    Ok(PlannedRecord {
        custom_id: record.custom_id,
        model,
        plan: job.keep_plan.then_some(plan),
        cost: cost_units,
        result_bound,
        host_bound,
    })
}

fn plan_body(
    planner: &RequestPlanner,
    body: &Value,
    budget: &PlanBudget,
    stop: Option<&StopToken>,
) -> Result<basal_core::engine::RequestPlan, PlanFailure> {
    planner.plan(body, Dialect::Upstream, budget, stop).map_err(|e| match e {
        PlanError::Decide(e) => PlanFailure::data(codes::INVALID_BODY, e.to_string()),
        PlanError::Budget(b) => budget_issue(b),
        PlanError::Interrupted(i) => PlanFailure::Interrupted(i.reason),
        PlanError::Unbounded(u) => PlanFailure::data(codes::UNBOUNDED_FEATURE, u.0),
    })
}
