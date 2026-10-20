//! Async feeder of the dedicated GPU lane. A lease never doubles as a GPU mutex or memory permit.
//!
//! Claims are made only after reserving host memory. The database remains the authority for result acceptance;
//! local monotonic deadlines stop wasted GPU work before lease loss. All commits retry the same operation id.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use basal_core::{StopReason, StopToken};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::artifacts::{ArtifactKey, ArtifactKind, ErrorClass, Sha256Digest};
use super::bg_lane::{BgLane, MicroBatch, Outcome, PlannedItem};
use super::budget::HostPool;
use super::config::Leases;
use super::device_budget::DeviceBudget;
use super::dto::{self, RecordOutcome};
use super::planning::{PlanFailure, PlanJob};
use super::scheduler::{self, Packing, Ready};
use super::store::{
    self, BatchStatus, ClaimOutcome, ClaimRequest, ClaimToken, ClaimedItem, LeaseTiming, LeaseUpdate, OperationId,
    ReleaseReason, ResultDisposition, ResultPayload, ShardParams, Store, StoreError, WorkClaim, WorkerTarget,
};
use super::{BatchService, ResolvedProfile};

/// Supplied only after the operator's numeric and device-memory qualification has been checked.
pub(in crate::serve) struct ExecutionLane {
    pub model: String,
    pub worker: BgLane,
    pub composition_qualified: bool,
    pub max_requests: usize,
    pub devices: Vec<usize>,
}

#[derive(Clone, Copy)]
pub(in crate::serve) struct OnlineReserve {
    pub device: usize,
    pub bytes: u64,
}

#[derive(Default)]
struct Counters {
    commits: AtomicU64,
    interruptions: AtomicU64,
    failed_groups: AtomicU64,
    /// Wall time, including pauses. This must not be presented as measured GPU time.
    discarded_wall_ms: AtomicU64,
    device_ns: AtomicU64,
    discarded_device_ns: AtomicU64,
    failed_group_device_ns: AtomicU64,
    deferred: AtomicU64,
    deferred_reason: Mutex<Option<String>>,
    abandoned_requeued: AtomicU64,
    abandoned_estimated: AtomicU64,
    next_eligible_at: AtomicI64,
    unavailable_lanes: AtomicU64,
    accepted: AtomicU64,
    requeued: AtomicU64,
    rejected: AtomicU64,
    replays: AtomicU64,
    max_failure_attempts: AtomicU64,
    unavailable_reason: Mutex<Option<String>>,
    discarded_by_reason: Mutex<std::collections::BTreeMap<&'static str, (u64, u64)>>,
}

pub(in crate::serve) struct Executor {
    stop: CancellationToken,
    lanes: Vec<ExecutionLane>,
    tasks: Vec<JoinHandle<()>>,
    counters: Arc<Counters>,
    devices: DeviceBudget,
}

impl Executor {
    pub fn start(
        service: Arc<BatchService>,
        lanes: Vec<ExecutionLane>,
        devices: DeviceBudget,
        reserves: Vec<OnlineReserve>,
    ) -> Self {
        let stop = service.stopping().child_token();
        let counters = Arc::new(Counters::default());
        let writes = Arc::new(Semaphore::new(service.settings().host.upload_concurrency as usize));
        let mut tasks: Vec<_> = lanes
            .iter()
            .map(|lane| {
                let feeder = Feeder {
                    service: service.clone(),
                    stop: stop.clone(),
                    counters: counters.clone(),
                    writes: writes.clone(),
                    devices: devices.clone(),
                    submit: lane.worker.submitter(),
                    model: lane.model.clone(),
                    composition_qualified: lane.composition_qualified,
                    max_requests: lane.max_requests,
                    device_ids: lane.devices.clone(),
                };
                tokio::spawn(feeder.run())
            })
            .collect();
        if !reserves.is_empty() {
            tasks.push(tokio::spawn(watch_memory(devices.clone(), reserves, stop.clone())));
        }
        let weak_service = Arc::downgrade(&service);
        let health_stop = stop.clone();
        let health_counters = counters.clone();
        let health_devices = devices.clone();
        let count = lanes.len();
        tasks.push(tokio::spawn(async move {
            loop {
                let Some(service) = weak_service.upgrade() else { return };
                service.set_execution_health(execution_health(&health_counters, &health_devices, count));
                drop(service);
                tokio::select! { _ = health_stop.cancelled() => return, _ = tokio::time::sleep(Duration::from_secs(1)) => {} }
            }
        }));
        let executor = Self { stop, lanes, tasks, counters, devices };
        service.set_execution_health(executor.health());
        executor
    }

    pub fn health(&self) -> serde_json::Value {
        execution_health(&self.counters, &self.devices, self.lanes.len())
    }

    pub async fn shutdown(mut self) -> Result<()> {
        self.stop.cancel();
        for lane in &self.lanes {
            lane.worker.stop();
        }
        let mut failed = false;
        for task in self.tasks.drain(..) {
            failed |= task.await.is_err();
        }
        for lane in self.lanes.drain(..) {
            failed |= lane.worker.shutdown().await.is_err();
        }
        anyhow::ensure!(!failed, "background executor stopped with a worker failure");
        Ok(())
    }
}

fn execution_health(counters: &Counters, devices: &DeviceBudget, lanes: usize) -> serde_json::Value {
    let snapshots = devices.snapshots().map(|snapshots| {
        snapshots
            .into_iter()
            .map(|device| {
                serde_json::json!({
                    "device": device.device,
                    "limit_bytes": device.limit,
                    "reserved_bytes": device.reserved,
                    "allocation_granularity_bytes": device.allocation_granularity,
                    "online_reserve_bytes": device.online_reserve_bytes,
                    "quarantined": device.quarantined,
                })
            })
            .collect::<Vec<_>>()
    });
    serde_json::json!({
        "configured_lanes": lanes,
        "unavailable_lanes": counters.unavailable_lanes.load(Ordering::Relaxed),
        "commits": counters.commits.load(Ordering::Relaxed),
        "interruptions": counters.interruptions.load(Ordering::Relaxed),
        "failed_groups": counters.failed_groups.load(Ordering::Relaxed),
        "discarded_wall_ms": counters.discarded_wall_ms.load(Ordering::Relaxed),
        "devices": snapshots.unwrap_or_default(),
        "cost_unit": "nanoseconds",
        "actual_cost_source": "background_device_gate_ownership",
        "device_ns": counters.device_ns.load(Ordering::Relaxed),
        "discarded_device_ns": counters.discarded_device_ns.load(Ordering::Relaxed),
        "failed_group_device_ns": counters.failed_group_device_ns.load(Ordering::Relaxed),
        "deferred": counters.deferred.load(Ordering::Relaxed),
        "deferred_reason": counters.deferred_reason.lock().map(|reason| reason.clone()).unwrap_or(None),
        "abandoned_requeued": counters.abandoned_requeued.load(Ordering::Relaxed),
        "abandoned_estimated": counters.abandoned_estimated.load(Ordering::Relaxed),
        "last_release_eligible_at_ms": counters.next_eligible_at.load(Ordering::Relaxed),
        "discarded_by_reason": counters.discarded_by_reason.lock().map(|reasons| reasons.clone()).unwrap_or_default(),
        "accepted_results": counters.accepted.load(Ordering::Relaxed),
        "requeued_results": counters.requeued.load(Ordering::Relaxed),
        "rejected_results": counters.rejected.load(Ordering::Relaxed),
        "commit_replays": counters.replays.load(Ordering::Relaxed),
        "max_observed_failure_attempts": counters.max_failure_attempts.load(Ordering::Relaxed),
        "unavailable_reason": counters.unavailable_reason.lock().map(|reason| reason.clone()).unwrap_or(None)
    })
}

async fn watch_memory(devices: DeviceBudget, reserves: Vec<OnlineReserve>, stop: CancellationToken) {
    let mut poll = tokio::time::interval(Duration::from_secs(1));
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! { _ = stop.cancelled() => return, _ = poll.tick() => {} }
        let captured = reserves.clone();
        // Driver/context queries run outside the Tokio executor and never synchronize the device or pool.
        let targets = tokio::task::spawn_blocking(move || {
            captured
                .iter()
                .filter_map(|reserve| {
                    let pressure = basal_gpu::gpu_memory(reserve.device)
                        .map_or(true, |memory| memory.available_bytes < reserve.bytes);
                    pressure.then_some(reserve.device)
                })
                .collect::<Vec<_>>()
        })
        .await;
        let targets = targets.unwrap_or_else(|_| reserves.iter().map(|reserve| reserve.device).collect());
        if !targets.is_empty() {
            let _ = devices.request_release(&targets);
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.stop.cancel();
        for lane in &self.lanes {
            lane.worker.stop();
        }
    }
}

struct Feeder {
    service: Arc<BatchService>,
    stop: CancellationToken,
    counters: Arc<Counters>,
    writes: Arc<Semaphore>,
    devices: DeviceBudget,
    submit: super::bg_lane::Submit,
    model: String,
    composition_qualified: bool,
    max_requests: usize,
    device_ids: Vec<usize>,
}

impl Feeder {
    async fn idle(&self, duration: Duration) {
        tokio::select! { _ = self.stop.cancelled() => {}, _ = tokio::time::sleep(duration) => {} }
    }

    async fn run(self) {
        let settings = self.service.settings();
        let min_poll = Duration::from_millis(settings.execution.poll_idle_min_ms);
        let max_poll = Duration::from_millis(settings.execution.poll_idle_max_ms);
        let mut delay = min_poll;
        let mut retry_delay = Duration::from_secs(1);
        let mut reserved = None;
        while !self.stop.is_cancelled() {
            if !self.service.processing_open() {
                self.idle(max_poll).await;
                continue;
            }
            match self.devices.ready(&self.device_ids) {
                Ok(true) => {}
                Ok(false) => {
                    self.idle(min_poll).await;
                    continue;
                }
                Err(error) => {
                    self.counters.unavailable_lanes.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut reason) = self.counters.unavailable_reason.lock() {
                        *reason = Some(error.to_string());
                    }
                    break;
                }
            }
            let Some(store) = self.service.store() else {
                self.idle(max_poll).await;
                continue;
            };
            let Some(profile) = self.service.profiles().into_iter().find(|profile| profile.name == self.model) else {
                self.idle(max_poll).await;
                continue;
            };
            if !profile.background.is_supported() || profile.max_record_cost == 0 {
                self.counters.unavailable_lanes.fetch_add(1, Ordering::Relaxed);
                break;
            }
            let mut host = reserved.take().unwrap_or_else(|| self.service.host_pool().reserve_available());
            let observed = Instant::now();
            let quantum = profile
                .cost_ns_per_token
                .checked_mul(u64::from(settings.execution.microbatch_tokens))
                .and_then(|cost| cost.checked_mul(self.max_requests.min(32) as u64))
                .unwrap_or(profile.max_record_cost)
                .min(profile.max_record_cost)
                .max(1);
            let request = ClaimRequest {
                worker: self.service.worker_id().clone(),
                target: WorkerTarget {
                    fingerprint: profile.fingerprint.clone(),
                    planner_version: profile.planner_version.clone(),
                },
                shard: ShardParams { quantum, max_record_cost: profile.max_record_cost },
                max_items: settings.execution.lookahead_records,
                max_input_bytes: settings.execution.lookahead_bytes,
                host_budget_free: host.num_permits() as u64,
            };
            match store.claim_items(&request).await {
                Ok(ClaimOutcome::Claimed(claim)) => {
                    let needed = claim.items.iter().try_fold(0u64, |total, item| total.checked_add(item.host_bound));
                    let Some(needed) = needed.filter(|&needed| needed <= host.num_permits() as u64) else {
                        let _ = store.release_claim(&claim.token, ReleaseReason::HostBudget).await;
                        self.counters.unavailable_lanes.fetch_add(1, Ordering::Relaxed);
                        break;
                    };
                    HostPool::shrink(&mut host, needed);
                    let result = self.process_claim(&store, profile, *claim, host, observed).await;
                    if matches!(result, Ok((false, _))) {
                        self.counters.unavailable_lanes.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    // I/O errors leave claims recoverable. Never move durable work into the online queues.
                    match result {
                        Ok((true, false)) => retry_delay = Duration::from_secs(1),
                        Ok(_) | Err(_) => {
                            self.idle(retry_delay).await;
                            retry_delay = retry_delay.saturating_mul(2).min(Duration::from_secs(30));
                        }
                    }
                    delay = min_poll;
                }
                Ok(ClaimOutcome::NeedsHostBudget { required }) => {
                    drop(host);
                    reserved = tokio::select! {
                        _ = self.stop.cancelled() => None,
                        permit = self.service.host_pool().reserve(required) => permit,
                    };
                    if reserved.is_none() && !self.stop.is_cancelled() {
                        self.counters.unavailable_lanes.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
                Ok(ClaimOutcome::QuotaBlocked { retry_after }) => {
                    drop(host);
                    self.idle(retry_after.min(Duration::from_secs(30))).await;
                }
                Ok(ClaimOutcome::Idle) | Err(_) => {
                    drop(host);
                    self.idle(delay).await;
                    delay = delay.saturating_mul(2).min(max_poll);
                }
            }
        }
    }

    async fn process_claim(
        &self,
        store: &Store,
        profile: Arc<ResolvedProfile>,
        claim: WorkClaim,
        mut host: OwnedSemaphorePermit,
        observed: Instant,
    ) -> Result<(bool, bool)> {
        let stop = StopToken::new();
        stop.set_pause_limit(Duration::from_millis(self.service.settings().execution.pause_limit_ms));
        let watch = Watch::start(
            store.clone(),
            claim.token.clone(),
            claim.timing,
            observed,
            self.service.settings().leases,
            self.service.settings().execution.cancel_poll_ms,
            self.stop.clone(),
            stop.clone(),
        );
        let mut measured: std::collections::BTreeMap<u64, u64> =
            claim.items.iter().map(|item| (item.line_no, 0)).collect();
        let result = self.process_held(store, profile, &claim, &mut host, &stop, &mut measured).await;
        let release = match stop.reason() {
            Some(StopReason::Shutdown) => ReleaseReason::Shutdown,
            Some(StopReason::PauseLimit) => ReleaseReason::PauseLimit,
            Some(StopReason::MemoryPressure) => ReleaseReason::MemoryPressure,
            Some(StopReason::LeaseLost) => ReleaseReason::LeaseLost,
            _ => ReleaseReason::Requeue,
        };
        let costs: Vec<_> =
            measured.into_iter().map(|(line_no, actual_cost)| store::AbandonedCost { line_no, actual_cost }).collect();
        if let Ok(LeaseUpdate::Applied(report)) = store.release_claim_with_cost(&claim.token, release, &costs).await {
            self.counters.abandoned_requeued.fetch_add(u64::from(report.requeued), Ordering::Relaxed);
            self.counters.abandoned_estimated.fetch_add(u64::from(report.estimated), Ordering::Relaxed);
            if let Some(at) = report.next_eligible_at {
                self.counters.next_eligible_at.store(at.0, Ordering::Relaxed);
            }
        }
        watch.finish().await;
        result.map(|enabled| (enabled, stop.reason().is_some()))
    }

    async fn process_held(
        &self,
        store: &Store,
        profile: Arc<ResolvedProfile>,
        claim: &WorkClaim,
        host: &mut OwnedSemaphorePermit,
        stop: &StopToken,
        measured: &mut std::collections::BTreeMap<u64, u64>,
    ) -> Result<bool> {
        let mut ready = Vec::new();
        let key = ArtifactKey::parse(claim.input.artifact.as_str())?;
        for record in &claim.items {
            if stop.reason().is_some() {
                return Ok(true);
            }
            let (Some(end), Ok(max)) = (record.offset.checked_add(record.length), usize::try_from(record.length))
            else {
                self.commit_error(store, &claim.token, record, dto::codes::INPUT_CHANGED).await?;
                continue;
            };
            let read = async {
                let reader = self
                    .service
                    .artifacts()
                    .read_range(
                        &key,
                        claim.input.bytes,
                        record.offset..end,
                        Some(Sha256Digest(record.input_sha256.0)),
                        Some(Instant::now() + Duration::from_millis(self.service.settings().leases.execution_ms)),
                    )
                    .await?;
                reader.read_all(max).await
            }
            .await;
            let line = match read {
                Ok(line) => line,
                Err(error) if error.class() == ErrorClass::Permanent => {
                    // A permanent failure belongs to this record, never to every other item in the claim.
                    self.commit_error(store, &claim.token, record, dto::codes::INPUT_CHANGED).await?;
                    continue;
                }
                Err(error) if error.class() == ErrorClass::Configuration => {
                    if let Ok(mut target) = self.counters.unavailable_reason.lock() {
                        *target = Some("background input storage configuration is unavailable".into());
                    }
                    return Ok(false);
                }
                Err(error) => return Err(error.into()),
            };
            let planned = self
                .service
                .planning()
                .plan(PlanJob {
                    profile: profile.clone(),
                    line,
                    keep_plan: true,
                    stop: Some(stop.clone()),
                    default_model: Some(profile.name.clone()),
                })
                .await;
            match planned {
                Ok(planned) => {
                    if planned.custom_id != record.custom_id.as_str() || planned.model != profile.name {
                        self.commit_error(store, &claim.token, record, dto::codes::INPUT_CHANGED).await?;
                        continue;
                    }
                    if planned.host_bound > record.host_bound || planned.result_bound > record.result_bound {
                        self.commit_error(store, &claim.token, record, dto::codes::HOST_BUDGET_EXCEEDED).await?;
                        continue;
                    }
                    let plan = planned.plan.context("planner did not retain the execution plan")?;
                    ready.push(Ready::new(PlannedItem { record: record.clone(), plan, spent_ns: 0 }));
                }
                Err(PlanFailure::Data { code, .. }) => {
                    self.commit_error(store, &claim.token, record, code).await?;
                }
                Err(PlanFailure::Interrupted(reason)) => {
                    stop.stop(reason);
                    return Ok(true);
                }
                Err(PlanFailure::Unavailable(_)) => return Ok(false),
            }
        }
        let limits = self.service.settings().execution;
        let packing = Packing {
            composition_qualified: self.composition_qualified,
            max_requests: (limits.microbatch_requests as usize).min(32).min(self.max_requests),
            max_tokens: limits.microbatch_tokens as usize,
            forward_max_tokens: profile.forward_max_tokens,
        };
        while !ready.is_empty() && stop.reason().is_none() {
            let selected = scheduler::take(&mut ready, packing);
            let bytes = selected.iter().try_fold(0usize, |total, item| {
                usize::try_from(item.record.host_bound).ok().and_then(|bytes| total.checked_add(bytes))
            });
            let permits = bytes.and_then(|bytes| host.split(bytes)).context("execution host reservation changed")?;
            // The worker may fail before replying. Only a completed reply supplies an absolute measurement;
            // unconfirmed work falls back to its durable estimate when the lease is released or recovered.
            for item in &selected {
                measured.remove(&item.record.line_no);
            }
            let receive = self
                .submit
                .run(MicroBatch { items: selected, host: permits, devices: self.devices.clone(), stop: stop.clone() })
                .await?;
            // Do not drop this receive on shutdown: the worker owns the plans and releases the device before reply.
            let reply = receive.await.context("background worker stopped before reporting cleanup")?;
            let super::bg_lane::Reply { mut items, outcome, elapsed, device_time, host: permits } = reply;
            host.merge(permits);
            let device_ns = u64::try_from(device_time.as_nanos()).context("device ownership time overflows")?;
            self.counters.device_ns.fetch_add(device_ns, Ordering::Relaxed);
            add_device_cost(&mut items, device_ns)?;
            if !matches!(outcome, Outcome::DeviceLost) {
                for item in &items {
                    measured.insert(item.record.line_no, item.spent_ns);
                }
            }
            match outcome {
                Outcome::Completed(outputs) => {
                    let mut results = Vec::new();
                    let mut inline = 0usize;
                    for (item, output) in items.into_iter().zip(outputs) {
                        let (disposition, body, code) = match output {
                            Ok(body) => (ResultDisposition::Output, Some((200, body)), None),
                            Err(basal_core::request::DecideError::Internal(_)) => {
                                (ResultDisposition::Retry, None, Some(dto::codes::EXECUTION_FAILED))
                            }
                            Err(error) => {
                                let (status, body) =
                                    crate::serve::error_parts(basal_core::request::Dialect::Upstream, error);
                                (ResultDisposition::Error, Some((status.as_u16(), body)), None)
                            }
                        };
                        let outcome = match &body {
                            Some((status, body)) => RecordOutcome::Response { status_code: *status, body },
                            None => RecordOutcome::Error {
                                code: code.unwrap_or(dto::codes::EXECUTION_FAILED),
                                message: "The record could not be executed.",
                            },
                        };
                        let result = self
                            .make_result(store, &claim.token, &item.record, disposition, outcome, item.spent_ns)
                            .await?;
                        let bytes = match &result.payload {
                            ResultPayload::Inline(bytes) => bytes.len(),
                            _ => 0,
                        };
                        if !results.is_empty()
                            && (results.len() >= limits.commit_items.min(32) as usize
                                || inline.saturating_add(bytes) > limits.commit_inline_bytes.min(1024 * 1024) as usize)
                        {
                            self.commit(store, &claim.token, &results).await?;
                            results.clear();
                            inline = 0;
                        }
                        inline += bytes;
                        results.push(result);
                    }
                    if !results.is_empty() {
                        self.commit(store, &claim.token, &results).await?;
                    }
                }
                Outcome::Failed(_) if items.len() > 1 => {
                    self.counters.failed_groups.fetch_add(1, Ordering::Relaxed);
                    self.counters.failed_group_device_ns.fetch_add(device_ns, Ordering::Relaxed);
                    let lines: Vec<_> = items.iter().map(|item| item.record.line_no).collect();
                    if !matches!(store.mark_solo(&claim.token, &lines).await?, LeaseUpdate::Applied(())) {
                        stop.stop(StopReason::LeaseLost);
                        return Ok(true);
                    }
                    for mut item in items {
                        item.record.requires_solo = true;
                        ready.push(Ready::new(item));
                    }
                }
                Outcome::Rejected { .. } | Outcome::MemoryLimited(_) if items.len() > 1 => {
                    let lines: Vec<_> = items.iter().map(|item| item.record.line_no).collect();
                    if !matches!(store.mark_solo(&claim.token, &lines).await?, LeaseUpdate::Applied(())) {
                        stop.stop(StopReason::LeaseLost);
                        return Ok(true);
                    }
                    for mut item in items {
                        item.record.requires_solo = true;
                        ready.push(Ready::new(item));
                    }
                }
                Outcome::Rejected { code, reason } => {
                    for item in items {
                        let message = dto::truncate_utf8(&reason, 256);
                        let result = self
                            .make_result(
                                store,
                                &claim.token,
                                &item.record,
                                ResultDisposition::Error,
                                RecordOutcome::Error { code, message },
                                item.spent_ns,
                            )
                            .await?;
                        self.commit(store, &claim.token, &[result]).await?;
                    }
                }
                Outcome::Failed(error) => {
                    let data_error = error.chain().find_map(|error| {
                        let error = error.downcast_ref::<basal_core::request::DecideError>()?;
                        (!matches!(error, basal_core::request::DecideError::Internal(_)))
                            .then(|| error.to_upstream_json())
                    });
                    for item in items {
                        let (disposition, outcome) = match &data_error {
                            Some(body) => {
                                (ResultDisposition::Error, RecordOutcome::Response { status_code: 422, body })
                            }
                            None => (
                                ResultDisposition::Retry,
                                RecordOutcome::Error {
                                    code: dto::codes::EXECUTION_FAILED,
                                    message: "The record could not be executed.",
                                },
                            ),
                        };
                        let result = self
                            .make_result(store, &claim.token, &item.record, disposition, outcome, item.spent_ns)
                            .await?;
                        self.commit(store, &claim.token, &[result]).await?;
                    }
                }
                Outcome::Interrupted(reason) => {
                    self.counters.interruptions.fetch_add(1, Ordering::Relaxed);
                    self.counters.discarded_device_ns.fetch_add(device_ns, Ordering::Relaxed);
                    if let Ok(mut reasons) = self.counters.discarded_by_reason.lock() {
                        let (count, ns) = reasons.entry(reason.as_str()).or_default();
                        *count = count.saturating_add(1);
                        *ns = ns.saturating_add(device_ns);
                    }
                    self.counters
                        .discarded_wall_ms
                        .fetch_add(u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX), Ordering::Relaxed);
                    stop.stop(reason);
                    return Ok(true);
                }
                Outcome::Unavailable(reason) => {
                    if let Ok(mut target) = self.counters.unavailable_reason.lock() {
                        *target = Some(dto::truncate_utf8(&reason, 256).to_string());
                    }
                    return Ok(false);
                }
                Outcome::Deferred(reason) | Outcome::MemoryLimited(reason) => {
                    self.counters.deferred.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut target) = self.counters.deferred_reason.lock() {
                        *target = Some(dto::truncate_utf8(&reason, 256).to_string());
                    }
                    anyhow::bail!("background work deferred by local device admission");
                }
                Outcome::DeviceLost => {
                    self.counters.discarded_device_ns.fetch_add(device_ns, Ordering::Relaxed);
                    if let Ok(mut reasons) = self.counters.discarded_by_reason.lock() {
                        let (count, ns) = reasons.entry("device_lost").or_default();
                        *count = count.saturating_add(1);
                        *ns = ns.saturating_add(device_ns);
                    }
                    if let Ok(mut target) = self.counters.unavailable_reason.lock() {
                        *target = Some("background device cleanup is unconfirmed".into());
                    }
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    async fn commit_error(&self, store: &Store, token: &ClaimToken, record: &ClaimedItem, code: &str) -> Result<()> {
        let result = self
            .make_result(
                store,
                token,
                record,
                ResultDisposition::Error,
                RecordOutcome::Error { code, message: "The record cannot execute under its admitted profile." },
                0,
            )
            .await?;
        self.commit(store, token, &[result]).await
    }

    async fn make_result(
        &self,
        store: &Store,
        token: &ClaimToken,
        record: &ClaimedItem,
        mut disposition: ResultDisposition,
        outcome: RecordOutcome<'_>,
        actual_cost: u64,
    ) -> Result<store::ItemResult> {
        let limit = record.result_bound.min(self.service.settings().limits.max_result_bytes);
        self.counters.max_failure_attempts.fetch_max(u64::from(record.failure_attempts), Ordering::Relaxed);
        let mut bytes = BoundedLine::new(limit)?;
        let request_id = dto::new_request_id();
        if dto::write_result_line(&mut bytes, &request_id, record.custom_id.as_str(), outcome).is_err() {
            bytes.bytes.clear();
            disposition = ResultDisposition::Error;
            dto::write_result_line(
                &mut bytes,
                &request_id,
                record.custom_id.as_str(),
                RecordOutcome::Error {
                    code: "result_too_large",
                    message: "The record result exceeds its admitted bound.",
                },
            )?;
        }
        let inline_limit = self.service.settings().limits.inline_result_bytes.min(store::INLINE_RESULT_MAX as u64);
        let payload = if bytes.bytes.len() as u64 <= inline_limit {
            ResultPayload::Inline(bytes.bytes)
        } else {
            let permit = self.writes.clone().acquire_owned().await.context("result writer pool closed")?;
            let key = ArtifactKey::generate(ArtifactKind::Result);
            let artifact = store::ArtifactRef::parse(&key.to_string())?;
            let session = store.begin_result_upload(token, record.line_no, &artifact).await?;
            let mut writer = self
                .service
                .artifacts()
                .create(
                    key,
                    limit,
                    Instant::now() + Duration::from_millis(self.service.settings().leases.execution_ms),
                    Some(permit),
                )
                .await?;
            let write = async {
                writer.write(&bytes.bytes).await?;
                writer.finish().await
            };
            let object = tokio::select! {
                _ = self.stop.cancelled() => anyhow::bail!("result upload stopped for shutdown"),
                result = write => result?,
            };
            ResultPayload::Stored {
                session,
                artifact,
                object: store::StoredObject { bytes: object.len, sha256: store::Sha256Digest(object.sha256.0) },
            }
        };
        Ok(store::ItemResult { line_no: record.line_no, disposition, payload, actual_cost })
    }

    async fn commit(&self, store: &Store, token: &ClaimToken, results: &[store::ItemResult]) -> Result<()> {
        let operation = OperationId::new_internal();
        let until = Instant::now() + Duration::from_millis(self.service.settings().leases.shutdown_grace_ms);
        loop {
            match store.commit_results(&operation, token, results).await {
                Ok(report) => {
                    self.counters.commits.fetch_add(1, Ordering::Relaxed);
                    self.counters.replays.fetch_add(u64::from(report.replayed), Ordering::Relaxed);
                    let mut rejected = false;
                    for (_, outcome) in report.items {
                        match outcome {
                            store::ItemCommitOutcome::Accepted => {
                                self.counters.accepted.fetch_add(1, Ordering::Relaxed);
                            }
                            store::ItemCommitOutcome::Requeued => {
                                self.counters.requeued.fetch_add(1, Ordering::Relaxed);
                            }
                            store::ItemCommitOutcome::Rejected(_) => {
                                rejected = true;
                                self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    if rejected || report.batch_status.is_terminal() {
                        return Err(StoreError::LeaseStale.into());
                    }
                    return Ok(());
                }
                Err(error)
                    if (matches!(error, StoreError::CommitUnknown(_)) || error.is_unavailable())
                        && Instant::now() < until =>
                {
                    tokio::time::sleep(Duration::from_millis(100)).await
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

/// Divide measured gate time in the ratio of the admitted estimates. The final share keeps the remainder,
/// so every nanosecond is accounted once. A failed group's shares travel with its plans into solo diagnosis.
fn add_device_cost(items: &mut [PlannedItem], measured: u64) -> Result<()> {
    let total = items
        .iter()
        .try_fold(0u128, |sum, item| sum.checked_add(u128::from(item.record.cost)))
        .context("record cost sum overflows")?;
    anyhow::ensure!(total > 0, "background records must have positive admitted costs");
    let count = items.len();
    let mut remaining = measured;
    for (index, item) in items.iter_mut().enumerate() {
        let share = if index + 1 == count {
            remaining
        } else {
            u64::try_from(u128::from(measured) * u128::from(item.record.cost) / total)
                .context("measured cost share overflows")?
        };
        item.spent_ns = item.spent_ns.checked_add(share).context("record device cost overflows")?;
        remaining = remaining.checked_sub(share).context("measured cost division exceeds its total")?;
    }
    Ok(())
}

/// Fixed capacity serialization; a result never allocates beyond the admitted line bound.
struct BoundedLine {
    bytes: Vec<u8>,
    limit: usize,
}
impl BoundedLine {
    fn new(limit: u64) -> Result<Self> {
        let limit = usize::try_from(limit).context("result bound is not addressable")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(limit).context("allocating an admitted result buffer")?;
        Ok(Self { bytes, limit })
    }
}
impl std::io::Write for BoundedLine {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("result exceeds its admitted bound"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Watch {
    done: CancellationToken,
    stop: StopToken,
    task: Option<JoinHandle<()>>,
}
impl Watch {
    #[expect(clippy::too_many_arguments, reason = "lease monitor captures one immutable claim and its clocks")]
    fn start(
        store: Store,
        token: ClaimToken,
        timing: LeaseTiming,
        observed: Instant,
        leases: Leases,
        poll_ms: u64,
        shutdown: CancellationToken,
        stop: StopToken,
    ) -> Self {
        let done = CancellationToken::new();
        let task = tokio::spawn(watch_claim(
            store,
            token,
            timing,
            observed,
            leases,
            poll_ms,
            shutdown,
            done.clone(),
            stop.clone(),
        ));
        Self { done, stop, task: Some(task) }
    }
    async fn finish(mut self) {
        self.done.cancel();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}
impl Drop for Watch {
    fn drop(&mut self) {
        if self.task.is_some() {
            self.stop.stop(StopReason::LeaseLost);
        }
        self.done.cancel();
    }
}

fn deadline(timing: LeaseTiming, observed: Instant, margin_ms: u64) -> (tokio::time::Instant, StopReason) {
    let (until, reason) = if timing.batch_expires_at <= timing.lease_until {
        (timing.batch_expires_at, StopReason::Expired)
    } else {
        (timing.lease_until, StopReason::LeaseLost)
    };
    let remaining = u64::try_from(until.0.saturating_sub(timing.server_now.0)).unwrap_or(0).saturating_sub(margin_ms);
    (tokio::time::Instant::from_std(observed + Duration::from_millis(remaining)), reason)
}

#[expect(
    clippy::too_many_arguments,
    reason = "owned task state keeps cancellation and database calls independent of the feeder"
)]
async fn watch_claim(
    store: Store,
    token: ClaimToken,
    timing: LeaseTiming,
    observed: Instant,
    leases: Leases,
    poll_ms: u64,
    shutdown: CancellationToken,
    done: CancellationToken,
    stop: StopToken,
) {
    let (mut until, mut reason) = deadline(timing, observed, leases.safety_margin_ms);
    let mut renew = tokio::time::interval(Duration::from_millis(leases.execution_renew_ms));
    let mut poll = tokio::time::interval(Duration::from_millis(poll_ms));
    renew.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            _ = done.cancelled() => return,
            _ = shutdown.cancelled() => { stop.stop(StopReason::Shutdown); return; }
            _ = tokio::time::sleep_until(until) => { stop.stop(reason); return; }
            _ = renew.tick() => {
                let observed = Instant::now();
                let result = tokio::select! {
                    _ = done.cancelled() => return,
                    _ = shutdown.cancelled() => { stop.stop(StopReason::Shutdown); return; }
                    _ = tokio::time::sleep_until(until) => { stop.stop(reason); return; }
                    result = store.renew_claim(&token) => result,
                };
                match result {
                    Ok(LeaseUpdate::Applied(renewal)) if renewal.accepting && renewal.control_epoch == token.control_epoch => {
                        (until, reason) = deadline(renewal.timing, observed, leases.safety_margin_ms);
                    }
                    Ok(_) => { stop.stop(StopReason::LeaseLost); return; }
                    Err(_) => {} // Old local deadline remains authoritative if the database cannot renew.
                }
            }
            _ = poll.tick() => {
                let result = tokio::select! {
                    _ = done.cancelled() => return,
                    _ = shutdown.cancelled() => { stop.stop(StopReason::Shutdown); return; }
                    _ = tokio::time::sleep_until(until) => { stop.stop(reason); return; }
                    result = store.get_batch(&token.batch.owner, &token.batch.id) => result,
                };
                if let Ok(batch) = result {
                    if batch.control_epoch != token.control_epoch || batch.status != BatchStatus::InProgress {
                        stop.stop(match batch.terminal_target {
                            Some(store::TerminalStatus::Cancelled) => StopReason::Cancelled,
                            Some(store::TerminalStatus::Expired) => StopReason::Expired,
                            _ => StopReason::LeaseLost,
                        });
                        return;
                    }
                }
            }
        }
    }
}
