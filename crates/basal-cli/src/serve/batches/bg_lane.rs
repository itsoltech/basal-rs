//! Dedicated owner of a background engine. Database and artifact I/O never run on this thread.
//!
//! A queued command owns every plan and its host and device reservations. Dropping the caller cannot release
//! those reservations while the GPU is still using the work. Replies keep the host charge through serialization
//! and commit; device admission is released only after execution cleanup has returned and device health is known.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use basal_core::engine::RequestPlan;
use basal_core::request::DecideError;
use basal_core::{Backend, Engine, StopReason, StopToken};
use serde_json::Value;
use tokio::sync::{mpsc, oneshot, OwnedSemaphorePermit};

use super::device_budget::{DeviceBudget, DeviceBudgetError, DeviceRequest};
use super::store::ClaimedItem;

pub(in crate::serve) struct PlannedItem {
    pub record: ClaimedItem,
    pub plan: RequestPlan,
    /// Measured device ownership already spent on a failed group, charged when the solo record settles.
    pub spent_ns: u64,
}

pub(in crate::serve) struct MicroBatch {
    pub items: Vec<PlannedItem>,
    pub host: OwnedSemaphorePermit,
    pub devices: DeviceBudget,
    pub stop: StopToken,
}

pub(in crate::serve) enum Outcome {
    Completed(Vec<std::result::Result<Value, DecideError>>),
    Interrupted(StopReason),
    /// A healthy grouped call failed. Persist `requires_solo` before retrying individual records.
    Failed(anyhow::Error),
    /// This shape or device has no usable memory profile. Release without charging a failure attempt.
    Unavailable(String),
    /// A transient host condition; keep the lane usable and defer the claim with bounded backoff.
    Deferred(String),
    /// A local shared-device allowance may be stricter than this profile on another compatible worker.
    /// Split groups first; a solo record must be deferred rather than permanently rejected by this host.
    MemoryLimited(String),
    /// No GPU work started. A permanent shape limit, isolated to a solo record by the feeder.
    Rejected {
        code: &'static str,
        reason: String,
    },
    /// Cleanup or the worker panicked, or device health was lost. Never diagnose the group on this device.
    DeviceLost,
}

pub(in crate::serve) struct Reply {
    pub items: Vec<PlannedItem>,
    pub outcome: Outcome,
    /// Includes CPU work and time parked for online traffic; this is deliberately not called GPU time.
    pub elapsed: Duration,
    /// Sum of background gate ownership, excluding waiting and pauses. Includes launch/drain CPU work in a gate.
    pub device_time: Duration,
    pub host: OwnedSemaphorePermit,
}

enum Command {
    Run { work: MicroBatch, reply: oneshot::Sender<Reply> },
    Shutdown,
}

/// Only an unwind while running device work makes its completion unknown. CPU preparation/response failures
/// are isolated to the background record; a cleanup failure fences background reservations, not online health.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkPhase {
    Preparing,
    Running,
    Cleaning,
    Responding,
}

#[derive(Default)]
struct Control {
    shutdown: bool,
    active: Option<StopToken>,
}

/// The feeder may submit at most one queued micro-batch in addition to the active one.
#[derive(Clone)]
pub(in crate::serve) struct Submit {
    tx: mpsc::Sender<Command>,
}

impl Submit {
    pub async fn run(&self, work: MicroBatch) -> Result<oneshot::Receiver<Reply>> {
        let (reply, receive) = oneshot::channel();
        self.tx.send(Command::Run { work, reply }).await.map_err(|_| anyhow::anyhow!("background worker closed"))?;
        Ok(receive)
    }
}

pub(in crate::serve) struct BgLane {
    submit: Submit,
    control: Arc<Mutex<Control>>,
    thread: Option<JoinHandle<()>>,
}

impl BgLane {
    /// The caller has already configured the fork, all device gates and its qualification policy.
    pub fn start<B: Backend + Send + 'static>(label: &str, name: &str, engine: Engine<B>) -> Result<Self> {
        anyhow::ensure!(engine.backend.work_health().is_some(), "background requires reported work health");
        let (tx, rx) = mpsc::channel(1);
        let control = Arc::new(Mutex::new(Control::default()));
        let shared = control.clone();
        let name = name.to_owned();
        let thread = std::thread::Builder::new()
            .name(format!("{label}-bg"))
            .spawn(move || worker(&name, engine, rx, shared))
            .context("starting a background GPU thread")?;
        Ok(Self { submit: Submit { tx }, control, thread: Some(thread) })
    }

    pub fn submitter(&self) -> Submit {
        self.submit.clone()
    }

    /// Signals active work and wakes an idle receiver even while a feeder still owns a sender.
    pub fn stop(&self) {
        let active = match self.control.lock() {
            Ok(mut state) => {
                state.shutdown = true;
                state.active.clone()
            }
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.shutdown = true;
                state.active.clone()
            }
        };
        if let Some(stop) = active {
            stop.stop(StopReason::Shutdown);
        }
        // Full means there is already a command to wake the receiver; it sees `shutdown` before executing it.
        let _ = self.submit.tx.try_send(Command::Shutdown);
    }

    /// Join outside the async executor. No timeout can make unconfirmed GPU cleanup safe to forget.
    pub async fn shutdown(mut self) -> Result<()> {
        self.stop();
        if let Some(thread) = self.thread.take() {
            tokio::task::spawn_blocking(move || thread.join())
                .await
                .context("joining a background GPU thread")?
                .map_err(|_| anyhow::anyhow!("background GPU worker panicked"))?;
        }
        Ok(())
    }
}

impl Drop for BgLane {
    fn drop(&mut self) {
        self.stop();
        if let Some(thread) = self.thread.take() {
            // Normal shutdown explicitly joins. Error paths must also retain the join operation: Tokio waits for
            // these blocking jobs before shutting its runtime down; no executor thread waits on the GPU here.
            match tokio::runtime::Handle::try_current() {
                Ok(runtime) => {
                    runtime.spawn_blocking(move || thread.join());
                }
                Err(_) => {
                    let _ = thread.join();
                }
            }
        }
    }
}

fn worker<B: Backend>(
    name: &str,
    mut engine: Engine<B>,
    mut rx: mpsc::Receiver<Command>,
    control: Arc<Mutex<Control>>,
) {
    while let Some(command) = rx.blocking_recv() {
        let Command::Run { work, reply } = command else { break };
        let shutting_down = match control.lock() {
            Ok(mut state) => {
                state.active = Some(work.stop.clone());
                state.shutdown
            }
            Err(_) => true,
        };
        if shutting_down {
            work.stop.stop(StopReason::Shutdown);
        }
        let MicroBatch { items, host, devices, stop } = work;
        let started = Instant::now();
        let meter = stop.clone();
        let prior = meter.device_time();
        // The permit lives outside catch_unwind: an unwinding backend must not drop the device charge before
        // the worker can quarantine it. The estimator itself is CPU-only and runs before GPU work is started.
        let mut device_permit = None;
        let mut phase = WorkPhase::Preparing;
        let result = catch_unwind(AssertUnwindSafe(|| {
            let plans: Vec<_> = items.iter().map(|item| &item.plan).collect();
            match engine.device_work_shape(&plans) {
                Ok(shape) if shape.forward_tokens.max(shape.longest_prompt) <= engine.forward_max_tokens as u64 => {}
                Ok(_) => {
                    return Outcome::Rejected {
                        code: "budget_exceeded",
                        reason: "work exceeds the qualified GEMM forward classes".into(),
                    }
                }
                Err(error) => return Outcome::Rejected { code: "unbounded_feature", reason: error.to_string() },
            }
            let estimate = match engine.device_work_bound(&plans) {
                basal_core::planner::DeviceWorkBound::Estimate(estimate) => estimate,
                basal_core::planner::DeviceWorkBound::Unknown(reason) => {
                    return Outcome::Rejected { code: "unbounded_feature", reason }
                }
            };
            let requests: Vec<_> = estimate
                .devices
                .iter()
                .map(|device| DeviceRequest {
                    device: device.device,
                    bytes: device.bytes,
                    allocations: device.allocations,
                })
                .collect();
            device_permit = match devices.try_reserve(&requests, stop.clone()) {
                Ok(Some(permit)) => Some(permit),
                Ok(None) => return Outcome::Interrupted(StopReason::MemoryPressure),
                Err(error @ DeviceBudgetError::ExceedsLimit(_)) => return Outcome::MemoryLimited(error.to_string()),
                Err(error @ DeviceBudgetError::Telemetry(_)) => return Outcome::Deferred(error.to_string()),
                Err(error) => return Outcome::Unavailable(error.to_string()),
            };
            execute(name, &mut engine, &items, stop, &mut phase)
        }));
        let health = engine.work_health();
        let mut lost = health.is_empty() || health.iter().any(|h| h.is_lost());
        let cleanup_unconfirmed = phase == WorkPhase::Cleaning;
        let outcome = match result {
            Ok(outcome) if !lost => outcome,
            Err(_) if !lost && phase != WorkPhase::Running => match phase {
                WorkPhase::Cleaning => Outcome::Unavailable("background cleanup did not complete".into()),
                _ => Outcome::Failed(anyhow::anyhow!("background CPU work panicked")),
            },
            _ => {
                lost = true;
                for health in engine.work_health() {
                    health.mark_lost();
                }
                Outcome::DeviceLost
            }
        };
        if lost || cleanup_unconfirmed {
            if let Some(permit) = device_permit {
                permit.quarantine();
            }
        } else {
            drop(device_permit);
        }
        let _ = reply.send(Reply {
            items,
            outcome,
            elapsed: started.elapsed(),
            device_time: meter.device_time().saturating_sub(prior),
            host,
        });
        let shutting_down = match control.lock() {
            Ok(mut state) => {
                state.active = None;
                state.shutdown
            }
            Err(_) => true,
        };
        if lost || shutting_down {
            break;
        }
    }
}

fn execute<B: Backend>(
    name: &str,
    engine: &mut Engine<B>,
    items: &[PlannedItem],
    stop: StopToken,
    phase: &mut WorkPhase,
) -> Outcome {
    let result = (|| -> Result<Vec<_>> {
        stop.check()?;
        anyhow::ensure!(!items.is_empty(), "empty background micro-batch");
        engine.set_stop_token(Some(stop))?;
        let plans: Vec<_> = items.iter().map(|item| &item.plan).collect();
        *phase = WorkPhase::Running;
        engine.run_plans(&plans)
    })();
    *phase = WorkPhase::Cleaning;
    if engine.reset_execution().is_err() {
        // The backend itself reports unconfirmed device work through work_health. A failure to release private
        // background state only fences its reservation; it must not manufacture a shared online device failure.
        return Outcome::Unavailable("background execution state could not be released".into());
    }
    *phase = WorkPhase::Responding;
    match result {
        Ok(outputs) if outputs.len() != items.len() => {
            Outcome::Failed(anyhow::anyhow!("background output count mismatch"))
        }
        Ok(outputs) => Outcome::Completed(
            items
                .iter()
                .zip(outputs)
                .map(|(item, out)| super::super::served_response(name, &item.plan, &out))
                .collect(),
        ),
        Err(error) => match basal_core::execution::interrupted(&error) {
            Some(reason) => Outcome::Interrupted(reason),
            None => Outcome::Failed(error),
        },
    }
}
