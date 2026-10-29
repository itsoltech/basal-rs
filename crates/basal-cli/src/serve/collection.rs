//! Opt-in online collection. The absent-controls path remains in the legacy scheduler.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use basal_core::{Backend, Engine};
use tokio::sync::mpsc;

use super::{plan_job, Job, Lane, Waiting};
use crate::models::SchedulingPolicy;

/// Checked options only; timing absence selects the qualified profile policy. Entry identity follows the lane's
/// effective packing/session policy: ordinary Budget main work is a ready primitive, exceptions remain atomic.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CollectionPolicy {
    max_batch: Option<usize>,
    timing: Option<(Duration, Duration)>,
}

impl CollectionPolicy {
    pub(crate) fn new(max_batch: Option<usize>, wait_ms: Option<f64>, batch_window_ms: Option<f64>) -> Result<Self> {
        ensure!(max_batch != Some(0), "max_batch must be positive");
        let timing = if wait_ms.is_some() || batch_window_ms.is_some() {
            Some((
                milliseconds(wait_ms.unwrap_or(0.0), "wait_ms")?,
                milliseconds(batch_window_ms.unwrap_or(0.0), "batch_window_ms")?,
            ))
        } else {
            None
        };
        Ok(Self { max_batch, timing })
    }

    pub(crate) fn max_batch(self) -> Option<usize> {
        self.max_batch
    }

    fn window(self, scheduling: SchedulingPolicy, last_batch: usize, returning: bool) -> (Duration, Option<usize>) {
        if let Some((wait, adaptive)) = self.timing {
            let window = wait.max(if last_batch > 1 { adaptive } else { Duration::ZERO });
            // A positive fixed wait collects toward the cap; adaptive-only collection stops at the previous size.
            (window, wait.is_zero().then_some(last_batch))
        } else {
            match scheduling {
                SchedulingPolicy::CoalesceReturning { window, .. } if last_batch > 1 && returning => (window, None),
                _ => (Duration::ZERO, None),
            }
        }
    }
}

fn milliseconds(ms: f64, name: &str) -> Result<Duration> {
    ensure!(ms.is_finite() && ms >= 0.0, "{name} must be finite and nonnegative milliseconds");
    let duration = Duration::try_from_secs_f64(ms / 1000.0).with_context(|| format!("{name} is too large"))?;
    ensure!(ms == 0.0 || !duration.is_zero(), "{name} must be zero or at least one nanosecond");
    ensure!(Instant::now().checked_add(duration).is_some(), "{name} exceeds the monotonic clock range");
    Ok(duration)
}

impl Lane {
    /// Preview actual admission, including the waiting backlog, without consuming a job or its queued permit.
    /// A token-rejected pending request ends collection: it remains available for the next call.
    fn collection_capacity(&mut self, now: Instant) -> (usize, bool) {
        self.waiting.retain(|w| !w.job.reply.is_closed());
        let selection = self.select(now);
        (selection.entries.len(), selection.full)
    }
}

/// Existing admission bound plus the server shutdown signal; shutdown skips waits while draining accepted jobs.
pub(super) struct CollectionRuntime {
    pub(super) max_inflight: usize,
    pub(super) stopping: Arc<AtomicBool>,
}

/// Runs only on the dedicated engine thread. At most `max_inflight` jobs are examined in one collection cycle,
/// even if clients replenish cancelled, invalid or long-routed jobs. There is no secondary raw-job buffer.
pub(super) fn scheduler<B: Backend>(
    mut engine: Engine<B>,
    mut rx: mpsc::UnboundedReceiver<Job>,
    mut lane: Lane,
    long: Option<(std::sync::mpsc::Sender<Waiting>, super::routing::RoutingPolicy)>,
    scheduling: SchedulingPolicy,
    policy: CollectionPolicy,
    runtime: CollectionRuntime,
) {
    let max_inflight = runtime.max_inflight;
    let mut last_batch = 1;
    loop {
        lane.waiting.retain(|w| !w.job.reply.is_closed());
        let had_backlog = !lane.waiting.is_empty();
        let wait_start = Instant::now();
        let mut returning = false;
        let mut examined = 0;
        let mut valid = true;
        if !had_backlog {
            match rx.blocking_recv() {
                Some(job) => {
                    // Legacy idle qualification measures the wait for a returning client, not its CPU planning.
                    returning = matches!(scheduling, SchedulingPolicy::CoalesceReturning { idle, .. } if wait_start.elapsed() < idle);
                    examined += 1;
                    valid = plan_job(&engine, job, &mut lane, &long);
                }
                None => return,
            }
        }
        // Plan the available snapshot before selection so HRRN still sees the waiting backlog even at cap=1.
        // The global permits bound both this snapshot and the planned queue; new arrivals cannot prolong it.
        let available = rx.len().min(max_inflight.saturating_sub(examined));
        for _ in 0..available {
            if !valid {
                break;
            }
            match rx.try_recv() {
                Ok(job) => {
                    examined += 1;
                    valid = plan_job(&engine, job, &mut lane, &long);
                }
                Err(_) => break,
            }
        }
        let (window, target) = policy.window(scheduling, last_batch, returning);
        // Conversion checked this range before device/model loading. Recheck rather than panic at runtime.
        let until = Instant::now().checked_add(window).unwrap_or_else(Instant::now);
        while valid && examined < max_inflight && !window.is_zero() && !runtime.stopping.load(Ordering::Relaxed) {
            let (count, full) = lane.collection_capacity(Instant::now());
            if full || count == 0 || target.is_some_and(|target| count >= target) || Instant::now() >= until {
                break;
            }
            match rx.try_recv() {
                Ok(job) => {
                    examined += 1;
                    valid = plan_job(&engine, job, &mut lane, &long);
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(until.saturating_duration_since(Instant::now()).min(Duration::from_micros(100)));
                }
            }
        }
        let start = Instant::now();
        let batch = lane.admit(start);
        // Explicit adaptive history follows engine calls, not empty invalid/cancelled/long-only cycles.
        if !batch.is_empty() || policy.timing.is_none() {
            last_batch = batch.entries.max(1);
        }
        if !batch.is_empty() {
            lane.run(&mut engine, batch, start);
        }
    }
}
