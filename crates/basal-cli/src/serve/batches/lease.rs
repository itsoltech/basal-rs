//! Renewal of a phase lease (validation, finalization) while its holder reads, plans or writes.
//!
//! The local deadline is derived from the database clock of each answer: `request start + (lease_until −
//! server_now) − safety margin`, so the round trip only shortens it. A renewal that fails is retried until the local
//! deadline; then, or on a stale answer, the lease counts as lost and the [`StopToken`] stops planning. The renewal
//! task never drops a store operation half-way: it is told to stop between renewals.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use basal_core::{StopReason, StopToken};
use tokio_util::sync::CancellationToken;

use super::store::{ClaimToken, LeaseTiming, LeaseUpdate, Store};

/// Local deadline of a lease answered with `timing` for a request sent at `sent`.
pub(super) fn local_deadline(sent: Instant, timing: &LeaseTiming, safety: Duration) -> Instant {
    let remaining = u64::try_from(timing.lease_until.0.saturating_sub(timing.server_now.0)).unwrap_or(0);
    (sent + Duration::from_millis(remaining)).checked_sub(safety).unwrap_or(sent)
}

struct State {
    deadline: Mutex<Instant>,
    lost: AtomicBool,
    stop: StopToken,
}

impl State {
    fn deadline(&self) -> Instant {
        *self.deadline.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn lose(&self) {
        self.lost.store(true, Ordering::SeqCst);
        self.stop.stop(StopReason::LeaseLost);
    }
}

/// Keeps one lease alive; dropping it ends the renewals after the current one.
pub(super) struct LeaseKeeper {
    token: ClaimToken,
    state: Arc<State>,
    done: CancellationToken,
}

impl LeaseKeeper {
    /// Start renewing `token` every `every`. `sent` is when the claim request was sent.
    pub(super) fn start(
        store: Store,
        token: ClaimToken,
        timing: &LeaseTiming,
        sent: Instant,
        every: Duration,
        safety: Duration,
    ) -> Self {
        let state = Arc::new(State {
            deadline: Mutex::new(local_deadline(sent, timing, safety)),
            lost: AtomicBool::new(false),
            stop: StopToken::new(),
        });
        let done = CancellationToken::new();
        // The deadline watchdog stops planning at the local deadline even while a renewal waits on the store; the
        // renewal itself is never dropped half-way.
        let (watch_state, watch_done) = (Arc::clone(&state), done.clone());
        tokio::spawn(async move {
            loop {
                let deadline = watch_state.deadline();
                tokio::select! {
                    () = watch_done.cancelled() => break,
                    () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {}
                }
                // A renewal may have moved the deadline meanwhile.
                if Instant::now() >= watch_state.deadline() {
                    watch_state.lose();
                    break;
                }
            }
        });
        let (task_state, task_done, task_token) = (Arc::clone(&state), done.clone(), token.clone());
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    () = task_done.cancelled() => break,
                    () = tokio::time::sleep(every) => {}
                }
                if task_state.lost.load(Ordering::SeqCst) {
                    // Lost by the deadline watchdog: a lost lease is never renewed back.
                    break;
                }
                let sent = Instant::now();
                match store.renew_claim(&task_token).await {
                    Ok(LeaseUpdate::Applied(renewal)) => {
                        *task_state.deadline.lock().unwrap_or_else(PoisonError::into_inner) =
                            local_deadline(sent, &renewal.timing, safety);
                    }
                    Ok(LeaseUpdate::Stale) => {
                        task_state.lose();
                        break;
                    }
                    // Retried at the next tick while the local deadline holds.
                    Err(_) if Instant::now() < task_state.deadline() => {}
                    Err(_) => {
                        task_state.lose();
                        break;
                    }
                }
            }
        });
        Self { token, state, done }
    }

    pub(super) fn token(&self) -> &ClaimToken {
        &self.token
    }

    /// The lease was lost (stale answer or local deadline passed without a renewal).
    pub(super) fn lost(&self) -> bool {
        if !self.state.lost.load(Ordering::SeqCst) && Instant::now() >= self.state.deadline() {
            self.state.lose();
        }
        self.state.lost.load(Ordering::SeqCst)
    }

    /// Current local deadline (moves with every renewal).
    pub(super) fn deadline(&self) -> Instant {
        self.state.deadline()
    }

    /// Stops planning when the lease is lost.
    pub(super) fn stop_token(&self) -> StopToken {
        self.state.stop.clone()
    }
}

impl Drop for LeaseKeeper {
    fn drop(&mut self) {
        self.done.cancel();
    }
}
