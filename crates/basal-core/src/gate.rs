//! Cooperative sharing of one GPU by the engines on it (the server's lanes), in three work classes
//! ([`WorkClass`]): the online main lane (short requests, formerly "urgent"), the online long lane (preemptible
//! between layers) and background batch work. Only one engine works on the GPU at a time. The forwards themselves do
//! not change: a paused forward continues with the same tensors, so results are the same as without the gate.
//!
//! Online semantics are those of the former two-lane gate. The long lane calls [`Gate::checkpoint`] between layers
//! and hands the GPU over when a main-lane caller waits and it has worked for at least `min_slice` since it last got
//! the GPU. A long lane that continues a started task (it yielded at a checkpoint, or a split model moves to the next
//! GPU with [`Gate::resume`]) takes a ticket in a FIFO queue; leaving hands the GPU to the first ticket atomically, so
//! new callers cannot overtake it. Main-lane callers keep their order among each other (no FIFO between them).
//! [`Gate::with_long_share`] additionally conditions long-lane checkpoint yielding on its recent share of online
//! occupied wall time, aggregated across every engine using this gate. [`Gate::new`] keeps the legacy policy.
//!
//! Background work takes the GPU only while no online work holds it, waits for it, holds a ticket or is announced by
//! an [`OnlineDemand`]. The claim is evaluated under the state lock at the moment of taking the GPU, so online demand
//! registered before that moment always wins (there is no unclaimed background hand-over to revoke). A background
//! holder at a checkpoint (after draining its own GPU work) yields to any pending online work at once, without a
//! minimum slice. Its waits observe its [`StopToken`]: a stop withdraws the waiter and returns [`Interrupted`]
//! without re-acquiring the GPU. Background waiters keep FIFO order among each other; resumes go first.
//!
//! [`GateGuard`] has an explicit possession state: a checkpoint consumes the guard and returns a guard only after the
//! GPU was taken again; an `Err` leaves nothing held.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use crate::execution::{Interrupted, StopReason, StopToken, StopWaker, WorkClass};

/// Longest wait of a background waiter between re-checks of its stop token, deadline and pause limit. A working
/// parameter to be measured; stops and demand changes wake waiters directly.
pub const BACKGROUND_POLL: Duration = Duration::from_millis(50);

/// A long-share target was not finite or was outside the inclusive range `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("long share must be finite and within [0, 1]")]
pub struct InvalidLongShare;

pub struct Gate {
    inner: Arc<Inner>,
    /// Minimum uninterrupted online long possession before a checkpoint may yield to a main caller.
    pub min_slice: Duration,
}

struct Inner {
    st: Mutex<State>,
    cv: Condvar,
}

impl StopWaker for Inner {
    fn wake(&self) {
        // Take the lock so that a waiter is either before its check (and sees the stop) or parked (and is woken).
        let _s = self.st.lock().unwrap_or_else(PoisonError::into_inner);
        self.cv.notify_all();
    }
}

#[derive(Default)]
struct State {
    busy: bool,
    holder: Option<WorkClass>,
    urgent_waiting: usize,
    /// Online long lanes waiting to start a new task ([`Gate::enter`] with `urgent = false`).
    long_waiting: usize,
    /// Tickets of online long lanes waiting to continue a started task, in arrival order. A ticket is queued only
    /// while its thread waits in [`Gate::checkpoint`] or [`Gate::resume`].
    reserved: VecDeque<u64>,
    next_ticket: u64,
    /// The GPU is reserved for this ticket (set by `leave` from the head of `reserved`); only its thread takes it.
    handoff: Option<u64>,
    /// When the current holder got the GPU.
    since: Option<Instant>,
    /// Live [`OnlineDemand`] registrations.
    online_demand: usize,
    /// Background waiters in claim order (resumes at the front). Only the head may take the GPU.
    background: VecDeque<u64>,
    /// Optional device-wide policy and history; absent for the legacy constructor.
    long_share: Option<LongShare>,
}

/// Exact exponential integration of held online wall time, with a two-second time constant. The timestamp
/// advances independently of `State::since`, which always marks an uninterrupted acquisition.
struct LongShare {
    target: f64,
    main: f64,
    long: f64,
    updated: Instant,
}

impl LongShare {
    fn account(&mut self, now: Instant, holder: Option<WorkClass>) {
        let exponent = -now.duration_since(self.updated).as_secs_f64() / 2.0;
        let decay = exponent.exp();
        // exp_m1 preserves small held intervals that would disappear in `1.0 - decay`.
        let occupied = -2.0 * exponent.exp_m1();
        self.main *= decay;
        self.long *= decay;
        match holder {
            Some(WorkClass::OnlineMain) => self.main += occupied,
            Some(WorkClass::OnlineLong) => self.long += occupied,
            None | Some(WorkClass::Background) => {}
        }
        self.updated = now;
    }
}

impl State {
    /// The ticket's thread takes the GPU it was handed: one step under the lock.
    fn take(&mut self, ticket: u64) {
        debug_assert_eq!(self.handoff, Some(ticket));
        self.handoff = None;
        self.hold(WorkClass::OnlineLong);
    }

    fn hold(&mut self, class: WorkClass) {
        let now = Instant::now();
        if let Some(share) = &mut self.long_share {
            share.account(now, self.holder);
        }
        self.busy = true;
        self.holder = Some(class);
        self.since = Some(now);
    }

    /// End actual possession without granting a queued handoff. A reserved ticket holds no occupied time.
    fn unhold(&mut self) {
        if let Some(share) = &mut self.long_share {
            share.account(Instant::now(), self.holder);
        }
        self.busy = false;
        self.holder = None;
        self.since = None;
    }

    /// Called only with a main waiter after the minimum slice. Include live held time in the comparison.
    fn long_share_allows_yield(&mut self) -> bool {
        let Some(share) = &mut self.long_share else {
            return true;
        };
        if share.target == 0.0 {
            return true;
        }
        if share.target == 1.0 {
            return false;
        }
        share.account(Instant::now(), self.holder);
        let total = share.main + share.long;
        // With no representable history, the current long holder supplies the entire online share.
        total == 0.0 || share.long / total >= share.target
    }

    fn ticket(&mut self) -> u64 {
        let id = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1);
        id
    }

    fn queue(&mut self) -> u64 {
        let id = self.ticket();
        self.reserved.push_back(id);
        id
    }

    /// Online work holds, waits for, is handed or announced the GPU.
    fn online_pending(&self) -> bool {
        self.online_demand > 0
            || self.urgent_waiting > 0
            || self.long_waiting > 0
            || !self.reserved.is_empty()
            || self.handoff.is_some()
            || (self.busy && self.holder != Some(WorkClass::Background))
    }

    /// Release the GPU: hand it to the first queued online long ticket, if any.
    fn release(&mut self) {
        self.unhold();
        self.handoff = self.reserved.pop_front();
    }
}

/// The GPU, until dropped (or consumed by [`Gate::checkpoint`]).
pub struct GateGuard<'a> {
    gate: &'a Gate,
    class: WorkClass,
    /// Invariant: true exactly while this guard's thread holds the GPU through it.
    armed: bool,
    /// Background guards with a stop token: the token and the instant the gate was acquired; the held interval is
    /// added to [`StopToken::device_time`] exactly once, when the guard releases or hands over the gate. Online
    /// guards carry none (and read no clock for it).
    meter: Option<(StopToken, Instant)>,
}

impl GateGuard<'_> {
    pub fn class(&self) -> WorkClass {
        self.class
    }

    /// Record the held interval (at most once per guard).
    fn record(&mut self) {
        if let Some((stop, since)) = self.meter.take() {
            stop.add_device_time(since.elapsed());
        }
    }

    /// The GPU leaves this guard without a release (the caller releases it under the same lock); its held interval
    /// is recorded first, before any hand-over.
    fn disarm(mut self) {
        self.record();
        self.armed = false;
    }
}

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // the held interval ends before the gate is handed over
            self.record();
            self.gate.leave();
        }
    }
}

/// A queued online long ticket until its thread takes the GPU. Only an unwinding wait drops it armed (the waits do
/// not panic: lock poisoning is tolerated); it then withdraws the ticket or passes on a granted hand-over. It never
/// waits for a turn on drop.
struct Ticket<'a> {
    gate: &'a Gate,
    id: u64,
    armed: bool,
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut s = self.gate.state();
        if s.handoff == Some(self.id) {
            s.handoff = s.reserved.pop_front();
        } else {
            s.reserved.retain(|&t| t != self.id);
        }
        self.gate.inner.cv.notify_all();
    }
}

/// A queued background waiter until it takes the GPU or gives up; dropping it armed withdraws it.
struct BackgroundTicket<'a> {
    gate: &'a Gate,
    id: u64,
    armed: bool,
}

impl Drop for BackgroundTicket<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut s = self.gate.state();
        s.background.retain(|&t| t != self.id);
        // the next background waiter may now be the head
        self.gate.inner.cv.notify_all();
    }
}

/// Outcome of a successful [`Gate::checkpoint`].
pub struct Checkpoint<'a> {
    /// The GPU, held again.
    pub guard: GateGuard<'a>,
    /// The holder let the GPU go and took it back: other work may have run (callers re-check device health).
    pub yielded: bool,
}

impl Gate {
    /// Create a legacy gate: a long holder yields whenever a main caller waits and `min_slice` has elapsed.
    pub fn new(min_slice: Duration) -> Self {
        Self { inner: Arc::new(Inner { st: Mutex::new(State::default()), cv: Condvar::new() }), min_slice }
    }

    /// Create a gate with a long-share target: a long holder yields at a checkpoint
    /// only when a main caller waits, `min_slice` has elapsed since acquisition, and its recent online occupied
    /// share is at least `long_share`. Zero uses the legacy yield condition; one never yields mid-forward.
    /// Starting a new long task still gives main callers priority, and an empty main queue never causes a pause.
    ///
    /// One policy and account cover all engines sharing this gate. The account integrates held main/long wall
    /// time with exponential memory of two seconds: over `dt`, both totals decay by `exp(-dt / 2)` and the online
    /// holder receives `2 * (1 - exp(-dt / 2))`. Checkpoints include the live interval. Idle, background and parked
    /// ticket time only decay history. With zero total history the current long holder's share is taken as one,
    /// subject to the explicit target endpoints above. Interval composition is independent of checkpoint count
    /// up to floating-point rounding. This is a native continuous interpretation, not upstream's lump sums.
    /// It measures gate occupation, not GPU utilization, and provides no speed or fairness guarantee.
    ///
    /// # Errors
    /// Returns [`InvalidLongShare`] for NaN, infinity or a value outside `[0, 1]`.
    pub fn with_long_share(min_slice: Duration, long_share: f64) -> Result<Self, InvalidLongShare> {
        if !long_share.is_finite() || !(0.0..=1.0).contains(&long_share) {
            return Err(InvalidLongShare);
        }
        let state = State {
            long_share: Some(LongShare { target: long_share, main: 0.0, long: 0.0, updated: Instant::now() }),
            ..State::default()
        };
        Ok(Self { inner: Arc::new(Inner { st: Mutex::new(state), cv: Condvar::new() }), min_slice })
    }

    /// The state lock. Every update under it is a few assignments that leave the state consistent, so a poisoned
    /// lock (a panic elsewhere while holding it) is taken over instead of panicking again.
    fn state(&self) -> MutexGuard<'_, State> {
        self.inner.st.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'s>(&self, s: MutexGuard<'s, State>) -> MutexGuard<'s, State> {
        self.inner.cv.wait(s).unwrap_or_else(PoisonError::into_inner)
    }

    fn guard(&self, class: WorkClass) -> GateGuard<'_> {
        GateGuard { gate: self, class, armed: true, meter: None }
    }

    /// Online work is holding, waiting for or announced on this GPU (diagnostics and background admission).
    pub fn online_pending(&self) -> bool {
        self.state().online_pending()
    }

    /// Wait until `ticket` is handed the GPU, then take it.
    fn wait_turn<'a>(&'a self, mut s: MutexGuard<'a, State>, ticket: &mut Ticket<'_>) -> MutexGuard<'a, State> {
        while s.handoff != Some(ticket.id) {
            s = self.wait(s);
        }
        s.take(ticket.id);
        ticket.armed = false;
        s
    }

    /// Online lane waiting for the GPU at the start of a task. A main-lane caller (`urgent`) goes before every
    /// online long and background caller that has not started yet and before queued tickets only while none is
    /// handed the GPU; a new long task waits for waiting main-lane callers and queued tickets.
    pub fn enter(&self, urgent: bool) -> GateGuard<'_> {
        let mut s = self.state();
        if urgent {
            s.urgent_waiting += 1;
            while s.busy || s.handoff.is_some() {
                s = self.wait(s);
            }
            s.urgent_waiting -= 1;
        } else {
            s.long_waiting += 1;
            while s.busy || s.handoff.is_some() || s.urgent_waiting > 0 || !s.reserved.is_empty() {
                s = self.wait(s);
            }
            s.long_waiting -= 1;
        }
        let class = WorkClass::from_urgent(urgent);
        s.hold(class);
        self.guard(class)
    }

    /// Online long lane continuing a started task (the next GPU of a split model, or the first one again): takes
    /// the GPU at once only when nobody holds or is handed it, no main-lane caller waits and no ticket is queued;
    /// otherwise queues a ticket and waits for its turn.
    pub fn resume(&self) -> GateGuard<'_> {
        let mut s = self.state();
        if !s.busy && s.handoff.is_none() && s.urgent_waiting == 0 && s.reserved.is_empty() {
            s.hold(WorkClass::OnlineLong);
            return self.guard(WorkClass::OnlineLong);
        }
        let id = s.queue();
        let mut ticket = Ticket { gate: self, id, armed: true };
        drop(self.wait_turn(s, &mut ticket));
        self.guard(WorkClass::OnlineLong)
    }

    /// Wait for the GPU at the start of a task in `class`. Online classes behave as [`Gate::enter`] and do not
    /// observe `stop`; background requires its stop token (`None` runs with a token that never stops).
    pub fn enter_class(&self, class: WorkClass, stop: Option<&StopToken>) -> Result<GateGuard<'_>, Interrupted> {
        match class {
            WorkClass::OnlineMain => Ok(self.enter(true)),
            WorkClass::OnlineLong => Ok(self.enter(false)),
            WorkClass::Background => self.background_wait(stop, false),
        }
    }

    /// Continue a started task in `class` (next stage of a split, or the same GPU again). The main lane enters as
    /// at the start (it never queues), the long lane queues a FIFO ticket, background goes first among background
    /// waiters but still after all pending online work.
    pub fn resume_class(&self, class: WorkClass, stop: Option<&StopToken>) -> Result<GateGuard<'_>, Interrupted> {
        match class {
            WorkClass::OnlineMain => Ok(self.enter(true)),
            WorkClass::OnlineLong => Ok(self.resume()),
            WorkClass::Background => self.background_wait(stop, true),
        }
    }

    fn leave(&self) {
        let mut s = self.state();
        s.release();
        self.inner.cv.notify_all();
    }

    /// Holder between two steps whose own GPU work has completed (the caller drained it). Consumes the guard:
    ///
    /// - main lane: returns it unchanged (the main lane does not yield);
    /// - long lane: if a main-lane caller waits and the lane has had the GPU for `min_slice`, queue a ticket, let the
    ///   GPU go and take it back at the ticket's turn (never fails), subject to [`Self::with_long_share`] if set;
    /// - background: once `stop` is stopped, release the GPU and return [`Interrupted`]; if any online work is
    ///   pending, release the GPU at once and wait for it again (first among background waiters), observing `stop`,
    ///   its deadline and its pause limit. `Err` leaves the GPU released.
    pub fn checkpoint<'a>(
        &'a self,
        guard: GateGuard<'a>,
        stop: Option<&StopToken>,
    ) -> Result<Checkpoint<'a>, Interrupted> {
        debug_assert!(std::ptr::eq(guard.gate, self), "a guard is checkpointed at its own gate");
        match guard.class {
            WorkClass::OnlineMain => Ok(Checkpoint { guard, yielded: false }),
            WorkClass::OnlineLong => {
                let mut s = self.state();
                if s.urgent_waiting == 0
                    || s.since.is_none_or(|t| t.elapsed() < self.min_slice)
                    || !s.long_share_allows_yield()
                {
                    return Ok(Checkpoint { guard, yielded: false });
                }
                guard.disarm();
                let id = s.queue();
                let mut ticket = Ticket { gate: self, id, armed: true };
                // Advance from the last accounting timestamp, charging any post-comparison time only once.
                s.unhold();
                self.inner.cv.notify_all();
                drop(self.wait_turn(s, &mut ticket));
                Ok(Checkpoint { guard: self.guard(WorkClass::OnlineLong), yielded: true })
            }
            WorkClass::Background => {
                let stopped = stop.and_then(StopToken::reason);
                let mut s = self.state();
                if stopped.is_none() && !s.online_pending() {
                    return Ok(Checkpoint { guard, yielded: false });
                }
                guard.disarm();
                s.release();
                self.inner.cv.notify_all();
                drop(s);
                if let Some(reason) = stopped {
                    return Err(Interrupted { reason });
                }
                let guard = self.background_wait(stop, true)?;
                Ok(Checkpoint { guard, yielded: true })
            }
        }
    }

    /// Background waiter: take the GPU when it is the head of the background queue, the GPU is free and no online
    /// work is pending. Observes the stop token, its deadline and its pause limit; on stop the waiter withdraws.
    /// The returned guard meters its held time into the caller's token from the moment of acquisition.
    ///
    /// Lock order: a stop is only *evaluated* under the state lock ([`StopToken::due`]); it is recorded after the
    /// lock is released, because recording wakes registered gates, which takes their state locks.
    fn background_wait(&self, stop: Option<&StopToken>, front: bool) -> Result<GateGuard<'_>, Interrupted> {
        let metered = stop;
        let never;
        let stop = match stop {
            Some(t) => t,
            None => {
                never = StopToken::new();
                &never
            }
        };
        let waker: Weak<dyn StopWaker> = {
            let strong: Arc<dyn StopWaker> = self.inner.clone();
            Arc::downgrade(&strong)
        };
        stop.register(waker);
        stop.check()?;
        let started = Instant::now();
        let pause = stop.pause_limit();
        let mut s = self.state();
        let id = s.ticket();
        if front {
            s.background.push_front(id);
        } else {
            s.background.push_back(id);
        }
        let mut ticket = BackgroundTicket { gate: self, id, armed: true };
        loop {
            // a stop (recorded, an elapsed deadline or the pause limit) wins over a free GPU
            let now = Instant::now();
            let due = stop.due().or_else(|| {
                pause.filter(|limit| now.duration_since(started) >= *limit).map(|_| StopReason::PauseLimit)
            });
            if let Some(reason) = due {
                drop(s);
                drop(ticket); // withdraws and wakes the next background waiter
                stop.stop(reason);
                // the first recorded reason wins
                return Err(Interrupted { reason: stop.reason().unwrap_or(reason) });
            }
            if !s.busy && !s.online_pending() && s.background.front() == Some(&id) {
                s.background.pop_front();
                ticket.armed = false;
                s.hold(WorkClass::Background);
                let mut guard = self.guard(WorkClass::Background);
                guard.meter = metered.map(|t| (t.clone(), Instant::now()));
                return Ok(guard);
            }
            let mut timeout = BACKGROUND_POLL;
            if let Some(limit) = pause {
                timeout = timeout.min(limit.saturating_sub(now.duration_since(started)));
            }
            if let Some((at, _)) = stop.deadline() {
                timeout = timeout.min(at.saturating_duration_since(now));
            }
            s = self.inner.cv.wait_timeout(s, timeout).unwrap_or_else(PoisonError::into_inner).0;
        }
    }
}

/// Announced online work on the GPUs of one instance, from admission until the response (all rounds and media),
/// including error, abandonment and worker panic (RAII). While any registration is live on a gate, background work
/// neither starts nor resumes there and yields at its next checkpoint.
///
/// Register the gates of every GPU of the instance (a layer split registers all of them): conservative for the
/// cards of that instance, without blocking devices of other models.
pub struct OnlineDemand {
    gates: Vec<Arc<Gate>>,
}

impl OnlineDemand {
    pub fn register(gates: impl IntoIterator<Item = Arc<Gate>>) -> Self {
        let mut unique: Vec<Arc<Gate>> = Vec::new();
        for gate in gates {
            if !unique.iter().any(|g| Arc::ptr_eq(g, &gate)) {
                unique.push(gate);
            }
        }
        for gate in &unique {
            gate.state().online_demand += 1;
        }
        Self { gates: unique }
    }

    /// Number of distinct gates of this registration.
    pub fn gates(&self) -> usize {
        self.gates.len()
    }
}

impl std::fmt::Debug for OnlineDemand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnlineDemand").field("gates", &self.gates.len()).finish()
    }
}

impl Drop for OnlineDemand {
    fn drop(&mut self) {
        for gate in &self.gates {
            let mut s = gate.state();
            s.online_demand = s.online_demand.saturating_sub(1);
            if s.online_demand == 0 {
                gate.inner.cv.notify_all();
            }
        }
    }
}
