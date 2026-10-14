//! Cooperative sharing of one GPU by two engines on two threads (the server's lanes): an urgent lane (short
//! requests) and a preemptible lane (long requests). Only one lane works on the GPU at a time. The preemptible lane
//! calls [`Gate::checkpoint`] between layers of its forwards and hands the GPU over when an urgent caller waits and
//! it has worked for at least `min_slice` since it last got the GPU. The forwards themselves do not change: a paused
//! forward continues with the same tensors, so results are the same as without the gate.
//!
//! A preemptible lane that continues a started task (it yielded at a checkpoint, or a model split over GPUs moves to
//! the next GPU with [`Gate::resume`]) takes a ticket in a FIFO queue. [`GateGuard`]'s `leave` hands the GPU to the
//! first ticket atomically, so new urgent or preemptible callers cannot overtake it: a queued lane waits at most for
//! the rest of the current turn, one turn of every ticket queued before it and one urgent turn per hand-over at a
//! checkpoint. Urgent callers keep their order among each other (no FIFO between them).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

pub struct Gate {
    st: Mutex<State>,
    cv: Condvar,
    /// Least work of the preemptible lane between two hand-overs (its share of the GPU under urgent load).
    pub min_slice: Duration,
}

#[derive(Default)]
struct State {
    busy: bool,
    urgent_waiting: usize,
    /// Tickets of preemptible lanes waiting to continue a started task, in arrival order. A ticket is queued only
    /// while its thread waits in [`Gate::checkpoint`] or [`Gate::resume`].
    reserved: VecDeque<u64>,
    next_ticket: u64,
    /// The GPU is reserved for this ticket (set by `leave` from the head of `reserved`); only its thread takes it.
    handoff: Option<u64>,
    /// When the current holder got the GPU.
    since: Option<Instant>,
}

/// The GPU, until dropped.
pub struct GateGuard<'a>(&'a Gate);

impl Drop for GateGuard<'_> {
    fn drop(&mut self) {
        self.0.leave();
    }
}

/// A queued ticket until its thread takes the GPU. Only an unwinding wait drops it armed (the waits below do not
/// panic: lock poisoning is tolerated); it then withdraws the ticket, so no hand-over is left to a thread that no
/// longer waits.
struct Ticket<'a> {
    gate: &'a Gate,
    id: u64,
    /// The thread already holds a [`GateGuard`] (parked at a checkpoint): it must end holding the GPU again, so that
    /// its guard's single `leave` stays correct.
    holder: bool,
    armed: bool,
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut s = self.gate.state();
        if self.holder {
            // wait for the turn (bounded as for any queued ticket), then hold the GPU for the guard's leave
            while s.handoff != Some(self.id) {
                s = self.gate.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
            }
            s.take(self.id);
        } else if s.handoff == Some(self.id) {
            // granted but never taken: pass the GPU on
            s.handoff = s.reserved.pop_front();
            self.gate.cv.notify_all();
        } else {
            s.reserved.retain(|&t| t != self.id);
        }
    }
}

impl State {
    /// The ticket's thread takes the GPU it was handed: one step under the lock.
    fn take(&mut self, ticket: u64) {
        debug_assert_eq!(self.handoff, Some(ticket));
        self.handoff = None;
        self.busy = true;
        self.since = Some(Instant::now());
    }

    fn queue(&mut self) -> u64 {
        let id = self.next_ticket;
        self.next_ticket = self.next_ticket.wrapping_add(1);
        self.reserved.push_back(id);
        id
    }
}

impl Gate {
    pub fn new(min_slice: Duration) -> Self {
        Self { st: Mutex::new(State::default()), cv: Condvar::new(), min_slice }
    }

    /// The state lock. Every update under it is a few assignments that leave the state consistent, so a poisoned
    /// lock (a panic elsewhere while holding it) is taken over instead of panicking again.
    fn state(&self) -> MutexGuard<'_, State> {
        self.st.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Wait until `ticket` is handed the GPU, then take it.
    fn wait_turn<'a>(&'a self, mut s: MutexGuard<'a, State>, ticket: &mut Ticket<'_>) -> MutexGuard<'a, State> {
        while s.handoff != Some(ticket.id) {
            s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
        }
        s.take(ticket.id);
        ticket.armed = false;
        s
    }

    /// Wait for the GPU at the start of a task. An urgent caller goes before every preemptible one that has not
    /// started yet and before queued tickets only while none is handed the GPU; a new preemptible task waits for
    /// waiting urgent callers and queued tickets.
    pub fn enter(&self, urgent: bool) -> GateGuard<'_> {
        let mut s = self.state();
        if urgent {
            s.urgent_waiting += 1;
            while s.busy || s.handoff.is_some() {
                s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
            }
            s.urgent_waiting -= 1;
        } else {
            while s.busy || s.handoff.is_some() || s.urgent_waiting > 0 || !s.reserved.is_empty() {
                s = self.cv.wait(s).unwrap_or_else(PoisonError::into_inner);
            }
        }
        s.busy = true;
        s.since = Some(Instant::now());
        GateGuard(self)
    }

    /// Preemptible lane continuing a started task (the next GPU of a split model, or the first one again): takes the
    /// GPU at once only when nobody holds or is handed it, no urgent caller waits and no ticket is queued; otherwise
    /// queues a ticket and waits for its turn.
    pub fn resume(&self) -> GateGuard<'_> {
        let mut s = self.state();
        if !s.busy && s.handoff.is_none() && s.urgent_waiting == 0 && s.reserved.is_empty() {
            s.busy = true;
            s.since = Some(Instant::now());
            return GateGuard(self);
        }
        let id = s.queue();
        let mut ticket = Ticket { gate: self, id, holder: false, armed: true };
        drop(self.wait_turn(s, &mut ticket));
        GateGuard(self)
    }

    fn leave(&self) {
        let mut s = self.state();
        s.busy = false;
        s.since = None;
        s.handoff = s.reserved.pop_front();
        self.cv.notify_all();
    }

    /// Preemptible holder between two steps whose GPU work has completed: if an urgent caller waits and the holder
    /// has had the GPU for `min_slice`, queue a ticket, let the GPU go and take it back at the ticket's turn. True if
    /// it yielded.
    pub fn checkpoint(&self) -> bool {
        let mut s = self.state();
        if s.urgent_waiting == 0 || s.since.is_none_or(|t| t.elapsed() < self.min_slice) {
            return false;
        }
        let id = s.queue();
        let mut ticket = Ticket { gate: self, id, holder: true, armed: true };
        s.busy = false;
        s.since = None;
        self.cv.notify_all();
        drop(self.wait_turn(s, &mut ticket));
        true
    }
}
