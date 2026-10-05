//! Cooperative sharing of one GPU by two engines on two threads (the server's lanes): an urgent lane (short
//! requests) and a preemptible lane (long requests). Only one lane works on the GPU at a time. The preemptible lane
//! calls [`Gate::checkpoint`] between layers of its forwards and hands the GPU over when an urgent caller waits and
//! it has worked for at least `min_slice` since it last got the GPU. The forwards themselves do not change: a paused
//! forward continues with the same tensors, so results are the same as without the gate.

use std::sync::{Condvar, Mutex};
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
    /// The preemptible lane yielded and waits for the end of the urgent holder's turn.
    parked: bool,
    /// The GPU is reserved for the parked lane (set when the urgent holder leaves).
    handoff: bool,
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

impl Gate {
    pub fn new(min_slice: Duration) -> Self {
        Self { st: Mutex::new(State::default()), cv: Condvar::new(), min_slice }
    }

    /// Wait for the GPU. An urgent caller goes before every preemptible one that has not started yet; a lane that
    /// yielded gets the GPU back after one urgent turn.
    pub fn enter(&self, urgent: bool) -> GateGuard<'_> {
        let mut s = self.st.lock().unwrap();
        if urgent {
            s.urgent_waiting += 1;
            while s.busy || s.handoff {
                s = self.cv.wait(s).unwrap();
            }
            s.urgent_waiting -= 1;
        } else {
            while s.busy || s.handoff || s.urgent_waiting > 0 {
                s = self.cv.wait(s).unwrap();
            }
        }
        s.busy = true;
        s.since = Some(Instant::now());
        GateGuard(self)
    }

    fn leave(&self) {
        let mut s = self.st.lock().unwrap();
        s.busy = false;
        s.since = None;
        if s.parked {
            s.parked = false;
            s.handoff = true;
        }
        self.cv.notify_all();
    }

    /// Preemptible holder between two steps whose GPU work has completed: if an urgent caller waits and the holder
    /// has had the GPU for `min_slice`, let one urgent turn run and take the GPU back. True if it yielded.
    pub fn checkpoint(&self) -> bool {
        let mut s = self.st.lock().unwrap();
        if s.urgent_waiting == 0 || s.since.is_none_or(|t| t.elapsed() < self.min_slice) {
            return false;
        }
        s.busy = false;
        s.parked = true;
        self.cv.notify_all();
        while !s.handoff {
            s = self.cv.wait(s).unwrap();
        }
        s.handoff = false;
        s.busy = true;
        s.since = Some(Instant::now());
        true
    }
}
