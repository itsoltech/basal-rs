//! Execution control shared by the engines of one device: the work class of an engine (online short lane, online
//! long lane or background batch work) and a cooperative stop signal for background work.
//!
//! Background work is never preempted inside a kernel. It observes its [`StopToken`] at gate entry, when it resumes
//! after yielding, at every checkpoint between layers or encoder blocks and between engine rounds. An observed stop
//! ends the call with a typed [`Interrupted`] error in the `anyhow` chain; callers recognise it with
//! [`interrupted`], never by message text.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

/// Priority class of an engine at the device gate. Online classes keep the semantics of the former `urgent` flag:
/// [`WorkClass::OnlineMain`] was `urgent = true`, [`WorkClass::OnlineLong`] `urgent = false`. Background work runs
/// only while no online work is waiting or announced ([`crate::gate::OnlineDemand`]) and yields at every checkpoint
/// when online work appears.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkClass {
    OnlineMain,
    OnlineLong,
    Background,
}

impl WorkClass {
    /// The former boolean lane flag (`urgent`) of an online class.
    pub fn from_urgent(urgent: bool) -> Self {
        if urgent {
            Self::OnlineMain
        } else {
            Self::OnlineLong
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::OnlineMain => "online_main",
            Self::OnlineLong => "online_long",
            Self::Background => "background",
        }
    }
}

/// Why background work was stopped. The first recorded reason wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StopReason {
    /// The batch was cancelled.
    Cancelled = 1,
    /// The batch completion window ended.
    Expired = 2,
    /// The claim of the work could not be renewed in time (or was taken over).
    LeaseLost = 3,
    /// The process is shutting down.
    Shutdown = 4,
    /// Online work needs the memory held by the paused background state.
    MemoryPressure = 5,
    /// The background work waited longer than its pause limit for the GPU.
    PauseLimit = 6,
}

impl StopReason {
    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::Cancelled,
            2 => Self::Expired,
            3 => Self::LeaseLost,
            4 => Self::Shutdown,
            5 => Self::MemoryPressure,
            6 => Self::PauseLimit,
            _ => return None,
        })
    }

    /// Stable label for metrics and logs (no client data).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
            Self::LeaseLost => "lease_lost",
            Self::Shutdown => "shutdown",
            Self::MemoryPressure => "memory_pressure",
            Self::PauseLimit => "pause_limit",
        }
    }
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Background work ended early at a safe point. It is not a device failure: the GPU work of the call was drained
/// before the error was returned, or no GPU work was started.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("background work interrupted: {reason}")]
pub struct Interrupted {
    pub reason: StopReason,
}

/// The stop reason when `error` (or any error in its chain) is [`Interrupted`].
pub fn interrupted(error: &anyhow::Error) -> Option<StopReason> {
    error.chain().find_map(|e| e.downcast_ref::<Interrupted>()).map(|i| i.reason)
}

/// Typed errors of the execution-control APIs of [`crate::Backend`] and [`crate::Engine`].
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// The backend cannot run this work class or control; the setting was not applied.
    #[error("unsupported execution control: {reason}")]
    Unsupported { reason: String },
    /// A gate for a device the backend does not run on.
    #[error("a gate for device {device}, this model runs on device(s) {devices:?}")]
    NoSuchDevice { device: usize, devices: Vec<usize> },
    /// The backend refused the gate of an online class.
    #[error("setting the device gate")]
    Gate(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A device operation of the execution control (such as releasing finished work) failed.
    #[error("device operation of the execution control failed")]
    Device(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The engine cannot change its execution control in its current state.
    #[error("invalid execution state: {reason}")]
    InvalidState { reason: String },
}

/// Whether a backend can run background work. A backend that cannot observe a [`StopToken`] at its checkpoints, or
/// whose GPU work cannot be drained and released cooperatively, reports `Unsupported` with the reason; background
/// stays disabled for it instead of ignoring the controls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackgroundSupport {
    Supported,
    Unsupported(String),
}

impl BackgroundSupport {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }
}

/// Something parked on behalf of a [`StopToken`] holder (a gate's condition variable) that must be woken when the
/// token stops, so that a cancelled waiter does not wait for the emergency timeout.
pub(crate) trait StopWaker: Send + Sync {
    fn wake(&self);
}

struct StopInner {
    /// 0: running; otherwise a [`StopReason`] discriminant (first writer wins).
    reason: AtomicU8,
    /// Local monotonic deadline and the reason it stops with (tightened only).
    deadline: Mutex<Option<(Instant, StopReason)>>,
    /// Longest single wait of background work for the GPU before it stops with [`StopReason::PauseLimit`].
    pause_limit: Mutex<Option<Duration>>,
    /// Waiters to wake on stop. Weak, deduplicated by pointer; dead entries are pruned on registration.
    wakers: Mutex<Vec<Weak<dyn StopWaker>>>,
    /// Nanoseconds background gate guards carrying this token held their gate (saturating, never wraps).
    device_ns: AtomicU64,
}

/// Cooperative stop signal of one unit of background work (a micro-batch). Cheap to clone; all clones observe the
/// same state. Checking it is lock-free unless a deadline is set.
#[derive(Clone)]
pub struct StopToken(Arc<StopInner>);

impl Default for StopToken {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StopToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StopToken")
            .field("reason", &self.reason())
            .field("deadline", &self.deadline())
            .field("device_time", &self.device_time())
            .finish()
    }
}

impl StopToken {
    pub fn new() -> Self {
        Self(Arc::new(StopInner {
            reason: AtomicU8::new(0),
            deadline: Mutex::new(None),
            pause_limit: Mutex::new(None),
            wakers: Mutex::new(Vec::new()),
            device_ns: AtomicU64::new(0),
        }))
    }

    /// Cumulative time background gate guards carrying this token held their device gate: from the moment a guard
    /// was actually acquired ([`crate::gate::Gate::enter_class`], [`crate::gate::Gate::resume_class`] or a yielding
    /// [`crate::gate::Gate::checkpoint`]) until it was released, handed over at a yielding or stopped checkpoint, or
    /// dropped; summed over every guard and every device (stages of a split) using clones of this token. A
    /// non-yielding checkpoint continues the running interval. Time spent waiting for a gate, parked at a
    /// checkpoint or after a stop adds nothing.
    ///
    /// This is the scheduler's GPU service occupancy as wall time on a monotonic clock: it includes host work done
    /// while holding the gate (preparing inputs, launching kernels, waiting for drains and synchronous readouts)
    /// and excludes nothing the holder did with the gate held. It is **not** hardware busy time: no GPU timer or
    /// event is read, and idle gaps between kernels count. Intervals still held are not included until released.
    /// Nondecreasing; saturates at `u64::MAX` nanoseconds.
    pub fn device_time(&self) -> Duration {
        Duration::from_nanos(self.0.device_ns.load(Ordering::Acquire))
    }

    /// Add one released gate interval to [`StopToken::device_time`] (lock-free, saturating).
    pub(crate) fn add_device_time(&self, held: Duration) {
        let ns = u64::try_from(held.as_nanos()).unwrap_or(u64::MAX);
        let counter = &self.0.device_ns;
        let mut current = counter.load(Ordering::Acquire);
        loop {
            match counter.compare_exchange_weak(
                current,
                current.saturating_add(ns),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(seen) => current = seen,
            }
        }
    }

    /// The reason the token stops with now (recorded, or an elapsed deadline) without recording it, so that it can
    /// be evaluated while a gate's state lock is held: recording wakes registered gates, which takes their locks.
    pub(crate) fn due(&self) -> Option<StopReason> {
        StopReason::from_u8(self.0.reason.load(Ordering::Acquire))
            .or_else(|| self.deadline().filter(|(at, _)| Instant::now() >= *at).map(|(_, reason)| reason))
    }

    /// Record `reason` unless another reason was recorded first; wakes registered waiters. True if this call set it.
    pub fn stop(&self, reason: StopReason) -> bool {
        let set = self.0.reason.compare_exchange(0, reason as u8, Ordering::AcqRel, Ordering::Acquire).is_ok();
        if set {
            self.wake_all();
        }
        set
    }

    /// The recorded reason, including an elapsed deadline (which is recorded on first observation).
    pub fn reason(&self) -> Option<StopReason> {
        if let Some(r) = StopReason::from_u8(self.0.reason.load(Ordering::Acquire)) {
            return Some(r);
        }
        let due = self.deadline().filter(|(at, _)| Instant::now() >= *at);
        if let Some((_, reason)) = due {
            self.stop(reason);
            return StopReason::from_u8(self.0.reason.load(Ordering::Acquire));
        }
        None
    }

    pub fn is_stopped(&self) -> bool {
        self.reason().is_some()
    }

    /// `Err` once stopped (or past the deadline).
    pub fn check(&self) -> Result<(), Interrupted> {
        match self.reason() {
            Some(reason) => Err(Interrupted { reason }),
            None => Ok(()),
        }
    }

    /// Set a local monotonic deadline at which the token stops with `reason`; an earlier existing deadline is kept.
    pub fn set_deadline(&self, at: Instant, reason: StopReason) {
        let mut d = self.0.deadline.lock().unwrap_or_else(PoisonError::into_inner);
        if d.is_none_or(|(old, _)| at < old) {
            *d = Some((at, reason));
        }
        drop(d);
        // a waiter computes its timeout from the deadline: let it recompute
        self.wake_all();
    }

    pub fn deadline(&self) -> Option<(Instant, StopReason)> {
        *self.0.deadline.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Longest single wait for the GPU (gate entry, resume or a checkpoint yield) before the token stops with
    /// [`StopReason::PauseLimit`].
    pub fn set_pause_limit(&self, limit: Duration) {
        *self.0.pause_limit.lock().unwrap_or_else(PoisonError::into_inner) = Some(limit);
    }

    pub fn pause_limit(&self) -> Option<Duration> {
        *self.0.pause_limit.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Register a waiter woken by [`StopToken::stop`] and deadline changes.
    pub(crate) fn register(&self, waker: Weak<dyn StopWaker>) {
        let mut w = self.0.wakers.lock().unwrap_or_else(PoisonError::into_inner);
        w.retain(|x| x.strong_count() > 0);
        if !w.iter().any(|x| Weak::ptr_eq(x, &waker)) {
            w.push(waker);
        }
    }

    fn wake_all(&self) {
        // Upgrade under the registry lock, wake after releasing it: a waker takes its own state lock.
        let live: Vec<Arc<dyn StopWaker>> = {
            let w = self.0.wakers.lock().unwrap_or_else(PoisonError::into_inner);
            w.iter().filter_map(Weak::upgrade).collect()
        };
        for waker in live {
            waker.wake();
        }
    }
}
