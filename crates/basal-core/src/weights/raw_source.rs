//! Held admission for exact requested layouts of private raw source buffers.

use std::alloc::{alloc, Layout};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

/// Shared allowance for requested layouts admitted by GGUF source routes.
///
/// Opt-in routes also use this ledger for declaration and immutable semantic
/// backing, BPE merge scratch, the built BPE merge, reverse and forward vocabulary
/// tables and decoder regexes' compiled backing. Each table's or regex's charge
/// remains held until its last shared owner drops it.
///
/// Create one handle before loads and pass it (or its sharing clones) to every
/// concurrent load/reload for that allowance. Independent handles have independent
/// limits. CLI/Engine do not provide this application context yet.
///
/// Excludes bootstrap/control blocks, objects, paths, names, records, errors,
/// other serde/native/tokenizer/cache heaps, GGUF scratch, staging and device memory.
/// Allocator rounding, OS/RSS and physical OOM protection are also excluded.
#[derive(Clone)]
pub struct GgufRawSourceBudget {
    ledger: Arc<RawLedger>,
}

impl GgufRawSourceBudget {
    /// Create an independent allowance. Zero admits no nonempty raw buffer;
    /// equality with the limit is accepted. Bootstrap allocation is not fallible.
    pub fn new(limit_bytes: usize) -> Self {
        let state = RawLedgerState { limit: limit_bytes, held: 0, failed_closed: false };
        Self { ledger: Arc::new(RawLedger { state: Mutex::new(state) }) }
    }

    pub(super) fn charge(&self) -> RawCharge {
        RawCharge { ledger: Arc::clone(&self.ledger), bytes: 0 }
    }

    /// The same ledger, for the per-block account of an opt-in BPE merge,
    /// reverse or forward vocabulary table or decoder regex.
    pub(super) fn table_account(&self) -> Arc<RawLedger> {
        Arc::clone(&self.ledger)
    }
}

/// Fixed raw-content admission/allocation causes; no input or path is retained.
#[derive(Debug, thiserror::Error)]
pub enum GgufRawSourceError {
    #[error("raw source ledger is busy")]
    Busy,
    #[error("raw source ledger is poisoned or closed")]
    Poisoned,
    #[error("raw source charge arithmetic overflows")]
    Overflow,
    #[error("raw source layout of {requested} bytes exceeds allowance (held {held}, limit {limit})")]
    Limit { requested: usize, held: usize, limit: usize },
    #[error("invalid raw source layout of {bytes} bytes")]
    Layout { bytes: usize },
    #[error("raw source allocation of {bytes} bytes failed")]
    Allocation { bytes: usize },
}

/// Fixed ledger refusal; `RawCharge` and the table account map it 1:1.
pub(super) enum LedgerRefusal {
    Busy,
    Poisoned,
    Overflow,
    Limit { requested: usize, held: usize, limit: usize },
}

impl From<LedgerRefusal> for GgufRawSourceError {
    fn from(refusal: LedgerRefusal) -> Self {
        match refusal {
            LedgerRefusal::Busy => Self::Busy,
            LedgerRefusal::Poisoned => Self::Poisoned,
            LedgerRefusal::Overflow => Self::Overflow,
            LedgerRefusal::Limit { requested, held, limit } => Self::Limit { requested, held, limit },
        }
    }
}

/// The shared state of one allowance. Every charge and charged table or regex
/// block of that allowance acquires and releases here.
pub(super) struct RawLedger {
    state: Mutex<RawLedgerState>,
}

struct RawLedgerState {
    limit: usize,
    held: usize,
    failed_closed: bool,
}

impl RawLedger {
    /// All-or-none admission of `requested` bytes; a refusal changes nothing.
    pub(super) fn try_acquire(&self, requested: usize) -> Result<(), LedgerRefusal> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(LedgerRefusal::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(LedgerRefusal::Poisoned),
        };
        if state.failed_closed {
            return Err(LedgerRefusal::Poisoned);
        }
        let held = state.held.checked_add(requested).ok_or(LedgerRefusal::Overflow)?;
        if held > state.limit {
            return Err(LedgerRefusal::Limit { requested, held: state.held, limit: state.limit });
        }
        // All-or-none arithmetic only: held equals the sum of outstanding charges.
        // Charge may conservatively exceed live layouts after a failed allocation.
        state.held = held;
        Ok(())
    }

    /// Returns `bytes` acquired earlier, without allocating; underflow fails closed.
    pub(super) fn release(&self, bytes: usize) {
        // Cleanup alone recovers poison; admission never clears or recovers it.
        // This may wait only for another short arithmetic critical section.
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match state.held.checked_sub(bytes) {
            Some(held) => state.held = held,
            None => state.failed_closed = true,
        }
    }
}

/// Non-Clone aggregate. Its owning field must follow every covered buffer, and
/// construction locals must precede raw locals, so release follows deallocation.
pub(super) struct RawCharge {
    ledger: Arc<RawLedger>,
    bytes: usize,
}

impl RawCharge {
    /// Used by private raw owners and opt-in BPE merge scratch. This checks the
    /// requested layout before acquiring its charge and allocating the backing.
    /// The returned Vec must stay private, never grow, and die before this charge.
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<Vec<u8>, GgufRawSourceError> {
        let layout = Layout::array::<u8>(bytes).map_err(|_| GgufRawSourceError::Layout { bytes })?;
        if bytes == 0 {
            return Ok(Vec::new());
        }
        self.acquire(layout.size())?;
        // SAFETY: Layout::array validated the nonzero size and isize bound;
        // u8 alignment is 1. The stable global allocator is also Vec's allocator.
        // No lock is held. Null is checked without dereference or adoption.
        let pointer = unsafe { alloc(layout) };
        if pointer.is_null() {
            return Err(GgufRawSourceError::Allocation { bytes });
        }
        // SAFETY: Immediately and uniquely transfer this global allocation into
        // Vec; no fallible/panicking work intervenes. Alignment 1 and capacity
        // bytes reproduce exactly the allocated layout at deallocation. Length 0
        // exposes no uninitialized data; bounded safe appends initialize <=bytes
        // elements and cannot reallocate. Raw owners retain their Vec; BPE scratch
        // moves its Vec in a private owner that exposes only a mutable slice to
        // the provider. Each owner's charge outlives the backing, including moves,
        // errors and unwind. Vec/Mutex supply synchronization without unsafe impls.
        Ok(unsafe { Vec::from_raw_parts(pointer, 0, bytes) })
    }

    // Also used by private declaration/immutable semantic builders for checked
    // aggregate layouts and identity bytes on this same ledger. The public
    // raw-only route still charges raw backing only.
    pub(super) fn acquire(&mut self, requested: usize) -> Result<(), GgufRawSourceError> {
        let bytes = self.bytes.checked_add(requested).ok_or(GgufRawSourceError::Overflow)?;
        self.ledger.try_acquire(requested)?;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for RawCharge {
    fn drop(&mut self) {
        self.ledger.release(self.bytes);
    }
}
