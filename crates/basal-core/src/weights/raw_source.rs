//! Held admission for exact requested layouts of private raw source buffers.

use std::alloc::{alloc, Layout};
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

/// Shared allowance for original raw source-content requested layouts only.
///
/// Create one handle before loads and pass it (or its sharing clones) to every
/// concurrent load/reload for that allowance. Independent handles have independent
/// limits. CLI/Engine do not provide this application context yet.
///
/// Excludes bootstrap/control blocks, objects, paths, names, records, errors,
/// serde/native/tokenizer/cache heaps, GGUF scratch, staging and device memory.
/// Allocator rounding, OS/RSS and physical OOM protection are also excluded.
#[derive(Clone)]
pub struct GgufRawSourceBudget {
    ledger: Arc<Mutex<RawLedgerState>>,
}

impl GgufRawSourceBudget {
    /// Create an independent allowance. Zero admits no nonempty raw buffer;
    /// equality with the limit is accepted. Bootstrap allocation is not fallible.
    pub fn new(limit_bytes: usize) -> Self {
        Self { ledger: Arc::new(Mutex::new(RawLedgerState { limit: limit_bytes, held: 0, failed_closed: false })) }
    }

    pub(super) fn charge(&self) -> RawCharge {
        RawCharge { ledger: Arc::clone(&self.ledger), bytes: 0 }
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

struct RawLedgerState {
    limit: usize,
    held: usize,
    failed_closed: bool,
}

/// Non-Clone aggregate. Its owning field must follow every covered buffer, and
/// construction locals must precede raw locals, so release follows deallocation.
pub(super) struct RawCharge {
    ledger: Arc<Mutex<RawLedgerState>>,
    bytes: usize,
}

impl RawCharge {
    /// Only the three private raw owners and the opt-in BPE merge scratch call
    /// this, after their length checks.
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
        // elements and cannot reallocate. No pointer or owned Vec escapes the
        // private raw owners; their charge outlives the backing, including moves,
        // errors and unwind. Vec/Mutex supply synchronization without unsafe impls.
        Ok(unsafe { Vec::from_raw_parts(pointer, 0, bytes) })
    }

    // Also used by private declaration/immutable semantic builders for checked
    // aggregate layouts and identity bytes on this same ledger. The public
    // raw-only route still charges raw backing only.
    pub(super) fn acquire(&mut self, requested: usize) -> Result<(), GgufRawSourceError> {
        let bytes = self.bytes.checked_add(requested).ok_or(GgufRawSourceError::Overflow)?;
        let mut state = match self.ledger.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return Err(GgufRawSourceError::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(GgufRawSourceError::Poisoned),
        };
        if state.failed_closed {
            return Err(GgufRawSourceError::Poisoned);
        }
        let held = state.held.checked_add(requested).ok_or(GgufRawSourceError::Overflow)?;
        if held > state.limit {
            return Err(GgufRawSourceError::Limit { requested, held: state.held, limit: state.limit });
        }
        // All-or-none arithmetic only: held equals the sum of outstanding charges.
        // Charge may conservatively exceed live layouts after a failed allocation.
        state.held = held;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for RawCharge {
    fn drop(&mut self) {
        // Cleanup alone recovers poison; admission never clears or recovers it.
        // This may wait only for another short arithmetic critical section.
        let mut state = self.ledger.lock().unwrap_or_else(PoisonError::into_inner);
        match state.held.checked_sub(self.bytes) {
            Some(held) => state.held = held,
            None => state.failed_closed = true,
        }
    }
}
