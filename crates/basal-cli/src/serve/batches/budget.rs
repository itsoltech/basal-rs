//! Execution pool `P` of host memory for background work: plans in lookahead and microseries, continuations and
//! result buffers. One semaphore permit is one byte, so a reservation is a `tokio::sync::OwnedSemaphorePermit`
//! that travels with the owned plans or results and releases on drop (also when a handler or task is cancelled or
//! panics). Planning slots `R` are separate (`planning.rs`); waiting for `P` never holds a lease.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// The pool; capacity is `batches.host.exec_pool_bytes` (validated ≤ `u32::MAX`).
#[derive(Debug)]
pub(crate) struct HostPool {
    semaphore: Arc<Semaphore>,
    capacity: u32,
}

impl HostPool {
    pub(crate) fn new(capacity: u64) -> Self {
        // The configuration bounds the pool; a larger value cannot be represented as one acquisition.
        let capacity = u32::try_from(capacity).unwrap_or(u32::MAX);
        Self { semaphore: Arc::new(Semaphore::new(capacity as usize)), capacity }
    }

    pub(crate) fn capacity(&self) -> u64 {
        u64::from(self.capacity)
    }

    pub(crate) fn free(&self) -> u64 {
        self.semaphore.available_permits() as u64
    }

    /// Every byte that is free now (possibly zero permits). Feeder: reserve, claim with `host_budget_free` equal to
    /// the permit size, then [`HostPool::shrink`] to what the claim needs.
    pub(crate) fn reserve_available(&self) -> OwnedSemaphorePermit {
        // The semaphore is never closed, so an acquisition fails only when another one took bytes after the free
        // count was read; the count is read again then.
        loop {
            let free = u32::try_from(self.semaphore.available_permits()).unwrap_or(self.capacity);
            if let Ok(permit) = self.semaphore.clone().try_acquire_many_owned(free) {
                return permit;
            }
        }
    }

    /// Wait (cancel-safe, FIFO) until `bytes` are reserved; `None` when `bytes` exceeds the capacity.
    pub(crate) async fn reserve(&self, bytes: u64) -> Option<OwnedSemaphorePermit> {
        let n = u32::try_from(bytes).ok().filter(|&n| n <= self.capacity)?;
        self.semaphore.clone().acquire_many_owned(n).await.ok()
    }

    /// Return the part of `permit` above `bytes` to the pool.
    pub(crate) fn shrink(permit: &mut OwnedSemaphorePermit, bytes: u64) {
        let keep = usize::try_from(bytes).unwrap_or(usize::MAX);
        let excess = permit.num_permits().saturating_sub(keep);
        if excess > 0 {
            drop(permit.split(excess));
        }
    }
}
