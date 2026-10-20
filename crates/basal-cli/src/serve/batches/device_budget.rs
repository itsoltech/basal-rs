//! Shared admission of private background memory on physical devices.
//!
//! Limits are supplied by the orchestrator from a qualified profile, after accounting for resident weights,
//! allocator retention and the online reserve. Free-memory telemetry alone must never create or increase a limit.
//! The same instance serves every model and replica in the process. A split acquires all reservations or none;
//! callers never wait with a partial reservation or while holding a device gate.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use basal_core::{StopReason, StopToken};

#[derive(Clone, Copy, Debug)]
pub(in crate::serve) struct DeviceLimit {
    pub device: usize,
    /// Qualified allowance for private background allocations, excluding the online reserve and resident data.
    pub background_bytes: u64,
    /// Qualified worst allocation granularity; each source allocation may waste up to granularity - 1 bytes.
    pub allocation_granularity: u64,
    pub online_reserve_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::serve) struct DeviceRequest {
    pub device: usize,
    pub bytes: u64,
    pub allocations: u64,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::serve) struct DeviceSnapshot {
    pub device: usize,
    pub limit: u64,
    pub reserved: u64,
    pub allocation_granularity: u64,
    pub online_reserve_bytes: u64,
    pub quarantined: bool,
}

#[derive(Debug, thiserror::Error)]
pub(in crate::serve) enum DeviceBudgetError {
    #[error("device memory limits must be positive and unique")]
    InvalidLimits,
    #[error("device memory reservation is empty, zero or overflows")]
    InvalidRequest,
    #[error("device {0} has no qualified background memory limit")]
    UnknownDevice(usize),
    #[error("device {0} is quarantined after an unconfirmed cleanup")]
    Quarantined(usize),
    #[error("work exceeds the qualified background memory allowance on device {0}")]
    ExceedsLimit(usize),
    #[error("device memory admission is unavailable after a synchronization failure")]
    Unavailable,
    #[error("device memory reservation sequence exhausted")]
    Exhausted,
    #[error("device {0} memory telemetry is unavailable")]
    Telemetry(usize),
}

struct Device {
    limit: u64,
    allocation_granularity: u64,
    online_reserve_bytes: u64,
    reserved: u64,
    quarantined: bool,
}

struct Active {
    requests: Vec<DeviceRequest>,
    stop: StopToken,
}

struct State {
    devices: BTreeMap<usize, Device>,
    active: BTreeMap<u64, Active>,
    next: u64,
}

#[derive(Clone)]
pub(in crate::serve) struct DeviceBudget(Arc<Mutex<State>>);

impl DeviceBudget {
    /// Advisory check before claiming and decoding another record. The all-or-none reservation still rechecks
    /// this under the same lock immediately before execution; a race can cost one replan, never GPU overlap.
    pub fn ready(&self, devices: &[usize]) -> Result<bool, DeviceBudgetError> {
        if devices.is_empty() {
            return Err(DeviceBudgetError::InvalidRequest);
        }
        let state = self.0.lock().map_err(|_| DeviceBudgetError::Unavailable)?;
        for id in devices {
            let device = state.devices.get(id).ok_or(DeviceBudgetError::UnknownDevice(*id))?;
            if device.quarantined {
                return Err(DeviceBudgetError::Quarantined(*id));
            }
            if device.reserved != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// An empty set deliberately admits no background work. Unknown profiles are never assigned guessed limits.
    pub fn new(limits: &[DeviceLimit]) -> Result<Self, DeviceBudgetError> {
        let mut devices = BTreeMap::new();
        for limit in limits {
            if limit.background_bytes == 0
                || limit.allocation_granularity == 0
                || limit.online_reserve_bytes == 0
                || devices
                    .insert(
                        limit.device,
                        Device {
                            limit: limit.background_bytes,
                            allocation_granularity: limit.allocation_granularity,
                            online_reserve_bytes: limit.online_reserve_bytes,
                            reserved: 0,
                            quarantined: false,
                        },
                    )
                    .is_some()
            {
                return Err(DeviceBudgetError::InvalidLimits);
            }
        }
        Ok(Self(Arc::new(Mutex::new(State { devices, active: BTreeMap::new(), next: 0 }))))
    }

    /// `None` means temporary memory pressure. Unknown/unqualified devices are explicit configuration errors.
    /// Reservation owns its stop signal and remains charged through pause, result delivery and cleanup.
    pub fn try_reserve(
        &self,
        requests: &[DeviceRequest],
        stop: StopToken,
    ) -> Result<Option<DevicePermit>, DeviceBudgetError> {
        let mut requests = normalize(requests)?;
        let mut state = self.0.lock().map_err(|_| DeviceBudgetError::Unavailable)?;
        // A paused forward still owns device state. Do not start a second background forward on any GPU it uses.
        if state.active.values().any(|active| {
            active.requests.iter().any(|held| requests.iter().any(|request| held.device == request.device))
        }) {
            return Ok(None);
        }
        let mut headroom = Vec::with_capacity(requests.len());
        for request in &mut requests {
            let device = state.devices.get(&request.device).ok_or(DeviceBudgetError::UnknownDevice(request.device))?;
            if device.quarantined {
                return Err(DeviceBudgetError::Quarantined(request.device));
            }
            let overhead = request
                .allocations
                .checked_mul(device.allocation_granularity - 1)
                .ok_or(DeviceBudgetError::InvalidRequest)?;
            request.bytes = request.bytes.checked_add(overhead).ok_or(DeviceBudgetError::InvalidRequest)?;
            if request.bytes > device.limit {
                return Err(DeviceBudgetError::ExceedsLimit(request.device));
            }
            if request.bytes > device.limit.saturating_sub(device.reserved) {
                return Ok(None);
            }
            headroom.push((
                request.device,
                request.bytes.checked_add(device.online_reserve_bytes).ok_or(DeviceBudgetError::InvalidRequest)?,
            ));
        }
        let id = state.next.checked_add(1).ok_or(DeviceBudgetError::Exhausted)?;
        // All arithmetic and devices have been checked; this mutation cannot leave a partial admission.
        for request in &requests {
            if let Some(device) = state.devices.get_mut(&request.device) {
                device.reserved += request.bytes;
            }
        }
        state.next = id;
        state.active.insert(id, Active { requests, stop });
        let permit = DevicePermit { budget: self.clone(), id, quarantine: false };
        drop(state);
        // Cached allocator blocks can outlive a previous model's reservation. A logical release cannot create
        // physical headroom. Refuse conservatively rather than counting another allocator's cache as reusable.
        // These read-only driver queries happen on the owning GPU thread, outside the accounting mutex.
        for (device, required) in headroom {
            let memory = basal_gpu::gpu_memory(device).map_err(|_| DeviceBudgetError::Telemetry(device))?;
            if memory.available_bytes < required {
                return Ok(None);
            }
        }
        Ok(Some(permit))
    }

    /// Ask work retaining memory on these devices to stop at a safe point. Signal outside the accounting lock:
    /// stop wakes device-gate waiters and must not invert the two locks. This does not release any reservation.
    pub fn request_release(&self, devices: &[usize]) -> Result<usize, DeviceBudgetError> {
        let targets: BTreeSet<_> = devices.iter().copied().collect();
        let stops: Vec<_> = {
            let state = self.0.lock().map_err(|_| DeviceBudgetError::Unavailable)?;
            state
                .active
                .values()
                .filter(|active| active.requests.iter().any(|request| targets.contains(&request.device)))
                .map(|active| active.stop.clone())
                .collect()
        };
        for stop in &stops {
            stop.stop(StopReason::MemoryPressure);
        }
        Ok(stops.len())
    }

    pub fn snapshots(&self) -> Result<Vec<DeviceSnapshot>, DeviceBudgetError> {
        let state = self.0.lock().map_err(|_| DeviceBudgetError::Unavailable)?;
        Ok(state
            .devices
            .iter()
            .map(|(&id, device)| DeviceSnapshot {
                device: id,
                limit: device.limit,
                reserved: device.reserved,
                allocation_granularity: device.allocation_granularity,
                online_reserve_bytes: device.online_reserve_bytes,
                quarantined: device.quarantined,
            })
            .collect())
    }
}

fn normalize(requests: &[DeviceRequest]) -> Result<Vec<DeviceRequest>, DeviceBudgetError> {
    if requests.is_empty() {
        return Err(DeviceBudgetError::InvalidRequest);
    }
    let mut sums = BTreeMap::<usize, (u64, u64)>::new();
    for request in requests {
        if request.bytes == 0 || request.allocations == 0 {
            return Err(DeviceBudgetError::InvalidRequest);
        }
        let total = sums.entry(request.device).or_default();
        total.0 = total.0.checked_add(request.bytes).ok_or(DeviceBudgetError::InvalidRequest)?;
        total.1 = total.1.checked_add(request.allocations).ok_or(DeviceBudgetError::InvalidRequest)?;
    }
    Ok(sums.into_iter().map(|(device, (bytes, allocations))| DeviceRequest { device, bytes, allocations }).collect())
}

/// Keep this alive until the worker confirms completion of its GPU work and releases private allocations.
pub(in crate::serve) struct DevicePermit {
    budget: DeviceBudget,
    id: u64,
    quarantine: bool,
}

impl DevicePermit {
    /// Cleanup could not confirm that the GPU stopped using the allocations. Retain the charge and fence admission
    /// on every device of the work; neither a timeout nor dropping the reply makes that memory reusable.
    pub fn quarantine(mut self) {
        self.quarantine = true;
    }
}

impl Drop for DevicePermit {
    fn drop(&mut self) {
        let Ok(mut state) = self.budget.0.lock() else { return };
        let Some(active) = state.active.remove(&self.id) else { return };
        for request in active.requests {
            if let Some(device) = state.devices.get_mut(&request.device) {
                if self.quarantine {
                    device.quarantined = true;
                } else if let Some(remaining) = device.reserved.checked_sub(request.bytes) {
                    device.reserved = remaining;
                } else {
                    // A corrupt accounting invariant must close admission, never undercount live memory.
                    device.quarantined = true;
                }
            }
        }
    }
}
