//! Resource snapshots of the actual device selected by candle, including CUDA_VISIBLE_DEVICES.

use anyhow::Result;
use candle_core::Device;
use serde::Serialize;

/// A checkpoint between forwards, not the peak allocation inside a forward.
#[derive(Clone, Debug, Serialize)]
pub struct GpuMemory {
    /// Hardware name, without a host name or device UUID.
    pub name: String,
    /// VRAM on CUDA; recommended process working set on Metal.
    pub budget_bytes: u64,
    /// Free VRAM on CUDA; recommended working set minus process allocations on Metal.
    pub available_bytes: u64,
    /// Metal's current process allocations; CUDA reports device-wide free memory instead.
    pub process_allocated_bytes: Option<u64>,
    /// Metal uses host RAM as well as its recommended working set limit.
    pub unified: bool,
}

/// Read-only telemetry handle retaining the engine's actual device, without retaining its weights.
pub struct GpuMemoryProbe(Device);

impl GpuMemoryProbe {
    /// Query a checkpoint without running a model forward.
    pub fn snapshot(&self) -> Result<GpuMemory> {
        snapshot(&self.0)
    }
}

fn snapshot(dev: &Device) -> Result<GpuMemory> {
    match dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(d) => {
            let stream = d.cuda_stream();
            let ctx = stream.context();
            let (free, total) = ctx.mem_get_info()?;
            Ok(GpuMemory {
                name: ctx.name()?,
                budget_bytes: total as u64,
                available_bytes: free as u64,
                process_allocated_bytes: None,
                unified: false,
            })
        }
        #[cfg(target_os = "macos")]
        Device::Metal(d) => {
            use objc2_metal::MTLDevice;
            let d = d.device();
            let budget = d.recommended_max_working_set_size() as u64;
            let allocated = d.current_allocated_size() as u64;
            Ok(GpuMemory {
                name: d.as_ref().name().to_string(),
                budget_bytes: budget,
                available_bytes: budget.saturating_sub(allocated),
                process_allocated_bytes: Some(allocated),
                unified: true,
            })
        }
        _ => anyhow::bail!("no GPU memory telemetry for this device"),
    }
}

/// Inspect the selected GPU before loading weights. Creates a device context, without a model forward.
pub fn gpu_memory() -> Result<GpuMemory> {
    snapshot(&crate::gpu_device()?)
}

impl crate::GpuBackend {
    /// Retain a read-only device handle for monitoring while this engine serves requests on another thread.
    pub fn memory_probe(&self) -> GpuMemoryProbe {
        GpuMemoryProbe(self.dev.clone())
    }
    /// Inspect this engine's device outside the measured inference interval.
    pub fn memory(&self) -> Result<GpuMemory> {
        snapshot(&self.dev)
    }
}
