//! Resource snapshots of the actual selected device, including CUDA_VISIBLE_DEVICES and Intel Vulkan heaps.

use anyhow::Result;
use candle_core::Device;
use serde::Serialize;

/// A checkpoint between forwards, not the peak allocation inside a forward.
#[derive(Clone, Debug, Serialize)]
pub struct GpuMemory {
    /// Hardware name, without a host name or device UUID.
    pub name: String,
    /// VRAM on CUDA; recommended process working set on Metal; driver heap budget on Vulkan.
    pub budget_bytes: u64,
    /// Free VRAM on CUDA; working-set/heap budget minus process usage on Metal/Vulkan.
    pub available_bytes: u64,
    /// Metal process allocations or Vulkan estimated heap usage; CUDA reports device-wide free memory.
    pub process_allocated_bytes: Option<u64>,
    /// The GPU shares host RAM (Metal or an integrated Intel device).
    pub unified: bool,
}

/// Read-only telemetry handle retaining the engine's actual device, without retaining its weights.
pub struct GpuMemoryProbe(Probe);

enum Probe {
    Candle(Device),
    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    Intel(crate::intel::telemetry::Probe),
}

impl GpuMemoryProbe {
    #[cfg(all(feature = "intel", not(target_os = "macos")))]
    pub(crate) fn intel(probe: crate::intel::telemetry::Probe) -> Self {
        Self(Probe::Intel(probe))
    }
    /// Query a checkpoint without running a model forward.
    pub fn snapshot(&self) -> Result<GpuMemory> {
        match &self.0 {
            Probe::Candle(dev) => snapshot(dev),
            #[cfg(all(feature = "intel", not(target_os = "macos")))]
            Probe::Intel(probe) => probe.snapshot(),
        }
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
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    return crate::intel::memory();
    #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
    snapshot(&crate::gpu_device()?)
}

impl crate::GpuBackend {
    /// Retain a read-only device handle for monitoring while this engine serves requests on another thread.
    pub fn memory_probe(&self) -> GpuMemoryProbe {
        GpuMemoryProbe(Probe::Candle(self.dev.clone()))
    }
    /// Inspect this engine's device outside the measured inference interval.
    pub fn memory(&self) -> Result<GpuMemory> {
        snapshot(&self.dev)
    }
}
