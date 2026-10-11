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

/// Inspect GPU `device` ([`crate::gpu_device`]; Intel: the adapter the backend selects, device 0 only) before loading
/// weights. Creates a device context, without a model forward.
pub fn gpu_memory(device: usize) -> Result<GpuMemory> {
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    return crate::intel::memory(device);
    #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
    snapshot(&crate::gpu_device(device)?)
}

/// Counters of a CUDA device's stream-ordered allocator pool (bytes): the pool `cuMemAllocAsync` allocates from, so
/// every Candle tensor, resident q8_0 block, temporary f16 tensor and cuBLASLt workspace of this process on the
/// device. Not in it: the CUDA context, loaded modules, library handles and their internal workspaces, synchronous
/// allocations. `*_high` are watermarks since the last [`PoolProbe::reset_peaks`] (or since the pool was created).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PoolUsage {
    /// Bytes of live allocations (`CU_MEMPOOL_ATTR_USED_MEM_CURRENT`).
    pub used_bytes: u64,
    /// Bytes the pool holds from the driver, live or cached for reuse (`CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT`).
    pub reserved_bytes: u64,
    /// Highest `used_bytes` (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`).
    pub used_high_bytes: u64,
    /// Highest `reserved_bytes` (`CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH`).
    pub reserved_high_bytes: u64,
}

/// Handle on the allocator pool of one CUDA device, independent of any engine: it retains the device's primary
/// context (the one Candle devices of this process use), so the pool and its counters stay in place between engines.
/// Reading is cheap; [`PoolProbe::reset_peaks`] changes pool state and belongs outside measured intervals.
pub struct PoolProbe {
    #[cfg(feature = "cuda")]
    ctx: std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaContext>,
}

/// The allocator pool of CUDA device `device` ([`crate::gpu_device`] numbering), or `None` when this build has no CUDA
/// backend or the device has no memory pools (allocations are then synchronous and have no pool counters).
pub fn pool_probe(device: usize) -> Result<Option<PoolProbe>> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda_backend::cudarc::driver::CudaContext;
        let count = CudaContext::device_count()?;
        anyhow::ensure!(
            device < usize::try_from(count).unwrap_or(0),
            "CUDA device {device} does not exist: this process sees {count} CUDA device(s)"
        );
        // the same primary-context flags gpu_device sets, before the context may be created here
        crate::set_cuda_sync(device)?;
        let ctx = CudaContext::new(device)?;
        Ok(ctx.has_async_alloc().then_some(PoolProbe { ctx }))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
impl PoolProbe {
    fn pool(&self) -> Result<candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool> {
        use candle_core::cuda_backend::cudarc::driver::result;
        self.ctx.bind_to_thread()?;
        // SAFETY: `cu_device` is the CUdevice the retained context was created for (cuDeviceGet of an ordinal below
        // the device count), as cuDeviceGetMemPool requires. The returned handle is the device's current pool (the
        // default pool: this process never sets another); it is not owned here, never destroyed, and stays valid
        // while the driver is initialised.
        Ok(unsafe { result::device::get_mem_pool(self.ctx.cu_device()) }?)
    }

    /// The pool's current and high-watermark usage. Pending stream-ordered frees count as used until the GPU
    /// executes them; synchronise the context first ([`PoolProbe::synchronize`]) when that matters.
    pub fn usage(&self) -> Result<PoolUsage> {
        use candle_core::cuda_backend::cudarc::driver::{result, sys::CUmemPool_attribute as A};
        let pool = self.pool()?;
        let read = |attribute: A| -> Result<u64> {
            let mut value: u64 = 0;
            // SAFETY: `pool` is a valid pool handle (above). The four attributes read here are `cuuint64_t` values,
            // so the driver writes exactly one u64 through the pointer, which points to `value`, alive and not
            // otherwise borrowed for the duration of the call.
            unsafe { result::mem_pool::get_attribute(pool, attribute, (&raw mut value).cast()) }?;
            Ok(value)
        };
        Ok(PoolUsage {
            used_bytes: read(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            reserved_bytes: read(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            used_high_bytes: read(A::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            reserved_high_bytes: read(A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        })
    }

    /// Reset the high watermarks (the driver accepts only 0 for them); later readings report the highest usage
    /// since the reset. Changes the pool's state for every user of the device; call it outside measured intervals.
    pub fn reset_peaks(&self) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::{result, sys::CUmemPool_attribute as A};
        let pool = self.pool()?;
        for attribute in [A::CU_MEMPOOL_ATTR_USED_MEM_HIGH, A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH] {
            let mut zero: u64 = 0;
            // SAFETY: `pool` is a valid pool handle (above). Both watermark attributes take a `cuuint64_t`, and 0 is
            // the only value the driver accepts for them; the pointer refers to `zero`, alive during the call and
            // only read by the driver.
            unsafe { result::mem_pool::set_attribute(pool, attribute, (&raw mut zero).cast()) }?;
        }
        Ok(())
    }

    /// Wait for all work of the device context, on every thread's stream (so stream-ordered frees have executed).
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.ctx.synchronize()?)
    }
}

#[cfg(not(feature = "cuda"))]
impl PoolProbe {
    /// The pool's current and high-watermark usage (CUDA only).
    pub fn usage(&self) -> Result<PoolUsage> {
        anyhow::bail!("memory pools need the CUDA backend")
    }
    /// Reset the high watermarks (CUDA only).
    pub fn reset_peaks(&self) -> Result<()> {
        anyhow::bail!("memory pools need the CUDA backend")
    }
    /// Wait for the device (CUDA only).
    pub fn synchronize(&self) -> Result<()> {
        anyhow::bail!("memory pools need the CUDA backend")
    }
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
