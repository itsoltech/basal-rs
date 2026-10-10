//! Driver-reported Vulkan heap budgets of the selected device, without inference or weight ownership.
//! wgpu has no safe API for `VK_EXT_memory_budget`, so this is the only Vulkan call outside wgpu.
use crate::GpuMemory;
use anyhow::{ensure, Context, Result};
use ash::vk;

pub(crate) struct Probe {
    /// Keeps the Vulkan instance, physical device and logical device alive for every query.
    device: wgpu::Device,
    name: String,
    unified: bool,
}

impl Probe {
    pub(super) fn new(device: &wgpu::Device, info: &wgpu::AdapterInfo) -> Self {
        Self {
            device: device.clone(),
            name: info.name.clone(),
            unified: info.device_type == wgpu::DeviceType::IntegratedGpu,
        }
    }

    pub(crate) fn snapshot(&self) -> Result<GpuMemory> {
        // SAFETY: `as_hal` requires that the returned device is not destroyed while wgpu may use it. Only
        // physical-device property queries below use the guard; nothing is created, destroyed or submitted,
        // and no raw handle outlives the guard, which itself keeps the device alive.
        let device = unsafe { self.device.as_hal::<wgpu::hal::api::Vulkan>() }.context("not a Vulkan device")?;
        ensure!(
            device.enabled_device_extensions().contains(&ash::ext::memory_budget::NAME),
            "Intel memory telemetry requires VK_EXT_memory_budget"
        );
        let instance = device.shared_instance();
        let raw = instance.raw_instance();
        let physical = device.raw_physical_device();
        // SAFETY: `physical` was enumerated from `raw`; the guard keeps both alive for this core 1.0 query.
        let version = unsafe { raw.get_physical_device_properties(physical) }.api_version;
        ensure!(
            instance.instance_api_version() >= vk::API_VERSION_1_1 && version >= vk::API_VERSION_1_1,
            "Intel memory telemetry requires Vulkan 1.1"
        );
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: as above, and the instance and device support this core 1.1 query. The enabled extension
        // makes the budget struct a valid pNext member; both output structs outlive the call. Vulkan permits
        // concurrent read-only physical-device queries, so no external synchronisation is needed.
        unsafe { raw.get_physical_device_memory_properties2(physical, &mut properties) };
        let memory = properties.memory_properties;
        let mut total = 0u64;
        let mut used = 0u64;
        for (i, heap) in memory.memory_heaps_as_slice().iter().enumerate() {
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                total = total.saturating_add(budget.heap_budget[i].min(heap.size));
                used = used.saturating_add(budget.heap_usage[i]);
            }
        }
        ensure!(total > 0, "Vulkan driver did not report a device-local heap budget");
        Ok(GpuMemory {
            name: self.name.clone(),
            budget_bytes: total,
            available_bytes: total.saturating_sub(used),
            process_allocated_bytes: Some(used),
            unified: self.unified,
        })
    }
}
