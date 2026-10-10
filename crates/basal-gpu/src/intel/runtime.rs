//! Vulkan resources. A backend owns its command encoders; immutable buffers and pipelines may be shared.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, ensure, Context, Result};

pub(super) struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    pub direct_upload: bool,
    lost: Arc<AtomicBool>,
    pipelines: BTreeMap<&'static str, wgpu::ComputePipeline>,
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        let adapter = adapters
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                i.vendor == 0x8086
                    && matches!(i.device_type, wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu)
            })
            .context("no Intel Vulkan GPU; install a Vulkan loader and the Intel Mesa driver (Arch: vulkan-intel)")?;
        let info = adapter.get_info();
        let mut features = wgpu::Features::SHADER_F16 | wgpu::Features::SUBGROUP;
        ensure!(
            adapter.features().contains(features),
            "{} requires Vulkan shaderFloat16 and compute subgroups",
            info.name
        );
        ensure!(
            info.subgroup_max_size <= 64 && info.subgroup_min_size >= 8,
            "unsupported Intel subgroup size {}",
            info.subgroup_max_size
        );
        // On a unified-memory GPU, map the final allocation instead of retaining a second staging copy.
        // Discrete cards keep device-local weights and use bounded staging uploads during model loading.
        let direct_upload = info.device_type == wgpu::DeviceType::IntegratedGpu
            && adapter.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        if direct_upload {
            features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("basal-intel"),
            required_features: features,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .context("creating Intel Vulkan device")?;
        let lost = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&lost);
        device.set_device_lost_callback(move |_, _| flag.store(true, Ordering::Release));
        Ok(Self { device, queue, info, limits, direct_upload, lost, pipelines: BTreeMap::new() })
    }

    /// Catch validation, allocation and driver errors instead of wgpu's default panic handler.
    /// Scopes are thread-local in wgpu 30; a fork on another worker cannot consume these errors.
    pub fn checked<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        ensure!(!self.lost.load(Ordering::Acquire), "Intel Vulkan device lost; restart the model");
        let scopes = [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal]
            .map(|filter| self.device.push_error_scope(filter));
        let result = f();
        let mut failure = None;
        for scope in scopes.into_iter().rev() {
            if let Some(error) = pollster::block_on(scope.pop()) {
                failure = Some(error);
            }
        }
        if let Some(error) = failure {
            return Err(error.into());
        }
        result
    }

    pub fn compile(&mut self, name: &'static str, source: &str) -> Result<()> {
        let pipeline = self
            .checked(|| {
                let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
                Ok(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }))
            })
            .with_context(|| format!("compiling Intel kernel {name}"))?;
        self.pipelines.insert(name, pipeline);
        Ok(())
    }

    pub fn upload<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> Result<wgpu::Buffer> {
        let bytes = bytemuck::cast_slice(data);
        self.check_size(bytes.len() as u64)?;
        self.initialized(label, bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC)
    }

    fn initialized(&self, label: &str, bytes: &[u8], usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        let usage = if self.direct_upload { usage | wgpu::BufferUsages::MAP_WRITE } else { usage };
        // create_buffer_init maps before the caller can inspect an OOM error and panics on an invalid buffer.
        // Finish the allocation error scope first, then use the fallible mapped-range API.
        let buffer = self
            .checked(|| {
                Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (bytes.len() as u64).next_multiple_of(4),
                    usage,
                    mapped_at_creation: true,
                }))
            })
            .with_context(|| format!("allocating Intel buffer {label} ({} bytes)", bytes.len()))?;
        {
            let mut mapped = buffer.get_mapped_range_mut(..).context("mapping Intel upload")?;
            mapped.slice(..bytes.len()).copy_from_slice(bytes);
        }
        self.checked(|| {
            buffer.unmap();
            Ok(())
        })?;
        Ok(buffer)
    }

    pub fn buffer(&self, label: &str, bytes: usize) -> Result<wgpu::Buffer> {
        self.check_size(bytes as u64)?;
        self.checked(|| {
            Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes as u64).next_multiple_of(4),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        })
    }

    fn check_size(&self, bytes: u64) -> Result<()> {
        ensure!(
            bytes > 0 && bytes <= self.limits.max_storage_buffer_binding_size && bytes <= self.limits.max_buffer_size,
            "Intel buffer size {bytes} exceeds Vulkan storage/buffer limits ({} / {})",
            self.limits.max_storage_buffer_binding_size,
            self.limits.max_buffer_size
        );
        Ok(())
    }

    pub fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        buffers: &[&wgpu::Buffer],
        params: [u32; 8],
        groups: [u32; 3],
    ) -> Result<()> {
        ensure!(
            groups.iter().all(|&n| n > 0 && n <= self.limits.max_compute_workgroups_per_dimension),
            "{name}: dispatch {groups:?} exceeds device limits"
        );
        let pipeline = self.pipelines.get(name).with_context(|| format!("missing kernel {name}"))?;
        let uniform = self.initialized(name, bytemuck::cast_slice(&params), wgpu::BufferUsages::UNIFORM)?;
        let entries: Vec<_> = buffers
            .iter()
            .copied()
            .chain(std::iter::once(&uniform))
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry { binding: i as u32, resource: buffer.as_entire_binding() })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes: None });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        Ok(())
    }

    pub fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit([encoder.finish()]);
    }

    pub fn flush_uploads(&self) -> Result<()> {
        self.checked(|| {
            self.queue.submit([]);
            self.sync()
        })
    }

    pub fn sync(&self) -> Result<()> {
        ensure!(!self.lost.load(Ordering::Acquire), "Intel Vulkan device lost; restart the model");
        // wgpu 30 routes DeviceLost from poll to a fatal handler instead of returning PollError.
        // Its loss callback marks this device unusable before the fatal-handler panic, which occurs
        // after core locks have been released. The callback itself only sets an atomic flag.
        // Only translate that pinned-version fatal-handler payload; unrelated panics keep unwinding
        // even if a concurrent fork lost the device. No partially completed GPU work is reused.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(60)) })
        })) {
            Ok(result) => result.context("waiting for Intel Vulkan GPU")?,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied());
                if self.lost.load(Ordering::Acquire)
                    && message
                        .is_some_and(|s| s.contains("Error in Device::poll:") && s.contains("Parent device is lost"))
                {
                    bail!("Intel Vulkan device lost while waiting; restart the model");
                }
                std::panic::resume_unwind(payload);
            }
        };
        ensure!(!self.lost.load(Ordering::Acquire), "Intel Vulkan device lost; restart the model");
        Ok(())
    }

    pub fn read(&self, buffer: &wgpu::Buffer, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes > 0 && bytes as u64 <= buffer.size() && bytes.is_multiple_of(4), "invalid Vulkan readback size");
        self.checked(|| {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = self.encoder();
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes as u64);
            self.submit(encoder);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            staging.map_async(wgpu::MapMode::Read, .., move |r| {
                let _ = tx.send(r);
            });
            self.sync()?;
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(r) => r.context("mapping Vulkan readback")?,
                Err(e) => bail!("Vulkan readback callback: {e}"),
            }
            let bytes = staging.get_mapped_range(..)?.to_vec();
            staging.unmap();
            Ok(bytes)
        })
    }
}
