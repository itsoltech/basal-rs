//! Hardware readings for the UI's server view, taken once a second on the stats sampler thread, never on a request
//! path. GPUs: on CUDA from NVML (`libnvidia-ml.so.1`, part of the driver: utilisation, memory, power, temperature,
//! SM clock, this process's memory), found by PCI address so the server's device ordinals match; without NVML, and on
//! Metal and Vulkan, the memory the backend's own probe reports ([`basal_gpu::GpuMemoryProbe`]). Host: Linux `/proc`
//! and POSIX calls; what a platform does not report stays `None`.

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use basal_gpu::GpuMemoryProbe;
use serde::Serialize;

use super::stats::{Stats, Telemetry};
use super::LoadCounts;

/// One GPU of the server. `None`: not reported by its source.
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct GpuSample {
    /// Device ordinal in the server's numbering (`device`, `replica_devices`, `layer_split`).
    pub(super) device: usize,
    /// `cuda:0`, `metal:0`, `vulkan:0`.
    pub(super) label: String,
    pub(super) name: Option<String>,
    /// `nvml`, or the backend's memory probe: `cuda`, `metal`, `vulkan`.
    pub(super) source: &'static str,
    pub(super) utilization_pct: Option<u32>,
    /// Share of time the memory bus was busy.
    pub(super) memory_bus_pct: Option<u32>,
    /// VRAM; Metal: the recommended working set; Vulkan: the driver's heap budget.
    pub(super) memory_total_bytes: Option<u64>,
    /// Used by all processes (Metal and Vulkan: by this process).
    pub(super) memory_used_bytes: Option<u64>,
    /// Used by this server process.
    pub(super) process_memory_bytes: Option<u64>,
    pub(super) power_w: Option<f64>,
    pub(super) power_limit_w: Option<f64>,
    pub(super) temperature_c: Option<u32>,
    pub(super) sm_clock_mhz: Option<u32>,
    /// The GPU shares host RAM.
    pub(super) unified: bool,
    /// Why nothing could be read.
    pub(super) error: Option<String>,
}

/// The host and this process.
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct HostSample {
    /// All CPUs busy since the previous reading, percent of the whole machine.
    pub(super) cpu_pct: Option<f64>,
    pub(super) cpus: Option<usize>,
    pub(super) memory_total_bytes: Option<u64>,
    /// Total minus available memory.
    pub(super) memory_used_bytes: Option<u64>,
    pub(super) load: Option<[f64; 3]>,
    pub(super) process_rss_bytes: Option<u64>,
    /// CPU time of this process since the previous reading, percent of one CPU.
    pub(super) process_cpu_pct: Option<f64>,
    pub(super) process_threads: Option<u64>,
}

/// Backend name of the device labels of this build.
const BACKEND: &str = if cfg!(feature = "cuda") {
    "cuda"
} else if cfg!(target_os = "macos") {
    "metal"
} else if cfg!(feature = "intel") {
    "vulkan"
} else {
    "gpu"
};

struct Gpu {
    device: usize,
    probe: GpuMemoryProbe,
    #[cfg(feature = "cuda")]
    nvml: Option<nvml::Device>,
}

/// Start the sampler: every second a hardware reading and a history point, until `stop` is dropped.
pub(super) fn start(
    stats: Arc<Stats>,
    load: Arc<Mutex<LoadCounts>>,
    probes: Vec<(usize, GpuMemoryProbe)>,
    stop: Receiver<()>,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    std::thread::Builder::new().name("ui-stats".into()).spawn(move || {
        let mut sampler = Sampler::new(probes);
        loop {
            let started = Instant::now();
            let mut telemetry = sampler.sample();
            telemetry.load = {
                let c = load.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                (c.inflight - c.queued, c.queued)
            };
            stats.tick(telemetry);
            match stop.recv_timeout(Duration::from_secs(1).saturating_sub(started.elapsed())) {
                Err(RecvTimeoutError::Timeout) => {}
                Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
            }
        }
    })
}

struct Sampler {
    gpus: Vec<Gpu>,
    host: Host,
}

impl Sampler {
    fn new(mut probes: Vec<(usize, GpuMemoryProbe)>) -> Self {
        probes.sort_by_key(|(device, _)| *device);
        probes.dedup_by_key(|(device, _)| *device);
        #[cfg(feature = "cuda")]
        let api = nvml::Api::load()
            .inspect_err(|e| crate::warn!("ui stats: NVML unavailable, GPU readings limited to memory: {e:#}"))
            .ok();
        let gpus = probes
            .into_iter()
            .map(|(device, probe)| Gpu {
                device,
                #[cfg(feature = "cuda")]
                nvml: api.and_then(|api| {
                    basal_gpu::pci_bus_id(device)
                        .and_then(|id| api.device(&id.unwrap_or_default()))
                        .inspect_err(|e| crate::warn!("ui stats: NVML has no GPU {device}: {e:#}"))
                        .ok()
                }),
                probe,
            })
            .collect();
        Self { gpus, host: Host::default() }
    }

    fn sample(&mut self) -> Telemetry {
        let gpus = self.gpus.iter().map(Gpu::sample).collect();
        Telemetry { at: super::stats::unix_ms(SystemTime::now()), load: (0, 0), gpus, host: self.host.sample() }
    }
}

impl Gpu {
    fn sample(&self) -> GpuSample {
        let mut s =
            GpuSample { device: self.device, label: format!("{BACKEND}:{}", self.device), ..GpuSample::default() };
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.nvml {
            device.sample(&mut s);
            return s;
        }
        match self.probe.snapshot() {
            Ok(m) => {
                s.source = BACKEND;
                s.name = Some(m.name);
                s.memory_total_bytes = Some(m.budget_bytes);
                s.memory_used_bytes = Some(m.budget_bytes.saturating_sub(m.available_bytes));
                s.process_memory_bytes = m.process_allocated_bytes;
                s.unified = m.unified;
            }
            Err(e) => s.error = Some(format!("{e:#}")),
        }
        s
    }
}

/// CPU counters of the previous reading.
#[derive(Default)]
struct Host {
    /// Busy and total jiffies of all CPUs.
    #[cfg(target_os = "linux")]
    cpu: Option<(u64, u64)>,
    /// CPU time of this process and when it was read.
    process: Option<(Duration, Instant)>,
}

impl Host {
    fn sample(&mut self) -> HostSample {
        let mut s = HostSample {
            cpus: std::thread::available_parallelism().ok().map(std::num::NonZeroUsize::get),
            load: load_average(),
            ..HostSample::default()
        };
        if let Some(cpu) = process_cpu_time() {
            let now = Instant::now();
            if let Some((before, at)) = self.process.replace((cpu, now)) {
                let wall = now.duration_since(at).as_secs_f64();
                s.process_cpu_pct = (wall > 0.0).then(|| cpu.saturating_sub(before).as_secs_f64() / wall * 100.0);
            }
        }
        #[cfg(target_os = "linux")]
        self.linux(&mut s);
        #[cfg(target_os = "macos")]
        {
            s.memory_total_bytes = macos_memory_bytes();
        }
        s
    }

    #[cfg(target_os = "linux")]
    fn linux(&mut self, s: &mut HostSample) {
        if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
            // cpu user nice system idle iowait irq softirq steal (guest time is already in user and nice)
            let fields: Vec<u64> = stat
                .lines()
                .next()
                .unwrap_or("")
                .split_whitespace()
                .skip(1)
                .take(8)
                .filter_map(|f| f.parse().ok())
                .collect();
            if fields.len() == 8 {
                let total: u64 = fields.iter().sum();
                let busy = total - fields[3] - fields[4];
                if let Some((busy0, total0)) = self.cpu.replace((busy, total)) {
                    let dt = total.saturating_sub(total0);
                    s.cpu_pct = (dt > 0).then(|| busy.saturating_sub(busy0) as f64 / dt as f64 * 100.0);
                }
            }
            s.cpus = Some(
                stat.lines()
                    .filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(u8::is_ascii_digit))
                    .count(),
            );
        }
        let kib = |text: &str, key: &str| {
            text.lines().find_map(|l| l.strip_prefix(key)?.split_whitespace().next()?.parse::<u64>().ok())
        };
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            let total = kib(&meminfo, "MemTotal:");
            s.memory_total_bytes = total.map(|k| k * 1024);
            s.memory_used_bytes = total.zip(kib(&meminfo, "MemAvailable:")).map(|(t, a)| t.saturating_sub(a) * 1024);
        }
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            s.process_rss_bytes = kib(&status, "VmRSS:").map(|k| k * 1024);
            s.process_threads = kib(&status, "Threads:");
        }
    }
}

fn load_average() -> Option<[f64; 3]> {
    let mut load = [0f64; 3];
    // SAFETY: `load` is a writable array of 3 doubles, the count passed; getloadavg writes at most that many.
    (unsafe { libc::getloadavg(load.as_mut_ptr(), 3) } == 3).then_some(load)
}

/// User and system CPU time of this process.
fn process_cpu_time() -> Option<Duration> {
    // SAFETY: an all-zero `rusage` is a valid value of this plain C struct (integers and timevals).
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: RUSAGE_SELF with a valid out-pointer to `usage`, alive and exclusively borrowed during the call.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return None;
    }
    let time = |t: libc::timeval| -> Option<Duration> {
        Some(Duration::from_secs(u64::try_from(t.tv_sec).ok()?) + Duration::from_micros(u64::try_from(t.tv_usec).ok()?))
    };
    Some(time(usage.ru_utime)? + time(usage.ru_stime)?)
}

#[cfg(target_os = "macos")]
fn macos_memory_bytes() -> Option<u64> {
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a NUL-terminated name of a 64-bit integer; `bytes` and `len` are valid out-pointers
    // and `len` holds the size of `bytes`. No new value is set (null pointer, length 0).
    let res = unsafe {
        libc::sysctlbyname(c"hw.memsize".as_ptr(), (&raw mut bytes).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    (res == 0 && len == std::mem::size_of::<u64>()).then_some(bytes)
}

/// The NVIDIA Management Library through its stable C ABI (`nvml.h`), loaded at run time like the CUDA driver in
/// `doctor`: no link-time dependency, and a server without it only loses these readings.
#[cfg(feature = "cuda")]
mod nvml {
    use std::ffi::{c_char, c_uint, c_ulonglong, c_void, CStr, CString};

    use anyhow::{Context, Result};

    type Ret = c_uint;
    type Handle = *mut c_void;
    const SUCCESS: Ret = 0;
    const TEMPERATURE_GPU: c_uint = 0;
    const CLOCK_SM: c_uint = 1;
    /// `NVML_VALUE_NOT_AVAILABLE` of `usedGpuMemory`.
    const NOT_AVAILABLE: c_ulonglong = c_ulonglong::MAX;
    const MAX_PROCESSES: usize = 64;

    #[repr(C)]
    #[derive(Default)]
    struct Utilization {
        gpu: c_uint,
        memory: c_uint,
    }

    #[repr(C)]
    #[derive(Default)]
    struct Memory {
        total: c_ulonglong,
        free: c_ulonglong,
        used: c_ulonglong,
    }

    /// `nvmlProcessInfo_t` of the `_v3` process query.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ProcessInfo {
        pid: c_uint,
        used_gpu_memory: c_ulonglong,
        gpu_instance_id: c_uint,
        compute_instance_id: c_uint,
    }

    /// The functions used here; the library stays loaded for the rest of the process.
    #[derive(Clone, Copy)]
    pub(super) struct Api {
        by_pci: unsafe extern "C" fn(*const c_char, *mut Handle) -> Ret,
        name: unsafe extern "C" fn(Handle, *mut c_char, c_uint) -> Ret,
        utilization: unsafe extern "C" fn(Handle, *mut Utilization) -> Ret,
        memory: unsafe extern "C" fn(Handle, *mut Memory) -> Ret,
        power: unsafe extern "C" fn(Handle, *mut c_uint) -> Ret,
        power_limit: unsafe extern "C" fn(Handle, *mut c_uint) -> Ret,
        temperature: unsafe extern "C" fn(Handle, c_uint, *mut c_uint) -> Ret,
        clock: unsafe extern "C" fn(Handle, c_uint, *mut c_uint) -> Ret,
        /// Missing before driver 510.
        processes: Option<unsafe extern "C" fn(Handle, *mut c_uint, *mut ProcessInfo) -> Ret>,
    }

    impl Api {
        pub(super) fn load() -> Result<Self> {
            // SAFETY: loads the NVIDIA driver's management library by its fixed name. The handle is leaked: NVML
            // keeps state for the rest of the process, so the library must stay mapped.
            let lib: &'static libloading::Library = Box::leak(Box::new(
                unsafe { libloading::Library::new("libnvidia-ml.so.1") }.context("loading libnvidia-ml.so.1")?,
            ));
            // SAFETY: the signatures of the NVML API (nvml.h, a stable C ABI; the `_v2`/`_v3` names are those nvml.h
            // maps the plain names to). The library stays loaded (above), so the functions stay valid.
            let api = unsafe {
                let init = *lib.get::<unsafe extern "C" fn() -> Ret>(b"nvmlInit_v2\0")?;
                let res = init();
                anyhow::ensure!(res == SUCCESS, "nvmlInit: NVML error {res}");
                Self {
                    by_pci: *lib.get(b"nvmlDeviceGetHandleByPciBusId_v2\0")?,
                    name: *lib.get(b"nvmlDeviceGetName\0")?,
                    utilization: *lib.get(b"nvmlDeviceGetUtilizationRates\0")?,
                    memory: *lib.get(b"nvmlDeviceGetMemoryInfo\0")?,
                    power: *lib.get(b"nvmlDeviceGetPowerUsage\0")?,
                    power_limit: *lib.get(b"nvmlDeviceGetEnforcedPowerLimit\0")?,
                    temperature: *lib.get(b"nvmlDeviceGetTemperature\0")?,
                    clock: *lib.get(b"nvmlDeviceGetClockInfo\0")?,
                    processes: lib.get(b"nvmlDeviceGetComputeRunningProcesses_v3\0").ok().map(|f| *f),
                }
            };
            Ok(api)
        }

        /// The GPU at PCI address `pci` (`domain:bus:device.function`).
        pub(super) fn device(self, pci: &str) -> Result<Device> {
            let id = CString::new(pci)?;
            let mut handle: Handle = std::ptr::null_mut();
            // SAFETY: NVML is initialised (`load`); `id` is NUL-terminated and alive during the call; `handle` is a
            // valid out-pointer.
            let res = unsafe { (self.by_pci)(id.as_ptr(), &mut handle) };
            anyhow::ensure!(res == SUCCESS, "nvmlDeviceGetHandleByPciBusId({pci}): NVML error {res}");
            let mut buf = [0u8; 96];
            // SAFETY: `handle` from NVML; `buf` is writable for the length passed and the name is read up to the first
            // NUL inside `buf`, so a missing terminator is harmless.
            let name = (unsafe { (self.name)(handle, buf.as_mut_ptr().cast(), 96) } == SUCCESS)
                .then(|| CStr::from_bytes_until_nul(&buf).ok().map(|s| s.to_string_lossy().into_owned()))
                .flatten();
            Ok(Device { api: self, handle, name })
        }
    }

    /// One GPU. NVML handles stay valid while the library is initialised (never shut down here).
    pub(super) struct Device {
        api: Api,
        handle: Handle,
        name: Option<String>,
    }

    impl Device {
        /// Read one `c_uint` query; `None` when not supported or failing.
        fn uint(&self, f: impl FnOnce(Handle, *mut c_uint) -> Ret) -> Option<u32> {
            let mut v: c_uint = 0;
            (f(self.handle, &mut v) == SUCCESS).then_some(v)
        }

        pub(super) fn sample(&self, s: &mut super::GpuSample) {
            let api = &self.api;
            s.source = "nvml";
            s.name.clone_from(&self.name);
            let mut u = Utilization::default();
            // SAFETY: `handle` is a device handle of the initialised library; the out-pointer refers to a live local
            // of the type nvml.h declares for it. The same holds for the reads below.
            let res = unsafe { (api.utilization)(self.handle, &mut u) };
            if res == SUCCESS {
                s.utilization_pct = Some(u.gpu);
                s.memory_bus_pct = Some(u.memory);
            }
            let mut m = Memory::default();
            // SAFETY: as above.
            if unsafe { (api.memory)(self.handle, &mut m) } == SUCCESS {
                s.memory_total_bytes = Some(m.total);
                s.memory_used_bytes = Some(m.used);
            }
            // SAFETY: as above.
            s.power_w = self.uint(|h, v| unsafe { (api.power)(h, v) }).map(|mw| f64::from(mw) / 1e3);
            // SAFETY: as above.
            s.power_limit_w = self.uint(|h, v| unsafe { (api.power_limit)(h, v) }).map(|mw| f64::from(mw) / 1e3);
            // SAFETY: as above; TEMPERATURE_GPU is NVML_TEMPERATURE_GPU.
            s.temperature_c = self.uint(|h, v| unsafe { (api.temperature)(h, TEMPERATURE_GPU, v) });
            // SAFETY: as above; CLOCK_SM is NVML_CLOCK_SM.
            s.sm_clock_mhz = self.uint(|h, v| unsafe { (api.clock)(h, CLOCK_SM, v) });
            s.process_memory_bytes = self.process_memory();
        }

        /// GPU memory of this process. Not found when NVML sees another PID namespace (some containers).
        fn process_memory(&self) -> Option<u64> {
            let processes = self.api.processes?;
            let mut infos = [ProcessInfo::default(); MAX_PROCESSES];
            let mut n = MAX_PROCESSES as c_uint;
            // SAFETY: `infos` holds `n` writable entries of the `_v3` struct; NVML writes at most `n` and stores the
            // number written in `n` (more processes than that: an error, and nothing is read).
            if unsafe { processes(self.handle, &mut n, infos.as_mut_ptr()) } != SUCCESS {
                return None;
            }
            let pid = std::process::id();
            infos[..(n as usize).min(MAX_PROCESSES)]
                .iter()
                .find(|p| p.pid == pid && p.used_gpu_memory != NOT_AVAILABLE)
                .map(|p| p.used_gpu_memory)
        }
    }
}
