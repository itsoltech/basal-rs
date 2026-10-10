//! Host checkpoints outside timing intervals. Swap counters are global, not attributed to basal.

use anyhow::{Context, Result};
use basal_gpu::{GpuMemory, GpuMemoryProbe};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const GIB: f64 = 1073741824.0;
const SWAP_GROWTH_LIMIT: u64 = 64 * 1024 * 1024;

pub(super) struct Guard {
    baseline: HostMemory,
    min_ram: u64,
    min_gpu: u64,
    max_gpu_allocated: Option<u64>,
    max_swap_growth: u64,
    checks: usize,
    ram_reserve: u64,
    gpu_reserve: u64,
}

impl Guard {
    pub fn new(host: HostMemory, gpu: &GpuMemory, options: &super::Options) -> Self {
        Self {
            min_ram: host.available_bytes,
            min_gpu: gpu.available_bytes,
            max_gpu_allocated: gpu.process_allocated_bytes,
            baseline: host,
            max_swap_growth: 0,
            checks: 0,
            ram_reserve: (options.ram_reserve_gib * GIB) as u64,
            gpu_reserve: (options.gpu_reserve_gib * GIB) as u64,
        }
    }

    pub fn check(&mut self, host: &HostMemory, gpu: &GpuMemory, weights: u64) -> Result<()> {
        self.checks += 1;
        self.min_ram = self.min_ram.min(host.available_bytes);
        self.min_gpu = self.min_gpu.min(gpu.available_bytes);
        if let Some(a) = gpu.process_allocated_bytes {
            self.max_gpu_allocated = Some(self.max_gpu_allocated.unwrap_or(0).max(a));
        }
        let swap = host
            .swap_used_bytes
            .saturating_sub(self.baseline.swap_used_bytes)
            .max(host.swap_out_bytes.saturating_sub(self.baseline.swap_out_bytes));
        self.max_swap_growth = self.max_swap_growth.max(swap);
        anyhow::ensure!(!host.pressure, "host memory pressure is elevated");
        anyhow::ensure!(swap < SWAP_GROWTH_LIMIT, "global swap activity grew by at least 64 MiB");
        let ram_need = self.ram_reserve.saturating_add(if gpu.unified { weights } else { 0 });
        anyhow::ensure!(
            host.available_bytes >= ram_need,
            "available RAM fell below the reserve / estimated weight requirement"
        );
        anyhow::ensure!(
            gpu.available_bytes >= self.gpu_reserve.saturating_add(weights),
            "GPU / Metal working-set headroom fell below the reserve / estimated weight requirement"
        );
        Ok(())
    }

    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"before_load": self.baseline, "checkpoints": self.checks,
            "min_available_ram_bytes_sampled": self.min_ram, "min_available_gpu_bytes_sampled": self.min_gpu,
            "max_process_gpu_allocated_bytes_sampled": self.max_gpu_allocated, "max_global_swap_growth_bytes": self.max_swap_growth,
            "ram_reserve_bytes": self.ram_reserve, "gpu_reserve_bytes": self.gpu_reserve, "swap_growth_limit_bytes": SWAP_GROWTH_LIMIT})
    }
}

struct Sample {
    at: Instant,
    host: HostMemory,
    gpu: GpuMemory,
    cpu_s: f64,
    rss: u64,
}

pub(super) struct ResourceReport {
    pub summary: serde_json::Value,
    samples: std::collections::VecDeque<Sample>,
    pub reason: Option<String>,
}

impl ResourceReport {
    pub fn window(&self, start: Instant, end: Instant) -> serde_json::Value {
        let samples = self.samples.iter().filter(|s| s.at >= start && s.at <= end).collect::<Vec<_>>();
        let cpu = samples.first().zip(samples.last()).and_then(|(a, b)| {
            let wall = (b.at - a.at).as_secs_f64();
            (wall > 0.0).then(|| ((b.cpu_s - a.cpu_s).max(0.0), wall))
        });
        serde_json::json!({"samples": samples.len(),
            "min_available_ram_bytes_sampled": samples.iter().map(|s| s.host.available_bytes).min(),
            "min_available_gpu_bytes_sampled": samples.iter().map(|s| s.gpu.available_bytes).min(),
            "max_process_gpu_allocated_bytes_sampled": samples.iter().filter_map(|s| s.gpu.process_allocated_bytes).max(),
            "lifetime_peak_rss_bytes": samples.iter().map(|s| s.rss).max(),
            "mean_process_cpu_cores": cpu.map(|(used, wall)| used / wall),
            "cpu_sample_window_s": cpu.map(|(_, wall)| wall),
            "scope": "same process HTTP server and load generator; sampled memory, not allocation peaks"})
    }
}

pub(super) struct Monitor {
    pub abort: Arc<AtomicBool>,
    stop: Option<std::sync::mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<ResourceReport>>,
}

impl Monitor {
    pub fn start(mut guard: Guard, probe: GpuMemoryProbe) -> Result<Self> {
        let abort = Arc::new(AtomicBool::new(false));
        let flag = abort.clone();
        let (stop, receiver) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("benchmark-resources".into())
            .spawn(move || {
                let mut samples = std::collections::VecDeque::new();
                let mut reason = None;
                loop {
                    let result = (|| -> Result<()> {
                        let host = host_memory()?;
                        let gpu = probe.snapshot()?;
                        let (cpu_s, rss) = process_usage()?;
                        let checked = guard.check(&host, &gpu, 0);
                        if samples.len() == 20_000 {
                            samples.pop_front();
                        }
                        samples.push_back(Sample { at: Instant::now(), host, gpu, cpu_s, rss });
                        checked
                    })();
                    if let Err(e) = result {
                        reason = Some(format!("{e:#}"));
                        flag.store(true, Ordering::Release);
                        break;
                    }
                    match receiver.recv_timeout(Duration::from_millis(200)) {
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        _ => break,
                    }
                }
                ResourceReport { summary: guard.report(), samples, reason }
            })
            .context("starting benchmark resource monitor")?;
        Ok(Self { abort, stop: Some(stop), thread: Some(thread) })
    }

    pub fn finish(mut self) -> Result<ResourceReport> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.thread
            .take()
            .context("resource monitor handle")?
            .join()
            .map_err(|_| anyhow::anyhow!("resource monitor panicked"))
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct HostMemory {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_out_bytes: u64,
    pub pressure: bool,
    pub cgroup_limit_bytes: Option<u64>,
}

#[cfg(target_os = "macos")]
fn command(cmd: &str, args: &[&str]) -> Result<String> {
    let out = std::process::Command::new(cmd).args(args).output().with_context(|| format!("running {cmd}"))?;
    anyhow::ensure!(out.status.success(), "{cmd}: {}", String::from_utf8_lossy(&out.stderr));
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

pub(super) fn host_memory() -> Result<HostMemory> {
    #[cfg(target_os = "macos")]
    {
        let vm = command("vm_stat", &[])?;
        let page_size: u64 = vm
            .lines()
            .next()
            .and_then(|l| l.split("page size of ").nth(1))
            .and_then(|v| v.split_whitespace().next())
            .context("vm_stat page size")?
            .parse()?;
        let pages = |name: &str| -> Result<u64> {
            Ok(vm
                .lines()
                .find_map(|l| l.strip_prefix(name))
                .with_context(|| format!("vm_stat: {name}"))?
                .trim()
                .trim_end_matches('.')
                .parse::<u64>()?
                * page_size)
        };
        let swap = command("sysctl", &["-n", "vm.swapusage"])?;
        let used = swap.split("used = ").nth(1).and_then(|s| s.split_whitespace().next()).context("swap used")?;
        let (value, unit) = used.split_at(used.len().saturating_sub(1));
        let multiplier = match unit {
            "M" => 1048576.0,
            "G" => 1073741824.0,
            _ => anyhow::bail!("unknown swap unit {unit}"),
        };
        let total = command("sysctl", &["-n", "hw.memsize"])?.parse()?;
        let pressure: u32 = command("sysctl", &["-n", "kern.memorystatus_vm_pressure_level"])?.parse()?;
        Ok(HostMemory {
            total_bytes: total,
            // A conservative checkpoint estimate, not macOS's exact allocatable memory.
            available_bytes: pages("Pages free:")? + pages("Pages inactive:")? + pages("Pages speculative:")?,
            swap_used_bytes: (value.parse::<f64>()? * multiplier) as u64,
            swap_out_bytes: pages("Swapouts:")?,
            pressure: pressure != 1,
            cgroup_limit_bytes: None,
        })
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mem = std::fs::read_to_string("/proc/meminfo").context("host memory telemetry needs /proc/meminfo")?;
        let bytes = |key: &str| -> Result<u64> {
            Ok(mem
                .lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|s| s.split_whitespace().next())
                .with_context(|| format!("meminfo: {key}"))?
                .parse::<u64>()?
                * 1024)
        };
        let vm = std::fs::read_to_string("/proc/vmstat").context("reading swap counters")?;
        let out: u64 = vm.lines().find_map(|l| l.strip_prefix("pswpout ")).context("pswpout")?.parse()?;
        // SAFETY: sysconf reads this constant system setting, retaining no pointers or resources.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        anyhow::ensure!(page_size > 0, "cannot read host page size");
        let mut host = HostMemory {
            total_bytes: bytes("MemTotal:")?,
            available_bytes: bytes("MemAvailable:")?,
            swap_used_bytes: bytes("SwapTotal:")?.saturating_sub(bytes("SwapFree:")?),
            swap_out_bytes: out.saturating_mul(page_size as u64),
            pressure: false,
            cgroup_limit_bytes: None,
        };
        cgroup_memory(&mut host)?;
        Ok(host)
    }
}

/// Respect process/ancestor cgroup limits as well as machine-wide MemAvailable.
#[cfg(not(target_os = "macos"))]
fn cgroup_memory(host: &mut HostMemory) -> Result<()> {
    use std::path::{Path, PathBuf};
    let groups = std::fs::read_to_string("/proc/self/cgroup").context("reading process memory limits")?;
    let v2 = groups.lines().find_map(|l| l.strip_prefix("0::"));
    let v1 = groups.lines().find_map(|l| {
        let mut fields = l.splitn(3, ':');
        fields.next()?;
        fields.next()?.split(',').any(|c| c == "memory").then(|| fields.next()).flatten()
    });
    // A hybrid host lists both; its memory controller is the v1 one, the unified v2 tree then has no memory.max.
    let (root, relative, limit_key, usage_key) = if let Some(path) = v1 {
        (Path::new("/sys/fs/cgroup/memory"), path, "memory.limit_in_bytes", "memory.usage_in_bytes")
    } else if let Some(path) = v2 {
        (Path::new("/sys/fs/cgroup"), path, "memory.max", "memory.current")
    } else {
        return Ok(());
    };
    let candidate = root.join(relative.trim_start_matches('/'));
    let root = root.canonicalize().context("locating cgroup memory controller")?;
    // A cgroup namespace may expose the current group as the mount root instead of its host path.
    let mut path: PathBuf = candidate.canonicalize().ok().filter(|p| p.starts_with(&root)).unwrap_or(root.clone());
    loop {
        let file = path.join(limit_key);
        if file.exists() {
            let value = std::fs::read_to_string(&file).context("reading cgroup memory limit")?;
            if value.trim() != "max" {
                let limit: u64 = value.trim().parse().context("cgroup memory limit")?;
                let used: u64 = std::fs::read_to_string(path.join(usage_key))
                    .context("reading cgroup memory usage")?
                    .trim()
                    .parse()
                    .context("cgroup memory usage")?;
                if limit < host.total_bytes {
                    host.cgroup_limit_bytes = Some(host.cgroup_limit_bytes.unwrap_or(u64::MAX).min(limit));
                }
                host.available_bytes = host.available_bytes.min(limit.saturating_sub(used));
            }
        }
        if path == root || !path.pop() {
            break;
        }
    }
    Ok(())
}

/// Both ends of every loopback connection share this process's descriptor limit; past it the server's accept loop
/// sleeps a second per failure, which would inflate latencies instead of failing requests.
pub(super) fn check_open_files(clients: usize) -> Result<()> {
    let mut limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    // SAFETY: getrlimit writes a complete rlimit to this aligned writable pointer on success;
    // nothing escapes the synchronous call and it is read only after checking the return code.
    anyhow::ensure!(unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, limit.as_mut_ptr()) } == 0, "getrlimit failed");
    // SAFETY: the successful call above initialized every field of rlimit.
    let soft = unsafe { limit.assume_init() }.rlim_cur;
    // Client and server socket per client, plus model, driver, runtime and monitor descriptors.
    let needed = (2 * clients + 64) as libc::rlim_t;
    anyhow::ensure!(
        soft == libc::RLIM_INFINITY || soft >= needed,
        "--concurrency {clients} needs about {needed} open files (client and server sockets share this process), \
         the limit is {soft}: raise it (e.g. `ulimit -n 4096`) or lower --concurrency"
    );
    Ok(())
}

/// Process CPU time and lifetime RSS high-water mark (including model loading).
pub(super) fn process_usage() -> Result<(f64, u64)> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: getrusage writes a complete rusage to this aligned writable pointer on success;
    // nothing escapes the synchronous call and it is read only after checking the return code.
    anyhow::ensure!(unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0, "getrusage failed");
    // SAFETY: the successful call above initialized every field of rusage.
    let usage = unsafe { usage.assume_init() };
    let seconds = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    let rss = usage.ru_maxrss.max(0) as u64;
    let rss = if cfg!(target_os = "macos") { rss } else { rss.saturating_mul(1024) };
    Ok((seconds(usage.ru_utime) + seconds(usage.ru_stime), rss))
}
