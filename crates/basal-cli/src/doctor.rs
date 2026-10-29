//! `basal doctor`: what this machine has for `basal serve` and what is missing, without loading a model.
//!
//! Checks the build, the system, the GPU (Apple Silicon and Metal; on Linux the NVIDIA driver, compute capability and
//! the CUDA libraries the CUDA build loads), the server configuration `basal serve` would use, whether its models are
//! cached or how much has to be downloaded, the free disk space and memory for them, the Hugging Face Hub and the port.
//! Every finding that needs action comes with the way to fix it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::config::{self, ServeConfig};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Level {
    Ok,
    Info,
    Warn,
    Fail,
}

impl Level {
    fn tag(self) -> &'static str {
        match self {
            Level::Ok => "ok",
            Level::Info => "info",
            Level::Warn => "warn",
            Level::Fail => "FAIL",
        }
    }
}

struct Check {
    area: &'static str,
    level: Level,
    what: String,
    fix: Option<String>,
}

#[derive(Default)]
struct Report(Vec<Check>);

impl Report {
    fn add(&mut self, area: &'static str, level: Level, what: impl Into<String>, fix: Option<String>) {
        self.0.push(Check { area, level, what: what.into(), fix });
    }
    fn ok(&mut self, area: &'static str, what: impl Into<String>) {
        self.add(area, Level::Ok, what, None);
    }
    fn info(&mut self, area: &'static str, what: impl Into<String>) {
        self.add(area, Level::Info, what, None);
    }
    fn warn(&mut self, area: &'static str, what: impl Into<String>, fix: impl Into<String>) {
        self.add(area, Level::Warn, what, Some(fix.into()));
    }
    fn fail(&mut self, area: &'static str, what: impl Into<String>, fix: impl Into<String>) {
        self.add(area, Level::Fail, what, Some(fix.into()));
    }
}

/// Lowest NVIDIA driver that loads the PTX of the CUDA 12.9 toolkit the Linux builds use.
const MIN_DRIVER: (u32, u32) = (575, 51);
/// Lowest compute capability of the kernels (mma.sync m16n8k16 with f16 operands, cp.async).
const MIN_COMPUTE_CAP: (u32, u32) = (8, 0);
/// Libraries the CUDA build links besides the driver's libcuda.so.1 (cuBLAS and cuBLASLt for GEMM, cuRAND and the
/// runtime through candle); `basal setup` installs them.
pub const CUDA_RUNTIME_LIBS: [&str; 4] = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcurand.so.10"];

const GB: f64 = 1e9;
/// memory sizes (RAM, GPU) in GiB, as the systems report them
const GIB: f64 = (1u64 << 30) as f64;

/// Run the checks; prints the report (text or `--json`) and returns false when something blocks `basal serve`.
pub fn run(file: Option<PathBuf>, model_refs: &[String], as_json: bool) -> Result<bool> {
    let mut r = Report::default();
    // Resolve scheduler/routing and local/cached host-PLE refusals before discovery.
    let mut config_report = Report::default();
    let config = configuration(&mut config_report, file, model_refs);
    let mut invalid_ple = false;
    let mut invalid_scheduler = false;
    if let Some(c) = &config {
        for model in &c.models {
            if let Err(e) = model.scheduler_policy() {
                config_report.fail(
                    "models",
                    format!("{e:#}"),
                    "long_lane: on needs positive long_tokens; short-first needs long_lane: off and positive long_tokens, or use scheduler: off",
                );
                invalid_scheduler = true;
            }
        }
    }
    if let Some(c) = config.as_ref().filter(|_| !invalid_scheduler) {
        for model in &c.models {
            // Device returns Ok before any check.
            if let Err(e) = config::check_ple_source(model.ple_placement, model.layer_split.is_some(), model) {
                config_report.fail(
                    "models",
                    format!("{}: {e:#}", config::describe(model)),
                    "host PLE needs a CUDA build and non-unified Gemma with PLE, without layer split",
                );
                invalid_ple = true;
            }
        }
    }
    // Resolve scheduler/routing conflicts before update notices and device discovery for Device and Host alike.
    build(&mut r, !invalid_scheduler);
    system(&mut r);
    let gpus = if invalid_ple || invalid_scheduler { None } else { gpu(&mut r) };
    r.0.extend(config_report.0);
    if let Some(c) = &config {
        if !invalid_scheduler {
            models(&mut r, c, gpus.as_deref());
        }
        port(&mut r, c);
    }
    let worst = r.0.iter().map(|c| c.level).max().unwrap_or(Level::Ok);
    if as_json {
        let checks: Vec<Value> =
            r.0.iter()
                .map(|c| {
                    let fix = c.fix.as_deref().map(crate::term::plain);
                    json!({"area": c.area, "level": c.level.tag().to_lowercase(), "what": crate::term::plain(&c.what), "fix": fix})
                })
                .collect();
        println!("{}", serde_json::to_string_pretty(&json!({"ok": worst < Level::Fail, "checks": checks}))?);
    } else {
        use crate::term::{paint, rich, tilde, Stream::Stdout as O, Style};
        let mark = |l: Level| match l {
            Level::Ok => paint(O, Style::Green, "✓"),
            Level::Info => paint(O, Style::Cyan, "·"),
            Level::Warn => paint(O, Style::Yellow, "!"),
            Level::Fail => paint(O, Style::Red, "✗"),
        };
        let mut area = "";
        for c in &r.0 {
            if c.area != area {
                area = c.area;
                println!("{}", paint(O, Style::Bold, area));
            }
            let base = match c.level {
                Level::Fail => Some(Style::Red),
                Level::Warn => Some(Style::Yellow),
                _ => None,
            };
            println!("  {} {}", mark(c.level), rich(O, base, &tilde(&c.what)));
            if let Some(f) = &c.fix {
                println!("    {} {}", paint(O, Style::Dim, "→"), rich(O, None, &tilde(f)));
            }
        }
        println!();
        let (style, line) = match worst {
            Level::Fail => (Style::Red, "✗ `basal serve` cannot run here yet: fix the ✗ lines above."),
            Level::Warn => (Style::Yellow, "! `basal serve` can run; the ! lines above may limit it."),
            _ => (Style::Green, "✓ `basal serve` can run."),
        };
        println!("{}", rich(O, Some(style), line));
    }
    Ok(worst < Level::Fail)
}

fn command(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn build(r: &mut Report, check_update: bool) {
    r.ok("basal", format!("basal {}", crate::version()));
    // the binary itself (Homebrew links bin/basal to Cellar/basal-rs/<version>/bin/basal)
    let exe = std::env::current_exe().ok().map(|e| e.canonicalize().unwrap_or(e));
    let how = match exe.as_deref().map(Path::to_string_lossy) {
        Some(p) if p == "/usr/local/bin/basal" && Path::new("/.dockerenv").exists() => {
            "container image (update: `docker compose pull`)".to_string()
        }
        Some(p) if p.contains("/Cellar/") => "with Homebrew (update: `brew upgrade basal-rs`)".to_string(),
        Some(p) => p.to_string(),
        None => "unknown location".to_string(),
    };
    r.info("basal", format!("installed: {how}"));
    if let Some(v) = check_update.then(crate::update::notice).flatten() {
        r.warn("basal", format!("version {v} is available"), crate::update::how());
    }
    if let Some(b) = crate::paths::cuda_binary() {
        r.info("basal", format!("GPU commands run the CUDA build {}", crate::term::P(&b)));
    } else if basal_gpu::BUILD_BACKEND == "no GPU backend" {
        r.fail(
            "basal",
            "this build has no GPU backend",
            "use a release package or the container image, or build with --features basal-cli/cuda",
        );
    }
}

fn system(r: &mut Report) {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let name = if os == "macos" {
        command("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}"))
    } else {
        std::fs::read_to_string("/etc/os-release").ok().and_then(|s| {
            s.lines().find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|v| v.trim_matches('"').to_string()))
        })
    };
    r.ok("system", format!("{} ({arch})", name.unwrap_or_else(|| os.to_string())));
    if os == "linux" {
        if let Some(v) = command("getconf", &["GNU_LIBC_VERSION"]) {
            r.info("system", v);
        }
    }
    if let Some(b) = memory_bytes() {
        r.ok("system", format!("memory {:.1} GiB", b as f64 / GIB));
    }
}

fn memory_bytes() -> Option<u64> {
    if cfg!(target_os = "macos") {
        command("sysctl", &["-n", "hw.memsize"])?.parse().ok()
    } else {
        let m = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb: u64 =
            m.lines().find_map(|l| l.strip_prefix("MemTotal:"))?.trim().trim_end_matches("kB").trim().parse().ok()?;
        Some(kb * 1024)
    }
}

/// A GPU `basal serve` selects with the `device` number of a model.
struct Gpu {
    name: String,
    /// memory available to the models (bytes), when known
    memory: Option<u64>,
    /// CUDA compute capability
    cap: Option<(u32, u32)>,
    /// CUDA streaming multiprocessors
    sms: Option<u32>,
}

/// GPU checks; returns the GPUs by `device` number, None when the numbers are unknown. CUDA: the devices the driver
/// shows this process, by ordinal (the numbering of `basal serve` started with the same environment).
fn gpu(r: &mut Report) -> Option<Vec<Gpu>> {
    #[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
    return intel_gpu(r).map(|g| vec![g]);
    #[cfg(not(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos"))))]
    candle_gpu(r)
}

/// The Intel Vulkan device the backend selects and its heap budget, queried through one device.
#[cfg(all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")))]
fn intel_gpu(r: &mut Report) -> Option<Gpu> {
    let info = match basal_gpu::intel::info() {
        Ok(info) => info,
        Err(e) => {
            r.fail("gpu", format!("{e:#}"), "install Vulkan loader and Intel Mesa driver (Arch: vulkan-intel)");
            return None;
        }
    };
    r.ok("gpu", format!("{}; Intel Vulkan, {}", info.name, info.driver));
    let memory = match info.memory {
        Ok(memory) => {
            if memory.unified {
                r.info("gpu", "GPU shares host RAM; weights and activations compete with the desktop");
            }
            r.info("gpu", format!("Vulkan heap budget: {:.1} GiB available", memory.available_bytes as f64 / GIB));
            Some(memory.available_bytes)
        }
        Err(e) => {
            r.warn(
                "gpu",
                format!("Vulkan memory telemetry unavailable: {e:#}"),
                "update the Intel Vulkan driver to support VK_EXT_memory_budget",
            );
            None
        }
    };
    Some(Gpu { name: info.name, memory, cap: None, sms: None })
}

/// Metal or NVIDIA CUDA checks.
#[cfg_attr(
    all(feature = "intel", not(feature = "cuda"), not(target_os = "macos")),
    expect(dead_code, reason = "the Intel build checks its Vulkan device instead")
)]
fn candle_gpu(r: &mut Report) -> Option<Vec<Gpu>> {
    if cfg!(target_os = "macos") {
        if std::env::consts::ARCH != "aarch64" {
            r.fail("gpu", "Intel Mac: the Metal backend needs Apple Silicon (M1 or newer)", "use an Apple Silicon Mac");
            return None;
        }
        #[cfg(target_os = "macos")]
        match basal_gpu::metal_info() {
            Some((name, ws)) => {
                r.ok("gpu", format!("{name}, Metal; recommended working set {:.1} GiB", ws as f64 / GIB));
                return Some(vec![Gpu { name, memory: Some(ws), cap: None, sms: None }]);
            }
            None => r.fail("gpu", "no Metal device", "Metal is part of macOS on Apple Silicon; check the system"),
        }
        return None;
    }
    if std::env::consts::OS != "linux" {
        r.fail("gpu", "no GPU backend for this system", "basal-rs runs on macOS (Apple Silicon) and Linux (NVIDIA)");
        return None;
    }
    let q = command(
        "nvidia-smi",
        &["--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader,nounits"],
    );
    let Some(q) = q else {
        r.fail(
            "gpu",
            "no NVIDIA driver (nvidia-smi not found or failing)",
            format!("install the NVIDIA driver {}.{} or newer", MIN_DRIVER.0, MIN_DRIVER.1),
        );
        cuda_libs(r);
        return None;
    };
    // the inventory of the machine, numbered as nvidia-smi numbers it (not necessarily as CUDA does)
    let mut caps = Vec::new();
    for (i, line) in q.lines().enumerate() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 4 {
            continue;
        }
        let (name, driver, cap, mem_mib) = (f[0], f[1], f[2], f[3]);
        let mem = mem_mib.parse::<u64>().ok().map(|m| m * 1024 * 1024);
        r.ok(
            "gpu",
            format!("nvidia-smi GPU {i}: {name}, {:.1} GiB, compute capability {cap}", mem.unwrap_or(0) as f64 / GIB),
        );
        caps.push(version_pair(cap));
        // one driver serves every GPU
        if i == 0 {
            let d = version_pair(driver);
            if d.is_some_and(|d| d >= MIN_DRIVER) {
                r.ok("gpu", format!("driver {driver}"));
            } else {
                r.fail(
                    "gpu",
                    format!(
                        "driver {driver}: the kernels (CUDA 12.9 PTX) need {}.{} or newer",
                        MIN_DRIVER.0, MIN_DRIVER.1
                    ),
                    "update the NVIDIA driver",
                );
            }
        }
    }
    // every GPU of the machine is too old, so none the process may see runs the kernels
    if !caps.is_empty() && caps.iter().all(|c| c.is_some_and(|c| c < MIN_COMPUTE_CAP)) {
        r.fail("gpu", min_cap_message("every GPU"), "use a newer GPU");
    }
    let devices = cuda_devices(r);
    cuda_libs(r);
    devices
}

fn min_cap_message(what: &str) -> String {
    format!(
        "{what}: the kernels need compute capability {}.{} or newer (A100, RTX 30xx and later)",
        MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1
    )
}

/// The CUDA devices of this process by ordinal, as `basal serve` started with the same environment numbers them;
/// None when the driver cannot be asked.
fn cuda_devices(r: &mut Report) -> Option<Vec<Gpu>> {
    for var in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"] {
        if let Some(v) = std::env::var_os(var) {
            r.info("gpu", format!("{var}={}", v.to_string_lossy()));
        }
    }
    let devices = match query_cuda_devices() {
        Ok(d) => d,
        Err(e) => {
            r.warn(
                "gpu",
                format!("CUDA devices unknown: {e:#}"),
                "the memory and compute capability of each model's `device` are not checked; check the NVIDIA \
                 driver (libcuda.so.1)",
            );
            return None;
        }
    };
    if devices.is_empty() {
        r.warn("gpu", "the CUDA driver shows this process no GPU", "check CUDA_VISIBLE_DEVICES");
    }
    for (i, d) in devices.iter().enumerate() {
        let cap = d.cap.map_or_else(|| "unknown".to_string(), |(a, b)| format!("{a}.{b}"));
        let mem = d.memory.map_or_else(|| "memory unknown".to_string(), |m| format!("{:.1} GiB", m as f64 / GIB));
        r.ok("gpu", format!("CUDA device {i}: {}, {mem}, compute capability {cap}", d.name));
    }
    Some(devices)
}

/// The devices from the CUDA driver API (libcuda.so.1, the driver's own library), which applies
/// `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` as the CUDA build does; no context is created.
fn query_cuda_devices() -> Result<Vec<Gpu>> {
    use std::ffi::{c_char, c_int, c_uint, CStr};
    // CUresult and CUdevice_attribute are C enums, CUdevice an int (cuda.h)
    type CuResult = c_uint;
    type CuDevice = c_int;
    const SUCCESS: CuResult = 0;
    const ERROR_NO_DEVICE: CuResult = 100;
    const CAP_MAJOR: c_uint = 75;
    const CAP_MINOR: c_uint = 76;
    const MULTIPROCESSORS: c_uint = 16;
    // SAFETY: loads the NVIDIA driver library by its fixed name. The handle is leaked: cuInit may start driver threads
    // running the library's code, so it must stay mapped for the rest of the process.
    let lib: &'static libloading::Library =
        Box::leak(Box::new(unsafe { libloading::Library::new("libcuda.so.1") }.context("loading libcuda.so.1")?));
    // SAFETY: the signatures of the CUDA driver API (cuda.h, a stable C ABI); cuDeviceTotalMem_v2 is the size_t
    // variant cuda.h maps cuDeviceTotalMem to. The library stays loaded (above), so the functions stay valid.
    let (init, count, get, name, total_mem, attribute) = unsafe {
        (
            *lib.get::<unsafe extern "C" fn(c_uint) -> CuResult>(b"cuInit\0")?,
            *lib.get::<unsafe extern "C" fn(*mut c_int) -> CuResult>(b"cuDeviceGetCount\0")?,
            *lib.get::<unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult>(b"cuDeviceGet\0")?,
            *lib.get::<unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult>(b"cuDeviceGetName\0")?,
            *lib.get::<unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult>(b"cuDeviceTotalMem_v2\0")?,
            *lib.get::<unsafe extern "C" fn(*mut c_int, c_uint, CuDevice) -> CuResult>(b"cuDeviceGetAttribute\0")?,
        )
    };
    let check = |what: &str, res: CuResult| {
        anyhow::ensure!(res == SUCCESS, "{what}: CUDA driver error {res}");
        Ok(())
    };
    // SAFETY: cuInit takes flags (0, the only valid value) and may be called any number of times
    match unsafe { init(0) } {
        SUCCESS => {}
        // CUDA_VISIBLE_DEVICES hides every GPU, or there is none
        ERROR_NO_DEVICE => return Ok(Vec::new()),
        res => check("cuInit", res)?,
    }
    let mut n: c_int = 0;
    // SAFETY: the driver is initialised; `n` is a valid out-pointer for the call
    check("cuDeviceGetCount", unsafe { count(&mut n) })?;
    (0..n)
        .map(|ordinal| {
            let mut dev: CuDevice = 0;
            // SAFETY: ordinal in 0..count; `dev` is a valid out-pointer
            check("cuDeviceGet", unsafe { get(&mut dev, ordinal) })?;
            let mut buf = [0u8; 256];
            // SAFETY: `buf` is writable for the length passed (256 fits c_int); the driver writes at most that many
            // bytes. The name is read up to the first NUL inside `buf` below, so a missing terminator is harmless.
            check("cuDeviceGetName", unsafe { name(buf.as_mut_ptr().cast(), 256, dev) })?;
            let name = CStr::from_bytes_until_nul(&buf)
                .map_or_else(|_| String::from_utf8_lossy(&buf).into_owned(), |s| s.to_string_lossy().into_owned());
            let mut bytes = 0usize;
            // SAFETY: `dev` from cuDeviceGet; `bytes` is a valid out-pointer
            let res = unsafe { total_mem(&mut bytes, dev) };
            let memory = (res == SUCCESS).then_some(bytes as u64);
            let attr = |a: c_uint| {
                let mut v: c_int = 0;
                // SAFETY: `dev` from cuDeviceGet, `a` a valid attribute; `v` is a valid out-pointer
                (unsafe { attribute(&mut v, a, dev) } == SUCCESS).then(|| u32::try_from(v).ok()).flatten()
            };
            let cap = attr(CAP_MAJOR).zip(attr(CAP_MINOR));
            Ok(Gpu { name, memory, cap, sms: attr(MULTIPROCESSORS) })
        })
        .collect()
}

fn version_pair(v: &str) -> Option<(u32, u32)> {
    let mut it = v.split('.');
    Some((it.next()?.parse().ok()?, it.next().unwrap_or("0").parse().ok()?))
}

/// Linux: the shared libraries the CUDA build needs, found as the dynamic loader would find them or installed by
/// `basal setup`; then whether the CUDA build of the package starts with them.
fn cuda_libs(r: &mut Report) {
    // SAFETY: only the known NVIDIA driver library is loaded and no symbols are retained after it is dropped.
    if unsafe { libloading::Library::new("libcuda.so.1") }.is_ok() {
        r.ok("cuda", "libcuda.so.1 (driver)");
    } else {
        r.fail("cuda", "libcuda.so.1 not found", "part of the NVIDIA driver: install the driver");
    }
    let managed = crate::setup::cuda_lib_dir();
    let mut missing = Vec::new();
    for lib in CUDA_RUNTIME_LIBS {
        if managed.join(lib).exists() {
            r.ok("cuda", format!("{lib} ({})", crate::term::P(&managed)));
        // SAFETY: lib is one of the fixed NVIDIA runtime library names; no symbols outlive this probe.
        } else if unsafe { libloading::Library::new(lib) }.is_ok() {
            r.ok("cuda", format!("{lib} (system)"));
        } else {
            missing.push(lib);
        }
    }
    if !missing.is_empty() {
        r.fail(
            "cuda",
            format!("not found: {}", missing.join(", ")),
            "`basal setup` (downloads the CUDA 12.9 libraries from NVIDIA, ~1 GB, into the user data directory)",
        );
        return;
    }
    if let Some(b) = crate::paths::cuda_binary() {
        match crate::cuda_command(&b).arg("--version").output() {
            Ok(o) if o.status.success() => {
                r.ok("cuda", format!("CUDA build starts: {}", String::from_utf8_lossy(&o.stdout).trim()))
            }
            Ok(o) => r.fail(
                "cuda",
                format!("CUDA build does not start: {}", String::from_utf8_lossy(&o.stderr).trim()),
                "`basal setup --force`; check that the package matches this system (glibc 2.28 or newer)",
            ),
            Err(e) => r.fail("cuda", format!("{}: {e}", crate::term::P(&b)), "reinstall basal"),
        }
    }
}

/// The configuration `basal serve` would use with the same `--config` / `--model`.
fn configuration(r: &mut Report, file: Option<PathBuf>, model_refs: &[String]) -> Option<ServeConfig> {
    match config::select(file, model_refs) {
        Ok((config::Source::File(p), c)) => {
            r.ok("config", format!("{} ({} model(s))", crate::term::P(&p), c.models.len()));
            Some(c)
        }
        Ok((config::Source::Models, c)) => {
            let ms: Vec<String> = c.models.iter().map(config::describe).collect();
            r.ok("config", format!("--model {}", ms.join(", ")));
            Some(c)
        }
        Ok((config::Source::Builtin, c)) => {
            r.info(
                "config",
                format!(
                    "no `--model` and no {} here: `basal serve` serves {} (`basal init` writes {0})",
                    crate::paths::CONFIG_NAME,
                    crate::models::default_model().repo
                ),
            );
            Some(c)
        }
        Err(e) => {
            let fix = if model_refs.is_empty() {
                "fix the file, write a new one (`basal init --force`) or give the model: `basal serve --model 2.0-4.5B`"
            } else {
                "`--model` takes 2.0-mini/2.0-4.5B/2.0-max, omni-mini/omni-max, a name (owner Remek), \
                 owner/name, each with @revision, or a model directory; mini/4.5B/max select basal-1.5"
            };
            r.fail("config", format!("{e:#}"), fix);
            None
        }
    }
}

fn models(r: &mut Report, c: &ServeConfig, gpus: Option<&[Gpu]>) {
    let mut download = 0u64;
    // checkpoint bytes of the models of each device
    let mut resident = std::collections::BTreeMap::<usize, u64>::new();
    let mut repos = Vec::new();
    let mut ple_host_load = 0u64;
    let mut ple_load_scratch = std::collections::BTreeMap::<usize, u64>::new();
    // replica groups and layer splits (several devices) whose GPUs are compared below
    let mut groups: Vec<(String, config::Placement)> = Vec::new();
    // the backend of `basal serve`: this build's, or the CUDA build of the Linux package
    let cuda = cfg!(feature = "cuda") || crate::paths::cuda_binary().is_some();
    for m in &c.models {
        let routing = match m.routing_policy() {
            Ok(routing) => routing,
            Err(e) => {
                r.fail(
                    "models",
                    format!("{e:#}"),
                    "set positive `long_tokens` for `long_lane: on`, or use `long_lane: off`",
                );
                continue;
            }
        };
        // the check `basal serve` makes before any download (with its default kernel set, fused)
        let checked = config::check_quantization(&config::describe(m), &m.quantization, &m.dtype, "fused", cuda);
        let quantization = match checked {
            Ok(q) => q,
            Err(e) => {
                r.fail("models", format!("{e:#}"), quantization_fix(&m.quantization));
                continue;
            }
        };
        let placement = match m.placement() {
            Ok(p) => p,
            Err(e) => {
                r.fail(
                    "models",
                    format!("{}: {e:#}", config::describe(m)),
                    "`device: N` for one GPU, `replica_devices: [N, M]` (distinct CUDA devices) for replicas, or \
                     `layer_split: [{device: N, layers: K}, ...]` for one model over several GPUs",
                );
                continue;
            }
        };
        // the static checks `basal serve` makes of a layer split before any download
        if let Some(split) = placement.split() {
            if let Err(e) = config::check_layer_split(&config::describe(m), split, &m.dtype, "fused", cuda) {
                r.fail(
                    "models",
                    format!("{e:#}"),
                    "`layer_split` runs on the CUDA build with `dtype: f16`; otherwise give the model one `device`",
                );
                continue;
            }
        }
        if placement.shared_devices().is_some() {
            groups.push((config::describe(m), placement.clone()));
        }
        // the checkpoint is loaded on every device of the model
        let mut bytes = 0u64;
        // q8_0 / q4_k: the weight estimate needs the model's config.json (local or cached)
        let mut model_dir = None;
        match (&m.path, &m.repo) {
            (Some(p), _) => match basal_core::checkpoint::bytes(p) {
                Ok(md) => {
                    bytes += md;
                    model_dir = Some(p.clone());
                    r.ok("models", format!("{}: local, {:.1} GB", crate::term::P(&p), md as f64 / GB));
                }
                Err(_) => r.fail(
                    "models",
                    format!("{}: no complete safetensors checkpoint", crate::term::P(&p)),
                    "a model directory holds config.json, basal.json, tokenizer.json and model.safetensors; or use a repository",
                ),
            },
            (None, Some(repo)) => {
                let size = config::known_size_gb(m).map(|g| (g * GB) as u64);
                let at = m.revision.as_deref().unwrap_or("main");
                match config::cached_model_dir(repo, m.revision.as_deref()) {
                    Some(dir) => {
                        let b = basal_core::checkpoint::bytes(&dir).unwrap_or(0);
                        bytes += b;
                        r.ok("models", format!("{repo}@{}: cached, {:.1} GB", short(at), b as f64 / GB));
                        model_dir = Some(dir);
                    }
                    None => {
                        download += size.unwrap_or(0);
                        bytes += size.unwrap_or(0);
                        repos.push((repo.clone(), at.to_string()));
                        r.info(
                            "models",
                            match size {
                                Some(s) => format!(
                                    "{repo}@{}: {:.1} GB to download at the first start",
                                    short(at),
                                    s as f64 / GB
                                ),
                                None => format!("{repo}@{}: to download at the first start", short(at)),
                            },
                        );
                    }
                }
            }
            _ => {}
        }
        let checkpoint = bytes;
        if m.ple_placement == basal_gpu::PleMode::Host {
            let estimate = (|| -> Result<_> {
                let dir = model_dir.as_deref().context("host PLE estimate requires a local or cached checkpoint")?;
                let value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
                let arch = basal_core::architecture::Architecture::parse(&value)?;
                m.ple_placement.resolve(&arch, placement.split().is_some())?;
                let qualification = m.batch_qualification()?;
                let background = c.batches.enabled
                    && matches!(
                        qualification,
                        crate::serve::batches::config::BackgroundQualification::Qualified { .. }
                    );
                let workers = routing.workers(background);
                let memory = basal_gpu::RuntimeBackend::host_ple_memory(dir, &arch)?;
                let replicas = u64::try_from(placement.instances().len())?;
                let host = memory
                    .host_load_bytes(workers)?
                    .checked_mul(replicas)
                    .context("host PLE replica estimate overflow")?;
                Ok((memory, host, workers, replicas))
            })();
            match estimate {
                Ok((memory, host, workers, replicas)) => {
                    bytes = memory.estimated_device_weight_bytes;
                    for devices in placement.instances() {
                        for device in devices {
                            let scratch = ple_load_scratch.entry(device).or_default();
                            *scratch = (*scratch).max(memory.estimated_device_load_scratch_bytes);
                        }
                    }
                    match ple_host_load.checked_add(host) {
                        Some(total) => ple_host_load = total,
                        None => {
                            r.fail("models", "host PLE server estimate overflow", "serve fewer replicas");
                            continue;
                        }
                    }
                    r.info("models", format!("{}: host PLE, {replicas} loaded replica(s), {workers} workers/replica; source estimate {}; checkpoint/mappings and scratch are separate, not allocator peaks or measured VRAM savings",
                        config::describe(m), serde_json::to_string(&memory).unwrap_or_default()));
                }
                Err(e) => {
                    r.fail("models", format!("{}: host PLE estimate unavailable ({e:#})", config::describe(m)),
                        "use the CUDA build with a local/cached non-unified Gemma PLE checkpoint and placement-specific Batch qualification");
                    continue;
                }
            }
        }
        if quantization != basal_gpu::Quantization::None {
            bytes = quantized_bytes(r, m, model_dir.as_deref(), bytes, quantization);
        }
        match placement.split() {
            // the checkpoint is loaded on every device of the model
            None => {
                for devices in placement.instances() {
                    for d in devices {
                        *resident.entry(d).or_default() += bytes;
                    }
                }
            }
            // each stage holds its part of the weights
            Some(split) => {
                for (d, b) in split_bytes(r, m, split, model_dir.as_deref(), checkpoint, bytes, quantization) {
                    *resident.entry(d).or_default() += b;
                }
            }
        }
    }
    if ple_host_load > 0 {
        let checked = (|| -> Result<u64> {
            let pool = c.batches.host_exec_pool_bytes()?;
            let total = ple_host_load.checked_add(pool).context("host PLE and Batch pool overflow")?;
            config::check_host_ple_headroom(total)?;
            Ok(total)
        })();
        match checked {
            Ok(total) => r.info("models", format!("host PLE additional source/load/worker estimate plus Batch execution pool: {total} bytes within current host/cgroup headroom; process overhead and allocator rounding need additional space")),
            Err(e) => r.fail("models", format!("{e:#}"), "serve fewer replicas or workers, or increase available host RAM"),
        }
    }
    for (model, placement) in &groups {
        let Some(devices) = placement.shared_devices() else { continue };
        let split = placement.split().is_some();
        if !cuda {
            // a layer split was refused above without the CUDA backend
            r.fail(
                "models",
                format!("{model}: replicas on devices {devices:?} need the CUDA build"),
                "remove `replica_devices` from the configuration",
            );
            continue;
        }
        // missing devices are reported per device below; GPUs of unknown type are not compared
        let Some(gpus) = gpus else { continue };
        let Some(kinds) = devices.iter().map(|&d| gpus.get(d)).collect::<Option<Vec<_>>>() else { continue };
        let kind = |g: &Gpu| (g.name.clone(), g.cap, g.sms);
        if kinds.iter().any(|g| kind(g) != kind(kinds[0])) {
            let listed: Vec<String> = devices
                .iter()
                .zip(&kinds)
                .map(|(d, g)| {
                    let cap = g.cap.map_or_else(|| "unknown".to_string(), |(a, b)| format!("{a}.{b}"));
                    let sms = g.sms.map_or_else(|| "unknown".to_string(), |n| n.to_string());
                    format!("device {d}: {}, compute capability {cap}, {sms} SMs", g.name)
                })
                .collect();
            if split {
                r.fail(
                    "models",
                    format!("{model}: a layer split needs GPUs of one type ({})", listed.join("; ")),
                    "list devices of one GPU type in `layer_split`; models of their own may use other GPUs",
                );
            } else {
                r.fail(
                    "models",
                    format!("{model}: replicas need GPUs of one type ({})", listed.join("; ")),
                    "list devices of one GPU type in `replica_devices`; models of their own may use other GPUs",
                );
            }
        } else if split {
            r.ok("models", format!("{model}: layer split over devices {devices:?} ({})", kinds[0].name));
        } else {
            r.ok("models", format!("{model}: {} replicas on devices {devices:?} ({})", devices.len(), kinds[0].name));
        }
    }
    let several = resident.keys().any(|&d| d > 0);
    for (&device, &resident) in &resident {
        if device > 0 && !cuda {
            r.fail(
                "models",
                format!("device {device}: this build's GPU backend runs on device 0 only"),
                "remove `device` from the configuration (several devices need the CUDA build)",
            );
            continue;
        }
        // numbering unknown (no GPU found, or the CUDA driver cannot be asked): reported in the GPU checks
        let Some(gpus) = gpus else { continue };
        let Some(g) = gpus.get(device) else {
            r.fail(
                "models",
                format!("device {device}: this process sees {} GPU(s), numbered from 0", gpus.len()),
                if cuda {
                    "set `device` of the model to one of the CUDA devices above (CUDA_VISIBLE_DEVICES selects and \
                     renumbers them)"
                } else {
                    "remove `device` from the configuration"
                },
            );
            continue;
        };
        let on = if several { format!(" on device {device} ({})", g.name) } else { String::new() };
        if g.cap.is_some_and(|c| c < MIN_COMPUTE_CAP) {
            r.fail("models", min_cap_message(&format!("device {device} ({})", g.name)), "give the models a newer GPU");
        }
        let Some(b) = g.memory else { continue };
        // f16 weights take about the checkpoint size (bf16) on the GPU, plus activations and caches (q8_0 / q4_k
        // models: their estimated weight bytes)
        let need = resident as f64 * 1.1 + ple_load_scratch.get(&device).copied().unwrap_or(0) as f64;
        if need > b as f64 {
            r.warn(
                "models",
                format!("models{on} need ~{:.1} GiB of GPU memory, {:.1} GiB available", need / GIB, b as f64 / GIB),
                if cuda {
                    "serve fewer or smaller models (`basal serve --model mini`) or spread them over GPUs \
                     (`device` per model)"
                } else {
                    "serve fewer or smaller models (`basal serve --model mini`)"
                },
            );
        } else if resident > 0 {
            r.ok("models", format!("GPU memory{on}: ~{:.1} of {:.1} GiB", need / GIB, b as f64 / GIB));
        }
    }
    if download > 0 {
        let dir = hf_cache_dir();
        match free_bytes(&dir) {
            Some(free) if free < download + download / 10 => r.fail(
                "disk",
                format!(
                    "{}: {:.1} GB free, {:.1} GB to download",
                    crate::term::P(&dir),
                    free as f64 / GB,
                    download as f64 / GB
                ),
                "free disk space or set `HF_HOME` to a larger disk",
            ),
            Some(free) => r.ok("disk", format!("{}: {:.1} GB free", crate::term::P(&dir), free as f64 / GB)),
            None => {}
        }
        hub(r, &repos);
    }
    if let Some(gc) = Some(&c.gemm_cache).filter(|_| cfg!(feature = "cuda")) {
        let gpus = crate::config::bundled_gpus();
        let built_in = if gpus.is_empty() { "none".to_string() } else { gpus.join(", ") };
        r.info("models", format!("GEMM tables: {}; built in for: {built_in}", crate::term::P(&gc)));
    }
}

const QUANTIZATION_FIX: &str =
    "`quantization: q8_0` runs on CUDA with `dtype: f16`; otherwise use `quantization: none` (default)";
const QUANTIZATION_FIX_Q4K: &str =
    "`quantization: q4_k` runs on CUDA with `dtype: f16`; otherwise use `quantization: none` (default)";

/// The fix of a rejected `quantization`: the q4_k text for `q4_k`, otherwise the q8_0 one (also for unknown values).
fn quantization_fix(quantization: &str) -> &'static str {
    match quantization.parse() {
        Ok(basal_gpu::Quantization::Q4K) => QUANTIZATION_FIX_Q4K,
        _ => QUANTIZATION_FIX,
    }
}

/// Estimated GPU bytes of a quantized model's weights (format arithmetic over its config.json); the checkpoint size
/// when the configuration is not on this machine yet. `q4_k` with a known configuration whose projection K is not a
/// multiple of 256 fails (the load would); every other estimate error is information, as for `q8_0`.
fn quantized_bytes(
    r: &mut Report,
    m: &config::ModelConfig,
    dir: Option<&Path>,
    checkpoint: u64,
    q: basal_gpu::Quantization,
) -> u64 {
    let cfg = model_config(dir);
    if let (Ok(cfg), basal_gpu::Quantization::Q4K, Some(block)) = (&cfg, q, q.block_values()) {
        if let Some((n, k)) = basal_gpu::projection_shapes(cfg).into_iter().find(|&(_, k)| !k.is_multiple_of(block)) {
            r.fail(
                "models",
                format!(
                    "{}: quantization q4_k needs every projection K to be a multiple of {block}; projection {n}x{k} \
                     is not",
                    config::describe(m)
                ),
                "use `quantization: none` (default) for this model, or `q8_0` when every projection K is a multiple \
                 of 32",
            );
            return checkpoint;
        }
    }
    let estimate = cfg.and_then(|cfg| basal_gpu::estimated_weight_bytes(&cfg, checkpoint, q));
    match estimate {
        Ok(b) => {
            r.ok("models", format!("{}: {q} weights ~{:.1} GB (estimate)", config::describe(m), b as f64 / GB));
            b
        }
        Err(e) => {
            r.info(
                "models",
                format!(
                    "{}: {q} weight estimate unavailable ({e:#}); the checkpoint size is used",
                    config::describe(m)
                ),
            );
            checkpoint
        }
    }
}

/// The configuration (`config.json`) of a model directory on this machine.
fn model_config(dir: Option<&Path>) -> Result<basal_core::manifest::LlamaConfig> {
    let d = dir.context("not downloaded yet")?;
    let text = std::fs::read_to_string(d.join("config.json")).context("reading config.json")?;
    basal_core::manifest::parse_config(&serde_json::from_str(&text).context("parsing config.json")?)
}

/// Estimated GPU bytes of each stage of a layer split (format arithmetic): with the model's configuration the layers'
/// share plus the embedding on the first and lm_head and norm on the last stage
/// ([`basal_gpu::estimated_stage_weight_bytes`]), after checking that the split covers the model's layers; without
/// it, `total` (the whole model's estimate) in proportion to the stages' layer counts.
fn split_bytes(
    r: &mut Report,
    m: &config::ModelConfig,
    split: &basal_gpu::LayerSplit,
    dir: Option<&Path>,
    checkpoint: u64,
    total: u64,
    q: basal_gpu::Quantization,
) -> Vec<(usize, u64)> {
    let stages = split.stages();
    let proportional = || {
        let layers: u128 = stages.iter().map(|s| s.layers as u128).sum();
        stages
            .iter()
            .map(|s| {
                (s.device, u64::try_from(u128::from(total) * s.layers as u128 / layers.max(1)).unwrap_or(u64::MAX))
            })
            .collect::<Vec<_>>()
    };
    let cfg = dir.context("not downloaded yet").and_then(|d| {
        let text = std::fs::read_to_string(d.join("config.json"))?;
        basal_core::architecture::Architecture::parse(&serde_json::from_str(&text)?)
    });
    let Ok(cfg) = cfg else { return proportional() };
    if let basal_core::architecture::Architecture::Gemma4(gemma) = &cfg {
        if q != basal_gpu::Quantization::None {
            r.fail(
                "models",
                format!("{}: Gemma layer split needs quantization none", config::describe(m)),
                "use `quantization: none`",
            );
            return proportional();
        }
        if let Err(e) = basal_gpu::RuntimeBackend::check_gemma_split(gemma) {
            r.fail(
                "models",
                format!("{}: {e:#}", config::describe(m)),
                "use one device or replicas for this Gemma configuration",
            );
            return proportional();
        }
    }
    if let Err(e) = config::check_split_coverage(&config::describe(m), split, cfg.num_layers()) {
        r.fail(
            "models",
            format!("{e:#}"),
            format!("give `layer_split` layer counts that add up to the model's {} layers", cfg.num_layers()),
        );
        return proportional();
    }
    let last = stages.len().saturating_sub(1);
    let estimates = stages
        .iter()
        .enumerate()
        .map(|(i, s)| {
            match &cfg {
                basal_core::architecture::Architecture::Qwen35(_) => {
                    anyhow::bail!("Qwen3.5 does not support layer split")
                }
                basal_core::architecture::Architecture::Llama(c) => {
                    basal_gpu::estimated_stage_weight_bytes(c, checkpoint, q, s.layers, i == 0, i == last)
                }
                basal_core::architecture::Architecture::Gemma4(c) => {
                    basal_gpu::RuntimeBackend::estimated_gemma_stage_bytes(
                        dir.context("downloaded model directory")?,
                        c,
                        split.ranges(c.layers.len())?[i].clone(),
                        m.media,
                    )
                }
            }
            .map(|b| (s.device, b))
        })
        .collect::<Result<Vec<_>>>();
    match estimates {
        Ok(e) => {
            let per: Vec<String> = e.iter().map(|(d, b)| format!("device {d} ~{:.1} GB", *b as f64 / GB)).collect();
            r.ok(
                "models",
                format!("{}: layer split {split}, weights {} (estimate)", config::describe(m), per.join(", ")),
            );
            e
        }
        Err(e) => {
            r.info(
                "models",
                format!(
                    "{}: layer split weight estimate unavailable ({e:#}); shares of the whole",
                    config::describe(m)
                ),
            );
            proportional()
        }
    }
}

fn short(rev: &str) -> &str {
    if rev.len() == 40 {
        &rev[..8]
    } else {
        rev
    }
}

/// The Hugging Face cache (hf-hub: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub).
fn hf_cache_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("HF_HUB_CACHE") {
        return PathBuf::from(p);
    }
    match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub"),
    }
}

/// Free space of the file system holding `p` (or its nearest existing ancestor).
fn free_bytes(p: &Path) -> Option<u64> {
    let mut d = p;
    while !d.exists() {
        d = d.parent()?;
    }
    let c = std::ffi::CString::new(d.as_os_str().as_encoded_bytes()).ok()?;
    // SAFETY: statvfs is a C struct of integers, for which an all-zero representation is valid.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a NUL-terminated path and a zeroed statvfs the call fills.
    (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0).then(|| s.f_bavail as u64 * s.f_frsize as u64)
}

/// The Hugging Face Hub serves the models still to be downloaded.
fn hub(r: &mut Report, repos: &[(String, String)]) {
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(_) => return,
    };
    let _guard = rt.enter(); // the client's connector needs the runtime
    let client = match reqwest::Client::builder().timeout(std::time::Duration::from_secs(10)).build() {
        Ok(c) => c,
        Err(_) => return,
    };
    for (repo, at) in repos {
        let url = format!("https://huggingface.co/api/models/{repo}/revision/{at}");
        let mut req = client.get(&url);
        if let Some(t) = &token {
            req = req.bearer_auth(t);
        }
        match rt.block_on(req.send()) {
            Ok(resp) if resp.status().is_success() => {
                r.ok("network", format!("huggingface.co serves {repo}@{}", short(at)))
            }
            Ok(resp) if resp.status().as_u16() == 401 || resp.status().as_u16() == 403 => r.fail(
                "network",
                format!("{repo}: access refused ({})", resp.status()),
                "set `HF_TOKEN` to a token with access to the repository",
            ),
            Ok(resp) => {
                r.fail("network", format!("{repo}@{at}: {}", resp.status()), "check the repository and revision")
            }
            Err(e) => r.fail(
                "network",
                format!("huggingface.co not reachable ({e})"),
                "check the network or proxy (`HTTPS_PROXY`); models can also be given as local paths",
            ),
        }
    }
}

/// The address `basal serve` would listen on is free.
fn port(r: &mut Report, c: &ServeConfig) {
    let addr = std::env::var("BASAL_ADDR").ok().and_then(|v| v.parse().ok()).unwrap_or(c.addr);
    match std::net::TcpListener::bind(addr) {
        Ok(_) => r.ok("server", format!("{addr} is free")),
        Err(e) => r.warn(
            "server",
            format!("{addr}: {e}"),
            "another process listens there (`basal serve` already running?); set `addr` in the configuration or `BASAL_ADDR`",
        ),
    }
}
