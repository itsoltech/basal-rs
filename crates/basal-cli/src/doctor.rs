//! `basal doctor`: what this machine has for `basal serve` and what is missing, without loading a model.
//!
//! Checks the build, the system, the GPU (Apple Silicon and Metal; on Linux the NVIDIA driver, compute capability and
//! the CUDA libraries the CUDA build loads), the server configuration `basal serve` would use, whether its models are
//! cached or how much has to be downloaded, the free disk space and memory for them, the Hugging Face Hub and the port.
//! Every finding that needs action comes with the way to fix it.

use std::path::{Path, PathBuf};

use anyhow::Result;
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
    build(&mut r);
    system(&mut r);
    let gpu_budget = gpu(&mut r);
    let config = configuration(&mut r, file, model_refs);
    if let Some(c) = &config {
        models(&mut r, c, gpu_budget);
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

fn build(r: &mut Report) {
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
    if let Some(v) = crate::update::notice() {
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

/// GPU checks; returns the memory available to the models (bytes) when known.
fn gpu(r: &mut Report) -> Option<u64> {
    if cfg!(target_os = "macos") {
        if std::env::consts::ARCH != "aarch64" {
            r.fail("gpu", "Intel Mac: the Metal backend needs Apple Silicon (M1 or newer)", "use an Apple Silicon Mac");
            return None;
        }
        #[cfg(target_os = "macos")]
        match basal_gpu::metal_info() {
            Some((name, ws)) => {
                r.ok("gpu", format!("{name}, Metal; recommended working set {:.1} GiB", ws as f64 / GIB));
                return Some(ws);
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
    let mut budget = None;
    for (i, line) in q.lines().enumerate() {
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() < 4 {
            continue;
        }
        let (name, driver, cap, mem_mib) = (f[0], f[1], f[2], f[3]);
        let mem = mem_mib.parse::<u64>().ok().map(|m| m * 1024 * 1024);
        r.ok("gpu", format!("GPU {i}: {name}, {:.1} GiB, compute capability {cap}", mem.unwrap_or(0) as f64 / GIB));
        if i == 0 {
            budget = mem;
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
            if version_pair(cap).is_some_and(|c| c < MIN_COMPUTE_CAP) {
                r.fail(
                    "gpu",
                    format!(
                        "compute capability {cap}: the kernels need {}.{} or newer (A100, RTX 30xx and later)",
                        MIN_COMPUTE_CAP.0, MIN_COMPUTE_CAP.1
                    ),
                    "use a newer GPU",
                );
            }
        }
    }
    cuda_libs(r);
    budget
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
                    config::DEFAULT_MODEL.repo
                ),
            );
            Some(c)
        }
        Err(e) => {
            let fix = if model_refs.is_empty() {
                "fix the file, write a new one (`basal init --force`) or give the model: `basal serve --model 4.5B`"
            } else {
                "`--model` takes mini, 4.5B, max, a name (owner Remek), owner/name, each with @revision, or a model directory"
            };
            r.fail("config", format!("{e:#}"), fix);
            None
        }
    }
}

fn models(r: &mut Report, c: &ServeConfig, gpu_budget: Option<u64>) {
    let (mut download, mut resident) = (0u64, 0u64);
    let mut repos = Vec::new();
    for m in &c.models {
        match (&m.path, &m.repo) {
            (Some(p), _) => match std::fs::metadata(p.join("model.safetensors")) {
                Ok(md) => {
                    resident += md.len();
                    r.ok("models", format!("{}: local, {:.1} GB", crate::term::P(&p), md.len() as f64 / GB));
                }
                Err(_) => r.fail(
                    "models",
                    format!("{}: no model.safetensors", crate::term::P(&p)),
                    "a model directory holds config.json, basal.json, tokenizer.json and model.safetensors; or use a repository",
                ),
            },
            (None, Some(repo)) => {
                let size = config::known_size_gb(m).map(|g| (g * GB) as u64);
                let at = m.revision.as_deref().unwrap_or("main");
                match config::cached_model_dir(repo, m.revision.as_deref()) {
                    Some(dir) => {
                        let b = std::fs::metadata(dir.join("model.safetensors")).map(|m| m.len()).unwrap_or(0);
                        resident += b;
                        r.ok("models", format!("{repo}@{}: cached, {:.1} GB", short(at), b as f64 / GB));
                    }
                    None => {
                        download += size.unwrap_or(0);
                        resident += size.unwrap_or(0);
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
    }
    // f16 weights take about the checkpoint size (bf16) on the GPU, plus activations and caches
    if let Some(b) = gpu_budget {
        let need = resident as f64 * 1.1;
        if need > b as f64 {
            r.warn(
                "models",
                format!("models need ~{:.1} GiB of GPU memory, {:.1} GiB available", need / GIB, b as f64 / GIB),
                "serve fewer or smaller models (`basal serve --model mini`)",
            );
        } else if resident > 0 {
            r.ok("models", format!("GPU memory: ~{:.1} of {:.1} GiB", need / GIB, b as f64 / GIB));
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
        r.info("models", format!("GEMM tables: {}", crate::term::P(&gc)));
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
