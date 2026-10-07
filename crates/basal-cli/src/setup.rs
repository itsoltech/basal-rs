//! `basal setup`: prepare this machine for `basal serve`.
//!
//! - Linux: the CUDA 12 runtime libraries the CUDA build links (cudart, cuBLAS, cuBLASLt, cuRAND), downloaded from
//!   NVIDIA's redistributable archives (`developer.download.nvidia.com/compute/cuda/redist`, SHA-256 from NVIDIA's
//!   manifest) into the user data directory, unless the system already has them. Only the NVIDIA driver has to be
//!   installed by the administrator. `basal` runs the CUDA build with that directory on `LD_LIBRARY_PATH`.
//! - The user configuration (`basal init`) when there is none.
//! - `--prefetch`: the models of the configuration into the Hugging Face cache.
//! - `--service`: a user service that runs `basal serve` (systemd on Linux, launchd on macOS).

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use crate::{config, paths};

/// CUDA release whose libraries are installed (the toolkit the Linux builds are compiled with).
pub const CUDA_RELEASE: &str = "12.9.1";
/// NVIDIA components holding the libraries of [`crate::doctor`]'s CUDA list (cuBLASLt is in libcublas).
const COMPONENTS: [&str; 3] = ["cuda_cudart", "libcublas", "libcurand"];
const REDIST: &str = "https://developer.download.nvidia.com/compute/cuda/redist";

/// Directory of the CUDA libraries installed by `basal setup`.
pub fn cuda_lib_dir() -> PathBuf {
    paths::data_dir().join("cuda").join(CUDA_RELEASE).join("lib")
}

pub struct Options {
    pub force: bool,
    pub prefetch: bool,
    pub service: bool,
}

pub fn run(o: &Options) -> Result<()> {
    if cfg!(target_os = "linux") {
        cuda_libraries(o.force)?;
    }
    let cfg = paths::config_file();
    if cfg.exists() {
        crate::note!("configuration {}", cfg.display());
    } else {
        if let Some(d) = cfg.parent() {
            std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        }
        std::fs::write(&cfg, config::template(&[config::DEFAULT_MODEL]))?;
        crate::done!(
            "wrote {} ({}; `basal init --force --model ...` changes it)",
            cfg.display(),
            config::DEFAULT_MODEL.repo
        );
    }
    if o.prefetch {
        let (_, c) = config::user_config()?;
        for m in &c.models {
            config::model_dir(m)?;
        }
    }
    if o.service {
        service()?;
    }
    crate::note!("next: `basal doctor` checks the machine, `basal serve` starts the server");
    Ok(())
}

/// Linux: the CUDA libraries, unless the dynamic loader already finds all of them.
fn cuda_libraries(force: bool) -> Result<()> {
    let dir = cuda_lib_dir();
    let libs = crate::doctor::CUDA_RUNTIME_LIBS;
    let found = |l: &str| {
        // SAFETY: loading a CUDA library runs its initialisers only; nothing is called.
        unsafe { libloading::Library::new(l) }.is_ok() || dir.join(l).exists()
    };
    if !force && libs.iter().all(|l| found(l)) {
        crate::done!("CUDA libraries present ({})", libs.join(", "));
        return Ok(());
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let _guard = rt.enter();
    let client = reqwest::Client::builder().build()?;
    let manifest: serde_json::Value = rt
        .block_on(async {
            client.get(format!("{REDIST}/redistrib_{CUDA_RELEASE}.json")).send().await?.error_for_status()?.json().await
        })
        .context("NVIDIA CUDA redistributable manifest")?;
    let arch = match std::env::consts::ARCH {
        "x86_64" => "linux-x86_64",
        "aarch64" => "linux-sbsa",
        a => bail!("no CUDA libraries for {a}"),
    };
    let tmp = paths::cache_dir().join("cuda-download");
    std::fs::create_dir_all(&tmp)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    for comp in COMPONENTS {
        let e = &manifest[comp][arch];
        let rel = e["relative_path"].as_str().with_context(|| format!("{comp}: not in the manifest for {arch}"))?;
        let sha = e["sha256"].as_str().context("sha256")?;
        let size: u64 = e["size"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0);
        let archive = tmp.join(Path::new(rel).file_name().context("archive name")?);
        crate::progress!(
            "{comp} {} ({:.0} MB) from NVIDIA",
            manifest[comp]["version"].as_str().unwrap_or("?"),
            size as f64 / 1e6
        );
        rt.block_on(download(&client, &format!("{REDIST}/{rel}"), &archive, size))?;
        let got = sha256(&archive)?;
        ensure!(got == sha, "{}: SHA-256 {got}, the manifest has {sha}", archive.display());
        // the shared libraries and their symlinks only (the archives also hold headers and static libraries)
        let out = tmp.join(comp);
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out)?;
        let st = std::process::Command::new("tar")
            .arg("-xJf")
            .arg(&archive)
            .arg("-C")
            .arg(&out)
            .args(["--wildcards", "*/lib/*.so*"])
            .status()
            .context("running tar (xz support needed)")?;
        ensure!(st.success(), "tar failed on {}", archive.display());
        for entry in std::fs::read_dir(&out)? {
            let lib = entry?.path().join("lib");
            for f in std::fs::read_dir(&lib).with_context(|| format!("{}", lib.display()))? {
                let f = f?;
                // libraries and their symlinks; not lib/stubs (link-time stubs, e.g. of libcuda.so)
                if f.file_type()?.is_dir() {
                    continue;
                }
                let dst = dir.join(f.file_name());
                let _ = std::fs::remove_file(&dst);
                std::fs::rename(f.path(), &dst)?;
            }
        }
        std::fs::remove_dir_all(&out)?;
        std::fs::remove_file(&archive)?;
    }
    for l in libs {
        ensure!(dir.join(l).exists() || found(l), "{l} missing after the installation");
    }
    crate::done!("CUDA {CUDA_RELEASE} libraries in {}", dir.display());
    Ok(())
}

async fn download(client: &reqwest::Client, url: &str, to: &Path, size: u64) -> Result<()> {
    use std::io::Write;
    let mut resp = client.get(url).send().await?.error_for_status()?;
    let mut f = std::fs::File::create(to).with_context(|| format!("creating {}", to.display()))?;
    let (mut done, mut last) = (0u64, std::time::Instant::now());
    while let Some(chunk) = resp.chunk().await? {
        f.write_all(&chunk)?;
        done += chunk.len() as u64;
        if last.elapsed().as_secs() >= 15 && size > 0 {
            crate::progress!("  {:.0} / {:.0} MB", done as f64 / 1e6, size as f64 / 1e6);
            last = std::time::Instant::now();
        }
    }
    Ok(())
}

fn sha256(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// A user service running `basal serve` with the user configuration.
fn service() -> Result<()> {
    let exe = std::env::current_exe()?;
    let home = std::env::var_os("HOME").map(PathBuf::from).context("HOME")?;
    if cfg!(target_os = "macos") {
        let plist = home.join("Library/LaunchAgents/tech.itsol.basal.plist");
        let log = paths::cache_dir().join("serve.log");
        std::fs::create_dir_all(plist.parent().unwrap())?;
        std::fs::create_dir_all(paths::cache_dir())?;
        std::fs::write(
            &plist,
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
                 \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\"><dict>\n  \
                 <key>Label</key><string>tech.itsol.basal</string>\n  <key>ProgramArguments</key><array>\
                 <string>{}</string><string>serve</string></array>\n  <key>RunAtLoad</key><true/>\n  \
                 <key>KeepAlive</key><true/>\n  <key>StandardOutPath</key><string>{log}</string>\n  \
                 <key>StandardErrorPath</key><string>{log}</string>\n</dict></plist>\n",
                exe.display(),
                log = log.display()
            ),
        )?;
        crate::done!("wrote {}; start: `launchctl load -w {0}`; log: {}", plist.display(), log.display());
    } else {
        let unit = home.join(".config/systemd/user/basal.service");
        std::fs::create_dir_all(unit.parent().unwrap())?;
        std::fs::write(
            &unit,
            format!(
                "[Unit]\nDescription=basal-rs server (basal serve)\nAfter=network-online.target\n\n[Service]\n\
                 ExecStart={} serve\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n",
                exe.display()
            ),
        )?;
        crate::done!(
            "wrote {}; start: `systemctl --user daemon-reload && systemctl --user enable --now basal`; log: \
             `journalctl --user -u basal -f` (without a login session: `loginctl enable-linger $USER`)",
            unit.display()
        );
    }
    Ok(())
}
