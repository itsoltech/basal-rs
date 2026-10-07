//! Releases on GitHub: `basal update` and the notice of a newer version (`basal serve`, `basal doctor`).
//!
//! A release has one package per target (`basal-<version>-<target>.tar.gz`, see tools/release/) and `SHA256SUMS`.
//! `basal update` replaces the binaries of the package installation this binary belongs to (`install.sh`); a Homebrew
//! or container installation is updated by its own tool. `BASAL_RELEASE_URL` (a URL or a local directory with the
//! packages and SHA256SUMS) replaces GitHub Releases, for testing. `BASAL_NO_UPDATE_CHECK=1` turns the notice off.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, ensure, Context, Result};

use crate::paths;

const REPO: &str = "itsoltech/basal-rs";
/// Interval between two automatic checks.
const CHECK_EVERY: Duration = Duration::from_secs(24 * 3600);

fn target() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("linux", "x86_64") => Some("x86_64-linux"),
        _ => None,
    }
}

/// `X.Y.Z[-pre]` -> comparable key (a pre-release sorts before its release).
fn version_key(v: &str) -> Option<(u64, u64, u64, bool, String)> {
    let v = v.trim_start_matches('v');
    let (core, pre) = match v.split_once('-') {
        Some((c, p)) => (c, Some(p)),
        None => (v, None),
    };
    let mut it = core.split('.').map(|x| x.parse::<u64>().ok());
    Some((it.next()??, it.next()??, it.next()??, pre.is_none(), pre.unwrap_or("").to_string()))
}

pub fn newer(a: &str, b: &str) -> bool {
    matches!((version_key(a), version_key(b)), (Some(x), Some(y)) if x > y)
}

fn client() -> Result<(tokio::runtime::Runtime, reqwest::Client)> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let c = {
        let _g = rt.enter();
        reqwest::Client::builder()
            .user_agent(concat!("basal-rs/", env!("CARGO_PKG_VERSION")))
            .timeout(Duration::from_secs(600))
            .build()?
    };
    Ok((rt, c))
}

fn source() -> Option<String> {
    std::env::var("BASAL_RELEASE_URL").ok().filter(|s| !s.is_empty())
}

/// Bytes of a file of release `version` (GitHub Releases or `BASAL_RELEASE_URL`).
fn fetch(rt: &tokio::runtime::Runtime, c: &reqwest::Client, version: &str, file: &str) -> Result<Vec<u8>> {
    let url = match source() {
        Some(s) if !s.starts_with("http://") && !s.starts_with("https://") => {
            return std::fs::read(Path::new(&s).join(file)).with_context(|| format!("{s}/{file}"));
        }
        Some(s) => format!("{}/{file}", s.trim_end_matches('/')),
        None => format!("https://github.com/{REPO}/releases/download/v{version}/{file}"),
    };
    rt.block_on(async {
        Ok::<_, reqwest::Error>(c.get(&url).send().await?.error_for_status()?.bytes().await?.to_vec())
    })
    .with_context(|| url)
}

/// The newest release (version without `v`), or None when there is none.
pub fn latest() -> Result<Option<String>> {
    let (rt, c) = client()?;
    if source().is_some() {
        // the newest package of this target listed in SHA256SUMS
        let sums = String::from_utf8(fetch(&rt, &c, "", "SHA256SUMS")?)?;
        let t = target().context("no package for this system")?;
        let mut vs: Vec<String> = sums
            .lines()
            .filter_map(|l| l.split_whitespace().nth(1))
            .filter_map(|f| f.trim_start_matches('*').strip_prefix("basal-")?.strip_suffix(&format!("-{t}.tar.gz")))
            .map(str::to_string)
            .collect();
        vs.sort_by_key(|v| version_key(v));
        return Ok(vs.pop());
    }
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let resp = rt.block_on(c.get(&url).timeout(Duration::from_secs(10)).send())?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    let v: serde_json::Value = rt.block_on(resp.error_for_status()?.json())?;
    Ok(v["tag_name"].as_str().map(|t| t.trim_start_matches('v').to_string()))
}

/// How this binary was installed.
enum Install {
    /// `install.sh` package: the prefix (PREFIX/bin/basal)
    Package(PathBuf),
    Homebrew,
    Container,
    /// a build of the repository (target/...)
    Source,
}

fn install() -> Result<Install> {
    let exe = std::env::current_exe()?.canonicalize()?;
    let s = exe.to_string_lossy();
    if s.contains("/Cellar/") {
        return Ok(Install::Homebrew);
    }
    if s == "/usr/local/bin/basal" && Path::new("/.dockerenv").exists() {
        return Ok(Install::Container);
    }
    if s.contains("/target/") {
        return Ok(Install::Source);
    }
    let prefix = exe.parent().and_then(Path::parent).context("installation prefix")?;
    ensure!(exe.parent().is_some_and(|b| b.ends_with("bin")), "{} is not PREFIX/bin/basal", exe.display());
    Ok(Install::Package(prefix.to_path_buf()))
}

/// `basal update [--version X] [--check]`.
pub fn run(version: Option<String>, check: bool) -> Result<()> {
    let current = env!("CARGO_PKG_VERSION");
    let want = match version {
        Some(v) => v.trim_start_matches('v').to_string(),
        None => match latest()? {
            Some(v) => v,
            None => bail!("no release published yet"),
        },
    };
    eprintln!("basal: installed {current}, {} {want}", if check { "available" } else { "target" });
    if check {
        if newer(&want, current) {
            eprintln!("basal: update available");
        }
        return Ok(());
    }
    match install()? {
        Install::Homebrew => {
            eprintln!("basal: installed with Homebrew: brew update && brew upgrade basal-rs");
            return Ok(());
        }
        Install::Container => {
            eprintln!("basal: container image: docker compose pull && docker compose up -d");
            return Ok(());
        }
        Install::Source => bail!("built from the repository: git pull && cargo build --release"),
        Install::Package(prefix) => {
            if want == current {
                eprintln!("basal: up to date");
                return Ok(());
            }
            replace(&prefix, &want)?;
        }
    }
    Ok(())
}

/// Download, check and install package `version` into `prefix`.
fn replace(prefix: &Path, version: &str) -> Result<()> {
    let t = target().context("no package for this system")?;
    let name = format!("basal-{version}-{t}");
    let file = format!("{name}.tar.gz");
    let (rt, c) = client()?;
    eprintln!("basal: downloading {file}");
    let sums = String::from_utf8(fetch(&rt, &c, version, "SHA256SUMS")?)?;
    let want = sums
        .lines()
        .find_map(|l| {
            let mut f = l.split_whitespace();
            let (h, n) = (f.next()?, f.next()?);
            (n.trim_start_matches('*') == file).then(|| h.to_string())
        })
        .with_context(|| format!("{file} is not in SHA256SUMS"))?;
    let bytes = fetch(&rt, &c, version, &file)?;
    let got = {
        use sha2::{Digest, Sha256};
        Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>()
    };
    ensure!(got == want, "{file}: SHA-256 {got}, expected {want}");
    let tmp = paths::cache_dir().join("update");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    std::fs::write(tmp.join(&file), &bytes)?;
    let st = std::process::Command::new("tar").arg("-xzf").arg(tmp.join(&file)).arg("-C").arg(&tmp).status()?;
    ensure!(st.success(), "tar failed on {file}");
    let dir = tmp.join(&name);
    // copy next to the old file, then rename over it (a running server keeps its binary)
    for rel in ["bin/basal", "libexec/basal/basal-cuda"] {
        let src = dir.join(rel);
        if !src.exists() {
            continue;
        }
        let dst = prefix.join(rel);
        std::fs::create_dir_all(dst.parent().unwrap())?;
        let new = dst.with_extension("new");
        std::fs::copy(&src, &new).with_context(|| format!("writing {}", new.display()))?;
        std::fs::rename(&new, &dst)?;
    }
    std::fs::remove_dir_all(&tmp)?;
    eprintln!("basal: updated to {version} in {}; restart a running server", prefix.display());
    Ok(())
}

/// Notice of a newer release, at most once a day (`BASAL_NO_UPDATE_CHECK=1`: never); network errors are ignored.
/// Returns the newer version, if any.
pub fn notice() -> Option<String> {
    if std::env::var("BASAL_NO_UPDATE_CHECK").is_ok_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
    let stamp = paths::cache_dir().join("update-check");
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    let (last, known) = std::fs::read_to_string(&stamp)
        .ok()
        .and_then(|s| {
            let (t, v) = s.trim().split_once(' ').unwrap_or((s.trim(), ""));
            Some((t.parse::<u64>().ok()?, v.to_string()))
        })
        .unwrap_or((0, String::new()));
    let latest = if now.saturating_sub(last) < CHECK_EVERY.as_secs() {
        known
    } else {
        let v = latest().ok().flatten().unwrap_or_default();
        let _ = std::fs::create_dir_all(paths::cache_dir());
        let _ = std::fs::write(&stamp, format!("{now} {v}\n"));
        v
    };
    newer(&latest, env!("CARGO_PKG_VERSION")).then_some(latest)
}

/// How to update this installation.
pub fn how() -> &'static str {
    match install() {
        Ok(Install::Homebrew) => "brew upgrade basal-rs",
        Ok(Install::Container) => "docker compose pull",
        Ok(Install::Source) => "git pull && cargo build --release",
        _ => "basal update",
    }
}
