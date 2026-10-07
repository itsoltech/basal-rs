//! Per-user locations of caches (GEMM tables, downloads) and data (CUDA libraries of
//! `basal setup`), and the CUDA build of the Linux package.
//!
//! | | macOS | Linux | `BASAL_HOME=DIR` |
//! |---|---|---|---|
//! | cache | `~/Library/Caches/basal` | `$XDG_CACHE_HOME/basal`, `~/.cache/basal` | `DIR/.cache` |
//! | data | `~/Library/Application Support/basal` | `$XDG_DATA_HOME/basal`, `~/.local/share/basal` | `DIR/.local` |
//!
//! The container image sets `BASAL_HOME=/data` (the volume), so its GEMM tables stay in `/data/.cache/gemm`.

use std::path::PathBuf;

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

fn basal_home() -> Option<PathBuf> {
    std::env::var_os("BASAL_HOME").filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// `$VAR/basal` when `VAR` is set to an absolute path (XDG), otherwise `~/<fallback>/basal`.
fn xdg(var: &str, fallback: &str) -> PathBuf {
    match std::env::var_os(var).map(PathBuf::from) {
        Some(p) if p.is_absolute() => p.join("basal"),
        _ => home().join(fallback).join("basal"),
    }
}

pub fn cache_dir() -> PathBuf {
    if let Some(h) = basal_home() {
        return h.join(".cache");
    }
    if cfg!(target_os = "macos") {
        home().join("Library/Caches/basal")
    } else {
        xdg("XDG_CACHE_HOME", ".cache")
    }
}

pub fn data_dir() -> PathBuf {
    if let Some(h) = basal_home() {
        return h.join(".local");
    }
    if cfg!(target_os = "macos") {
        home().join("Library/Application Support/basal")
    } else {
        xdg("XDG_DATA_HOME", ".local/share")
    }
}

/// The CUDA build of the Linux package next to this binary (`PREFIX/libexec/basal/basal-cuda` for
/// `PREFIX/bin/basal`), when present.
pub fn cuda_binary() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    let p = exe.parent()?.parent()?.join("libexec/basal/basal-cuda");
    p.exists().then_some(p)
}

/// Name of the configuration file `basal serve` reads from the working directory.
pub const CONFIG_NAME: &str = "basal-serve.yml";

/// The configuration file of `basal serve` without `--config` / `--model`: `BASAL_CONFIG`, else `basal-serve.yml` in
/// the working directory when it exists.
pub fn config_file() -> Option<PathBuf> {
    match std::env::var_os("BASAL_CONFIG").filter(|v| !v.is_empty()) {
        Some(p) => Some(PathBuf::from(p)),
        None => Some(PathBuf::from(CONFIG_NAME)).filter(|p| p.exists()),
    }
}
