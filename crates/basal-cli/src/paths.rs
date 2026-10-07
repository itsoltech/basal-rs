//! Per-user locations of the server configuration and caches (GEMM tables).
//!
//! | | macOS | Linux | `BASAL_HOME=DIR` |
//! |---|---|---|---|
//! | configuration | `~/Library/Application Support/basal` | `$XDG_CONFIG_HOME/basal`, `~/.config/basal` | `DIR` |
//! | cache | `~/Library/Caches/basal` | `$XDG_CACHE_HOME/basal`, `~/.cache/basal` | `DIR/.cache` |
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

pub fn config_dir() -> PathBuf {
    if let Some(h) = basal_home() {
        return h;
    }
    if cfg!(target_os = "macos") {
        home().join("Library/Application Support/basal")
    } else {
        xdg("XDG_CONFIG_HOME", ".config")
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

/// Server configuration of `basal serve` without `--config` / `--model`: `BASAL_CONFIG`, else `serve.yml` in
/// [`config_dir`].
pub fn config_file() -> PathBuf {
    match std::env::var_os("BASAL_CONFIG").filter(|v| !v.is_empty()) {
        Some(p) => PathBuf::from(p),
        None => config_dir().join("serve.yml"),
    }
}
