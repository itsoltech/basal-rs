//! `basal uninstall`: remove what basal put on this machine, after listing it (and asking, unless `--yes`).
//!
//! The user service (`basal setup --service`, stopped first), the configuration, the caches (GEMM tables, downloads,
//! update check), the data (CUDA libraries of `basal setup`), with `--models` the basal models in the Hugging Face
//! cache (other repositories there are kept), and last the binaries of a package installation (`install.sh`). A
//! Homebrew installation is removed by `brew uninstall`, which is printed instead. A configuration file given by
//! `BASAL_CONFIG` outside these directories is kept.

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::paths;
use crate::term::{paint, path, Stream::Stderr as E, Style};

pub struct Options {
    pub models: bool,
    pub yes: bool,
    pub dry_run: bool,
}

fn add(targets: &mut Vec<PathBuf>, p: PathBuf) {
    if p.exists() && !targets.contains(&p) {
        targets.push(p);
    }
}

/// Bytes under `p` (files and symlinks themselves, not their targets).
fn size(p: &Path) -> u64 {
    let Ok(m) = std::fs::symlink_metadata(p) else { return 0 };
    if m.is_dir() {
        std::fs::read_dir(p).map(|d| d.flatten().map(|e| size(&e.path())).sum()).unwrap_or(0)
    } else {
        m.len()
    }
}

fn human(b: u64) -> String {
    match b {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / 1e9),
        b if b >= 1 << 20 => format!("{:.0} MB", b as f64 / 1e6),
        b if b >= 1000 => format!("{:.0} kB", b as f64 / 1e3),
        _ => "< 1 kB".to_string(),
    }
}

fn run_quiet(cmd: &str, args: &[&str]) {
    let _ = std::process::Command::new(cmd).args(args).output();
}

pub fn run(o: &Options) -> Result<()> {
    let home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let mut targets: Vec<PathBuf> = Vec::new();
    // the service first (stopped before its files go)
    let launchd = home.join("Library/LaunchAgents/tech.itsol.basal.plist");
    let systemd = home.join(".config/systemd/user/basal.service");
    add(&mut targets, launchd.clone());
    add(&mut targets, systemd.clone());
    // configuration, caches, data: the basal directories; with BASAL_HOME only what basal writes there
    if std::env::var_os("BASAL_HOME").is_some_and(|v| !v.is_empty()) {
        add(&mut targets, paths::config_dir().join("serve.yml"));
        add(&mut targets, paths::cache_dir());
        add(&mut targets, paths::data_dir());
    } else {
        add(&mut targets, paths::config_dir());
        add(&mut targets, paths::cache_dir());
        add(&mut targets, paths::data_dir());
    }
    let config = paths::config_file();
    let config_kept = config.exists() && !targets.iter().any(|t| config.starts_with(t));
    if o.models {
        for dir in hf_model_dirs() {
            add(&mut targets, dir);
        }
    }
    // binaries of a package installation (PREFIX/bin/basal, PREFIX/libexec/basal, PREFIX/share/doc/basal)
    let exe = std::env::current_exe()?.canonicalize()?;
    let s = exe.to_string_lossy().to_string();
    let homebrew = s.contains("/Cellar/");
    let source = s.contains("/target/");
    if !homebrew && !source && exe.parent().is_some_and(|b| b.ends_with("bin")) {
        let prefix = exe.parent().and_then(Path::parent).unwrap().to_path_buf();
        add(&mut targets, prefix.join("libexec/basal"));
        add(&mut targets, prefix.join("share/doc/basal"));
        add(&mut targets, exe.clone());
    }

    if targets.is_empty() {
        crate::note!("nothing to remove");
    } else {
        eprintln!("{}", paint(E, Style::Bold, "basal uninstall removes:"));
        let mut total = 0;
        for t in &targets {
            let b = size(t);
            total += b;
            eprintln!("  {} {:>8}  {}", paint(E, Style::Red, "-"), human(b), path(t));
        }
        eprintln!("  {:>10}  total", human(total));
    }
    if config_kept {
        crate::note!("kept {} (BASAL_CONFIG, outside the basal directories)", path(&config));
    }
    if !o.models {
        let n = hf_model_dirs();
        if !n.is_empty() {
            let b: u64 = n.iter().map(|d| size(d)).sum();
            crate::note!(
                "kept {} basal model(s) in the Hugging Face cache ({}); --models removes them",
                n.len(),
                human(b)
            );
        }
    }
    if homebrew {
        crate::note!(
            "installed with Homebrew: brew services stop basal-rs; brew uninstall basal-rs; brew untap itsoltech/tap"
        );
    }
    if source {
        crate::note!("built from the repository: the binary stays in target/");
    }
    if targets.is_empty() || o.dry_run {
        if o.dry_run {
            crate::done!("dry run, nothing removed");
        }
        return Ok(());
    }
    if !o.yes {
        if !std::io::stdin().is_terminal() {
            bail!("not a terminal: confirm with --yes");
        }
        eprint!("Remove these? [y/N] ");
        std::io::stderr().flush()?;
        let mut a = String::new();
        std::io::stdin().lock().read_line(&mut a)?;
        if !matches!(a.trim(), "y" | "Y" | "yes") {
            crate::done!("nothing removed");
            return Ok(());
        }
    }
    if targets.contains(&launchd) {
        run_quiet("launchctl", &["unload", "-w", &launchd.to_string_lossy()]);
    }
    if targets.contains(&systemd) {
        run_quiet("systemctl", &["--user", "disable", "--now", "basal"]);
    }
    for t in &targets {
        let r =
            if std::fs::symlink_metadata(t)?.is_dir() { std::fs::remove_dir_all(t) } else { std::fs::remove_file(t) };
        match r {
            Ok(()) => eprintln!("  {} {}", paint(E, Style::Green, "✓"), path(t)),
            Err(e) => eprintln!("  {} {}: {e}", paint(E, Style::Red, "✗"), path(t)),
        }
    }
    if targets.contains(&systemd) {
        run_quiet("systemctl", &["--user", "daemon-reload"]);
    }
    crate::done!("removed");
    Ok(())
}

/// The basal models in the Hugging Face cache (repositories Remek/basal-*).
fn hf_model_dirs() -> Vec<PathBuf> {
    let hub = match (std::env::var_os("HF_HUB_CACHE"), std::env::var_os("HF_HOME")) {
        (Some(c), _) => PathBuf::from(c),
        (None, Some(h)) => PathBuf::from(h).join("hub"),
        (None, None) => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub"),
    };
    let mut v: Vec<PathBuf> = std::fs::read_dir(&hub)
        .map(|d| {
            d.flatten()
                .map(|e| e.path())
                .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("models--Remek--basal-")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}
