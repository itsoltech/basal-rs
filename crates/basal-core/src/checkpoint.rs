//! Discover one dense checkpoint or the files named by a bounded Hugging Face safetensors index.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Checkpoint files in deterministic filename order. Index entries must be plain safetensors filenames.
/// HF snapshot symlinks remain usable; an index cannot introduce a path outside the supplied directory.
pub fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let index = dir.join("model.safetensors.index.json");
    let file = std::fs::File::open(&index).with_context(|| format!("opening {}", index.display()))?;
    ensure!(file.metadata()?.len() <= 16 * 1024 * 1024, "checkpoint index exceeds 16 MiB");
    let value: Value = serde_json::from_reader(file).context("checkpoint index JSON")?;
    let weights = value
        .get("weight_map")
        .and_then(Value::as_object)
        .filter(|m| !m.is_empty())
        .context("checkpoint index needs a nonempty weight_map")?;
    let mut names = std::collections::BTreeSet::new();
    for value in weights.values() {
        let name = value.as_str().context("checkpoint shard name must be a string")?;
        ensure!(
            name.ends_with(".safetensors")
                && !name.starts_with('.')
                && name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
            "checkpoint shard must be a plain safetensors filename"
        );
        names.insert(name.to_string());
        ensure!(names.len() <= 64, "at most 64 checkpoint shards");
    }
    names
        .into_iter()
        .map(|name| {
            let path = dir.join(name);
            ensure!(path.is_file(), "checkpoint shard {} missing", path.display());
            Ok(path)
        })
        .collect()
}

/// Actual bytes on disk, including every shard's header; suitable for memory/download diagnostics.
pub fn bytes(dir: &Path) -> Result<u64> {
    files(dir)?
        .iter()
        .try_fold(0u64, |sum, p| sum.checked_add(p.metadata()?.len()).context("checkpoint byte count overflow"))
}
