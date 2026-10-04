//! Reading the reference exported by tools/reference/export_reference.py.

use std::path::Path;

use anyhow::{Context, Result};
use basal_core::prompt::lang_of;
use basal_core::request::{Item, QType};
use serde_json::Value;

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    s.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect()).unwrap_or_default()
}

pub fn str_of<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// A basal-bench item `{state, question, options, type?, gold?}` as a readout item (keys = option indices).
pub fn bench_item(rec: &Value) -> Result<Item> {
    let state = str_of(rec, "state").to_string();
    let question = str_of(rec, "question").to_string();
    let options: Vec<String> =
        rec["options"].as_array().context("options")?.iter().map(|o| o.as_str().unwrap_or("").to_string()).collect();
    let kind = QType::parse(rec.get("type").and_then(Value::as_str).unwrap_or("choice")).context("type")?;
    Ok(Item {
        name: rec.get("id").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| rec["index"].to_string()),
        kind,
        keys: (0..options.len()).map(|i| i.to_string()).collect(),
        legend: if kind == QType::Score { options.iter().map(|o| Value::String(o.clone())).collect() } else { vec![] },
        lang: lang_of(&format!("{state}{question}")),
        options,
        state,
        question,
        ext: Default::default(),
        evidence: false,
        uncalibrated: false,
    })
}
