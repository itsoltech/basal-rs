//! Normalized identity of a GEMM table, shared by build.rs (bundled tables) and the runtime (`gemm_artifact`).
//!
//! The normalized form keeps exactly what decides the GEMM algorithms at run time and drops the rest (model name,
//! the basal build that made it, timings, notes, the list of searched shapes). It is UTF-8 text, one field per
//! line, every line ended by `\n`:
//!
//! ```text
//! basal-gemm-artifact.1
//! gpu <the GPU name as a JSON string>
//! cublaslt_version <decimal>
//! dtype <the table's dtype as a JSON string>
//! invariant <true|false>
//! gemm_search_version <decimal; 1 when absent>
//! entry <dtype> <n> <k> <m_class> <algo word 0> ... <algo word 7>
//! ```
//!
//! with one `entry` line per table entry, sorted by (dtype, n, k, m_class); integers in decimal without leading
//! zeros, entry dtypes `f16` or `bf16`. Two entries with the same (dtype, n, k, m_class) are refused. The digest of
//! a table is the lowercase hex SHA-256 of this text: tables with equal digests choose the same algorithms.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// First line of the normalized form.
pub const ARTIFACT_SCHEMA: &str = "basal-gemm-artifact.1";

/// The normalized text of table `v`.
pub fn normalized(v: &Value) -> Result<String, String> {
    let gpu = v["gpu"].as_str().ok_or("no gpu")?;
    let cublaslt = v["cublaslt_version"].as_u64().ok_or("no cublaslt_version")?;
    let dtype = v["dtype"].as_str().ok_or("no dtype")?;
    let invariant = v["invariant"].as_bool().unwrap_or(false);
    let search = match &v["gemm_search_version"] {
        Value::Null => 1,
        s => s.as_u64().ok_or("gemm_search_version is not an integer")?,
    };
    let mut entries = Vec::new();
    for e in v["entries"].as_array().ok_or("no entries")? {
        let field = |name: &str| e[name].as_u64().ok_or_else(|| format!("entry without integer {name}"));
        let entry_dtype = match e["dtype"].as_str() {
            Some(d @ ("f16" | "bf16")) => d,
            d => return Err(format!("entry dtype {d:?}")),
        };
        let algo = e["algo"].as_array().filter(|a| a.len() == 8).ok_or("entry algo must have 8 words")?;
        let words =
            algo.iter().map(|w| w.as_u64().ok_or("algo word is not an integer")).collect::<Result<Vec<_>, _>>()?;
        entries.push((entry_dtype, field("n")?, field("k")?, field("m_class")?, words));
    }
    entries.sort();
    if let Some(w) =
        entries.windows(2).find(|w| w[0].0 == w[1].0 && w[0].1 == w[1].1 && w[0].2 == w[1].2 && w[0].3 == w[1].3)
    {
        return Err(format!("two entries for {} n={} k={} m_class={}", w[0].0, w[0].1, w[0].2, w[0].3));
    }
    let quote = |s: &str| serde_json::to_string(s).map_err(|e| e.to_string());
    let mut out = format!(
        "{ARTIFACT_SCHEMA}\ngpu {}\ncublaslt_version {cublaslt}\ndtype {}\ninvariant {invariant}\ngemm_search_version {search}\n",
        quote(gpu)?,
        quote(dtype)?
    );
    for (d, n, k, m, words) in &entries {
        out.push_str(&format!("entry {d} {n} {k} {m}"));
        for w in words {
            out.push_str(&format!(" {w}"));
        }
        out.push('\n');
    }
    Ok(out)
}

/// Lowercase hex SHA-256 of the normalized text of table `v`.
pub fn digest(v: &Value) -> Result<String, String> {
    let hash = Sha256::digest(normalized(v)?.as_bytes());
    Ok(hash.iter().map(|b| format!("{b:02x}")).collect())
}
