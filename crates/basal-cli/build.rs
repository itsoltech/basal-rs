//! BASAL_GIT_SHA for `basal --version`: the environment variable of the same name (release builds, container image)
//! or the commit of the checkout; "unknown" without either. gemm_tables.rs: the batch-invariant GEMM tables of
//! gemm-tables/*.json compiled into the CUDA binary (config::gemm_table uses one for a matching GPU instead of a
//! search), checked and reduced by [`gemm_tables`].

fn main() {
    println!("cargo:rerun-if-env-changed=BASAL_GIT_SHA");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for f in [".git/HEAD", ".git/index"] {
        if root.join(f).exists() {
            println!("cargo:rerun-if-changed={}", root.join(f).display());
        }
    }
    let sha =
        std::env::var("BASAL_GIT_SHA").ok().filter(|s| !s.is_empty()).map(|s| s.chars().take(8).collect()).or_else(
            || {
                let out = std::process::Command::new("git")
                    .args(["rev-parse", "--short=8", "HEAD"])
                    .current_dir(&root)
                    .output()
                    .ok()?;
                out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
            },
        );
    println!("cargo:rustc-env=BASAL_GIT_SHA={}", sha.unwrap_or_else(|| "unknown".into()));

    gemm_tables();
}

/// Limits of the tables compiled into the binary: the growth of the binary stays bounded by GPUs x models.
const MAX_TABLE_BYTES: usize = 32 * 1024;
const MAX_TOTAL_BYTES: usize = 1024 * 1024;

/// gemm-tables/*.json -> `$OUT_DIR/gemm_tables.rs`, CUDA builds only (other builds get none). Each table is checked
/// and reduced to what `config::gemm_table` and `basal_gpu::check_gemm_table` read (no timings, notes or sources).
/// A table breaking a rule fails the build, so CI (Clippy with CUDA) stops it in the pull request:
/// - batch-invariant, f16, of the current search version (`INVARIANT_SEARCH_VERSION` of basal-gpu) and of the
///   cuBLASLt version of the CUDA toolkit (`cublas_api.h`), so that a toolkit change requires new tables;
/// - file name `<gpu>--<model>--f16--cublaslt<version>.json`, one table per GPU and weight shapes;
/// - at most MAX_TABLE_BYTES per reduced table and MAX_TOTAL_BYTES in all.
fn gemm_tables() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("gemm-tables");
    let gpu_src = manifest.join("../basal-gpu/src/cublaslt.rs");
    println!("cargo:rerun-if-changed={}", dir.display());
    println!("cargo:rerun-if-changed={}", gpu_src.display());
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let mut out = String::from(
        "/// (file name, reduced table JSON) of gemm-tables/*.json\npub static TABLES: &[(&str, &str)] = &[\n",
    );
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let search: u64 = std::fs::read_to_string(&gpu_src)
            .ok()
            .and_then(|s| {
                s.lines().find_map(|l| {
                    l.trim().strip_prefix("pub const INVARIANT_SEARCH_VERSION: u64 = ")?.strip_suffix(';')?.parse().ok()
                })
            })
            .expect("INVARIANT_SEARCH_VERSION in basal-gpu/src/cublaslt.rs");
        let cublaslt = toolkit_cublaslt_version();
        if cublaslt.is_none() {
            println!("cargo:warning=cublas_api.h not found: the cuBLASLt version of gemm-tables/*.json is not checked");
        }
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|x| x == "json"))
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        let (mut total, mut seen) = (0, Vec::new());
        for f in &files {
            println!("cargo:rerun-if-changed={}", f.display());
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let table = reduce(f, &name, search, cublaslt).unwrap_or_else(|e| panic!("gemm-tables/{name}: {e}"));
            let key = (table["gpu"].clone(), table["shapes"].clone());
            assert!(!seen.contains(&key), "gemm-tables/{name}: a second table for this GPU and these weight shapes");
            seen.push(key);
            let text = serde_json::to_string(&table).unwrap();
            assert!(
                text.len() <= MAX_TABLE_BYTES,
                "gemm-tables/{name}: {} bytes reduced, more than {MAX_TABLE_BYTES}",
                text.len()
            );
            total += text.len();
            out += &format!("    ({name:?}, {text:?}),\n");
        }
        assert!(total <= MAX_TOTAL_BYTES, "gemm-tables: {total} bytes reduced, more than {MAX_TOTAL_BYTES}");
    }
    out += "];\n";
    let dst = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("gemm_tables.rs");
    std::fs::write(dst, out).unwrap();
}

/// The checked table of `f`, with the fields the runtime reads.
fn reduce(f: &std::path::Path, name: &str, search: u64, cublaslt: Option<u64>) -> Result<serde_json::Value, String> {
    use serde_json::{json, Value};
    let v: Value =
        serde_json::from_str(&std::fs::read_to_string(f).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let gpu = v["gpu"].as_str().ok_or("no gpu")?;
    let model = v["model"].as_str().ok_or("no model")?;
    let version = v["cublaslt_version"].as_u64().ok_or("no cublaslt_version")?;
    if v["invariant"].as_bool() != Some(true) {
        return Err("not a batch-invariant table".into());
    }
    if v["dtype"].as_str() != Some("f16") {
        return Err(format!("dtype {}, only f16 tables are built in", v["dtype"]));
    }
    if v["gemm_search_version"].as_u64() != Some(search) {
        return Err(format!("search version {}, this build searches with {search}", v["gemm_search_version"]));
    }
    if let Some(c) = cublaslt.filter(|&c| c != version) {
        return Err(format!("cuBLASLt {version}, the CUDA toolkit has {c}"));
    }
    let slug: String = gpu
        .to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    let expected = format!("{slug}--{model}--f16--cublaslt{version}.json");
    if name != expected {
        return Err(format!("the file name must be {expected}"));
    }
    let shapes = v["shapes"].as_array().filter(|s| !s.is_empty()).ok_or("no shapes")?;
    let mut entries = Vec::new();
    for e in v["entries"].as_array().filter(|e| !e.is_empty()).ok_or("no entries")? {
        let (n, k) = (&e["n"], &e["k"]);
        if !shapes.iter().any(|s| s[0] == *n && s[1] == *k) {
            return Err(format!("entry for {n}x{k}, not one of the shapes"));
        }
        if e["dtype"].as_str() != Some("f16") || e["algo"].as_array().is_none_or(|a| a.len() != 8) {
            return Err("entry without f16 dtype or 8 algo words".into());
        }
        entries.push(json!({"m_class": e["m_class"], "n": n, "k": k, "dtype": "f16", "algo": e["algo"]}));
    }
    Ok(json!({
        "gpu": gpu, "model": model, "cublaslt_version": version, "dtype": "f16", "invariant": true,
        "gemm_search_version": search, "shapes": shapes, "entries": entries,
    }))
}

/// cuBLAS version of the CUDA toolkit headers (`CUBLAS_VER_*` of cublas_api.h, as `cublasLtGetVersion` reports it).
fn toolkit_cublaslt_version() -> Option<u64> {
    let roots =
        ["CUDA_PATH", "CUDA_HOME"].iter().filter_map(|v| std::env::var(v).ok()).chain(["/usr/local/cuda".to_string()]);
    let text = roots.map(|r| std::path::Path::new(&r).join("include/cublas_api.h")).find_map(|p| {
        println!("cargo:rerun-if-changed={}", p.display());
        std::fs::read_to_string(p).ok()
    })?;
    let def = |n: &str| -> Option<u64> {
        text.lines().find_map(|l| l.trim().strip_prefix(&format!("#define {n} "))?.trim().parse().ok())
    };
    Some(def("CUBLAS_VER_MAJOR")? * 10000 + def("CUBLAS_VER_MINOR")? * 100 + def("CUBLAS_VER_PATCH")?)
}
