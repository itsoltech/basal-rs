//! BASAL_GIT_SHA for `basal --version`: the environment variable of the same name (release builds, container image)
//! or the commit of the checkout; "unknown" without either. gemm_tables.rs: the batch-invariant GEMM tables of
//! gemm-tables/*.json, compiled into the binary (config::gemm_table uses one for a matching GPU instead of a search).

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

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("gemm-tables");
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect()
        })
        .unwrap_or_default();
    files.sort();
    let mut out =
        String::from("/// (file name, table JSON) of gemm-tables/*.json\npub static TABLES: &[(&str, &str)] = &[\n");
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
        let name = f.file_name().unwrap().to_string_lossy();
        out += &format!("    ({name:?}, include_str!({:?})),\n", f.display().to_string());
    }
    out += "];\n";
    let dst = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("gemm_tables.rs");
    std::fs::write(dst, out).unwrap();
}
