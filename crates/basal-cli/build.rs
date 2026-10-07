//! BASAL_GIT_SHA for `basal --version`: the environment variable of the same name (release builds, container image)
//! or the commit of the checkout; "unknown" without either.

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
}
