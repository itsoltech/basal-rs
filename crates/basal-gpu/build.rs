//! With the `cuda` feature: compile src/kernels.cu to PTX with nvcc, once per GPU architecture of CUDA_COMPUTE_CAPS
//! (comma-separated, default 80,89,90: A100 / RTX 30xx, RTX 6000 Ada / L40S / RTX 40xx, H100), else only for
//! CUDA_COMPUTE_CAP when that is set (candle's own kernels are compiled for CUDA_COMPUTE_CAP, which candle needs
//! without a GPU in the build machine; their PTX for 8.0 runs on newer GPUs too). At run time the PTX of the highest architecture not above the GPU's is loaded
//! (`fused::ptx`); a GPU older than every one of them is refused.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAPS");
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let caps: Vec<u32> = match (std::env::var("CUDA_COMPUTE_CAPS"), std::env::var("CUDA_COMPUTE_CAP")) {
        (Ok(cs), _) if !cs.is_empty() => cs.split(',').map(|c| c.trim().parse().expect("CUDA_COMPUTE_CAPS")).collect(),
        (_, Ok(c)) if !c.is_empty() => vec![c.trim().parse().expect("CUDA_COMPUTE_CAP")],
        _ => vec![80, 89, 90],
    };
    let mut caps = caps;
    caps.sort_unstable();
    caps.dedup();
    let list: Vec<String> = caps.iter().map(|c| c.to_string()).collect();
    println!("cargo:rustc-env=BASAL_CUDA_COMPUTE_CAPS={}", list.join(","));
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let mut table = String::from("pub const PTX: &[(u32, &str)] = &[\n");
    for cap in &caps {
        let out = out_dir.join(format!("kernels_{cap}.ptx"));
        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-O3", "-std=c++17", &format!("-arch=compute_{cap}"), "src/kernels.cu", "-o"])
            .arg(&out)
            .status()
            .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
        assert!(status.success(), "nvcc failed on src/kernels.cu (compute_{cap})");
        table += &format!("    ({cap}, include_str!({:?})),\n", out.display().to_string());
    }
    table += "];\n";
    std::fs::write(out_dir.join("kernels_ptx.rs"), table).unwrap();
}
