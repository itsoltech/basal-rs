//! With the `cuda` feature: compile src/kernels.cu to PTX with nvcc (CUDA_COMPUTE_CAP, default 89 = RTX 6000 Ada).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "89".into());
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    let status = std::process::Command::new(&nvcc)
        .args(["--ptx", "-O3", "-std=c++17", &format!("-arch=compute_{cap}"), "src/kernels.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
    assert!(status.success(), "nvcc failed on src/kernels.cu");
}
