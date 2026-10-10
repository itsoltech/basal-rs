//! With the `cuda` feature: compile src/kernels.cu to PTX with nvcc, once per GPU architecture of CUDA_COMPUTE_CAPS
//! (comma-separated, default 80,89,90: A100 / RTX 30xx, RTX 6000 Ada / L40S / RTX 40xx, H100), else only for
//! CUDA_COMPUTE_CAP when that is set (candle's own kernels are compiled for CUDA_COMPUTE_CAP, which candle needs
//! without a GPU in the build machine; their PTX for 8.0 runs on newer GPUs too). At run time the PTX of the highest architecture not above the GPU's is loaded
//! (`fused::ptx`); a GPU older than every one of them is refused. With 90 among them, also a compute_90a build of the
//! same file with the Hopper (wgmma) attention, used on compute capability 9.0 only.

fn main() {
    #[cfg(feature = "intel")]
    if let Err(error) = validate_intel_shaders() {
        eprintln!("{error}");
        std::process::exit(1);
    }
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
    // Hopper: the same kernels plus the wgmma attention (BASAL_WGMMA), for compute capability 9.0 only (an `a` target
    // runs on no other architecture)
    if caps.contains(&90) {
        let out = out_dir.join("kernels_90a.ptx");
        let status = std::process::Command::new(&nvcc)
            .args(["--ptx", "-O3", "-std=c++17", "-arch=compute_90a", "-DBASAL_WGMMA", "src/kernels.cu", "-o"])
            .arg(&out)
            .status()
            .unwrap_or_else(|e| panic!("running {nvcc}: {e}"));
        assert!(status.success(), "nvcc failed on src/kernels.cu (compute_90a)");
        table += &format!("pub const PTX_90A: Option<&str> = Some(include_str!({:?}));\n", out.display().to_string());
    } else {
        table += "pub const PTX_90A: Option<&str> = None;\n";
    }
    std::fs::write(out_dir.join("kernels_ptx.rs"), table).unwrap();
}

/// Compile-time WGSL syntax/type validation. This requires no Vulkan driver or GPU and does not run inference.
#[cfg(feature = "intel")]
fn validate_intel_shaders() -> Result<(), String> {
    for name in ["gemm", "norm", "rope", "attention", "elementwise", "gather", "readout"] {
        let path = format!("src/intel/{name}.wgsl");
        println!("cargo:rerun-if-changed={path}");
        let source = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string_with_path(&source, &path))?;
        let capabilities = naga::valid::Capabilities::SHADER_FLOAT16 | naga::valid::Capabilities::SUBGROUP;
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .map_err(|e| e.emit_to_string_with_path(&source, &path))?;
    }
    Ok(())
}
