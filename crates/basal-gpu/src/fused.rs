//! Fused kernels as candle custom ops: Metal (kernels.metal) and CUDA (kernels.cu, same arithmetic and rounding
//! points). Each replaces several candle element-wise kernels and the strided copies between them; the GEMMs, RMSNorm
//! and attention stay candle/MLX/cuBLAS kernels.

#[cfg(any(target_os = "macos", feature = "cuda"))]
use candle_core::backend::BackendStorage;
#[cfg(target_os = "macos")]
use candle_core::MetalStorage;
#[cfg(feature = "cuda")]
use candle_core::{cuda_backend::CudaStorageSlice as Cs, CudaStorage};
use candle_core::{CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

#[cfg(target_os = "macos")]
use metal::{launch, pipeline_suffix as suffix};

#[cfg_attr(
    not(any(feature = "cuda", target_os = "macos")),
    allow(dead_code, reason = "GPU kernels require contiguous layouts; the CPU fallback does not use this helper")
)]
fn contiguous(l: &Layout, what: &str) -> Result<usize> {
    match l.contiguous_offsets() {
        Some((start, _)) => Ok(start),
        None => candle_core::bail!("fused kernels: {what} must be contiguous"),
    }
}

#[cfg(target_os = "macos")]
mod metal {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::{DType, MetalStorage, Result};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Library};
    use objc2_metal::MTLSize;

    const SOURCE: &str = include_str!("kernels.metal");

    static LIBRARY: Mutex<Option<Library>> = Mutex::new(None);
    static PIPELINES: OnceLock<Mutex<HashMap<String, ComputePipeline>>> = OnceLock::new();

    pub(super) fn pipeline(dev: &candle_core::MetalDevice, name: &str) -> Result<ComputePipeline> {
        let cache = PIPELINES.get_or_init(Default::default);
        if let Some(p) = cache.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let mut lib = LIBRARY.lock().unwrap();
        if lib.is_none() {
            *lib = Some(dev.device().new_library_with_source(SOURCE, None).map_err(candle_core::Error::wrap)?);
        }
        let func = lib.as_ref().unwrap().get_function(name, None).map_err(candle_core::Error::wrap)?;
        let p = dev.device().new_compute_pipeline_state_with_function(&func).map_err(candle_core::Error::wrap)?;
        cache.lock().unwrap().insert(name.to_string(), p.clone());
        Ok(p)
    }

    pub fn pipeline_suffix(dt: DType) -> Result<&'static str> {
        match dt {
            DType::F32 => Ok("f32"),
            DType::F16 => Ok("f16"),
            DType::BF16 => Ok("bf16"),
            d => candle_core::bail!("fused kernels: unsupported dtype {d:?}"),
        }
    }

    /// One thread per element of `total`.
    pub fn launch(
        dev: &candle_core::MetalDevice,
        name: &str,
        inputs: &[(&MetalStorage, usize)],
        output: &candle_metal_kernels::metal::Buffer,
        consts: &[u32],
        total: usize,
    ) -> Result<()> {
        let p = pipeline(dev, name)?;
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&p);
        let mut i = 0;
        for (s, off) in inputs {
            enc.set_input_buffer(i, Some(s.buffer()), off * s.dtype().size_in_bytes());
            i += 1;
        }
        enc.set_output_buffer(i, Some(output), 0);
        i += 1;
        for c in consts {
            enc.set_bytes(i, c);
            i += 1;
        }
        let width = p.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize { width: total, height: 1, depth: 1 }, MTLSize { width, height: 1, depth: 1 });
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cu {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaStream, LaunchArgs, LaunchConfig};
    use candle_core::cuda_backend::{CudaDevice, DeviceId};

    // PTX: [(compute capability, PTX)], ascending (build.rs)
    include!(concat!(env!("OUT_DIR"), "/kernels_ptx.rs"));

    static SELECTED: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

    /// Choose the PTX for a GPU of compute capability `cap` (major * 10 + minor): the highest architecture not
    /// above it. None when the GPU is older than every compiled architecture.
    pub fn select(cap: u32) -> Option<u32> {
        let &(c, p) = PTX.iter().rev().find(|(c, _)| *c <= cap)?;
        let _ = SELECTED.set(p);
        let _ = CAP.set(cap);
        Some(c)
    }

    /// The compute_90a PTX with the Hopper attention, when the build has it and the GPU is of compute capability 9.0.
    pub fn ptx_90a() -> Option<&'static str> {
        PTX_90A.filter(|_| CAP.get() == Some(&90))
    }

    /// The PTX chosen by [`select`] (the lowest architecture when nothing was selected).
    pub fn ptx() -> &'static str {
        SELECTED.get().copied().unwrap_or(PTX[0].1)
    }

    /// One thread per element of `n`.
    pub fn cfg(n: usize) -> LaunchConfig {
        LaunchConfig::for_num_elems(n as u32)
    }

    /// A kernel of a module and the device's stream to launch it on.
    pub struct Kernel {
        func: CudaFunction,
        stream: std::sync::Arc<CudaStream>,
        /// looked up now for the first time on this device: function attributes are still to be set
        pub fresh: bool,
    }

    impl std::ops::Deref for Kernel {
        type Target = CudaFunction;
        fn deref(&self) -> &CudaFunction {
            &self.func
        }
    }

    impl Kernel {
        pub fn builder(&self) -> LaunchArgs<'_> {
            self.stream.launch_builder(&self.func)
        }
    }

    type Funcs = HashMap<(DeviceId, &'static str, String), CudaFunction>;
    static FUNCS: std::sync::OnceLock<Mutex<Funcs>> = std::sync::OnceLock::new();

    /// The kernel `name` of `module` (loaded from `ptx` the first time), looked up once per device: candle's
    /// `get_or_load_custom_func` asks the driver for the function on every call, several hundred times per forward.
    pub fn func(dev: &CudaDevice, name: &str, module: &'static str, ptx: &'static str) -> candle_core::Result<Kernel> {
        let key = (dev.id(), module, name.to_string());
        let mut funcs = FUNCS.get_or_init(Default::default).lock().unwrap();
        let fresh = !funcs.contains_key(&key);
        if fresh {
            let f = dev.get_or_load_custom_func(name, module, ptx)?.into_cuda_function();
            funcs.insert(key.clone(), f);
        }
        Ok(Kernel { func: funcs[&key].clone(), stream: dev.cuda_stream(), fresh })
    }
}

/// Run `$body` with `$t` bound to the Rust element type and `$s` to the kernel suffix of `$dt` (f32, f16, bf16).
#[cfg(feature = "cuda")]
macro_rules! by_dtype {
    ($dt:expr, |$t:ident, $s:ident| $body:expr) => {
        match $dt {
            DType::F32 => {
                type $t = f32;
                const $s: &str = "f32";
                $body
            }
            DType::F16 => {
                type $t = half::f16;
                const $s: &str = "f16";
                $body
            }
            DType::BF16 => {
                type $t = half::bf16;
                const $s: &str = "bf16";
                $body
            }
            d => candle_core::bail!("fused kernels: unsupported dtype {d:?}"),
        }
    };
}

/// The typed device slice of `$st` (a `CudaStorage`) for the element type `$t`, from offset `$off`.
#[cfg(feature = "cuda")]
macro_rules! view {
    ($st:expr, $t:ty, $off:expr) => {
        <$t as candle_core::cuda_backend::CudaDType>::as_cuda_slice(&$st)?.slice($off..)
    };
}

fn no_cpu() -> Result<(CpuStorage, Shape)> {
    candle_core::bail!("fused kernels run on Metal or CUDA only")
}

struct BiasSiluMul;

impl CustomOp2 for BiasSiluMul {
    fn name(&self) -> &'static str {
        "bias-silu-mul"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, gu: &CudaStorage, gl: &Layout, b: &CudaStorage, bl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc::driver::PushKernelArg, WrapErr};
        let (m, two_i) = gl.shape().dims2()?;
        let i = two_i / 2;
        let total = m * i;
        let dev = gu.device.clone();
        let (go, bo) = (contiguous(gl, "gate_up")?, contiguous(bl, "bias")?);
        let st = by_dtype!(gu.dtype(), |T, S| {
            let (g, bb) = (view!(gu, T, go), view!(b, T, bo));
            // SAFETY: every element is written by the kernel.
            let out = unsafe { dev.alloc::<T>(total)? };
            // 8 outputs per thread with 16-byte accesses when the layout allows it (same arithmetic per element)
            let vec8 = S != "f32" && i % 8 == 0 && go % 8 == 0 && bo % 8 == 0;
            let name = if vec8 { format!("bias_silu_mul8_{S}") } else { format!("bias_silu_mul_{S}") };
            let f = cu::func(&dev, &name, "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let threads = if vec8 { total / 8 } else { total };
            let (i_, n_) = (i as u32, threads as u32);
            a.arg(&g).arg(&bb).arg(&out).arg(&i_).arg(&n_);
            // SAFETY: argument types and counts match the kernel signature; with vec8 every operand offset and I are
            // multiples of 8 elements (16 bytes), as the kernel's 16-byte accesses need.
            unsafe { a.launch(cu::cfg(threads)) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, Shape::from((m, i))))
    }
    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        gu: &MetalStorage,
        gl: &Layout,
        b: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (m, two_i) = gl.shape().dims2()?;
        let i = two_i / 2;
        let dev = gu.device();
        let total = m * i;
        let out = dev.new_buffer_builder().with_size_for(total, gu.dtype()).with_label("bias_silu_mul").build()?;
        let name = format!("bias_silu_mul_{}", suffix(gu.dtype())?);
        launch(
            dev,
            &name,
            &[(gu, contiguous(gl, "gate_up")?), (b, contiguous(bl, "bias")?)],
            &out,
            &[i as u32, total as u32],
            total,
        )?;
        Ok((MetalStorage::new(out, dev.clone(), total, gu.dtype()), Shape::from((m, i))))
    }
}

/// `silu(gate + b_gate) * (up + b_up)` for `gu = [M, 2I]` (gate | up) and `bias = [2I]`.
pub fn bias_silu_mul(gu: &Tensor, bias: &Tensor) -> Result<Tensor> {
    gu.apply_op2_no_bwd(bias, &BiasSiluMul)
}

/// `(T(x + T(y + b)), RMSNorm(that) * alpha)` in one CUDA kernel: the residual add of a linear output and the RMSNorm
/// of the next block, bitwise as [`bias_residual`] followed by candle's `rms_norm` (same arithmetic and summation
/// order), with one read of the residual stream instead of two. Returns (x, normed) as views of one [2, M, H] buffer.
#[cfg(feature = "cuda")]
pub fn residual_rmsnorm(x: &Tensor, y: &Tensor, b: &Tensor, alpha: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    let out = x.apply_op3_no_bwd(y, b, &ResidualRmsNorm { alpha: alpha.clone(), eps })?;
    Ok((out.get(0)?, out.get(1)?))
}

#[cfg(feature = "cuda")]
struct ResidualRmsNorm {
    alpha: Tensor,
    eps: f32,
}

#[cfg(feature = "cuda")]
impl CustomOp3 for ResidualRmsNorm {
    fn name(&self) -> &'static str {
        "residual-rmsnorm"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        y: &CudaStorage,
        yl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{
            cudarc::driver::{LaunchConfig, PushKernelArg},
            WrapErr,
        };
        let (m, h) = xl.shape().dims2()?;
        let dev = x.device.clone();
        let (xo, yo, bo) = (contiguous(xl, "x")?, contiguous(yl, "y")?, contiguous(bl, "bias")?);
        let (ast, al) = self.alpha.storage_and_layout();
        let candle_core::Storage::Cuda(acs) = &*ast else {
            candle_core::bail!("residual-rmsnorm: alpha must be on the CUDA device");
        };
        let ao = contiguous(al, "alpha")?;
        let st = by_dtype!(x.dtype(), |T, S| {
            let (xv, yv, bv, av) = (view!(x, T, xo), view!(y, T, yo), view!(b, T, bo), view!(acs, T, ao));
            // SAFETY: the kernel writes all 2 * m * h elements (both halves of every row).
            let out = unsafe { dev.alloc::<T>(2 * m * h)? };
            let f = cu::func(&dev, &format!("residual_rmsnorm_{S}"), "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let (h_, m_, eps) = (h as u32, m as u32, self.eps);
            a.arg(&xv).arg(&yv).arg(&bv).arg(&av).arg(&out).arg(&h_).arg(&m_).arg(&eps);
            // the block size of candle's rmsnorm, on which the summation order depends
            let bs = if h < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (bs, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: argument types and counts match the kernel signature; one block per row of m.
            unsafe { a.launch(cfg) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, Shape::from((2, m, h))))
    }
}

struct BiasResidual;

impl CustomOp3 for BiasResidual {
    fn name(&self) -> &'static str {
        "bias-residual"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        y: &CudaStorage,
        yl: &Layout,
        b: &CudaStorage,
        bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc::driver::PushKernelArg, WrapErr};
        let (_, h) = xl.shape().dims2()?;
        let total = xl.shape().elem_count();
        let dev = x.device.clone();
        let (xo, yo, bo) = (contiguous(xl, "x")?, contiguous(yl, "y")?, contiguous(bl, "bias")?);
        let st = by_dtype!(x.dtype(), |T, S| {
            let (xv, yv, bv) = (view!(x, T, xo), view!(y, T, yo), view!(b, T, bo));
            // SAFETY: every element is written by the kernel.
            let out = unsafe { dev.alloc::<T>(total)? };
            let f = cu::func(&dev, &format!("bias_residual_{S}"), "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let (h_, n_) = (h as u32, total as u32);
            a.arg(&xv).arg(&yv).arg(&bv).arg(&out).arg(&h_).arg(&n_);
            // SAFETY: argument types and counts match the kernel signature.
            unsafe { a.launch(cu::cfg(total)) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, xl.shape().clone()))
    }
    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        x: &MetalStorage,
        xl: &Layout,
        y: &MetalStorage,
        yl: &Layout,
        b: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (_, h) = xl.shape().dims2()?;
        let total = xl.shape().elem_count();
        let dev = x.device();
        let out = dev.new_buffer_builder().with_size_for(total, x.dtype()).with_label("bias_residual").build()?;
        let name = format!("bias_residual_{}", suffix(x.dtype())?);
        let ins = [(x, contiguous(xl, "x")?), (y, contiguous(yl, "y")?), (b, contiguous(bl, "bias")?)];
        launch(dev, &name, &ins, &out, &[h as u32, total as u32], total)?;
        Ok((MetalStorage::new(out, dev.clone(), total, x.dtype()), xl.shape().clone()))
    }
}

/// `x + (y + b)` with `x, y = [M, H]`, `b = [H]`.
pub fn bias_residual(x: &Tensor, y: &Tensor, b: &Tensor) -> Result<Tensor> {
    x.apply_op3_no_bwd(y, b, &BiasResidual)
}

#[derive(Clone, Copy)]
pub struct HeadDims {
    pub b: usize,
    pub l: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
}

#[cfg_attr(
    not(any(feature = "cuda", target_os = "macos")),
    allow(dead_code, reason = "Head dimensions are read by GPU kernels; the CPU implementation is unsupported")
)]
struct QkvRope {
    d: HeadDims,
    /// CUDA: device address of the f16 `[K hi | K lo | V hi | V lo]` buffer (0: none) and the kernel's QR_* flags
    #[cfg_attr(not(feature = "cuda"), expect(dead_code, reason = "read by the CUDA kernel only"))]
    hl: u64,
    #[cfg_attr(not(feature = "cuda"), expect(dead_code, reason = "read by the CUDA kernel only"))]
    flags: u32,
}

impl CustomOp3 for QkvRope {
    fn name(&self) -> &'static str {
        "qkv-rope"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        qkv: &CudaStorage,
        ql: &Layout,
        b: &CudaStorage,
        bl: &Layout,
        cs: &CudaStorage,
        cl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc::driver::PushKernelArg, WrapErr};
        let d = self.d;
        let n = d.b * d.l * (d.nh + 2 * d.nkv) * d.hd;
        let dev = qkv.device.clone();
        let (qo, bo, co) = (contiguous(ql, "qkv")?, contiguous(bl, "bias")?, contiguous(cl, "cos/sin")?);
        let csv = view!(cs, f32, co);
        // SAFETY: every element is written by the kernel.
        let out = unsafe { dev.alloc::<f32>(n)? };
        by_dtype!(qkv.dtype(), |T, S| {
            let (qv, bv) = (view!(qkv, T, qo), view!(b, T, bo));
            let f = cu::func(&dev, &format!("qkv_rope_{S}"), "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let c = [d.b as u32, d.l as u32, d.nh as u32, d.nkv as u32, d.hd as u32];
            a.arg(&qv).arg(&bv).arg(&csv).arg(&out).arg(&c[0]).arg(&c[1]).arg(&c[2]).arg(&c[3]).arg(&c[4]);
            a.arg(&self.hl).arg(&self.flags);
            // SAFETY: argument types and counts match the kernel signature.
            unsafe { a.launch(cu::cfg(n / 2)) }.w()?;
        });
        Ok((CudaStorage { slice: Cs::F32(out), device: dev }, Shape::from((n,))))
    }
    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        qkv: &MetalStorage,
        ql: &Layout,
        b: &MetalStorage,
        bl: &Layout,
        cs: &MetalStorage,
        cl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let d = self.d;
        if cs.dtype() != DType::F32 {
            candle_core::bail!("qkv_rope: cos/sin must be f32");
        }
        let n = d.b * d.l * (d.nh + 2 * d.nkv) * d.hd;
        let dev = qkv.device();
        let out = dev.new_buffer_builder().with_size_for(n, DType::F32).with_label("qkv_rope").build()?;
        let name = format!("qkv_rope_{}", suffix(qkv.dtype())?);
        let ins = [(qkv, contiguous(ql, "qkv")?), (b, contiguous(bl, "bias")?), (cs, contiguous(cl, "cos/sin")?)];
        let consts = [d.b as u32, d.l as u32, d.nh as u32, d.nkv as u32, d.hd as u32];
        launch(dev, &name, &ins, &out, &consts, n / 2)?;
        Ok((MetalStorage::new(out, dev.clone(), n, DType::F32), Shape::from((n,))))
    }
}

/// Bias + RoPE + head split: returns the flat f32 buffer `[q | k | v]` and its views `q [B, NH, L, HD]`,
/// `k, v [B, NKV, L, HD]`.
pub fn qkv_rope(
    qkv: &Tensor,
    bias: &Tensor,
    cos_sin: &Tensor,
    d: HeadDims,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let flat = qkv.apply_op3_no_bwd(bias, cos_sin, &QkvRope { d, hl: 0, flags: 0 })?;
    let (nq, nk) = (d.b * d.nh * d.l * d.hd, d.b * d.nkv * d.l * d.hd);
    let q = flat.narrow(0, 0, nq)?.reshape((d.b, d.nh, d.l, d.hd))?;
    let k = flat.narrow(0, nq, nk)?.reshape((d.b, d.nkv, d.l, d.hd))?;
    let v = flat.narrow(0, nq + nk, nk)?.reshape((d.b, d.nkv, d.l, d.hd))?;
    Ok((flat, q, k, v))
}

/// [`qkv_rope`] for the tensor-core attention (CUDA): also returns the layer's K and V split into f16 hi / lo planes,
/// `[K hi | K lo | V hi | V lo]` of `B*NKV*L*HD` each, as [`split_hilo`] of K and of V would give them. `v_lo`: write
/// the V lo plane (not read by the attention when V is exact f16); `kv_f32`: also write K and V into the flat f32
/// buffer (a prefix capture needs them; otherwise only its Q part is read).
#[cfg(feature = "cuda")]
pub fn qkv_rope_hilo(
    qkv: &Tensor,
    bias: &Tensor,
    cos_sin: &Tensor,
    d: HeadDims,
    v_lo: bool,
    kv_f32: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let nk = d.b * d.nkv * d.l * d.hd;
    // SAFETY: qkv_rope writes the K hi, K lo and V hi planes, and the V lo plane when it is read (`v_lo`).
    let hl = unsafe { Tensor::empty((4 * nk,), DType::F16, qkv.device())? };
    let flags = 1 | if v_lo { 2 } else { 0 } | if kv_f32 { 0 } else { 4 };
    let flat = qkv.apply_op3_no_bwd(bias, cos_sin, &QkvRope { d, hl: device_addr_f16(&hl)?, flags })?;
    let nq = d.b * d.nh * d.l * d.hd;
    let q = flat.narrow(0, 0, nq)?.reshape((d.b, d.nh, d.l, d.hd))?;
    let k = flat.narrow(0, nq, nk)?.reshape((d.b, d.nkv, d.l, d.hd))?;
    let v = flat.narrow(0, nq + nk, nk)?.reshape((d.b, d.nkv, d.l, d.hd))?;
    Ok((flat, q, k, v, hl))
}

#[cfg_attr(
    not(any(feature = "cuda", target_os = "macos")),
    allow(dead_code, reason = "Head dimensions and dtype are read only by the GPU implementations")
)]
struct MergeHeads(HeadDims, DType);

impl CustomOp1 for MergeHeads {
    fn name(&self) -> &'static str {
        "merge-heads"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{cudarc::driver::PushKernelArg, WrapErr};
        let (d, dt) = (self.0, self.1);
        let n = d.b * d.l * d.nh * d.hd;
        let dev = x.device.clone();
        let xv = view!(x, f32, contiguous(xl, "attention output")?);
        let st = by_dtype!(dt, |T, S| {
            // SAFETY: every element is written by the kernel.
            let out = unsafe { dev.alloc::<T>(n)? };
            let f = cu::func(&dev, &format!("merge_heads_{S}"), "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let c = [d.b as u32, d.l as u32, d.nh as u32, d.hd as u32];
            a.arg(&xv).arg(&out).arg(&c[0]).arg(&c[1]).arg(&c[2]).arg(&c[3]);
            // SAFETY: argument types and counts match the kernel signature.
            unsafe { a.launch(cu::cfg(n)) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, Shape::from((d.b * d.l, d.nh * d.hd))))
    }
    #[cfg(target_os = "macos")]
    fn metal_fwd(&self, x: &MetalStorage, xl: &Layout) -> Result<(MetalStorage, Shape)> {
        let (d, dt) = (self.0, self.1);
        if x.dtype() != DType::F32 {
            candle_core::bail!("merge_heads: input must be f32");
        }
        let n = d.b * d.l * d.nh * d.hd;
        let dev = x.device();
        let out = dev.new_buffer_builder().with_size_for(n, dt).with_label("merge_heads").build()?;
        let name = format!("merge_heads_{}", suffix(dt)?);
        let consts = [d.b as u32, d.l as u32, d.nh as u32, d.hd as u32];
        launch(dev, &name, &[(x, contiguous(xl, "attention output")?)], &out, &consts, n)?;
        Ok((MetalStorage::new(out, dev.clone(), n, dt), Shape::from((d.b * d.l, d.nh * d.hd))))
    }
}

/// f32 `[B, NH, L, HD]` -> `[B*L, NH*HD]` in `dt`.
pub fn merge_heads(x: &Tensor, d: HeadDims, dt: DType) -> Result<Tensor> {
    x.apply_op1_no_bwd(&MergeHeads(d, dt))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code, reason = "Softmax dimensions are used only by the CUDA kernel"))]
struct MaskedSoftmax {
    l: usize,
    scale: f32,
}

impl CustomOp2 for MaskedSoftmax {
    fn name(&self) -> &'static str {
        "masked-softmax"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s: &CudaStorage, sl: &Layout, m: &CudaStorage, ml: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        if s.dtype() != DType::F32 || m.dtype() != DType::F32 {
            candle_core::bail!("masked_softmax: f32 only");
        }
        let lk = sl.shape().dims().last().copied().unwrap_or(0);
        let n = sl.shape().elem_count();
        let rows = n / lk.max(1);
        if ml.shape().elem_count() != self.l * lk || !rows.is_multiple_of(self.l) {
            candle_core::bail!("masked_softmax: mask {:?} does not match scores {:?}", ml.shape(), sl.shape());
        }
        let dev = s.device.clone();
        let (sv, mv) = (view!(s, f32, contiguous(sl, "scores")?), view!(m, f32, contiguous(ml, "mask")?));
        // SAFETY: every element is written by the kernel.
        let out = unsafe { dev.alloc::<f32>(n)? };
        let f = cu::func(&dev, "masked_softmax_f32", "basal_fused", cu::ptx())?;
        let mut a = f.builder();
        let (l_, lk_) = (self.l as u32, lk as u32);
        a.arg(&sv).arg(&mv).arg(&out).arg(&l_).arg(&lk_).arg(&self.scale);
        let threads = if lk >= 1024 { 256 } else { 128 };
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        // SAFETY: argument types and counts match the kernel signature.
        unsafe { a.launch(cfg) }.w()?;
        Ok((CudaStorage { slice: Cs::F32(out), device: dev }, sl.shape().clone()))
    }
}

/// `softmax(scores * scale + mask)` over the last dimension (CUDA, f32): `scores [.., R, LK]` whose rows are query
/// positions `0..l` repeated per head, `mask [l, LK]` shared by all heads.
pub fn masked_softmax(scores: &Tensor, mask: &Tensor, l: usize, scale: f32) -> Result<Tensor> {
    scores.apply_op2_no_bwd(mask, &MaskedSoftmax { l, scale })
}

/// Device address of the first element of a contiguous f32 CUDA tensor (for prefix K/V read by a kernel).
#[cfg(feature = "cuda")]
pub fn device_addr_f32(t: &Tensor) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(cs) = &*st else { candle_core::bail!("device_addr_f32: not a CUDA tensor") };
    let start = contiguous(l, "prefix K/V")?;
    let slice = <f32 as candle_core::cuda_backend::CudaDType>::as_cuda_slice(cs)?;
    let stream = cs.device.cuda_stream();
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p + (start * 4) as u64)
}

#[cfg(feature = "cuda")]
struct LetterLogits;

#[cfg(feature = "cuda")]
impl CustomOp3 for LetterLogits {
    fn name(&self) -> &'static str {
        "letter-logits"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(
        &self,
        h: &CudaStorage,
        hl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        ids: &CudaStorage,
        il: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;
        let (r, hd) = hl.shape().dims2()?;
        let (_, hd2) = wl.shape().dims2()?;
        let u = il.shape().elem_count();
        if hd != hd2 || w.dtype() != DType::BF16 || ids.dtype() != DType::U32 {
            candle_core::bail!("letter_logits: hidden [R, H], bf16 lm_head [V, H], u32 ids expected");
        }
        let dev = h.device.clone();
        let wv = view!(w, half::bf16, contiguous(wl, "lm_head")?);
        let iv = view!(ids, u32, contiguous(il, "ids")?);
        let ho = contiguous(hl, "hidden")?;
        // SAFETY: every element is written by the kernel.
        let out = unsafe { dev.alloc::<f32>(r * u)? };
        by_dtype!(h.dtype(), |T, S| {
            let hv = view!(h, T, ho);
            let f = cu::func(&dev, &format!("letter_logits_{S}"), "basal_fused", cu::ptx())?;
            let mut a = f.builder();
            let c = [r as u32, u as u32, hd as u32];
            a.arg(&hv).arg(&wv).arg(&iv).arg(&out).arg(&c[0]).arg(&c[1]).arg(&c[2]);
            // SAFETY: argument types and counts match the kernel signature.
            unsafe { a.launch(cu::cfg(r * u * 32)) }.w()?;
        });
        Ok((CudaStorage { slice: Cs::F32(out), device: dev }, Shape::from((r, u))))
    }
}

/// f32 logits `[R, U]` of the letter rows `ids` of the bf16 `lm_head` for the hidden states `[R, H]` (CUDA), with a
/// summation order that does not depend on R or U.
#[cfg(feature = "cuda")]
pub fn letter_logits(hidden: &Tensor, lm_head: &Tensor, ids: &Tensor) -> Result<Tensor> {
    hidden.apply_op3_no_bwd(lm_head, ids, &LetterLogits)
}

/// Ranges per unit: the depth of a packed row ([`basal_core::pack::MAX_TREE_DEPTH`]).
#[cfg(feature = "cuda")]
const TREE_MAXR: usize = 8;
/// Units per launch; the table is a kernel parameter (128 × 120 B, within the 32 KiB parameter limit of CUDA 12.1+
/// on sm_70+).
#[cfg(feature = "cuda")]
const TREE_MAXU: usize = 128;

/// One work unit of [`attention_tree`]: the queries `q_off..q_off + q_len` (one block of a packed row, compact token
/// indices) and their keys: `np` prefix keys (`pk`, `pv`: device addresses of f32 `[NKV, np, HD]`, 0 when `np == 0`)
/// followed by the token ranges `ranges` (ancestor blocks from the root down, then the own block, causal).
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub struct TreeUnit {
    pub pk: u64,
    pub pv: u64,
    /// the prefix K / V split into f16 hi / lo planes (tensor-core kernel), 0 when unused
    pub pkh: u64,
    pub pvh: u64,
    pub np: usize,
    pub q_off: usize,
    pub q_len: usize,
    pub ranges: Vec<(usize, usize)>,
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TreeUnitC {
    pk: u64,
    pv: u64,
    pkh: u64,
    pvh: u64,
    np: u32,
    q_off: u32,
    q_len: u32,
    tile0: u32,
    nr: u32,
    pad: u32,
    r_off: [u32; TREE_MAXR],
    r_len: [u32; TREE_MAXR],
}

#[cfg(feature = "cuda")]
#[repr(C)]
#[derive(Clone, Copy)]
struct TreeTable {
    u: [TreeUnitC; TREE_MAXU],
    n: u32,
    pad: u32,
}

// SAFETY: plain-old-data passed by value as a kernel parameter.
#[cfg(feature = "cuda")]
unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for TreeTable {}

#[cfg(feature = "cuda")]
struct AttentionTree<'a> {
    units: &'a [TreeUnit],
    total: usize,
    nh: usize,
    nkv: usize,
    scale: f32,
    /// tensor-core kernel (`attn_tree_tc*`, see [`tc_kernel`]) instead of the f32 SIMT kernel (attn_tree_f32)
    tc: bool,
    tc_kernel: &'static str,
    /// V values are exact f16 (f16 forward): the lo plane of V is zero and skipped
    v_exact: bool,
    /// tensor-core kernels: output merged heads `[T, NH * HD]` in this dtype (f16 / bf16) instead of f32 heads
    merged: Option<DType>,
}

/// Dynamic shared memory of attn_tree_tc_pipe: two stages of K and of V tiles (8 x 32 x 136 x 2); the other tensor-core
/// kernels take half: one Q plane of up to 128 rows, then the K / V tiles (4 x 32 x 136 x 2) in the same memory.
#[cfg(feature = "cuda")]
const TC_SMEM: usize = 8 * 32 * 136 * 2;

#[cfg(feature = "cuda")]
impl CustomOp3 for AttentionTree<'_> {
    fn name(&self) -> &'static str {
        "attention-tree"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    /// `khl`, `vhl`: the layer's K and V split into f16 hi / lo planes (tensor-core kernel; ignored by the f32 one).
    fn cuda_fwd(
        &self,
        qkv: &CudaStorage,
        ql: &Layout,
        khl: &CudaStorage,
        kl: &Layout,
        vhl: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        const HD: usize = 128;
        let (t, nh, nkv) = (self.total, self.nh, self.nkv);
        if qkv.dtype() != DType::F32 || ql.shape().elem_count() != (nh + 2 * nkv) * t * HD {
            candle_core::bail!("attention_tree: expects the f32 qkv_rope output with head_dim {HD}");
        }
        let dev = qkv.device.clone();
        let qv = view!(qkv, f32, contiguous(ql, "qkv")?);
        let n = nh * t * HD;
        let out = match self.merged {
            // SAFETY: every query of every unit is written; the units cover the token list.
            None => Cs::F32(unsafe { dev.alloc::<f32>(n)? }),
            // SAFETY: as above.
            Some(DType::F16) => Cs::F16(unsafe { dev.alloc::<half::f16>(n)? }),
            // SAFETY: as above.
            Some(DType::BF16) => Cs::BF16(unsafe { dev.alloc::<half::bf16>(n)? }),
            Some(d) => candle_core::bail!("attention_tree: merged output in {d:?}"),
        };
        let out_mode: u32 = match self.merged {
            None => 0,
            Some(DType::F16) => 1,
            _ => 2,
        };
        // rows per block and threads: attn_tree_tc* TC_BM / TC_THREADS, attn_tree_f32 ATT_BM / ATT_THREADS
        let (name, bm, threads, smem) = match self.tc_kernel {
            _ if !self.tc => ("attn_tree_f32", 16, 128, 0),
            k @ "attn_tree_tc_pipe" => (k, 128, 256, TC_SMEM),
            // Q hi / lo of 128 rows, then the K / V tiles
            k @ "attn_tree_wg" => (k, 128, 256, (2 * 128 * 128 + 4 * 32 * 128) * 2),
            // Q hi / lo, then two stages of K hi / lo and V hi
            k @ "attn_tree_wgp" => (k, 128, 256, (2 * 128 * 128 + 2 * 3 * 32 * 128) * 2),
            k => (k, 128, 256, TC_SMEM / 2),
        };
        let f = match (name, cu::ptx_90a()) {
            ("attn_tree_wg" | "attn_tree_wgp", Some(ptx)) => cu::func(&dev, name, "basal_fused_90a", ptx)?,
            ("attn_tree_wg" | "attn_tree_wgp", None) => {
                candle_core::bail!("BASAL_ATT={name} needs a Hopper GPU (compute capability 9.0)")
            }
            _ => cu::func(&dev, name, "basal_fused", cu::ptx())?,
        };
        if smem > 0 && f.fresh {
            use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute;
            f.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32).w()?;
            if name.starts_with("attn_tree_wg") {
                // the largest shared memory carveout: two blocks of 96 KiB per SM
                f.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100).w()?;
            }
        }
        let rep = nh / nkv;
        for chunk in self.units.chunks(TREE_MAXU) {
            let mut tab = TreeTable { u: [TreeUnitC::default(); TREE_MAXU], n: chunk.len() as u32, pad: 0 };
            let mut tiles = 0u32;
            for (d, s) in tab.u.iter_mut().zip(chunk) {
                if s.ranges.is_empty() || s.ranges.len() > TREE_MAXR || s.ranges.last().unwrap().0 != s.q_off {
                    candle_core::bail!("attention_tree: unit needs 1..={TREE_MAXR} ranges ending with its own block");
                }
                if self.tc && s.np > 0 && (s.pkh == 0 || s.pvh == 0) {
                    candle_core::bail!("attention_tree: the tensor-core kernel needs split prefix K/V");
                }
                let mut c = TreeUnitC {
                    pk: s.pk,
                    pv: s.pv,
                    pkh: s.pkh,
                    pvh: s.pvh,
                    np: s.np as u32,
                    q_off: s.q_off as u32,
                    q_len: s.q_len as u32,
                    tile0: tiles,
                    nr: s.ranges.len() as u32,
                    ..Default::default()
                };
                for (i, &(o, l)) in s.ranges.iter().enumerate() {
                    c.r_off[i] = o as u32;
                    c.r_len[i] = l as u32;
                }
                *d = c;
                tiles += (rep * s.q_len).div_ceil(bm) as u32;
            }
            let mut a = f.builder();
            let (t_, nh_, nkv_, vx) = (t as u32, nh as u32, nkv as u32, u32::from(self.v_exact));
            if self.tc {
                let (kv, vv) = (
                    view!(khl, half::f16, contiguous(kl, "K hi/lo")?),
                    view!(vhl, half::f16, contiguous(vl, "V hi/lo")?),
                );
                a.arg(&qv).arg(&kv).arg(&vv);
                match &out {
                    Cs::F32(o) => a.arg(o),
                    Cs::F16(o) => a.arg(o),
                    Cs::BF16(o) => a.arg(o),
                    _ => unreachable!("allocated above"),
                };
                a.arg(&tab).arg(&t_).arg(&nh_).arg(&nkv_).arg(&self.scale).arg(&vx);
                if self.tc_kernel != "attn_tree_tc_pipe" {
                    a.arg(&out_mode);
                }
                // SAFETY: argument types and counts match the kernel signature; prefix addresses are live tensors.
                unsafe {
                    a.launch(LaunchConfig {
                        grid_dim: (tiles, nkv as u32, 1),
                        block_dim: (threads, 1, 1),
                        shared_mem_bytes: smem as u32,
                    })
                }
                .w()?;
                continue;
            }
            let Cs::F32(out) = &out else { candle_core::bail!("attention_tree: the f32 kernel has f32 output") };
            a.arg(&qv).arg(out).arg(&tab).arg(&t_).arg(&nh_).arg(&nkv_).arg(&self.scale);
            let cfg = LaunchConfig {
                grid_dim: (tiles, nkv as u32, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: smem as u32,
            };
            // SAFETY: argument types and counts match the kernel signature; prefix addresses are live tensors.
            unsafe { a.launch(cfg) }.w()?;
        }
        let shape = if self.merged.is_some() { Shape::from((t, nh * HD)) } else { Shape::from((1, nh, t, HD)) };
        Ok((CudaStorage { slice: out, device: dev }, shape))
    }
}

/// Tensor-core attention kernel: `tc` (Q·K with f16 hi/lo operands, three MMAs, and P split into hi/lo for P·V:
/// products close to f32), on compute capability 9.0 the same with Hopper warpgroup MMAs and the next K / V tile
/// loading during the current one (`wgp`; `wg` without that overlap; both bitwise the same outputs on H100);
/// `BASAL_ATT` selects one for A/B: `tc`, `wg`, `wgp`, `tc-pv1` (P rounded to f16), `tc-qk1` (Q·K from the f16 hi
/// parts only), `tc-f16` (both), `tc-pipe`. Softmax, accumulation and output are f32 in every variant.
#[cfg(feature = "cuda")]
pub fn tc_kernel() -> &'static str {
    static K: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    K.get_or_init(|| match std::env::var("BASAL_ATT").as_deref() {
        Ok("tc") => "attn_tree_tc",
        Ok("tc-pv1") => "attn_tree_tc_pv1",
        Ok("tc-qk1") => "attn_tree_tc_qk1",
        Ok("tc-f16") => "attn_tree_tc_f16",
        Ok("tc-pipe") => "attn_tree_tc_pipe",
        Ok("wg") => "attn_tree_wg",
        Ok("wgp") => "attn_tree_wgp",
        _ if cu::ptx_90a().is_some() => "attn_tree_wgp",
        _ => "attn_tree_tc",
    })
}

/// f32 attention of all units (blocks of packed rows) of a forward (CUDA), invariant to how questions are packed:
/// `qkv` = the flat [`qkv_rope`] output over `d.l` tokens. Returns `[1, NH, total, HD]`.
/// `tc = Some((khl, vhl, v_exact))`: tensor cores with f16 hi/lo operands (about f32 accuracy) on the layer's K / V
/// split by [`split_hilo`]; `None`: the f32 SIMT kernel.
#[cfg(feature = "cuda")]
pub fn attention_tree(
    qkv: &Tensor,
    units: &[TreeUnit],
    d: HeadDims,
    scale: f32,
    tc: Option<(&Tensor, &Tensor, bool)>,
    merge: Option<DType>,
) -> Result<Tensor> {
    // the tensor-core kernels (except tc-pipe) write merged heads themselves, bitwise as merge_heads would
    let in_kernel = merge
        .filter(|&dt| tc.is_some() && tc_kernel() != "attn_tree_tc_pipe" && matches!(dt, DType::F16 | DType::BF16));
    let att = std::env::var("BASAL_ATT");
    let op = AttentionTree {
        units,
        total: d.l,
        nh: d.nh,
        nkv: d.nkv,
        scale,
        tc: tc.is_some(),
        // the Hopper kernels are measured for f16 forwards: a bf16 forward (V lo plane) keeps tc unless BASAL_ATT asks
        // for wg / wgp; wgp has no V lo stage, so wg runs instead
        tc_kernel: match (tc_kernel(), tc.is_some_and(|t| t.2), matches!(att.as_deref(), Ok("wg" | "wgp"))) {
            ("attn_tree_wg" | "attn_tree_wgp", false, false) => "attn_tree_tc",
            ("attn_tree_wgp", false, true) => "attn_tree_wg",
            (k, _, _) => k,
        },
        v_exact: tc.is_some_and(|t| t.2),
        merged: in_kernel,
    };
    let o = match tc {
        Some((k, v, _)) => qkv.apply_op3_no_bwd(k, v, &op)?,
        None => qkv.apply_op3_no_bwd(qkv, qkv, &op)?,
    };
    match merge {
        Some(dt) if in_kernel.is_none() => merge_heads(&o, d, dt),
        _ => Ok(o),
    }
}

/// One work unit of [`attention_tree_metal`]: the queries `q_off..q_off + q_len` (one block of a packed row, compact
/// token indices) and their keys: the precomputed prefix `past` (an index into the `pasts` of the call) followed by
/// the token ranges `ranges` (ancestor blocks from the root down, then the own block, causal).
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct MetalTreeUnit {
    pub past: Option<usize>,
    pub q_off: usize,
    pub q_len: usize,
    pub ranges: Vec<(usize, usize)>,
}

/// `TreeUnit` of kernels.metal.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MetalTreeUnitC {
    np: u32,
    q_off: u32,
    q_len: u32,
    tile0: u32,
    nr: u32,
    pad: [u32; 3],
    r_off: [u32; 8],
    r_len: [u32; 8],
}

#[cfg(target_os = "macos")]
struct AttentionTreeMetal<'a> {
    units: &'a [MetalTreeUnit],
    /// prefix K / V of the layer, f32 `[1, NKV, np, HD]`
    pasts: &'a [(&'a Tensor, &'a Tensor)],
    total: usize,
    nh: usize,
    nkv: usize,
    scale: f32,
}

#[cfg(target_os = "macos")]
impl CustomOp1 for AttentionTreeMetal<'_> {
    fn name(&self) -> &'static str {
        "attention-tree"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn metal_fwd(&self, qkv: &MetalStorage, ql: &Layout) -> Result<(MetalStorage, Shape)> {
        use objc2_metal::MTLSize;
        const HD: usize = 128;
        /// Units per dispatch: the table goes through setBytes (at most 4 KiB).
        const MAXU: usize = 32;
        let (t, nh, nkv) = (self.total, self.nh, self.nkv);
        if qkv.dtype() != DType::F32 || ql.shape().elem_count() != (nh + 2 * nkv) * t * HD {
            candle_core::bail!("attention_tree: expects the f32 qkv_rope output with head_dim {HD}");
        }
        let dev = qkv.device();
        let n = nh * t * HD;
        let out = dev.new_buffer_builder().with_size_for(n, DType::F32).with_label("attention_tree").build()?;
        let p = metal::pipeline(dev, "attn_tree_f32")?;
        let qo = contiguous(ql, "qkv")? * 4;
        let rep = nh / nkv;
        // simdgroups (8 query rows each) per threadgroup, at most ATT_SG = 8; the result does not depend on it
        let nsg: usize = std::env::var("BASAL_ATT_SG").ok().and_then(|v| v.parse().ok()).unwrap_or(8).clamp(1, 8);
        // one dispatch per prefix (it is bound as a buffer) and per MAXU units
        let mut groups: Vec<(Option<usize>, Vec<&MetalTreeUnit>)> = Vec::new();
        for u in self.units {
            match groups.iter_mut().find(|g| g.0 == u.past) {
                Some(g) => g.1.push(u),
                None => groups.push((u.past, vec![u])),
            }
        }
        for (past, us) in &groups {
            let guards = match past {
                Some(i) => {
                    let (k, v) = self.pasts[*i];
                    Some((k.storage_and_layout(), v.storage_and_layout(), k.dim(2)?))
                }
                None => None,
            };
            let (pk, pv, np) = match &guards {
                Some(((ks, kl), (vs, vl), np)) => {
                    let (candle_core::Storage::Metal(k), candle_core::Storage::Metal(v)) = (&**ks, &**vs) else {
                        candle_core::bail!("attention_tree: prefix K/V not on Metal")
                    };
                    if k.dtype() != DType::F32 || v.dtype() != DType::F32 {
                        candle_core::bail!("attention_tree: prefix K/V must be f32");
                    }
                    ((k.buffer(), contiguous(kl, "prefix K")? * 4), (v.buffer(), contiguous(vl, "prefix V")? * 4), *np)
                }
                None => ((qkv.buffer(), qo), (qkv.buffer(), qo), 0),
            };
            for chunk in us.chunks(MAXU) {
                let mut tab = Vec::with_capacity(chunk.len());
                let mut tiles = 0u32;
                for u in chunk {
                    if u.ranges.is_empty() || u.ranges.len() > 8 || u.ranges.last().unwrap().0 != u.q_off {
                        candle_core::bail!("attention_tree: unit needs 1..=8 ranges ending with its own block");
                    }
                    let mut c = MetalTreeUnitC {
                        np: np as u32,
                        q_off: u.q_off as u32,
                        q_len: u.q_len as u32,
                        tile0: tiles,
                        nr: u.ranges.len() as u32,
                        ..Default::default()
                    };
                    for (i, &(o, l)) in u.ranges.iter().enumerate() {
                        c.r_off[i] = o as u32;
                        c.r_len[i] = l as u32;
                    }
                    tab.push(c);
                    tiles += (rep * u.q_len).div_ceil(8 * nsg) as u32;
                }
                let guard = dev.command_encoder()?;
                let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
                enc.set_compute_pipeline_state(&p);
                enc.set_input_buffer(0, Some(qkv.buffer()), qo);
                enc.set_input_buffer(1, Some(pk.0), pk.1);
                enc.set_input_buffer(2, Some(pv.0), pv.1);
                enc.set_output_buffer(3, Some(&out), 0);
                enc.set_bytes_directly(4, std::mem::size_of_val(tab.as_slice()), tab.as_ptr().cast());
                let c = [tab.len() as u32, t as u32, nh as u32, nkv as u32];
                for (i, x) in c.iter().enumerate() {
                    enc.set_bytes(5 + i, x);
                }
                enc.set_bytes(9, &self.scale);
                enc.dispatch_thread_groups(
                    MTLSize { width: tiles as usize, height: nkv, depth: 1 },
                    MTLSize { width: 32 * nsg, height: 1, depth: 1 },
                );
            }
        }
        Ok((MetalStorage::new(out, dev.clone(), n, DType::F32), Shape::from((1, nh, t, HD))))
    }
}

/// f32 attention of all units (blocks of packed rows) of a forward (Metal), invariant to how questions are packed:
/// `qkv` = the flat [`qkv_rope`] output over `d.l` tokens, `pasts` = the prefix K / V the units refer to. Returns
/// `[1, NH, total, HD]`.
#[cfg(target_os = "macos")]
pub fn attention_tree_metal(
    qkv: &Tensor,
    units: &[MetalTreeUnit],
    pasts: &[(&Tensor, &Tensor)],
    d: HeadDims,
    scale: f32,
) -> Result<Tensor> {
    qkv.apply_op1_no_bwd(&AttentionTreeMetal { units, pasts, total: d.l, nh: d.nh, nkv: d.nkv, scale })
}

#[cfg(feature = "cuda")]
struct SplitHiLo;

#[cfg(feature = "cuda")]
impl CustomOp1 for SplitHiLo {
    fn name(&self) -> &'static str {
        "split-hilo"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;
        if x.dtype() != DType::F32 {
            candle_core::bail!("split_hilo: f32 input expected");
        }
        let n = xl.shape().elem_count();
        let dev = x.device.clone();
        let xv = view!(x, f32, contiguous(xl, "x")?);
        // SAFETY: both planes are written by the kernel.
        let out = unsafe { dev.alloc::<half::f16>(2 * n)? };
        let f = cu::func(&dev, "split_hilo_f32", "basal_fused", cu::ptx())?;
        let mut a = f.builder();
        let n64 = n as u64;
        a.arg(&xv).arg(&out).arg(&n64);
        // SAFETY: argument types and counts match the kernel signature.
        unsafe { a.launch(cu::cfg(n)) }.w()?;
        Ok((CudaStorage { slice: Cs::F16(out), device: dev }, Shape::from((2 * n,))))
    }
}

/// f32 `x` (contiguous, n elements) -> f16 `[hi: n | lo: n]` with hi = f16(x), lo = f16(x - hi) (CUDA).
#[cfg(feature = "cuda")]
pub fn split_hilo(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&SplitHiLo)
}

/// Device address of the first element of a contiguous f16 CUDA tensor.
#[cfg(feature = "cuda")]
pub fn device_addr_f16(t: &Tensor) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    let (st, l) = t.storage_and_layout();
    let candle_core::Storage::Cuda(cs) = &*st else { candle_core::bail!("device_addr_f16: not a CUDA tensor") };
    let start = contiguous(l, "split prefix K/V")?;
    let slice = <half::f16 as candle_core::cuda_backend::CudaDType>::as_cuda_slice(cs)?;
    let stream = cs.device.cuda_stream();
    let (p, _g) = slice.device_ptr(&stream);
    Ok(p + (start * 2) as u64)
}
