//! Qwen3.5 CUDA operations (kernels.cu `qwen35_*`): decoder norms, SwiGLU, short-context attention, half-precision
//! attention scores and the DeltaNet convolution, preparation, recurrence and gated norm. The shared positional
//! masked softmax stays with the Gemma mask in the parent module.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaStorageSlice as Cs;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, Result, Shape, Tensor};

use super::{contiguous, cu, no_cpu};

/// Half-precision Q/K products with FP32 accumulation and FP32 score storage. cuBLAS sees the transpose
/// of the row-major output: K * Q^T, with one independent matrix per query head.
struct Qwen35AttentionScores;

impl CustomOp2 for Qwen35AttentionScores {
    fn name(&self) -> &'static str {
        "qwen35-attention-scores"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, q: &CudaStorage, ql: &Layout, k: &CudaStorage, kl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::{
            cublas::{result, sys},
            driver::{DevicePtr, DevicePtrMut},
        };
        use candle_core::cuda_backend::WrapErr;
        let (heads, queries, dim) = ql.shape().dims3()?;
        let (kh, keys, kd) = kl.shape().dims3()?;
        if heads == 0
            || queries == 0
            || keys == 0
            || dim == 0
            || kh != heads
            || kd != dim
            || q.dtype() != k.dtype()
            || !matches!(q.dtype(), DType::BF16 | DType::F16)
            || q.device.id() != k.device.id()
        {
            candle_core::bail!("qwen35 attention scores: incompatible shapes, dtype or device");
        }
        let checked = |v: usize| i32::try_from(v).map_err(candle_core::Error::wrap);
        let (h, m, n, d) = (checked(heads)?, checked(keys)?, checked(queries)?, checked(dim)?);
        let total = heads
            .checked_mul(queries)
            .and_then(|v| v.checked_mul(keys))
            .ok_or_else(|| candle_core::Error::Msg("qwen35 attention scores: size overflow".into()))?;
        let dev = q.device.clone();
        let stream = dev.cuda_stream();
        let handle = dev.cublas_handle();
        let qoff = contiguous(ql, "attention query")?;
        let koff = contiguous(kl, "attention key")?;
        // SAFETY: beta is zero and GEMM initializes every element of the newly owned output before use.
        let mut out = unsafe { dev.alloc::<f32>(total)? };
        let (output, output_guard) = out.device_ptr_mut(&stream);
        let alpha = 1.0_f32 / (dim as f32).sqrt();
        let beta = 0.0_f32;
        macro_rules! gemm {
            ($ty:ty, $dtype:expr) => {{
                let query = view!(q, $ty, qoff);
                let key = view!(k, $ty, koff);
                let (qp, _qguard) = query.device_ptr(&stream);
                let (kp, _kguard) = key.device_ptr(&stream);
                // SAFETY: validated contiguous [head,row,dim] slices have matching dimensions and device.
                // Checked i32 dimensions bound each i64 stride product. The output is unique, correctly
                // sized FP32 storage; alpha/beta stay alive through the call in the handle's default host
                // pointer mode. Candle owns the handle on this same stream; no handle settings are changed.
                // Device guards retain input/output allocation lifetimes through the enqueued GEMM.
                unsafe {
                    result::gemm_strided_batched_ex(
                        *handle.handle(),
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        m,
                        n,
                        d,
                        (&alpha as *const f32).cast(),
                        kp as *const _,
                        $dtype,
                        d,
                        i64::from(m) * i64::from(d),
                        qp as *const _,
                        $dtype,
                        d,
                        i64::from(n) * i64::from(d),
                        (&beta as *const f32).cast(),
                        output as *mut _,
                        sys::cudaDataType::CUDA_R_32F,
                        m,
                        i64::from(m) * i64::from(n),
                        h,
                        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .w()?;
            }};
        }
        match q.dtype() {
            DType::BF16 => gemm!(half::bf16, sys::cudaDataType::CUDA_R_16BF),
            DType::F16 => gemm!(half::f16, sys::cudaDataType::CUDA_R_16F),
            _ => unreachable!("validated half precision"),
        }
        drop(output_guard);
        Ok((CudaStorage { slice: Cs::F32(out), device: dev }, Shape::from((heads, queries, keys))))
    }
}

pub(crate) fn qwen35_attention_scores(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    q.apply_op2_no_bwd(k, &Qwen35AttentionScores)
}

struct Qwen35Attention {
    start: usize,
}

impl CustomOp3 for Qwen35Attention {
    fn name(&self) -> &'static str {
        "qwen35-attention"
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
        q: &CudaStorage,
        ql: &Layout,
        k: &CudaStorage,
        kl: &Layout,
        v: &CudaStorage,
        vl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (h, n, d) = ql.shape().dims3()?;
        let (kv, total, kd) = kl.shape().dims3()?;
        if h == 0
            || kv == 0
            || h % kv != 0
            || n == 0
            || n > 512
            || d != 256
            || kd != d
            || self.start.checked_add(n) != Some(total)
            || total > 2048
            || vl.shape() != kl.shape()
            || !matches!(q.dtype(), DType::F16 | DType::BF16)
            || q.dtype() != k.dtype()
            || k.dtype() != v.dtype()
            || q.device.id() != k.device.id()
            || k.device.id() != v.device.id()
        {
            candle_core::bail!("Qwen3.5 fused attention: incompatible operands");
        }
        let dev = q.device.clone();
        let count = n * h * d;
        let storage = by_dtype!(q.dtype(), |T, S| {
            let qq = view!(q, T, contiguous(ql, "attention Q")?);
            let kk = view!(k, T, contiguous(kl, "attention K")?);
            let vv = view!(v, T, contiguous(vl, "attention V")?);
            // SAFETY: each query/head warp writes its 256 output elements exactly once.
            let out = unsafe { dev.alloc::<T>(count)? };
            let f = cu::func(&dev, &format!("qwen35_attention_{S}"), "basal_fused", cu::ptx)?;
            let mut a = f.builder();
            let (n_, total_, h_, kv_, start_) = (n as u32, total as u32, h as u32, kv as u32, self.start as u32);
            a.arg(&qq).arg(&kk).arg(&vv).arg(&out).arg(&n_).arg(&total_).arg(&h_).arg(&kv_).arg(&start_);
            let cfg =
                LaunchConfig { grid_dim: ((n_ * h_).div_ceil(4), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: checked contiguous half-precision layouts and absolute positions match the signature. Four
            // complete warps per block own distinct queries/heads; tail warps return before warp collectives.
            // Inputs remain immutable, output is fresh, and all allocations live on this stream through launch.
            unsafe { a.launch(cfg) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((storage, Shape::from((n, h, d))))
    }
}

pub fn qwen35_attention(q: &Tensor, k: &Tensor, v: &Tensor, start: usize) -> Result<Tensor> {
    q.apply_op3_no_bwd(k, v, &Qwen35Attention { start })
}

struct Qwen35SwiGlu;

impl CustomOp1 for Qwen35SwiGlu {
    fn name(&self) -> &'static str {
        "qwen35-swiglu"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, gu: &CudaStorage, gl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
        use candle_core::cuda_backend::WrapErr;
        let (n, two_width) = gl.shape().dims2()?;
        if n == 0 || two_width == 0 || two_width % 2 != 0 {
            candle_core::bail!("qwen35 SwiGLU: invalid shape");
        }
        let width = two_width / 2;
        let total = n * width;
        let dev = gu.device.clone();
        let st = by_dtype!(gu.dtype(), |T, S| {
            let g = view!(gu, T, contiguous(gl, "gate/up projection")?);
            // SAFETY: every output element is written once by its owning thread.
            let out = unsafe { dev.alloc::<T>(total)? };
            let f = cu::func(&dev, &format!("qwen35_swiglu_{S}"), "basal_fused", cu::ptx)?;
            let mut a = f.builder();
            let (w, count) = (width as u32, total as u32);
            a.arg(&g).arg(&out).arg(&w).arg(&count);
            // SAFETY: checked contiguous [N,2I] input matches the kernel's two [N,I] halves. Launch bounds
            // cover the fresh [N,I] output. Both allocations remain alive on the owning stream through launch.
            unsafe { a.launch(cu::cfg(total)) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, Shape::from((n, width))))
    }
}

pub fn qwen35_swiglu(gu: &Tensor) -> Result<Tensor> {
    gu.apply_op1_no_bwd(&Qwen35SwiGlu)
}

struct Qwen35Unary {
    eps: Option<f32>,
}

impl CustomOp2 for Qwen35Unary {
    fn name(&self) -> &'static str {
        "qwen35-norm-or-conv"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        no_cpu()
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let dims = xl.shape().dims();
        let width = dims.last().copied().unwrap_or(0);
        if width == 0 || w.dtype() != DType::F32 || w.device.id() != x.device.id() {
            candle_core::bail!("qwen35 norm/conv: invalid width, weight dtype or device");
        }
        let mut shape = xl.shape().clone();
        if self.eps.is_some() {
            if wl.shape().dims() != [width] {
                candle_core::bail!("qwen35 norm: invalid weights");
            }
        } else {
            let (rows, _) = xl.shape().dims2()?;
            if rows <= 3 || wl.shape().dims() != [width, 4] {
                candle_core::bail!("qwen35 convolution: invalid shapes");
            }
            shape = Shape::from((rows - 3, width));
        }
        let total = shape.elem_count();
        let dev = x.device.clone();
        let ww = view!(w, f32, contiguous(wl, "norm/conv weights")?);
        let st = by_dtype!(x.dtype(), |T, S| {
            let xx = view!(x, T, contiguous(xl, "norm/conv input")?);
            // SAFETY: each kernel writes every element of the new output exactly once.
            let out = unsafe { dev.alloc::<T>(total)? };
            let name = format!("qwen35_{}_{S}", if self.eps.is_some() { "norm" } else { "conv" });
            let f = cu::func(&dev, &name, "basal_fused", cu::ptx)?;
            let mut a = f.builder();
            let (width_, total_) = (width as u32, total as u32);
            let eps = self.eps.unwrap_or(0.0);
            let power = -0.5f32;
            a.arg(&xx).arg(&ww).arg(&out).arg(&width_);
            let cfg = if self.eps.is_some() {
                a.arg(&eps).arg(&power);
                LaunchConfig {
                    grid_dim: ((total / width) as u32, 1, 1),
                    block_dim: (width.min(1024).next_power_of_two() as u32, 1, 1),
                    shared_mem_bytes: 0,
                }
            } else {
                a.arg(&total_);
                cu::cfg(total)
            };
            // SAFETY: checked contiguous shapes and FP32 weights match the selected signature; the convolution
            // reads exactly three history rows plus the output rows. Norm uses at most 1024 shared slots.
            // Inputs and the fresh, unaliased output remain alive on the owning device's stream through launch.
            unsafe { a.launch(cfg) }.w()?;
            <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
        });
        Ok((st, shape))
    }
}

pub fn qwen35_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(&w.to_dtype(DType::F32)?, &Qwen35Unary { eps: Some(eps as f32) })
}

pub fn qwen35_conv(all: &Tensor, w: &Tensor) -> Result<Tensor> {
    all.apply_op2_no_bwd(w, &Qwen35Unary { eps: None })
}

struct Qwen35DeltaOp {
    eps: Option<f32>,
}

impl CustomOp3 for Qwen35DeltaOp {
    fn name(&self) -> &'static str {
        "qwen35-delta-prepare-or-gate"
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
        w: &CudaStorage,
        wl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        if x.dtype() != y.dtype()
            || w.dtype() != DType::F32
            || x.device.id() != y.device.id()
            || x.device.id() != w.device.id()
        {
            candle_core::bail!("qwen35 DeltaNet: incompatible dtype or device");
        }
        let prepare = self.eps.is_none();
        let (tokens, heads, width, shape) = if prepare {
            let (n, channels) = xl.shape().dims2()?;
            let (planes, yn, h) = yl.shape().dims3()?;
            if n == 0 || planes != 2 || yn != n || h == 0 || channels != 3 * h * 128 || wl.shape().dims() != [2, h] {
                candle_core::bail!("qwen35 DeltaNet: invalid preparation shapes");
            }
            (n, h, 128, Shape::from(n * h * (3 * 128 + 2)))
        } else {
            let (n, h, d) = xl.shape().dims3()?;
            if n == 0 || h == 0 || d == 0 || d > 256 || yl.shape() != xl.shape() || wl.shape().dims() != [d] {
                candle_core::bail!("qwen35 DeltaNet: invalid gated norm shapes");
            }
            (n, h, d, xl.shape().clone())
        };
        let dev = x.device.clone();
        let ww = view!(w, f32, contiguous(wl, "DeltaNet parameters")?);
        let storage = by_dtype!(x.dtype(), |T, S| {
            let xx = view!(x, T, contiguous(xl, "DeltaNet input")?);
            let yy = view!(y, T, contiguous(yl, "DeltaNet gates")?);
            let name = format!("qwen35_{}_{S}", if prepare { "prepare" } else { "gated_norm" });
            let f = cu::func(&dev, &name, "basal_fused", cu::ptx)?;
            let (n, h, d) = (tokens as u32, heads as u32, width as u32);
            let eps = self.eps.unwrap_or(0.0);
            let power = -0.5f32;
            let cfg =
                LaunchConfig { grid_dim: (n * h, 1, 1), block_dim: (d.next_power_of_two(), 1, 1), shared_mem_bytes: 0 };
            if prepare {
                // SAFETY: one block writes all three 128-element vectors and both scalar gates for its token/head.
                let out = unsafe { dev.alloc::<f32>(shape.elem_count())? };
                let mut a = f.builder();
                a.arg(&xx).arg(&yy).arg(&ww).arg(&out).arg(&n).arg(&h).arg(&power);
                // SAFETY: contiguous validated layouts match the preparation signature. Exactly 128 threads
                // reduce each Q/K head and lane zero writes its gates; operands live on this stream through launch.
                unsafe { a.launch(cfg) }.w()?;
                CudaStorage { slice: Cs::F32(out), device: dev.clone() }
            } else {
                // SAFETY: each valid lane writes its output; padding lanes participate only in the reduction.
                let out = unsafe { dev.alloc::<T>(shape.elem_count())? };
                let mut a = f.builder();
                a.arg(&xx).arg(&yy).arg(&ww).arg(&out).arg(&d).arg(&eps).arg(&power);
                // SAFETY: checked equal contiguous input shapes and the FP32 weight match the gated-norm
                // signature. At most 256 threads use 256 shared slots. Fresh output has no input aliases;
                // every allocation remains alive on the owning device's stream through launch.
                unsafe { a.launch(cfg) }.w()?;
                <T as candle_core::cuda_backend::CudaDType>::wrap_cuda_slice(out, dev.clone())
            }
        });
        Ok((storage, shape))
    }
}

pub fn qwen35_prepare(mixed: &Tensor, a: &Tensor, b: &Tensor, params: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n, h) = a.dims2()?;
    if b.shape() != a.shape() {
        candle_core::bail!("qwen35 DeltaNet: scalar gate shapes disagree");
    }
    let ab = Tensor::stack(&[a, b], 0)?;
    let out = mixed.apply_op3_no_bwd(&ab, params, &Qwen35DeltaOp { eps: None })?;
    let plane = n * h * 128;
    // Immutable contiguous views of the same allocation: no repacking or change to kernel arithmetic.
    Ok((
        out.narrow(0, 0, 3 * plane)?.reshape((3, n, h, 128))?,
        out.narrow(0, 3 * plane, 2 * n * h)?.reshape((2, n, h))?,
    ))
}

pub fn qwen35_gated_norm(x: &Tensor, z: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    x.apply_op3_no_bwd(&z.contiguous()?, w, &Qwen35DeltaOp { eps: Some(eps as f32) })
}

struct Qwen35Recurrent {
    wide: bool,
}

impl CustomOp3 for Qwen35Recurrent {
    fn name(&self) -> &'static str {
        "qwen35-recurrent"
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
        qkv: &CudaStorage,
        ql: &Layout,
        gates: &CudaStorage,
        gl: &Layout,
        state: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (planes, tokens, heads, dim) = ql.shape().dims4()?;
        if planes != 3
            || tokens == 0
            || tokens > 2048
            || heads == 0
            || dim != 128
            || gl.shape().dims() != [2, tokens, heads]
            || sl.shape().dims() != [heads, 128, 128]
            || [qkv.dtype(), gates.dtype(), state.dtype()].iter().any(|d| *d != DType::F32)
            || qkv.device.id() != gates.device.id()
            || qkv.device.id() != state.device.id()
        {
            candle_core::bail!("qwen35 recurrence: incompatible shapes, dtype or device");
        }
        let dev = qkv.device.clone();
        let q = view!(qkv, f32, contiguous(ql, "qkv")?);
        let g = view!(gates, f32, contiguous(gl, "decay/beta")?);
        let s = view!(state, f32, contiguous(sl, "recurrent state")?);
        let total = tokens * heads * 128 + heads * 128 * 128;
        // SAFETY: the kernel writes every output and final-state element exactly once.
        let out = unsafe { dev.alloc::<f32>(total)? };
        let name = if self.wide { "qwen35_recurrent_wide_f32" } else { "qwen35_recurrent_f32" };
        let f = cu::func(&dev, name, "basal_fused", cu::ptx)?;
        let mut a = f.builder();
        let (n, h) = (tokens as u32, heads as u32);
        a.arg(&q).arg(&g).arg(&s).arg(&out).arg(&n).arg(&h);
        let cfg = LaunchConfig {
            grid_dim: (h, if self.wide { 16 } else { 32 }, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: validated contiguous FP32 operands match the kernel signature. Each warp owns one or two
        // distinct value columns; the grid covers all 128 columns per head, with 128 key rows. Inputs remain
        // alive and immutable on the owning stream
        // through launch, and the newly allocated output has no aliases. All accesses fit the checked shapes.
        unsafe { a.launch(cfg) }.w()?;
        Ok((CudaStorage { slice: Cs::F32(out), device: dev }, Shape::from(total)))
    }
}

/// Reference FP32 DeltaNet arithmetic with a register-resident `[head,value,key]` state (128-dimensional heads).
pub fn qwen35_recurrent(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    wide: bool,
) -> Result<(Tensor, Tensor)> {
    let (n, h, _) = q.dims3()?;
    if k.shape() != q.shape() || v.shape() != q.shape() || decay.dims() != [n, h] || beta.shape() != decay.shape() {
        candle_core::bail!("qwen35 recurrence: input shapes disagree");
    }
    let qkv = Tensor::stack(&[q, k, v], 0)?;
    let gates = Tensor::stack(&[decay, beta], 0)?;
    qwen35_recurrent_packed(&qkv, &gates, state, wide)
}

/// Consume the contiguous prepared planes directly. CustomOp3 validates dtype, dimensions and device identity;
/// input views stay immutable and own their underlying allocation through the asynchronous launch.
pub fn qwen35_recurrent_packed(qkv: &Tensor, gates: &Tensor, state: &Tensor, wide: bool) -> Result<(Tensor, Tensor)> {
    let (_, n, h, d) = qkv.dims4()?;
    let out = qkv.apply_op3_no_bwd(gates, state, &Qwen35Recurrent { wide })?;
    Ok((out.narrow(0, 0, n * h * d)?.reshape((n, h, d))?, out.narrow(0, n * h * d, h * d * d)?.reshape((h, d, d))?))
}
