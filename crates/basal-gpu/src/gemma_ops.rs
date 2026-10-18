//! CUDA element-wise operations with Gemma's original FP16 rounding points.

use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::{
    cudarc::driver::{LaunchConfig, PushKernelArg},
    CudaStorageSlice, WrapErr,
};
use candle_core::{CpuStorage, CudaStorage, CustomOp1, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

fn contiguous(layout: &Layout) -> Result<(usize, usize)> {
    layout.contiguous_offsets().ok_or_else(|| candle_core::Error::Msg("Gemma fused operand must be contiguous".into()))
}

struct Rope;

impl CustomOp3 for Rope {
    fn name(&self) -> &'static str {
        "gemma-rope-f16"
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
        candle_core::bail!("Gemma fused RoPE requires CUDA FP16")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let (batch, tokens, heads, dim) = xl.shape().dims4()?;
        let count = xl.shape().elem_count();
        if [batch, tokens, heads, dim].iter().any(|&d| d == 0 || d > u32::MAX as usize)
            || dim % 2 != 0
            || count > u32::MAX as usize
            || count.checked_mul(std::mem::size_of::<half::f16>()).is_none()
            || cl.shape().dims4()? != (1, tokens, 1, dim)
            || sl.shape() != cl.shape()
            || !x.device.same_device(&cos.device)
            || !x.device.same_device(&sin.device)
        {
            candle_core::bail!("invalid Gemma fused RoPE shape or device");
        }
        let (xs, xe) = contiguous(xl)?;
        let (cs, ce) = contiguous(cl)?;
        let (ss, se) = contiguous(sl)?;
        let dev = x.device.clone();
        let (CudaStorageSlice::F16(x), CudaStorageSlice::F16(cos), CudaStorageSlice::F16(sin)) =
            (&x.slice, &cos.slice, &sin.slice)
        else {
            candle_core::bail!("Gemma fused RoPE requires FP16 operands");
        };
        let (x, cos, sin) = (x.slice(xs..xe), cos.slice(cs..ce), sin.slice(ss..se));
        // SAFETY: the owned output has count FP16 elements; the kernel below initializes
        // every element on the device's stream before any tensor can read it.
        let mut output = unsafe { dev.alloc::<half::f16>(count)? };
        let kernel = crate::fused::cu::func(&dev, "gemma_rope_f16", "basal_fused", crate::fused::cu::ptx)?;
        let mut args = kernel.builder();
        let (tokens, heads, dim, count) = (tokens as u32, heads as u32, dim as u32, count as u32);
        args.arg(&x).arg(&cos).arg(&sin).arg(&mut output).arg(&tokens).arg(&heads).arg(&dim).arg(&count);
        // SAFETY: validated contiguous X[B,N,H,D] and cos/sin[1,N,1,D] share one
        // device; D is nonzero/even and every dimension and element index fits u32.
        // The kernel writes each output once and reads only the paired feature and
        // the current token's cos/sin. The builder tracks buffer access on this
        // stream; input views and owned output stay alive through enqueue.
        unsafe { args.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Ok((CudaStorage { slice: CudaStorageSlice::F16(output), device: dev }, xl.shape().clone()))
    }
}

pub(crate) fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op3_no_bwd(cos, sin, &Rope)
}

enum Pointwise {
    Scale,
    GeluMul,
}

impl CustomOp2 for Pointwise {
    fn name(&self) -> &'static str {
        match self {
            Self::Scale => "gemma-scale-f16",
            Self::GeluMul => "gemma-gelu-mul-f16",
        }
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Gemma fused pointwise operation requires CUDA FP16")
    }

    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, rhs: &CudaStorage, rl: &Layout) -> Result<(CudaStorage, Shape)> {
        let count = xl.shape().elem_count();
        let rhs_valid = match self {
            Self::Scale => rl.shape().elem_count() == 1,
            Self::GeluMul => rl.shape() == xl.shape(),
        };
        if count == 0
            || count > u32::MAX as usize
            || count.checked_mul(std::mem::size_of::<half::f16>()).is_none()
            || !rhs_valid
            || !x.device.same_device(&rhs.device)
        {
            candle_core::bail!("invalid Gemma fused pointwise shape or device");
        }
        let (xs, xe) = contiguous(xl)?;
        let (ss, se) = contiguous(rl)?;
        let dev = x.device.clone();
        let (CudaStorageSlice::F16(x), CudaStorageSlice::F16(rhs)) = (&x.slice, &rhs.slice) else {
            candle_core::bail!("Gemma fused pointwise operation requires FP16 operands");
        };
        let (x, rhs) = (x.slice(xs..xe), rhs.slice(ss..se));
        // SAFETY: this owned FP16 output is fully initialized by the same-stream
        // kernel below before it becomes visible through a returned tensor.
        let mut output = unsafe { dev.alloc::<half::f16>(count)? };
        let name = match self {
            Self::Scale => "gemma_scale_f16",
            Self::GeluMul => "gemma_gelu_mul_f16",
        };
        let kernel = crate::fused::cu::func(&dev, name, "basal_fused", crate::fused::cu::ptx)?;
        let mut args = kernel.builder();
        let count = count as u32;
        args.arg(&x).arg(&rhs).arg(&mut output).arg(&count);
        // SAFETY: contiguous FP16 operands share this device. The selected kernel
        // reads either the validated one-element scale or a same-shaped second
        // input. X and output have count elements and count fits u32. Each thread
        // writes its output once. The
        // builder tracks stream accesses and all buffers survive through enqueue.
        unsafe { args.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        Ok((CudaStorage { slice: CudaStorageSlice::F16(output), device: dev }, xl.shape().clone()))
    }
}

pub(crate) fn scale(x: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(scale, &Pointwise::Scale)
}

/// Keep Candle's FP16 GELU intermediates, then its FP16 element-wise product.
pub(crate) fn gelu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.contiguous()?.apply_op2_no_bwd(&up.contiguous()?, &Pointwise::GeluMul)
}

struct AttentionPack {
    heads: usize,
    transpose: bool,
}

impl CustomOp1 for AttentionPack {
    fn name(&self) -> &'static str {
        "gemma-attention-pack-f32"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Gemma attention packing requires CUDA FP16")
    }

    fn cuda_fwd(&self, x: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let (batch, tokens, source_heads, dim) = layout.shape().dims4()?;
        let count = batch.checked_mul(self.heads).and_then(|v| v.checked_mul(tokens)).and_then(|v| v.checked_mul(dim));
        let Some(count) = count.filter(|&v| v > 0 && v <= u32::MAX as usize) else {
            candle_core::bail!("invalid Gemma attention packing size");
        };
        if source_heads == 0
            || !self.heads.is_multiple_of(source_heads)
            || layout.shape().elem_count() > u32::MAX as usize
            || count.checked_mul(std::mem::size_of::<f32>()).is_none()
        {
            candle_core::bail!("invalid Gemma attention packing heads");
        }
        let (start, end) = contiguous(layout)?;
        let CudaStorageSlice::F16(input) = &x.slice else {
            candle_core::bail!("Gemma attention packing requires FP16 input");
        };
        let input = input.slice(start..end);
        let dev = x.device.clone();
        // SAFETY: the kernel initializes all count owned FP32 elements on this
        // device's stream before the returned tensor can be consumed.
        let mut output = unsafe { dev.alloc::<f32>(count)? };
        let kernel = crate::fused::cu::func(&dev, "gemma_attention_pack_f32", "basal_fused", crate::fused::cu::ptx)?;
        let mut args = kernel.builder();
        let (tokens, source_heads, heads, dim, count, transpose) = (
            tokens as u32,
            source_heads as u32,
            self.heads as u32,
            dim as u32,
            count as u32,
            u32::from(self.transpose),
        );
        args.arg(&input)
            .arg(&mut output)
            .arg(&tokens)
            .arg(&source_heads)
            .arg(&heads)
            .arg(&dim)
            .arg(&count)
            .arg(&transpose);
        // SAFETY: a nonempty contiguous [B,N,H,D] FP16 input has H > 0 and the
        // target head count is a positive multiple of H. Both input and output
        // element counts fit u32, so each mapped source index is in bounds.
        // Each thread writes one distinct output. Buffer access is tracked on
        // this stream and both allocations survive through enqueue.
        unsafe { args.launch(LaunchConfig::for_num_elems(count)) }.w()?;
        let shape = if self.transpose {
            (batch * self.heads, dim as usize, tokens as usize)
        } else {
            (batch * self.heads, tokens as usize, dim as usize)
        };
        Ok((CudaStorage { slice: CudaStorageSlice::F32(output), device: dev }, shape.into()))
    }
}

/// Reorder `[B,N,H,D]`, repeat GQA heads and widen FP16 to FP32 in one pass.
pub(crate) fn attention_pack(x: &Tensor, heads: usize, transpose: bool) -> Result<Tensor> {
    x.contiguous()?.apply_op1_no_bwd(&AttentionPack { heads, transpose })
}
