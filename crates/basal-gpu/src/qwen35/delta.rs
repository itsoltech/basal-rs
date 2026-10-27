//! Gated DeltaNet mixer: projections, causal depthwise convolution, gate preparation, FP32 recurrence and the
//! gated RMS norm. Each stage has a fused CUDA path and a Candle reference with the same rounding points.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Tensor};
use serde_json::json;

use super::{normalized, DeltaNet, Qwen35Backend, State};
use crate::StageGates;

/// Recurrence inputs: FP32 `q`, `k`, `v` `[n, value_heads, dim]` and `decay`, `beta` `[n, value_heads]`.
struct Prepared {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    decay: Tensor,
    beta: Tensor,
    /// The CUDA preparation output `([3,n,h,128] qkv, [2,n,h] gates)` that the tensors above view.
    #[cfg_attr(not(feature = "cuda"), expect(dead_code, reason = "read only by the CUDA packed recurrence"))]
    packed: Option<(Tensor, Tensor)>,
}

impl Qwen35Backend {
    /// Starts opt-in synchronized diagnostic timings; see [`Self::trace_mark`].
    fn trace_start(&self) -> Result<Option<Instant>> {
        if crate::plan::DiagnosticOverrides::process().host_timing() {
            self.dev.synchronize()?;
            Ok(Some(Instant::now()))
        } else {
            Ok(None)
        }
    }

    /// Opt-in synchronized diagnostic timings; the ordinary forward never adds a synchronization here.
    fn trace_mark(&self, start: Option<Instant>, stage: &str, tokens: usize) -> Result<Option<Instant>> {
        if let Some(start) = start {
            self.dev.synchronize()?;
            eprintln!(
                "{}",
                json!({"qwen35_stage":stage,"tokens":tokens,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0})
            );
            Ok(Some(Instant::now()))
        } else {
            Ok(None)
        }
    }

    /// `projected` is this row's slice of a GEMM shared by the group; without it the merged input projection, or
    /// else each separate projection, runs on `x` alone.
    pub(super) fn delta(
        &self,
        w: &DeltaNet,
        x: &Tensor,
        state: &mut State,
        projected: Option<&Tensor>,
        gates: &mut StageGates<'_>,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let n = x.dim(0)?;
        let (vh, vd) = (c.linear_value_heads, c.linear_value_dim);
        let channels = 2 * c.linear_key_heads * c.linear_key_dim + vh * vd;
        let mut trace = self.trace_start()?;
        let projected = match projected {
            Some(p) => Some(p.clone()),
            None => w.input.as_ref().map(|w| self.linear(x, w)).transpose()?,
        };
        let project = |weight: &Tensor, offset: usize, width: usize| -> Result<Tensor> {
            match &projected {
                Some(p) => Ok(p.narrow(1, offset, width)?),
                None => self.linear(x, weight),
            }
        };
        let raw = project(&w.qkv, 0, channels)?;
        let a = project(&w.a, channels + vh * vd, vh)?;
        let b = project(&w.b, channels + vh * vd + vh, vh)?;
        trace = self.trace_mark(trace, "delta-projection", n)?;
        let mixed = self.convolve(w, raw, x.dtype(), state)?;
        trace = self.trace_mark(trace, "delta-convolution", n)?;
        let prepared = self.prepare(w, &mixed, &a, &b)?;
        let recurrent = match state.recurrent.take() {
            Some(recurrent) => recurrent,
            None => Tensor::zeros((vh, c.linear_key_dim, vd), DType::F32, &self.dev)?,
        };
        trace = self.trace_mark(trace, "delta-preparation", n)?;
        let (outputs, recurrent) = self.recur(&prepared, recurrent, gates)?;
        state.recurrent = Some(recurrent);
        trace = self.trace_mark(trace, "delta-recurrence", n)?;
        let out = if outputs.len() == 1 { outputs[0].clone() } else { Tensor::cat(&outputs, 0)? };
        let out = out.to_dtype(x.dtype())?;
        let z = project(&w.z, channels, vh * vd)?.reshape((n, vh, vd))?;
        let out = self.gated_norm(&out, &z, &w.norm, x.dtype())?;
        self.trace_mark(trace, "delta-gated-norm", n)?;
        Ok(out.reshape((n, vh * vd))?)
    }

    /// Causal depthwise convolution of `raw` after the cached history, followed by SiLU. The last
    /// `conv_kernel - 1` input rows become the next history.
    fn convolve(&self, w: &DeltaNet, raw: Tensor, dtype: DType, state: &mut State) -> Result<Tensor> {
        let c = &self.cfg;
        let (n, channels) = raw.dims2()?;
        let history = match state.conv.take() {
            Some(history) => history,
            None => Tensor::zeros((c.conv_kernel - 1, channels), dtype, &self.dev)?,
        };
        let all = Tensor::cat(&[history, raw], 0)?;
        let conv_w = w.conv.to_dtype(DType::F32)?.reshape((channels, c.conv_kernel))?;
        let mixed = self.depthwise_conv(&all, &conv_w, n, dtype)?;
        state.conv = Some(all.narrow(0, n, c.conv_kernel - 1)?.contiguous()?);
        Ok(mixed)
    }

    fn depthwise_conv(&self, all: &Tensor, weight: &Tensor, n: usize, dtype: DType) -> Result<Tensor> {
        let c = &self.cfg;
        #[cfg(feature = "cuda")]
        if self.dev.is_cuda() && c.conv_kernel == 4 {
            return Ok(crate::fused::qwen35_conv(all, weight)?);
        }
        let mut conv = Tensor::zeros((n, weight.dim(0)?), DType::F32, &self.dev)?;
        for tap in 0..c.conv_kernel {
            conv = (conv
                + all.narrow(0, tap, n)?.to_dtype(DType::F32)?.broadcast_mul(&weight.narrow(1, tap, 1)?.t()?)?)?;
        }
        // Depthwise convolution rounds its accumulation before SiLU in the model dtype.
        Ok(candle_nn::ops::silu(&conv.to_dtype(dtype)?)?)
    }

    /// L2-normalized, head-expanded Q/K, FP32 V and the decay/beta gates of the convolved `mixed` channels.
    fn prepare(&self, w: &DeltaNet, mixed: &Tensor, a: &Tensor, b: &Tensor) -> Result<Prepared> {
        let c = &self.cfg;
        let (kh, vh, kd, vd) = (c.linear_key_heads, c.linear_value_heads, c.linear_key_dim, c.linear_value_dim);
        #[cfg(feature = "cuda")]
        if self.dev.is_cuda() && kh == vh && kd == 128 && vd == 128 {
            let (qkv, gates) = crate::fused::qwen35_prepare(mixed, a, b, &w.params)?;
            return Ok(Prepared {
                q: qkv.get(0)?,
                k: qkv.get(1)?,
                v: qkv.get(2)?,
                decay: gates.get(0)?,
                beta: gates.get(1)?,
                packed: Some((qkv, gates)),
            });
        }
        let n = mixed.dim(0)?;
        let q = normalized(&mixed.narrow(1, 0, kh * kd)?.reshape((n, kh, kd))?, 1e-6, false)?;
        let k = normalized(&mixed.narrow(1, kh * kd, kh * kd)?.reshape((n, kh, kd))?, 1e-6, false)?;
        let v = mixed.narrow(1, 2 * kh * kd, vh * vd)?.reshape((n, vh, vd))?.to_dtype(DType::F32)?;
        let map = Tensor::from_vec((0..vh).map(|h| (h / (vh / kh)) as u32).collect::<Vec<_>>(), vh, &self.dev)?;
        let q = (q.index_select(&map, 1)? / (kd as f64).sqrt())?;
        let k = k.index_select(&map, 1)?;
        let beta = candle_nn::ops::sigmoid(b)?.to_dtype(DType::F32)?;
        let a = a.to_dtype(DType::F32)?.broadcast_add(&w.dt_bias.to_dtype(DType::F32)?)?;
        // Stable softplus, including large positive inputs.
        let softplus = (a.maximum(0.0)? + (a.abs()?.neg()?.exp()? + 1.0)?.log()?)?;
        let decay = softplus.broadcast_mul(&w.a_log.exp()?)?.neg()?.exp()?;
        Ok(Prepared { q, k, v, decay, beta, packed: None })
    }

    /// Runs the recurrence from `recurrent`, returning the FP32 outputs (one `[n,h,d]` tensor, or one `[1,h,d]`
    /// tensor per token) and the final state.
    fn recur(&self, p: &Prepared, mut recurrent: Tensor, gates: &mut StageGates<'_>) -> Result<(Vec<Tensor>, Tensor)> {
        let c = &self.cfg;
        #[cfg(feature = "cuda")]
        if self.dev.is_cuda() && c.linear_key_dim == 128 && c.linear_value_dim == 128 {
            gates.checkpoint()?;
            let wide = self.kernel_set == crate::KernelSet::Qwen35V4;
            let (out, next) = if let Some((qkv, packed_gates)) =
                p.packed.as_ref().filter(|_| crate::plan::DiagnosticOverrides::process().qwen_packed_recurrence())
            {
                crate::fused::qwen35_recurrent_packed(qkv, packed_gates, &recurrent, wide)?
            } else {
                crate::fused::qwen35_recurrent(&p.q, &p.k, &p.v, &p.decay, &p.beta, &recurrent, wide)?
            };
            return Ok((vec![out], next));
        }
        let vh = c.linear_value_heads;
        let n = p.q.dim(0)?;
        let mut outputs = Vec::with_capacity(n);
        for t in 0..n {
            gates.checkpoint()?;
            let key = p.k.get(t)?.unsqueeze(2)?;
            recurrent = recurrent.broadcast_mul(&p.decay.get(t)?.reshape((vh, 1, 1))?)?;
            let memory = recurrent.broadcast_mul(&key)?.sum(1)?;
            let delta = (p.v.get(t)? - memory)?.broadcast_mul(&p.beta.get(t)?.unsqueeze(1)?)?;
            recurrent = (recurrent + key.broadcast_mul(&delta.unsqueeze(1)?)?)?;
            outputs.push(recurrent.broadcast_mul(&p.q.get(t)?.unsqueeze(2)?)?.sum(1)?.unsqueeze(0)?);
        }
        Ok((outputs, recurrent))
    }

    /// Gated RMS uses direct FP32 weights, unlike zero-centered decoder RMSNorm.
    fn gated_norm(&self, x: &Tensor, z: &Tensor, weight: &Tensor, dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.dev.is_cuda() && self.cfg.linear_value_dim <= 256 {
            return Ok(crate::fused::qwen35_gated_norm(x, z, weight, self.cfg.rms_eps)?);
        }
        Ok(normalized(x, self.cfg.rms_eps, true)?
            .to_dtype(dtype)?
            .to_dtype(DType::F32)?
            .broadcast_mul(weight)?
            .mul(&candle_nn::ops::silu(&z.to_dtype(DType::F32)?)?)?
            .to_dtype(dtype)?)
    }
}
