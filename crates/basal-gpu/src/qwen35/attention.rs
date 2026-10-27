//! Gated full attention with partial RoPE. Three paths share the projections and K/V cache: fused short-context
//! CUDA attention, query-tiled CUDA v3/v4 attention and the Candle reference.

use anyhow::Result;
use candle_core::{DType, Tensor};

use super::{norm, Attention, PrefillRow, Qwen35Backend, State};
use crate::Precision;

impl Qwen35Backend {
    /// Text positions coincide on all three MRoPE axes, reducing interleaved MRoPE to ordinary partial RoPE.
    fn rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (n, heads, d) = x.dims3()?;
        let r = self.cfg.rotary_dim;
        let half = r / 2;
        let a = x.narrow(2, 0, half)?;
        let b = x.narrow(2, half, half)?;
        let left = (a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?;
        let right = (b.broadcast_mul(cos)? + a.broadcast_mul(sin)?)?;
        let mut parts = vec![left, right];
        if r < d {
            parts.push(x.narrow(2, r, d - r)?);
        }
        Ok(Tensor::cat(&parts, 2)?.reshape((n, heads, d))?)
    }

    pub(super) fn attention(&self, w: &Attention, x: &Tensor, state: &mut State, row: &PrefillRow) -> Result<Tensor> {
        let c = &self.cfg;
        let (cos, sin) = (&row.cos, &row.sin);
        let n = x.dim(0)?;
        let qg = self.linear(x, &w.q)?.reshape((n, c.heads, 2 * c.head_dim))?;
        let q = self
            .rotary(&norm(&qg.narrow(2, 0, c.head_dim)?, &w.q_norm, c.rms_eps)?, cos, sin)?
            .transpose(0, 1)?
            .contiguous()?;
        let gate = candle_nn::ops::sigmoid(&qg.narrow(2, c.head_dim, c.head_dim)?.reshape((n, c.heads * c.head_dim))?)?;
        let k = self
            .rotary(
                &norm(&self.linear(x, &w.k)?.reshape((n, c.kv_heads, c.head_dim))?, &w.k_norm, c.rms_eps)?,
                cos,
                sin,
            )?
            .transpose(0, 1)?
            .contiguous()?;
        let v = self.linear(x, &w.v)?.reshape((n, c.kv_heads, c.head_dim))?.transpose(0, 1)?.contiguous()?;
        let (k, v) = match state.kv.take() {
            Some((old_k, old_v)) => (Tensor::cat(&[old_k, k], 1)?, Tensor::cat(&[old_v, v], 1)?),
            None => (k, v),
        };
        let out = self.attend(&q, &k, &v, row, x.dtype())?;
        state.kv = Some((k, v));
        Ok((out * gate)?)
    }

    /// `[n, heads * head_dim]` output of `q` `[heads, n, head_dim]` at the end of the cached `k`/`v`
    /// `[kv_heads, total, head_dim]`, rounded to the input `dtype`.
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, row: &PrefillRow, dtype: DType) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let c = &self.cfg;
            let (n, total) = (q.dim(1)?, k.dim(1)?);
            if super::optimized_kernels(self.kernel_set)
                && self.precision != Precision::F32
                && self.dev.is_cuda()
                && c.head_dim == 256
                && total <= 2048
                && (self.kernel_set == crate::KernelSet::Qwen35V2 || row.short_context)
            {
                return Ok(crate::fused::qwen35_attention(q, k, v, total - n)?.reshape((n, c.heads * c.head_dim))?);
            }
            if self.positional_attention() {
                return self.tiled_attention(q, k, v, dtype);
            }
        }
        self.reference_attention(q, k, v, &row.mask, dtype)
    }

    /// CUDA v3/v4: query tiles of at most 16 Mi scores per head group, with causal positions applied on the GPU.
    #[cfg(feature = "cuda")]
    fn tiled_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, dtype: DType) -> Result<Tensor> {
        let c = &self.cfg;
        let (n, total) = (q.dim(1)?, k.dim(1)?);
        let map = &self.weights.get().head_map;
        let dt = if self.tensor_attention { dtype } else { DType::F32 };
        let key = k.index_select(map, 0)?.to_dtype(dt)?;
        let value = v.index_select(map, 0)?.to_dtype(dt)?;
        let query = q.to_dtype(dt)?;
        let tile = (16 * 1024 * 1024 / (total * c.heads)).clamp(1, n);
        let mut outputs = Vec::new();
        for start in (0..n).step_by(tile) {
            let len = tile.min(n - start);
            let query = query.narrow(1, start, len)?.contiguous()?;
            let scores = if self.tensor_attention {
                crate::fused::qwen35_attention_scores(&query, &key)?
            } else {
                (query.matmul(&key.transpose(1, 2)?)? / (c.head_dim as f64).sqrt())?
            };
            let probabilities = crate::fused::qwen35_masked_softmax(
                &scores,
                map,
                crate::fused::GemmaMask { query_start: total - n + start, key_start: 0, window: 0, use_groups: false },
                dt,
            )?;
            outputs.push(probabilities.to_dtype(dt)?.matmul(&value)?.to_dtype(dtype)?);
        }
        Ok(Tensor::cat(&outputs, 1)?.transpose(0, 1)?.contiguous()?.reshape((n, c.heads * c.head_dim))?)
    }

    /// Candle FP32 attention with a dense causal `mask`: all heads at once for v2 up to 16 Mi scores, otherwise
    /// one head at a time.
    fn reference_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, dtype: DType) -> Result<Tensor> {
        let c = &self.cfg;
        let (n, total) = (q.dim(1)?, k.dim(1)?);
        let batched = self.kernel_set == crate::KernelSet::Qwen35V2
            && self.precision != Precision::F32
            && self.dev.is_cuda()
            && n.saturating_mul(total).saturating_mul(c.heads) <= 16 * 1024 * 1024;
        if batched {
            let map = &self.weights.get().head_map;
            let key = k.index_select(map, 0)?.to_dtype(DType::F32)?;
            let value = v.index_select(map, 0)?.to_dtype(DType::F32)?;
            let scores = q.to_dtype(DType::F32)?.matmul(&key.transpose(1, 2)?)?;
            let scores = (scores / (c.head_dim as f64).sqrt())?;
            let probabilities = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
            let output = probabilities.matmul(&value)?;
            return Ok(output.to_dtype(dtype)?.transpose(0, 1)?.contiguous()?.reshape((n, c.heads * c.head_dim))?);
        }
        let mut outputs = Vec::with_capacity(c.heads);
        // Large contexts keep one head's scores at a time; the batched path above caps scores at 64 MiB.
        for h in 0..c.heads {
            let kh = k.get(h / (c.heads / c.kv_heads))?.to_dtype(DType::F32)?;
            let vh = v.get(h / (c.heads / c.kv_heads))?.to_dtype(DType::F32)?;
            let scores = (q.get(h)?.to_dtype(DType::F32)?.matmul(&kh.t()?)? / (c.head_dim as f64).sqrt())?;
            let probabilities = candle_nn::ops::softmax_last_dim(&(scores + mask)?)?;
            outputs.push(probabilities.matmul(&vh)?.to_dtype(dtype)?.unsqueeze(1)?);
        }
        Ok(Tensor::cat(&outputs, 1)?.reshape((n, c.heads * c.head_dim))?)
    }
}
