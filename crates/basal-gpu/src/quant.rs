//! Optional weight quantization of the transformer projections.
//!
//! `q8_0` (W8A16, CUDA with `--dtype f16` and `--kernels fused` only): the four projection matrices of every layer
//! (q|k|v, o, gate|up, down) stay on the GPU as GGML Q8_0 blocks (32 values along K: an f16 scale and 32 int8),
//! quantized once on the host from the exact bf16 checkpoint values and uploaded with Candle's
//! `quantized::cuda::load_quantized`. Before every projection GEMM the matrix is dequantized into a temporary f16
//! tensor that feeds the same cuBLASLt f16 GEMM as dense weights; activations stay f16. Embedding, norms, biases,
//! lm_head, attention, K/V caches, readout and the evidence head are unchanged. `none` keeps the dense weights.

use std::fmt;
use std::str::FromStr;

use anyhow::{ensure, Context, Result};
use basal_core::manifest::LlamaConfig;
#[cfg(feature = "cuda")]
use candle_core::quantized::k_quants::BlockQ8_0;
#[cfg(feature = "cuda")]
use candle_core::quantized::GgmlType;
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use serde_json::{json, Value};

use crate::{projection_shapes, Kernels, Precision};

/// Values of one Q8_0 block, consecutive along K.
pub(crate) const Q8_BLOCK: usize = 32;
/// Bytes of one Q8_0 block: f16 scale and 32 int8.
pub(crate) const Q8_BLOCK_BYTES: usize = 34;
/// Zeroed bytes Candle allocates after the blocks of every matrix (`MATRIX_ROW_PADDING` = 512 values of Q8_0).
pub(crate) const Q8_PADDING_BYTES: usize = 544;

#[cfg(feature = "cuda")]
const _: () = {
    assert!(std::mem::size_of::<BlockQ8_0>() == Q8_BLOCK_BYTES);
    assert!(BlockQ8_0::BLCK_SIZE == Q8_BLOCK);
    assert!(
        Q8_PADDING_BYTES
            == candle_core::quantized::cuda::MATRIX_ROW_PADDING * Q8_BLOCK_BYTES
                / candle_core::quantized::k_quants::QK8_0
    );
};

/// Format of the resident projection weights: `quantization` of a served model, `--quantization` of the commands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quantization {
    /// Dense weights in the storage precision of `--dtype` (default; the forward is unchanged).
    #[default]
    None,
    /// GGML Q8_0 projection weights, dequantized to f16 before each projection GEMM (CUDA, `--dtype f16`,
    /// `--kernels fused`).
    Q8_0,
}

impl Quantization {
    /// The values `quantization` / `--quantization` accept, for messages.
    pub const VALUES: &'static str = "none, q8_0";

    /// The configuration value of this format.
    pub fn as_str(self) -> &'static str {
        match self {
            Quantization::None => "none",
            Quantization::Q8_0 => "q8_0",
        }
    }

    /// Whether this format runs with `precision` and `kernels` on a backend: `q8_0` needs the CUDA backend
    /// (`cuda_backend`; a build with the `cuda` feature, or the CUDA build a Linux package runs), f16 and fused
    /// kernels. `none` runs everywhere.
    pub fn check(self, precision: Precision, kernels: Kernels, cuda_backend: bool) -> Result<()> {
        if self == Quantization::None {
            return Ok(());
        }
        let need = "quantization q8_0 runs on CUDA with dtype f16 and kernels fused; quantization none runs with \
                    every backend, dtype and kernel set";
        ensure!(cuda_backend, "quantization q8_0 needs the CUDA backend, this build has {} ({need})", backend_name());
        ensure!(
            precision == Precision::F16,
            "quantization q8_0 needs dtype f16 (f16 activations), not {} ({need})",
            precision_name(precision)
        );
        ensure!(kernels == Kernels::Fused, "quantization q8_0 needs kernels fused, not candle ({need})");
        Ok(())
    }
}

fn backend_name() -> &'static str {
    crate::BUILD_BACKEND
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::Bf16 => "bf16",
        Precision::F16 => "f16",
        Precision::F32 => "f32",
    }
}

impl fmt::Display for Quantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Quantization {
    type Err = ParseQuantizationError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "none" => Ok(Quantization::None),
            "q8_0" => Ok(Quantization::Q8_0),
            _ => Err(ParseQuantizationError { value: s.to_string() }),
        }
    }
}

/// An unknown weight quantization name ([`Quantization::VALUES`] lists the known ones).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseQuantizationError {
    value: String,
}

impl ParseQuantizationError {
    /// The rejected value.
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Display for ParseQuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown quantization {:?} ({})", self.value, Quantization::VALUES)
    }
}

impl std::error::Error for ParseQuantizationError {}

/// Estimated GPU bytes of a model's weights (format arithmetic, not a measurement), for preflight checks: with
/// `None` exactly `checkpoint_bytes` (the estimate used without quantization); with `Q8_0` the checkpoint size minus
/// the 2-byte values of every projection matrix of every layer plus its Q8_0 blocks and Candle's padding
/// (`N·K/32·34 + 544` bytes per matrix).
pub fn estimated_weight_bytes(cfg: &LlamaConfig, checkpoint_bytes: u64, quantization: Quantization) -> Result<u64> {
    if quantization == Quantization::None {
        return Ok(checkpoint_bytes);
    }
    let overflow = || anyhow::anyhow!("projection sizes overflow");
    let (mut dense, mut q8) = (0u64, 0u64);
    for (n, k) in projection_shapes(cfg) {
        ensure!(
            k.is_multiple_of(Q8_BLOCK),
            "projection {n}x{k}: K is not a multiple of the Q8_0 block size {Q8_BLOCK}"
        );
        let values = u64::try_from(n.checked_mul(k).ok_or_else(overflow)?)?;
        dense = values.checked_mul(2).and_then(|b| dense.checked_add(b)).ok_or_else(overflow)?;
        let blocks = values / Q8_BLOCK as u64;
        let bytes = blocks
            .checked_mul(Q8_BLOCK_BYTES as u64)
            .and_then(|b| b.checked_add(Q8_PADDING_BYTES as u64))
            .ok_or_else(overflow)?;
        q8 = q8.checked_add(bytes).ok_or_else(overflow)?;
    }
    let layers = u64::try_from(cfg.num_layers)?;
    let dense = dense.checked_mul(layers).ok_or_else(overflow)?;
    let q8 = q8.checked_mul(layers).ok_or_else(overflow)?;
    let rest = checkpoint_bytes.checked_sub(dense).with_context(|| {
        format!("checkpoint of {checkpoint_bytes} bytes is smaller than its projection weights ({dense} bytes)")
    })?;
    rest.checked_add(q8).ok_or_else(overflow)
}

/// Load diagnostics of the Q8_0 projection matrices of a model (sums over matrices).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Q8Stats {
    pub matrices: u64,
    pub values: u64,
    pub blocks: u64,
    /// Device bytes of the matrices with Candle's padding.
    pub allocated_bytes: u64,
    /// Bytes of the largest matrix as the temporary f16 tensor of its projection.
    pub largest_f16_bytes: u64,
    /// Blocks whose 32 values are all zero (scale 0, exact).
    pub zero_blocks: u64,
    /// Blocks with a nonzero value whose f16 scale rounds to 0 (amax / 127 <= 2^-25): they dequantize to zeros.
    pub zero_scale_blocks: u64,
    /// Blocks with a subnormal f16 scale (0 < scale < 2^-14): reduced scale precision.
    pub subnormal_scale_blocks: u64,
}

#[cfg(feature = "cuda")]
impl Q8Stats {
    /// Add the diagnostics of another matrix.
    pub fn add(&mut self, o: &Q8Stats) {
        self.matrices += o.matrices;
        self.values += o.values;
        self.blocks += o.blocks;
        self.allocated_bytes += o.allocated_bytes;
        self.largest_f16_bytes = self.largest_f16_bytes.max(o.largest_f16_bytes);
        self.zero_blocks += o.zero_blocks;
        self.zero_scale_blocks += o.zero_scale_blocks;
        self.subnormal_scale_blocks += o.subnormal_scale_blocks;
    }

    /// The `quantization` object of the backend description.
    pub fn describe(&self) -> Value {
        json!({
            "format": "q8_0",
            "scope": "q|k|v, o, gate|up and down projections of every layer; embedding, norms, biases, lm_head, attention, K/V caches and readout unchanged",
            "source": "bf16 checkpoint values (exact in f32), quantized once on the host",
            "blocks_layout": "GGML Q8_0: 32 values along K per block, f16 scale (amax / 127, round to nearest even) and 32 int8",
            "forward": "each projection dequantized to a temporary f16 tensor (Candle dequantize_block_q8_0_f16) feeding the f16 cuBLASLt GEMM; f16 activations",
            "matrices": self.matrices,
            "values": self.values,
            "blocks": self.blocks,
            "block_bytes": self.blocks * Q8_BLOCK_BYTES as u64,
            "allocated_bytes_with_padding": self.allocated_bytes,
            "dense_f16_bytes_replaced": self.values * 2,
            "largest_temporary_f16_bytes": self.largest_f16_bytes,
            "zero_blocks": self.zero_blocks,
            "zero_scale_blocks": self.zero_scale_blocks,
            "subnormal_scale_blocks": self.subnormal_scale_blocks,
        })
    }
}

/// A projection weight of a decoder layer.
///
/// Invariant of `Q8`: the blocks live on the CUDA device of the backend that loaded them (the same Candle `DeviceId`,
/// checked at load) and are dequantized only onto `q.device()`, so a dequantization never transfers between devices.
#[derive(Clone)]
pub(crate) enum ProjWeight {
    /// Dense `[N, K]` weight in the storage dtype.
    Dense(Tensor),
    /// Resident GGML Q8_0 blocks of an `[N, K]` weight, shared by the forks of a backend.
    #[cfg(feature = "cuda")]
    Q8(std::sync::Arc<candle_core::quantized::QTensor>),
}

impl ProjWeight {
    /// The dense tensor (the stock-candle forward needs one; `q8_0` with `--kernels candle` is rejected before load).
    pub fn dense(&self) -> Result<&Tensor> {
        match self {
            ProjWeight::Dense(t) => Ok(t),
            #[cfg(feature = "cuda")]
            ProjWeight::Q8(_) => anyhow::bail!("q8_0 projection weights need --kernels fused"),
        }
    }

    /// Values of the weight.
    #[cfg(feature = "cuda")]
    pub fn elem_count(&self) -> usize {
        match self {
            ProjWeight::Dense(t) => t.elem_count(),
            ProjWeight::Q8(q) => q.shape().elem_count(),
        }
    }
}

/// Little-endian bf16 bytes of a row-major `[rows, cols]` matrix -> its Q8_0 blocks, row by row on every core
/// (a row converted to f32 at a time, no f32 copy of the matrix). Blocks are independent, so the result is bitwise
/// the blocks of `BlockQ8_0::from_float` over the whole matrix. Every value must be finite (Candle's `amax` would
/// skip a NaN) and every f16 scale finite.
#[cfg(feature = "cuda")]
pub(crate) fn quantize_q8_0(bf16: &[u8], rows: usize, cols: usize) -> Result<(Vec<BlockQ8_0>, Q8Stats)> {
    ensure!(rows > 0 && cols > 0, "empty matrix [{rows}, {cols}]");
    ensure!(cols.is_multiple_of(Q8_BLOCK), "K = {cols} is not a multiple of the Q8_0 block size {Q8_BLOCK}");
    let values = rows.checked_mul(cols).with_context(|| format!("[{rows}, {cols}]: element count overflows"))?;
    let bytes = values.checked_mul(2).with_context(|| format!("[{rows}, {cols}]: byte count overflows"))?;
    ensure!(bf16.len() == bytes, "{} bytes of bf16 data, expected {bytes} for [{rows}, {cols}]", bf16.len());
    // exact: cols is a multiple of the block size
    let per_row = cols / Q8_BLOCK;
    let n_blocks = values / Q8_BLOCK;
    ensure!(n_blocks.checked_mul(Q8_BLOCK) == Some(values), "[{rows}, {cols}]: block count is not exact");
    let mut blocks = vec![BlockQ8_0::zeros(); n_blocks];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let rows_per_thread = rows.div_ceil(threads).max(1);
    let parts: Vec<Result<Q8Stats>> = std::thread::scope(|s| {
        let handles: Vec<_> = blocks
            .chunks_mut(rows_per_thread * per_row)
            .zip(bf16.chunks(rows_per_thread * cols * 2))
            .enumerate()
            .map(|(i, (ys, xs))| s.spawn(move || quantize_rows(xs, ys, cols, i * rows_per_thread)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("a quantization thread panicked"))))
            .collect()
    });
    let mut stats = Q8Stats::default();
    for p in parts {
        stats.add(&p?);
    }
    ensure!(stats.blocks == n_blocks as u64, "quantized {} blocks of {n_blocks}", stats.blocks);
    stats.matrices = 1;
    stats.values = values as u64;
    stats.allocated_bytes = (n_blocks as u64)
        .checked_mul(Q8_BLOCK_BYTES as u64)
        .and_then(|b| b.checked_add(Q8_PADDING_BYTES as u64))
        .context("block bytes overflow")?;
    stats.largest_f16_bytes = (values as u64).checked_mul(2).context("f16 bytes overflow")?;
    Ok((blocks, stats))
}

/// Quantize whole rows (`src`: bf16 bytes of `dst.len() / (cols / 32)` rows starting at `first_row`).
#[cfg(feature = "cuda")]
fn quantize_rows(src: &[u8], dst: &mut [BlockQ8_0], cols: usize, first_row: usize) -> Result<Q8Stats> {
    let per_row = cols / Q8_BLOCK;
    ensure!(
        dst.len().is_multiple_of(per_row) && Some(src.len()) == dst.len().checked_mul(Q8_BLOCK * 2),
        "rows from {first_row}: {} bytes for {} blocks",
        src.len(),
        dst.len()
    );
    let mut row = vec![0f32; cols];
    let mut stats = Q8Stats::default();
    for (r, (xs, ys)) in src.chunks_exact(cols * 2).zip(dst.chunks_exact_mut(per_row)).enumerate() {
        let at = first_row + r;
        for (o, c) in row.iter_mut().zip(xs.chunks_exact(2)) {
            *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
        }
        for (b, block) in row.chunks_exact(Q8_BLOCK).enumerate() {
            let mut amax = 0f32;
            for (j, &x) in block.iter().enumerate() {
                ensure!(x.is_finite(), "non-finite weight {x} at row {at}, column {}", b * Q8_BLOCK + j);
                amax = amax.max(x.abs());
            }
            // the scale BlockQ8_0::from_float stores: f16 of amax / 127 (round to nearest even)
            let scale = half::f16::from_f32(amax / 127.0);
            ensure!(
                scale.is_finite(),
                "row {at}, columns {}..{}: Q8_0 scale {} overflows f16",
                b * Q8_BLOCK,
                (b + 1) * Q8_BLOCK,
                amax / 127.0
            );
            if amax == 0.0 {
                stats.zero_blocks += 1;
            } else if scale == half::f16::ZERO {
                stats.zero_scale_blocks += 1;
            } else if scale.classify() == std::num::FpCategory::Subnormal {
                stats.subnormal_scale_blocks += 1;
            }
        }
        // from_float checks lengths only in debug builds: exactly 32 values per block
        ensure!(row.len() == Q8_BLOCK * ys.len(), "row {at}: {} values for {} blocks", row.len(), ys.len());
        BlockQ8_0::from_float(&row, ys);
        stats.blocks += ys.len() as u64;
    }
    Ok(stats)
}
