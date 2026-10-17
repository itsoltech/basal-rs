//! Optional weight quantization of the transformer projections.
//!
//! CUDA with `--dtype f16` and `--kernels fused` only: the four projection matrices of every layer (q|k|v, o,
//! gate|up, down) stay on the GPU as GGML blocks, quantized once on the host from the exact bf16 checkpoint values and
//! uploaded with Candle's `quantized::cuda::load_quantized`. Before every projection GEMM the matrix is dequantized
//! into a temporary f16 tensor that feeds the same cuBLASLt f16 GEMM as dense weights; activations stay f16.
//! Embedding, norms, biases, lm_head, attention, K/V caches, readout and the evidence head are unchanged. `none` keeps
//! the dense weights.
//!
//! - `q8_0` (W8A16): Q8_0 blocks, 32 values along K with an f16 scale and 32 int8.
//! - `q4_k` (W4A16): Q4_K super-blocks, 256 values along K in 8 groups of 32 with 6-bit group scales and minima
//!   relative to an f16 `d` and `dmin`, and 4-bit codes. The host pass also reconstructs every matrix
//!   (`BlockQ4K::to_float`, f16 round to nearest even), checks that every source value and reconstruction is finite
//!   and that no group of 32 is a constant non-zero (this pinned quantizer reconstructs it as zeros), and sums the
//!   reconstruction error in a fixed order (`quantize_q4_k`). The Q4_K dequantization kernel has no bounds check, so
//!   the loader checks its memory invariants (K a multiple of 256, at most `i32::MAX` values, the exact block count
//!   and bytes) before any launch.

use std::fmt;
use std::str::FromStr;

#[cfg(feature = "cuda")]
use anyhow::{anyhow, bail};
use anyhow::{ensure, Context, Result};
use basal_core::manifest::LlamaConfig;
#[cfg(feature = "cuda")]
use candle_core::quantized::k_quants::{BlockQ4K, BlockQ8_0};
#[cfg(feature = "cuda")]
use candle_core::quantized::{GgmlDType, GgmlType, QTensor};
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

/// Values of one Q4_K super-block, consecutive along K (8 groups of 32).
pub(crate) const Q4K_BLOCK: usize = 256;
/// Bytes of one Q4_K super-block: f16 `d` and `dmin`, 12 bytes of 6-bit group scales and minima, 128 bytes of codes.
pub(crate) const Q4K_BLOCK_BYTES: usize = 144;
/// Zeroed bytes Candle allocates after the blocks of every matrix (`MATRIX_ROW_PADDING` = 512 values of Q4_K).
pub(crate) const Q4K_PADDING_BYTES: usize = 288;
/// Values of a group of the q4_k load diagnostics: a Q4_K sub-block with its own 6-bit scale and minimum.
#[cfg(feature = "cuda")]
const Q4K_GROUP: usize = 32;
/// Rows per chunk of the canonical reduction of the q4_k load statistics, the value the W4 P0 weight diagnostic
/// recorded. Fixed and independent of the core count; it shapes only the summation tree of the diagnostics, not the
/// blocks or any inference result.
#[cfg(feature = "cuda")]
const Q4K_CHUNK_ROWS: usize = 16;

#[cfg(feature = "cuda")]
const _: () = {
    assert!(std::mem::size_of::<BlockQ4K>() == Q4K_BLOCK_BYTES);
    assert!(BlockQ4K::BLCK_SIZE == Q4K_BLOCK);
    assert!(candle_core::quantized::k_quants::QK_K == Q4K_BLOCK);
    assert!(Q4K_BLOCK.is_multiple_of(Q4K_GROUP));
    assert!(Q4K_PADDING_BYTES == candle_core::quantized::cuda::MATRIX_ROW_PADDING * Q4K_BLOCK_BYTES / Q4K_BLOCK);
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
    /// GGML Q4_K projection weights, dequantized to f16 before each projection GEMM (CUDA, `--dtype f16`,
    /// `--kernels fused`; every projection's K a multiple of 256).
    Q4K,
}

/// Block layout of a quantized format: the format arithmetic of the estimates (the CUDA build asserts it against
/// Candle's block types and padding).
#[derive(Clone, Copy)]
struct Layout {
    /// GGML name, for messages.
    ggml: &'static str,
    /// Values per block along K: K must be a multiple.
    block: usize,
    block_bytes: usize,
    /// Zeroed bytes Candle allocates after the blocks of every matrix.
    padding_bytes: usize,
}

impl Quantization {
    /// The values `quantization` / `--quantization` accept, for messages.
    pub const VALUES: &'static str = "none, q8_0, q4_k";

    /// The configuration value of this format.
    pub fn as_str(self) -> &'static str {
        match self {
            Quantization::None => "none",
            Quantization::Q8_0 => "q8_0",
            Quantization::Q4K => "q4_k",
        }
    }

    fn layout(self) -> Option<Layout> {
        match self {
            Quantization::None => None,
            Quantization::Q8_0 => Some(Layout {
                ggml: "Q8_0",
                block: Q8_BLOCK,
                block_bytes: Q8_BLOCK_BYTES,
                padding_bytes: Q8_PADDING_BYTES,
            }),
            Quantization::Q4K => Some(Layout {
                ggml: "Q4_K",
                block: Q4K_BLOCK,
                block_bytes: Q4K_BLOCK_BYTES,
                padding_bytes: Q4K_PADDING_BYTES,
            }),
        }
    }

    /// Values per block along K of a quantized format (every projection's K must be a multiple of it); `None` for
    /// `none`.
    pub fn block_values(self) -> Option<usize> {
        self.layout().map(|l| l.block)
    }

    /// Whether this format runs with `precision` and `kernels` on a backend: a quantized format needs the CUDA
    /// backend (`cuda_backend`; a build with the `cuda` feature, or the CUDA build a Linux package runs), f16 and fused
    /// kernels. `none` runs everywhere.
    pub fn check(self, precision: Precision, kernels: Kernels, cuda_backend: bool) -> Result<()> {
        if self == Quantization::None {
            return Ok(());
        }
        let name = self.as_str();
        let need = format!(
            "quantization {name} runs on CUDA with dtype f16 and kernels fused; quantization none runs with \
             every backend, dtype and kernel set"
        );
        ensure!(cuda_backend, "quantization {name} needs the CUDA backend, this build has {} ({need})", backend_name());
        ensure!(
            precision == Precision::F16,
            "quantization {name} needs dtype f16 (f16 activations), not {} ({need})",
            precision_name(precision)
        );
        ensure!(kernels == Kernels::Fused, "quantization {name} needs kernels fused, not candle ({need})");
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
            "q4_k" => Ok(Quantization::Q4K),
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
/// `None` exactly `checkpoint_bytes` (the estimate used without quantization); with a quantized format the checkpoint
/// size minus the 2-byte values of every projection matrix of every layer plus its blocks and Candle's padding
/// (`Q8_0`: `N·K/32·34 + 544` bytes per matrix; `Q4K`: `N·K/256·144 + 288`).
pub fn estimated_weight_bytes(cfg: &LlamaConfig, checkpoint_bytes: u64, quantization: Quantization) -> Result<u64> {
    let Some(layout) = quantization.layout() else {
        return Ok(checkpoint_bytes);
    };
    let overflow = || anyhow::anyhow!("projection sizes overflow");
    let (mut dense, mut quantized) = (0u64, 0u64);
    for (n, k) in projection_shapes(cfg) {
        ensure!(
            k.is_multiple_of(layout.block),
            "projection {n}x{k}: K is not a multiple of the {} block size {}",
            layout.ggml,
            layout.block
        );
        let values = u64::try_from(n.checked_mul(k).ok_or_else(overflow)?)?;
        dense = values.checked_mul(2).and_then(|b| dense.checked_add(b)).ok_or_else(overflow)?;
        let blocks = values / layout.block as u64;
        let bytes = blocks
            .checked_mul(layout.block_bytes as u64)
            .and_then(|b| b.checked_add(layout.padding_bytes as u64))
            .ok_or_else(overflow)?;
        quantized = quantized.checked_add(bytes).ok_or_else(overflow)?;
    }
    let layers = u64::try_from(cfg.num_layers)?;
    let dense = dense.checked_mul(layers).ok_or_else(overflow)?;
    let quantized = quantized.checked_mul(layers).ok_or_else(overflow)?;
    let rest = checkpoint_bytes.checked_sub(dense).with_context(|| {
        format!("checkpoint of {checkpoint_bytes} bytes is smaller than its projection weights ({dense} bytes)")
    })?;
    rest.checked_add(quantized).ok_or_else(overflow)
}

/// Estimated GPU bytes of the weights of one stage of a layer split (format arithmetic for preflight checks, not a
/// measurement): the decoder layers' part of [`estimated_weight_bytes`] in proportion to the stage's `layers` of
/// `cfg.num_layers`, plus the embedding on the `first` stage and lm_head and final norm on the `last` one (each
/// `vocab × hidden` or `hidden` bf16 values of the checkpoint). Not included: the copy of the next stage's input norm
/// (`hidden` values) and the evidence head, a separate file.
pub fn estimated_stage_weight_bytes(
    cfg: &LlamaConfig,
    checkpoint_bytes: u64,
    quantization: Quantization,
    layers: usize,
    first: bool,
    last: bool,
) -> Result<u64> {
    ensure!(
        cfg.num_layers > 0 && layers <= cfg.num_layers,
        "a stage of {layers} layers in a model of {}",
        cfg.num_layers
    );
    let overflow = || anyhow::anyhow!("weight sizes overflow");
    let total = estimated_weight_bytes(cfg, checkpoint_bytes, quantization)?;
    let matrix = cfg.vocab.checked_mul(cfg.hidden).and_then(|v| v.checked_mul(2)).ok_or_else(overflow)?;
    let matrix = u64::try_from(matrix)?;
    let norm = u64::try_from(cfg.hidden.checked_mul(2).ok_or_else(overflow)?)?;
    let ends = matrix.checked_mul(2).and_then(|b| b.checked_add(norm)).ok_or_else(overflow)?;
    let decoder = total.checked_sub(ends).with_context(|| {
        format!("estimated weights of {total} bytes are smaller than the embedding, lm_head and norm ({ends} bytes)")
    })?;
    let share = u128::from(decoder) * layers as u128 / cfg.num_layers as u128;
    let mut bytes = u64::try_from(share)?;
    if first {
        bytes = bytes.checked_add(matrix).ok_or_else(overflow)?;
    }
    if last {
        bytes = bytes.checked_add(matrix).and_then(|b| b.checked_add(norm)).ok_or_else(overflow)?;
    }
    Ok(bytes)
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

/// Load diagnostics of the quantized projection matrices of a model: the `quantization` object of the backend
/// description.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub(crate) enum QuantStats {
    Q8(Q8Stats),
    Q4K(Box<Q4KStats>),
}

#[cfg(feature = "cuda")]
impl QuantStats {
    /// The `quantization` object of the backend description (`q8_0`: exactly [`Q8Stats::describe`]).
    pub fn describe(&self) -> Value {
        match self {
            QuantStats::Q8(s) => s.describe(),
            QuantStats::Q4K(s) => s.describe(),
        }
    }
}

/// A projection matrix of a decoder layer (the loader's order within a layer, also the canonical order of the q4_k
/// statistics).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Projection {
    /// [q; k; v]
    Qkv,
    O,
    /// [gate; up]
    GateUp,
    Down,
}

#[cfg(feature = "cuda")]
impl Projection {
    /// Every projection, in the order of [`projection_shapes`].
    pub const ALL: [Projection; 4] = [Projection::Qkv, Projection::O, Projection::GateUp, Projection::Down];

    /// Name in the q4_k diagnostics (as in the W4 P0 record).
    pub fn name(self) -> &'static str {
        match self {
            Projection::Qkv => "qkv",
            Projection::O => "o",
            Projection::GateUp => "gate_up",
            Projection::Down => "down",
        }
    }

    fn index(self) -> usize {
        match self {
            Projection::Qkv => 0,
            Projection::O => 1,
            Projection::GateUp => 2,
            Projection::Down => 3,
        }
    }
}

/// The source of a q4_k projection matrix: its layer and projection and its checkpoint tensors, concatenated along N in
/// this order (name, rows). Positions in messages and diagnostics name the original tensor and its row.
#[cfg(feature = "cuda")]
pub(crate) struct Source<'a> {
    pub layer: usize,
    pub projection: Projection,
    pub parts: &'a [(String, usize)],
}

#[cfg(feature = "cuda")]
impl Source<'_> {
    /// The tensor and its row of matrix row `row`.
    fn tensor_row(&self, row: usize) -> Option<(&str, usize)> {
        let mut first = 0usize;
        for (name, rows) in self.parts {
            let end = first.checked_add(*rows)?;
            if row < end {
                return Some((name, row - first));
            }
            first = end;
        }
        None
    }

    /// A value position for messages: tensor, its row and the column, then the matrix row.
    fn at(&self, row: usize, col: usize) -> String {
        let (layer, projection) = (self.layer, self.projection.name());
        match self.tensor_row(row) {
            Some((t, r)) => format!("{t} row {r}, column {col} (layer {layer} {projection} matrix row {row})"),
            None => format!("layer {layer} {projection} matrix row {row}, column {col}"),
        }
    }

    fn pos(&self, c: Cell) -> SourcePos {
        let (tensor, tensor_row) = match self.tensor_row(c.row) {
            Some((t, r)) => (Some(t.to_string()), Some(r)),
            None => (None, None),
        };
        SourcePos { layer: self.layer, projection: self.projection, tensor, tensor_row, matrix_row: c.row, col: c.col }
    }
}

/// Row of a projection matrix (after concatenation along N) and column.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
struct Cell {
    row: usize,
    col: usize,
}

/// A value of a projection's source tensors: layer, projection, tensor and its row, matrix row and column.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub(crate) struct SourcePos {
    layer: usize,
    projection: Projection,
    tensor: Option<String>,
    tensor_row: Option<usize>,
    matrix_row: usize,
    col: usize,
}

#[cfg(feature = "cuda")]
impl SourcePos {
    fn describe(&self) -> Value {
        json!({
            "layer": self.layer,
            "projection": self.projection.name(),
            "tensor": self.tensor,
            "tensor_row": self.tensor_row,
            "matrix_row": self.matrix_row,
            "col": self.col,
        })
    }
}

/// A counter with its first position; merged in canonical order, `first` is the canonical first.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
struct Count<P> {
    n: u64,
    first: Option<P>,
}

#[cfg(feature = "cuda")]
impl<P> Default for Count<P> {
    fn default() -> Self {
        Self { n: 0, first: None }
    }
}

#[cfg(feature = "cuda")]
impl<P: Clone> Count<P> {
    fn hit(&mut self, at: P) {
        self.n += 1;
        if self.first.is_none() {
            self.first = Some(at);
        }
    }

    fn merge(&mut self, o: &Count<P>) {
        self.n += o.n;
        if self.first.is_none() {
            self.first.clone_from(&o.first);
        }
    }
}

/// Reconstruction error and group counters of q4_k blocks (a chunk of rows, a matrix, a projection type or a model),
/// merged in the canonical order. `w` = source value (bf16, exact in f32 and f64), `r` = `BlockQ4K::to_float` rounded
/// to f16 (round to nearest even) as f64, `e = r − w`. Every value of a loaded matrix is finite (checked before the
/// sums), so every value is in the sums.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
pub(crate) struct Q4KErrors<P> {
    rows: u64,
    values: u64,
    sum_e2: f64,
    sum_w2: f64,
    max_abs_e: f64,
    /// Groups of 32 consecutive values along K.
    groups: u64,
    /// Source groups equal to 0 (f32 `==`, so a mixture of ±0 is a zero group).
    zero_groups: u64,
    /// Non-constant source groups with a constant (finite) f16 reconstruction: a 6-bit group scale 0 relative to the
    /// super-block, or a degenerate fit. Their error is in the sums; no threshold.
    flattened_groups: Count<P>,
    /// Zero source groups whose f16 reconstruction is not all 0 (the pinned quantizer gives exact zeros).
    zero_groups_nonzero_reconstruction: Count<P>,
}

#[cfg(feature = "cuda")]
impl<P> Default for Q4KErrors<P> {
    fn default() -> Self {
        Self {
            rows: 0,
            values: 0,
            sum_e2: 0.0,
            sum_w2: 0.0,
            max_abs_e: 0.0,
            groups: 0,
            zero_groups: 0,
            flattened_groups: Count::default(),
            zero_groups_nonzero_reconstruction: Count::default(),
        }
    }
}

#[cfg(feature = "cuda")]
impl<P: Clone> Q4KErrors<P> {
    fn merge(&mut self, o: &Q4KErrors<P>) {
        self.rows += o.rows;
        self.values += o.values;
        self.sum_e2 += o.sum_e2;
        self.sum_w2 += o.sum_w2;
        self.max_abs_e = self.max_abs_e.max(o.max_abs_e);
        self.groups += o.groups;
        self.zero_groups += o.zero_groups;
        self.flattened_groups.merge(&o.flattened_groups);
        self.zero_groups_nonzero_reconstruction.merge(&o.zero_groups_nonzero_reconstruction);
    }

    /// The f64 sums are finite (always with finite values in f64; checked anyway, so the description has no NaN).
    fn check_finite(&self, what: &str) -> Result<()> {
        ensure!(
            self.sum_e2.is_finite() && self.sum_w2.is_finite() && self.max_abs_e.is_finite(),
            "{what}: q4_k error sums are not finite (sum_e2 {}, sum_w2 {}, max |e| {})",
            self.sum_e2,
            self.sum_w2,
            self.max_abs_e
        );
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Q4KErrors<Cell> {
    fn locate(&self, source: &Source<'_>) -> Q4KErrors<SourcePos> {
        let count = |c: &Count<Cell>| Count { n: c.n, first: c.first.map(|p| source.pos(p)) };
        Q4KErrors {
            rows: self.rows,
            values: self.values,
            sum_e2: self.sum_e2,
            sum_w2: self.sum_w2,
            max_abs_e: self.max_abs_e,
            groups: self.groups,
            zero_groups: self.zero_groups,
            flattened_groups: count(&self.flattened_groups),
            zero_groups_nonzero_reconstruction: count(&self.zero_groups_nonzero_reconstruction),
        }
    }
}

#[cfg(feature = "cuda")]
impl Q4KErrors<SourcePos> {
    fn describe(&self) -> Value {
        let count = |c: &Count<SourcePos>| json!({ "count": c.n, "first": c.first.as_ref().map(SourcePos::describe) });
        // R = sum_e2 / sum_w2 (as in the W4 P0 record) is defined only for a finite, positive norm; otherwise null with
        // the reason, never NaN
        let ratio = if self.sum_w2.is_finite() && self.sum_w2 > 0.0 && (self.sum_e2 / self.sum_w2).is_finite() {
            let r = self.sum_e2 / self.sum_w2;
            json!({ "value": r, "sqrt": r.sqrt(), "status": "valid" })
        } else {
            json!({ "value": null, "sqrt": null, "status": "invalid", "reason": format!("sum_w2 is {} (must be finite and positive)", self.sum_w2) })
        };
        json!({
            "rows": self.rows,
            "values": self.values,
            "sum_e2": self.sum_e2,
            "sum_w2": self.sum_w2,
            "max_abs_error": self.max_abs_e,
            "rms_error": if self.values > 0 { json!((self.sum_e2 / self.values as f64).sqrt()) } else { Value::Null },
            "relative_error_ratio": ratio,
            "groups": {
                "total": self.groups,
                "zero": self.zero_groups,
                "flattened": count(&self.flattened_groups),
                "zero_with_nonzero_reconstruction": count(&self.zero_groups_nonzero_reconstruction),
            },
        })
    }
}

/// Load diagnostics of one q4_k projection matrix ([`quantize_q4_k`]).
#[cfg(feature = "cuda")]
pub(crate) struct Q4KMatrix {
    projection: Projection,
    blocks: u64,
    errors: Q4KErrors<SourcePos>,
}

#[cfg(feature = "cuda")]
impl Q4KMatrix {
    /// Q4_K blocks of the matrix.
    pub fn blocks(&self) -> u64 {
        self.blocks
    }
}

/// Load diagnostics of the q4_k projection matrices of a model. Deterministic: the same checkpoint gives the same
/// object on every device and with any number of host threads (no timings or thread counts).
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Default)]
pub(crate) struct Q4KStats {
    matrices: u64,
    blocks: u64,
    /// Device bytes of the matrices with Candle's padding.
    allocated_bytes: u64,
    /// Bytes of the largest matrix as the temporary f16 tensor of its projection.
    largest_f16_bytes: u64,
    /// Sums over every matrix: layers ascending, within a layer qkv, o, gate_up, down.
    model: Q4KErrors<SourcePos>,
    /// Sums per projection type (index of [`Projection`]), layers ascending.
    per_projection: [Q4KErrors<SourcePos>; 4],
    /// The matrix of the warm-up check: layer, projection, values.
    warmup: Option<(usize, Projection, u64)>,
}

#[cfg(feature = "cuda")]
impl Q4KStats {
    /// Add the diagnostics of the next matrix in the loader's (canonical) order.
    pub fn add(&mut self, m: &Q4KMatrix) -> Result<()> {
        let bytes = m
            .blocks
            .checked_mul(Q4K_BLOCK_BYTES as u64)
            .and_then(|b| b.checked_add(Q4K_PADDING_BYTES as u64))
            .context("block bytes overflow")?;
        self.matrices += 1;
        self.blocks = self.blocks.checked_add(m.blocks).context("block count overflows")?;
        self.allocated_bytes = self.allocated_bytes.checked_add(bytes).context("block bytes overflow")?;
        let f16_bytes = m.errors.values.checked_mul(2).context("f16 bytes overflow")?;
        self.largest_f16_bytes = self.largest_f16_bytes.max(f16_bytes);
        self.model.merge(&m.errors);
        self.model.check_finite("model")?;
        let p = &mut self.per_projection[m.projection.index()];
        p.merge(&m.errors);
        p.check_finite(m.projection.name())
    }

    /// Record the matrix the warm-up checked bitwise against the host reconstruction.
    pub fn set_warmup(&mut self, layer: usize, projection: Projection, values: u64) {
        self.warmup = Some((layer, projection, values));
    }

    /// The `quantization` object of the backend description.
    pub fn describe(&self) -> Value {
        let per_projection: serde_json::Map<String, Value> =
            Projection::ALL.iter().map(|p| (p.name().to_string(), self.per_projection[p.index()].describe())).collect();
        json!({
            "format": "q4_k",
            "scope": "q|k|v, o, gate|up and down projections of every layer; embedding, norms, biases, lm_head, attention, K/V caches and readout unchanged",
            "source": "bf16 checkpoint values (exact in f32), quantized once on the host",
            "blocks_layout": "GGML Q4_K: 256 values along K per super-block, 8 groups of 32 with 6-bit scales and minima relative to an f16 d and dmin, 4-bit codes (Candle BlockQ4K::from_float)",
            "forward": "each projection dequantized to a temporary f16 tensor (Candle dequantize_block_q4_K_f16) feeding the f16 cuBLASLt GEMM; f16 activations",
            "matrices": self.matrices,
            "values": self.model.values,
            "blocks": self.blocks,
            "block_bytes": self.blocks * Q4K_BLOCK_BYTES as u64,
            "allocated_bytes_with_padding": self.allocated_bytes,
            "dense_f16_bytes_replaced": self.model.values * 2,
            "largest_temporary_f16_bytes": self.largest_f16_bytes,
            "load_checks": "every source value, BlockQ4K::to_float value and its f16 reconstruction finite, and no constant non-zero source group of 32, in every matrix before its upload (a load error otherwise); K a multiple of 256, at most i32::MAX values, block count and bytes checked before any dequantization",
            "warmup_check": self.warmup.map(|(layer, projection, values)| json!({
                "layer": layer,
                "projection": projection.name(),
                "values": values,
                "result": "GPU dequantization bitwise equal to the host reconstruction (f16 bit patterns, signed zeros included)",
                "scope": "one representative matrix per load, not a check of the other matrices",
            })),
            "reconstruction_error": {
                "definition": "w = bf16 source value (exact in f64), r = f16 (round to nearest even) of BlockQ4K::to_float as f64, e = r - w; sum_e2, sum_w2, max_abs_error over every value; rms_error = sqrt(sum_e2 / values); relative_error_ratio = sum_e2 / sum_w2 (sqrt: relative RMS error)",
                "reduction": "chunks of chunk_rows rows: sequential over rows then elements (f64); matrix: sequential fold of chunk partials by chunk index from 0.0; model: layers ascending, within a layer qkv, o, gate_up, down; per projection type: layers ascending. Independent of the host thread count",
                "chunk_rows": Q4K_CHUNK_ROWS,
                "model": self.model.describe(),
                "per_projection": per_projection,
            },
        })
    }
}

/// A projection weight of a decoder layer.
///
/// Invariant of `Quantized`: the blocks live on the CUDA device of the backend that loaded them (the same Candle
/// `DeviceId`, checked at load) and are dequantized only onto `q.device()`, so a dequantization never transfers between
/// devices.
#[derive(Clone)]
pub(crate) enum ProjWeight {
    /// Dense `[N, K]` weight in the storage dtype.
    Dense(Tensor),
    /// Resident GGML blocks (Q8_0 or Q4_K, the type the `QTensor` carries) of an `[N, K]` weight, shared by the forks
    /// of a backend.
    #[cfg(feature = "cuda")]
    Quantized(std::sync::Arc<QTensor>),
}

impl ProjWeight {
    /// The dense tensor (the stock-candle forward needs one; a quantized format with `--kernels candle` is rejected
    /// before load).
    pub fn dense(&self) -> Result<&Tensor> {
        match self {
            ProjWeight::Dense(t) => Ok(t),
            #[cfg(feature = "cuda")]
            ProjWeight::Quantized(q) => match q.dtype() {
                GgmlDType::Q8_0 => anyhow::bail!("q8_0 projection weights need --kernels fused"),
                GgmlDType::Q4K => anyhow::bail!("q4_k projection weights need --kernels fused"),
                d => anyhow::bail!("{d:?} projection weights need --kernels fused"),
            },
        }
    }

    /// Values of the weight.
    #[cfg(feature = "cuda")]
    pub fn elem_count(&self) -> usize {
        match self {
            ProjWeight::Dense(t) => t.elem_count(),
            ProjWeight::Quantized(q) => q.shape().elem_count(),
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

/// Little-endian bf16 bytes of a row-major `[rows, cols]` projection matrix -> its Q4_K blocks and load diagnostics.
///
/// Checked before any work: positive dimensions, K a multiple of 256, `rows · cols` without overflow and at most
/// `i32::MAX` (the Q4_K dequantization kernel computes offsets in `int` and has no bounds check), the exact byte length
/// and the source rows of `source`. Per row, in this order (the first failing row of the matrix is reported, whatever
/// the thread count):
///
/// 1. exact f32 of every bf16 value; a NaN or ±inf is an error naming the tensor, row and column, with the row's NaN,
///    +inf and −inf counts;
/// 2. groups of 32 along K: a constant non-zero group is an error (this pinned Candle quantizer reconstructs it as
///    zeros: `make_qkx1_quants` returns `(0, 0)` for `min == max`), all-zero groups are counted;
/// 3. `BlockQ4K::from_float` into the row's blocks and `BlockQ4K::to_float` (both check lengths only in debug builds,
///    so the exact lengths are checked before each call);
/// 4. every `to_float` value and its f16 reconstruction (round to nearest even) must be finite, before any statistic
///    or upload (finite input can still give NaN or ±inf when `d` or `dmin` overflows f16); an error names the
///    position with the row's counts of both stages;
/// 5. f64 error sums over the row's values in column order; flattened groups (non-constant source, constant
///    reconstruction) counted with their error in the sums.
///
/// Rows form chunks of 16 ([`Q4K_CHUNK_ROWS`]); at most `available_parallelism` scoped workers each process one
/// contiguous range of whole chunks into the chunk-indexed partials, which are then folded in chunk order: the blocks
/// and the statistics do not depend on the number of workers. A worker that cannot be started is an error (no panic);
/// every started worker is joined before the first error is returned.
#[cfg(feature = "cuda")]
pub(crate) fn quantize_q4_k(
    bf16: &[u8],
    rows: usize,
    cols: usize,
    source: &Source<'_>,
) -> Result<(Vec<BlockQ4K>, Q4KMatrix)> {
    ensure!(rows > 0 && cols > 0, "empty matrix [{rows}, {cols}]");
    ensure!(cols.is_multiple_of(Q4K_BLOCK), "K = {cols} is not a multiple of the Q4_K block size {Q4K_BLOCK}");
    let values = rows.checked_mul(cols).with_context(|| format!("[{rows}, {cols}]: element count overflows"))?;
    let limit = usize::try_from(i32::MAX).context("i32::MAX as usize")?;
    ensure!(
        values <= limit,
        "[{rows}, {cols}]: {values} values exceed i32::MAX, the limit of the Q4_K dequantization kernel"
    );
    let row_bytes = cols.checked_mul(2).with_context(|| format!("[{rows}, {cols}]: row bytes overflow"))?;
    let bytes = rows.checked_mul(row_bytes).with_context(|| format!("[{rows}, {cols}]: byte count overflows"))?;
    ensure!(bf16.len() == bytes, "{} bytes of bf16 data, expected {bytes} for [{rows}, {cols}]", bf16.len());
    let part_rows = source.parts.iter().try_fold(0usize, |sum, (_, r)| sum.checked_add(*r));
    ensure!(part_rows == Some(rows), "source tensors of {part_rows:?} rows for a matrix of {rows} rows");
    // exact: cols is a multiple of the block size
    let per_row = cols / Q4K_BLOCK;
    let n_blocks = values / Q4K_BLOCK;
    ensure!(n_blocks.checked_mul(Q4K_BLOCK) == Some(values), "[{rows}, {cols}]: block count is not exact");
    let chunks = rows.div_ceil(Q4K_CHUNK_ROWS);
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(chunks).max(1);
    let mut blocks = vec![BlockQ4K::zeros(); n_blocks];
    let mut partials = vec![Q4KErrors::<Cell>::default(); chunks];
    let (base, extra) = (chunks / workers, chunks % workers);
    let results: Vec<Result<()>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(workers);
        let mut rest_partials: &mut [Q4KErrors<Cell>] = &mut partials;
        let mut rest_blocks: &mut [BlockQ4K] = &mut blocks;
        let mut first_chunk = 0usize;
        for w in 0..workers {
            // workers <= chunks, so every worker gets base >= 1 whole chunks (plus one for the first `extra`); the
            // row bounds stay below rows + Q4K_CHUNK_ROWS and their offsets within bf16 and the blocks
            let n = base + usize::from(w < extra);
            let r0 = first_chunk * Q4K_CHUNK_ROWS;
            let r1 = ((first_chunk + n) * Q4K_CHUNK_ROWS).min(rows);
            first_chunk += n;
            let Some((mine, tail)) = std::mem::take(&mut rest_partials).split_at_mut_checked(n) else {
                handles.push(Err(anyhow!("chunk partition: worker {w} of {workers} has no chunks")));
                break;
            };
            rest_partials = tail;
            let Some((ys, tail)) = std::mem::take(&mut rest_blocks).split_at_mut_checked((r1 - r0) * per_row) else {
                handles.push(Err(anyhow!("rows {r0}..{r1}: outside the blocks")));
                break;
            };
            rest_blocks = tail;
            let Some(xs) = bf16.get(r0 * row_bytes..r1 * row_bytes) else {
                handles.push(Err(anyhow!("rows {r0}..{r1} outside the matrix")));
                break;
            };
            // `Scope::spawn` panics when the OS refuses a thread; `spawn_scoped` returns the error instead
            let spawned = std::thread::Builder::new()
                .spawn_scoped(s, move || q4k_rows(xs, ys, mine, r0, cols, source))
                .with_context(|| format!("starting q4_k worker {w} of {workers} (rows {r0}..{r1})"));
            let failed = spawned.is_err();
            handles.push(spawned);
            if failed {
                break;
            }
        }
        // join every started worker (the collect does not stop at an error), so none outlives this call; in worker
        // order, so with every worker started the first error is that of the first failing row
        handles
            .into_iter()
            .map(|h| h.and_then(|h| h.join().unwrap_or_else(|_| Err(anyhow!("a q4_k quantization thread panicked")))))
            .collect()
    });
    for r in results {
        r?;
    }
    let mut errors = Q4KErrors::<Cell>::default();
    for p in &partials {
        errors.merge(p);
    }
    ensure!(errors.rows == rows as u64, "quantized {} rows of {rows}", errors.rows);
    ensure!(errors.values == values as u64, "quantized {} values of {values}", errors.values);
    errors.check_finite(&format!("layer {} {}", source.layer, source.projection.name()))?;
    let matrix = Q4KMatrix { projection: source.projection, blocks: n_blocks as u64, errors: errors.locate(source) };
    Ok((blocks, matrix))
}

/// Per-worker row buffers of [`q4k_rows`], reused for every row.
#[cfg(feature = "cuda")]
struct Q4KScratch {
    /// Source row, exact f32 of the bf16 values.
    x: Vec<f32>,
    /// `to_float` row, then its f16 reconstruction (as f32).
    y: Vec<f32>,
    group_constant: Vec<bool>,
    group_zero: Vec<bool>,
}

/// Quantize a contiguous range of whole chunks: `src` = its rows (first matrix row `first_row`), `dst` = their blocks,
/// `out` = the partials of its chunks.
#[cfg(feature = "cuda")]
fn q4k_rows(
    src: &[u8],
    dst: &mut [BlockQ4K],
    out: &mut [Q4KErrors<Cell>],
    first_row: usize,
    cols: usize,
    source: &Source<'_>,
) -> Result<()> {
    let row_bytes = cols.checked_mul(2).context("row bytes overflow")?;
    ensure!(src.len().is_multiple_of(row_bytes), "rows from {first_row}: partial row");
    let rows = src.len() / row_bytes;
    let per_row = cols / Q4K_BLOCK;
    ensure!(
        out.len() == rows.div_ceil(Q4K_CHUNK_ROWS) && Some(dst.len()) == rows.checked_mul(per_row),
        "rows from {first_row}: {} partials and {} blocks for {rows} rows",
        out.len(),
        dst.len()
    );
    let groups = cols / Q4K_GROUP;
    let mut s = Q4KScratch {
        x: vec![0f32; cols],
        y: vec![0f32; cols],
        group_constant: vec![false; groups],
        group_zero: vec![false; groups],
    };
    let chunk_bytes = Q4K_CHUNK_ROWS.checked_mul(row_bytes).context("chunk bytes overflow")?;
    let chunk_blocks = Q4K_CHUNK_ROWS.checked_mul(per_row).context("chunk blocks overflow")?;
    for (c, ((st, xs), ys)) in out.iter_mut().zip(src.chunks(chunk_bytes)).zip(dst.chunks_mut(chunk_blocks)).enumerate()
    {
        for (i, (row, blocks)) in xs.chunks_exact(row_bytes).zip(ys.chunks_exact_mut(per_row)).enumerate() {
            q4k_row(row, blocks, &mut s, st, first_row + c * Q4K_CHUNK_ROWS + i, source)?;
        }
    }
    Ok(())
}

/// NaN, +inf and −inf counts of `values`, for messages.
#[cfg(feature = "cuda")]
fn nonfinite_counts(values: impl Iterator<Item = f32>) -> String {
    let (mut nan, mut pos, mut neg) = (0usize, 0usize, 0usize);
    for v in values {
        if v.is_nan() {
            nan += 1;
        } else if v == f32::INFINITY {
            pos += 1;
        } else if v == f32::NEG_INFINITY {
            neg += 1;
        }
    }
    format!("{nan} NaN, {pos} +inf, {neg} -inf")
}

/// One row of [`quantize_q4_k`] (steps 1–5) into `blocks` and the chunk partial `st`.
#[cfg(feature = "cuda")]
fn q4k_row(
    row: &[u8],
    blocks: &mut [BlockQ4K],
    s: &mut Q4KScratch,
    st: &mut Q4KErrors<Cell>,
    at: usize,
    source: &Source<'_>,
) -> Result<()> {
    let cols = s.x.len();
    let (pairs, rest) = row.as_chunks::<2>();
    ensure!(rest.is_empty() && pairs.len() == cols, "matrix row {at}: {} bytes for {cols} values", row.len());

    // 1. exact source values, all finite
    for (x, &b) in s.x.iter_mut().zip(pairs) {
        *x = half::bf16::from_le_bytes(b).to_f32();
    }
    if let Some(col) = s.x.iter().position(|v| !v.is_finite()) {
        bail!(
            "non-finite weight {} at {}; this row has {} values",
            s.x[col],
            source.at(at, col),
            nonfinite_counts(s.x.iter().copied())
        );
    }
    st.rows += 1;
    st.values += cols as u64;

    // 2. source groups of 32 (f32 `==`: ±0 are equal)
    for (g, ((xs, constant), zero)) in
        s.x.chunks_exact(Q4K_GROUP).zip(s.group_constant.iter_mut()).zip(s.group_zero.iter_mut()).enumerate()
    {
        let first = xs.first().copied();
        *constant = xs.iter().all(|&v| Some(v) == first);
        *zero = *constant && first == Some(0.0);
        st.groups += 1;
        if *zero {
            st.zero_groups += 1;
        } else if *constant {
            bail!(
                "constant non-zero group of {Q4K_GROUP} values ({}) at {}: the pinned Candle Q4_K quantizer \
                 reconstructs it as zeros; use quantization q8_0 or none for this model",
                first.unwrap_or(f32::NAN),
                source.at(at, g * Q4K_GROUP)
            );
        }
    }

    // 3. quantize and reconstruct; Candle checks these lengths only in debug builds
    ensure!(
        blocks.len().checked_mul(Q4K_BLOCK) == Some(s.x.len()),
        "matrix row {at}: {} values for {} blocks",
        s.x.len(),
        blocks.len()
    );
    BlockQ4K::from_float(&s.x, blocks);
    ensure!(
        blocks.len().checked_mul(Q4K_BLOCK) == Some(s.y.len()),
        "matrix row {at}: {} outputs for {} blocks",
        s.y.len(),
        blocks.len()
    );
    BlockQ4K::to_float(blocks, &mut s.y);

    // 4. every to_float value and its f16 reconstruction finite, before any statistic (and before the upload)
    let mut first_bad = None;
    for (col, y) in s.y.iter_mut().enumerate() {
        let t = *y;
        let r = half::f16::from_f32(t).to_f32();
        if first_bad.is_none() && !(t.is_finite() && r.is_finite()) {
            first_bad = Some((col, t, r));
        }
        *y = r;
    }
    if let Some((col, t, r)) = first_bad {
        // the counts of both stages in this row: to_float again into a new buffer (error path only; the length is
        // the one checked above)
        let mut t_row = vec![0f32; s.x.len()];
        BlockQ4K::to_float(blocks, &mut t_row);
        bail!(
            "non-finite Q4_K reconstruction at {} (to_float {t}, f16 {r}); this row: to_float {} values, f16 \
             reconstruction {} values",
            source.at(at, col),
            nonfinite_counts(t_row.iter().copied()),
            nonfinite_counts(s.y.iter().copied())
        );
    }

    // 5. f64 sums in column order (the W4 P0 expressions), then the reconstruction groups
    for (&r, &x) in s.y.iter().zip(&s.x) {
        let w = f64::from(x);
        let e = f64::from(r) - w;
        st.sum_e2 += e * e;
        st.sum_w2 += w * w;
        st.max_abs_e = st.max_abs_e.max(e.abs());
    }
    for (g, ((ys, &constant), &zero)) in
        s.y.chunks_exact(Q4K_GROUP).zip(&s.group_constant).zip(&s.group_zero).enumerate()
    {
        let first = ys.first().copied();
        if !constant && ys.iter().all(|&v| Some(v) == first) {
            st.flattened_groups.hit(Cell { row: at, col: g * Q4K_GROUP });
        }
        if zero && ys.iter().any(|&v| v != 0.0) {
            st.zero_groups_nonzero_reconstruction.hit(Cell { row: at, col: g * Q4K_GROUP });
        }
    }
    Ok(())
}

/// The memory invariants of Candle's Q4_K dequantization kernel, checked on an uploaded tensor before any launch: the
/// kernel has no length argument and no bounds check (one thread block writes 256 values per super-block, offsets in
/// `int`), so the tensor must be a matrix with positive dimensions, K a multiple of 256, at most `i32::MAX` values and
/// exactly `blocks` Q4_K blocks of 144 bytes in its storage.
#[cfg(feature = "cuda")]
pub(crate) fn check_q4k_resident(q: &QTensor, blocks: u64) -> Result<()> {
    ensure!(q.dtype() == GgmlDType::Q4K, "uploaded blocks of type {:?}, expected Q4K", q.dtype());
    let (rows, cols) = q.shape().dims2()?;
    ensure!(
        rows > 0 && cols > 0 && cols.is_multiple_of(Q4K_BLOCK),
        "q4_k tensor [{rows}, {cols}]: dimensions must be positive and K a multiple of {Q4K_BLOCK}"
    );
    let values = rows.checked_mul(cols).context("element count overflows")?;
    ensure!(
        values <= usize::try_from(i32::MAX).context("i32::MAX as usize")?,
        "q4_k tensor [{rows}, {cols}]: {values} values exceed i32::MAX"
    );
    ensure!(
        u64::try_from(values / Q4K_BLOCK)? == blocks,
        "q4_k tensor [{rows}, {cols}]: {} blocks of the shape, {blocks} uploaded",
        values / Q4K_BLOCK
    );
    let expected = blocks.checked_mul(Q4K_BLOCK_BYTES as u64).context("block bytes overflow")?;
    ensure!(
        u64::try_from(q.storage_size_in_bytes())? == expected,
        "{} bytes of q4_k blocks on the device, expected {expected}",
        q.storage_size_in_bytes()
    );
    Ok(())
}

/// Warm-up control of one q4_k matrix, at load: dequantize the resident blocks onto their own device with Candle's
/// kernel (which also loads the `QUANTIZED` module and its q4_K function now, not at the first request), copy the f16
/// result to the host and compare every value's f16 bit pattern, signed zeros included, with
/// `f16::from_f32(BlockQ4K::to_float)` of the host blocks that were uploaded. A difference is a load error (no
/// fallback). One representative matrix per load, not a check of the other matrices. Returns the values compared.
#[cfg(feature = "cuda")]
pub(crate) fn check_q4k_warmup(q: &QTensor, blocks: &[BlockQ4K], source: &Source<'_>) -> Result<u64> {
    let (rows, cols) = q.shape().dims2()?;
    ensure!(rows > 0 && cols > 0 && cols.is_multiple_of(Q4K_BLOCK), "warm-up matrix [{rows}, {cols}]");
    let values = rows.checked_mul(cols).context("element count overflows")?;
    ensure!(
        blocks.len().checked_mul(Q4K_BLOCK) == Some(values),
        "warm-up: {} host blocks for [{rows}, {cols}]",
        blocks.len()
    );
    let gpu = q.dequantize_f16(&q.device())?.flatten_all()?.to_vec1::<half::f16>()?;
    ensure!(gpu.len() == values, "warm-up: {} dequantized values of {values}", gpu.len());
    let per_row = cols / Q4K_BLOCK;
    let mut host = vec![0f32; cols];
    let mut mismatches = 0u64;
    let mut first = None;
    for (r, (bs, gs)) in blocks.chunks_exact(per_row).zip(gpu.chunks_exact(cols)).enumerate() {
        // to_float checks lengths only in debug builds
        ensure!(bs.len().checked_mul(Q4K_BLOCK) == Some(host.len()), "warm-up row {r}: {} blocks", bs.len());
        BlockQ4K::to_float(bs, &mut host);
        for (c, (&h, &g)) in host.iter().zip(gs).enumerate() {
            let h = half::f16::from_f32(h);
            if h.to_bits() != g.to_bits() {
                mismatches += 1;
                first.get_or_insert((r, c, h, g));
            }
        }
    }
    if let Some((r, c, h, g)) = first {
        bail!(
            "q4_k warm-up: the GPU dequantization differs from the host reconstruction in {mismatches} of {values} \
             values; first at {}: GPU f16 {g} (bits {:#06x}), host {h} (bits {:#06x})",
            source.at(r, c),
            g.to_bits(),
            h.to_bits()
        );
    }
    Ok(values as u64)
}
