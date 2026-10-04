//! `x[M, K] @ w[N, K]^T` with the MLX steel GEMM kernels shipped by candle-metal-kernels, dispatched here so the tile
//! configuration and the threadgroup swizzle can be chosen. candle always dispatches with `swizzle_log = 0`, so the
//! row tiles (M) that share a weight tile run far apart and, at the small M of one question, every row of tiles
//! streams the whole weight matrix from memory again. Swizzling groups `2^s` row tiles of the same column tile.

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ConstantValues, Value};
use candle_metal_kernels::source::Source;
use objc2_metal::MTLSize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    pub swizzle_log: i32,
}

/// Tile configurations instantiated in candle's mlx_gemm.metal.
pub const TILES: [(usize, usize, usize, usize, usize); 5] =
    [(32, 32, 16, 2, 2), (64, 64, 16, 2, 2), (64, 64, 16, 1, 2), (64, 32, 32, 2, 2), (32, 64, 16, 1, 2)];

impl GemmConfig {
    pub const fn new(tile: (usize, usize, usize, usize, usize), swizzle_log: i32) -> Self {
        let (bm, bn, bk, wm, wn) = tile;
        Self { bm, bn, bk, wm, wn, swizzle_log }
    }
}

#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

struct GemmNt(GemmConfig);

impl CustomOp2 for GemmNt {
    fn name(&self) -> &'static str {
        "gemm-nt"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("gemm_nt is Metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, xl: &Layout, w: &MetalStorage, wl: &Layout) -> Result<(MetalStorage, Shape)> {
        let c = self.0;
        let (m, k) = xl.shape().dims2()?;
        let (n, k2) = wl.shape().dims2()?;
        if k != k2 || x.dtype() != w.dtype() {
            candle_core::bail!("gemm_nt: shapes {:?} x {:?}^T or dtypes differ", xl.dims(), wl.dims());
        }
        let (Some((xo, _)), Some((wo, _))) = (xl.contiguous_offsets(), wl.contiguous_offsets()) else {
            candle_core::bail!("gemm_nt: inputs must be contiguous");
        };
        let ty = match x.dtype() {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            d => candle_core::bail!("gemm_nt: dtype {d:?}"),
        };
        let dev = x.device();
        let out = dev.new_buffer_builder().with_size_for(m * n, x.dtype()).with_label("gemm_nt").build()?;
        let name = format!("gemm_nt_{ty}_{ty}_{}_{}_{}_{}_{}", c.bm, c.bn, c.bk, c.wm, c.wn);
        let constants = ConstantValues::new(vec![
            (10, Value::Bool(false)),
            (100, Value::Bool(false)),
            (110, Value::Bool(false)),
            (200, Value::Bool(m % c.bm == 0)),
            (201, Value::Bool(n % c.bn == 0)),
            (202, Value::Bool(k % c.bk == 0)),
            (300, Value::Bool(false)),
        ]);
        let pipeline = dev
            .kernels()
            .load_pipeline_with_constants(dev.device(), Source::Gemm, name, Some(constants))
            .map_err(candle_core::Error::wrap)?;
        let sw = 1usize << c.swizzle_log;
        let tn = n.div_ceil(c.bn) * sw;
        let tm = m.div_ceil(c.bm).div_ceil(sw);
        let params = GemmParams {
            m: m as i32,
            n: n as i32,
            k: k as i32,
            lda: k as i32,
            ldb: k as i32,
            ldd: n as i32,
            tiles_n: n.div_ceil(c.bn) as i32,
            tiles_m: m.div_ceil(c.bm) as i32,
            batch_stride_a: (m * k) as isize,
            batch_stride_b: (n * k) as isize,
            batch_stride_d: (m * n) as isize,
            swizzle_log: c.swizzle_log,
            gemm_k_iterations_aligned: (k / c.bk) as i32,
            batch_ndim: 1,
        };
        let batch_strides = [params.batch_stride_a, params.batch_stride_b];
        let guard = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        let sz = x.dtype().size_in_bytes();
        enc.set_input_buffer(0, Some(x.buffer()), xo * sz);
        enc.set_input_buffer(1, Some(w.buffer()), wo * sz);
        enc.set_output_buffer(3, Some(&out), 0);
        enc.set_bytes(4, &params);
        enc.set_bytes(6, &1i32);
        enc.set_bytes_directly(7, std::mem::size_of_val(&batch_strides), batch_strides.as_ptr().cast());
        enc.dispatch_thread_groups(
            MTLSize { width: tn, height: tm, depth: 1 },
            MTLSize { width: 32, height: c.wn, depth: c.wm },
        );
        Ok((MetalStorage::new(out, dev.clone(), m * n, x.dtype()), Shape::from((m, n))))
    }
}

/// `x @ w^T` for contiguous `x [M, K]`, `w [N, K]`.
pub fn gemm_nt(x: &Tensor, w: &Tensor, cfg: GemmConfig) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &GemmNt(cfg))
}
