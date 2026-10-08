//! `x @ w^T` (f16 / bf16, f32 accumulation) through cuBLASLt with a measured algorithm choice (CUDA only).
//!
//! cuBLAS (candle's matmul) picks one kernel per shape by heuristic; for the small-M shapes of a decision forward
//! (M = a few hundred tokens, weights 2048 x 2560 ... 22016) that choice is not the fastest. The algorithm of a shape
//! class (N, K, dtype, M rounded up by `m_class`) comes from, in order: a table produced offline by an exhaustive
//! search of cuBLASLt configurations (`search`: algorithm, tile, stages, split-K with f32 reduction, CTA swizzle;
//! `basal gemm-search`), or a timing of up to `CANDIDATES` heuristic algorithms on the first call of the class. A
//! later M of the class reuses the algorithm after `cublasLtMatmulAlgoCheck` accepts it (otherwise the heuristic's first
//! choice is used). Every configuration accumulates in f32 (split-K partial sums are reduced in f32, never in the
//! output type); results may differ from cuBLAS in the last bits (summation order).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::Instant;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::cublaslt::sys;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle_core::cuda_backend::{CudaStorageSlice as Cs, WrapErr};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

const CANDIDATES: usize = 16;

/// A configuration and its measured time (ms).
type Timed = (f64, sys::cublasLtMatmulAlgo_t);
const WORKSPACE: usize = 32 << 20;

/// Shape class of M: up to 512 rounded up to a multiple of 16, above that to a multiple of 128.
pub fn m_class(m: usize) -> usize {
    if m <= 512 {
        m.div_ceil(16) * 16
    } else {
        m.div_ceil(128) * 128
    }
}

/// A tuned class: its algorithm (opaque cuBLASLt bytes, valid for this GPU and cuBLASLt version) and timings.
#[derive(Clone, Debug)]
pub struct Tuned {
    pub m_class: usize,
    pub n: usize,
    pub k: usize,
    pub dtype: DType,
    pub algo: [u64; 8],
    pub ms: f64,
    pub heuristic_ms: f64,
    pub tried: usize,
}

pub fn version() -> usize {
    // SAFETY: plain query.
    unsafe { sys::cublasLtGetVersion() }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    n: usize,
    k: usize,
    m_class: usize,
    dt: DType,
}

/// Descriptors of one exact (M, N, K, dtype); created once, never destroyed (one per distinct shape).
#[derive(Clone, Copy)]
struct Plan {
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
}

/// Algorithms of a weight shape by M class (ascending), all with bitwise identical results.
type ClassAlgos = Vec<(usize, sys::cublasLtMatmulAlgo_t)>;

pub struct Lt {
    handle: sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    plans: Mutex<HashMap<(usize, usize, usize, DType), Plan>>,
    algos: Mutex<HashMap<Key, sys::cublasLtMatmulAlgo_t>>,
    /// Classes tuned in this process (searched or timed heuristics).
    pub tuned: Mutex<Vec<Tuned>>,
    /// Classes taken from a table.
    pub from_table: Mutex<usize>,
    /// Batch-invariant mode: one algorithm per (N, K, dtype) for every M, without split-K, so that a row's result
    /// does not depend on the other rows of the call. A shape listed here never uses another algorithm.
    invariant: Mutex<HashMap<(usize, usize, DType), ClassAlgos>>,
}

// SAFETY: the handle and descriptors are only used under the plan/algo locks on candle's single stream.
unsafe impl Send for Lt {}
// SAFETY: shared access uses the same plan/algo locks and single candle stream as owned access.
unsafe impl Sync for Lt {}

fn st(s: sys::cublasStatus_t, what: &str) -> Result<()> {
    if s == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        candle_core::bail!("cuBLASLt {what}: {s:?}")
    }
}

fn data_type(dt: DType) -> Result<sys::cudaDataType> {
    match dt {
        DType::F16 => Ok(sys::cudaDataType::CUDA_R_16F),
        DType::BF16 => Ok(sys::cudaDataType::CUDA_R_16BF),
        d => candle_core::bail!("cuBLASLt matmul: unsupported dtype {d:?}"),
    }
}

impl Lt {
    pub fn new(dev: &candle_core::CudaDevice) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        // SAFETY: plain handle creation.
        st(unsafe { sys::cublasLtCreate(&mut handle) }, "create")?;
        // SAFETY: scratch memory for cuBLASLt, never read by us.
        let workspace = unsafe { dev.alloc::<u8>(WORKSPACE)? };
        Ok(Self {
            handle,
            workspace,
            plans: Mutex::new(HashMap::new()),
            algos: Mutex::new(HashMap::new()),
            tuned: Mutex::new(Vec::new()),
            from_table: Mutex::new(0),
            invariant: Mutex::new(HashMap::new()),
        })
    }

    /// Batch-invariant table: for each (N, K, dtype) the algorithms of its M classes, all of them giving bitwise the
    /// same results (one algorithm for every class in tables written before the grouping).
    pub fn load_invariant(&self, entries: &[Tuned]) -> Result<()> {
        let mut inv = self.invariant.lock().unwrap();
        for e in entries {
            let list = inv.entry((e.n, e.k, e.dtype)).or_default();
            list.push((e.m_class, sys::cublasLtMatmulAlgo_t { data: e.algo }));
            list.sort_by_key(|(c, _)| *c);
            list.dedup_by_key(|(c, _)| *c);
        }
        *self.from_table.lock().unwrap() += entries.len();
        Ok(())
    }

    /// A table entry (batch-invariant or per class) exists for the weight shape (N, K) in `dt`.
    pub fn covers(&self, n: usize, k: usize, dt: DType) -> bool {
        self.invariant.lock().unwrap().contains_key(&(n, k, dt))
            || self.algos.lock().unwrap().keys().any(|key| key.n == n && key.k == k && key.dt == dt)
    }

    pub fn is_invariant(&self) -> bool {
        !self.invariant.lock().unwrap().is_empty()
    }

    fn plan_cached(&self, m: usize, n: usize, k: usize, dt: DType) -> Result<Plan> {
        let mut plans = self.plans.lock().unwrap();
        Ok(*match plans.entry((m, n, k, dt)) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(self.plan(m, n, k, dt)?),
        })
    }

    /// Mean time of `reps` runs of `a`, reading the weight copies in turn starting at copy `start`.
    ///
    /// # Safety
    /// As [`Lt::run`], with `w` holding `copies` weights of `w_bytes`.
    #[expect(clippy::too_many_arguments, reason = "Timing requires the plan, device buffers, weight copies and stream")]
    unsafe fn time_algo(
        &self,
        p: &Plan,
        a: &sys::cublasLtMatmulAlgo_t,
        w: u64,
        copies: usize,
        w_bytes: u64,
        x: u64,
        y: u64,
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
        reps: usize,
    ) -> Result<f64> {
        let wi = |i: usize| w + (i % copies.max(1)) as u64 * w_bytes;
        // SAFETY: the caller guarantees the plan's buffers; wi selects a complete in-bounds weight copy.
        unsafe { self.run(p, a, wi(reps + 1), x, y, stream)? };
        stream.synchronize().w()?;
        let t = Instant::now();
        for i in 0..reps {
            // SAFETY: the same live buffers and plan are reused, selecting a valid weight copy.
            unsafe { self.run(p, a, wi(i), x, y, stream)? };
        }
        stream.synchronize().w()?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
    }

    /// Batch-invariant choice for one weight shape: candidates without split-K from searches at a few M, each timed
    /// at every class in `classes`; the chosen one has the smallest mean slowdown against the best candidate of each
    /// class. `x` holds at least `max(classes)` rows. Returns (chosen algorithm, per-class (ms, best ms), candidates).
    ///
    /// # Safety
    /// As [`Lt::run`] for each class: x and y hold at least max(classes) rows, and w holds copies full weights.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "The search receives GPU buffers and returns an opaque algorithm plus timings for each M class"
    )]
    unsafe fn invariant_search(
        &self,
        n: usize,
        k: usize,
        dt: DType,
        w: u64,
        copies: usize,
        w_bytes: u64,
        x: u64,
        y: u64,
        classes: &[usize],
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(Vec<(sys::cublasLtMatmulAlgo_t, f64, f64)>, usize)> {
        let mut cands: Vec<sys::cublasLtMatmulAlgo_t> = Vec::new();
        for m in [64, 128, 192, 256, 384, 512, 1024, 2048, 4096, 8192, 16384] {
            if !classes.contains(&m) {
                continue;
            }
            let p = self.plan_cached(m, n, k, dt)?;
            // SAFETY: m is in classes, so the caller's max-class buffers cover this plan and every weight copy.
            let (top, _, _) = unsafe { self.search(&p, dt, w, copies, w_bytes, x, y, stream, false)? };
            for (_, a) in top {
                if !cands.iter().any(|c| c.data == a.data) {
                    cands.push(a);
                }
            }
        }
        let mut times = vec![vec![f64::INFINITY; classes.len()]; cands.len()];
        for (ci, &m) in classes.iter().enumerate() {
            let p = self.plan_cached(m, n, k, dt)?;
            for (ai, a) in cands.iter().enumerate() {
                if self.check(&p, a) {
                    // SAFETY: the algorithm passed the plan check; buffers cover every class and weight copy.
                    times[ai][ci] = unsafe { self.time_algo(&p, a, w, copies, w_bytes, x, y, stream, 10)? };
                }
            }
        }
        let best: Vec<f64> =
            (0..classes.len()).map(|ci| times.iter().map(|t| t[ci]).fold(f64::INFINITY, f64::min)).collect();
        // groups of candidates with bitwise identical outputs; the group with the smallest geometric-mean slowdown
        // of its fastest member per class (every class weighs the same) gives one algorithm per class
        // SAFETY: the caller guarantees live buffers large enough for every class on this stream.
        let group = unsafe { self.equivalence_groups(n, k, dt, w, x, y, classes, &cands, stream)? };
        let geo = |f: &dyn Fn(usize) -> f64| {
            ((0..classes.len()).map(|ci| (f(ci) / best[ci]).ln()).sum::<f64>() / classes.len() as f64).exp()
        };
        let mut roots: Vec<usize> = group.clone();
        roots.sort();
        roots.dedup();
        let fastest = |r: usize, ci: usize| {
            (0..cands.len()).filter(|&ai| group[ai] == r).min_by(|&a, &b| times[a][ci].total_cmp(&times[b][ci]))
        };
        let scored: Vec<(usize, f64)> =
            roots.iter().map(|&r| (r, geo(&|ci| fastest(r, ci).map_or(f64::INFINITY, |ai| times[ai][ci])))).collect();
        if std::env::var("BASAL_GEMM_EQUIV").is_ok() {
            let single = (0..cands.len()).map(|ai| geo(&|ci| times[ai][ci])).fold(f64::INFINITY, f64::min);
            eprintln!(
                "equivalence n={n} k={k}: {} candidates, {} groups; best single algorithm {single:.3}",
                cands.len(),
                roots.len()
            );
            for &(r, sc) in &scored {
                eprintln!("  group of {:2}: best member per class {sc:.3}", group.iter().filter(|&&g| g == r).count());
            }
        }
        let (root, _) =
            scored.iter().copied().filter(|(_, sc)| sc.is_finite()).min_by(|a, b| a.1.total_cmp(&b.1)).ok_or_else(
                || candle_core::Error::Msg(format!("no batch-invariant group valid for every M (n={n} k={k})")),
            )?;
        let per_class = (0..classes.len())
            .map(|ci| {
                let ai = fastest(root, ci).unwrap();
                (cands[ai], times[ai][ci], best[ci])
            })
            .collect();
        Ok((per_class, cands.len()))
    }

    /// Groups of candidates whose outputs are bitwise identical on the same operands at every class (where both are
    /// valid): the group id of each candidate. Candidates without split-K or sliced K reduce over K in the same
    /// order and fall into one group.
    ///
    /// # Safety
    /// As [`Lt::run`] for every class: x and y must hold at least max(classes) rows on stream.
    #[expect(clippy::too_many_arguments, reason = "Equivalence compares GPU buffers for each shape and algorithm")]
    unsafe fn equivalence_groups(
        &self,
        n: usize,
        k: usize,
        dt: DType,
        w: u64,
        x: u64,
        y: u64,
        classes: &[usize],
        cands: &[sys::cublasLtMatmulAlgo_t],
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<Vec<usize>> {
        use candle_core::cuda_backend::cudarc::driver::sys as drv;
        let mut sig: Vec<Vec<Option<u64>>> = vec![Vec::new(); cands.len()];
        for &m in classes {
            let p = self.plan_cached(m, n, k, dt)?;
            let bytes = m * n * dt.size_in_bytes();
            let mut host = vec![0u8; bytes];
            for (ai, a) in cands.iter().enumerate() {
                if !self.check(&p, a) {
                    sig[ai].push(None);
                    continue;
                }
                // SAFETY: this candidate passed the plan check and the caller's buffers cover the class.
                unsafe { self.run(&p, a, w, x, y, stream)? };
                stream.synchronize().w()?;
                // SAFETY: y holds m*n elements, host has exactly bytes capacity, and the stream is synchronized.
                let r = unsafe { drv::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut c_void, y, bytes) };
                if r != drv::CUresult::CUDA_SUCCESS {
                    candle_core::bail!("cuBLASLt equivalence: copy failed {r:?}");
                }
                let mut h: u64 = 0xcbf29ce484222325;
                for c in host.chunks(8) {
                    let mut v = [0u8; 8];
                    v[..c.len()].copy_from_slice(c);
                    h = (h ^ u64::from_le_bytes(v)).wrapping_mul(0x100000001b3);
                }
                sig[ai].push(Some(h));
            }
        }
        let mut group: Vec<usize> = (0..cands.len()).collect();
        for i in 0..cands.len() {
            for j in 0..i {
                if group[j] == j
                    && sig[i].iter().zip(&sig[j]).all(|(a, b)| a.is_none() || b.is_none() || a == b)
                    && sig[i].iter().zip(&sig[j]).any(|(a, b)| a.is_some() && b.is_some())
                {
                    group[i] = j;
                    break;
                }
            }
        }
        Ok(group)
    }

    /// Use the algorithms of a search table for their classes.
    pub fn load(&self, entries: &[Tuned]) {
        let mut algos = self.algos.lock().unwrap();
        for e in entries {
            let key = Key { n: e.n, k: e.k, m_class: e.m_class, dt: e.dtype };
            algos.insert(key, sys::cublasLtMatmulAlgo_t { data: e.algo });
        }
        *self.from_table.lock().unwrap() += entries.len();
    }

    /// Exhaustive search of cuBLASLt configurations for one shape on the given operands (f32 accumulation; split-K
    /// only with f32 reduction). Returns the fastest valid algorithm, its time, the heuristic's first choice time and
    /// the number of valid configurations timed.
    ///
    /// # Safety
    /// As [`Lt::run`].
    #[expect(clippy::too_many_arguments, reason = "Algorithm search needs a plan, dtype, live GPU buffers and stream")]
    unsafe fn search(
        &self,
        p: &Plan,
        dt: DType,
        w: u64,
        w_copies: usize,
        w_bytes: u64,
        x: u64,
        y: u64,
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
        allow_splitk: bool,
    ) -> Result<(Vec<Timed>, f64, usize)> {
        use sys::cublasLtMatmulAlgoCapAttributes_t as Cap;
        use sys::cublasLtMatmulAlgoConfigAttributes_t as Cfg;
        let t = data_type(dt)?;
        let ct = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
        let st32 = sys::cudaDataType::CUDA_R_32F;
        // Successive runs read successive copies of the weight (together larger than L2), as the layers of a
        // forward do; timing one resident copy would favour configurations that re-read the weight.
        let next = std::cell::Cell::new(0usize);
        let wi = || {
            let i = next.get();
            next.set(i + 1);
            w + (i % w_copies.max(1)) as u64 * w_bytes
        };
        let time = |a: &sys::cublasLtMatmulAlgo_t, reps: usize| -> Result<f64> {
            // SAFETY: the caller guarantees all plan buffers; wi selects one of the complete weight copies.
            unsafe { self.run(p, a, wi(), x, y, stream)? };
            stream.synchronize().w()?;
            let t = Instant::now();
            for _ in 0..reps {
                // SAFETY: the same live buffers and checked plan are reused with an in-bounds weight copy.
                unsafe { self.run(p, a, wi(), x, y, stream)? };
            }
            stream.synchronize().w()?;
            Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
        };
        let heur = self.heuristics(p)?;
        let heuristic_ms = match heur.first() {
            Some(r) => time(&r.algo, 20)?,
            None => f64::NAN,
        };
        let mut ids = vec![0i32; 128];
        let mut nids = 0;
        // SAFETY: the handle is valid and ids/nids provide initialized output storage for the requested count.
        st(
            unsafe {
                sys::cublasLtMatmulAlgoGetIds(self.handle, ct, st32, t, t, t, t, 128, ids.as_mut_ptr(), &mut nids)
            },
            "ids",
        )?;
        ids.truncate(nids as usize);
        let cap_u32s = |algo: &sys::cublasLtMatmulAlgo_t, attr: Cap| -> Vec<u32> {
            let mut size = 0usize;
            // SAFETY: with a null buffer and zero size, cuBLASLt only writes the required size.
            unsafe { sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, std::ptr::null_mut(), 0, &mut size) };
            let mut v = vec![0u32; size / 4];
            if !v.is_empty() {
                // SAFETY: queried tile/stage ID arrays are u32 values and v provides the reported byte size.
                unsafe {
                    sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, v.as_mut_ptr() as *mut c_void, size, &mut size)
                };
            }
            v
        };
        let cap_u32 = |algo: &sys::cublasLtMatmulAlgo_t, attr: Cap| -> u32 {
            let (mut v, mut size) = (0u32, 0usize);
            // SAFETY: the queried scalar capabilities are u32 values; v and size are valid output pointers.
            unsafe {
                sys::cublasLtMatmulAlgoCapGetAttribute(algo, attr, &mut v as *mut u32 as *mut c_void, 4, &mut size)
            };
            v
        };
        let set = |algo: &mut sys::cublasLtMatmulAlgo_t, attr: Cfg, v: u32| {
            // SAFETY: these configuration attributes take u32 values and v outlives the synchronous call.
            unsafe { sys::cublasLtMatmulAlgoConfigSetAttribute(algo, attr, &v as *const u32 as *const c_void, 4) };
        };
        let mut cands: Vec<(f64, sys::cublasLtMatmulAlgo_t)> = Vec::new();
        for &id in &ids {
            // SAFETY: the opaque algorithm struct contains a fixed-size integer array, valid when zeroed.
            let mut base: sys::cublasLtMatmulAlgo_t = unsafe { std::mem::zeroed() };
            // SAFETY: the handle and data types are valid; base is writable storage for the initialized algorithm.
            if unsafe { sys::cublasLtMatmulAlgoInit(self.handle, ct, st32, t, t, t, t, id, &mut base) }
                != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
            {
                continue;
            }
            let mut tiles = cap_u32s(&base, Cap::CUBLASLT_ALGO_CAP_TILE_IDS);
            if tiles.is_empty() {
                tiles.push(0);
            }
            let mut stages = cap_u32s(&base, Cap::CUBLASLT_ALGO_CAP_STAGES_IDS);
            if stages.is_empty() {
                stages.push(0);
            }
            let splitk = cap_u32(&base, Cap::CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) != 0;
            let red_f32 = cap_u32(&base, Cap::CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & 2 != 0; // COMPUTE_TYPE
            let swz = cap_u32(&base, Cap::CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
            let splits: &[u32] = if allow_splitk && splitk && red_f32 { &[1, 2, 3, 4, 6, 8] } else { &[1] };
            for &tile in &tiles {
                for &stage in &stages {
                    for &sk in splits {
                        for sw in 0..=swz.min(1) {
                            let mut a = base;
                            set(&mut a, Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID, tile);
                            set(&mut a, Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID, stage);
                            set(&mut a, Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, sk);
                            set(&mut a, Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, if sk > 1 { 2 } else { 0 });
                            set(&mut a, Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, sw);
                            if !self.check(p, &a) {
                                continue;
                            }
                            if let Ok(ms) = time(&a, 5) {
                                cands.push((ms, a));
                            }
                        }
                    }
                }
            }
        }
        let tried = cands.len();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.truncate(8);
        if allow_splitk {
            for c in heur.iter().take(4) {
                cands.push((0.0, c.algo));
            }
        }
        let mut timed = Vec::with_capacity(cands.len());
        for (_, a) in &cands {
            timed.push((time(a, 30)?, *a));
        }
        timed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if timed.is_empty() {
            candle_core::bail!("cuBLASLt search: no valid configuration");
        }
        Ok((timed, heuristic_ms, tried))
    }

    /// Row-major `y[m, n] = x[m, k] @ w[n, k]^T` is column-major `y^T[n, m] = op_T(w)[n, k] x^T[k, m]`.
    fn plan(&self, m: usize, n: usize, k: usize, dt: DType) -> Result<Plan> {
        let t = data_type(dt)?;
        let mut desc = std::ptr::null_mut();
        let (mut a, mut b, mut c) = (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: descriptor creation with valid out-pointers; attribute buffers outlive the calls.
        unsafe {
            st(
                sys::cublasLtMatmulDescCreate(
                    &mut desc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    sys::cudaDataType::CUDA_R_32F,
                ),
                "desc",
            )?;
            let tr: i32 = 1; // CUBLAS_OP_T
            st(
                sys::cublasLtMatmulDescSetAttribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                    &tr as *const _ as *const c_void,
                    std::mem::size_of_val(&tr),
                ),
                "transa",
            )?;
            st(sys::cublasLtMatrixLayoutCreate(&mut a, t, k as u64, n as u64, k as i64), "layout a")?;
            st(sys::cublasLtMatrixLayoutCreate(&mut b, t, k as u64, m as u64, k as i64), "layout b")?;
            st(sys::cublasLtMatrixLayoutCreate(&mut c, t, n as u64, m as u64, n as i64), "layout c")?;
        }
        Ok(Plan { desc, a, b, c })
    }

    fn heuristics(&self, p: &Plan) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>> {
        let mut pref = std::ptr::null_mut();
        let ws = WORKSPACE as u64;
        // SAFETY: preference descriptor with a valid out-pointer; results buffer sized to the request.
        unsafe {
            st(sys::cublasLtMatmulPreferenceCreate(&mut pref), "pref")?;
            st(
                sys::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &ws as *const _ as *const c_void,
                    std::mem::size_of_val(&ws),
                ),
                "pref workspace",
            )?;
            let mut res: Vec<sys::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); CANDIDATES];
            let mut found = 0;
            let s = sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                p.desc,
                p.a,
                p.b,
                p.c,
                p.c,
                pref,
                CANDIDATES as i32,
                res.as_mut_ptr(),
                &mut found,
            );
            sys::cublasLtMatmulPreferenceDestroy(pref);
            st(s, "heuristic")?;
            res.truncate(found as usize);
            Ok(res.into_iter().filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS).collect())
        }
    }

    fn check(&self, p: &Plan, algo: &sys::cublasLtMatmulAlgo_t) -> bool {
        // SAFETY: the check only reads the descriptors and the algorithm.
        unsafe {
            let mut r: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            sys::cublasLtMatmulAlgoCheck(self.handle, p.desc, p.a, p.b, p.c, p.c, algo, &mut r)
                == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && r.workspaceSize <= WORKSPACE
        }
    }

    /// # Safety
    /// `w`, `x`, `y` must be device pointers of `n*k`, `m*k`, `m*n` elements of the plan's dtype on `stream`.
    unsafe fn run(
        &self,
        p: &Plan,
        algo: &sys::cublasLtMatmulAlgo_t,
        w: u64,
        x: u64,
        y: u64,
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<()> {
        let (alpha, beta) = (1f32, 0f32);
        let (ws, _g) = self.workspace.device_ptr(stream);
        st(
            // SAFETY: the caller guarantees the plan's live device buffers on stream; scalar pointers and
            // the guarded workspace remain valid throughout this enqueue call.
            unsafe {
                sys::cublasLtMatmul(
                    self.handle,
                    p.desc,
                    &alpha as *const f32 as *const c_void,
                    w as *const c_void,
                    p.a,
                    x as *const c_void,
                    p.b,
                    &beta as *const f32 as *const c_void,
                    y as *mut c_void,
                    p.c,
                    y as *mut c_void,
                    p.c,
                    algo,
                    ws as *mut c_void,
                    WORKSPACE,
                    stream.cu_stream() as _,
                )
            },
            "matmul",
        )
    }

    fn matmul(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        mode: &Mode,
    ) -> Result<(CudaStorage, Shape)> {
        let (m, k) = xl.shape().dims2()?;
        // With a search, `w` holds stacked copies of the weight.
        let copies = match mode {
            Mode::Plain => 1,
            Mode::Search(c) | Mode::Invariant(c, _) => (*c).max(1),
        };
        let (n_all, k2) = wl.shape().dims2()?;
        let n = n_all / copies;
        if k != k2 || x.dtype() != w.dtype() {
            candle_core::bail!(
                "cuBLASLt matmul: x {:?} {:?}, w {:?} {:?}",
                xl.shape(),
                x.dtype(),
                wl.shape(),
                w.dtype()
            );
        }
        let (Some((xo, _)), Some((wo, _))) = (xl.contiguous_offsets(), wl.contiguous_offsets()) else {
            candle_core::bail!("cuBLASLt matmul: operands must be contiguous");
        };
        let dt = x.dtype();
        let dev = x.device.clone();
        let stream = dev.cuda_stream();
        macro_rules! ptrs {
            ($v:ident, $t:ty) => {{
                let (Cs::$v(xs), Cs::$v(ws)) = (&x.slice, &w.slice) else { unreachable!() };
                // SAFETY: every element of the output is written by the matmul.
                let mut out = unsafe { dev.alloc::<$t>(m * n)? };
                {
                    let (xv, wv) = (xs.slice(xo..), ws.slice(wo..));
                    let (xp, _gx) = xv.device_ptr(&stream);
                    let (wp, _gw) = wv.device_ptr(&stream);
                    let (yp, _gy) = out.device_ptr_mut(&stream);
                    let w_bytes = (n * k * dt.size_in_bytes()) as u64;
                    let search = match mode {
                        Mode::Plain => 0,
                        Mode::Search(c) => *c,
                        Mode::Invariant(c, classes) => {
                            // SAFETY: x holds m >= max(classes) rows, w holds c copies, y holds m*n elements.
                            let (per_class, ncand) =
                                unsafe { self.invariant_search(n, k, dt, wp, *c, w_bytes, xp, yp, classes, &stream)? };
                            let mut tuned = self.tuned.lock().unwrap();
                            let mut list = Vec::new();
                            for (&mc, (a, ms, best)) in classes.iter().zip(per_class) {
                                tuned.push(Tuned {
                                    m_class: mc,
                                    n,
                                    k,
                                    dtype: dt,
                                    algo: a.data,
                                    ms,
                                    heuristic_ms: best,
                                    tried: ncand,
                                });
                                list.push((mc, a));
                            }
                            list.sort_by_key(|(c, _)| *c);
                            self.invariant.lock().unwrap().insert((n, k, dt), list);
                            0
                        }
                    };
                    self.dispatch(m, n, k, dt, wp, xp, yp, &stream, search)?;
                }
                Cs::$v(out)
            }};
        }
        let slice = match dt {
            DType::F16 => ptrs!(F16, half::f16),
            DType::BF16 => ptrs!(BF16, half::bf16),
            d => candle_core::bail!("cuBLASLt matmul: unsupported dtype {d:?}"),
        };
        Ok((CudaStorage { slice, device: dev }, Shape::from((m, n))))
    }

    #[expect(clippy::too_many_arguments, reason = "Dispatch maps one validated matmul shape and its device pointers")]
    fn dispatch(
        &self,
        m: usize,
        n: usize,
        k: usize,
        dt: DType,
        w: u64,
        x: u64,
        y: u64,
        stream: &candle_core::cuda_backend::cudarc::driver::CudaStream,
        search: usize,
    ) -> Result<()> {
        let p = &self.plan_cached(m, n, k, dt)?;
        let inv = self.invariant.lock().unwrap().get(&(n, k, dt)).cloned();
        if let Some(list) = inv {
            // the algorithm of the smallest class >= M's class (else the largest), or any other valid one: all of
            // them give the same results
            let mc = m_class(m);
            let start = list.iter().position(|(c, _)| *c >= mc).unwrap_or(list.len() - 1);
            let order = (start..list.len()).chain((0..start).rev());
            let Some(a) = order.map(|i| list[i].1).find(|a| self.check(p, a)) else {
                candle_core::bail!("batch-invariant cuBLASLt algorithms rejected for M={m} (n={n} k={k})");
            };
            // SAFETY: device pointers of the operands and of a fresh output of m*n elements, all on `stream`.
            return unsafe { self.run(p, &a, w, x, y, stream) };
        }
        let key = Key { n, k, m_class: m_class(m), dt };
        if search > 0 {
            let w_bytes = (n * k * dt.size_in_bytes()) as u64;
            // SAFETY: operand pointers as for the final run below; `w` holds `search` copies of n*k elements.
            let (top, heuristic_ms, tried) = unsafe { self.search(p, dt, w, search, w_bytes, x, y, stream, true)? };
            let (ms, a) = top[0];
            self.tuned.lock().unwrap().push(Tuned {
                m_class: key.m_class,
                n,
                k,
                dtype: dt,
                algo: a.data,
                ms,
                heuristic_ms,
                tried,
            });
            self.algos.lock().unwrap().insert(key, a);
        }
        let known = self.algos.lock().unwrap().get(&key).copied();
        let algo = match known {
            Some(a) if self.check(p, &a) => a,
            Some(_) => match self.heuristics(p)?.first() {
                Some(r) => r.algo,
                None => candle_core::bail!("cuBLASLt: no algorithm for {m}x{n}x{k}"),
            },
            None => {
                let cands = self.heuristics(p)?;
                if cands.is_empty() {
                    candle_core::bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
                }
                // Time every candidate on the call's operands (the output is rewritten by the final run).
                let time = |a: &sys::cublasLtMatmulAlgo_t| -> Result<f64> {
                    // SAFETY: pointers as for the final run below.
                    unsafe { self.run(p, a, w, x, y, stream)? };
                    stream.synchronize().w()?;
                    let t = Instant::now();
                    for _ in 0..5 {
                        // SAFETY: as above.
                        unsafe { self.run(p, a, w, x, y, stream)? };
                    }
                    stream.synchronize().w()?;
                    Ok(t.elapsed().as_secs_f64() * 1e3 / 5.0)
                };
                let mut best = (f64::INFINITY, cands[0].algo);
                let mut first = f64::NAN;
                for (i, c) in cands.iter().enumerate() {
                    let ms = time(&c.algo)?;
                    if i == 0 {
                        first = ms;
                    }
                    if ms < best.0 {
                        best = (ms, c.algo);
                    }
                }
                self.tuned.lock().unwrap().push(Tuned {
                    m_class: key.m_class,
                    n,
                    k,
                    dtype: dt,
                    algo: best.1.data,
                    ms: best.0,
                    heuristic_ms: first,
                    tried: cands.len(),
                });
                self.algos.lock().unwrap().insert(key, best.1);
                best.1
            }
        };
        // SAFETY: device pointers of the operands and of a fresh output of m*n elements, all on `stream`.
        unsafe { self.run(p, &algo, w, x, y, stream) }
    }
}

enum Mode {
    Plain,
    /// exhaustive search for the class of this M first; `w` holds the given number of stacked weight copies
    Search(usize),
    /// batch-invariant choice over the M classes first (x has at least max(classes) rows)
    Invariant(usize, Vec<usize>),
}

struct LtMatmul<'a>(&'a Lt, Mode);

impl CustomOp2 for LtMatmul<'_> {
    fn name(&self) -> &'static str {
        "cublaslt-matmul-nt"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("cuBLASLt matmul runs on CUDA only")
    }
    fn cuda_fwd(&self, x: &CudaStorage, xl: &Layout, w: &CudaStorage, wl: &Layout) -> Result<(CudaStorage, Shape)> {
        self.0.matmul(x, xl, w, wl, &self.1)
    }
}

/// `x[m, k] @ w[n, k]^T` (f16 or bf16) with the measured cuBLASLt algorithm of its shape class.
pub fn matmul_nt(lt: &Lt, x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &LtMatmul(lt, Mode::Plain))
}

/// Exhaustive configuration search for the class of `x @ w0^T` (see [`Lt::tuned`]); `w = [copies * n, k]` holds
/// `copies` stacked weights that the timing runs read in turn. Returns `x @ w0^T`.
pub fn search_nt(lt: &Lt, x: &Tensor, w: &Tensor, copies: usize) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &LtMatmul(lt, Mode::Search(copies.max(1))))
}

/// Batch-invariant algorithm choice for the weight shape of `w` (see [`Lt::load_invariant`]); `x` has
/// `max(classes)` rows, `w = [copies * n, k]`. Afterwards every call of this shape uses the chosen algorithm.
pub fn invariant_nt(lt: &Lt, x: &Tensor, w: &Tensor, copies: usize, classes: &[usize]) -> Result<Tensor> {
    x.apply_op2_no_bwd(w, &LtMatmul(lt, Mode::Invariant(copies.max(1), classes.to_vec())))
}
