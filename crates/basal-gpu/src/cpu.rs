//! Host CPU execution of a dense BF16 Llama checkpoint (`backend: cpu`): one stage on `Device::Cpu`, loaded by the
//! shared Llama loader with BF16 resident weights widened to F32 per use, and computed by the unchanged stock-Candle
//! padded decoder (Candle kernels, F32 readout, no quantization, no prefix or state cache, text only).
//!
//! The host bounds below are source-derived arithmetic of the named tensor payloads of the loader and of one padded
//! forward, all in checked `u64` (an overflow counts as exceeding). They are not measured RSS, not allocator peaks and
//! no protection against process OOM: the allocations themselves stay infallible. Excluded and never claimed bounded:
//! gemm packing buffers and its per-thread slabs, Rayon thread stacks, allocator metadata, rounding and
//! fragmentation, Tensor objects and other metadata, the read-only checkpoint mapping and page cache, the JSON header
//! value, the ZIP/pickle metadata, decompressor buffers and skipped offsets of the evidence head reader (and its
//! behavior on a malformed file), tokenizer, request planning, engine plans, HTTP/Tokio and tracing.
//!
//! Every call is preflighted before the first host input of a forward exists: rows are grouped greedily in order
//! while the bound of the group fits the frozen ceiling, a single row (or an evidence forward) above it is refused
//! with [`CpuWorkExceeded`], and only then are the groups run once each. The output vectors already accumulated for
//! a call and the planner's control data are request state outside the per-forward ceiling.

use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use basal_core::manifest::LlamaConfig;
use basal_core::pack::Packed;
use candle_core::DType;
use serde_json::{json, Value};

use crate::{GpuBackend, Kernels, Precision, Quantization, Readout, StageDevice, StageGates, MASKED};

/// Optional evidence head of a model directory (`EvidenceHead::load`).
const EVIDENCE_FILE: &str = "evidence_head.pt";
/// The six tensors `EvidenceHead::load` accepts: four `[k, H]` projections and two `[1, H]` biases.
const EVIDENCE_PROJECTIONS: [&str; 4] = ["ws.weight", "us.weight", "we.weight", "ue.weight"];
const EVIDENCE_BIASES: [&str; 2] = ["bs.weight", "be.weight"];

/// Frozen CPU execution limits of a loaded model, copied to its forks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CpuWork {
    /// Host working-set ceiling of one CPU forward (bytes), at most `usize::MAX`.
    pub(crate) ceiling: NonZeroU64,
    /// Candle's host GEMM thread count captured by the plan (reported, never set).
    pub(crate) gemm_threads: usize,
}

/// Source arithmetic of one CPU model before it is loaded: named tensor payloads, not allocator peaks, measured RSS
/// or a guarantee against process OOM.
#[derive(Clone, Debug, serde::Serialize)]
pub struct CpuHostMemory {
    /// Bytes of the checkpoint files (for the report; the data is read, not counted as resident).
    pub checkpoint_bytes: u64,
    /// BF16 weights and zero biases, the RoPE table and inverse frequencies, and the F32 evidence head.
    pub resident_bytes: u64,
    /// Transient load bytes: the largest concatenated projection read twice, the safetensors header, the RoPE copy
    /// into shared storage and the evidence head's source tensors.
    pub load_scratch_bytes: u64,
    /// Logical evidence-head payload in its stored dtypes (0 without `evidence_head.pt`).
    pub evidence_source_bytes: u64,
    /// The same tensors as F32 (0 without `evidence_head.pt`).
    pub evidence_f32_bytes: u64,
}

impl CpuHostMemory {
    /// Resident + load scratch + `workers` independent forward ceilings of `work_bytes` (workers are counted even when
    /// a gate serializes them).
    pub fn startup_bytes(&self, workers: u64, work_bytes: u64) -> Result<u64> {
        workers
            .checked_mul(work_bytes)
            .and_then(|n| n.checked_add(self.resident_bytes))
            .and_then(|n| n.checked_add(self.load_scratch_bytes))
            .context("CPU startup estimate overflow")
    }
}

/// A CPU forward whose source bound exceeds the frozen ceiling; returned before any host input of it is allocated.
#[derive(Debug)]
pub(crate) struct CpuWorkExceeded {
    what: &'static str,
    rows: usize,
    tokens: usize,
    /// None: the bound does not fit in u64.
    bytes: Option<u64>,
    ceiling: u64,
}

impl std::fmt::Display for CpuWorkExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { what, rows, tokens, bytes, ceiling } = self;
        let needs = bytes.map_or_else(|| "more than u64::MAX".to_string(), |b| b.to_string());
        write!(
            f,
            "CPU {what} of {rows} row(s) padded to {tokens} tokens needs {needs} host bytes by the source working-set \
             bound, above cpu_work_mb ({ceiling} bytes); shorten the input or raise cpu_work_mb"
        )
    }
}

impl std::error::Error for CpuWorkExceeded {}

/// The model dimensions of the formulas (`docs/ARCHITECTURE.md`), in u64.
struct Dims {
    h: u64,
    i: u64,
    v: u64,
    layers: u64,
    nh: u64,
    hd: u64,
    positions: u64,
    /// `nh·hd`, `nkv·hd`, `qd + 2·kvd`, `hd / 2`.
    qd: u64,
    kvd: u64,
    q: u64,
    half: u64,
}

impl Dims {
    fn of(c: &LlamaConfig) -> Option<Self> {
        let (h, i, v, layers) = (n(c.hidden)?, n(c.intermediate)?, n(c.vocab)?, n(c.num_layers)?);
        let (nh, nkv, hd, positions) = (n(c.heads)?, n(c.kv_heads)?, n(c.head_dim)?, n(c.max_positions)?);
        let qd = nh.checked_mul(hd)?;
        let kvd = nkv.checked_mul(hd)?;
        let q = sum(&[Some(qd), kvd.checked_mul(2)])?;
        Some(Self { h, i, v, layers, nh, hd, positions, qd, kvd, q, half: hd / 2 })
    }
}

/// One padded CPU forward: rows, longest row, readouts, distinct letter ids, letter ids.
#[derive(Clone, Copy)]
struct PaddedShape {
    rows: u64,
    len: u64,
    readouts: u64,
    letters: u64,
    letter_ids: u64,
}

fn n(x: usize) -> Option<u64> {
    u64::try_from(x).ok()
}

fn prod(factors: &[u64]) -> Option<u64> {
    factors.iter().try_fold(1u64, |a, &b| a.checked_mul(b))
}

fn sum(terms: &[Option<u64>]) -> Option<u64> {
    terms.iter().try_fold(0u64, |a, &t| a.checked_add(t?))
}

/// `size_of::<Vec<f32>>()`: one returned row vector.
fn vec_bytes() -> u64 {
    std::mem::size_of::<Vec<f32>>() as u64
}

/// Host inputs of `padded_inputs` and their `Tensor::from_slice` copies: ids, cos/sin, mask and read indices, plus
/// the per-row visibility matrix: `2·(4T + 8T·half + 4b·l² + 4r) + l²`.
fn inputs_work(d: &Dims, b: u64, l: u64, r: u64) -> Option<u64> {
    let t = b.checked_mul(l)?;
    let host = sum(&[prod(&[4, t]), prod(&[8, t, d.half]), prod(&[4, b, l, l]), prod(&[4, r])])?;
    sum(&[host.checked_mul(2), prod(&[l, l])])
}

/// `index_select` of BF16 embedding rows and their F32 copy: `6·T·H`.
fn embed_work(d: &Dims, t: u64) -> Option<u64> {
    prod(&[6, t, d.h])
}

/// Every allocation of one layer iteration of `forward_padded_exit`, live until the iteration ends:
/// `4·(T·(9H + 8qd + 9kvd + 6I) + 3·b·nh·l² + H·(Q + qd + 3I) + 4H + Q + 2I)`.
fn layer_work(d: &Dims, b: u64, l: u64) -> Option<u64> {
    let t = b.checked_mul(l)?;
    let per_token = sum(&[prod(&[9, d.h]), prod(&[8, d.qd]), prod(&[9, d.kvd]), prod(&[6, d.i])])?;
    let widened = d.h.checked_mul(sum(&[Some(d.q), Some(d.qd), prod(&[3, d.i])])?)?;
    let inner = sum(&[
        t.checked_mul(per_token),
        prod(&[3, b, d.nh, l, l]),
        Some(widened),
        prod(&[4, d.h]),
        Some(d.q),
        prod(&[2, d.i]),
    ])?;
    inner.checked_mul(4)
}

/// Gathered rows, final norm output and widened norm: `4·(2rH + H)`.
fn final_rows_work(d: &Dims, r: u64) -> Option<u64> {
    sum(&[prod(&[2, r, d.h]), Some(d.h)])?.checked_mul(4)
}

/// `W` of one padded letter forward (`docs/ARCHITECTURE.md`): inputs + embedding + one layer + readout, where the
/// readout is `4(2rH + H) + 4u + 6uH + 8ru + r·VEC + 8n + (b + r)·VEC`.
fn forward_work(d: &Dims, s: &PaddedShape) -> Option<u64> {
    let PaddedShape { rows: b, len: l, readouts: r, letters: u, letter_ids: ids } = *s;
    let vec = vec_bytes();
    sum(&[
        inputs_work(d, b, l, r),
        embed_work(d, b.checked_mul(l)?),
        layer_work(d, b, l),
        final_rows_work(d, r),
        prod(&[4, u]),
        prod(&[6, u, d.h]),
        prod(&[8, r, u]),
        prod(&[r, vec]),
        prod(&[8, ids]),
        prod(&[sum(&[Some(b), Some(r)])?, vec]),
    ])
}

/// `W_ev` of one padded causal evidence forward of `tokens` tokens, span `span` and head width `k`, `r = span + 1`:
/// inputs + embedding + one layer (b = 1) + `4(2rH + H) + 4rH + 16(r+1)H + 8(r+1)k + 32r + 8·span`.
fn evidence_work(d: &Dims, tokens: u64, span: u64, k: u64) -> Option<u64> {
    let r = span.checked_add(1)?;
    let r1 = r.checked_add(1)?;
    sum(&[
        inputs_work(d, 1, tokens, r),
        embed_work(d, tokens),
        layer_work(d, 1, tokens),
        final_rows_work(d, r),
        prod(&[4, r, d.h]),
        prod(&[16, r1, d.h]),
        prod(&[8, r1, k]),
        prod(&[32, r]),
        prod(&[8, span]),
    ])
}

/// Resident bytes of a loaded CPU model:
/// `2·(2VH + H + NL·(2H + H(Q + qd + 3I) + Q + 2H + 2I)) + 4·P·hd + 8·half + E_f32`.
fn resident_work(d: &Dims, evidence_f32: u64) -> Option<u64> {
    let per_layer = sum(&[
        prod(&[2, d.h]),
        d.h.checked_mul(sum(&[Some(d.q), Some(d.qd), prod(&[3, d.i])])?),
        Some(d.q),
        prod(&[2, d.h]),
        prod(&[2, d.i]),
    ])?;
    let weights = sum(&[prod(&[2, d.v, d.h]), Some(d.h), d.layers.checked_mul(per_layer)])?;
    sum(&[weights.checked_mul(2), prod(&[4, d.positions, d.hd]), prod(&[8, d.half]), Some(evidence_f32)])
}

/// Transient load bytes: `4·max(VH, QH, H·qd, 2IH, HI) + header + 4·P·hd + E_src`.
fn load_scratch_work(d: &Dims, header: u64, evidence_source: u64) -> Option<u64> {
    let largest =
        [prod(&[d.v, d.h])?, prod(&[d.q, d.h])?, prod(&[d.h, d.qd])?, prod(&[2, d.i, d.h])?, prod(&[d.h, d.i])?]
            .into_iter()
            .max()?;
    sum(&[largest.checked_mul(4), Some(header), prod(&[4, d.positions, d.hd]), Some(evidence_source)])
}

/// The logical payloads `(E_src, E_f32)` of the evidence head at `path` from its pickle metadata, before any tensor is
/// read: exactly the six tensors `EvidenceHead::load` accepts, unique, `[k, hidden]` projections with one `k` and
/// `[1, hidden]` biases, in a layout its reader supports (contiguous or Fortran-contiguous). Both are 0 without the
/// file. The checkpoint directory is assumed stable: the loader opens the file again afterwards.
fn evidence_payload(path: &Path, hidden: usize) -> Result<(u64, u64)> {
    if !path.exists() {
        return Ok((0, 0));
    }
    let infos = candle_core::pickle::read_pth_tensor_info(path, false, Some("state_dict"))
        .with_context(|| format!("reading the tensor metadata of {}", path.display()))?;
    let file = path.display();
    ensure!(infos.len() == 6, "{file}: {} tensors, the evidence head has exactly six", infos.len());
    let mut seen = BTreeSet::new();
    let mut k = None;
    let (mut source, mut f32_bytes) = (0u64, 0u64);
    for info in &infos {
        let name = info.name.as_str();
        ensure!(seen.insert(name), "{file}: tensor {name} appears twice");
        let dims = info.layout.dims();
        if EVIDENCE_PROJECTIONS.contains(&name) {
            ensure!(dims.len() == 2 && dims[1] == hidden, "{file}: {name} has shape {dims:?}, expected [k, {hidden}]");
            ensure!(
                *k.get_or_insert(dims[0]) == dims[0],
                "{file}: {name} has shape {dims:?}, the projections differ in k"
            );
        } else {
            ensure!(EVIDENCE_BIASES.contains(&name), "{file}: unexpected tensor {name} in the evidence head");
            ensure!(dims == [1, hidden], "{file}: {name} has shape {dims:?}, expected [1, {hidden}]");
        }
        ensure!(
            info.layout.is_contiguous() || info.layout.is_fortran_contiguous(),
            "{file}: {name} has a strided layout the tensor reader does not support"
        );
        let elements = dims.iter().try_fold(1u64, |a, &x| a.checked_mul(n(x)?));
        let bytes = elements.and_then(|e| e.checked_mul(n(info.dtype.size_in_bytes())?));
        let (Some(elements), Some(bytes)) = (elements, bytes) else {
            anyhow::bail!("{file}: {name} payload size overflows");
        };
        // the reader allocates the payload as one vector after skipping `start_offset` bytes of its storage
        ensure!(
            usize::try_from(bytes).is_ok_and(|b| isize::try_from(b).is_ok())
                && n(info.layout.start_offset()).and_then(|o| o.checked_add(bytes)).is_some(),
            "{file}: {name} payload of {bytes} bytes does not fit the tensor reader"
        );
        source = source.checked_add(bytes).context("evidence head payload overflow")?;
        f32_bytes = elements
            .checked_mul(4)
            .and_then(|b| f32_bytes.checked_add(b))
            .context("evidence head F32 payload overflow")?;
    }
    Ok((source, f32_bytes))
}

/// [`CpuHostMemory`] of the dense BF16 Llama checkpoint in `dir` (`model.safetensors`) and its optional evidence head.
pub(crate) fn host_memory(dir: &Path, cfg: &LlamaConfig) -> Result<CpuHostMemory> {
    let d = Dims::of(cfg).context("model dimensions overflow")?;
    let (evidence_source_bytes, evidence_f32_bytes) = evidence_payload(&dir.join(EVIDENCE_FILE), cfg.hidden)?;
    let header = crate::TensorReader::open(&dir.join("model.safetensors"))?.base;
    Ok(CpuHostMemory {
        checkpoint_bytes: basal_core::checkpoint::bytes(dir)?,
        resident_bytes: resident_work(&d, evidence_f32_bytes).context("CPU resident estimate overflow")?,
        load_scratch_bytes: load_scratch_work(&d, header, evidence_source_bytes)
            .context("CPU load scratch estimate overflow")?,
        evidence_source_bytes,
        evidence_f32_bytes,
    })
}

/// The consecutive row groups of one letter call, all checked before any of them runs: a row joins the current group
/// while the group's bound stays within `ceiling` and its padded tokens fit the u32 readout indices; a row that does
/// not fit alone is refused. Positions and non-empty groups are checked here too, so a call that passes runs every
/// group without a refusal of the forward's own input checks.
fn letter_groups(
    d: &Dims,
    cfg: &LlamaConfig,
    rows: &[&Packed],
    letters: &[&[Vec<u32>]],
    ceiling: u64,
) -> Result<Vec<Range<usize>>> {
    ensure!(!rows.is_empty() && rows.len() == letters.len(), "CPU letter forward: rows and letter groups differ");
    // positions restart in every branch of a tree row: the limit applies to the longest prompt, not the row
    let max_pos = rows.iter().flat_map(|r| r.pos.iter()).copied().max().unwrap_or(0) as usize;
    ensure!(
        max_pos < cfg.max_positions,
        "a prompt of {} tokens exceeds the model limit {}",
        max_pos + 1,
        cfg.max_positions
    );
    let fits = |s: &PaddedShape| {
        let work = forward_work(d, s);
        let indexed = s.rows.checked_mul(s.len).is_some_and(|t| t <= u64::from(u32::MAX));
        (work, work.is_some_and(|w| w <= ceiling) && indexed)
    };
    let mut groups = Vec::new();
    let mut start = 0;
    let mut ids = BTreeSet::new();
    let mut group = PaddedShape { rows: 0, len: 0, readouts: 0, letters: 0, letter_ids: 0 };
    for (i, (row, codes)) in rows.iter().zip(letters).enumerate() {
        let row_ids: BTreeSet<u32> = codes.iter().flatten().copied().collect();
        let one = PaddedShape {
            rows: 1,
            len: n(row.ids.len()).context("row length")?,
            readouts: n(row.last.len()).context("readouts")?,
            letters: n(row_ids.len()).context("letters")?,
            letter_ids: n(codes.iter().map(Vec::len).sum()).context("letter ids")?,
        };
        if i > start {
            let joined = (|| {
                Some(PaddedShape {
                    rows: group.rows.checked_add(1)?,
                    len: group.len.max(one.len),
                    readouts: group.readouts.checked_add(one.readouts)?,
                    letters: group.letters.checked_add(n(row_ids.difference(&ids).count())?)?,
                    letter_ids: group.letter_ids.checked_add(one.letter_ids)?,
                })
            })();
            if let Some(joined) = joined.filter(|s| fits(s).1) {
                group = joined;
                ids.extend(row_ids);
                continue;
            }
            ensure!(group.len > 0, "empty batch");
            groups.push(start..i);
            start = i;
        }
        let (work, ok) = fits(&one);
        if !ok {
            ensure!(
                one.len <= u64::from(u32::MAX),
                "CPU letter forward: a row of {} tokens exceeds the u32 readout indices",
                one.len
            );
            return Err(CpuWorkExceeded {
                what: "letter forward",
                rows: 1,
                tokens: row.ids.len(),
                bytes: work,
                ceiling,
            }
            .into());
        }
        group = one;
        ids = row_ids;
    }
    ensure!(group.len > 0, "empty batch");
    groups.push(start..rows.len());
    Ok(groups)
}

impl GpuBackend {
    /// Load `cfg` from `dir` on the host CPU with the shared loader (one [`StageDevice::Cpu`] stage) and the fixed CPU
    /// options; the evidence head's metadata is validated first. [`crate::RuntimeBackend::load_planned`] applies the
    /// CPU plan next.
    pub(crate) fn load_cpu(dir: &Path, cfg: &LlamaConfig, work: CpuWork) -> Result<Self> {
        ensure!(usize::try_from(work.ceiling.get()).is_ok(), "the cpu work ceiling exceeds this host's address space");
        evidence_payload(&dir.join(EVIDENCE_FILE), cfg.hidden)?;
        let mut backend = Self::load_stages(
            dir,
            cfg,
            Precision::F32,
            Readout::F32,
            Kernels::Candle,
            Quantization::None,
            &[(StageDevice::Cpu, 0..cfg.num_layers)],
        )?;
        backend.cpu = Some(work);
        Ok(backend)
    }

    fn cpu_work(&self) -> Result<(CpuWork, Dims)> {
        let work = self.cpu.context("not a CPU model")?;
        Ok((work, Dims::of(&self.cfg).context("model dimensions overflow")?))
    }

    /// [`basal_core::Backend::letter_logits`] on the host CPU: every row group is planned and checked first
    /// ([`letter_groups`]), then each group runs once through `letter_logits_gated` inside its own gate entry, in row
    /// order. The gate is released between groups.
    pub(crate) fn cpu_letter_logits(
        &mut self,
        rows: &[&Packed],
        letters: &[&[Vec<u32>]],
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let (work, d) = self.cpu_work()?;
        let groups = letter_groups(&d, &self.cfg, rows, letters, work.ceiling.get())?;
        let mut out = Vec::with_capacity(rows.len());
        for group in groups {
            let owners = self.gate_owners();
            let mut gates = StageGates::new(&owners);
            let result = self.letter_logits_gated(&mut gates, &rows[group.clone()], &letters[group]);
            out.extend(gates.end(result)?);
        }
        Ok(out)
    }

    /// [`basal_core::Backend::evidence_scores`] on the host CPU: the plain causal forward of the whole prompt as one
    /// padded row with an explicit mask, then the unchanged evidence head; refused before any input exists when its
    /// bound exceeds the ceiling.
    pub(crate) fn cpu_evidence_scores(&self, ids: &[u32], t0: usize, t1: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (work, d) = self.cpu_work()?;
        let tokens = ids.len();
        ensure!(t0 < t1 && t1 <= tokens, "evidence: token range {t0}..{t1} outside {tokens} tokens");
        ensure!(tokens <= self.cfg.max_positions, "evidence: {tokens} tokens exceed the model limit");
        let read_end = u32::try_from(tokens).context("evidence: tokens exceed the u32 readout indices")?;
        let k = self.head()?.evidence.as_ref().context("this model has no evidence head")?.k;
        let bytes = (|| evidence_work(&d, n(tokens)?, n(t1 - t0)?, n(k)?))();
        let ceiling = work.ceiling.get();
        if bytes.is_none_or(|b| b > ceiling) {
            return Err(CpuWorkExceeded { what: "evidence forward", rows: 1, tokens, bytes, ceiling }.into());
        }
        let read: Vec<u32> = (u32::try_from(t0)?..u32::try_from(t1)?).chain(std::iter::once(read_end - 1)).collect();
        let owners = self.gate_owners();
        let mut gates = StageGates::new(&owners);
        let result = self.cpu_evidence_gated(&mut gates, ids, t1 - t0, &read);
        gates.end(result)
    }

    /// The evidence forward inside the call's gate: RoPE rows of positions `0..n`, a causal mask (`MASKED` above the
    /// diagonal), the span's `m` rows then the answer position read, the evidence head on F32 rows.
    fn cpu_evidence_gated(
        &self,
        gates: &mut StageGates<'_>,
        ids: &[u32],
        m: usize,
        read: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        gates.switch_to(0)?;
        let tokens = ids.len();
        let half = self.cfg.head_dim / 2;
        let mut cs = vec![0f32; 2 * tokens * half];
        for t in 0..tokens {
            let (cos, sin) = cs.split_at_mut(tokens * half);
            self.rope_row(t as f64, &mut cos[t * half..(t + 1) * half], &mut sin[t * half..(t + 1) * half]);
        }
        let mut mask = vec![MASKED; tokens * tokens];
        for t in 0..tokens {
            mask[t * tokens..t * tokens + t + 1].fill(0.0);
        }
        let h = self.forward_padded(ids, &cs, &mask, 1, tokens, read)?.to_dtype(DType::F32)?;
        let head = self.head()?.evidence.as_ref().context("this model has no evidence head")?;
        let scores = head.scores(&h.narrow(0, 0, m)?, &h.narrow(0, m, 1)?)?;
        // the second side's copy to host vectors was the last work of this call
        gates.mark_drained();
        Ok(scores)
    }

    /// [`basal_core::Backend::describe`] of a CPU model.
    pub(crate) fn cpu_describe(&self, cpu: &CpuWork) -> Value {
        let mut d = json!({
            "backend": "cpu",
            "device": "host CPU through candle Device::Cpu",
            "tensor_library": "candle-core 0.11.0 (CPU; matmul = gemm 0.19.0)",
            "weights": "bf16 resident, widened to f32 per layer and use",
            "activations": "f32 everywhere",
            "attention": "f32: RoPE (candle rope), scores matmul, additive block mask, softmax, weighted sum",
            "kernels": "candle ops only: GEMM + broadcast bias add, separate SiLU, mul and residual add",
            "readout": "letter rows of lm_head only, f32",
            "padding": "rows padded on the right to the longest packed row of a CPU sub-batch",
            "attention_kernel": "padded",
            "gemm": "gemm 0.19.0 f32 matmul of the candle CPU backend, instruction set chosen at run time",
            "prefix_kv": [],
            "state_cache_bytes": self.state_cache_bytes,
            "last_layer": "full",
            "cpu": {
                "schema": "basal-cpu-execution.1",
                "work_bytes": cpu.ceiling.get(),
                "gemm_threads": cpu.gemm_threads,
                "thread_pool": "the process-global Rayon pool (RAYON_NUM_THREADS, else the available parallelism)",
                "sub_batches": "consecutive rows grouped while the source working-set bound of a group fits work_bytes; all groups are checked before the first runs",
                "bit_identity": "none across hosts, instruction sets, thread counts or groupings",
            },
        });
        if self.head().is_ok_and(|h| h.evidence.is_some()) {
            d["evidence"] = json!("padded causal f32 forward of the whole prompt, then the f32 evidence head");
        }
        d
    }
}
