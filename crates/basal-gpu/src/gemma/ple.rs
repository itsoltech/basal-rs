//! Immutable host PLE storage. Selection and transfer buffers belong to one call, never to a fork.

use anyhow::{ensure, Context, Result};
use basal_core::architecture::GemmaConfig;
use candle_core::{Device, Tensor, WithDType};

use crate::Precision;

/// Placement of Gemma's per-layer token table, independent of its contextual projection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlePlacement {
    #[default]
    Device,
    /// Single-CUDA Gemma only. The ceiling covers one call's selected rows, a transfer staging
    /// copy and its PLE IDs. Concurrent workers need separate reservations in addition to residency.
    Host { max_staging_bytes: u64 },
}

pub(super) enum PleEmbedding {
    Device(Tensor),
    Host(HostPle),
}

enum Rows {
    Bf16(Vec<half::bf16>),
    F16(Vec<half::f16>),
}

pub(super) struct HostPle {
    rows: Rows,
    width: usize,
    vocab: usize,
    pub resident_bytes: u64,
    pub max_staging_bytes: u64,
}

// Short buckets have <=8192 padded tokens; non-unified long/media chunks have <=2048.
const MAX_ROWS: usize = 8192;
// One gathered Vec plus one conservative pageable transfer copy. Candle's ordinary from_slice
// CUDA path borrows the Vec without copying it; no CPU Tensor or index tensor is constructed.
const ROW_COPIES: u64 = 2;

pub(super) fn dimensions(cfg: &GemmaConfig) -> Result<(usize, u64)> {
    cfg.validate()?;
    ensure!(
        cfg!(feature = "cuda") && !cfg.unified && cfg.per_layer_input > 0,
        "host PLE requires single-CUDA non-unified Gemma with per-layer input embeddings"
    );
    let width = cfg.layers.len().checked_mul(cfg.per_layer_input).context("host PLE row width overflow")?;
    let bytes = cfg
        .per_layer_vocab
        .checked_mul(width)
        .and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= isize::MAX as usize)
        .context("host PLE table allocation overflow")?;
    Ok((width, u64::try_from(bytes)?))
}

impl HostPle {
    pub fn load(data: &[u8], cfg: &GemmaConfig, precision: Precision, max_staging_bytes: u64) -> Result<Self> {
        let (width, resident_bytes) = dimensions(cfg)?;
        ensure!(u64::try_from(data.len())? == resident_bytes, "host PLE checkpoint byte length mismatch");
        let elements = data.len() / 2;
        let rows = if precision == Precision::F16 {
            let mut rows = Vec::new();
            rows.try_reserve_exact(elements).context("allocating host F16 PLE table")?;
            rows.extend(
                data.chunks_exact(2).map(|b| half::f16::from_f32(half::bf16::from_le_bytes([b[0], b[1]]).to_f32())),
            );
            Rows::F16(rows)
        } else {
            // F32 computation also keeps BF16 resident weights, widening only after selection.
            let mut rows = Vec::new();
            rows.try_reserve_exact(elements).context("allocating host BF16 PLE table")?;
            rows.extend(data.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]])));
            Rows::Bf16(rows)
        };
        Ok(Self { rows, width, vocab: cfg.per_layer_vocab, resident_bytes, max_staging_bytes })
    }

    /// Additional coefficients over the unchanged device-placement host bound. Padding contributes
    /// at most 31 rows per prompt/readout; chunks reuse the reservation, with no retained host rows.
    pub fn staging_coefficients(width: usize) -> Result<(u64, u64)> {
        let per_token = u64::try_from(width)?
            .checked_mul(2)
            .and_then(|n| n.checked_mul(ROW_COPIES))
            .context("host PLE staging coefficient overflow")?;
        Ok((per_token, per_token.checked_mul(31).context("host PLE padding coefficient overflow")?))
    }

    /// Per-call ceiling `rows*(4*width+4)`: selected rows, one transfer copy and u32 PLE IDs.
    pub fn staging_bytes(width: usize, rows: usize) -> Result<u64> {
        let (per_token, _) = Self::staging_coefficients(width)?;
        per_token
            .checked_add(4)
            .and_then(|n| n.checked_mul(rows as u64))
            .context("host PLE staging byte count overflow")
    }

    /// Check before allocating even the substituted host ID vector.
    pub fn check_staging(&self, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= MAX_ROWS, "host PLE step bounds");
        let bytes = Self::staging_bytes(self.width, rows)?;
        ensure!(
            bytes <= self.max_staging_bytes,
            "host PLE staging requires {bytes} bytes, limit {}",
            self.max_staging_bytes
        );
        Ok(())
    }

    pub fn select(&self, ids: &[u32], dev: &Device, health: &basal_core::DeviceHealth) -> Result<Tensor> {
        ensure!(dev.is_cuda(), "host PLE requires CUDA");
        self.check_staging(ids.len())?;
        ensure!(ids.iter().all(|id| (*id as usize) < self.vocab), "host PLE token outside per-layer vocabulary");
        match &self.rows {
            Rows::Bf16(rows) => self.upload(rows, ids, dev, health),
            Rows::F16(rows) => self.upload(rows, ids, dev, health),
        }
    }

    fn upload<T: WithDType>(
        &self,
        rows: &[T],
        ids: &[u32],
        dev: &Device,
        health: &basal_core::DeviceHealth,
    ) -> Result<Tensor> {
        // try_reserve_exact rejects byte sizes above isize::MAX without panicking.
        let elements = ids.len().checked_mul(self.width).context("host PLE selected allocation overflow")?;
        let mut selected = Vec::new();
        selected.try_reserve_exact(elements).context("allocating selected host PLE rows")?;
        for &id in ids {
            let start = (id as usize).checked_mul(self.width).context("host PLE row offset overflow")?;
            let end = start.checked_add(self.width).context("host PLE row end overflow")?;
            selected.extend_from_slice(rows.get(start..end).context("host PLE row outside table")?);
        }
        // from_slice -> CUDA storage_from_slice -> clone_htod borrows selected. cudarc's
        // Vec/slice HostSlice does NOT synchronize on drop, even with event tracking enabled.
        // Keep selected alive through an explicit drain, including a failed upload. On two failed
        // drains retain the bounded buffer and quarantine the lineage rather than free in-flight data.
        let uploaded = Tensor::from_slice(&selected, (ids.len(), self.width), dev).context("uploading host PLE rows");
        match crate::twice(|| dev.synchronize()) {
            crate::Twice::First => uploaded,
            crate::Twice::Second(error) => {
                let context = format!("host PLE transfer drain failed on first attempt ({error}); retry completed");
                Err(uploaded.err().unwrap_or_else(|| error.into()).context(context))
            }
            crate::Twice::Neither(first, second) => {
                health.mark_lost();
                std::mem::forget(selected);
                let context = format!("host PLE transfer drain failed twice ({first}; {second}); {}", crate::LOST);
                Err(uploaded.err().unwrap_or_else(|| first.into()).context(context))
            }
        }
    }
}
