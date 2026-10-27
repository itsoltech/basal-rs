//! Packed tree rows: per-readout prompts, immutable shared-prefix forks and letter readouts.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use basal_core::pack::Packed;
use candle_core::{DType, Tensor};
use serde_json::json;

use super::{is_short, norm, Qwen35Backend, State, CHUNK, MAX_GROUP};
use crate::{Readout, StageGates};

/// The prompt of every readout of a packed tree row: its ancestor-or-self tokens, which must have contiguous
/// positions from zero.
fn row_prompts(row: &Packed, readouts: usize) -> Result<Vec<Vec<u32>>> {
    ensure!(
        row.ids.len() == row.pos.len() && row.ids.len() == row.node.len() && row.last.len() == readouts,
        "invalid packed Qwen3.5 row"
    );
    ensure!(
        !row.parent.is_empty()
            && row.parent[0] == -1
            && row.parent.iter().enumerate().skip(1).all(|(i, p)| *p >= 0 && (*p as usize) < i)
            && row.node.iter().all(|n| (*n as usize) < row.parent.len()),
        "invalid Qwen3.5 tree"
    );
    row.last
        .iter()
        .map(|&last| {
            ensure!(last < row.ids.len(), "invalid Qwen3.5 readout position");
            let mut ids = Vec::new();
            for i in 0..=last {
                if row.is_ancestor_or_self(row.node[i], row.node[last]) {
                    ensure!(row.pos[i] as usize == ids.len(), "non-contiguous Qwen3.5 prompt positions");
                    ids.push(row.ids[i]);
                }
            }
            Ok(ids)
        })
        .collect()
}

impl Qwen35Backend {
    fn valid_prompt(&self, ids: &[u32]) -> bool {
        !ids.is_empty() && ids.len() <= self.cfg.max_positions && ids.iter().all(|id| (*id as usize) < self.cfg.vocab)
    }

    fn valid_codes(&self, codes: &[u32]) -> bool {
        !codes.is_empty() && codes.iter().all(|id| (*id as usize) < self.cfg.vocab)
    }

    /// Letter logits of every readout of one packed row.
    pub(super) fn read_row(
        &self,
        row: &Packed,
        codes: &[Vec<u32>],
        gates: &mut StageGates<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let prompts = row_prompts(row, codes.len())?;
        // Share only whole reference chunks inside this call. Identical GEMM shapes, causal masks
        // and absolute positions preserve the separate-prompt arithmetic. Tensor clones are immutable
        // snapshots: each child writes fresh convolution, recurrent and K/V storage.
        let first = prompts.first().context("Qwen3.5 row has no readouts")?;
        let common = self.shared_prefix(&prompts);
        let prefix = self.prefill_prefix(&first[..common], first.len(), gates)?;
        if self.chunk > CHUNK {
            return self.read_grouped(&prompts, codes, common, &prefix, gates);
        }
        prompts
            .iter()
            .zip(codes)
            .map(|(ids, codes)| {
                self.read_prompt(ids, codes, (common > 0).then_some((common, prefix.as_slice())), gates)
            })
            .collect()
    }

    /// Tokens shared by every prompt before its last token. Without qualified arbitrary prefixes for short
    /// prompts, only whole prefill chunks are shared.
    fn shared_prefix(&self, prompts: &[Vec<u32>]) -> usize {
        if prompts.len() < 2 {
            return 0;
        }
        let first = &prompts[0];
        let common =
            (0..first.len()).take_while(|&i| prompts.iter().all(|p| i + 1 < p.len() && p[i] == first[i])).count();
        if self.arbitrary_prefix && prompts.iter().all(|p| is_short(p.len())) {
            common
        } else {
            common / self.chunk * self.chunk
        }
    }

    /// States after prefilling the shared `prefix` of a `prompt_len`-token prompt; fresh states when it is empty.
    fn prefill_prefix(&self, prefix: &[u32], prompt_len: usize, gates: &mut StageGates<'_>) -> Result<Vec<State>> {
        let mut states = self.fresh_states();
        if !prefix.is_empty() {
            gates.switch_to(0)?;
            self.prefill(prefix, is_short(prompt_len), 0, &mut states, gates)?;
        }
        Ok(states)
    }

    /// Optimized kernel sets: each group of up to eight suffixes forks from the prefix states and prefills together.
    fn read_grouped(
        &self,
        prompts: &[Vec<u32>],
        codes: &[Vec<u32>],
        common: usize,
        prefix: &[State],
        gates: &mut StageGates<'_>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(prompts.len());
        for (group, codes) in prompts.chunks(MAX_GROUP).zip(codes.chunks(MAX_GROUP)) {
            ensure!(group.iter().all(|ids| self.valid_prompt(ids)), "invalid Qwen3.5 prompt");
            gates.switch_to(0)?;
            let mut states = vec![prefix.to_vec(); group.len()];
            let short_contexts = group.iter().map(|p| is_short(p.len())).collect::<Vec<_>>();
            let suffixes: Vec<_> = group.iter().map(|p| &p[common..]).collect();
            let hidden = self.prefill_many(&suffixes, &short_contexts, common, &mut states, gates)?;
            for (last, codes) in hidden.iter().zip(codes) {
                out.push(self.readout(last, codes)?);
            }
        }
        Ok(out)
    }

    /// Reference chunking: one prompt at a time, continuing from the `prefix` states when given.
    fn read_prompt(
        &self,
        ids: &[u32],
        codes: &[u32],
        prefix: Option<(usize, &[State])>,
        gates: &mut StageGates<'_>,
    ) -> Result<Vec<f32>> {
        ensure!(self.valid_prompt(ids) && self.valid_codes(codes), "invalid Qwen3.5 prompt or readout codes");
        gates.switch_to(0)?;
        let (start, mut states) = match prefix {
            Some((len, states)) => (len, states.to_vec()),
            None => (0, self.fresh_states()),
        };
        ensure!(start < ids.len() && start % self.chunk == 0, "invalid Qwen3.5 prefix boundary");
        let last = self.prefill(&ids[start..], is_short(ids.len()), start, &mut states, gates)?;
        self.readout(&last, codes)
    }

    fn readout(&self, last: &Tensor, codes: &[u32]) -> Result<Vec<f32>> {
        let timing = crate::plan::DiagnosticOverrides::process().phase_timing().then(Instant::now);
        ensure!(self.valid_codes(codes), "invalid Qwen3.5 readout codes");
        let weights = self.weights.get();
        let hidden = norm(last, &weights.norm, self.cfg.rms_eps)?.to_dtype(DType::F32)?;
        let codes = Tensor::new(codes, &self.dev)?;
        let head = weights.embed.index_select(&codes, 0)?.to_dtype(DType::F32)?;
        let logits = hidden.matmul(&head.t()?)?;
        let logits = if self.readout == Readout::Bf16Rounded {
            logits.to_dtype(DType::BF16)?.to_dtype(DType::F32)?
        } else {
            logits
        };
        let result = logits.squeeze(0)?.to_vec1()?;
        if let Some(t) = timing {
            eprintln!("{}", json!({"pico_readout_and_wait":true,"elapsed_ms":t.elapsed().as_secs_f64()*1000.0}));
        }
        Ok(result)
    }
}
