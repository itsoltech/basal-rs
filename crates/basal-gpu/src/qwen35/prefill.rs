//! Chunked prefill: up to eight prompts visit each decoder layer together, each with its own chunk boundaries,
//! GEMM shapes and reductions. Shared GEMMs are used only where a row-invariant table qualifies them.

use std::time::Instant;

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Tensor};
use serde_json::json;

use super::{norm, Layer, Mixer, PrefillRow, Qwen35Backend, State, Weights, MAX_GROUP};
use crate::StageGates;

impl Qwen35Backend {
    /// Prefills every prompt from absolute position `start` into its `states` and returns each prompt's last
    /// hidden row.
    pub(super) fn prefill_many(
        &self,
        prompts: &[&[u32]],
        short_contexts: &[bool],
        start: usize,
        states: &mut [Vec<State>],
        gates: &mut StageGates<'_>,
    ) -> Result<Vec<Tensor>> {
        ensure!(
            !prompts.is_empty()
                && prompts.len() <= MAX_GROUP
                && states.len() == prompts.len()
                && short_contexts.len() == prompts.len(),
            "invalid Qwen3.5 prefill group"
        );
        let weights = self.weights.get();
        if crate::plan::DiagnosticOverrides::process().host_timing() {
            eprintln!(
                "{}",
                json!({"qwen35_group":prompts.len(),"chunk":self.chunk,"start":start,"lengths":prompts.iter().map(|p|p.len()).collect::<Vec<_>>()})
            );
        }
        let mut last = vec![None; prompts.len()];
        let phase_start = crate::plan::DiagnosticOverrides::process().phase_timing().then(Instant::now);
        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        // A prompt keeps its own chunk boundaries and attention path when another prompt joins the group.
        // Visit short boundaries too; long rows are active only at their fixed larger boundaries.
        let stride = self.chunk.min(512);
        for chunk_start in (0..longest).step_by(stride) {
            let offset = start + chunk_start;
            let mut rows = prompts
                .iter()
                .zip(short_contexts)
                .map(|(ids, &short_context)| -> Result<Option<PrefillRow>> {
                    let chunk = if short_context { stride } else { self.chunk };
                    if chunk_start >= ids.len() || chunk_start % chunk != 0 {
                        return Ok(None);
                    }
                    let ids = &ids[chunk_start..ids.len().min(chunk_start + chunk)];
                    self.prefill_row(weights, ids, offset, short_context).map(Some)
                })
                .collect::<Result<Vec<_>>>()?;
            if rows.iter().all(Option::is_none) {
                continue;
            }
            for (i, layer) in weights.layers.iter().enumerate() {
                self.layer(i, layer, &mut rows, states, gates)?;
            }
            for (row, last) in rows.into_iter().zip(&mut last) {
                if let Some(row) = row {
                    *last = Some(row.x.narrow(0, row.x.dim(0)? - 1, 1)?);
                }
            }
        }
        if let Some(t) = phase_start {
            eprintln!(
                "{}",
                json!({"pico_enqueue":true,"tokens":prompts.iter().map(|p|p.len()).sum::<usize>(),
                "rows":prompts.len(),"elapsed_ms":t.elapsed().as_secs_f64()*1000.0})
            );
        }
        last.into_iter().map(|x| x.context("Qwen3.5 final hidden state")).collect()
    }

    pub(super) fn prefill(
        &self,
        ids: &[u32],
        short_context: bool,
        start: usize,
        states: &mut Vec<State>,
        gates: &mut StageGates<'_>,
    ) -> Result<Tensor> {
        self.prefill_many(&[ids], &[short_context], start, std::slice::from_mut(states), gates)?
            .pop()
            .context("Qwen3.5 prefill output")
    }

    /// Embeddings, partial-RoPE tables and causal mask of the chunk `ids` at absolute position `offset`.
    fn prefill_row(&self, weights: &Weights, ids: &[u32], offset: usize, short_context: bool) -> Result<PrefillRow> {
        let dtype = self.precision.compute();
        let n = ids.len();
        let positions =
            Tensor::arange(offset as u32, (offset + n) as u32, &self.dev)?.to_dtype(DType::F32)?.reshape((n, 1))?;
        let freq = positions.broadcast_mul(&weights.rope_inv)?;
        let cos = freq.cos()?.to_dtype(dtype)?.unsqueeze(1)?;
        let sin = freq.sin()?.to_dtype(dtype)?.unsqueeze(1)?;
        let total = offset + n;
        let mask = if self.positional_attention() {
            // The positional attention path never reads a dense mask.
            Tensor::zeros((1, 1), DType::F32, &self.dev)?
        } else {
            Tensor::from_vec(
                (0..n)
                    .flat_map(|i| (0..total).map(move |j| if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }))
                    .collect::<Vec<_>>(),
                (n, total),
                &self.dev,
            )?
        };
        let token_ids = Tensor::new(ids, &self.dev)?;
        let x = weights.embed.index_select(&token_ids, 0)?.to_dtype(dtype)?;
        Ok(PrefillRow { x, cos, sin, mask, short_context })
    }

    /// Decoder layer `i` for every active row. Visiting the same layer of all prompts consecutively lets mixer and
    /// MLP weights stay in the GPU's cache between prompts.
    fn layer(
        &self,
        i: usize,
        layer: &Layer,
        rows: &mut [Option<PrefillRow>],
        states: &mut [Vec<State>],
        gates: &mut StageGates<'_>,
    ) -> Result<()> {
        let inputs = rows
            .iter()
            .map(|row| row.as_ref().map(|r| norm(&r.x, &layer.input_norm, self.cfg.rms_eps)).transpose())
            .collect::<Result<Vec<_>>>()?;
        let active_inputs = inputs.iter().flatten().cloned().collect::<Vec<_>>();
        let projected = match &layer.mixer {
            Mixer::DeltaNet(w) => {
                w.input.as_ref().map(|w| self.shared_linear(&active_inputs, w)).transpose()?.flatten()
            }
            Mixer::Attention(_) => None,
        };
        let mut mixed_rows = Vec::with_capacity(active_inputs.len());
        for ((row, states), input) in rows.iter().zip(&mut *states).zip(&inputs) {
            let Some(row) = row else {
                continue;
            };
            gates.checkpoint()?;
            let input = input.as_ref().context("Qwen3.5 active prefill input")?;
            let active = mixed_rows.len();
            let mixed = match &layer.mixer {
                Mixer::Attention(w) => self.attention(w, input, &mut states[i], row)?,
                Mixer::DeltaNet(w) => {
                    self.delta(w, input, &mut states[i], projected.as_ref().map(|p| &p[active]), gates)?
                }
            };
            mixed_rows.push(mixed);
        }
        let output_weight = layer.mixer.output();
        let outputs = match self.shared_linear(&mixed_rows, output_weight)? {
            Some(out) => out,
            None => mixed_rows.iter().map(|x| self.linear(x, output_weight)).collect::<Result<Vec<_>>>()?,
        };
        for (row, output) in rows.iter_mut().flatten().zip(outputs) {
            row.x = (&row.x + output)?;
        }
        if self.shared_mlp(layer, rows)? {
            return Ok(());
        }
        for row in rows.iter_mut().flatten() {
            gates.checkpoint()?;
            row.x = self.mlp(layer, &row.x)?;
        }
        Ok(())
    }

    /// Post-attention RMSNorm, SwiGLU MLP and residual.
    fn mlp(&self, layer: &Layer, x: &Tensor) -> Result<Tensor> {
        let input = norm(x, &layer.post_norm, self.cfg.rms_eps)?;
        let hidden = self.swiglu(layer, &input)?;
        Ok((x + self.linear(&hidden, &layer.down)?)?)
    }

    fn swiglu(&self, layer: &Layer, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(gate_up) = &layer.gate_up {
            return Ok(crate::fused::qwen35_swiglu(&self.linear(input, gate_up)?)?);
        }
        Ok((candle_nn::ops::silu(&self.linear(input, &layer.gate)?)? * self.linear(input, &layer.up)?)?)
    }

    /// Runs `f` once on the row-concatenated `inputs` and splits its output back per input. Only for several
    /// inputs under a qualified row-invariant table that covers every `weights` shape at the total row count;
    /// otherwise returns `None`, and callers keep the separate-prompt fallback of a one-prompt forward.
    fn shared_rows(
        &self,
        inputs: &[Tensor],
        weights: &[&Tensor],
        f: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Option<Vec<Tensor>>> {
        #[cfg(feature = "cuda")]
        if inputs.len() > 1 {
            if let Some(lt) = &self.weights.get().lt {
                let rows = inputs.iter().map(|x| x.dim(0)).sum::<candle_core::Result<usize>>()?;
                let dtype = inputs[0].dtype();
                let mut covered = lt.is_invariant();
                for w in weights {
                    covered = covered && lt.covers_classes(w.dim(0)?, w.dim(1)?, dtype, rows);
                }
                if covered {
                    let output = f(&Tensor::cat(inputs, 0)?)?;
                    let mut offset = 0;
                    return inputs
                        .iter()
                        .map(|x| {
                            let n = x.dim(0)?;
                            let part = output.narrow(0, offset, n)?;
                            offset += n;
                            Ok(part)
                        })
                        .collect::<Result<Vec<_>>>()
                        .map(Some);
                }
            }
        }
        let _ = (inputs, weights, f);
        Ok(None)
    }

    fn shared_linear(&self, inputs: &[Tensor], weight: &Tensor) -> Result<Option<Vec<Tensor>>> {
        self.shared_rows(inputs, &[weight], |x| self.linear(x, weight))
    }

    /// The MLP of all active rows as one merged gate/up GEMM, when [`Self::shared_rows`] qualifies it.
    fn shared_mlp(&self, layer: &Layer, rows: &mut [Option<PrefillRow>]) -> Result<bool> {
        let Some(gate_up) = &layer.gate_up else {
            return Ok(false);
        };
        let inputs = rows.iter().flatten().map(|row| row.x.clone()).collect::<Vec<_>>();
        let Some(outputs) = self.shared_rows(&inputs, &[gate_up, &layer.down], |x| self.mlp(layer, x))? else {
            return Ok(false);
        };
        for (row, x) in rows.iter_mut().flatten().zip(outputs) {
            row.x = x;
        }
        Ok(true)
    }
}
