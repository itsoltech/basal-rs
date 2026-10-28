//! Published trained heads on the single-device CUDA Candle Llama path.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use basal_core::early_exit::{ExitArtifacts, ExitCaps};
use candle_core::{DType, Device, Tensor};

pub(super) struct ExitHeads {
    pub caps: ExitCaps,
    heads: BTreeMap<usize, Head>,
    pub resident_bytes: usize,
}

struct Head {
    norm: Tensor,
    a: Tensor,
    b: Tensor,
}

/// A final-normalized hidden readout or the actual trained-head logits (tail not executed).
pub(super) enum PaddedOutput {
    Hidden(Tensor),
    Exit(Vec<Vec<Vec<f32>>>),
}

/// Called after the complete decoder residual update, never on the next layer's normalized input.
pub(super) type ExitCheck<'a> = dyn FnMut(usize, &Tensor) -> Result<Option<Vec<Vec<Vec<f32>>>>> + 'a;

impl ExitHeads {
    pub fn load(artifacts: &ExitArtifacts, hidden: usize, layers: usize) -> Result<Self> {
        let tensors = candle_core::pickle::read_all_with_key(artifacts.heads_path(), Some("state"))
            .context("reading trained exit heads state without Python execution")?;
        ensure!(!tensors.is_empty(), "empty trained exit state");
        let mut groups: BTreeMap<usize, BTreeMap<String, Tensor>> = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for (name, tensor) in tensors {
            let (layer, suffix) = name.split_once('.').context("exit tensor requires <L>.<name>")?;
            let layer: usize = layer.parse().context("exit layer must be an integer depth")?;
            ensure!(layer > 0 && layer < layers, "exit head {layer} outside intermediate decoder range");
            ensure!(matches!(suffix, "norm.weight" | "a.weight" | "b.weight"), "unknown exit tensor {name}");
            ensure!(
                matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16),
                "exit tensor {name} is not floating point"
            );
            let tensor = tensor.to_dtype(DType::F32)?;
            ensure!(
                tensor.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()),
                "nonfinite exit tensor {name}"
            );
            resident_bytes = resident_bytes
                .checked_add(tensor.elem_count().checked_mul(4).context("exit tensor bytes overflow")?)
                .context("resident exit bytes overflow")?;
            ensure!(
                groups.entry(layer).or_default().insert(suffix.into(), tensor).is_none(),
                "duplicate exit tensor {name}"
            );
        }
        let mut heads = BTreeMap::new();
        for (layer, mut tensors) in groups {
            let mut take = |name| tensors.remove(name).with_context(|| format!("exit {layer}: missing {name}"));
            let (norm, a, b) = (take("norm.weight")?, take("a.weight")?, take("b.weight")?);
            ensure!(norm.dims() == [hidden], "exit {layer}: invalid norm shape");
            let (rank, width) = a.dims2().context("exit a.weight must be [rank,hidden]")?;
            ensure!(
                rank > 0 && width == hidden && b.dims() == [hidden, rank],
                "exit {layer}: invalid adapter shapes/rank"
            );
            heads.insert(layer, Head { norm, a, b });
        }
        for layer in artifacts.caps().layers() {
            ensure!(heads.contains_key(&layer), "threshold refers to missing trained head {layer}");
        }
        Ok(Self { caps: artifacts.caps().clone(), heads, resident_bytes })
    }

    /// The stage owns this value before the first upload, including partial failure and publication failure.
    pub fn upload(&mut self, dev: &Device) -> Result<()> {
        for head in self.heads.values_mut() {
            head.norm = head.norm.to_device(dev)?;
            head.a = head.a.to_device(dev)?;
            head.b = head.b.to_device(dev)?;
        }
        dev.synchronize().context("publishing trained heads")?;
        Ok(())
    }

    pub fn readout(
        &self,
        layer: usize,
        raw: &Tensor,
        eps: f64,
        lm: &Tensor,
        letters: &[&[Vec<u32>]],
        policy: &str,
    ) -> Result<Option<Vec<Vec<Vec<f32>>>>> {
        let Some(&tau) = self.caps.thresholds(policy)?.get(&layer).filter(|&&t| t <= 1.0) else { return Ok(None) };
        let head = self.heads.get(&layer).context("validated trained head missing")?;
        let h = raw.to_dtype(DType::F32)?;
        // Trained RMSNorm in FP32, model epsilon; no final norm or next-layer norm is substituted.
        let x = h.broadcast_div(&(h.sqr()?.mean_keepdim(1)? + eps)?.sqrt()?)?.broadcast_mul(&head.norm)?;
        let adapted = (&x + x.matmul(&head.a.t()?)?.matmul(&head.b.t()?)?)?;
        let logits = project(&adapted, lm, letters)?;
        ensure!(logits.iter().flatten().flatten().all(|x| x.is_finite()), "nonfinite trained-head logits");
        let confident = logits
            .iter()
            .flatten()
            .all(|values| basal_core::decision::softmax32(values).into_iter().fold(0.0f64, f64::max) >= tau);
        Ok(confident.then_some(logits))
    }
}

/// Frozen base LM rows, gathered per readout. Every softmax contains exactly that readout's allowed codes.
pub(super) fn project(hidden: &Tensor, lm: &Tensor, letters: &[&[Vec<u32>]]) -> Result<Vec<Vec<Vec<f32>>>> {
    let mut readout = 0;
    let mut out = Vec::with_capacity(letters.len());
    for row in letters {
        let mut values = Vec::with_capacity(row.len());
        for ids in *row {
            ensure!(!ids.is_empty(), "exit readout has no allowed codes");
            let index = Tensor::from_slice(ids, ids.len(), hidden.device())?;
            let weights = lm.index_select(&index, 0)?.to_dtype(DType::F32)?;
            values.push(
                hidden.narrow(0, readout, 1)?.to_dtype(DType::F32)?.matmul(&weights.t()?)?.flatten_all()?.to_vec1()?,
            );
            readout += 1;
        }
        out.push(values);
    }
    ensure!(readout == hidden.dim(0)?, "exit readout count mismatch");
    Ok(out)
}
