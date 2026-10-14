//! Layer placement of one model on several CUDA devices of one process (`--layer-split 0:30,1:30`, YAML
//! `layer_split`): consecutive ranges of decoder layers, one range per GPU, in layer order. Only the CUDA backend
//! with f16 activations and the fused kernels runs a split model; one GPU keeps using `device`.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Kernels, Precision};

/// One stage of a layer split: the next `layers` decoder layers on CUDA device `device` (an ordinal of this process,
/// after `CUDA_VISIBLE_DEVICES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageSpec {
    pub device: usize,
    pub layers: usize,
}

/// Stages of one model in layer order. Invariants (checked by every constructor): at least two stages, every stage
/// with at least one layer, every device at most once, and the layer counts add up without overflow. Whether the sum
/// equals the model's layer count is checked against its configuration ([`LayerSplit::ranges`]).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "Vec<StageSpec>")]
pub struct LayerSplit {
    stages: Vec<StageSpec>,
}

/// A layer split that cannot be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerSplitError {
    /// `--layer-split` text that is not `DEVICE:LAYERS` pairs separated by commas.
    Parse { input: String, reason: String },
    /// Fewer than two stages (one GPU uses `device`).
    TooFewStages { stages: usize },
    /// A stage without layers.
    ZeroLayers { stage: usize, device: usize },
    /// A device given for two stages (each GPU holds one consecutive range).
    DuplicateDevice { device: usize },
    /// The stages do not cover exactly the model's layers.
    Coverage { covered: usize, layers: usize },
    /// The backend, dtype or kernel set cannot run a split model.
    Unsupported { reason: String },
}

impl fmt::Display for LayerSplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse { input, reason } => {
                write!(f, "layer split {input:?}: {reason} (expected DEVICE:LAYERS pairs in layer order, e.g. 0:30,1:30)")
            }
            Self::TooFewStages { stages } => write!(
                f,
                "a layer split needs at least two stages, got {stages} (one GPU: give `device` instead of a layer split)"
            ),
            Self::ZeroLayers { stage, device } => {
                write!(f, "layer split: stage {stage} on device {device} has no layers (give at least one)")
            }
            Self::DuplicateDevice { device } => write!(
                f,
                "layer split: device {device} is given twice (each GPU holds one consecutive range of layers)"
            ),
            Self::Coverage { covered, layers } => {
                write!(f, "layer split covers {covered} of {layers} layers (the counts must add up to the model's layers)")
            }
            Self::Unsupported { reason } => write!(f, "layer split: {reason}"),
        }
    }
}

impl std::error::Error for LayerSplitError {}

impl LayerSplit {
    /// Validate stages given in layer order.
    pub fn new(stages: Vec<StageSpec>) -> Result<Self, LayerSplitError> {
        if stages.len() < 2 {
            return Err(LayerSplitError::TooFewStages { stages: stages.len() });
        }
        let mut total = 0usize;
        for (i, s) in stages.iter().enumerate() {
            if s.layers == 0 {
                return Err(LayerSplitError::ZeroLayers { stage: i, device: s.device });
            }
            if stages[..i].iter().any(|o| o.device == s.device) {
                return Err(LayerSplitError::DuplicateDevice { device: s.device });
            }
            total = total.checked_add(s.layers).ok_or_else(|| LayerSplitError::Parse {
                input: text(&stages),
                reason: "the layer counts overflow".into(),
            })?;
        }
        Ok(Self { stages })
    }

    /// The stages in layer order.
    pub fn stages(&self) -> &[StageSpec] {
        &self.stages
    }

    /// Device ordinals of the stages in layer order.
    pub fn devices(&self) -> Vec<usize> {
        self.stages.iter().map(|s| s.device).collect()
    }

    /// Global layer range of every stage: stage `s` gets `[Σ_{t<s} layers_t, Σ_{t≤s} layers_t)`, so ranges neither
    /// leave gaps nor overlap. Fails unless they cover exactly `num_layers` layers.
    pub fn ranges(&self, num_layers: usize) -> Result<Vec<Range<usize>>, LayerSplitError> {
        let mut ranges = Vec::with_capacity(self.stages.len());
        let mut start = 0usize;
        for s in &self.stages {
            // `new` checked that the sum fits
            let end = start + s.layers;
            ranges.push(start..end);
            start = end;
        }
        if start != num_layers {
            return Err(LayerSplitError::Coverage { covered: start, layers: num_layers });
        }
        Ok(ranges)
    }

    /// Whether a split runs with `precision` and `kernels` on a backend (`cuda_backend`: a build with the `cuda`
    /// feature, or the CUDA build a Linux package runs): CUDA, f16 activations, fused kernels.
    pub fn check(&self, precision: Precision, kernels: Kernels, cuda_backend: bool) -> Result<(), LayerSplitError> {
        let unsupported = |reason: String| Err(LayerSplitError::Unsupported { reason });
        if !cuda_backend {
            return unsupported(format!("needs the CUDA backend, this build has {}", crate::BUILD_BACKEND));
        }
        if precision != Precision::F16 {
            return unsupported(format!("needs dtype f16, not {}", precision_name(precision)));
        }
        if kernels != Kernels::Fused {
            return unsupported("needs kernels fused, not candle (the padded candle forward is not split)".into());
        }
        Ok(())
    }
}

fn precision_name(p: Precision) -> &'static str {
    match p {
        Precision::Bf16 => "bf16",
        Precision::F16 => "f16",
        Precision::F32 => "f32",
    }
}

/// `DEVICE:LAYERS,...` of `stages`.
fn text(stages: &[StageSpec]) -> String {
    stages.iter().map(|s| format!("{}:{}", s.device, s.layers)).collect::<Vec<_>>().join(",")
}

impl fmt::Display for LayerSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&text(&self.stages))
    }
}

impl TryFrom<Vec<StageSpec>> for LayerSplit {
    type Error = LayerSplitError;

    fn try_from(stages: Vec<StageSpec>) -> Result<Self, Self::Error> {
        Self::new(stages)
    }
}

impl FromStr for LayerSplit {
    type Err = LayerSplitError;

    /// `DEVICE:LAYERS` pairs separated by commas, in layer order (`0:30,1:30`): decimal digits only, no spaces or
    /// signs.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parse = |reason: String| LayerSplitError::Parse { input: s.to_string(), reason };
        if s.is_empty() {
            return Err(parse("empty".into()));
        }
        let number = |v: &str, what: &str| -> Result<usize, LayerSplitError> {
            if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
                return Err(parse(format!("{what} {v:?} is not a non-negative decimal number")));
            }
            v.parse().map_err(|e| parse(format!("{what} {v:?}: {e}")))
        };
        let stages = s
            .split(',')
            .map(|pair| {
                let (device, layers) =
                    pair.split_once(':').ok_or_else(|| parse(format!("{pair:?} is not DEVICE:LAYERS")))?;
                Ok(StageSpec { device: number(device, "device")?, layers: number(layers, "layer count")? })
            })
            .collect::<Result<Vec<_>, LayerSplitError>>()?;
        Self::new(stages)
    }
}
