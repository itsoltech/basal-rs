//! Validated per-model lane controls, independent of packing/admission and numeric profiles.

use anyhow::{ensure, Result};
use basal_core::{engine::RequestPlan, Backend, Engine};

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LongLane {
    #[default]
    Legacy,
    On,
    Off,
}

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LongTokenBasis {
    #[default]
    Packed,
    Prompt,
}

/// Requested controls stay readable; all worker/gate/routing decisions use this checked snapshot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoutingPolicy {
    requested_lane: Option<LongLane>,
    requested_basis: Option<LongTokenBasis>,
    requested_tokens: usize,
    threshold: Option<usize>,
    basis: LongTokenBasis,
}

impl RoutingPolicy {
    pub(crate) fn new(tokens: usize, lane: Option<LongLane>, basis: Option<LongTokenBasis>) -> Result<Self> {
        ensure!(
            !matches!(lane, Some(LongLane::On)) || tokens > 0,
            "long_lane=on (--long-lane on) requires positive long_tokens (--long-tokens); requested threshold={tokens}, basis={basis:?}"
        );
        let enabled = !matches!(lane, Some(LongLane::Off)) && tokens > 0;
        Ok(Self {
            requested_lane: lane,
            requested_basis: basis,
            requested_tokens: tokens,
            threshold: enabled.then_some(tokens),
            basis: basis.unwrap_or_default(),
        })
    }

    pub(crate) fn enabled(self) -> bool {
        self.threshold.is_some()
    }

    pub(crate) fn workers(self, background: bool) -> u64 {
        1 + u64::from(self.enabled()) + u64::from(background)
    }

    pub(crate) fn routes_long<B: Backend>(self, engine: &Engine<B>, plan: &RequestPlan, packed: usize) -> bool {
        let Some(threshold) = self.threshold else { return false };
        match self.basis {
            LongTokenBasis::Packed => packed > threshold,
            // An unproved bound must never classify potentially long dependent work as short.
            LongTokenBasis::Prompt => plan.prompt_token_bound(&engine.shape_limits()).is_none_or(|n| n > threshold),
        }
    }

    pub(crate) fn identity(self) -> serde_json::Value {
        serde_json::json!({
            "requested": {"long_lane": self.requested_lane, "long_tokens": self.requested_tokens,
                "long_token_basis": self.requested_basis},
            "effective": {"long_lane": self.enabled(), "long_tokens": self.threshold, "long_token_basis": self.basis}
        })
    }
}
