//! Explicit native short-first backlog. This policy never changes engine/worker or numeric identities.

use anyhow::{ensure, Result};

use super::{routing::RoutingPolicy, Schedule};

/// Independent prompt allowance for long entries, matching the pinned upstream constant.
pub(super) const LONG_CALL_TOKENS: usize = 8192;

#[derive(Clone, Copy, Debug, clap::ValueEnum, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Scheduler {
    Auto,
    On,
    Off,
}

/// Resolve only after CLI/YAML overrides. Private fields prevent unchecked effective modes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SchedulerPolicy {
    requested: Option<Scheduler>,
    schedule: Schedule,
    threshold: Option<usize>,
}

impl SchedulerPolicy {
    pub(crate) fn new(
        requested: Option<Scheduler>,
        schedule: Schedule,
        routing: RoutingPolicy,
        tokens: usize,
    ) -> Result<Self> {
        ensure!(
            !matches!(requested, Some(Scheduler::On)) || !routing.enabled(),
            "scheduler=on (--scheduler on) requires a disabled effective long lane; use --long-lane off"
        );
        let backlog = matches!(requested, Some(Scheduler::On | Scheduler::Auto)) && !routing.enabled();
        ensure!(
            !backlog || tokens > 0,
            "short-first scheduler requires positive long_tokens (--long-tokens); requested threshold={tokens}"
        );
        Ok(Self { requested, schedule, threshold: backlog.then_some(tokens) })
    }

    pub(super) fn threshold(self) -> Option<usize> {
        self.threshold
    }

    pub(super) fn schedule(self) -> Schedule {
        self.schedule
    }

    /// Omission adds no benchmark keys; explicit auto/off still retain the effective native order.
    pub(crate) fn identity(self) -> Option<serde_json::Value> {
        self.requested.map(|requested| {
            serde_json::json!({
                "requested": requested,
                "effective": {
                    "mode": if self.threshold.is_some() { "short-first" } else { "native" },
                    "native_schedule": format!("{:?}", self.schedule).to_lowercase(),
                    "long_tokens": self.threshold,
                    "long_call_tokens": self.threshold.map(|_| LONG_CALL_TOKENS)
                }
            })
        })
    }
}
