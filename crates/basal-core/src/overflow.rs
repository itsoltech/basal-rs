//! CPU context-overflow policy. Refusal is the default, including for media.

/// What to do when a text prompt exceeds the model's position limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContextOverflow {
    #[default]
    Refuse,
    Truncate,
}

impl ContextOverflow {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Refuse => "refuse",
            Self::Truncate => "truncate",
        }
    }
}

/// Validated settings: the head share is always finite and in `[0, 1]`.
#[derive(Clone, Copy, Debug)]
pub struct OverflowSettings {
    policy: ContextOverflow,
    head_share: f64,
}

#[derive(Debug, thiserror::Error)]
#[error("truncate head must be finite and in [0, 1]")]
pub struct InvalidHeadShare;

impl Default for OverflowSettings {
    fn default() -> Self {
        Self { policy: ContextOverflow::Refuse, head_share: 1.0 / 3.0 }
    }
}

impl OverflowSettings {
    pub fn new(policy: ContextOverflow, head_share: f64) -> Result<Self, InvalidHeadShare> {
        if !head_share.is_finite() || !(0.0..=1.0).contains(&head_share) {
            return Err(InvalidHeadShare);
        }
        Ok(Self { policy, head_share })
    }

    pub fn policy(self) -> ContextOverflow {
        self.policy
    }

    pub fn head_share(self) -> f64 {
        self.head_share
    }
}

/// Tokens selected from the original (facts-augmented, if requested) state. The marker is not counted as state.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Truncation {
    pub state_tokens: usize,
    pub head_tokens: usize,
    pub tail_tokens: usize,
    pub dropped_tokens: usize,
    pub head_share: f64,
}
