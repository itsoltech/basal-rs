//! Bounded local packing after the database has granted cost credit to one owner and batch.
//!
//! This never changes global DRR credit or mixes claims. Unselected records keep their claim and host permits;
//! the executor renews them while waiting. Composition is an explicit qualification input, never inferred from
//! `forward_max_tokens` or a GEMM table. Media and persisted poison records always run alone.

use basal_core::pack::TrieSize;
use basal_core::planner::plan_cost;

use super::bg_lane::PlannedItem;

pub(super) struct Ready {
    pub item: PlannedItem,
    skipped: u8,
}

impl Ready {
    pub fn new(item: PlannedItem) -> Self {
        Self { item, skipped: 0 }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Packing {
    pub composition_qualified: bool,
    pub max_requests: usize,
    pub max_tokens: usize,
    /// Limit of one engine forward; one admitted record may require several forwards.
    pub forward_max_tokens: usize,
}

pub(super) fn take(ready: &mut Vec<Ready>, packing: Packing) -> Vec<PlannedItem> {
    if ready.is_empty() {
        return Vec::new();
    }
    // An expensive record skipped eight times becomes the first record regardless of the fit of newer work.
    let first = ready.iter().position(|record| record.skipped >= 8).unwrap_or(0);
    let first_cost = plan_cost(&ready[first].item.plan);
    // Total packed tokens are a scheduling estimate, not a per-forward bound. Single/Budget/Tree engines may
    // split one valid record into several forwards. Validation and the worker use DeviceWorkShape for that bound.
    let first = ready.remove(first).item;
    let solo = first.record.requires_solo || first_cost.media || !packing.composition_qualified;
    let mut trie = TrieSize::default();
    trie.insert(&crate::serve::plan_prompts(&first.plan));
    let mut selected = vec![first];
    let limit = packing.max_tokens.min(packing.forward_max_tokens);
    while !solo && selected.len() < packing.max_requests.max(1) {
        let candidate = ready
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| {
                let cost = plan_cost(&entry.item.plan);
                if entry.item.record.requires_solo || cost.media || cost.packed_tokens > packing.forward_max_tokens {
                    return None;
                }
                let growth = trie.growth(&crate::serve::plan_prompts(&entry.item.plan));
                (trie.tokens.saturating_add(growth) <= limit).then_some((
                    index,
                    growth,
                    first_cost.longest_prompt.abs_diff(cost.longest_prompt),
                ))
            })
            .min_by_key(|&(index, growth, length_difference)| (length_difference, growth, index));
        let Some((index, _, _)) = candidate else { break };
        let item = ready.remove(index).item;
        trie.insert(&crate::serve::plan_prompts(&item.plan));
        selected.push(item);
    }
    for record in ready {
        record.skipped = record.skipped.saturating_add(1).min(8);
    }
    selected
}
