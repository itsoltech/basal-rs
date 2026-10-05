//! Evidence spans (basal-1.5 `"evidence": true`): port of upstream `basal/evidence.py`. A pointer head over the
//! state's tokens (in the backend) gives start / end log-probabilities from the final-layer states of the question's
//! prompt in the original option order; this module builds that prompt, maps tokens to character offsets of the state
//! and picks the top non-overlapping spans exactly as `best_spans`. Offsets and texts are in characters (Python
//! string indices), so a span quotes the state verbatim.

use anyhow::Result;
use serde_json::{json, Value};

use crate::prompt::render;
use crate::request::Item;
use crate::tokenize::BasalTokenizer;

/// Longest span in tokens (`MAX_SPAN`).
pub const MAX_SPAN: usize = 80;
/// Spans returned per question.
pub const TOP: usize = 3;

/// Head input of one question: prompt token ids, the token range `t0..t1` that covers the state, and each of those
/// tokens' character offsets relative to the state start.
pub struct EvidenceInput {
    pub ids: Vec<u32>,
    pub t0: usize,
    pub t1: usize,
    pub rel: Vec<(usize, usize)>,
}

/// `Evidence.spans` up to the head call; `None` when no token lies inside the state (upstream returns no spans).
pub fn input(tok: &BasalTokenizer, bos: &str, item: &Item) -> Result<Option<EvidenceInput>> {
    let opts: Vec<&str> = item.options.iter().map(String::as_str).collect();
    let prompt = render(bos, &item.state, &item.question, &opts, item.lang);
    let needle = format!("\n{}", item.state);
    let Some(byte) = prompt.find(&needle) else { return Ok(None) };
    let s0 = prompt[..byte].chars().count() + 1;
    let s1 = s0 + item.state.chars().count();
    let (ids, offsets) = tok.encode_char_offsets(&prompt)?;
    let idx: Vec<usize> =
        offsets.iter().enumerate().filter(|(_, &(a, b))| a >= s0 && b <= s1 && b > a).map(|(i, _)| i).collect();
    let (Some(&t0), Some(&last)) = (idx.first(), idx.last()) else { return Ok(None) };
    let t1 = last + 1;
    let rel = offsets[t0..t1].iter().map(|&(a, b)| (a.saturating_sub(s0), b.saturating_sub(s0))).collect();
    Ok(Some(EvidenceInput { ids, t0, t1, rel }))
}

/// `best_spans`: scores `ls[i] + le[j]` (f32) for `i <= j < i + MAX_SPAN` (other pairs -1e9), the `4 * top` best of
/// all `n * n` pairs (ties: lower flat index first), greedily kept when not overlapping a kept span.
pub fn best_spans(ls: &[f32], le: &[f32], rel: &[(usize, usize)], state: &[char], top: usize) -> Vec<Value> {
    let n = ls.len();
    let want = (top * 4).min(n * n);
    let mut cand: Vec<(f32, usize)> = Vec::new();
    for (i, &s) in ls.iter().enumerate() {
        for (j, &e) in le.iter().enumerate().take((i + MAX_SPAN).min(n)).skip(i) {
            cand.push((s + e, i * n + j));
        }
    }
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    if cand.len() < want {
        // masked pairs (score -1e9), in flat index order
        let valid: std::collections::HashSet<usize> = cand.iter().map(|c| c.1).collect();
        cand.extend((0..n * n).filter(|k| !valid.contains(k)).map(|k| (-1e9f32, k)));
    }
    cand.truncate(want);
    let mut out: Vec<(usize, usize, f64)> = Vec::new();
    for (v, k) in cand {
        let (i, j) = (k / n, k % n);
        let (a, b) = (rel[i].0, rel[j].1);
        if out.iter().any(|&(s, e, _)| !(b <= s || a >= e)) {
            continue;
        }
        out.push((a, b, (v as f64).exp()));
        if out.len() == top {
            break;
        }
    }
    out.into_iter()
        .map(|(a, b, p)| {
            let text: String = slice(state, a, b);
            json!({"text": text, "start": a, "end": b, "probability": p})
        })
        .collect()
}

/// Python `state[a:b]` on characters (out-of-range bounds clamp, an empty range gives "").
fn slice(state: &[char], a: usize, b: usize) -> String {
    let (a, b) = (a.min(state.len()), b.min(state.len()));
    if a >= b {
        String::new()
    } else {
        state[a..b].iter().collect()
    }
}

/// Final spans: `best_spans` with `top + 2` candidates, those starting at or after `limit` (the appended facts block)
/// dropped and the rest clipped to it, cut to `top`.
pub fn spans(ls: &[f32], le: &[f32], inp: &EvidenceInput, state: &str, limit: Option<usize>) -> Value {
    let chars: Vec<char> = state.chars().collect();
    let mut out = best_spans(ls, le, &inp.rel, &chars, TOP + 2);
    if let Some(l) = limit {
        out.retain(|x| x["start"].as_u64().is_some_and(|s| (s as usize) < l));
        for x in out.iter_mut() {
            let s = x["start"].as_u64().unwrap() as usize;
            let e = (x["end"].as_u64().unwrap() as usize).min(l);
            x["end"] = json!(e);
            x["text"] = json!(slice(&chars, s, e));
        }
    }
    out.truncate(TOP);
    Value::Array(out)
}
