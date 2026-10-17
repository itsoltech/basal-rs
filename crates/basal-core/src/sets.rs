//! Port of upstream v2 `basal/sets.py`: independent label sets and their constraints.

use serde::Serialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Objective {
    #[default]
    Threshold,
    Exact,
    Iou,
    F1,
}

impl Objective {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "threshold" => Some(Self::Threshold),
            "exact" => Some(Self::Exact),
            "iou" => Some(Self::Iou),
            "f1" => Some(Self::F1),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Threshold => "threshold",
            Self::Exact => "exact",
            Self::Iou => "iou",
            Self::F1 => "f1",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Rules {
    pub objective: Objective,
    pub exclusive: Vec<usize>,
    pub fallback: Option<usize>,
    pub fallback_threshold: Option<f64>,
    pub fallback_threshold_source: Option<&'static str>,
}

impl Rules {
    pub fn extended(&self) -> bool {
        self.objective != Objective::Threshold || !self.exclusive.is_empty() || self.fallback.is_some()
    }
}

/// Stable upstream calibration key: compact UTF-8 JSON of sorted (key, label) pairs and the fallback key.
pub fn label_set_id(keys: &[String], labels: &[String], fallback: &str) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut pairs: Vec<_> = keys.iter().zip(labels).collect();
    pairs.sort_unstable();
    let bytes = serde_json::to_vec(&(pairs, fallback))?;
    Ok(format!("{:x}", Sha256::digest(bytes))[..16].to_string())
}

/// Gauss-Legendre nodes and weights on [0, 1]; enough nodes integrate the score polynomials exactly up to rounding.
fn nodes(m: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (m as f64 + 0.5)).cos();
        let mut derivative = 0.0;
        for _ in 0..64 {
            let (mut p0, mut p1) = (1.0, z);
            for j in 2..=m {
                let p = ((2 * j - 1) as f64 * z * p1 - (j - 1) as f64 * p0) / j as f64;
                p0 = p1;
                p1 = p;
            }
            derivative = m as f64 * (z * p1 - p0) / (z * z - 1.0);
            let next = z - p1 / derivative;
            let done = (next - z).abs() <= 4.0 * f64::EPSILON;
            z = next;
            if done {
                break;
            }
        }
        out.push(((z + 1.0) / 2.0, 1.0 / ((1.0 - z * z) * derivative * derivative)));
    }
    out
}

/// Expected score of selecting each prefix. Callers validate at most 256 labels for IoU/F1.
pub fn expected_scores(p: &[f64], objective: Objective) -> Vec<f64> {
    let n = p.len();
    let mut out = vec![0.0; n + 1];
    out[0] = p.iter().map(|v| 1.0 - v).product();
    if objective == Objective::Exact {
        let mut prefix = vec![1.0; n + 1];
        let mut suffix = vec![1.0; n + 1];
        for i in 0..n {
            prefix[i + 1] = prefix[i] * p[i];
        }
        for i in (0..n).rev() {
            suffix[i] = suffix[i + 1] * (1.0 - p[i]);
        }
        return prefix.iter().zip(suffix).map(|(a, b)| a * b).collect();
    }
    if n == 0 {
        return out;
    }
    let mut m = 8;
    let degree = if objective == Objective::Iou { n - 1 } else { 2 * n - 1 };
    while 2 * m - 1 < degree {
        m *= 2;
    }
    let sum_p: Vec<f64> = p
        .iter()
        .scan(0.0, |sum, v| {
            *sum += v;
            Some(*sum)
        })
        .collect();
    for (t, weight) in nodes(m) {
        let factors: Vec<f64> = p.iter().map(|v| 1.0 + v * (t - 1.0)).collect();
        if objective == Objective::Iou {
            let mut suffix = vec![1.0; n + 1];
            for i in (0..n).rev() {
                suffix[i] = suffix[i + 1] * factors[i];
            }
            let mut power = 1.0;
            for k in 1..=n {
                out[k] += weight * power * suffix[k] * sum_p[k - 1];
                power *= t;
            }
        } else {
            let product: f64 = factors.iter().product();
            let (mut sum, mut power) = (0.0, t);
            for k in 1..=n {
                sum += p[k - 1] / factors[k - 1] * product;
                out[k] += 2.0 * weight * power * sum;
                power *= t;
            }
        }
    }
    for v in &mut out[1..] {
        *v = v.min(1.0);
    }
    out
}

pub(crate) struct Selection {
    pub indices: Vec<usize>,
    pub score: Option<f64>,
    pub constraints: Option<Value>,
}

/// Select labels under upstream objectives, keeping input order on probability ties.
pub(crate) fn select(keys: &[String], p: &[f64], spec: &crate::request::MultiSpec) -> Selection {
    let rules = &spec.rules;
    let mut excluded = rules.exclusive.clone();
    if let Some(i) = rules.fallback {
        if !excluded.contains(&i) {
            excluded.push(i);
        }
    }
    let alone: Vec<usize> = excluded.iter().copied().filter(|i| Some(*i) != rules.fallback).collect();
    let mut order: Vec<usize> = (0..p.len()).filter(|i| !excluded.contains(i)).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    let one_ok = spec.min.unwrap_or(0) <= 1 && spec.max.is_none_or(|n| n >= 1);
    let mut chosen = "labels";
    let (mut indices, mut score) = if rules.objective == Objective::Threshold {
        let mut indices: Vec<usize> = order.iter().copied().filter(|&i| p[i] >= spec.threshold).collect();
        if let Some(max) = spec.max {
            let length = if max < 0 { indices.len().saturating_sub(max.unsigned_abs() as usize) } else { max as usize };
            indices.truncate(length);
        }
        if let Some(min) = spec.min {
            if (indices.len() as i64) < min {
                indices = order.iter().copied().take(min.max(0) as usize).collect();
            }
        }
        (indices, None)
    } else {
        let lo = (spec.min.unwrap_or(0).max(0) as usize).min(order.len());
        let hi = spec.max.map_or(order.len(), |n| n.max(0) as usize).min(order.len());
        let all: Vec<f64> = order.iter().chain(&excluded).map(|&i| p[i]).collect();
        let scores = expected_scores(&all, rules.objective);
        let mut best = lo;
        if rules.objective == Objective::Exact {
            best = order.iter().filter(|&&i| p[i] >= 0.5).count().clamp(lo.min(hi), hi);
        } else {
            for k in lo..=hi {
                if scores[k] > scores[best] {
                    best = k;
                }
            }
        }
        (order[..best].to_vec(), Some(scores[best]))
    };
    let solo_score = |i| {
        let probabilities: Vec<_> =
            std::iter::once(p[i]).chain(p.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, v)| *v)).collect();
        expected_scores(&probabilities, rules.objective)[1]
    };
    if one_ok {
        if rules.objective == Objective::Threshold {
            let mut best = None;
            for &i in &alone {
                if p[i] >= spec.threshold && best.is_none_or(|j| p[i] > p[j]) {
                    best = Some(i);
                }
            }
            if let Some(i) = best {
                if indices.is_empty() || indices.iter().all(|&j| p[i] > p[j]) {
                    indices = vec![i];
                    chosen = "exclusive";
                }
            }
        } else {
            for &i in &alone {
                let s = solo_score(i);
                if s > score.unwrap_or(0.0) {
                    indices = vec![i];
                    score = Some(s);
                    chosen = "exclusive";
                }
            }
        }
        if let Some(i) = rules.fallback {
            if indices.is_empty() {
                let take = if let Some(t) = rules.fallback_threshold {
                    p[i] >= t
                } else if rules.objective == Objective::Threshold {
                    p[i] >= spec.threshold
                } else {
                    solo_score(i) > score.unwrap_or(0.0)
                };
                if take {
                    indices = vec![i];
                    if score.is_some() {
                        score = Some(solo_score(i));
                    }
                    chosen = "fallback";
                }
            }
        }
    }
    let constraints = (!excluded.is_empty()).then(|| {
        let mut value = json!({"exclusive": alone.iter().map(|&i| &keys[i]).collect::<Vec<_>>(), "fallback": rules.fallback.map(|i| &keys[i]), "chosen": if indices.is_empty() { "empty" } else { chosen }});
        if rules.fallback.is_some() {
            value["fallback_threshold"] = rules.fallback_threshold.or_else(|| (rules.objective == Objective::Threshold).then_some(spec.threshold)).map_or(Value::Null, crate::decision::num);
            value["fallback_threshold_source"] = rules.fallback_threshold_source.unwrap_or("default").into();
        }
        value
    });
    Selection { indices, score, constraints }
}
