//! Small statistics used by the parity and bench reports.

use serde_json::{json, Value};

pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((q * sorted.len() as f64) as usize).min(sorted.len() - 1)]
}

pub fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// max / p50 / p95 / p99 / mean of absolute values.
pub fn summary(xs: &[f64]) -> Value {
    let mut s: Vec<f64> = xs.iter().map(|v| v.abs()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if s.is_empty() { f64::NAN } else { s.iter().sum::<f64>() / s.len() as f64 };
    json!({"n": s.len(), "max": s.last().copied().unwrap_or(f64::NAN), "p50": median(&s),
           "p95": quantile(&s, 0.95), "p99": quantile(&s, 0.99), "mean": mean})
}

pub fn tv(p: &[f64], q: &[f64]) -> f64 {
    0.5 * p.iter().zip(q).map(|(a, b)| (a - b).abs()).sum::<f64>()
}

/// Quality of distributions against gold indices: accuracy, NLL, multi-class Brier and top-label ECE (10 bins).
pub fn quality(dists: &[Vec<f64>], gold: &[usize]) -> Value {
    let n = dists.len() as f64;
    let mut acc = 0.0;
    let mut nll = 0.0;
    let mut brier = 0.0;
    let mut bins = vec![(0.0f64, 0.0f64, 0usize); 10];
    for (p, &g) in dists.iter().zip(gold) {
        let top = basal_core::decision::argmax(p);
        let hit = (top == g) as u8 as f64;
        acc += hit;
        nll -= p[g].max(1e-12).ln();
        brier += p.iter().enumerate().map(|(i, v)| (v - (i == g) as u8 as f64).powi(2)).sum::<f64>();
        let b = ((p[top] * 10.0) as usize).min(9);
        bins[b].0 += p[top];
        bins[b].1 += hit;
        bins[b].2 += 1;
    }
    let ece: f64 = bins.iter().filter(|b| b.2 > 0).map(|b| (b.0 - b.1).abs() / n).sum();
    json!({"n": dists.len(), "correct": acc as usize, "accuracy": acc / n, "nll": nll / n, "brier": brier / n, "ece_10bin": ece})
}
