//! From letter logits to System One answers: per-order softmax, un-permutation, average of probabilities,
//! per-type temperature and TypeSafe confidence. All host math in f64.

use indexmap::IndexMap;
use serde_json::{Number, Value};

use crate::request::{Item, QType};

/// Option orders asked for each question: original and reversed (upstream default `--orders 2`).
pub fn orders(k: usize) -> Vec<Vec<usize>> {
    vec![(0..k).collect(), (0..k).rev().collect()]
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// `p[j] = mean over orders of probs_order[position of option j]`; `perm[k]` is the option shown at position `k`.
pub fn canonical_average(perms: &[Vec<usize>], probs: &[Vec<f64>]) -> Vec<f64> {
    let k = perms[0].len();
    let mut out = vec![0.0; k];
    for (perm, p) in perms.iter().zip(probs) {
        for (pos, &j) in perm.iter().enumerate() {
            out[j] += p[pos];
        }
    }
    out.iter().map(|v| v / perms.len() as f64).collect()
}

/// `softmax(log(max(p, 1e-12)) / T)`, identity for `T == 1` (upstream `Server.decide`).
pub fn calibrate(p: &[f64], t: f64) -> Vec<f64> {
    if t == 1.0 {
        return p.to_vec();
    }
    softmax(&p.iter().map(|v| v.max(1e-12).ln() / t).collect::<Vec<_>>())
}

/// First index of the maximum (Python `max` keeps the first of equal values).
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in p.iter().enumerate() {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

/// TypeSafe Choice confidence `(max(p) - 1/n) / (1 - 1/n)`; `None` for n < 2.
pub fn choice_confidence(p: &[f64]) -> Option<f64> {
    let n = p.len() as f64;
    (p.len() >= 2).then(|| (p[argmax(p)] - 1.0 / n) / (1.0 - 1.0 / n))
}

/// TypeSafe Score confidence `max(0, 1 - sum p[i]|i-m| / uniform_mad)`, m = most probable level (first on ties);
/// `None` for n < 2.
pub fn score_confidence(p: &[f64]) -> Option<f64> {
    let n = p.len();
    if n < 2 {
        return None;
    }
    let m = argmax(p) as f64;
    let mid = (n as f64 - 1.0) / 2.0;
    let uniform_mad = (0..n).map(|i| (i as f64 - mid).abs()).sum::<f64>() / n as f64;
    let spread: f64 = p.iter().enumerate().map(|(i, v)| v * (i as f64 - m).abs()).sum();
    Some((1.0 - spread / uniform_mad).max(0.0))
}

pub fn num(v: f64) -> Value {
    Number::from_f64(v).map(Value::Number).unwrap_or(Value::Null)
}

/// System One answer for one question from its final (averaged, calibrated) distribution.
/// `multi` answer from the calibrated P(yes) of every label (upstream `multi_answer`): labels with P >= threshold by
/// descending P (stable), cut to `max`, at least the top `min`; `set_confidence` = probability of exactly that set.
fn multi_answer(item: &Item, m: &crate::request::MultiSpec, p_yes: &[f64]) -> Value {
    let mut order: Vec<usize> = (0..p_yes.len()).collect();
    order.sort_by(|&a, &b| p_yes[b].partial_cmp(&p_yes[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut sel: Vec<usize> = order.iter().copied().filter(|&i| p_yes[i] >= m.threshold).collect();
    if let Some(mx) = m.max {
        sel.truncate(mx.max(0) as usize);
    }
    if let Some(mn) = m.min {
        if (sel.len() as i64) < mn {
            sel = order.iter().copied().take(mn.max(0) as usize).collect();
        }
    }
    let conf: f64 = (0..p_yes.len()).map(|i| if sel.contains(&i) { p_yes[i] } else { 1.0 - p_yes[i] }).product();
    let probs: serde_json::Map<String, Value> = item.keys.iter().cloned().zip(p_yes.iter().map(|v| num(*v))).collect();
    let mut a = serde_json::Map::new();
    a.insert("type".into(), "multi".into());
    a.insert("selected".into(), Value::Array(sel.iter().map(|&i| Value::String(item.keys[i].clone())).collect()));
    a.insert("probabilities".into(), Value::Object(probs));
    a.insert("threshold".into(), num(m.threshold));
    a.insert("set_confidence".into(), num(conf));
    Value::Object(a)
}

/// `act` answer (upstream `act_answer`): minimum expected cost action (first on ties); with `max_error`, an automatic
/// action below the certified confidence is replaced by the defer action.
fn act_answer(item: &Item, a: &crate::request::ActSpec, p: &[f64]) -> Result<Value, String> {
    let exp: Vec<f64> = a.costs.iter().map(|(_, row)| row.iter().zip(p).map(|(c, q)| q * c).sum()).collect();
    let mut best = 0;
    for (i, e) in exp.iter().enumerate() {
        if *e < exp[best] {
            best = i;
        }
    }
    let conf = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probs: serde_json::Map<String, Value> = item.keys.iter().cloned().zip(p.iter().map(|v| num(*v))).collect();
    let round6 = |v: f64| (v * 1e6).round() / 1e6;
    let mut ans = serde_json::Map::new();
    ans.insert("type".into(), "act".into());
    ans.insert("action".into(), Value::String(a.costs[best].0.clone()));
    ans.insert(
        "expected_costs".into(),
        Value::Object(a.costs.iter().zip(&exp).map(|((n, _), e)| (n.clone(), num(round6(*e)))).collect()),
    );
    ans.insert("answer".into(), Value::String(item.keys[argmax(p)].clone()));
    ans.insert("probabilities".into(), Value::Object(probs));
    ans.insert("confidence".into(), num(conf));
    if a.max_error.is_some() {
        match a.threshold {
            None => {
                ans.insert("max_error".into(), "no certified threshold for this target".into());
            }
            Some(thr) if conf < thr && Some(&a.costs[best].0) != a.defer.as_ref() => {
                let Some(d) = &a.defer else {
                    return Err("max_error needs a defer action (\"defer\"/\"human\" or defer_action)".into());
                };
                ans.insert("action".into(), Value::String(d.clone()));
                ans.insert("refused".into(), Value::String(a.costs[best].0.clone()));
            }
            Some(_) => {}
        }
    }
    let validated = a.calibrated && (a.base == QType::Noul || a.hide);
    ans.insert("calibration".into(), if validated { "validated" } else { "unvalidated" }.into());
    Ok(Value::Object(ans))
}

/// Answer object of one question; `Err` for a request that cannot be answered (an `act` refusal without a defer
/// action, as upstream).
pub fn answer_checked(item: &Item, p: &[f64], dialect: crate::request::Dialect) -> Result<Value, String> {
    match &item.ext {
        crate::request::Ext::Multi(m) => Ok(multi_answer(item, m, p)),
        crate::request::Ext::Act(a) => act_answer(item, a, p),
        crate::request::Ext::None if dialect == crate::request::Dialect::Upstream => Ok(answer_upstream(item, p)),
        crate::request::Ext::None => Ok(answer(item, p)),
    }
}

/// Upstream `Server.decide` answer objects: probabilities for every type, `confidence = max(p)`, choice = first key
/// of maximal probability, score legend = shown option texts.
pub fn answer_upstream(item: &Item, p: &[f64]) -> Value {
    let probs: serde_json::Map<String, Value> = item.keys.iter().cloned().zip(p.iter().map(|v| num(*v))).collect();
    let conf = num(p.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let mut a = serde_json::Map::new();
    match item.kind {
        QType::Noul => {
            a.insert("type".into(), "noul".into());
            a.insert("noul".into(), num(p[0]));
        }
        QType::Score => {
            a.insert("type".into(), "score".into());
            a.insert("score".into(), num(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()));
            let legend = item.keys.iter().cloned().zip(item.options.iter().cloned().map(Value::String)).collect();
            a.insert("legend".into(), Value::Object(legend));
        }
        _ => {
            a.insert("type".into(), "choice".into());
            a.insert("choice".into(), Value::String(item.keys[argmax(p)].clone()));
        }
    }
    a.insert("probabilities".into(), Value::Object(probs));
    a.insert("confidence".into(), conf);
    Value::Object(a)
}

pub fn answer(item: &Item, p: &[f64]) -> Value {
    let probs: serde_json::Map<String, Value> = item.keys.iter().cloned().zip(p.iter().map(|v| num(*v))).collect();
    let mut a: IndexMap<&str, Value> = IndexMap::new();
    a.insert("type", Value::String(item.kind.as_str().into()));
    match item.kind {
        QType::Noul => {
            a.insert("noul", num(p[0]));
        }
        QType::Choice => {
            a.insert("choice", Value::String(item.keys[argmax(p)].clone()));
            a.insert("probabilities", Value::Object(probs));
            a.insert("confidence", choice_confidence(p).map(num).unwrap_or(Value::Null));
        }
        QType::Multi | QType::Act => unreachable!("multi and act answers come from answer_checked"),
        QType::Score => {
            a.insert("score", num(p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()));
            let legend = item.keys.iter().cloned().zip(item.legend.iter().cloned()).collect();
            a.insert("legend", Value::Object(legend));
            a.insert("probabilities", Value::Object(probs));
            a.insert("confidence", score_confidence(p).map(num).unwrap_or(Value::Null));
        }
    }
    Value::Object(a.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
