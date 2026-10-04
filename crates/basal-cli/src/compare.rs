//! `basal compare`: two exports (Python reference or `basal export`) side by side.
//! Exact: prompt text, token ids, letter ids, permutations and packing. Numeric: letter logits, per-order
//! distributions, order average and calibrated distribution, total variation. Decisions: argmax agreement and every
//! flip. Quality against gold labels. System One: errors and answers of both sides.
//! `basal check-prompts` runs only the exact part against this runtime's preparation, without a model.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use basal_core::decision::argmax;
use basal_core::engine::Prepared;
use basal_core::{Backend, DecideError, Engine};
use serde_json::{json, Value};

use crate::reference::{bench_item, f64s, read_jsonl, str_of};
use crate::stats::{quality, summary, tv};

/// Intended differences of this runtime from upstream `to_items` (docs/SYSTEM_ONE.md). Token mismatches of these items
/// are reported separately and excluded from numeric parity.
pub fn known_deviation(case: &str, item: &str) -> Option<&'static str> {
    match (case, item) {
        ("so-null-and-missing-instructions", "explicit_null") => {
            Some("instructions=null is treated as missing (empty); upstream renders the text \"null\"")
        }
        ("so-empty-structures", "empty_true") => {
            Some("noul description \"\"/{} is kept; upstream replaces falsy descriptions with Tak/Nie")
        }
        ("so-unknown-type", _) => Some("unknown type is a validation error; upstream treats it as choice"),
        ("so-missing-type", _) => Some("missing type is a validation error; upstream defaults to choice"),
        _ => None,
    }
}

#[derive(Default)]
struct TokenCheck {
    orders: usize,
    mismatches: Vec<Value>,
    deviations: Vec<Value>,
}

impl TokenCheck {
    fn to_json(&self) -> Value {
        json!({"orders_compared": self.orders, "mismatches": self.mismatches, "known_deviations": self.deviations})
    }

    /// Records with `orders[{perm, prompt, input_ids, letter_ids}]` and `pack{prefix_len, packed_len}`.
    fn compare(&mut self, tag: &str, a: &Value, b: &Value, deviation: Option<&str>) -> bool {
        let mut diffs = Vec::new();
        let (ao, bo) =
            (a["orders"].as_array().cloned().unwrap_or_default(), b["orders"].as_array().cloned().unwrap_or_default());
        if ao.len() != bo.len() {
            diffs.push(format!("orders {} vs {}", ao.len(), bo.len()));
        }
        for (k, (x, y)) in ao.iter().zip(&bo).enumerate() {
            self.orders += 1;
            for field in ["prompt", "input_ids", "letter_ids", "perm"] {
                if x[field] != y[field] {
                    diffs.push(format!("order {k}: {field}"));
                }
            }
        }
        for field in ["prefix_len", "packed_len", "readout_cols"] {
            if a["pack"][field] != b["pack"][field] {
                diffs.push(format!("pack.{field}"));
            }
        }
        if diffs.is_empty() {
            return true;
        }
        let rec = json!({"item": tag, "differences": diffs, "reason": deviation});
        if deviation.is_some() {
            self.deviations.push(rec);
        } else {
            self.mismatches.push(rec);
        }
        false
    }
}

fn prepared_json(p: &Prepared) -> Value {
    json!({
        "name": p.item.name,
        "orders": p.orders.iter().map(|o| json!({"perm": o.perm, "prompt": o.prompt, "input_ids": o.input_ids, "letter_ids": o.letter_ids})).collect::<Vec<_>>(),
        "pack": {"prefix_len": p.packed.prefix_len, "packed_len": p.packed.ids.len(), "readout_cols": p.packed.last},
    })
}

fn error_of(case: &Value) -> Option<Value> {
    case.get("upstream_error").or_else(|| case.get("error")).filter(|v| !v.is_null()).cloned()
}

fn answer_of(case: &Value) -> Option<&Value> {
    case.get("upstream_answer").or_else(|| case.get("answer")).filter(|v| !v.is_null())
}

fn item_named<'a>(case: &'a Value, name: &str) -> Option<&'a Value> {
    case["items"].as_array().and_then(|xs| xs.iter().find(|x| str_of(x, "name") == name))
}

/// Exact checks of this runtime's preparation against a reference, without a model.
pub fn check_prompts<B: Backend>(engine: &Engine<B>, reference: &Path) -> Result<Value> {
    let mut tc = TokenCheck::default();
    for rec in read_jsonl(&reference.join("bench.jsonl"))? {
        let p = engine.prepare(bench_item(&rec)?)?;
        tc.compare(&format!("bench/{}", rec["index"]), &rec, &prepared_json(&p), None);
    }
    let mut cases = Vec::new();
    for case in read_jsonl(&reference.join("systemone.jsonl"))? {
        let id = str_of(&case, "id");
        let mut items = Vec::new();
        let ours_error = match engine.prepare_request(&case["request"]) {
            Ok(prep) => {
                for p in &prep {
                    let same = match item_named(&case, &p.item.name) {
                        Some(r) => tc.compare(
                            &format!("systemone/{id}/{}", p.item.name),
                            r,
                            &prepared_json(p),
                            known_deviation(id, &p.item.name),
                        ),
                        None => false,
                    };
                    items.push(json!({"item": p.item.name, "tokens_identical": same}));
                }
                None
            }
            Err(DecideError::Internal(e)) => return Err(e),
            Err(e) => Some(e.to_json()),
        };
        cases.push(json!({"id": id, "ours_error": ours_error, "reference_error": error_of(&case),
                          "deviation": known_deviation(id, ""), "items": items}));
    }
    Ok(json!({"tokens": tc.to_json(), "systemone_cases": cases}))
}

#[derive(Default)]
struct Numeric {
    logit: Vec<f64>,
    order_prob: Vec<f64>,
    avg: Vec<f64>,
    cal: Vec<f64>,
    tv_avg: Vec<f64>,
    tv_cal: Vec<f64>,
    flips_avg: Vec<Value>,
    flips_cal: Vec<Value>,
    items: usize,
}

impl Numeric {
    fn add(&mut self, tag: &str, a: &Value, b: &Value) {
        self.items += 1;
        for (x, y) in a["orders"].as_array().into_iter().flatten().zip(b["orders"].as_array().into_iter().flatten()) {
            self.logit.extend(f64s(&x["letter_logits"]).iter().zip(f64s(&y["letter_logits"])).map(|(p, q)| q - p));
            self.order_prob.extend(f64s(&x["probs"]).iter().zip(f64s(&y["probs"])).map(|(p, q)| q - p));
        }
        let (aa, ba, ac, bc) = (f64s(&a["p_avg"]), f64s(&b["p_avg"]), f64s(&a["p_cal"]), f64s(&b["p_cal"]));
        self.avg.extend(aa.iter().zip(&ba).map(|(p, q)| q - p));
        self.cal.extend(ac.iter().zip(&bc).map(|(p, q)| q - p));
        self.tv_avg.push(tv(&aa, &ba));
        self.tv_cal.push(tv(&ac, &bc));
        if argmax(&aa) != argmax(&ba) {
            self.flips_avg.push(
                json!({"item": tag, "a": argmax(&aa), "b": argmax(&ba), "gold": a.get("gold"), "a_p": aa, "b_p": ba}),
            );
        }
        if argmax(&ac) != argmax(&bc) {
            self.flips_cal.push(
                json!({"item": tag, "a": argmax(&ac), "b": argmax(&bc), "gold": a.get("gold"), "a_p": ac, "b_p": bc}),
            );
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "items": self.items,
            "letter_logit_diff": summary(&self.logit), "order_prob_diff": summary(&self.order_prob),
            "avg_prob_diff": summary(&self.avg), "cal_prob_diff": summary(&self.cal),
            "tv_avg": summary(&self.tv_avg), "tv_cal": summary(&self.tv_cal),
            "argmax_agreement_avg": format!("{}/{}", self.items - self.flips_avg.len(), self.items),
            "argmax_agreement_cal": format!("{}/{}", self.items - self.flips_cal.len(), self.items),
            "flips_avg": self.flips_avg, "flips_cal": self.flips_cal,
        })
    }
}

/// Differences of two System One answers for the same question (probabilities, value, choice). The confidence fields
/// are listed but not differenced: upstream uses max(p), this runtime the TypeSafe definitions.
fn answer_diff(a: &Value, b: &Value) -> Value {
    let mut out = json!({"type_a": a["type"], "type_b": b["type"]});
    let max_prob = a["probabilities"].as_object().map(|pa| {
        pa.iter()
            .filter_map(|(k, v)| Some((v.as_f64()? - b["probabilities"].get(k)?.as_f64()?).abs()))
            .fold(0.0, f64::max)
    });
    out["max_abs_prob_diff"] = json!(max_prob);
    for f in ["noul", "score"] {
        if let (Some(x), Some(y)) = (a[f].as_f64(), b[f].as_f64()) {
            out[format!("{f}_diff")] = json!(y - x);
        }
    }
    if a["choice"].is_string() || b["choice"].is_string() {
        out["choice_a"] = a["choice"].clone();
        out["choice_b"] = b["choice"].clone();
        out["choice_same"] = json!(a["choice"] == b["choice"]);
    }
    out["confidence_a"] = a["confidence"].clone();
    out["confidence_b"] = b["confidence"].clone();
    if let (Some(x), Some(y)) = (a["confidence"].as_f64(), b["confidence"].as_f64()) {
        out["confidence_diff"] = json!(y - x);
    }
    let fields = |v: &Value| v.as_object().map(|o| o.keys().cloned().collect::<std::collections::BTreeSet<_>>());
    out["fields_same"] = json!(fields(a) == fields(b));
    if !a["legend"].is_null() || !b["legend"].is_null() {
        out["legend_same"] = json!(a["legend"] == b["legend"]);
    }
    out["type_same"] = json!(a["type"] == b["type"]);
    // multi / act (basal-1.5 extensions)
    for f in ["selected", "action", "refused", "answer", "calibration", "max_error", "threshold"] {
        if !a[f].is_null() || !b[f].is_null() {
            out[format!("{f}_same")] = json!(a[f] == b[f]);
        }
    }
    if let (Some(ea), Some(eb)) = (a["evidence"].as_array(), b["evidence"].as_array()) {
        let span = |x: &Value| (x["start"].clone(), x["end"].clone(), x["text"].clone());
        out["evidence_spans_same"] = json!(ea.len() == eb.len() && ea.iter().zip(eb).all(|(x, y)| span(x) == span(y)));
        out["evidence_first_same"] = json!(ea.first().map(span) == eb.first().map(span));
        out["evidence_max_abs_probability_diff"] = json!(ea
            .iter()
            .zip(eb)
            .filter_map(|(x, y)| Some((x["probability"].as_f64()? - y["probability"].as_f64()?).abs()))
            .fold(0.0, f64::max));
        out["evidence_a"] = json!(ea.iter().map(span).collect::<Vec<_>>());
        out["evidence_b"] = json!(eb.iter().map(span).collect::<Vec<_>>());
    } else if !a["evidence"].is_null() || !b["evidence"].is_null() {
        out["evidence_spans_same"] = json!(false);
    }
    if let (Some(x), Some(y)) = (a["set_confidence"].as_f64(), b["set_confidence"].as_f64()) {
        out["set_confidence_diff"] = json!(y - x);
    }
    if let (Some(ea), Some(eb)) = (a["expected_costs"].as_object(), b["expected_costs"].as_object()) {
        out["expected_costs_keys_same"] = json!(ea.keys().eq(eb.keys()));
        out["expected_costs_max_abs_diff"] =
            json!(ea.iter().filter_map(|(k, v)| Some((v.as_f64()? - eb.get(k)?.as_f64()?).abs())).fold(0.0, f64::max));
    }
    out
}

pub fn compare(a_dir: &Path, b_dir: &Path) -> Result<Value> {
    let mut tc = TokenCheck::default();
    let a_bench = read_jsonl(&a_dir.join("bench.jsonl"))?;
    let b_bench: HashMap<String, Value> =
        read_jsonl(&b_dir.join("bench.jsonl"))?.into_iter().map(|r| (r["index"].to_string(), r)).collect();
    let mut num = Numeric::default();
    let (mut qa_avg, mut qb_avg, mut qa_cal, mut qb_cal, mut gold) = (vec![], vec![], vec![], vec![], vec![]);
    let mut missing = Vec::new();
    for a in &a_bench {
        let key = a["index"].to_string();
        let Some(b) = b_bench.get(&key) else {
            missing.push(key);
            continue;
        };
        let tag = format!("bench/{key}");
        if tc.compare(&tag, a, b, None) {
            num.add(&tag, a, b);
            if let Some(g) = a["gold"].as_u64() {
                gold.push(g as usize);
                qa_avg.push(f64s(&a["p_avg"]));
                qb_avg.push(f64s(&b["p_avg"]));
                qa_cal.push(f64s(&a["p_cal"]));
                qb_cal.push(f64s(&b["p_cal"]));
            }
        }
    }

    let b_so: HashMap<String, Value> =
        read_jsonl(&b_dir.join("systemone.jsonl"))?.into_iter().map(|r| (str_of(&r, "id").to_string(), r)).collect();
    let mut so_num = Numeric::default();
    let mut cases = Vec::new();
    let mut upstream_cases = Vec::new();
    for a in read_jsonl(&a_dir.join("systemone.jsonl"))? {
        let id = str_of(&a, "id").to_string();
        let Some(b) = b_so.get(&id) else {
            missing.push(id);
            continue;
        };
        let mut items = Vec::new();
        for bi in b["items"].as_array().into_iter().flatten() {
            let name = str_of(bi, "name");
            let Some(ai) = item_named(&a, name) else {
                items.push(json!({"item": name, "note": "only in b"}));
                continue;
            };
            let tag = format!("systemone/{id}/{name}");
            let same = tc.compare(&tag, ai, bi, known_deviation(&id, name));
            if same {
                so_num.add(&tag, ai, bi);
            }
            let answer = match (
                answer_of(&a).and_then(|x| x["answers"].get(name)),
                answer_of(b).and_then(|x| x["answers"].get(name)),
            ) {
                (Some(x), Some(y)) => answer_diff(x, y),
                _ => Value::Null,
            };
            items.push(json!({"item": name, "tokens_identical": same, "tv_cal": tv(&f64s(&ai["p_cal"]), &f64s(&bi["p_cal"])), "answer": answer}));
        }
        cases.push(json!({"id": id, "error_a": error_of(&a), "error_b": error_of(b), "deviation": known_deviation(&id, ""), "items": items}));
        // upstream-compatible endpoint: its answer against upstream Server.decide
        if let Some(ub) = b.get("answer_upstream") {
            let ua = answer_of(&a);
            let err_a = a.get("upstream_error").filter(|v| !v.is_null());
            let err_b = ub.get("error");
            let mut qs = Vec::new();
            if let (Some(ua), None) = (ua, err_b) {
                let names: std::collections::BTreeSet<&String> = ua["answers"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .chain(ub["answers"].as_object().into_iter().flatten())
                    .map(|(k, _)| k)
                    .collect();
                for n in names {
                    qs.push(match (ua["answers"].get(n), ub["answers"].get(n)) {
                        (Some(x), Some(y)) => json!({"question": n, "diff": answer_diff(x, y)}),
                        _ => json!({"question": n, "only_in_one": true}),
                    });
                }
            }
            upstream_cases.push(json!({"id": id, "error_upstream": err_a, "error_ours": err_b,
                                       "errors_agree": err_a.is_some() == err_b.is_some(), "questions": qs}));
        }
    }

    Ok(json!({
        "a": a_dir.display().to_string(), "b": b_dir.display().to_string(),
        "sign": "differences are b - a",
        "missing": missing,
        "tokens": tc.to_json(),
        "bench": num.to_json(),
        "quality": {"a_avg": quality(&qa_avg, &gold), "b_avg": quality(&qb_avg, &gold),
                    "a_cal": quality(&qa_cal, &gold), "b_cal": quality(&qb_cal, &gold)},
        "systemone": {"numeric_identical_tokens": so_num.to_json(), "cases": cases},
        "upstream_endpoint": upstream_cases,
    }))
}
