//! `basal export`: run this runtime on the inputs of a reference directory and write the results in the same format as
//! tools/reference/export_reference.py (bench.jsonl, systemone.jsonl, manifest.json), so any two exports can be
//! compared with `basal compare`. By default each question is a separate forward (both orders packed), as in the Python
//! export; `--batching budget|tree` runs the bench items through the batched paths instead.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use basal_core::decision::argmax;
use basal_core::engine::{ItemResult, Prepared};
use basal_core::{Backend, Batching, DecideError, Engine};
use serde_json::{json, Value};

use crate::reference::{bench_item, read_jsonl, str_of};

fn orders_json(p: &Prepared, r: &ItemResult) -> Value {
    let orders: Vec<Value> = p
        .orders
        .iter()
        .zip(&r.letter_logits)
        .zip(&r.probs)
        .map(|((o, lg), pr)| {
            json!({"perm": o.perm, "prompt": o.prompt, "input_ids": o.input_ids, "letter_ids": o.letter_ids,
                   "letter_logits": lg, "probs": pr})
        })
        .collect();
    json!(orders)
}

fn pack_json(p: &Prepared) -> Value {
    json!({"prefix_len": p.packed.prefix_len, "packed_len": p.packed.ids.len(), "readout_cols": p.packed.last})
}

pub fn export<B: Backend>(
    engine: &mut Engine<B>,
    inputs: &Path,
    out: &Path,
    batching: Batching,
    extra: Value,
) -> Result<()> {
    std::fs::create_dir(out).with_context(|| format!("creating {} (must not exist)", out.display()))?;
    let t0 = Instant::now();
    let mut f = std::fs::File::create(out.join("bench.jsonl"))?;
    let recs = read_jsonl(&inputs.join("bench.jsonl"))?;
    let prepared: Vec<Prepared> = recs.iter().map(|r| engine.prepare(bench_item(r)?)).collect::<Result<_>>()?;
    // All bench items in one run call, grouped into forwards by `batching` (Single = one question per forward).
    let results = engine.run(&prepared, batching)?;
    for ((rec, p), r) in recs.iter().zip(&prepared).zip(results) {
        let line = json!({
            "index": rec["index"], "source": rec["source"], "line": rec["line"], "id": rec["id"], "type": rec["type"],
            "state": rec["state"], "question": rec["question"], "options": rec["options"], "gold": rec["gold"],
            "lang": p.item.lang.to_string(), "orders": orders_json(p, &r), "pack": pack_json(p),
            "p_avg": r.p_avg, "argmax_avg": argmax(&r.p_avg), "temperature": r.temperature,
            "p_cal": r.p_cal, "argmax_cal": argmax(&r.p_cal),
        });
        writeln!(f, "{line}")?;
    }
    let bench_s = t0.elapsed().as_secs_f64();
    let mut f = std::fs::File::create(out.join("systemone.jsonl"))?;
    for case in read_jsonl(&inputs.join("systemone.jsonl"))? {
        let mut line = json!({"id": case["id"], "source": case["source"], "request": case["request"]});
        match engine.prepare_request(&case["request"]) {
            Ok(prep) => {
                let res = engine.run(&prep, Batching::Single)?;
                let items: Vec<Value> = prep
                    .iter()
                    .zip(&res)
                    .map(|(p, r)| {
                        json!({"name": p.item.name, "type": p.item.kind, "keys": p.item.keys, "options": p.item.options,
                               "lang": p.item.lang.to_string(), "state_text": p.item.state,
                               "question_text": p.item.question, "orders": orders_json(p, r), "pack": pack_json(p),
                               "p_avg": r.p_avg, "temperature": r.temperature, "p_cal": r.p_cal})
                    })
                    .collect();
                line["items"] = json!(items);
                // The served answer: questions of the request in one shared-prefix row (Engine::request_batching).
                // Its distributions are compared with the per-question forwards above.
                let answer = match engine.decide(&case["request"]) {
                    Ok(a) => a,
                    Err(e) => e.to_json(),
                };
                let mut served_diff: f64 = 0.0;
                for (p, r) in prep.iter().zip(&res) {
                    let a = &answer["answers"][p.item.name.as_str()];
                    let served: Vec<f64> = match a["noul"].as_f64() {
                        Some(v) => vec![v, 1.0 - v],
                        None => {
                            p.item.keys.iter().map(|k| a["probabilities"][k].as_f64().unwrap_or(f64::NAN)).collect()
                        }
                    };
                    for (x, y) in served.iter().zip(&r.p_cal) {
                        served_diff = served_diff.max((x - y).abs());
                    }
                }
                line["answer"] = answer;
                line["answer_vs_items_max_abs_prob_diff"] = json!(served_diff);
            }
            Err(DecideError::Internal(e)) => return Err(e),
            Err(e) => {
                line["items"] = json!([]);
                line["error"] = e.to_json();
            }
        }
        // The answer of the upstream-compatible endpoint (/v1/basal) for the same request.
        line["answer_upstream"] = match engine.decide_as(&case["request"], basal_core::request::Dialect::Upstream) {
            Ok(a) => a,
            Err(DecideError::Internal(e)) => return Err(e),
            Err(e) => e.to_upstream_json(),
        };
        writeln!(f, "{line}")?;
        eprintln!("systemone {}", str_of(&case, "id"));
    }
    let manifest = json!({
        "tool": "basal export", "inputs": inputs.display().to_string(),
        "model": engine.manifest.name, "model_dir": engine.manifest.dir.display().to_string(),
        "backend": engine.backend.describe(), "bench_batching": format!("{batching:?}"),
        "host_math": "softmax, order average and calibration in f64",
        "calibration": {"temperatures": engine.manifest.temperatures.iter().map(|(k, v)| (k.as_str().to_string(), json!(v))).collect::<serde_json::Map<_, _>>()},
        "request_batching": format!("{:?}", engine.request_batching),
        "bench_s": bench_s, "total_s": t0.elapsed().as_secs_f64(), "run": extra,
    });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&manifest)? + "\n")?;
    Ok(())
}
