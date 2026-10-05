//! Offline benchmark with the methodology of upstream `basal-bench` (`bench.measure`), on the items of the reference:
//!   lat1  one decision with ONE option order at batch 1
//!   lat2  one decision with BOTH orders (shared prefix) at batch 1; the first 5 measured items are discarded
//!   dec_s two-order decisions per second, questions processed in groups of 16 (token-budget chunks inside)
//! Timings include prompt rendering, tokenization, packing, the GPU forward with readout and host math
//! (upstream timings include tokenization and packing as well, not rendering).

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use basal_core::decision::argmax;
use basal_core::engine::Prepared;
use basal_core::pack::pack;
use basal_core::{Backend, Batching, Engine, Item};
use serde_json::{json, Value};

use crate::reference::{bench_item, read_jsonl};
use crate::stats::{median, quantile, summary};

fn single_order(mut p: Prepared) -> Prepared {
    p.orders.truncate(1);
    p.packed = pack(&[p.orders[0].input_ids.clone()]);
    p
}

pub fn bench<B: Backend>(engine: &mut Engine<B>, reference: &Path, lat_n: usize, batch: Batching) -> Result<Value> {
    let recs = read_jsonl(&reference.join("bench.jsonl"))?;
    let items: Vec<Item> = recs.iter().map(bench_item).collect::<Result<_>>()?;
    let n = items.len();

    // Decisions of all items, 32 questions per call as upstream (token-budget chunks inside).
    let mut correct = 0;
    let mut with_gold = 0;
    let t_all = Instant::now();
    for (c, rc) in items.chunks(32).zip(recs.chunks(32)) {
        let prep: Vec<Prepared> = c.iter().cloned().map(|it| engine.prepare(it)).collect::<Result<_>>()?;
        for (res, rec) in engine.run(&prep, Batching::Budget)?.iter().zip(rc) {
            if let Some(g) = rec.get("gold").and_then(Value::as_u64) {
                with_gold += 1;
                correct += (argmax(&res.p_avg) == g as usize) as usize;
            }
        }
    }
    let all_s = t_all.elapsed().as_secs_f64();

    let lat_n = lat_n.min(n.saturating_sub(5));
    let (mut one, mut two, mut prep_ms, mut fwd_ms, mut tokens) = (vec![], vec![], vec![], vec![], vec![]);
    let mut per_item = Vec::new();
    for (idx, it) in items.iter().enumerate().take(lat_n + 5) {
        // warm-up of the shapes, as upstream
        let p = engine.prepare(it.clone())?;
        engine.run(&[single_order(p.clone())], Batching::Single)?;
        engine.run(&[p], Batching::Single)?;

        let t = Instant::now();
        let p1 = single_order(engine.prepare(it.clone())?);
        engine.run(&[p1], Batching::Single)?;
        one.push(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        let p = engine.prepare(it.clone())?;
        let t_mid = Instant::now();
        engine.run(std::slice::from_ref(&p), Batching::Single)?;
        let t_end = Instant::now();
        two.push((t_end - t).as_secs_f64() * 1e3);
        prep_ms.push((t_mid - t).as_secs_f64() * 1e3);
        fwd_ms.push((t_end - t_mid).as_secs_f64() * 1e3);
        tokens.push(p.packed.ids.len() as f64);
        if idx >= 5 {
            per_item.push(json!({"index": idx, "packed_tokens": p.packed.ids.len(), "lat1_ms": one[idx],
                                 "lat2_ms": two[idx], "lat2_prepare_ms": prep_ms[idx], "lat2_forward_ms": fwd_ms[idx]}));
        }
    }
    let skip = |v: &mut Vec<f64>| -> Vec<f64> {
        let mut s = v.split_off(5.min(v.len()));
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s
    };
    let (one, two, prep_ms, fwd_ms, tokens) =
        (skip(&mut one), skip(&mut two), skip(&mut prep_ms), skip(&mut fwd_ms), skip(&mut tokens));

    let chunk: Vec<Item> = items.iter().take(16 * 8).cloned().collect();
    let warm: Vec<Prepared> = chunk.iter().take(16).cloned().map(|it| engine.prepare(it)).collect::<Result<_>>()?;
    engine.run(&warm, batch)?;
    // The throughput phase is repeated: single runs varied by about 10 %.
    let mut dec_s_runs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        for c in chunk.chunks(16) {
            let prep: Vec<Prepared> = c.iter().cloned().map(|it| engine.prepare(it)).collect::<Result<_>>()?;
            engine.run(&prep, batch)?;
        }
        dec_s_runs.push(chunk.len() as f64 / t.elapsed().as_secs_f64());
    }
    let mut sorted = dec_s_runs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dec_s = median(&sorted);

    Ok(json!({
        "n": n, "lat_n": two.len(),
        "lat1_ms": median(&one), "lat2_ms": median(&two), "lat2_p95_ms": quantile(&two, 0.95),
        "lat2_ms_all": two, "lat1_ms_all": one, "per_item": per_item,
        "lat2_prepare_ms": summary(&prep_ms), "lat2_forward_ms": summary(&fwd_ms),
        "lat2_packed_tokens": summary(&tokens),
        "dec_s": dec_s, "dec_s_runs": dec_s_runs, "dec_s_questions": chunk.len(), "dec_s_batching": format!("{batch:?}"),
        "all_items_s": all_s,
        "acc": if with_gold > 0 { json!(correct as f64 / with_gold as f64) } else { Value::Null },
        "correct": correct, "with_gold": with_gold,
    }))
}

/// Latency of whole System One requests (`Engine::decide`), one client, each request warmed up once and timed
/// `reps` times; counterpart of tools/reference/bench_requests.py.
pub fn bench_requests<B: Backend>(engine: &mut Engine<B>, requests: &Path, reps: usize) -> Result<Value> {
    let mut rows = Vec::new();
    for r in read_jsonl(requests)? {
        let body = &r["request"];
        let prepared = engine.prepare_request(body).map_err(|e| anyhow::anyhow!("{e}"))?;
        let separate: usize = prepared.iter().map(|p| p.packed.ids.len()).sum();
        let tree: usize = {
            let toks: Vec<Vec<Vec<u32>>> =
                prepared.iter().map(|p| p.orders.iter().map(|o| o.input_ids.clone()).collect()).collect();
            basal_core::pack::pack_tree(&toks).ids.len()
        };
        let resp = engine.decide(body).map_err(|e| anyhow::anyhow!("{e}"))?; // warm-up
        let mut lat = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            engine.decide(body).map_err(|e| anyhow::anyhow!("{e}"))?;
            lat.push(t.elapsed().as_secs_f64() * 1e3);
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let row = json!({
            "id": r["id"], "questions": prepared.len(), "input_tokens": resp["usage"]["input_tokens"],
            "packed_tokens_per_question_rows": separate, "packed_tokens_shared_tree": tree,
            "median_ms": median(&lat), "min_ms": lat[0], "max_ms": lat[lat.len() - 1], "all_ms": lat,
            "answers": resp["answers"],
        });
        eprintln!("{} q={} median {:.1} ms", r["id"], prepared.len(), median(&lat));
        rows.push(row);
    }
    Ok(json!({"batching": format!("{:?}", engine.request_batching), "reps": reps, "requests": rows}))
}
