//! `basal eval-large-choice`: evaluation of the grouped 11..255 choice strategy on the reference items.
//!
//! The original Basal cannot answer more than 10 options, so there is no reference behaviour. Each reference item
//! keeps its options and gold label and receives distractor options taken from items of other topics (`TOPICS`),
//! up to `n` options, shuffled with a seed. Reported per size: accuracy, probability mass on the original options,
//! agreement and total variation of the original options' renormalised distribution with the direct answer
//! (the item asked alone, 2..10 options), agreement between seeds (sensitivity to positions and grouping) and cost.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use basal_core::decision::argmax;
use basal_core::prompt::lang_of;
use basal_core::request::{Item, QType};
use basal_core::{Backend, Engine};
use serde_json::{json, Value};

use crate::reference::{read_jsonl, str_of};
use crate::stats::{quality, summary, tv};

/// Topic of each reference item (index in `bench.jsonl`). Distractors come only from items of another topic, so that a
/// distractor is not a synonym or translation of a correct answer (e.g. "High" for a Polish urgency scale, "Yes" for
/// another yes/no question).
const TOPICS: [&str; 44] = [
    "yesno",
    "yesno",
    "level",
    "yesno",
    "money",
    "routing",
    "alert",
    "yesno",
    "level",
    "money",
    "money", // 0..=10
    "warranty",
    "yesno",
    "routing",
    "yesno",
    "yesno",
    "routing",
    "level",
    "next-step",
    "yesno",
    "yesno", // 11..=20
    "correctness",
    "yesno",
    "days",
    "sentiment",
    "level",
    "level",
    "sentiment",
    "correctness",
    "date",
    "days", // 21..=30
    "level",
    "sentiment",
    "sentiment",
    "document",
    "yesno",
    "yesno",
    "class",
    "yesno",
    "yesno",
    "routing", // 31..=40
    "money",
    "sentiment",
    "routing", // 41..=43
];

/// Deterministic xorshift64* generator (no dependency, reproducible across platforms).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

pub(crate) fn choice_item(name: String, state: &str, question: &str, options: Vec<String>) -> Item {
    Item {
        name,
        kind: QType::Choice,
        keys: (0..options.len()).map(|i| format!("o{i}")).collect(),
        options,
        legend: vec![],
        state: state.to_string(),
        question: question.to_string(),
        lang: lang_of(&format!("{state}{question}")),
        ext: Default::default(),
        evidence: false,
        uncalibrated: false,
    }
}

/// One answered question: final distribution, prompts of all rounds, packed tokens of the first round, time and the
/// final group of a grouped large choice.
pub(crate) struct Answered {
    pub(crate) p: Vec<f64>,
    pub(crate) prompts: usize,
    pub(crate) tokens: usize,
    pub(crate) ms: f64,
    pub(crate) finalists: Option<Vec<usize>>,
}

/// Run planned items one by one (each item = one shared-prefix tree forward per round) and return the answer.
pub(crate) fn answer<B: Backend>(engine: &mut Engine<B>, item: Item, duty: f64) -> Result<Answered> {
    let t = Instant::now();
    let plan = engine.plan_items(vec![item])?;
    let mut out = engine.run_plan(&plan)?;
    let p = out.finals.remove(0).1;
    let prompts = plan.prepared.iter().map(|p| p.orders.len()).sum::<usize>() + out.extra_prompts;
    let toks: Vec<Vec<Vec<u32>>> =
        plan.prepared.iter().map(|p| p.orders.iter().map(|o| o.input_ids.clone()).collect()).collect();
    let tokens = basal_core::pack::pack_tree(&toks).ids.len();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    if duty < 1.0 {
        std::thread::sleep(std::time::Duration::from_secs_f64(ms / 1e3 * (1.0 / duty.max(0.05) - 1.0)));
    }
    Ok(Answered { p, prompts, tokens, ms, finalists: out.finalists.remove(0) })
}

pub fn eval<B: Backend>(
    engine: &mut Engine<B>,
    reference: &Path,
    sizes: &[usize],
    seeds: &[u64],
    duty: f64,
    out: &Path,
) -> Result<Value> {
    std::fs::create_dir(out).with_context(|| format!("creating {} (must not exist)", out.display()))?;
    let recs = read_jsonl(&reference.join("bench.jsonl"))?;
    anyhow::ensure!(recs.len() == TOPICS.len(), "the topic map covers the 44 default basal-bench items only");
    let opts = |r: &Value| -> Vec<String> {
        r["options"]
            .as_array()
            .map(|a| a.iter().map(|o| o.as_str().unwrap_or("").to_string()).collect())
            .unwrap_or_default()
    };
    let mut pool: Vec<(usize, String)> = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        for o in opts(r) {
            if !pool.iter().any(|(_, x)| *x == o) {
                pool.push((i, o));
            }
        }
    }
    // direct answers (the item alone, as a choice question)
    let mut direct = Vec::new();
    for r in &recs {
        let it = choice_item(r["index"].to_string(), str_of(r, "state"), str_of(r, "question"), opts(r));
        direct.push(answer(engine, it, duty)?.p);
    }
    let mut f = std::fs::File::create(out.join("items.jsonl"))?;
    let mut sizes_out = Vec::new();
    for &n in sizes {
        let mut per_seed: Vec<Vec<Option<Value>>> = Vec::new();
        let (mut dists, mut golds, mut tvs, mut mass, mut agree, mut ms, mut toks, mut prompts) =
            (vec![], vec![], vec![], vec![], 0usize, vec![], vec![], vec![]);
        let (mut gold_in_final, mut with_gold) = (0usize, 0usize);
        let mut evaluated = 0usize;
        for &seed in seeds {
            let mut row = Vec::new();
            for (i, r) in recs.iter().enumerate() {
                let orig = opts(r);
                let k = orig.len();
                let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1) ^ ((n as u64) << 32));
                let mut cand: Vec<&String> = pool
                    .iter()
                    .filter(|(src, o)| TOPICS[*src] != TOPICS[i] && !orig.contains(o))
                    .map(|(_, o)| o)
                    .collect();
                if cand.len() < n.saturating_sub(k) {
                    row.push(None);
                    continue;
                }
                rng.shuffle(&mut cand);
                // positions: original option j sits at pos[j]
                let mut all: Vec<(Option<usize>, String)> =
                    orig.iter().cloned().enumerate().map(|(j, o)| (Some(j), o)).collect();
                all.extend(cand.into_iter().take(n - k).map(|o| (None, o.clone())));
                rng.shuffle(&mut all);
                let pos: Vec<usize> = (0..k).map(|j| all.iter().position(|(s, _)| *s == Some(j)).unwrap()).collect();
                let item = choice_item(
                    format!("{}-{n}-{seed}", r["index"]),
                    str_of(r, "state"),
                    str_of(r, "question"),
                    all.iter().map(|(_, o)| o.clone()).collect(),
                );
                let a = answer(engine, item, duty)?;
                let (p, np, nt, t) = (a.p, a.prompts, a.tokens, a.ms);
                let restricted: Vec<f64> = {
                    let v: Vec<f64> = pos.iter().map(|&q| p[q]).collect();
                    let s: f64 = v.iter().sum();
                    v.iter().map(|x| x / s).collect()
                };
                let m: f64 = pos.iter().map(|&q| p[q]).sum();
                let top = argmax(&p);
                let top_orig = all[top].0; // Some(j) if an original option won
                let gold = r["gold"].as_u64().map(|g| g as usize);
                let in_final = gold.zip(a.finalists.as_ref()).map(|(g, f)| f.contains(&pos[g]));
                if let Some(g) = gold {
                    // quality of the joint distribution over all n options, gold at its shuffled position
                    dists.push(p.clone());
                    golds.push(pos[g]);
                    with_gold += 1;
                    gold_in_final += usize::from(in_final == Some(true));
                }
                tvs.push(tv(&restricted, &direct[i]));
                mass.push(m);
                agree += usize::from(argmax(&restricted) == argmax(&direct[i]));
                evaluated += 1;
                ms.push(t);
                toks.push(nt as f64);
                prompts.push(np as f64);
                let rec = json!({"n": n, "seed": seed, "index": r["index"], "gold": gold, "winner_original": top_orig, "winner": all[top].1,
                                 "mass_original": m, "tv_restricted_vs_direct": tvs.last(), "ms": t, "prompts": np, "gold_in_final": in_final,
                                 "tokens": nt, "restricted": restricted, "direct": direct[i]});
                writeln!(f, "{rec}")?;
                row.push(Some(json!({"winner": all[top].1})));
            }
            per_seed.push(row);
        }
        let mut seed_agree = (0usize, 0usize);
        if per_seed.len() >= 2 {
            for (a, b) in per_seed[0].iter().zip(&per_seed[1]) {
                if let (Some(a), Some(b)) = (a, b) {
                    seed_agree.1 += 1;
                    seed_agree.0 += usize::from(a["winner"] == b["winner"]);
                }
            }
        }
        let s = json!({
            "n": n, "evaluated": evaluated, "quality": quality(&dists, &golds),
            "restricted_argmax_agrees_with_direct": format!("{agree}/{evaluated}"),
            "gold_in_final_group": format!("{gold_in_final}/{with_gold}"),
            "tv_restricted_vs_direct": summary(&tvs), "mass_on_original_options": summary(&mass),
            "winner_same_across_seeds": format!("{}/{}", seed_agree.0, seed_agree.1),
            "ms_per_question": summary(&ms), "tokens_per_question": summary(&toks), "prompts_per_question": summary(&prompts),
        });
        eprintln!("n={n}: {}", s["quality"]);
        sizes_out.push(s);
    }
    let golds: Vec<usize> = recs.iter().filter_map(|r| r["gold"].as_u64().map(|g| g as usize)).collect();
    let summary = json!({
        "strategy": engine.large_choice_strategy.name(),
        "strategy_detail": format!("{:?}", engine.large_choice_strategy),
        "direct_quality": quality(&direct, &golds), "seeds": seeds, "sizes": sizes_out,
        "backend": engine.backend.describe(), "duty": duty, "distractor_pool": pool.len(), "distractors": "options of items with another topic (TOPICS)",
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)? + "\n")?;
    Ok(summary)
}
