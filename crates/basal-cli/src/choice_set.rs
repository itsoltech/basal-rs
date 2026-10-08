//! `basal eval-choice-set`: choice questions with more than 10 options from labelled sets (tools/choice-sets), each
//! answered with every given strategy of `basal_core::large_choice`. Reported per strategy and set: accuracy against
//! the labels, how often the label reached the final group, prompts and time per question.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use basal_core::decision::argmax;
use basal_core::large_choice::Strategy;
use basal_core::{Backend, Engine};
use serde_json::{json, Value};

use crate::large_eval::{answer, choice_item};
use crate::reference::read_jsonl;
use crate::stats::{quality, summary};

/// Per set: distributions, gold indices, questions with the label in the final group, ms, prompts.
type SetStats = (Vec<Vec<f64>>, Vec<usize>, usize, Vec<f64>, Vec<f64>);

pub fn eval<B: Backend>(
    engine: &mut Engine<B>,
    set: &Path,
    strategies: &[String],
    limit: Option<usize>,
    out: &Path,
) -> Result<Value> {
    std::fs::create_dir(out).with_context(|| format!("creating {} (must not exist)", out.display()))?;
    let mut recs = read_jsonl(set)?;
    if let Some(l) = limit {
        recs.truncate(l);
    }
    let mut by_strategy = Vec::new();
    engine.trace_large_choice = true;
    for name in strategies {
        let strategy = Strategy::parse(name).with_context(|| format!("unknown strategy {name:?}"))?;
        engine.large_choice_strategy = strategy;
        let mut f = std::fs::File::create(out.join(format!("{name}.jsonl")))?;
        let mut sets: BTreeMap<String, SetStats> = BTreeMap::new();
        for r in &recs {
            let options: Vec<String> = r["options"]
                .as_array()
                .context("options")?
                .iter()
                .map(|o| o.as_str().unwrap_or_default().to_string())
                .collect();
            let gold = r["gold"].as_u64().context("gold")? as usize;
            let item = choice_item(
                r["id"].to_string(),
                r["state"].as_str().unwrap_or_default(),
                r["question"].as_str().unwrap_or_default(),
                options,
            );
            let a = answer(engine, item, 1.0)?;
            let in_final = a.finalists.as_ref().map(|f| f.contains(&gold));
            // rank of the label (1 = strongest) in the joint fit after the first round
            let first_rank = a.first_round.as_ref().map(|q| 1 + q.iter().filter(|&&v| v > q[gold]).count());
            let top = argmax(&a.p);
            writeln!(
                f,
                "{}",
                json!({"id": r["id"], "dataset": r["dataset"], "gold": gold, "top": top, "p_gold": a.p[gold],
                       "p_top": a.p[top], "gold_in_final": in_final, "gold_rank_first_round": first_rank, "finalists": a.finalists, "prompts": a.prompts,
                       "ms": a.ms, "p": a.p})
            )?;
            let e = sets.entry(r["dataset"].as_str().unwrap_or("?").to_string()).or_default();
            e.0.push(a.p);
            e.1.push(gold);
            e.2 += usize::from(in_final == Some(true));
            e.3.push(a.ms);
            e.4.push(a.prompts as f64);
        }
        let per_set: serde_json::Map<String, Value> = sets
            .into_iter()
            .map(|(k, (d, g, fin, ms, pr))| {
                let n = g.len();
                (
                    k,
                    json!({"quality": quality(&d, &g), "gold_in_final_group": format!("{fin}/{n}"),
                           "ms_per_question": summary(&ms), "prompts_per_question": summary(&pr)}),
                )
            })
            .collect();
        eprintln!(
            "{name}: {}",
            serde_json::to_string(
                &per_set
                    .iter()
                    .map(|(k, v)| (k.clone(), v["quality"]["accuracy"].clone()))
                    .collect::<serde_json::Map<_, _>>()
            )?
        );
        by_strategy.push(json!({"strategy": name, "detail": format!("{strategy:?}"), "sets": per_set}));
    }
    let s = json!({"set": set.display().to_string(), "questions": recs.len(), "strategies": by_strategy,
                   "backend": engine.backend.describe()});
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&s)? + "\n")?;
    Ok(s)
}
