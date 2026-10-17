//! Manual host/protocol comparison with pinned upstream observations, not a CI test gate.
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{env, fs, path::Path};
fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let i = args.iter().position(|s| s == "--measurement-artifacts").context("--measurement-artifacts required")?;
    let root = Path::new(args.get(i + 1).context("missing artifact directory")?);
    for line in fs::read_to_string(root.join("host-inputs.jsonl"))?.lines() {
        let input: Value = serde_json::from_str(line)?;
        let result = match basal_core::tasks::expand(&input["request"], 255) {
            Ok(Some(mut expansion)) => {
                let second = expansion.plan.second_round(&input["first"], 255)?;
                let finished = expansion.plan.finish(
                    input["first"].clone(),
                    input.get("second"),
                    input["calibration"]["task_thresholds"].as_object().unwrap_or(&serde_json::Map::new()),
                )?;
                json!({"id":input["id"],"expanded":expansion.request,"second_request":second,"response":finished})
            }
            Ok(None) => json!({"id":input["id"],"unchanged":true}),
            Err(e) => json!({"id":input["id"],"error":e.to_string()}),
        };
        println!("{}", serde_json::to_string(&result)?);
    }
    Ok(())
}
