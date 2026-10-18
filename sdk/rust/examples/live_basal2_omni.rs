//! Manual live SDK/model comparison. This example is never a CI test gate.
use basal_client::*;
use serde_json::{json, Value};
use std::{env, error::Error as StdError, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn StdError>>;

fn selection<'a>(q: &'a Value, exclusive: &'a [&'a str]) -> MultiOptions<'a> {
    MultiOptions {
        threshold: q["threshold"].as_f64(),
        min: q["min"].as_u64().map(|n| n as usize),
        max: q["max"].as_u64().map(|n| n as usize),
        objective: serde_json::from_value(q.get("objective").cloned().unwrap_or(json!("threshold")))
            .expect("diagnostic objective"),
        exclusive,
        fallback: q["fallback"].as_str(),
        fallback_threshold: q["fallback_threshold"].as_f64(),
    }
}
fn tool_definition(q: &Value) -> Result<ToolQuestion> {
    let extras: Vec<_> = q["extra"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|v| match v.as_str() {
            Some("none") => ToolExtra::None,
            Some("ask_user") => ToolExtra::AskUser,
            Some("answer_directly") => ToolExtra::AnswerDirectly,
            _ => panic!("diagnostic extra"),
        })
        .collect();
    let exclusive: Vec<_> = q["exclusive"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    Ok(tool_with_options(
        q["instructions"].as_str().ok_or("missing instructions")?,
        q["tools"].as_array().ok_or("missing tools")?,
        ToolOptions {
            extra: q.get("extra").map(|_| extras.as_slice()),
            rank: q["rank"].as_u64().map(|n| n as usize),
            multi: q["multi"].as_bool().unwrap_or(false),
            max_error: q["max_error"].as_f64(),
            selection: selection(q, &exclusive),
        },
    )?)
}
fn definitions(case: &Value) -> Result<Questions> {
    let mut result = Questions::new();
    for (id, q) in case["questions"].as_object().ok_or("missing questions")? {
        let instructions = q["instructions"].as_str().unwrap_or("");
        let values: Vec<_> = q["criteria"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, v)| Ok((k.as_str(), v.as_str().ok_or("criterion text")?)))
            .collect::<Result<_>>()?;
        let exclusive: Vec<_> = q["exclusive"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        let errors: Vec<f64> = match q.get("max_error") {
            None => vec![],
            Some(Value::Array(v)) => {
                v.iter().map(|v| v.as_f64().ok_or("error target")).collect::<std::result::Result<_, _>>()?
            }
            Some(v) => vec![v.as_f64().ok_or("error target")?],
        };
        let max_error = q.get("max_error").map(|_| errors.as_slice());
        result = match q["type"].as_str() {
            Some("classify") => result.with(
                id,
                &classify(
                    instructions,
                    q["labels"].clone(),
                    ClassifyOptions {
                        multi_label: q["multi_label"].as_bool().unwrap_or(false),
                        none: q["none"].as_bool().unwrap_or(false),
                        max_error,
                        hide_keys: q["option_keys"] == "hide",
                        selection: selection(q, &exclusive),
                    },
                )?,
            )?,
            Some("verify") => result.with(
                id,
                &verify(
                    instructions,
                    q["claims"].clone(),
                    VerifyOptions { binary: q["binary"].as_bool().unwrap_or(false), max_error },
                )?,
            )?,
            Some("compress") => result.with(
                id,
                &compress(
                    instructions,
                    CompressOptions {
                        unit: serde_json::from_value(q.get("unit").cloned().unwrap_or(json!("sentence")))?,
                        keep_ratio: q["keep_ratio"].as_f64(),
                        threshold: q["threshold"].as_f64(),
                        min_keep: q["min_keep"].as_u64().map(|v| v as usize),
                    },
                )?,
            )?,
            Some("extract") => result.with(id, &extract(instructions, q["schema"].clone())?)?,
            Some("choice") => result.with(id, &choice(instructions, &values)?)?,
            Some("noul") => result.with(id, &noul(instructions)?)?,
            Some("score") => {
                let levels: Vec<_> = q["criteria"]
                    .as_array()
                    .ok_or("missing levels")?
                    .iter()
                    .map(|v| v.as_str().ok_or("level text"))
                    .collect::<std::result::Result<_, _>>()?;
                result.with(id, &score(instructions, &levels)?)?
            }
            Some("multi") => result.with(id, &multi_with_options(instructions, &values, selection(q, &exclusive))?)?,
            Some("tool") => result.with(id, &tool_definition(q)?)?,
            Some("act") => {
                let costs: CostMatrix = serde_json::from_value(q["costs"].clone())?;
                result.with(
                    id,
                    &act_with_options(
                        instructions,
                        &costs,
                        ActOptions {
                            criteria: q.get("criteria").map(|_| values.as_slice()),
                            defer_action: q["defer_action"].as_str(),
                            max_error: q["max_error"].as_f64(),
                            ..Default::default()
                        },
                    )?,
                )?
            }
            _ => return Err("unsupported diagnostic question".into()),
        };
    }
    Ok(result)
}
fn selection_projection(a: MultiAnswer<'_>) -> Value {
    json!({"type":"multi","selected":a.selected,"threshold":a.threshold,"objective":a.objective,"expected_score":a.expected_score,"constraints_applied":a.constraints_applied,"set_confidence":a.set_confidence,"probabilities":a.probabilities})
}
fn project(answer: AnswerResult<'_>) -> Value {
    match answer {
        AnswerResult::Classify(a) => {
            json!({"type":"classify","experimental":true,"scores":a.scores,"label":a.label,"top_label":a.top_label,"confidence":a.confidence,"automated":a.automated,"labels":a.labels,"set_confidence":a.set_confidence})
        }
        AnswerResult::Verify(a) => {
            json!({"type":"verify","experimental":true,"claims":a.claims,"all_supported":a.all_supported,"not_supported":a.not_supported})
        }
        AnswerResult::Compress(a) => {
            json!({"type":"compress","experimental":true,"compressed":a.compressed,"kept":a.kept,"units":a.units,"unit":a.unit,"original_chars":a.original_chars,"compressed_chars":a.compressed_chars,"ratio":a.ratio})
        }
        AnswerResult::Extract(a) => json!({"type":"extract","experimental":true,"data":a.data,"fields":a.fields}),
        AnswerResult::Choice(a) => {
            json!({"type":"choice","choice":a.choice,"confidence":a.confidence,"probabilities":a.probabilities})
        }
        AnswerResult::Noul(a) => {
            json!({"type":"noul","noul":a.noul,"confidence":a.confidence,"probabilities":a.probabilities})
        }
        AnswerResult::Score(a) => {
            json!({"type":"score","score":a.score,"legend":a.legend,"confidence":a.confidence,"probabilities":a.probabilities})
        }
        AnswerResult::Multi(a) => selection_projection(a),
        AnswerResult::Act(a) => {
            json!({"type":"act","action":a.action,"answer":a.answer,"expected_costs":a.expected_costs,"confidence":a.confidence,"calibration":a.calibration,"deferred":a.raw.deferred,"refused":a.raw.refused,"max_error":a.raw.max_error,"probabilities":a.probabilities})
        }
        AnswerResult::Tool(a) => {
            let selection = a.selection.map(selection_projection);
            json!({"type":"tool","tool":a.tool,"call":a.call,"multi":a.multi,"tools":a.tools,"confidence":a.confidence,"ranking":a.ranking.iter().map(|r|json!({"tool":r.tool,"probability":r.probability})).collect::<Vec<_>>(),"selection":selection,"deferred":a.raw.deferred,"refused":a.raw.refused,"max_error":a.raw.max_error,"probabilities":a.probabilities})
        }
    }
}
fn raw_metadata(response: &DecisionResponse) -> Value {
    let u = &response.usage;
    let media=u.media.as_ref().map(|m|json!({"images":m.images,"audio":m.audio,"media_tokens":m.media_tokens,"detail":match m.detail {MediaDetail::Default=>"default",MediaDetail::High=>"high"}}));
    json!({"model":response.model,"usage":{"input_tokens":u.input_tokens,"output_tokens":u.output_tokens,"questions":u.questions,"branches":u.branches,"longest_prompt_tokens":u.longest_prompt_tokens,"second_order_branches":u.second_order_branches,"media":media}})
}
#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let index = args.iter().position(|a| a == "--measurement-artifacts").ok_or("--measurement-artifacts required")?;
    let root = Path::new(args.get(index + 1).ok_or("missing artifact directory")?);
    let url = env::var("BASAL_URL").unwrap_or_else(|_| "http://127.0.0.1:19439".into());
    let client = Client::with_options(ClientOptions {
        base_url: &url,
        timeout: std::time::Duration::from_secs(300),
        ..Default::default()
    })?;
    let model = env::var("BASAL_MODEL")?;
    let discovery = client.models().await?;
    println!(
        "{}",
        json!({"sdk":"rust","models":discovery.models.iter().map(|m|json!({"name":m.name,"prompt_format":m.prompt_format,"orders":m.orders,"temperature_per_prim":m.temperature_per_prim,"media":m.media.as_ref().map(|ks|ks.iter().map(|k|match k {MediaKind::Image=>"image",MediaKind::Audio=>"audio"}).collect::<Vec<_>>())})).collect::<Vec<_>>()})
    );
    let diagnostic = Questions::new().with("n", &noul("Is this a diagnostic?")?)?;
    match client
        .decide_with_options(
            json!("diagnostic"),
            &diagnostic,
            DecisionOptions { model: Some("__sdk_unknown_model__"), ..Default::default() },
        )
        .await
    {
        Err(Error::Http { status: 422, .. }) => {
            println!("{}", json!({"sdk":"rust-map","check":"unknown-model","status":422}))
        }
        _ => return Err("unexpected unknown-model result".into()),
    }
    match client
        .decide_with_options(
            json!("diagnostic"),
            &diagnostic,
            DecisionOptions { media: Some(&[]), ..Default::default() },
        )
        .await
    {
        Err(Error::InvalidRequest(_)) => {
            println!("{}", json!({"sdk":"rust-map","check":"empty-media","status":"local-rejection"}))
        }
        _ => return Err("unexpected empty-media result".into()),
    }
    for line in fs::read_to_string(root.join("inputs/cases.jsonl"))?.lines() {
        let case: Value = serde_json::from_str(line)?;
        if case.get("media").is_some() && env::var("BASAL_MEDIA").unwrap_or_default() != "1" {
            continue;
        }
        let mut media = Vec::new();
        for item in case["media"].as_array().into_iter().flatten() {
            let bytes = fs::read(root.join(item["file"].as_str().ok_or("media file")?))?;
            let kind = match item["type"].as_str() {
                Some("image") => MediaKind::Image,
                Some("audio") => MediaKind::Audio,
                _ => return Err("media type".into()),
            };
            let encoded = match kind {
                MediaKind::Image => MediaInput::image(&bytes),
                MediaKind::Audio => MediaInput::audio(&bytes),
            };
            let input = match item["encoding"].as_str() {
                Some("bytes") => encoded,
                Some("base64") => MediaInput::from_base64(kind, encoded.data)?,
                Some("data-url") => MediaInput::from_base64(
                    kind,
                    format!("data:{};base64,{}", item["mime"].as_str().ok_or("media mime")?, encoded.data),
                )?,
                _ => return Err("media encoding".into()),
            };
            media.push(input);
        }
        let options = DecisionOptions {
            model: Some(&model),
            lang: match case["lang"].as_str() {
                Some("pl") => Some(Language::Pl),
                Some("en") => Some(Language::En),
                _ => None,
            },
            detail: match case["detail"].as_str() {
                Some("high") => Some(MediaDetail::High),
                Some("default") => Some(MediaDetail::Default),
                _ => None,
            },
            media: case.get("media").map(|_| media.as_slice()),
            ..Default::default()
        };
        let response = client.decide_with_options(case["state"].clone(), &definitions(&case)?, options).await?;
        let answers: serde_json::Map<_, _> =
            response.answers()?.into_iter().map(|(k, v)| (k.to_owned(), project(v))).collect();
        println!(
            "{}",
            json!({"sdk":"rust-map","case":case["id"],"answers":answers,"raw":raw_metadata(response.raw())})
        );
        if case["id"] == "named-api" {
            let q = &case["questions"]["labels"];
            let values: Vec<_> = q["criteria"]
                .as_object()
                .ok_or("labels")?
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str().expect("diagnostic text")))
                .collect();
            let route = tool_definition(&case["questions"]["route"])?;
            let labels =
                multi_with_options(q["instructions"].as_str().ok_or("instructions")?, &values, selection(q, &[]))?;
            let named = questions! {route:&route,labels:&labels}?;
            let result = client.decide_with_options(case["state"].clone(), &named, options).await?;
            let answers = result.answers()?;
            println!(
                "{}",
                json!({"sdk":"rust-named","case":"named-api","answers":{"route":project(AnswerResult::Tool(answers.route)),"labels":project(AnswerResult::Multi(answers.labels))},"raw":raw_metadata(result.raw())})
            );
        }
    }
    Ok(())
}
