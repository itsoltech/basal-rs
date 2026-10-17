//! Experimental upstream v2.1 task layer. Tasks become ordinary decisions; extraction has at most two rounds.
//! Ported from rkinas/basal v2.1.0 (Apache-2.0); source offsets count Unicode scalar values.
mod schema;
mod text;
mod words;

use crate::prompt::{lang_of, Lang};
use crate::request::{DecideError, FieldError};
use anyhow::{anyhow, bail, ensure, Context, Result};
use schema::{Field, Kind};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use text::{Span, Text};
use words::word;

const SEP: &str = "::";

pub(crate) fn invalid(message: impl Into<String>) -> DecideError {
    DecideError::Validation(vec![FieldError::new(vec!["body".into(), "questions".into()], message, "value_error")])
}

pub fn is_task(q: &Value) -> bool {
    matches!(q.get("type").and_then(Value::as_str), Some("classify" | "verify" | "compress" | "extract"))
}

#[derive(Clone, Debug)]
enum TaskKind {
    Classify { multi: bool, risks: Vec<f64> },
    Verify { keys: Vec<String>, binary: bool, risks: Vec<f64> },
    Compress { spans: Vec<Span>, unit: String, keep_ratio: Option<f64>, threshold: f64, min_keep: usize },
    Extract { fields: Vec<Field>, regions: Vec<Span>, pre: String },
}
#[derive(Clone, Debug)]
struct Task {
    name: String,
    lang: Lang,
    kind: TaskKind,
}

#[derive(Clone, Debug)]
pub struct TaskPlan {
    continuation: Map<String, Value>,
    text: Arc<Text>,
    tasks: Vec<Task>,
    count: usize,
}

pub struct Expansion {
    pub request: Value,
    pub plan: TaskPlan,
}

fn finite(value: Option<&Value>, default: f64, name: &str) -> Result<f64> {
    let n = match value {
        None => default,
        Some(v) => v.as_f64().ok_or_else(|| anyhow!("{name} must be a number"))?,
    };
    ensure!(n.is_finite(), "{name} must be finite");
    Ok(n)
}
fn risks(q: &Value) -> Result<Vec<f64>> {
    let values = match q.get("max_error") {
        None => vec![json!(0.01), json!(0.05)],
        Some(Value::Array(v)) => v.clone(),
        Some(v) => vec![v.clone()],
    };
    ensure!(values.len() <= 64, "at most 64 requested error rates");
    values
        .iter()
        .map(|v| {
            let n = finite(Some(v), 0.0, "max_error")?;
            ensure!(n > 0.0 && n < 1.0, "max_error must be in (0, 1)");
            Ok(n)
        })
        .collect()
}
fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => {
            // Python repr prefers single quotes unless only single quotes need escaping.
            let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
            let printable = {
                static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
                RE.get_or_init(|| {
                    regex::Regex::new(r"^[\p{L}\p{M}\p{N}\p{P}\p{S} ]$").expect("static Python printable categories")
                })
            };
            let mut out = String::new();
            out.push(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if !printable.is_match(c.encode_utf8(&mut [0; 4])) => {
                        let n = c as u32;
                        if n <= 0xff {
                            out.push_str(&format!("\\x{n:02x}"));
                        } else if n <= 0xffff {
                            out.push_str(&format!("\\u{n:04x}"));
                        } else {
                            out.push_str(&format!("\\U{n:08x}"));
                        }
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Array(v) => format!("[{}]", v.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(v) => format!(
            "{{{}}}",
            v.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(_) => crate::pyjson::dumps(v),
    }
}
fn py_string(v: &Value) -> String {
    if let Value::String(s) = v {
        s.clone()
    } else {
        py_repr(v)
    }
}
fn instruction(q: &Value) -> String {
    q.get("instructions").filter(|v| crate::request::truthy(Some(v))).map(py_string).unwrap_or_default()
}
fn render(lang: Lang, key: &str, values: &[(&str, &str)]) -> String {
    let mut rest = word(lang, key);
    let mut out = String::new();
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let end = rest[start..].find('}').expect("static task template braces") + start;
        let key = &rest[start + 1..end];
        out.push_str(values.iter().find(|(k, _)| *k == key).expect("static task template arguments").1);
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}
fn labels(value: &Value) -> Result<Map<String, Value>> {
    match value {
        Value::Object(m) if !m.is_empty() => {
            ensure!(m.len() <= 256, "at most 256 task labels");
            Ok(m.clone())
        }
        Value::Array(a) if !a.is_empty() => {
            ensure!(a.len() <= 256, "at most 256 task labels");
            let objects = a.iter().all(|v| v.is_object() && v.get("name").is_some());
            let mut out = Map::new();
            for v in a {
                let (key, rubric) = if objects {
                    (
                        py_string(&v["name"]),
                        v.get("rubric")
                            .filter(|v| crate::request::truthy(Some(v)))
                            .or_else(|| v.get("description"))
                            .cloned()
                            .unwrap_or(Value::Null),
                    )
                } else {
                    (py_string(v), Value::Null)
                };
                ensure!(!key.is_empty(), "label keys must be nonempty");
                out.insert(key, rubric);
            }
            Ok(out)
        }
        _ => bail!("labels must be a non-empty object or list"),
    }
}
fn claims(q: &Value) -> Result<Map<String, Value>> {
    match q.get("claims") {
        Some(Value::Object(m)) if !m.is_empty() => Ok(m.clone()),
        Some(Value::Array(a)) if !a.is_empty() => {
            Ok(a.iter().enumerate().map(|(i, v)| (i.to_string(), v.clone())).collect())
        }
        _ => bail!("claims must be a non-empty object or list"),
    }
}

/// Conservative work bound before task expansion duplicates any state. Includes a possible second extraction round.
pub(crate) fn estimate(q: &Value, state: &Value) -> Result<(usize, usize), DecideError> {
    fn inner(q: &Value, state: &Value) -> Result<(usize, usize)> {
        let size = if let Some(s) = state.as_str() { s.len() } else { crate::pyjson::dumps(state).len() };
        let extra = crate::pyjson::dumps(q).len();
        let mut questions = 1usize;
        let mut option_bytes = extra;
        let mut second = 0usize;
        match q["type"].as_str() {
            Some("classify") => {
                let l = labels(
                    q.get("labels")
                        .filter(|v| crate::request::truthy(Some(v)))
                        .or_else(|| q.get("criteria"))
                        .unwrap_or(&Value::Null),
                )?;
                if crate::request::truthy(q.get("multi_label")) {
                    questions = l.len();
                }
                risks(q)?;
            }
            Some("verify") => {
                questions = claims(q)?.len();
                risks(q)?;
            }
            Some("compress") => {
                let s = state.as_str().filter(|s| !s.is_empty()).context("compress needs a text state")?;
                let text = Text::new(s.to_string());
                let units = text.units(q.get("unit").and_then(Value::as_str).unwrap_or("sentence"))?;
                text::check_unit_count(units.len())?;
                questions = units.len();
                option_bytes = option_bytes.saturating_add(size);
            }
            Some("extract") => {
                let fields = schema::fields(&q["schema"])?;
                questions = 0;
                let t = Text::new(state.as_str().unwrap_or("").to_string());
                let regions = schema::regions(&t, &fields, 254)?;
                let max_region = regions.iter().map(|s| t.slice(*s).len()).max().unwrap_or(0);
                let previews = regions.iter().map(|s| text::shown(t.slice(*s)).len()).sum::<usize>();
                for f in fields {
                    match f.kind {
                        Kind::String | Kind::Number | Kind::Integer => {
                            questions = questions.saturating_add(1);
                            second = second.saturating_add(1);
                            option_bytes =
                                option_bytes.max(previews).max(max_region.saturating_mul(254).saturating_add(2048));
                        }
                        Kind::EnumList => {
                            questions =
                                questions.saturating_add(schema::consts(&f.spec["items"]).map_or(0, |v| v.len()))
                        }
                        _ => questions = questions.saturating_add(1),
                    }
                }
            }
            _ => bail!("unknown task"),
        }
        let branches = questions.saturating_add(second);
        let bytes = size
            .saturating_add(extra)
            .saturating_add(option_bytes)
            .saturating_add(2048)
            .saturating_mul(branches)
            .saturating_mul(2);
        Ok((branches, bytes))
    }
    inner(q, state).map_err(|e| invalid(format!("{e:#}")))
}

pub fn expand(body: &Value, max_options: usize) -> Result<Option<Expansion>, DecideError> {
    if body.get("questions").and_then(Value::as_object).is_some_and(|qs| qs.values().any(is_task)) {
        crate::request::prepared_input_bytes(body)?;
    }
    fn inner(body: &Value, max_options: usize) -> Result<Option<Expansion>> {
        let Some(input) = body.get("questions").and_then(Value::as_object) else { return Ok(None) };
        if !input.values().any(is_task) {
            return Ok(None);
        }
        ensure!((2..=255).contains(&max_options), "task option limit must be in 2..=255");
        let text = Arc::new(Text::new(body.get("state").and_then(Value::as_str).unwrap_or("").to_string()));
        let continuation = body
            .as_object()
            .context("request must be an object")?
            .iter()
            .filter(|(k, _)| matches!(k.as_str(), "state" | "lang" | "facts"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut questions = Map::new();
        let mut tasks = Vec::new();
        for (name, q) in input {
            if !is_task(q) {
                questions.insert(name.clone(), q.clone());
                continue;
            }
            ensure!(!name.contains(SEP), "a task question's name may not contain {SEP:?}");
            let prefix = format!("{name}{SEP}");
            ensure!(
                !input.keys().any(|key| key.starts_with(&prefix)),
                "task internal question names collide with a user question"
            );
            let lang = match body.get("lang").and_then(Value::as_str) {
                Some("pl") => Lang::Pl,
                Some("en") => Lang::En,
                None => lang_of(&format!("{}{}", text.value, instruction(q))),
                _ => bail!("lang must be pl or en"),
            };
            let mut sub = Map::new();
            let pre = instruction(q);
            let pre = if pre.is_empty() { pre } else { pre + "\n" };
            let kind = match q["type"].as_str() {
                Some("classify") => {
                    let mut labels = labels(
                        q.get("labels")
                            .filter(|v| crate::request::truthy(Some(v)))
                            .or_else(|| q.get("criteria"))
                            .unwrap_or(&Value::Null),
                    )?;
                    let multi = crate::request::truthy(q.get("multi_label"));
                    if crate::request::truthy(q.get("none")) && !multi {
                        labels.insert("none".into(), word(lang, "none").into());
                    }
                    let instructions = q
                        .get("instructions")
                        .filter(|v| crate::request::truthy(Some(v)))
                        .cloned()
                        .unwrap_or_else(|| word(lang, if multi { "classify_multi" } else { "classify" }).into());
                    let mut spec = json!({"type":if multi {"multi"} else {"choice"},"criteria":labels,"instructions":instructions,"option_keys":q.get("option_keys").cloned().unwrap_or(json!("show"))});
                    if multi {
                        for key in ["threshold", "min", "max", "objective"] {
                            if let Some(v) = q.get(key) {
                                spec[key] = v.clone();
                            }
                        }
                    }
                    sub.insert("labels".into(), spec);
                    TaskKind::Classify { multi, risks: risks(q)? }
                }
                Some("verify") => {
                    let claims = claims(q)?;
                    let binary = crate::request::truthy(q.get("binary"));
                    for (key, claim) in &claims {
                        let claim = crate::pyjson::text(claim);
                        let instructions = pre.clone()
                            + &render(lang, if binary { "verify_binary" } else { "verify" }, &[("claim", &claim)]);
                        let spec = if binary {
                            json!({"type":"noul","instructions":instructions})
                        } else {
                            json!({"type":"choice","instructions":instructions,"criteria":{"supported":word(lang,"supported"),"contradicted":word(lang,"contradicted"),"not_stated":word(lang,"not_stated")},"option_keys":"hide"})
                        };
                        sub.insert(format!("c{SEP}{key}"), spec);
                    }
                    TaskKind::Verify { keys: claims.keys().cloned().collect(), binary, risks: risks(q)? }
                }
                Some("compress") => {
                    ensure!(!text.value.is_empty(), "compress needs a text state");
                    let unit = q.get("unit").and_then(Value::as_str).unwrap_or("sentence");
                    let spans = text.units(unit)?;
                    text::check_unit_count(spans.len())?;
                    let query = instruction(q).trim().to_string();
                    for (i, span) in spans.iter().enumerate() {
                        let unit = text.slice(*span);
                        let instructions = if query.is_empty() {
                            render(lang, "compress_generic", &[("unit", unit)])
                        } else {
                            render(lang, "compress", &[("unit", unit), ("q", &query)])
                        };
                        sub.insert(format!("u{SEP}{i}"), json!({"type":"noul","instructions":instructions}));
                    }
                    let keep_ratio = q
                        .get("keep_ratio")
                        .filter(|v| !v.is_null())
                        .map(|v| finite(Some(v), 0.0, "keep_ratio"))
                        .transpose()?;
                    ensure!(keep_ratio.is_none_or(|r| r > 0.0 && r <= 1.0), "keep_ratio must be in (0, 1]");
                    let threshold = finite(q.get("threshold"), 0.5, "threshold")?;
                    ensure!((0.0..=1.0).contains(&threshold), "threshold must be in [0, 1]");
                    let min_keep = q
                        .get("min_keep")
                        .map_or(Some(1), Value::as_u64)
                        .and_then(|n| usize::try_from(n).ok())
                        .context("min_keep must be a nonnegative integer")?;
                    TaskKind::Compress { spans, unit: unit.into(), keep_ratio, threshold, min_keep }
                }
                Some("extract") => {
                    let fields = schema::fields(&q["schema"])?;
                    let regions = schema::regions(&text, &fields, max_options - 1)?;
                    for field in &fields {
                        let f = field.description();
                        let spec = match field.kind {
                            Kind::Enum | Kind::EnumList => {
                                let multi = matches!(field.kind, Kind::EnumList);
                                let vals = schema::consts(if multi { &field.spec["items"] } else { &field.spec })
                                    .context("enum values")?;
                                let mut criteria: Map<String, Value> = vals
                                    .iter()
                                    .map(|(v, d)| {
                                        let key = py_string(v);
                                        let shown = d
                                            .as_ref()
                                            .filter(|v| crate::request::truthy(Some(v)))
                                            .map_or(key.clone(), |d| format!("{key} ({})", py_string(d)));
                                        (key, shown.into())
                                    })
                                    .collect();
                                ensure!(!criteria.contains_key("__missing__"), "enum value __missing__ is reserved");
                                if !multi {
                                    criteria.insert("__missing__".into(), word(lang, "missing").into());
                                }
                                let mut spec = json!({"type":if multi {"multi"} else {"choice"},"criteria":criteria,"instructions":pre.clone()+&render(lang,if multi {"enum_multi"} else {"enum"},&[("field",&f)])});
                                if !multi {
                                    spec["option_keys"] = "hide".into();
                                }
                                spec
                            }
                            Kind::Boolean => {
                                json!({"type":"choice","instructions":pre.clone()+&render(lang,"enum",&[("field",&f)]),"criteria":{"true":word(lang,"yes"),"false":word(lang,"no"),"__missing__":word(lang,"missing")},"option_keys":"hide"})
                            }
                            _ => {
                                let mut criteria: Map<String, Value> = regions
                                    .iter()
                                    .enumerate()
                                    .map(|(i, s)| (format!("L{i}"), text::shown(text.slice(*s)).into()))
                                    .collect();
                                criteria.insert("__missing__".into(), word(lang, "missing").into());
                                json!({"type":"choice","criteria":criteria,"option_keys":"hide","instructions":pre.clone()+&render(lang,"line",&[("field",&f)])})
                            }
                        };
                        sub.insert(format!("f{SEP}{}", field.key), spec);
                    }
                    TaskKind::Extract { fields, regions, pre }
                }
                _ => bail!("unknown task"),
            };
            for (key, value) in sub {
                questions.insert(prefix.clone() + &key, value);
            }
            tasks.push(Task { name: name.clone(), lang, kind });
        }
        let mut request = body.clone();
        request["questions"] = questions.into();
        Ok(Some(Expansion { request, plan: TaskPlan { continuation, text, tasks, count: input.len() } }))
    }
    inner(body, max_options).map_err(|e| invalid(format!("{e:#}")))
}

fn answer<'a>(out: &'a Value, key: &str) -> Result<&'a Value> {
    out.get("answers").and_then(|a| a.get(key)).with_context(|| format!("missing task decision {key}"))
}
fn probability(value: &Value) -> Result<f64> {
    let p = value.as_f64().context("missing task probability")?;
    ensure!(p.is_finite() && (0.0..=1.0).contains(&p), "invalid task probability");
    Ok(p)
}
fn chosen(a: &Value) -> Result<&str> {
    a.get("choice").and_then(Value::as_str).context("missing task choice")
}
fn automated(cal: &Map<String, Value>, task: &str, confidence: f64, risks: &[f64]) -> Value {
    risks
        .iter()
        .map(|r| {
            let key = crate::pyjson::float_repr(*r);
            let value = cal
                .get(task)
                .and_then(|v| v.get(&key))
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .map(|t| Value::Bool(confidence >= t))
                .unwrap_or(Value::Null);
            (key, value)
        })
        .collect::<Map<_, _>>()
        .into()
}

impl TaskPlan {
    pub fn second_round(&mut self, out: &Value, max_options: usize) -> Result<Option<Value>> {
        ensure!((2..=255).contains(&max_options), "task option limit must be in 2..=255");
        let mut questions = Map::new();
        for task in &mut self.tasks {
            let TaskKind::Extract { fields, regions, pre } = &mut task.kind else { continue };
            for field in fields {
                if !matches!(field.kind, Kind::String | Kind::Number | Kind::Integer) {
                    continue;
                }
                let choice = chosen(answer(out, &format!("{}{SEP}f{SEP}{}", task.name, field.key))?)?;
                if choice == "__missing__" {
                    continue;
                }
                let region = choice
                    .strip_prefix('L')
                    .and_then(|s| s.parse::<usize>().ok())
                    .and_then(|i| regions.get(i))
                    .context("invalid extract line selection")?;
                field.candidates = self.text.candidates(*region, max_options - 1);
                if field.candidates.len() <= 1 {
                    continue;
                }
                let mut criteria: Map<String, Value> = field
                    .candidates
                    .iter()
                    .enumerate()
                    .map(|(i, s)| (format!("S{i}"), self.text.slice(*s).into()))
                    .collect();
                criteria.insert("__missing__".into(), word(task.lang, "missing").into());
                let instructions = pre.clone()
                    + &render(
                        task.lang,
                        "span",
                        &[("line", &text::shown(self.text.slice(*region))), ("field", &field.description())],
                    );
                questions.insert(
                    format!("{}{SEP}s{SEP}{}", task.name, field.key),
                    json!({"type":"choice","criteria":criteria,"option_keys":"hide","instructions":instructions}),
                );
            }
        }
        if questions.is_empty() {
            return Ok(None);
        }
        let mut body = self.continuation.clone();
        body.insert("questions".into(), questions.into());
        Ok(Some(body.into()))
    }

    pub fn finish(&self, mut out: Value, second: Option<&Value>, cal: &Map<String, Value>) -> Result<Value> {
        for task in &self.tasks {
            let prefix = format!("{}{SEP}", task.name);
            let part = |suffix: &str| answer(&out, &(prefix.clone() + suffix));
            let mut result = match &task.kind {
                TaskKind::Classify { multi, risks } => {
                    let a = part("labels")?;
                    if *multi {
                        json!({"type":"classify","labels":a["selected"],"scores":a["probabilities"],"set_confidence":a["set_confidence"]})
                    } else {
                        let confidence = probability(&a["confidence"])?;
                        json!({"type":"classify","label":a["choice"],"top_label":a["choice"],"scores":a["probabilities"],"confidence":confidence,"automated":automated(cal,"classify",confidence,risks)})
                    }
                }
                TaskKind::Verify { keys, binary, risks } => {
                    let mut claims = Map::new();
                    let mut flagged = Vec::new();
                    for key in keys {
                        let a = part(&format!("c{SEP}{key}"))?;
                        let (pr, verdict) = if *binary {
                            let p = probability(&a["noul"])?;
                            (
                                json!({"supported":p,"not_supported":1.0-p}),
                                if p >= 0.5 { "supported" } else { "not_supported" },
                            )
                        } else {
                            (a["probabilities"].clone(), chosen(a)?)
                        };
                        let conf = pr
                            .as_object()
                            .context("claim probabilities")?
                            .values()
                            .map(probability)
                            .collect::<Result<Vec<_>>>()?
                            .into_iter()
                            .fold(0.0, f64::max);
                        claims.insert(key.clone(),json!({"verdict":verdict,"probabilities":pr,"confidence":conf,"automated":automated(cal,"verify",conf,risks)}));
                        if verdict != "supported" {
                            flagged.push(key.clone());
                        }
                    }
                    json!({"type":"verify","claims":claims,"all_supported":flagged.is_empty(),"not_supported":flagged})
                }
                TaskKind::Compress { spans, unit, keep_ratio, threshold, min_keep } => {
                    let probabilities = (0..spans.len())
                        .map(|i| probability(&part(&format!("u{SEP}{i}"))?["noul"]))
                        .collect::<Result<Vec<_>>>()?;
                    let mut order: Vec<_> = (0..spans.len()).collect();
                    order.sort_by(|&i, &j| probabilities[j].total_cmp(&probabilities[i]));
                    let mut keep = vec![false; spans.len()];
                    if let Some(ratio) = keep_ratio {
                        let total = spans.iter().map(|s| s.end - s.start).sum::<usize>();
                        let mut n = 0;
                        for &i in &order {
                            if n as f64 >= ratio * total as f64 {
                                break;
                            }
                            keep[i] = true;
                            n += spans[i].end - spans[i].start;
                        }
                    } else {
                        for (i, &p) in probabilities.iter().enumerate() {
                            keep[i] = p >= *threshold;
                        }
                    }
                    for &i in order.iter().take(*min_keep) {
                        keep[i] = true;
                    }
                    let kept: Vec<_> = keep.iter().enumerate().filter_map(|(i, k)| k.then_some(i)).collect();
                    let compressed = self.text.join(&kept, spans);
                    let compressed_chars = compressed.chars().count();
                    let units: Vec<_> = spans
                        .iter()
                        .zip(&probabilities)
                        .map(|(s, p)| json!({"start":s.start,"end":s.end,"probability":round(*p,6)}))
                        .collect();
                    json!({"type":"compress","compressed":compressed,"kept":kept,"units":units,"unit":unit,"original_chars":self.text.len(),"compressed_chars":compressed_chars,"ratio":round(compressed_chars as f64/self.text.len().max(1) as f64,4)})
                }
                TaskKind::Extract { fields, .. } => {
                    let mut data = json!({});
                    let mut results = Map::new();
                    for field in fields {
                        let a = part(&format!("f{SEP}{}", field.key))?;
                        let mut f = match field.kind {
                            Kind::Enum | Kind::EnumList => {
                                let multi = matches!(field.kind, Kind::EnumList);
                                let vals = schema::consts(if multi { &field.spec["items"] } else { &field.spec })
                                    .context("enum values")?;
                                let value = |key: &str| -> Result<Value> {
                                    if key == "__missing__" {
                                        return Ok(Value::Null);
                                    }
                                    vals.iter()
                                        .find(|(v, _)| py_string(v) == key)
                                        .map(|(v, _)| v.clone())
                                        .context("invalid extract enum choice")
                                };
                                let result = if multi {
                                    Value::Array(
                                        a["selected"]
                                            .as_array()
                                            .context("enum list selection")?
                                            .iter()
                                            .map(|v| value(v.as_str().context("enum key")?))
                                            .collect::<Result<Vec<_>>>()?,
                                    )
                                } else {
                                    value(chosen(a)?)?
                                };
                                json!({"value":result,"probabilities":a["probabilities"],"confidence":a[if multi {"set_confidence"} else {"confidence"}],"source":"choice"})
                            }
                            Kind::Boolean => {
                                let c = chosen(a)?;
                                json!({"value":if c=="__missing__" {Value::Null} else {Value::Bool(c=="true")},"probabilities":a["probabilities"],"confidence":a["confidence"],"source":"choice"})
                            }
                            _ => {
                                let missing = probability(&a["probabilities"]["__missing__"])?;
                                let mut f = json!({"value":null,"present_probability":1.0-missing,"source":"span"});
                                if chosen(a)? == "__missing__" || field.candidates.is_empty() {
                                    f["confidence"] = missing.into();
                                } else {
                                    let next = second.and_then(|s| {
                                        s["answers"].get(format!("{}{SEP}s{SEP}{}", task.name, field.key))
                                    });
                                    let (candidate, span_p) = match next {
                                        None => (Some(0), 1.0),
                                        Some(b) if chosen(b)? == "__missing__" => {
                                            (None, probability(&b["probabilities"]["__missing__"])?)
                                        }
                                        Some(b) => (
                                            Some(
                                                chosen(b)?
                                                    .strip_prefix('S')
                                                    .and_then(|s| s.parse::<usize>().ok())
                                                    .context("invalid span choice")?,
                                            ),
                                            probability(&b["confidence"])?,
                                        ),
                                    };
                                    if let Some(i) = candidate {
                                        let s = field.candidates.get(i).context("unknown span candidate")?;
                                        let text = self.text.slice(*s);
                                        let line_p = probability(&a["confidence"])?;
                                        f["text"] = text.into();
                                        f["start"] = s.start.into();
                                        f["end"] = s.end.into();
                                        f["line_probability"] = line_p.into();
                                        f["span_probability"] = span_p.into();
                                        f["confidence"] = (line_p * span_p).into();
                                        f["value"] = if matches!(field.kind, Kind::String) {
                                            text.into()
                                        } else {
                                            text::number(text, matches!(field.kind, Kind::Integer))
                                                .unwrap_or(Value::Null)
                                        };
                                        if f["value"].is_null() {
                                            f["error"] = format!(
                                                "not a {}",
                                                if matches!(field.kind, Kind::Integer) { "integer" } else { "number" }
                                            )
                                            .into();
                                        }
                                    } else {
                                        f["confidence"] = span_p.into();
                                    }
                                }
                                f
                            }
                        };
                        f["calibration"] = if f["source"] == "choice" { "validated" } else { "per decision" }.into();
                        schema::set_path(&mut data, &field.path, f["value"].clone());
                        results.insert(field.key.clone(), f);
                    }
                    json!({"type":"extract","data":data,"fields":results})
                }
            };
            result["experimental"] = true.into();
            let answers = out["answers"].as_object_mut().context("answers object")?;
            answers.retain(|k, _| !k.starts_with(&prefix));
            answers.insert(task.name.clone(), result);
        }
        out["usage"]["questions"] = self.count.into();
        Ok(out)
    }
}
fn round(value: f64, places: u32) -> f64 {
    format!("{:.*}", places as usize, value).parse().expect("finite rounded decimal invariant")
}
