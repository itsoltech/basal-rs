//! OpenAI Decisions API (`POST /v1/decisions`), translated to a TypeSafe System One request and its answers back.
//! The translated request runs the same admission, planning and model path as `/v1/systemone`. Mapping and limits:
//! docs/OPENAI_DECISIONS.md.

use std::collections::HashSet;

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use basal_core::DecideError;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::systemone::valid_json_value;

const MAX_QUESTIONS: usize = 400;
const MAX_CHOICES: usize = 255;
const MAX_LEVELS: usize = 10;
const TOP_LEVEL: &[&str] =
    &["model", "input", "questions", "prompt_cache_key", "prompt_cache_options", "safety_identifier"];
const USER_PARTS: &[&str] = &["input_text", "input_image", "input_json"];
const TEXT_PARTS: &[&str] = &["input_text"];
const ASSISTANT_PARTS: &[&str] = &["output_text", "refusal"];
const OUTPUT_PARTS: &[&str] = &["input_text", "input_image"];

/// An OpenAI error answer: `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug)]
pub(super) struct Error {
    status: StatusCode,
    kind: &'static str,
    message: String,
    param: Option<String>,
    code: Option<&'static str>,
}

impl Error {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), param: None, code: None }
    }

    fn invalid(param: impl Into<String>, message: impl Into<String>) -> Self {
        Self { param: Some(param.into()), ..Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message) }
    }

    fn code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }

    pub(super) fn json(e: &serde_json::Error) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("We could not parse the JSON body of your request: {e}"),
        )
    }

    /// A body that could not be read (too large, or a transport error).
    pub(super) fn body(status: StatusCode, message: String) -> Self {
        Self::new(status, "invalid_request_error", message)
    }

    pub(super) fn unknown_model(model: &str) -> Self {
        Self::new(StatusCode::NOT_FOUND, "invalid_request_error", format!("The model `{model}` does not exist."))
            .code("model_not_found")
    }

    /// 503 with `Retry-After`; OpenAI SDKs retry it with backoff.
    pub(super) fn overloaded(message: &str) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "server_error", message).code("server_overloaded")
    }

    pub(super) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "server_error", message)
    }

    /// Engine errors of the translated request. Question IDs `q{i}` are reported as `questions[i]`.
    pub(super) fn decide(e: DecideError) -> Self {
        let message = e.to_upstream_json()["error"].as_str().unwrap_or_default().to_string();
        let fields = match &e {
            DecideError::Validation(fields) | DecideError::Unsupported(fields) => fields.as_slice(),
            DecideError::Usage(_) => &[],
            DecideError::Internal(_) => return Self::internal(message),
        };
        let mut error = Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message);
        error.param = fields.first().and_then(|f| param(&f.loc));
        if matches!(e, DecideError::Unsupported(_)) {
            error.code = Some(if error.message.contains("maximum context length") {
                "context_length_exceeded"
            } else {
                "unsupported_value"
            });
        }
        error
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let body =
            json!({"error": {"message": self.message, "type": self.kind, "param": self.param, "code": self.code}});
        let mut response = (self.status, Json(body)).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            response.headers_mut().insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// OpenAI parameter of a TypeSafe error location of the translated body.
fn param(loc: &[Value]) -> Option<String> {
    match (loc.get(1).and_then(Value::as_str), loc.get(2).and_then(Value::as_str)) {
        (Some("questions"), Some(id)) => id.strip_prefix('q').map(|i| format!("questions[{i}]")),
        (Some("state" | "assets"), _) => Some("input".into()),
        _ => None,
    }
}

/// The TypeSafe body of one request and how to read its answers back.
pub(super) struct Translated {
    pub(super) body: Value,
    pub(super) model: String,
    pub(super) answers: Answers,
}

/// Question names and answer keys in request order; the TypeSafe question of index `i` is `q{i}`.
pub(super) struct Answers(Vec<(Value, Kind)>);

enum Kind {
    Predicate,
    /// Typed choice values; the TypeSafe key of each is its text.
    Choice(Vec<Value>),
    /// Level labels, lowest first.
    Score(Vec<String>),
}

/// Base64 data-URL images as TypeSafe assets. A reference is derived from the image data, so equal images share one
/// asset and a request translates to the same body every time.
#[derive(Default)]
struct Images(Map<String, Value>);

impl Images {
    fn reference(&mut self, url: &str, param: &str) -> Result<String, Error> {
        let Some(rest) = url.strip_prefix("data:") else {
            let error = Error::invalid(param, "Only base64 data URLs are supported for images; URLs are not fetched.");
            return Err(error.code("unsupported_value"));
        };
        let (meta, data) = rest.split_once(',').ok_or_else(|| Error::invalid(param, "Invalid image data URL."))?;
        let format = match meta.to_ascii_lowercase().as_str() {
            "image/png;base64" => "png",
            "image/jpeg;base64" | "image/jpg;base64" => "jpg",
            _ => return Err(Error::invalid(param, "Images must be base64 PNG or JPEG data URLs.")),
        };
        let digest = Sha256::digest(data.as_bytes());
        let id: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        let key = format!("[image:{id}]");
        self.0.entry(key.clone()).or_insert_with(|| json!({"type": "image", "format": format, "data_base64": data}));
        Ok(key)
    }
}

fn object(v: Value, param: &str) -> Result<Map<String, Value>, Error> {
    match v {
        Value::Object(m) => Ok(m),
        _ => Err(Error::invalid(param, format!("`{param}` must be an object."))),
    }
}

fn string(m: &mut Map<String, Value>, key: &str, param: &str) -> Result<String, Error> {
    match m.remove(key) {
        Some(Value::String(s)) => Ok(s),
        _ => Err(Error::invalid(format!("{param}.{key}"), format!("`{param}.{key}` must be a string."))),
    }
}

/// An optional string field; `null` is the same as a missing field.
fn optional_string(m: &mut Map<String, Value>, key: &str, param: &str) -> Result<Option<String>, Error> {
    match m.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        _ => Err(Error::invalid(format!("{param}.{key}"), format!("`{param}.{key}` must be a string or null."))),
    }
}

/// One content part as a state value: text as a string, JSON data as itself, an image as its asset reference.
fn part(v: Value, param: &str, allowed: &[&str], images: &mut Images) -> Result<Value, Error> {
    let mut p = object(v, param)?;
    let kind = p.get("type").and_then(Value::as_str).unwrap_or_default().to_string();
    if !allowed.contains(&kind.as_str()) {
        return Err(Error::invalid(
            format!("{param}.type"),
            format!("Content part type must be one of: {}.", allowed.join(", ")),
        ));
    }
    Ok(match kind.as_str() {
        "input_text" | "output_text" => string(&mut p, "text", param)?.into(),
        "refusal" => json!({"refusal": string(&mut p, "refusal", param)?}),
        "input_json" => match p.remove("data") {
            Some(data @ (Value::Object(_) | Value::Array(_))) => data,
            _ => return Err(Error::invalid(format!("{param}.data"), "`data` must be an object or an array.")),
        },
        _ => {
            let detail = optional_string(&mut p, "detail", param)?;
            if detail.is_some_and(|d| !matches!(d.as_str(), "low" | "high" | "auto" | "original")) {
                return Err(Error::invalid(format!("{param}.detail"), "`detail` must be low, high, auto or original."));
            }
            images.reference(&string(&mut p, "image_url", param)?, &format!("{param}.image_url"))?.into()
        }
    })
}

/// Message content or question instructions: a string, or a non-empty part array. One text part reads as a string.
fn content(v: Option<Value>, param: &str, allowed: &[&str], images: &mut Images) -> Result<Value, Error> {
    match v {
        Some(Value::String(s)) => Ok(s.into()),
        Some(Value::Array(parts)) if !parts.is_empty() => {
            let mut values = parts
                .into_iter()
                .enumerate()
                .map(|(i, p)| part(p, &format!("{param}[{i}]"), allowed, images))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(if values.len() == 1 && values[0].is_string() { values.remove(0) } else { values.into() })
        }
        _ => Err(Error::invalid(param, format!("`{param}` must be a string or a non-empty array of content parts."))),
    }
}

/// Tool calls seen so far, by `call_id`, with their item type; outputs must answer one of them.
type Calls = HashSet<(String, String)>;

/// One `input` item as a state value. Item ids, statuses, phases, image detail and cache breakpoints are dropped.
fn item(v: Value, param: &str, calls: &mut Calls, images: &mut Images) -> Result<Value, Error> {
    let mut m = object(v, param)?;
    let kind = optional_string(&mut m, "type", param)?.unwrap_or_else(|| "message".into());
    match kind.as_str() {
        "message" => {
            let role = string(&mut m, "role", param)?;
            let allowed = match role.as_str() {
                "user" => USER_PARTS,
                "developer" | "system" => TEXT_PARTS,
                "assistant" => ASSISTANT_PARTS,
                _ => {
                    return Err(Error::invalid(
                        format!("{param}.role"),
                        "`role` must be user, developer, system or assistant.",
                    ))
                }
            };
            // System messages are developer messages in the Decisions API.
            let role = if role == "system" { "developer".into() } else { role };
            Ok(
                json!({"role": role, "content": content(m.remove("content"), &format!("{param}.content"), allowed, images)?}),
            )
        }
        "function_call" | "custom_tool_call" => {
            let field = if kind == "function_call" { "arguments" } else { "input" };
            let call_id = string(&mut m, "call_id", param)?;
            let mut call = json!({"type": kind.as_str(), "name": string(&mut m, "name", param)?});
            if let Some(namespace) = optional_string(&mut m, "namespace", param)? {
                call["namespace"] = namespace.into();
            }
            call[field] = string(&mut m, field, param)?.into();
            call["call_id"] = call_id.clone().into();
            calls.insert((kind, call_id));
            Ok(call)
        }
        "function_call_output" | "custom_tool_call_output" => {
            let call_id = string(&mut m, "call_id", param)?;
            let call = kind.trim_end_matches("_output");
            if !calls.contains(&(call.to_string(), call_id.clone())) {
                return Err(Error::invalid(
                    format!("{param}.call_id"),
                    format!("No preceding {call} with call_id `{call_id}`."),
                ));
            }
            let output = content(m.remove("output"), &format!("{param}.output"), OUTPUT_PARTS, images)?;
            Ok(json!({"type": kind, "call_id": call_id, "output": output}))
        }
        _ => Err(Error::invalid(format!("{param}.type"), format!("Unsupported input item type `{kind}`."))),
    }
}

/// One question as a TypeSafe question and the key of its answer.
fn question(v: Value, param: &str, images: &mut Images) -> Result<(Value, (Value, Kind)), Error> {
    let mut q = object(v, param)?;
    let kind = string(&mut q, "type", param)?;
    let name = optional_string(&mut q, "name", param)?.map_or(Value::Null, Value::String);
    let instructions = content(q.remove("instructions"), &format!("{param}.instructions"), USER_PARTS, images)?;
    let (question, answer) = match kind.as_str() {
        "predicate" => (json!({"type": "noul", "instructions": instructions}), Kind::Predicate),
        "choice" => {
            let choices = options(q.remove("choices"), &format!("{param}.choices"), 2, MAX_CHOICES)?;
            let mut criteria = Map::new();
            let mut values = Vec::with_capacity(choices.len());
            for (i, choice) in choices.into_iter().enumerate() {
                let param = format!("{param}.choices[{i}]");
                let mut choice = object(choice, &param)?;
                let value = choice.remove("value").unwrap_or_default();
                let key = match &value {
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => b.to_string(),
                    _ => {
                        return Err(Error::invalid(format!("{param}.value"), "`value` must be a string or a boolean."))
                    }
                };
                let description = optional_string(&mut choice, "description", &param)?;
                if criteria.insert(key, description.map_or(Value::Null, Value::String)).is_some() {
                    return Err(Error::invalid(
                        format!("{param}.value"),
                        "Choice values must be unique; this runtime also requires distinct text (`true` and \"true\").",
                    ));
                }
                values.push(value);
            }
            (json!({"type": "choice", "instructions": instructions, "criteria": criteria}), Kind::Choice(values))
        }
        "score" => {
            let levels = options(q.remove("levels"), &format!("{param}.levels"), 2, MAX_LEVELS)?;
            let mut criteria = Map::new();
            let mut labels = Vec::with_capacity(levels.len());
            for (i, level) in levels.into_iter().enumerate() {
                let param = format!("{param}.levels[{i}]");
                let mut level = object(level, &param)?;
                let label = string(&mut level, "label", &param)?;
                let description = optional_string(&mut level, "description", &param)?;
                if criteria.insert(label.clone(), description.map_or(Value::Null, Value::String)).is_some() {
                    return Err(Error::invalid(format!("{param}.label"), "Level labels must be unique."));
                }
                labels.push(label);
            }
            (json!({"type": "score", "instructions": instructions, "criteria": criteria}), Kind::Score(labels))
        }
        _ => return Err(Error::invalid(format!("{param}.type"), "`type` must be predicate, choice or score.")),
    };
    Ok((question, (name, answer)))
}

fn options(v: Option<Value>, param: &str, min: usize, max: usize) -> Result<Vec<Value>, Error> {
    match v {
        Some(Value::Array(xs)) if (min..=max).contains(&xs.len()) => Ok(xs),
        _ => Err(Error::invalid(param, format!("`{param}` must be an array of {min} to {max} items."))),
    }
}

/// Validate an OpenAI Decisions request and build its TypeSafe body. Cache and safety fields are checked and ignored.
pub(super) fn translate(body: Value) -> Result<Translated, Error> {
    let Value::Object(mut body) = body else {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "The request body must be a JSON object.",
        ));
    };
    if let Some(key) = body.keys().find(|k| !TOP_LEVEL.contains(&k.as_str())) {
        return Err(Error::invalid(key.clone(), format!("Unrecognized request argument supplied: {key}")));
    }
    for (key, max) in [("prompt_cache_key", 64), ("safety_identifier", 128)] {
        if body.get(key).is_some_and(|v| !v.is_null() && v.as_str().is_none_or(|s| s.chars().count() > max)) {
            return Err(Error::invalid(key, format!("`{key}` must be a string of at most {max} characters.")));
        }
    }
    if !matches!(body.get("prompt_cache_options"), None | Some(Value::Null | Value::Object(_))) {
        return Err(Error::invalid("prompt_cache_options", "`prompt_cache_options` must be an object or null."));
    }
    let Some(Value::String(model)) = body.remove("model") else {
        return Err(Error::invalid("model", "`model` must be a string."));
    };
    let mut images = Images::default();
    let state = match body.remove("input") {
        Some(Value::String(s)) => Value::String(s),
        Some(Value::Array(items)) if !items.is_empty() => {
            let mut calls = Calls::new();
            items
                .into_iter()
                .enumerate()
                .map(|(i, v)| item(v, &format!("input[{i}]"), &mut calls, &mut images))
                .collect::<Result<Vec<_>, _>>()?
                .into()
        }
        _ => return Err(Error::invalid("input", "`input` must be a string or a non-empty array of input items.")),
    };
    valid_json_value(&state, 0).map_err(|e| Error::invalid("input", e))?;
    let questions = options(body.remove("questions"), "questions", 1, MAX_QUESTIONS)?;
    let mut typesafe = Map::new();
    let mut answers = Vec::with_capacity(questions.len());
    for (i, q) in questions.into_iter().enumerate() {
        let param = format!("questions[{i}]");
        let (question, answer) = question(q, &param, &mut images)?;
        valid_json_value(&question["instructions"], 0)
            .map_err(|e| Error::invalid(format!("{param}.instructions"), e))?;
        typesafe.insert(format!("q{i}"), question);
        answers.push(answer);
    }
    let mut out = json!({"model": model, "state": state, "questions": typesafe});
    if !images.0.is_empty() {
        out["assets"] = images.0.into();
    }
    Ok(Translated { body: out, model, answers: Answers(answers) })
}

impl Answers {
    /// The `Decision` object of a TypeSafe answer to the translated request.
    pub(super) fn decision(self, response: &Value, model: String) -> Result<Value, Error> {
        let missing = || Error::internal("decision answer is missing a question");
        let mut answers = Vec::with_capacity(self.0.len());
        for (i, (name, kind)) in self.0.into_iter().enumerate() {
            let a = response["answers"].get(format!("q{i}")).ok_or_else(missing)?;
            let probability = |key: &str| a["probabilities"].get(key).cloned().ok_or_else(missing);
            answers.push(match kind {
                Kind::Predicate => json!({"type": "predicate", "name": name, "probability": a["noul"]}),
                Kind::Choice(values) => {
                    let chosen = a["choice"].as_str().ok_or_else(missing)?;
                    let mut choice = Value::Null;
                    let mut probabilities = Vec::with_capacity(values.len());
                    for value in values {
                        let key = value.as_str().map_or_else(|| value.to_string(), str::to_string);
                        if key == chosen {
                            choice = value.clone();
                        }
                        probabilities.push(json!({"value": value, "probability": probability(&key)?}));
                    }
                    json!({"type": "choice", "name": name, "choice": choice, "probabilities": probabilities,
                           "confidence": a["confidence"]})
                }
                Kind::Score(labels) => {
                    let probabilities = labels
                        .iter()
                        .enumerate()
                        .map(|(i, label)| Ok(json!({"value": i, "label": label, "probability": probability(label)?})))
                        .collect::<Result<Vec<_>, Error>>()?;
                    json!({"type": "score", "name": name, "score": a["score"], "probabilities": probabilities,
                           "confidence": a["confidence"]})
                }
            });
        }
        let usage = &response["usage"];
        let input = usage["input_tokens"].as_u64().unwrap_or(0);
        let output = usage["output_tokens"].as_u64().unwrap_or(0);
        Ok(json!({
            "model": model,
            "answers": answers,
            "usage": {
                "input_tokens": input,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": output,
                "output_tokens_details": {"reasoning_tokens": output},
                "total_tokens": input + output,
            },
        }))
    }
}
