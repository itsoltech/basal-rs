//! Construct requests without introducing a second question or calibration contract.

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Map, Value};

use super::{ClientArgs, Failure};
use crate::config;

pub(super) struct Requests {
    template: Option<Value>,
    model: Option<String>,
    state_pointer: Option<String>,
    value: bool,
}

pub(super) fn model_name(name: &str) -> String {
    config::KNOWN_MODELS
        .iter()
        .find(|m| m.short.eq_ignore_ascii_case(name) || m.repo.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_string(), |m| m.repo.rsplit('/').next().unwrap_or(name).to_string())
}

fn criteria(values: &[String]) -> Result<Map<String, Value>> {
    let mut map = Map::new();
    for value in values {
        let (key, description) = match value.split_once('=') {
            Some((key, description)) => (key, Value::String(description.to_string())),
            None => (value.as_str(), Value::Null),
        };
        ensure!(!key.is_empty(), "empty option key");
        ensure!(!map.contains_key(key), "duplicate option key {key:?}");
        map.insert(key.to_string(), description);
    }
    Ok(map)
}

impl Requests {
    pub fn new(args: &ClientArgs) -> Result<Self> {
        let model = match &args.model {
            Some(model) => {
                ensure!(!model.trim().is_empty(), "--model cannot be empty");
                Some(model.clone())
            }
            None => super::environment("BASAL_MODEL")?,
        };
        let template = if let Some(path) = &args.template {
            // Templates are configuration, not streaming data; bound them before allocating their content.
            let file = std::fs::File::open(path).with_context(|| format!("opening template {}", path.display()))?;
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut std::io::Read::take(file, args.max_input_bytes as u64 + 1), &mut bytes)
                .with_context(|| format!("reading template {}", path.display()))?;
            ensure!(bytes.len() <= args.max_input_bytes, "template exceeds --max-input-bytes");
            Some(serde_json::from_slice::<Value>(&bytes).context("parsing request template")?)
        } else if let Some(ask) = &args.ask {
            ensure!(!ask.trim().is_empty(), "--ask cannot be empty");
            let question = if !args.choice.is_empty() {
                json!({"type":"choice", "instructions":ask, "criteria":criteria(&args.choice)?})
            } else if !args.level.is_empty() {
                json!({"type":"score", "instructions":ask, "criteria":args.level})
            } else if !args.label.is_empty() {
                json!({"type":"multi", "instructions":ask, "criteria":criteria(&args.label)?, "threshold":args.threshold})
            } else {
                json!({"type":"noul", "instructions":ask})
            };
            Some(json!({"questions":{"result":question}}))
        } else {
            None
        };
        let requests = Self { template, model, state_pointer: args.state_pointer.clone(), value: args.value };
        if let Some(template) = &requests.template {
            ensure!(template.is_object(), "template must be a request object containing questions");
            let mut probe = template.clone();
            probe["state"] = json!("");
            probe["model"] = json!("client-validation");
            basal_core::request::parse_request(&probe).context("invalid questions in template")?;
            if args.value {
                ensure!(
                    probe["questions"].as_object().is_some_and(|q| q.len() == 1),
                    "--value requires exactly one question"
                );
            }
        }
        Ok(requests)
    }

    /// The request/template name is a fallback; an explicit CLI/environment override wins.
    pub fn configured_model(&self) -> Option<&str> {
        self.model.as_deref().or_else(|| self.template.as_ref()?.get("model")?.as_str())
    }

    /// Builds the request for one input record. With `echo` the input is returned unchanged for the output envelope
    /// and the request gets a copy; without it the input is consumed and its state moves into the request.
    pub fn build(&self, mut input: Value, echo: bool) -> (Option<Value>, Result<Value, Failure>) {
        let body = self.body(&mut input, echo);
        (echo.then_some(input), body)
    }

    fn body(&self, input: &mut Value, echo: bool) -> Result<Value, Failure> {
        let extract = |value: &mut Value| if echo { value.clone() } else { value.take() };
        let mut body = match &self.template {
            Some(template) => {
                let state = match &self.state_pointer {
                    Some(pointer) => input
                        .pointer_mut(pointer)
                        .ok_or_else(|| Failure::message("input", format!("missing state at {pointer:?}")))?,
                    None => input,
                };
                let mut body = template.clone();
                body["state"] = extract(state);
                body
            }
            None => extract(input),
        };
        let Some(object) = body.as_object_mut() else {
            return Err(Failure::message("input", "request must be a JSON object"));
        };
        if self.value && object.get("questions").and_then(Value::as_object).is_none_or(|q| q.len() != 1) {
            return Err(Failure::message("input", "--value requires exactly one question"));
        }
        if let Some(model) = &self.model {
            object.insert("model".into(), json!(model_name(model)));
        } else if let Some(model) = object.get("model").and_then(Value::as_str) {
            object.insert("model".into(), json!(model_name(model)));
        }
        Ok(body)
    }

    pub fn validate(body: &Value) -> Result<(), Failure> {
        basal_core::request::parse_request(body).map(|_| ()).map_err(Failure::decision)
    }

    pub fn validate_pointer(pointer: &str) -> Result<()> {
        ensure!(pointer.is_empty() || pointer.starts_with('/'), "--state-pointer must be a JSON pointer, e.g. /text");
        for segment in pointer.split('/').skip(1) {
            let mut chars = segment.chars();
            while let Some(c) = chars.next() {
                if c == '~' && !matches!(chars.next(), Some('0' | '1')) {
                    bail!("invalid escape in --state-pointer (use ~0 for ~ and ~1 for /)");
                }
            }
        }
        Ok(())
    }
}
