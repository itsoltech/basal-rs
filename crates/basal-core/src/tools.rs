//! Upstream v2 tool selection. Definitions become decision options; this module never invokes a tool.

use serde::Serialize;
use serde_json::{json, Value};

use crate::prompt::Lang;
use crate::request::{qerr, FieldError, MultiSpec, QType};

const EXTRA: [&str; 3] = ["answer_directly", "ask_user", "none"];

#[derive(Clone, Debug, Serialize)]
pub struct ToolSpec {
    pub names: Vec<String>,
    pub rank: Option<usize>,
    pub max_error: Option<String>,
    pub threshold: Option<f64>,
    pub multi: Option<MultiSpec>,
}

pub struct ParsedTool {
    pub keys: Vec<String>,
    pub options: Vec<String>,
    pub question: String,
    pub spec: ToolSpec,
}

fn extra_text(key: &str, lang: Lang) -> &'static str {
    match (key, lang) {
        ("answer_directly", Lang::Pl) => "Odpowiedz bezpośrednio, bez wywoływania narzędzia",
        ("answer_directly", _) => "Answer directly without calling a tool",
        ("ask_user", Lang::Pl) => "Dopytaj użytkownika o brakujące informacje",
        ("ask_user", _) => "Ask the user for the missing information",
        (_, Lang::Pl) => "Brak odpowiedniego narzędzia — poinformuj użytkownika",
        _ => "No suitable tool — tell the user",
    }
}

pub fn parse(q: &serde_json::Map<String, Value>, lang: Lang, name: &str) -> Result<ParsedTool, Vec<FieldError>> {
    let error = |field: &str, message: String| vec![qerr(name, &[field], message, "value_error")];
    if q.contains_key("criteria") {
        return Err(error("criteria", "tool takes tools, not criteria".into()));
    }
    let definitions = q
        .get("tools")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| error("tools", "tools must be a non-empty list of definitions".into()))?;
    let is_multi = match q.get("multi") {
        None => false,
        Some(Value::Bool(v)) => *v,
        Some(_) => return Err(error("multi", "multi must be true or false".into())),
    };
    if !is_multi && definitions.len() > crate::prompt::PromptFormat::Two.max_options() {
        return Err(error("tools", "tool exceeds the 255-option format limit".into()));
    }
    let cap = match q.get("description_chars") {
        None => 300,
        Some(v) => v
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| error("description_chars", "description_chars must be a non-negative integer".into()))?,
    };
    let mut names = Vec::with_capacity(definitions.len());
    let mut seen = std::collections::HashSet::with_capacity(definitions.len());
    let mut options = Vec::with_capacity(definitions.len() + EXTRA.len());
    for (i, definition) in definitions.iter().enumerate() {
        let field = format!("tools[{i}]");
        let d = definition.as_object().ok_or_else(|| error(&field, "tool definition must be an object".into()))?;
        if d.get("type").is_some_and(|v| v.as_str() != Some("function")) {
            return Err(error(&field, "function tools only".into()));
        }
        let f = d
            .get("function")
            .unwrap_or(definition)
            .as_object()
            .ok_or_else(|| error(&field, "function must be an object".into()))?;
        let tool = f
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && !s.chars().any(python_whitespace))
            .ok_or_else(|| error(&field, "tool name must be non-empty without whitespace".into()))?;
        if !seen.insert(tool) {
            return Err(error(&field, format!("duplicate tool name {tool:?}")));
        }
        let description = match f.get("description") {
            None | Some(Value::Null) => "",
            Some(Value::String(s)) => s,
            Some(_) => return Err(error(&field, "description must be a string".into())),
        };
        let schema = ["parameters", "inputSchema", "input_schema"].iter().find_map(|key| f.get(*key));
        let mut parameters = Vec::new();
        if let Some(schema) = schema.filter(|v| !v.is_null()) {
            let s = schema.as_object().ok_or_else(|| error(&field, "parameter schema must be an object".into()))?;
            if s.get("type").is_some_and(|v| v.as_str() != Some("object")) {
                return Err(error(&field, "parameter schema type must be object".into()));
            }
            let empty = serde_json::Map::new();
            let properties = match s.get("properties") {
                None => &empty,
                Some(v) => v.as_object().ok_or_else(|| error(&field, "properties must be an object".into()))?,
            };
            if properties.values().any(|v| !v.is_object() && !v.is_boolean()) {
                return Err(error(&field, "properties must contain schemas".into()));
            }
            let mut required: Vec<&str> = Vec::new();
            if let Some(value) = s.get("required") {
                for value in value.as_array().ok_or_else(|| error(&field, "required must be a list".into()))? {
                    let key = value.as_str().ok_or_else(|| error(&field, "required names must be strings".into()))?;
                    if required.contains(&key) || !properties.contains_key(key) {
                        return Err(error(&field, "required must name distinct properties".into()));
                    }
                    required.push(key);
                }
            }
            parameters = properties
                .keys()
                .map(|key| if required.contains(&key.as_str()) { key.clone() } else { format!("{key}?") })
                .collect();
        }
        let mut desc = description.split(python_whitespace).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = desc.chars().collect();
        if cap > 0 && chars.len() > cap {
            let mut head: String = chars[..cap].iter().collect();
            if chars[cap] != ' ' {
                if let Some(pos) = head.rfind(' ') {
                    head.truncate(pos);
                }
            }
            desc = format!("{}…", head.trim_end_matches([' ', ',', ';', ':']));
        }
        let mut text = format!("{tool}({})", parameters.join(", "));
        if !desc.is_empty() {
            text.push_str(" — ");
            text.push_str(&desc);
        }
        names.push(tool.to_string());
        options.push(text);
    }
    let defaults = if is_multi { Vec::new() } else { EXTRA.iter().map(|s| Value::String((*s).into())).collect() };
    let extras = match q.get("extra") {
        None => &defaults,
        Some(v) => v.as_array().ok_or_else(|| error("extra", "extra must be a list".into()))?,
    };
    let mut keys = names.clone();
    for extra in extras {
        let key = extra
            .as_str()
            .filter(|s| EXTRA.contains(s))
            .ok_or_else(|| error("extra", "unknown extra option".into()))?;
        if keys.iter().any(|k| k == key) {
            return Err(error("extra", format!("duplicate or clashing extra {key:?}")));
        }
        keys.push(key.into());
        options.push(extra_text(key, lang).into());
    }
    let rank = match q.get("rank").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|v| (1..=keys.len()).contains(v))
                .ok_or_else(|| error("rank", "rank must be an integer in 1..option count".into()))?,
        ),
    };
    let max_error = match q.get("max_error").filter(|v| !v.is_null()) {
        None => None,
        Some(v) if !is_multi => Some(crate::pyjson::float_repr(
            v.as_f64()
                .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
                .ok_or_else(|| error("max_error", "max_error must be a number between 0 and 1".into()))?,
        )),
        Some(_) => return Err(error("max_error", "max_error is not supported with multi".into())),
    };
    let question = q.get("instructions").map(crate::pyjson::text).filter(|s| !s.is_empty()).unwrap_or_else(|| {
        match (is_multi, lang) {
            (true, Lang::Pl) => "Które narzędzia asystent powinien teraz wywołać?",
            (true, Lang::En) => "Which tools should the assistant call now?",
            (false, Lang::Pl) => "Co asystent powinien teraz zrobić: które narzędzie wywołać?",
            (false, Lang::En) => "What should the assistant do next: which tool should it call?",
        }
        .into()
    });
    let multi = if is_multi {
        let mut m = q.clone();
        m.insert(
            "criteria".into(),
            Value::Object(keys.iter().cloned().zip(options.iter().cloned().map(Value::String)).collect()),
        );
        let mut exclusive: Vec<Value> =
            keys.iter().filter(|k| !names.contains(k)).cloned().map(Value::String).collect();
        if let Some(value) = q.get("exclusive").filter(|v| !v.is_null()) {
            exclusive.extend(
                value.as_array().ok_or_else(|| error("exclusive", "exclusive must be a list".into()))?.iter().cloned(),
            );
        }
        m.insert("exclusive".into(), Value::Array(exclusive));
        let (parts, spec) = crate::request::multi(&m, lang, name)?;
        options = parts.1;
        Some(spec)
    } else {
        None
    };
    Ok(ParsedTool { keys, options, question, spec: ToolSpec { names, rank, max_error, threshold: None, multi } })
}

fn ranking(keys: &[String], p: &[f64], n: usize) -> Value {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));
    json!(order.iter().take(n).map(|&i| json!({"tool": keys[i], "probability": p[i]})).collect::<Vec<_>>())
}

pub fn answer(item: &crate::request::Item, spec: &ToolSpec, p: &[f64]) -> Value {
    if let Some(multi) = &spec.multi {
        let mut value = crate::decision::multi_answer(item, multi, p);
        let selected = value.as_object_mut().and_then(|m| m.remove("selected")).unwrap_or_else(|| json!([]));
        let call = selected
            .as_array()
            .is_some_and(|xs| xs.iter().any(|v| v.as_str().is_some_and(|s| spec.names.iter().any(|n| n == s))));
        value["type"] = "tool".into();
        value["multi"] = true.into();
        value["tools"] = selected;
        value["call"] = call.into();
        if let Some(n) = spec.rank {
            value["ranking"] = ranking(&item.keys, p, n);
        }
        return value;
    }
    let top = crate::decision::argmax(p);
    let key = &item.keys[top];
    let probabilities: serde_json::Map<_, _> =
        item.keys.iter().cloned().zip(p.iter().map(|p| crate::decision::num(*p))).collect();
    let mut value = json!({"type": "tool", "tool": key, "call": spec.names.contains(key), "probabilities": probabilities, "confidence": p[top]});
    if let Some(n) = spec.rank {
        value["ranking"] = ranking(&item.keys, p, n);
    }
    if spec.max_error.is_some() {
        let deferred = spec.threshold.is_none_or(|t| p[top] < t);
        value["deferred"] = deferred.into();
        if spec.threshold.is_none() {
            value["max_error"] = "no certified threshold for this type at this max_error".into();
        }
        if deferred {
            value["tool"] = Value::Null;
            value["call"] = false.into();
            value["refused"] = key.clone().into();
        }
    }
    value
}

pub fn calibration_kind(spec: &ToolSpec) -> QType {
    if spec.multi.is_some() {
        QType::Noul
    } else {
        QType::Choice
    }
}

fn python_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
