//! Bound host-side prompt expansion before cloning shared state into questions and branches.

use serde_json::Value;

use crate::request::{DecideError, FieldError};

pub(crate) const MAX_TEXT_BYTES: usize = 32 * 1024 * 1024;
const MAX_BRANCHES: usize = 4096;

/// Conservative rendered text budget for both potential option orders. JSON media bytes are admitted separately.
pub(crate) fn prepared_input_bytes(body: &Value) -> Result<usize, DecideError> {
    let state = text_bytes(&body["state"]);
    let Some(questions) = body.get("questions").and_then(Value::as_object) else {
        return Ok(0);
    };
    let mut bytes = 0usize;
    let mut branches = 0usize;
    for q in questions.values() {
        let multi = q.get("type").and_then(Value::as_str) == Some("multi")
            || (q.get("type").and_then(Value::as_str) == Some("tool") && crate::request::truthy(q.get("multi")));
        let labels = match q.get("type").and_then(Value::as_str) {
            Some("tool") => q.get("tools").and_then(Value::as_array).map_or(1, |v| v.len()).saturating_add(3),
            _ => match q.get("criteria") {
                Some(Value::Array(v)) => v.len(),
                Some(Value::Object(v)) => v.len(),
                _ => 1,
            },
        };
        let count = if multi { labels.max(1) } else { 1 };
        branches = branches.saturating_add(count);
        let shared = state.saturating_add(text_bytes(&q["instructions"])).saturating_add(512);
        // Include both score fields conservatively: `levels` is an accepted alias of `criteria`.
        let options = text_bytes(&q["criteria"])
            .saturating_add(q.get("levels").map_or(0, text_bytes))
            .saturating_add(text_bytes(&q["tools"]))
            .saturating_add(text_bytes(&q["costs"]));
        bytes = bytes.saturating_add(shared.saturating_mul(count).saturating_add(options).saturating_mul(2));
        if branches > MAX_BRANCHES || bytes > MAX_TEXT_BYTES {
            return Err(DecideError::Unsupported(vec![FieldError::new(
                vec!["body".into(), "questions".into()],
                "request exceeds the host prompt budget (4096 branches or 32 MiB of expanded text for two orders)",
                "request_size_exceeded",
            )]));
        }
    }
    Ok(bytes)
}

fn string_bytes(s: &str) -> usize {
    let escapes = s.bytes().filter(|c| *c < 0x20 || *c == b'"' || *c == b'\\').count();
    s.len().saturating_add(escapes.saturating_mul(5)).saturating_add(2)
}

/// Upper bound for the UTF-8 text/JSON conversion, including keys and escaped control characters.
fn text_bytes(v: &Value) -> usize {
    match v {
        Value::String(s) => string_bytes(s),
        Value::Array(v) => v.iter().fold(2usize, |n, v| n.saturating_add(text_bytes(v)).saturating_add(2)),
        Value::Object(v) => v.iter().fold(2usize, |n, (k, v)| {
            n.saturating_add(string_bytes(k)).saturating_add(text_bytes(v)).saturating_add(4)
        }),
        _ => 32,
    }
}
