//! TypeSafe image assets. Native `/v1/basal` media do not use this adapter.

use std::io::Cursor;

use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::request::{DecideError, Dialect, FieldError, Item};

pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;

pub(crate) fn unsupported(path: Vec<Value>, message: impl Into<String>) -> DecideError {
    DecideError::Unsupported(vec![FieldError::new(path, message, "unsupported_feature")])
}

/// Recognize contract features before a text-only parser could ignore them.
pub fn check_features(body: &Value) -> Result<(), DecideError> {
    if let Some(questions) = body.get("questions").and_then(Value::as_object) {
        for (name, q) in questions {
            if q.get("type").and_then(Value::as_str) == Some("box") {
                return Err(unsupported(
                    vec!["body".into(), "questions".into(), name.clone().into(), "type".into()],
                    "box questions are not supported by basal-rs models",
                ));
            }
        }
    }
    Ok(())
}

fn invalid(key: &str, message: impl Into<String>) -> DecideError {
    DecideError::Validation(vec![FieldError::new(
        vec!["body".into(), "assets".into(), key.into()],
        message,
        "value_error",
    )])
}

/// Validated wire assets, before model-specific image processing. References are replaced only in JSON values.
pub(crate) struct Assets {
    pub entries: Value,
    pub markers: Vec<String>,
    references: Vec<String>,
    sizes: Vec<usize>,
    matcher: regex::Regex,
}

impl Assets {
    pub fn parse(body: &Value) -> Result<Option<Self>, DecideError> {
        let Some(value) = body.get("assets") else { return Ok(None) };
        let assets = value
            .as_object()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("", "assets must be a non-empty object"))?;
        if body.get("media").is_some_and(|v| !v.is_null()) {
            return Err(unsupported(
                vec!["body".into(), "assets".into()],
                "assets cannot be combined with native media",
            ));
        }
        if assets.len() > crate::media::MAX_ITEMS {
            return Err(unsupported(
                vec!["body".into(), "assets".into()],
                "this runtime supports at most 8 distinct images",
            ));
        }
        if body.get("facts").is_some_and(|v| v != "off" && !v.is_null()) {
            return Err(unsupported(vec!["body".into(), "facts".into()], "facts are not supported with image assets"));
        }
        if let Some(questions) = body.get("questions").and_then(Value::as_object) {
            for (name, q) in questions {
                let kind = q.get("type").and_then(Value::as_str);
                if !matches!(kind, Some("noul" | "choice" | "score"))
                    && (kind.and_then(crate::request::QType::parse).is_some() || crate::tasks::is_task(q))
                {
                    return Err(unsupported(
                        vec!["body".into(), "questions".into(), name.clone().into()],
                        "image assets support choice, noul and score questions",
                    ));
                }
            }
        }
        let mut references = Vec::new();
        let mut entries = Vec::new();
        let mut sizes = Vec::new();
        let mut key_bytes = 0usize;
        let mut image_bytes = 0usize;
        for (key, asset) in assets {
            key_bytes = key_bytes.saturating_add(key.len());
            if key.trim().is_empty() || key.len() > 1024 || key_bytes > 65536 {
                return Err(invalid(
                    key,
                    "asset references must be non-blank, at most 1024 bytes each and 65536 bytes together",
                ));
            }
            let asset = asset.as_object().ok_or_else(|| invalid(key, "asset must be an image object"))?;
            if asset.get("type").is_some_and(|v| v != "image") {
                return Err(invalid(key, "asset type must be image"));
            }
            if asset.contains_key("part") {
                return Err(invalid(key, "part requires a matching multipart image part"));
            }
            if !matches!(asset.get("format").and_then(Value::as_str), Some("png" | "jpg" | "jpeg")) {
                return Err(invalid(key, "format must be png, jpg or jpeg"));
            }
            let data = asset
                .get("data_base64")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| invalid(key, "data_base64 must be a non-empty string"))?;
            if data.len() > MAX_IMAGE_BYTES.div_ceil(3) * 4 {
                return Err(invalid(key, "image exceeds 16 MiB"));
            }
            let raw = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|_| invalid(key, "invalid standard padded base64 image"))?;
            if raw.len() > MAX_IMAGE_BYTES {
                return Err(invalid(key, "image exceeds 16 MiB"));
            }
            let reader = image::ImageReader::new(Cursor::new(&raw))
                .with_guessed_format()
                .map_err(|_| invalid(key, "invalid image"))?;
            let format = reader.format();
            if !matches!(format, Some(image::ImageFormat::Png | image::ImageFormat::Jpeg)) {
                return Err(invalid(key, "only PNG and JPEG images are accepted"));
            }
            let (width, height) = reader.into_dimensions().map_err(|_| invalid(key, "invalid image dimensions"))?;
            if !(280..=1568).contains(&width) || !(280..=1568).contains(&height) {
                return Err(invalid(key, "image width and height must each be 280..1568 pixels"));
            }
            if format == Some(image::ImageFormat::Png) {
                let png = image::codecs::png::PngDecoder::new(Cursor::new(&raw))
                    .map_err(|_| invalid(key, "invalid PNG image"))?;
                if png.is_apng().map_err(|_| invalid(key, "invalid PNG image"))? {
                    return Err(invalid(key, "animated PNG images are not supported"));
                }
            }
            image_bytes = image_bytes.saturating_add(raw.len());
            if image_bytes > MAX_BODY_BYTES {
                return Err(invalid(key, "images together exceed 32 MiB"));
            }
            references.push(key.clone());
            sizes.push(raw.len());
            entries.push(json!({"type":"image", "data":data}));
        }
        // A fixed-length namespace derived from the complete request avoids unbounded prefix searches on
        // adversarial text. Reject a collision explicitly; no user text may become an internal placeholder.
        let serialized = body.to_string();
        let prefix = format!("\u{e000}basal-asset:{:x}:", Sha256::digest(serialized.as_bytes()));
        if serialized.contains(&prefix) {
            return Err(unsupported(vec!["body".into(), "assets".into()], "image placeholder namespace collision"));
        }
        let markers = (0..references.len()).map(|i| format!("{prefix}{i}\u{e001}")).collect();
        let matcher = build_matcher(&references).map_err(|e| DecideError::Internal(e.into()))?;
        Ok(Some(Self { entries: Value::Array(entries), markers, references, sizes, matcher }))
    }

    pub fn apply(&self, body: &Value, items: &mut [Item], fmt: crate::prompt::PromptFormat) -> Result<(), DecideError> {
        let mut transformed = serde_json::Map::new();
        let mut used = vec![false; self.references.len()];
        let mut expanded = 0usize;
        for field in ["model", "lang", "detail"] {
            if let Some(v) = body.get(field) {
                transformed.insert(field.into(), v.clone());
            }
        }
        transformed.insert("state".into(), self.transform(&body["state"], &mut used, &mut expanded)?);
        expanded = expanded.saturating_mul(items.len());
        if expanded > MAX_BODY_BYTES {
            return Err(unsupported(
                vec!["body".into(), "assets".into()],
                "expanded image content exceeds the 32 MiB runtime budget",
            ));
        }
        let mut questions = body["questions"].clone();
        if let Some(questions) = questions.as_object_mut() {
            for q in questions.values_mut().filter_map(Value::as_object_mut) {
                for field in ["instructions", "criteria", "levels"] {
                    if let Some(v) = q.get_mut(field) {
                        *v = self.transform(v, &mut used, &mut expanded)?;
                    }
                }
            }
        }
        if used.iter().any(|v| !v) {
            return Err(crate::request::api_usage_error(
                "every asset must be referenced in state, instructions or criteria",
            ));
        }
        transformed.insert("questions".into(), questions);
        let rendered = crate::request::parse_request_for_format(&Value::Object(transformed), Dialect::TypeSafe, fmt)?;
        // Preserve original IDs, legends, language and all decision metadata. Only prompt content contains markers.
        for (item, rendered) in items.iter_mut().zip(rendered.items) {
            item.state = rendered.state;
            item.question = rendered.question;
            item.options = rendered.options;
        }
        Ok(())
    }

    /// Reject repeated-image expansion before building large placeholder strings, token arrays or packed rows.
    pub fn check_prompt_budget(
        &self,
        body: &Value,
        items: &[Item],
        media: &crate::media::ProcessedMedia,
        max_positions: usize,
    ) -> Result<(), DecideError> {
        let mut bytes = crate::request::prepared_input_bytes(body)?;
        for item in items {
            let mut tokens = 0usize;
            for text in std::iter::once(&item.state).chain(std::iter::once(&item.question)).chain(&item.options) {
                for (marker, image) in self.markers.iter().zip(&media.items) {
                    tokens = tokens.saturating_add(text.matches(marker).count().saturating_mul(image.soft_tokens()));
                }
            }
            if tokens > max_positions {
                return Err(DecideError::Unsupported(vec![FieldError::new(
                    vec!["body".into(), "questions".into(), item.name.clone().into()],
                    format!("image tokens exceed the maximum context length of {max_positions} tokens"),
                    "context_length_exceeded",
                )]));
            }
            // Both possible option orders are prepared, including an adaptive reverse order.
            bytes = bytes.saturating_add(tokens.saturating_mul("<|image|>".len()).saturating_mul(2));
            if bytes > MAX_BODY_BYTES {
                return Err(unsupported(
                    vec!["body".into(), "assets".into()],
                    "expanded image placeholders exceed the 32 MiB host prompt budget",
                ));
            }
        }
        Ok(())
    }

    fn transform(&self, value: &Value, used: &mut [bool], expanded: &mut usize) -> Result<Value, DecideError> {
        Ok(match value {
            Value::String(s) => {
                *expanded = expanded.saturating_add(s.len());
                let mut rendered = String::new();
                let mut cursor = 0;
                // Enforce the expansion budget while scanning, before allocating all replacement markers/indices.
                for found in self.matcher.find_iter(s) {
                    let i = self
                        .references
                        .iter()
                        .position(|r| r == found.as_str())
                        .expect("the matcher contains only escaped, non-empty asset references");
                    used[i] = true;
                    *expanded = expanded.saturating_add(self.sizes[i]);
                    if *expanded > MAX_BODY_BYTES {
                        return Err(unsupported(
                            vec!["body".into(), "assets".into()],
                            "expanded image content exceeds the 32 MiB runtime budget",
                        ));
                    }
                    rendered.push_str(&s[cursor..found.start()]);
                    rendered.push_str(&self.markers[i]);
                    cursor = found.end();
                }
                if *expanded > MAX_BODY_BYTES {
                    return Err(unsupported(
                        vec!["body".into(), "assets".into()],
                        "expanded image content exceeds the 32 MiB runtime budget",
                    ));
                }
                rendered.push_str(&s[cursor..]);
                Value::String(rendered)
            }
            Value::Array(values) => {
                Value::Array(values.iter().map(|v| self.transform(v, used, expanded)).collect::<Result<_, _>>()?)
            }
            Value::Object(values) => Value::Object(
                values
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), self.transform(v, used, expanded)?)))
                    .collect::<Result<_, DecideError>>()?,
            ),
            _ => value.clone(),
        })
    }
}

/// Whole-value matches take priority; otherwise take the next leftmost non-overlapping match. Equal-start
/// substring matches prefer the longest reference, then insertion order, independent of hash iteration order.
pub(crate) fn substitute(
    text: &str,
    references: &[String],
    replacements: &[String],
) -> Result<(String, Vec<usize>), regex::Error> {
    let matcher = build_matcher(references)?;
    Ok(substitute_matches(text, references, replacements, &matcher))
}

fn build_matcher(references: &[String]) -> Result<regex::Regex, regex::Error> {
    let mut patterns: Vec<_> = references.iter().collect();
    patterns.sort_by_key(|r| std::cmp::Reverse(r.len()));
    regex::Regex::new(&patterns.into_iter().map(|s| regex::escape(s)).collect::<Vec<_>>().join("|"))
}

fn substitute_matches(
    text: &str,
    references: &[String],
    replacements: &[String],
    matcher: &regex::Regex,
) -> (String, Vec<usize>) {
    if let Some(i) = references.iter().position(|r| r == text) {
        return (replacements[i].clone(), vec![i]);
    }
    let mut out = String::new();
    let mut order = Vec::new();
    let mut cursor = 0;
    // One multi-pattern scan avoids rescanning the remaining text for each occurrence and each absent reference.
    for found in matcher.find_iter(text) {
        let i = references
            .iter()
            .position(|r| r == found.as_str())
            .expect("the matcher contains only escaped, non-empty asset references");
        out.push_str(&text[cursor..found.start()]);
        out.push_str(&replacements[i]);
        order.push(i);
        cursor = found.end();
    }
    out.push_str(&text[cursor..]);
    (out, order)
}
