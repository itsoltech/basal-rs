use super::text::{Span, Text};
use anyhow::{bail, ensure, Result};
use serde_json::Value;

#[derive(Clone, Copy, Debug)]
pub(super) enum Kind {
    Enum,
    EnumList,
    Boolean,
    String,
    Number,
    Integer,
}
#[derive(Clone, Debug)]
pub(super) struct Field {
    pub path: Vec<String>,
    pub key: String,
    pub kind: Kind,
    pub spec: Value,
    pub candidates: Vec<Span>,
}
impl Field {
    pub fn description(&self) -> String {
        let name = self.key.replace('_', " ").replace('.', " / ");
        self.spec
            .get("description")
            .filter(|v| crate::request::truthy(Some(v)))
            .or_else(|| self.spec.get("title").filter(|v| crate::request::truthy(Some(v))))
            .map_or(name.clone(), |v| format!("{name} ({})", super::py_string(v)))
    }
}

pub(super) fn consts(schema: &Value) -> Option<Vec<(Value, Option<Value>)>> {
    for key in ["oneOf", "anyOf"] {
        if let Some(alts) = schema
            .get(key)
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty() && v.iter().all(|a| a.get("const").is_some()))
        {
            return Some(
                alts.iter()
                    .map(|a| (a["const"].clone(), a.get("description").or_else(|| a.get("title")).cloned()))
                    .collect(),
            );
        }
    }
    schema
        .get("enum")
        .and_then(Value::as_array)
        .map(|xs| xs.iter().filter(|v| !v.is_null()).map(|v| (v.clone(), None)).collect())
}

pub(super) fn fields(schema: &Value) -> Result<Vec<Field>> {
    fn visit(schema: &Value, path: &mut Vec<String>, out: &mut Vec<Field>) -> Result<()> {
        ensure!(path.len() <= 16, "extract schema exceeds 16 nested levels");
        ensure!(out.len() < 512, "extract schema exceeds 512 fields");
        ensure!(schema.is_object(), "extract schema must be an object");
        let ty = match schema.get("type") {
            Some(Value::Array(v)) => v.iter().filter_map(Value::as_str).find(|s| *s != "null"),
            Some(Value::String(v)) => Some(v.as_str()),
            _ => None,
        };
        if ty == Some("object") || schema.get("properties").is_some() {
            let props = schema
                .get("properties")
                .and_then(Value::as_object)
                .filter(|p| !p.is_empty())
                .ok_or_else(|| anyhow::anyhow!("extract object has no properties"))?;
            for (key, value) in props {
                ensure!(
                    !key.is_empty() && !key.contains('.'),
                    "extract property names must be nonempty and may not contain dots"
                );
                path.push(key.clone());
                visit(value, path, out)?;
                path.pop();
            }
            return Ok(());
        }
        ensure!(!path.is_empty(), "extract schema root must have object properties");
        let kind = if consts(schema).is_some_and(|v| !v.is_empty()) {
            Kind::Enum
        } else {
            match ty {
                Some("boolean") => Kind::Boolean,
                Some("array") if consts(&schema["items"]).is_some_and(|v| !v.is_empty()) => Kind::EnumList,
                Some("array") => bail!("extract arrays require enum items"),
                Some("string") | None => Kind::String,
                Some("number") => Kind::Number,
                Some("integer") => Kind::Integer,
                Some(t) => bail!("extract field type {t:?} is unsupported"),
            }
        };
        out.push(Field { path: path.clone(), key: path.join("."), kind, spec: schema.clone(), candidates: vec![] });
        Ok(())
    }
    let mut out = Vec::new();
    visit(schema, &mut Vec::new(), &mut out)?;
    Ok(out)
}

pub(super) fn set_path(value: &mut Value, path: &[String], field: Value) {
    let mut current = value;
    for key in &path[..path.len() - 1] {
        current = current
            .as_object_mut()
            .expect("extraction object invariant")
            .entry(key)
            .or_insert_with(|| serde_json::json!({}));
    }
    current.as_object_mut().expect("extraction object invariant").insert(path[path.len() - 1].clone(), field);
}

pub(super) fn regions(text: &Text, fields: &[Field], cap: usize) -> Result<Vec<Span>> {
    if fields.iter().any(|f| matches!(f.kind, Kind::String | Kind::Number | Kind::Integer)) {
        ensure!(
            !text.value.is_empty(),
            "extract text fields need a text state (media supports enum and boolean fields only)"
        );
        text.regions(cap)
    } else {
        Ok(vec![])
    }
}
