//! TypeSafe System One request: structural validation (independent of the backend) and conversion of questions into
//! readout items (port of upstream `server.to_items` / `named_options`, with the deviations listed in
//! docs/SYSTEM_ONE.md). Capability limits of a backend are checked separately in [`check_capability`].

use indexmap::IndexMap;
use serde::Serialize;
use serde_json::{json, Value};

use crate::prompt::{lang_of, Lang, MAX_OPTIONS};
use crate::pyjson::text;

/// System One documents up to 255 choice options.
pub const MAX_CHOICE_OPTIONS: usize = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    Choice,
    Noul,
    Score,
    /// basal-1.5 extension: one yes/no branch per label, the labels whose P(yes) passes a threshold.
    Multi,
    /// basal-1.5 extension: a choice or yes/no readout followed by the minimum expected cost action.
    Act,
}

impl QType {
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Noul => "noul",
            QType::Score => "score",
            QType::Multi => "multi",
            QType::Act => "act",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(QType::Choice),
            "noul" => Some(QType::Noul),
            "score" => Some(QType::Score),
            "multi" => Some(QType::Multi),
            "act" => Some(QType::Act),
            _ => None,
        }
    }
}

/// `multi`: labels (shown texts) in key order and the selection rule.
#[derive(Clone, Debug, Serialize)]
pub struct MultiSpec {
    pub labels: Vec<String>,
    pub threshold: f64,
    pub max: Option<i64>,
    pub min: Option<i64>,
}

/// `act`: cost matrix `costs[action][outcome]` (actions in request order, outcomes in key order), the defer action
/// and the certified error target. `threshold` (the certified confidence for `max_error`) and `calibrated` are filled
/// from the model's CALIBRATION.json when the request is planned.
#[derive(Clone, Debug, Serialize)]
pub struct ActSpec {
    pub base: QType,
    pub costs: Vec<(String, Vec<f64>)>,
    pub defer: Option<String>,
    /// Python `str()` of the requested `max_error` (the key of CALIBRATION.json thresholds)
    pub max_error: Option<String>,
    pub hide: bool,
    pub threshold: Option<f64>,
    pub calibrated: bool,
}

/// Type-specific part of an item.
#[derive(Clone, Debug, Default, Serialize)]
pub enum Ext {
    #[default]
    None,
    Multi(MultiSpec),
    Act(ActSpec),
}

/// Request and answer conventions of an endpoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dialect {
    /// TypeSafe System One: strict schema, TypeSafe answer objects and confidence (`/v1/systemone`).
    #[default]
    TypeSafe,
    /// Upstream basal `Server.decide` (`/v1/basal`): its lenient parsing (`to_items`), answer objects
    /// (`confidence = max(p)`, noul with probabilities, score legend of shown option texts), 2..10 options.
    Upstream,
}

/// One validation problem, in the shape of TypeSafe `ValidationError` (`type`, `loc`, `msg`, `input`, `ctx`; the
/// pydantic error of the TypeSafe API, docs/SYSTEM_ONE.md).
#[derive(Clone, Debug, Serialize)]
pub struct FieldError {
    #[serde(rename = "type")]
    pub kind: String,
    pub loc: Vec<Value>,
    pub msg: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx: Option<Value>,
}

impl FieldError {
    pub fn new(loc: Vec<Value>, msg: impl Into<String>, kind: &str) -> Self {
        FieldError { kind: kind.into(), loc, msg: msg.into(), input: None, ctx: None }
    }

    fn input(mut self, v: &Value) -> Self {
        self.input = Some(v.clone());
        self
    }

    fn ctx(mut self, v: Value) -> Self {
        self.ctx = Some(v);
        self
    }
}

#[derive(Debug)]
pub enum DecideError {
    /// The request does not satisfy the System One schema (HTTP 422).
    Validation(Vec<FieldError>),
    /// A schema-valid request the TypeSafe API refuses (HTTP 400): the value of its `detail`, a message or
    /// `{"error_type": "api_usage_error", "message": ...}`.
    Usage(Value),
    /// Valid request that this runtime/backend cannot answer yet (explicit, never a silent fallback).
    Unsupported(Vec<FieldError>),
    Internal(anyhow::Error),
}

impl std::fmt::Display for DecideError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecideError::Validation(e) => write!(f, "validation error: {}", serde_json::to_string(e).unwrap()),
            DecideError::Usage(d) => write!(f, "invalid request: {d}"),
            DecideError::Unsupported(e) => write!(f, "unsupported: {}", serde_json::to_string(e).unwrap()),
            DecideError::Internal(e) => write!(f, "internal error: {e:#}"),
        }
    }
}

/// `{"error_type": "api_usage_error", "message": ...}`, the `detail` of TypeSafe's 400 answers about the request
/// as a whole (unknown model, unknown question type).
pub fn api_usage_error(message: impl Into<String>) -> DecideError {
    DecideError::Usage(json!({"error_type": "api_usage_error", "message": message.into()}))
}

impl std::error::Error for DecideError {}

impl From<anyhow::Error> for DecideError {
    fn from(e: anyhow::Error) -> Self {
        DecideError::Internal(e)
    }
}

impl DecideError {
    /// Upstream basal error body `{"error": message}` (it answers every failed request with 422).
    pub fn to_upstream_json(&self) -> Value {
        let msg = match self {
            DecideError::Validation(d) | DecideError::Unsupported(d) => {
                d.iter().map(|e| e.msg.clone()).collect::<Vec<_>>().join("; ")
            }
            DecideError::Usage(Value::String(s)) => s.clone(),
            DecideError::Usage(d) => d.get("message").and_then(Value::as_str).unwrap_or_default().to_string(),
            DecideError::Internal(e) => format!("{e:#}"),
        };
        json!({"error": msg})
    }

    pub fn to_json(&self) -> Value {
        match self {
            DecideError::Validation(d) => json!({"detail": d}),
            DecideError::Usage(d) => json!({"detail": d}),
            DecideError::Unsupported(d) => json!({"error": "unsupported", "detail": d}),
            DecideError::Internal(e) => json!({"error": "internal", "detail": format!("{e:#}")}),
        }
    }
}

/// One question prepared for the letter readout.
#[derive(Clone, Debug, Serialize)]
pub struct Item {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: QType,
    /// Answer keys in option order (choice names, "true"/"false", score level indices or names).
    pub keys: Vec<String>,
    /// Option texts shown to the model, in canonical order.
    pub options: Vec<String>,
    /// Original JSON values of score levels (legend); empty for other types.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub legend: Vec<Value>,
    pub state: String,
    pub question: String,
    pub lang: Lang,
    #[serde(skip)]
    pub ext: Ext,
    /// `"evidence"` requested (truthy): return the supporting spans of the state.
    #[serde(skip)]
    pub evidence: bool,
    /// Upstream dialect: the question type has no calibration temperature (an unknown type read as choice).
    #[serde(skip)]
    pub uncalibrated: bool,
}

impl Item {
    /// Calibration key of the item's readout: `act` uses its base type, a `multi` branch is a yes/no question.
    pub fn temp_kind(&self) -> QType {
        match &self.ext {
            Ext::Act(a) => a.base,
            Ext::Multi(_) => QType::Noul,
            Ext::None => self.kind,
        }
    }
}

/// Question text of the yes/no branch of a `multi` label (upstream `MULTI_Q`).
pub fn multi_question(lang: Lang, instructions: &str, label: &str) -> String {
    match lang {
        Lang::Pl => format!("{instructions}\nCzy dotyczy: \u{201e}{label}\u{201d}?"),
        Lang::En => format!("{instructions}\nDoes this apply: \"{label}\"?"),
    }
}

impl Serialize for Lang {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

pub struct ParsedRequest {
    pub model: String,
    pub items: Vec<Item>,
    /// `facts: "auto"`: characters of the request's own state (evidence spans are kept inside it).
    pub facts_limit: Option<usize>,
}

fn loc(path: &[&str]) -> Vec<Value> {
    std::iter::once("body").chain(path.iter().copied()).map(|s| Value::String(s.to_string())).collect()
}

fn err(path: &[&str], msg: impl Into<String>, kind: &str) -> FieldError {
    FieldError::new(loc(path), msg, kind)
}

fn qerr(name: &str, rest: &[&str], msg: impl Into<String>, kind: &str) -> FieldError {
    let path: Vec<&str> = ["questions", name].into_iter().chain(rest.iter().copied()).collect();
    err(&path, msg, kind)
}

/// `["body", "questions", name, rest...]`; [`question`] adds the question type after the name, as the TypeSafe API
/// (a pydantic discriminated union) reports it.
fn qloc(name: &str, rest: &[Value]) -> Vec<Value> {
    ["body", "questions", name].into_iter().map(Value::from).chain(rest.iter().cloned()).collect()
}

/// `Field required` of a missing field of `parent`.
fn missing(loc: Vec<Value>, parent: &Value) -> FieldError {
    FieldError::new(loc, "Field required", "missing").input(parent)
}

/// The errors of a value that is none of string, object and array (the description type of System One: one per
/// member of the union, as pydantic reports them).
fn union_errs(loc: Vec<Value>, input: &Value) -> Vec<FieldError> {
    [
        ("str", "string_type", "Input should be a valid string"),
        ("dict[any,any]", "dict_type", "Input should be a valid dictionary"),
        ("list[any]", "list_type", "Input should be a valid list"),
    ]
    .into_iter()
    .map(|(member, kind, msg)| {
        let mut l = loc.clone();
        l.push(member.into());
        FieldError::new(l, msg, kind).input(input)
    })
    .collect()
}

/// pydantic `too_short` of an empty list or object.
fn too_short(loc: Vec<Value>, input: &Value) -> FieldError {
    let what = if input.is_array() { "List" } else { "Dictionary" };
    FieldError::new(loc, format!("{what} should have at least 1 item after validation, not 0"), "too_short")
        .input(input)
        .ctx(json!({"field_type": what, "min_length": 1, "actual_length": 0}))
}

/// Kind of an error the TypeSafe API answers with 400 instead of 422 (its `detail` is in `input`); it is reported
/// only when the request has no schema error.
const USAGE: &str = "api_usage";

/// A 400 answer of the TypeSafe API: `detail` is a message or an `api_usage_error` object.
fn usage(detail: Value) -> FieldError {
    FieldError::new(vec![], "", USAGE).input(&detail)
}

/// Python truthiness of an optional JSON value (`q.get("evidence")`).
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// String, object or array (and null when `allow_null`): the "description" type of System One.
fn is_description(v: &Value, allow_null: bool) -> bool {
    matches!(v, Value::String(_) | Value::Object(_) | Value::Array(_)) || (allow_null && v.is_null())
}

/// upstream `named_options`: option texts for `{key: description}`.
fn named_options(crit: &IndexMap<String, Value>, hide: bool) -> (Vec<String>, Vec<String>) {
    let texts: Vec<String> = crit.iter().map(|(k, v)| if v.is_null() { k.clone() } else { text(v) }).collect();
    let dup = |t: &String| texts.iter().filter(|x| *x == t).count() > 1;
    let shown = crit
        .iter()
        .zip(&texts)
        .map(|((k, v), t)| if v.is_null() || (hide && !dup(t)) { t.clone() } else { format!("{k}: {t}") })
        .collect();
    (crit.keys().cloned().collect(), shown)
}

fn to_map(m: &serde_json::Map<String, Value>) -> IndexMap<String, Value> {
    m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Validate a System One request body and build readout items. All schema errors are collected.
pub fn parse_request(body: &Value) -> Result<ParsedRequest, DecideError> {
    parse_request_as(body, Dialect::TypeSafe)
}

/// [`parse_request`] for an endpoint dialect.
pub fn parse_request_as(body: &Value, dialect: Dialect) -> Result<ParsedRequest, DecideError> {
    let mut errs = Vec::new();
    let Some(obj) = body.as_object() else {
        return Err(DecideError::Validation(vec![FieldError::new(
            loc(&[]),
            "Input should be a valid dictionary or object to extract fields from",
            "model_attributes_type",
        )
        .input(body)]));
    };
    let model = match obj.get("model") {
        // upstream ignores the model field
        _ if dialect == Dialect::Upstream => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => {
            errs.push(FieldError::new(loc(&["model"]), "Input should be a valid string", "string_type").input(v));
            String::new()
        }
        None => {
            errs.push(missing(loc(&["model"]), body));
            String::new()
        }
    };
    let state = match obj.get("state") {
        Some(v) if is_description(v, false) || (dialect == Dialect::Upstream && !v.is_null()) => Some(text(v)),
        // a null state is a missing one for the TypeSafe API
        None | Some(Value::Null) => {
            errs.push(missing(loc(&["state"]), body));
            None
        }
        Some(v) => {
            errs.append(&mut union_errs(loc(&["state"]), v));
            None
        }
    };
    let questions = match obj.get("questions") {
        Some(Value::Object(m)) if !m.is_empty() => Some(m),
        Some(v @ Value::Object(_)) => {
            errs.push(too_short(loc(&["questions"]), v));
            None
        }
        Some(v) => {
            errs.push(FieldError::new(loc(&["questions"]), "Input should be a valid dictionary", "dict_type").input(v));
            None
        }
        None => {
            errs.push(missing(loc(&["questions"]), body));
            None
        }
    };
    // basal-1.5 `facts: "auto"`: computed calendar and amount facts appended to the state (upstream facts.inject,
    // before the language of the questions is detected)
    let mut state = state;
    let mut facts_limit = None;
    match obj.get("facts") {
        None | Some(Value::Null) => {}
        Some(Value::String(f)) if f == "off" => {}
        Some(Value::String(f)) if f == "auto" => {
            if let Some(s) = &state {
                facts_limit = Some(s.chars().count());
                match crate::facts::try_inject(s) {
                    Ok(x) => state = Some(x),
                    Err(e) => errs.push(err(&["state"], e.0, "value_error")),
                }
            }
        }
        Some(_) => errs.push(err(&["facts"], "facts must be \"off\" or \"auto\"", "literal_error")),
    }
    if let Some(questions) = questions {
        for (name, q) in questions {
            if truthy(q.get("evidence")) && q.get("type").and_then(Value::as_str) == Some("multi") {
                errs.push(qerr(name, &["evidence"], "evidence is not supported for multi", "value_error"));
            }
        }
    }
    // TypeSafe: questions are checked also without a valid state, so that all schema errors are reported together
    let mut items = Vec::new();
    let state = if dialect == Dialect::TypeSafe { Some(state.unwrap_or_default()) } else { state };
    if let (Some(state), Some(questions)) = (state, questions) {
        for (name, q) in questions {
            let built = match dialect {
                Dialect::TypeSafe => question(name, q, &state),
                Dialect::Upstream => question_upstream(name, q, &state),
            };
            match built {
                Ok(it) => items.push(it),
                Err(mut e) => errs.append(&mut e),
            }
        }
    }
    // schema errors first (422), then the first error the TypeSafe API answers with 400
    let (usage, errs): (Vec<FieldError>, Vec<FieldError>) = errs.into_iter().partition(|e| e.kind == USAGE);
    if !errs.is_empty() {
        Err(DecideError::Validation(errs))
    } else if let Some(u) = usage.into_iter().next() {
        Err(DecideError::Usage(u.input.unwrap_or_default()))
    } else {
        Ok(ParsedRequest { model, items, facts_limit })
    }
}

fn question(name: &str, qv: &Value, state: &str) -> Result<Item, Vec<FieldError>> {
    let Some(q) = qv.as_object() else {
        return Err(vec![FieldError::new(
            qloc(name, &[]),
            "Input should be a valid dictionary or object to extract fields from",
            "model_attributes_type",
        )
        .input(qv)]);
    };
    let mut errs = Vec::new();
    let kind = match q.get("type") {
        Some(Value::String(s)) => match QType::parse(s) {
            Some(t) => Some(t),
            None => {
                errs.push(usage(json!({
                    "error_type": "api_usage_error",
                    "message": format!("Invalid request. Unknown question type {s:?} (choice, noul, score): {name}"),
                })));
                None
            }
        },
        None => {
            errs.push(
                FieldError::new(
                    qloc(name, &[]),
                    "Unable to extract tag using discriminator 'type'",
                    "union_tag_not_found",
                )
                .input(qv)
                .ctx(json!({"discriminator": "'type'"})),
            );
            None
        }
        Some(t) => {
            errs.push(usage(json!({
                "error_type": "api_usage_error",
                "message": format!("Invalid request. Question type {t} is not a string: {name}"),
            })));
            None
        }
    };
    // Missing and null instructions both mean "no instructions" (empty question text). Upstream renders null as the
    // literal text "null"; see docs/SYSTEM_ONE.md.
    let instructions = match q.get("instructions") {
        None | Some(Value::Null) => String::new(),
        Some(v) if is_description(v, false) => text(v),
        Some(v) => {
            errs.append(&mut union_errs(qloc(name, &["instructions".into()]), v));
            String::new()
        }
    };
    let hide = match q.get("option_keys") {
        None | Some(Value::Null) => false,
        Some(Value::String(s)) if s == "show" => false,
        Some(Value::String(s)) if s == "hide" => true,
        Some(_) => {
            errs.push(qerr(name, &["option_keys"], "option_keys must be \"show\" or \"hide\"", "literal_error"));
            false
        }
    };
    let Some(kind) = kind else { return Err(errs) };
    // the question type after the name in the location of every error, as in the TypeSafe API
    let tag = |mut errs: Vec<FieldError>| {
        for e in &mut errs {
            if e.loc.len() > 3 && e.loc[2] == name {
                e.loc.insert(3, kind.as_str().into());
            }
        }
        errs
    };
    let errs = tag(errs);
    if kind == QType::Noul
        && q.get("instructions").is_none_or(Value::is_null)
        && q.get("criteria").is_none_or(Value::is_null)
    {
        return Err(vec![usage(format!("Noul question must have criteria or instructions: {name}").into())]);
    }
    let mut errs = errs;
    let lang = lang_of(&format!("{state}{instructions}"));
    let mut ext = Ext::None;
    let built = match kind {
        QType::Noul => noul(q, lang, name),
        QType::Choice => choice(q, hide, name),
        QType::Score => score(q, hide, name),
        QType::Multi => multi(q, lang, name).map(|(b, m)| {
            ext = Ext::Multi(m);
            b
        }),
        QType::Act => act(q, lang, hide, name).map(|(b, a)| {
            ext = Ext::Act(a);
            b
        }),
    };
    match built {
        Ok((keys, options, legend)) if errs.is_empty() => Ok(Item {
            name: name.to_string(),
            kind,
            keys,
            options,
            legend,
            state: state.to_string(),
            question: instructions,
            lang,
            ext,
            evidence: truthy(q.get("evidence")),
            uncalibrated: false,
        }),
        Ok(_) => Err(errs),
        Err(e) => {
            errs.append(&mut tag(e));
            Err(errs)
        }
    }
}

/// keys, option texts, score legend
type Parts = (Vec<String>, Vec<String>, Vec<Value>);
type Built = Result<Parts, Vec<FieldError>>;

/// Upstream `to_items` for one question: type defaults to choice (an unknown type is read as choice without
/// calibration), `instructions` is any JSON (null renders as "null"), falsy noul descriptions fall back to Tak/Nie,
/// criteria values may be any JSON, a score legend holds the shown option texts, and every type but multi needs
/// 2..10 options.
fn question_upstream(name: &str, q: &Value, state: &str) -> Result<Item, Vec<FieldError>> {
    let Some(q) = q.as_object() else {
        return Err(vec![qerr(name, &[], "question must be an object", "model_type")]);
    };
    let t = q.get("type").and_then(Value::as_str).unwrap_or("choice");
    let (kind, uncalibrated) = match QType::parse(t) {
        Some(k) => (k, false),
        None => (QType::Choice, true),
    };
    let instructions = q.get("instructions").map(text).unwrap_or_default();
    let lang = lang_of(&format!("{state}{instructions}"));
    let hide = match q.get("option_keys") {
        None => false,
        Some(Value::String(s)) if s == "show" => false,
        Some(Value::String(s)) if s == "hide" => true,
        Some(v) => {
            return Err(vec![qerr(
                name,
                &["option_keys"],
                format!("option_keys must be \"show\" or \"hide\", got {v}"),
                "value_error",
            )])
        }
    };
    let crit_or = |keys: &[&str]| -> Value {
        keys.iter().filter_map(|k| q.get(*k)).find(|v| truthy(Some(v))).cloned().unwrap_or(Value::Null)
    };
    let as_map = |crit: &Value, path: &str| -> Result<IndexMap<String, Value>, Vec<FieldError>> {
        match crit {
            Value::Null => Ok(IndexMap::new()),
            Value::Object(m) => Ok(to_map(m)),
            Value::Array(xs) => {
                let mut m = IndexMap::new();
                for x in xs {
                    m.entry(text(x)).or_insert(Value::Null);
                }
                Ok(m)
            }
            _ => Err(vec![qerr(name, &[path], format!("{path} must be an object or a list"), "type_error")]),
        }
    };
    let (yes, no) = if lang == Lang::Pl { ("Tak", "Nie") } else { ("Yes", "No") };
    let mut ext = Ext::None;
    let (keys, options, legend) = match kind {
        QType::Multi => {
            let (b, m) = multi(q, lang, name)?;
            ext = Ext::Multi(m);
            b
        }
        QType::Act => {
            let (b, a) = act(q, lang, hide, name)?;
            ext = Ext::Act(a);
            b
        }
        QType::Noul => {
            let crit = match q.get("criteria") {
                Some(v) if truthy(Some(v)) => match v {
                    Value::Object(m) => m.clone(),
                    _ => return Err(vec![qerr(name, &["criteria"], "noul criteria must be an object", "type_error")]),
                },
                _ => serde_json::Map::new(),
            };
            let side =
                |k: &str, d: &str| crit.get(k).filter(|v| truthy(Some(v))).map(text).unwrap_or_else(|| d.to_string());
            (vec!["true".into(), "false".into()], vec![side("true", yes), side("false", no)], vec![])
        }
        QType::Score => match crit_or(&["criteria", "levels"]) {
            Value::Object(m) => {
                let (k, o) = named_options(&to_map(&m), hide);
                let legend = o.iter().cloned().map(Value::String).collect();
                (k, o, legend)
            }
            Value::Null => (vec![], vec![], vec![]),
            Value::Array(xs) => {
                let o: Vec<String> = xs.iter().map(text).collect();
                ((0..xs.len()).map(|i| i.to_string()).collect(), o.clone(), o.into_iter().map(Value::String).collect())
            }
            _ => {
                return Err(vec![qerr(name, &["criteria"], "score criteria must be a list or an object", "type_error")])
            }
        },
        QType::Choice => {
            let crit = as_map(&crit_or(&["criteria"]), "criteria")?;
            let (k, o) = named_options(&crit, hide);
            (k, o, vec![])
        }
    };
    if kind != QType::Multi && !(2..=MAX_OPTIONS).contains(&options.len()) {
        return Err(vec![qerr(
            name,
            &["criteria"],
            format!("question {name:?}: {} options (supported: 2..{MAX_OPTIONS})", options.len()),
            "value_error",
        )]);
    }
    Ok(Item {
        name: name.to_string(),
        kind,
        keys,
        options,
        legend,
        state: state.to_string(),
        question: instructions,
        lang,
        ext,
        evidence: truthy(q.get("evidence")),
        uncalibrated,
    })
}

/// Python `str()` of a JSON number (CALIBRATION.json threshold keys are written that way).
fn py_str(v: &Value) -> Option<String> {
    match v {
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(n.to_string()),
        Value::Number(n) => n.as_f64().map(crate::pyjson::float_repr),
        _ => None,
    }
}

/// `multi`: criteria `{key: description}` or `[keys]`, any number of labels; options of every branch are the default
/// yes/no texts. Keys: the label keys; options: the yes/no texts.
fn multi(q: &serde_json::Map<String, Value>, lang: Lang, name: &str) -> Result<(Parts, MultiSpec), Vec<FieldError>> {
    let crit: IndexMap<String, Value> = match q.get("criteria") {
        Some(Value::Object(m)) => to_map(m),
        Some(Value::Array(xs)) => {
            let mut m = IndexMap::new();
            for (i, x) in xs.iter().enumerate() {
                let Value::String(s) = x else {
                    return Err(vec![qerr(
                        name,
                        &["criteria", &i.to_string()],
                        "labels must be strings",
                        "string_type",
                    )]);
                };
                m.entry(s.clone()).or_insert(Value::Null);
            }
            m
        }
        None | Some(Value::Null) => IndexMap::new(),
        Some(_) => {
            return Err(vec![qerr(name, &["criteria"], "multi criteria must be an object or a list", "dict_type")])
        }
    };
    if crit.is_empty() {
        return Err(vec![qerr(name, &["criteria"], "multi needs criteria", "missing")]);
    }
    let num = |k: &str| -> Result<Option<f64>, Vec<FieldError>> {
        match q.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(n)) => Ok(n.as_f64()),
            Some(_) => Err(vec![qerr(name, &[k], format!("{k} must be a number"), "float_type")]),
        }
    };
    let threshold = num("threshold")?.unwrap_or(0.5);
    let int = |k: &str| -> Result<Option<i64>, Vec<FieldError>> { Ok(num(k)?.map(|v| v.trunc() as i64)) };
    let labels: Vec<String> = crit.iter().map(|(k, v)| if v.is_null() { k.clone() } else { text(v) }).collect();
    let (yes, no) = if lang == Lang::Pl { ("Tak", "Nie") } else { ("Yes", "No") };
    let spec = MultiSpec { labels, threshold, max: int("max")?, min: int("min")? };
    Ok(((crit.keys().cloned().collect(), vec![yes.into(), no.into()], vec![]), spec))
}

/// `act`: readout as `noul` (criteria keys within {true, false}, or `base: "noul"`) or as a choice over named
/// options, plus the cost matrix (full form `{action: {outcome: cost}}` or short form `{wrong: W, defer: D}`).
fn act(
    q: &serde_json::Map<String, Value>,
    lang: Lang,
    hide: bool,
    name: &str,
) -> Result<(Parts, ActSpec), Vec<FieldError>> {
    let crit = q.get("criteria").cloned().unwrap_or(Value::Null);
    let yes_no = match &crit {
        Value::Object(m) => m.keys().all(|k| k == "true" || k == "false"),
        _ => false,
    } || q.get("base").and_then(Value::as_str) == Some("noul")
        || (crit.is_null() && q.get("base").is_none());
    let (base, built) = if yes_no {
        let mut qq = q.clone();
        if !matches!(crit, Value::Object(_)) {
            qq.remove("criteria");
        }
        (QType::Noul, noul(&qq, lang, name)?)
    } else {
        let base = match q.get("base").and_then(Value::as_str) {
            None | Some("choice") => QType::Choice,
            Some("score") => QType::Score,
            Some(b) => return Err(vec![qerr(name, &["base"], format!("unsupported act base {b:?}"), "literal_error")]),
        };
        (base, choice(q, hide, name)?)
    };
    let outcomes = &built.0;
    let costs = match q.get("costs") {
        Some(Value::Object(c)) if !c.is_empty() => c,
        _ => return Err(vec![qerr(name, &["costs"], "act needs costs", "missing")]),
    };
    let f = |v: &Value, path: &[&str]| -> Result<f64, Vec<FieldError>> {
        v.as_f64().ok_or_else(|| vec![qerr(name, path, "cost must be a number", "float_type")])
    };
    let short = costs.keys().all(|k| k == "wrong" || k == "defer") && !costs.get("wrong").is_some_and(Value::is_object);
    let (rows, defer) = if short {
        let w = f(costs.get("wrong").unwrap_or(&Value::Null), &["costs", "wrong"])?;
        let mut rows: Vec<(String, Vec<f64>)> = outcomes
            .iter()
            .map(|y| (y.clone(), outcomes.iter().map(|z| if z == y { 0.0 } else { w }).collect()))
            .collect();
        let mut defer = None;
        if let Some(d) = costs.get("defer") {
            let d = f(d, &["costs", "defer"])?;
            rows.push(("defer".into(), vec![d; outcomes.len()]));
            defer = Some("defer".to_string());
        }
        (rows, defer)
    } else {
        let mut rows = Vec::new();
        for (a, row) in costs {
            let Some(row) =
                row.as_object().filter(|r| r.len() == outcomes.len() && outcomes.iter().all(|z| r.contains_key(z)))
            else {
                return Err(vec![qerr(
                    name,
                    &["costs", a],
                    format!("costs[{a:?}] must give a cost for every outcome {outcomes:?}"),
                    "value_error",
                )]);
            };
            let mut v = Vec::with_capacity(outcomes.len());
            for z in outcomes {
                v.push(f(&row[z], &["costs", a, z])?);
            }
            rows.push((a.clone(), v));
        }
        let defer = match q.get("defer_action").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            Some(d) => Some(d.to_string()),
            None => ["defer", "human"].into_iter().find(|a| rows.iter().any(|(r, _)| r == a)).map(String::from),
        };
        (rows, defer)
    };
    let max_error = match q.get("max_error") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            py_str(v).ok_or_else(|| vec![qerr(name, &["max_error"], "max_error must be a number", "float_type")])?,
        ),
    };
    let spec = ActSpec { base, costs: rows, defer, max_error, hide, threshold: None, calibrated: false };
    Ok((built, spec))
}
fn noul(q: &serde_json::Map<String, Value>, lang: Lang, name: &str) -> Built {
    let crit = match q.get("criteria") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(v) => {
            return Err(vec![FieldError::new(
                qloc(name, &["criteria".into()]),
                "Input should be a valid dictionary or instance of NoulCriteria",
                "model_type",
            )
            .input(v)])
        }
    };
    let mut errs = Vec::new();
    let mut side = |key: &str, default: &str| -> String {
        match crit.get(key) {
            // Only a missing or null description falls back to the default text; "" / {} / [] are kept as given
            // (upstream used truthiness, see docs/SYSTEM_ONE.md).
            None | Some(Value::Null) => default.to_string(),
            Some(v) if is_description(v, false) => text(v),
            Some(v) => {
                errs.append(&mut union_errs(qloc(name, &["criteria".into(), key.into()]), v));
                String::new()
            }
        }
    };
    let (yes, no) = if lang == Lang::Pl { ("Tak", "Nie") } else { ("Yes", "No") };
    let opts = vec![side("true", yes), side("false", no)];
    if errs.is_empty() {
        Ok((vec!["true".into(), "false".into()], opts, vec![]))
    } else {
        Err(errs)
    }
}

fn choice(q: &serde_json::Map<String, Value>, hide: bool, name: &str) -> Built {
    let crit: IndexMap<String, Value> = match q.get("criteria") {
        Some(Value::Object(m)) => {
            let bad: Vec<_> = m
                .iter()
                .filter(|(_, v)| !is_description(v, true))
                .flat_map(|(k, v)| union_errs(qloc(name, &["criteria".into(), k.as_str().into()]), v))
                .collect();
            if !bad.is_empty() {
                return Err(bad);
            }
            to_map(m)
        }
        // Basal extension: a list of choice names (duplicates collapse, first position kept, as a Python dict).
        Some(Value::Array(xs)) => {
            let mut m = IndexMap::new();
            for (i, x) in xs.iter().enumerate() {
                match x {
                    Value::String(s) => {
                        m.entry(s.clone()).or_insert(Value::Null);
                    }
                    _ => {
                        return Err(vec![FieldError::new(
                            qloc(name, &["criteria".into(), i.into()]),
                            "Input should be a valid string",
                            "string_type",
                        )
                        .input(x)])
                    }
                }
            }
            m
        }
        Some(v) => {
            return Err(vec![FieldError::new(
                qloc(name, &["criteria".into()]),
                "Input should be a valid dictionary",
                "dict_type",
            )
            .input(v)])
        }
        None => return Err(vec![missing(qloc(name, &["criteria".into()]), &Value::Object(q.clone()))]),
    };
    if crit.is_empty() {
        return Err(vec![usage(format!("Choice question must have at least one choice: {name}").into())]);
    }
    if crit.len() > MAX_CHOICE_OPTIONS {
        return Err(vec![usage(format!("Too many choices. Must have at most {MAX_CHOICE_OPTIONS} choices.").into())]);
    }
    let (keys, opts) = named_options(&crit, hide);
    Ok((keys, opts, vec![]))
}

fn score(q: &serde_json::Map<String, Value>, hide: bool, name: &str) -> Built {
    // `levels` is accepted as an alias of `criteria` (upstream Basal extension).
    let (field, crit) = match (q.get("criteria"), q.get("levels")) {
        (Some(c), _) if !c.is_null() => ("criteria", c),
        (_, Some(l)) if !l.is_null() => ("levels", l),
        _ => return Err(vec![missing(qloc(name, &["criteria".into()]), &Value::Object(q.clone()))]),
    };
    let too_many = || usage(format!("Too many score levels. Must have at most {MAX_OPTIONS} levels.").into());
    match crit {
        Value::Array(xs) => {
            if xs.is_empty() {
                return Err(vec![too_short(qloc(name, &[field.into()]), crit)]);
            }
            let bad: Vec<_> = xs
                .iter()
                .enumerate()
                .filter(|(_, v)| !is_description(v, false))
                .flat_map(|(i, v)| union_errs(qloc(name, &[field.into(), i.into()]), v))
                .collect();
            if !bad.is_empty() {
                return Err(bad);
            }
            if xs.len() > MAX_OPTIONS {
                return Err(vec![too_many()]);
            }
            let keys = (0..xs.len()).map(|i| i.to_string()).collect();
            Ok((keys, xs.iter().map(text).collect(), xs.clone()))
        }
        // Basal extension: ordered {name: description}; answers keyed by the names.
        Value::Object(m) => {
            if m.is_empty() {
                return Err(vec![too_short(qloc(name, &[field.into()]), crit)]);
            }
            if m.len() > MAX_OPTIONS {
                return Err(vec![too_many()]);
            }
            let map = to_map(m);
            let (keys, opts) = named_options(&map, hide);
            Ok((keys, opts, map.values().cloned().collect()))
        }
        _ => Err(vec![
            FieldError::new(qloc(name, &[field.into()]), "Input should be a valid list", "list_type").input(crit)
        ]),
    }
}

/// Limits of the A–J letter readout (2..=10 options). Requests outside of it that the schema allows are answered with
/// an explicit error, see docs/SYSTEM_ONE.md.
/// `large_choice`: choice questions with 11..=255 options are answered by the grouped strategy
/// (`crate::large_choice`) instead of an error.
/// `single_option`: choice and score questions with one option are answered without the model (probability 1, as the
/// TypeSafe API does); without it they are an error too.
pub fn check_capability(items: &[Item], large_choice: bool, single_option: bool) -> Result<(), DecideError> {
    let mut errs = Vec::new();
    for it in items {
        let n = it.options.len();
        let path = ["questions", it.name.as_str(), "criteria"];
        let fixed = single_option && matches!(it.kind, QType::Choice | QType::Score) && matches!(it.ext, Ext::None);
        if n == 1 && !fixed {
            errs.push(err(
                &path,
                format!(
                    "{} with a single option/level is not supported yet: its expected behaviour is unresolved",
                    it.kind.as_str()
                ),
                "unsupported_single_option",
            ));
        } else if n > MAX_OPTIONS && !(large_choice && it.kind == QType::Choice) {
            errs.push(err(&path, format!("{n} options: the letter readout of the model reads 2..{MAX_OPTIONS}; only choice questions take more (the grouped 11..255 strategy)"), "unsupported_option_count"));
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(DecideError::Unsupported(errs))
    }
}
