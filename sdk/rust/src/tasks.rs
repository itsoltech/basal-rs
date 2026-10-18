//! Experimental Basal 2.1 tasks. Source offsets count Unicode scalar values, not UTF-8 bytes.
use crate::{Answer, AnswerKind, Error, MultiObjective, Question, QuestionSpec, QuestionType};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Certified automation by requested error target. `None` is an unavailable task certificate.
pub type Automation = IndexMap<String, Option<bool>>;
/// Claim verification verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Supported by the source.
    Supported,
    /// Contradicted by the source.
    Contradicted,
    /// Not stated in the source.
    NotStated,
    /// Binary verification did not support the claim.
    NotSupported,
}
impl Verdict {
    fn key(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Contradicted => "contradicted",
            Self::NotStated => "not_stated",
            Self::NotSupported => "not_supported",
        }
    }
}
/// Per-claim calibrated verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResult {
    /// Verdict.
    pub verdict: Verdict,
    /// Outcome probabilities.
    pub probabilities: IndexMap<String, f64>,
    /// Maximum outcome probability.
    pub confidence: f64,
    /// Certification for this task.
    pub automated: Automation,
}
/// Compression segmentation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionUnitKind {
    /// Sentence segmentation.
    #[default]
    Sentence,
    /// Nonempty lines.
    Line,
    /// Paragraphs separated by blank lines.
    Paragraph,
}
/// One compression unit in the original text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionUnit {
    /// Inclusive Unicode code point offset.
    pub start: usize,
    /// Exclusive Unicode code point offset.
    pub end: usize,
    /// Retention probability.
    pub probability: f64,
}
/// Extraction decision source.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtractSource {
    /// Enum/boolean decision.
    Choice,
    /// Source span selection.
    Span,
}
/// Extraction calibration scope; span products are not certified field probabilities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExtractCalibration {
    /// Categorical decision.
    #[serde(rename = "validated")]
    Validated,
    /// Each constituent decision is calibrated separately.
    #[serde(rename = "per decision")]
    PerDecision,
}
/// Extracted field, including missing values and source provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractField {
    /// Native JSON value, null when missing or numeric parsing failed.
    pub value: Value,
    /// Decision confidence or line/span product.
    pub confidence: f64,
    /// Choice or span selection.
    pub source: ExtractSource,
    /// Calibration scope.
    pub calibration: ExtractCalibration,
    /// Categorical probabilities.
    pub probabilities: Option<IndexMap<String, f64>>,
    /// Probability the field is stated.
    pub present_probability: Option<f64>,
    /// Verbatim source text, when selected.
    pub text: Option<String>,
    /// Inclusive Unicode code point offset.
    pub start: Option<usize>,
    /// Exclusive Unicode code point offset.
    pub end: Option<usize>,
    /// Line selection probability.
    pub line_probability: Option<f64>,
    /// Span selection probability.
    pub span_probability: Option<f64>,
    /// Numeric conversion diagnostic.
    pub error: Option<String>,
}
/// Checked single- or multi-label classification.
#[derive(Debug, Clone, Copy)]
pub struct ClassifyAnswer<'a> {
    /// Label scores.
    pub scores: &'a IndexMap<String, f64>,
    /// Single selected label; absent in multi mode.
    pub label: Option<&'a str>,
    /// Top label; absent in multi mode.
    pub top_label: Option<&'a str>,
    /// Single-label confidence.
    pub confidence: Option<f64>,
    /// Task certification; absent in multi mode.
    pub automated: Option<&'a Automation>,
    /// Multi-label set; absent in single mode.
    pub labels: Option<&'a [String]>,
    /// Multi-label set confidence.
    pub set_confidence: Option<f64>,
}
/// Checked claim verification.
#[derive(Debug, Clone, Copy)]
pub struct VerifyAnswer<'a> {
    /// Per-claim results.
    pub claims: &'a IndexMap<String, ClaimResult>,
    /// Whether every claim is supported.
    pub all_supported: bool,
    /// Flagged claim keys.
    pub not_supported: &'a [String],
}
/// Checked extractive compression.
#[derive(Debug, Clone, Copy)]
pub struct CompressAnswer<'a> {
    /// Compressed source text.
    pub compressed: &'a str,
    /// Kept source unit indices.
    pub kept: &'a [usize],
    /// Source spans and retention probabilities.
    pub units: &'a [CompressionUnit],
    /// Segmentation unit.
    pub unit: CompressionUnitKind,
    /// Original Unicode code point count.
    pub original_chars: usize,
    /// Compressed Unicode code point count.
    pub compressed_chars: usize,
    /// Character ratio.
    pub ratio: f64,
}
/// Checked structured extraction.
#[derive(Debug, Clone, Copy)]
pub struct ExtractAnswer<'a> {
    /// Nested output data.
    pub data: &'a IndexMap<String, Value>,
    /// Field provenance and calibrated constituent decisions.
    pub fields: &'a IndexMap<String, ExtractField>,
}
fn required<T>(v: Option<T>) -> Result<T, Error> {
    v.ok_or(Error::InvalidResponse("missing task answer field"))
}
fn probability(v: f64) -> Result<(), Error> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(Error::InvalidResponse("invalid task probability"))
    }
}
fn probabilities(v: &IndexMap<String, f64>) -> Result<(), Error> {
    v.values().try_for_each(|n| probability(*n))
}
macro_rules! task_kind {
    ($marker:ident,$view:ident,$variant:ident,$a:ident,$body:block) => {
        #[doc=concat!("Result marker for ",stringify!($variant)," tasks.")]
        #[derive(Debug)]
        pub struct $marker;
        impl crate::questions::sealed::Sealed for $marker {}
        impl AnswerKind for $marker {
            type Output<'a> = $view<'a>;
            fn parse($a: &Answer) -> Result<Self::Output<'_>, Error> {
                if !matches!($a.kind, QuestionType::$variant) || $a.experimental != Some(true) {
                    return Err(Error::InvalidResponse("mismatched experimental task"));
                }
                $body
            }
        }
    };
}
task_kind!(ClassifyKind, ClassifyAnswer, Classify, a, {
    let scores = required(a.scores.as_ref())?;
    probabilities(scores)?;
    if let Some(labels) = &a.labels {
        if labels.iter().any(|k| !scores.contains_key(k)) {
            return Err(Error::InvalidResponse("unknown classification label"));
        }
        probability(required(a.set_confidence)?)?;
    } else {
        let label = required(a.label.as_deref())?;
        if !scores.contains_key(label) || a.top_label.as_deref() != Some(label) {
            return Err(Error::InvalidResponse("invalid classification label"));
        }
        probability(required(a.confidence)?)?;
        required(a.automated.as_ref())?;
    }
    Ok(ClassifyAnswer {
        scores,
        label: a.label.as_deref(),
        top_label: a.top_label.as_deref(),
        confidence: a.confidence,
        automated: a.automated.as_ref(),
        labels: a.labels.as_deref(),
        set_confidence: a.set_confidence,
    })
});
task_kind!(VerifyKind, VerifyAnswer, Verify, a, {
    let claims = required(a.claims.as_ref())?;
    for c in claims.values() {
        probabilities(&c.probabilities)?;
        probability(c.confidence)?;
        if !c.probabilities.contains_key(c.verdict.key()) {
            return Err(Error::InvalidResponse("invalid claim verdict"));
        }
    }
    let flagged = required(a.not_supported.as_deref())?;
    if flagged.iter().any(|k| !claims.contains_key(k)) {
        return Err(Error::InvalidResponse("invalid flagged claim"));
    }
    Ok(VerifyAnswer { claims, all_supported: required(a.all_supported)?, not_supported: flagged })
});
task_kind!(CompressKind, CompressAnswer, Compress, a, {
    let units = required(a.units.as_deref())?;
    for u in units {
        probability(u.probability)?;
        if u.end < u.start {
            return Err(Error::InvalidResponse("reversed compression span"));
        }
    }
    let kept = required(a.kept.as_deref())?;
    if kept.iter().any(|i| *i >= units.len()) {
        return Err(Error::InvalidResponse("invalid kept unit"));
    }
    let ratio = required(a.ratio)?;
    if !ratio.is_finite() || ratio < 0.0 {
        return Err(Error::InvalidResponse("invalid compression ratio"));
    }
    Ok(CompressAnswer {
        compressed: required(a.compressed.as_deref())?,
        kept,
        units,
        unit: required(a.unit)?,
        original_chars: required(a.original_chars)?,
        compressed_chars: required(a.compressed_chars)?,
        ratio,
    })
});
task_kind!(ExtractKind, ExtractAnswer, Extract, a, {
    let fields = required(a.fields.as_ref())?;
    for f in fields.values() {
        probability(f.confidence)?;
        if let Some(p) = &f.probabilities {
            probabilities(p)?;
        }
        for p in [f.present_probability, f.line_probability, f.span_probability].into_iter().flatten() {
            probability(p)?;
        }
        if f.text.is_some() && required(f.end)? < required(f.start)? {
            return Err(Error::InvalidResponse("reversed extraction span"));
        }
    }
    Ok(ExtractAnswer { data: required(a.data.as_ref())?, fields })
});
/// Classification question definition.
pub type ClassifyQuestion = QuestionSpec<ClassifyKind>;
/// Verification question definition.
pub type VerifyQuestion = QuestionSpec<VerifyKind>;
/// Compression question definition.
pub type CompressQuestion = QuestionSpec<CompressKind>;
/// Extraction question definition.
pub type ExtractQuestion = QuestionSpec<ExtractKind>;
/// Experimental classification options.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassifyOptions<'a> {
    /// Independent multi-label classification.
    pub multi_label: bool,
    /// Add a none-of-the-above label in single mode.
    pub none: bool,
    /// Requested certified error targets; omitted preserves server defaults.
    pub max_error: Option<&'a [f64]>,
    /// Hide category names in the decision prompt.
    pub hide_keys: bool,
    /// Multi-label selection options.
    pub selection: crate::MultiOptions<'a>,
}
/// Experimental verification options.
#[derive(Debug, Clone, Copy, Default)]
pub struct VerifyOptions<'a> {
    /// Supported/not-supported instead of three outcomes.
    pub binary: bool,
    /// Requested certified error targets.
    pub max_error: Option<&'a [f64]>,
}
/// Extractive compression options.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompressOptions {
    /// Segmentation, sentences by default.
    pub unit: CompressionUnitKind,
    /// Character retention target, whole units may exceed it.
    pub keep_ratio: Option<f64>,
    /// Independent retention threshold.
    pub threshold: Option<f64>,
    /// Minimum retained units; omitted uses server default 1.
    pub min_keep: Option<usize>,
}
fn task(kind: QuestionType, instructions: &str) -> Question {
    let mut q = Question::new(kind, instructions);
    if instructions.is_empty() {
        q.instructions = None;
    }
    q
}
fn risks(q: &mut Question, values: Option<&[f64]>) -> Result<(), Error> {
    if let Some(values) = values {
        if values.len() > 64 || values.iter().any(|v| !v.is_finite() || *v <= 0.0 || *v >= 1.0) {
            return Err(Error::InvalidRequest("invalid task max_error"));
        }
        q.extra.insert("max_error".into(), serde_json::json!(values));
    }
    Ok(())
}
/// Classify using a label object or ordered list, normalized by the server.
pub fn classify(instructions: &str, labels: Value, options: ClassifyOptions<'_>) -> Result<ClassifyQuestion, Error> {
    let count = match &labels {
        Value::Object(m) => m.len(),
        Value::Array(a) => a.len(),
        _ => 0,
    };
    if !(1..=256).contains(&count) {
        return Err(Error::InvalidRequest("classify requires 1..256 labels"));
    }
    let mut q = task(QuestionType::Classify, instructions);
    q.extra.insert("labels".into(), labels);
    q.extra.insert("multi_label".into(), options.multi_label.into());
    q.extra.insert("none".into(), options.none.into());
    risks(&mut q, options.max_error)?;
    if options.hide_keys {
        q.extra.insert("option_keys".into(), "hide".into());
    }
    if options.multi_label {
        if !options.selection.exclusive.is_empty()
            || options.selection.fallback.is_some()
            || options.selection.fallback_threshold.is_some()
        {
            return Err(Error::InvalidRequest("classify does not support exclusive/fallback"));
        }
        q.extra.extend(crate::questions::selection_fields(options.selection, None)?);
    } else if options.selection.threshold.is_some()
        || options.selection.min.is_some()
        || options.selection.max.is_some()
        || !matches!(options.selection.objective, MultiObjective::Threshold)
        || !options.selection.exclusive.is_empty()
        || options.selection.fallback.is_some()
        || options.selection.fallback_threshold.is_some()
    {
        return Err(Error::InvalidRequest("selection options require multi_label"));
    }
    Ok(QuestionSpec::new(q))
}
/// Verify an ordered map or list of claims.
pub fn verify(instructions: &str, claims: Value, options: VerifyOptions<'_>) -> Result<VerifyQuestion, Error> {
    if match &claims {
        Value::Object(m) => m.is_empty(),
        Value::Array(a) => a.is_empty(),
        _ => true,
    } {
        return Err(Error::InvalidRequest("verify requires claims"));
    }
    let mut q = task(QuestionType::Verify, instructions);
    q.extra.insert("claims".into(), claims);
    q.extra.insert("binary".into(), options.binary.into());
    risks(&mut q, options.max_error)?;
    Ok(QuestionSpec::new(q))
}
/// Compress verbatim source units; the model chooses units, it does not generate text.
pub fn compress(instructions: &str, options: CompressOptions) -> Result<CompressQuestion, Error> {
    let mut q = task(QuestionType::Compress, instructions);
    q.extra.insert("unit".into(), serde_json::to_value(options.unit).map_err(Error::Encode)?);
    for (k, v) in [("keep_ratio", options.keep_ratio), ("threshold", options.threshold)] {
        if let Some(v) = v {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) || (k == "keep_ratio" && v == 0.0) {
                return Err(Error::InvalidRequest("invalid compression probability"));
            }
            q.extra.insert(k.into(), v.into());
        }
    }
    if let Some(n) = options.min_keep {
        q.extra.insert("min_keep".into(), n.into());
    }
    Ok(QuestionSpec::new(q))
}
/// Extract enum, boolean, scalar and nested object fields from a JSON schema.
pub fn extract(instructions: &str, schema: Value) -> Result<ExtractQuestion, Error> {
    if !schema.is_object() {
        return Err(Error::InvalidRequest("schema must be an object"));
    }
    let mut q = task(QuestionType::Extract, instructions);
    q.extra.insert("schema".into(), schema);
    Ok(QuestionSpec::new(q))
}
