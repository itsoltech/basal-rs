//! Typed question factories and checked result views, without a schema/runtime dependency.

use crate::{Answer, DecisionRequest, DecisionResponse, Error, EvidenceSpan, Question, QuestionType, Usage};
use indexmap::IndexMap;
use serde_json::{Map, Value};
use std::{marker::PhantomData, sync::Arc};

/// Display named criterion keys.
#[derive(Debug, Clone, Copy, Default)]
pub enum OptionKeys {
    /// Show names (native default).
    #[default]
    Show,
    /// Hide names.
    Hide,
}
/// Choice display options.
#[derive(Debug, Clone, Copy, Default)]
pub struct ChoiceOptions {
    /// Request supporting spans.
    pub evidence: bool,
    /// Show or hide keys.
    pub option_keys: OptionKeys,
}
/// Supporting spans for noul and score.
#[derive(Debug, Clone, Copy, Default)]
pub struct EvidenceOptions {
    /// Request supporting spans.
    pub evidence: bool,
}
/// Multi options intentionally have no evidence field.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultiOptions {
    /// Selection threshold in [0, 1], default 0.5.
    pub threshold: Option<f64>,
    /// Minimum number of labels.
    pub min: Option<usize>,
    /// Maximum number of labels.
    pub max: Option<usize>,
}
/// Act options. Omitted criteria use true/false outcomes.
#[derive(Debug, Clone, Copy, Default)]
pub struct ActOptions<'a> {
    /// Named outcomes, in request order.
    pub criteria: Option<&'a [(&'a str, &'a str)]>,
    /// Request supporting spans.
    pub evidence: bool,
    /// Action to defer to, which must exist in costs.
    pub defer_action: Option<&'a str>,
    /// Certified error target; calibration capability belongs to the model.
    pub max_error: Option<f64>,
}
/// Explicit action/outcome cost matrix.
pub type CostMatrix = IndexMap<String, IndexMap<String, f64>>;
/// Native facts preprocessing mode.
#[derive(Debug, Clone, Copy)]
pub enum FactsMode {
    /// No preprocessing.
    Off,
    /// Let the model preprocess facts.
    Auto,
}
/// Decision routing/preprocessing options.
#[derive(Debug, Clone, Copy, Default)]
pub struct DecisionOptions<'a> {
    /// Exact model name, or server default.
    pub model: Option<&'a str>,
    /// Native facts mode.
    pub facts: Option<FactsMode>,
}

mod sealed {
    pub trait Sealed {}
}
/// Closed association between a question and its required result fields.
pub trait AnswerKind: sealed::Sealed {
    /// Checked result borrowed from a response.
    type Output<'a>;
    /// Validate required fields and return the typed view.
    fn parse(answer: &Answer) -> Result<Self::Output<'_>, Error>;
}

/// Checked choice answer.
#[derive(Debug, Clone, Copy)]
pub struct ChoiceAnswer<'a> {
    /// Selected criterion (never optional).
    pub choice: &'a str,
    /// Native confidence.
    pub confidence: f64,
    /// Native probabilities.
    pub probabilities: &'a IndexMap<String, f64>,
    /// Supporting spans, empty if none were returned.
    pub evidence: &'a [EvidenceSpan],
    /// Original answer, including evidence/extensions.
    pub raw: &'a Answer,
}
/// Checked yes/no answer.
#[derive(Debug, Clone, Copy)]
pub struct NoulAnswer<'a> {
    /// P(yes), never optional.
    pub noul: f64,
    /// Native confidence.
    pub confidence: f64,
    /// Native probabilities.
    pub probabilities: &'a IndexMap<String, f64>,
    /// Supporting spans, empty if none were returned.
    pub evidence: &'a [EvidenceSpan],
    /// Original answer.
    pub raw: &'a Answer,
}
/// Checked score answer.
#[derive(Debug, Clone, Copy)]
pub struct ScoreAnswer<'a> {
    /// Weighted score, never optional.
    pub score: f64,
    /// Native confidence.
    pub confidence: f64,
    /// Ordered level legend.
    pub legend: &'a IndexMap<String, String>,
    /// Native probabilities.
    pub probabilities: &'a IndexMap<String, f64>,
    /// Supporting spans, empty if none were returned.
    pub evidence: &'a [EvidenceSpan],
    /// Original answer.
    pub raw: &'a Answer,
}
/// Checked multi answer.
#[derive(Debug, Clone, Copy)]
pub struct MultiAnswer<'a> {
    /// Selected labels, possibly empty.
    pub selected: &'a [String],
    /// Applied threshold.
    pub threshold: f64,
    /// Confidence of exactly the selected set.
    pub set_confidence: f64,
    /// Independent P(yes) values.
    pub probabilities: &'a IndexMap<String, f64>,
    /// Original answer.
    pub raw: &'a Answer,
}
/// Checked act answer.
#[derive(Debug, Clone, Copy)]
pub struct ActAnswer<'a> {
    /// Chosen action, never optional.
    pub action: &'a str,
    /// Most probable outcome.
    pub answer: &'a str,
    /// Native confidence.
    pub confidence: f64,
    /// Calibration status.
    pub calibration: &'a str,
    /// Expected cost per action.
    pub expected_costs: &'a IndexMap<String, f64>,
    /// Native probabilities.
    pub probabilities: &'a IndexMap<String, f64>,
    /// Supporting spans, empty if none were returned.
    pub evidence: &'a [EvidenceSpan],
    /// Original answer.
    pub raw: &'a Answer,
}

fn result_number(value: Option<f64>) -> Result<f64, Error> {
    value.filter(|v| v.is_finite()).ok_or(Error::InvalidResponse("missing or non-finite answer field"))
}
macro_rules! kind {
    ($name:ident, $view:ident, $variant:ident, $answer:ident, $body:block) => {
        #[doc = concat!("Result marker for ", stringify!($variant), " questions.")]
        #[derive(Debug)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl AnswerKind for $name {
            type Output<'a> = $view<'a>;
            fn parse($answer: &Answer) -> Result<Self::Output<'_>, Error> {
                if !matches!($answer.kind, QuestionType::$variant) {
                    return Err(Error::InvalidResponse("mismatched answer type"));
                }
                $body
            }
        }
    };
}
kind!(ChoiceKind, ChoiceAnswer, Choice, a, {
    Ok(ChoiceAnswer {
        choice: a.choice.as_deref().ok_or(Error::InvalidResponse("missing choice"))?,
        confidence: result_number(a.confidence)?,
        probabilities: &a.probabilities,
        evidence: a.evidence.as_deref().unwrap_or_default(),
        raw: a,
    })
});
kind!(NoulKind, NoulAnswer, Noul, a, {
    Ok(NoulAnswer {
        noul: result_number(a.noul)?,
        confidence: result_number(a.confidence)?,
        probabilities: &a.probabilities,
        evidence: a.evidence.as_deref().unwrap_or_default(),
        raw: a,
    })
});
kind!(ScoreKind, ScoreAnswer, Score, a, {
    Ok(ScoreAnswer {
        score: result_number(a.score)?,
        confidence: result_number(a.confidence)?,
        legend: a.legend.as_ref().ok_or(Error::InvalidResponse("missing legend"))?,
        probabilities: &a.probabilities,
        evidence: a.evidence.as_deref().unwrap_or_default(),
        raw: a,
    })
});
kind!(MultiKind, MultiAnswer, Multi, a, {
    Ok(MultiAnswer {
        selected: a.selected.as_deref().ok_or(Error::InvalidResponse("missing selected labels"))?,
        threshold: result_number(a.threshold)?,
        set_confidence: result_number(a.set_confidence)?,
        probabilities: &a.probabilities,
        raw: a,
    })
});
kind!(ActKind, ActAnswer, Act, a, {
    let costs = a.expected_costs.as_ref().ok_or(Error::InvalidResponse("missing expected costs"))?;
    if costs.values().any(|v| !v.is_finite()) {
        return Err(Error::InvalidResponse("non-finite expected cost"));
    }
    let calibration = a
        .calibration
        .as_deref()
        .filter(|v| matches!(*v, "validated" | "unvalidated"))
        .ok_or(Error::InvalidResponse("invalid calibration"))?;
    Ok(ActAnswer {
        action: a.action.as_deref().ok_or(Error::InvalidResponse("missing action"))?,
        answer: a.answer.as_deref().ok_or(Error::InvalidResponse("missing outcome"))?,
        confidence: result_number(a.confidence)?,
        calibration,
        expected_costs: costs,
        probabilities: &a.probabilities,
        evidence: a.evidence.as_deref().unwrap_or_default(),
        raw: a,
    })
});

/// Immutable typed question definition. It can be reused under multiple request keys.
#[derive(Debug)]
pub struct QuestionSpec<A: AnswerKind> {
    data: Arc<Question>,
    marker: PhantomData<A>,
}
impl<A: AnswerKind> Clone for QuestionSpec<A> {
    fn clone(&self) -> Self {
        Self { data: self.data.clone(), marker: PhantomData }
    }
}
impl<A: AnswerKind> QuestionSpec<A> {
    fn new(question: Question) -> Self {
        Self { data: Arc::new(question), marker: PhantomData }
    }
    /// Borrow the immutable native question.
    pub fn to_wire(&self) -> &Question {
        &self.data
    }
}
/// A typed choice question.
pub type ChoiceQuestion = QuestionSpec<ChoiceKind>;
/// A typed yes/no question.
pub type NoulQuestion = QuestionSpec<NoulKind>;
/// A typed score question.
pub type ScoreQuestion = QuestionSpec<ScoreKind>;
/// A typed multi question.
pub type MultiQuestion = QuestionSpec<MultiKind>;
/// A typed act question.
pub type ActQuestion = QuestionSpec<ActKind>;

fn text(value: &str) -> Result<(), Error> {
    if value.trim().is_empty() {
        Err(Error::InvalidRequest("expected nonempty text"))
    } else {
        Ok(())
    }
}
fn finite(value: f64, probability: bool) -> Result<Value, Error> {
    if !value.is_finite() || (probability && !(0.0..=1.0).contains(&value)) {
        return Err(Error::InvalidRequest("invalid finite number/range"));
    }
    Ok(Value::from(value))
}
fn criteria(values: &[(&str, &str)], min: usize, max: usize) -> Result<Value, Error> {
    if values.len() < min || values.len() > max {
        return Err(Error::InvalidRequest("invalid number of criteria"));
    }
    let mut fields = Map::new();
    for &(key, value) in values {
        text(key)?;
        text(value)?;
        if fields.insert(key.into(), Value::String(value.into())).is_some() {
            return Err(Error::InvalidRequest("duplicate criterion key"));
        }
    }
    Ok(Value::Object(fields))
}
fn base(kind: QuestionType, instructions: &str, evidence: bool) -> Result<Question, Error> {
    text(instructions)?;
    let mut q = Question::new(kind, instructions);
    if evidence {
        q.extra.insert("evidence".into(), Value::Bool(true));
    }
    Ok(q)
}
/// Choose one of 2–10 ordered named options.
pub fn choice_with_options(
    instructions: &str,
    values: &[(&str, &str)],
    options: ChoiceOptions,
) -> Result<ChoiceQuestion, Error> {
    let mut q = base(QuestionType::Choice, instructions, options.evidence)?;
    q.criteria = Some(criteria(values, 2, 10)?);
    if matches!(options.option_keys, OptionKeys::Hide) {
        q.extra.insert("option_keys".into(), Value::String("hide".into()));
    }
    Ok(QuestionSpec::new(q))
}
/// Read a yes/no probability.
pub fn noul_with_options(instructions: &str, options: EvidenceOptions) -> Result<NoulQuestion, Error> {
    Ok(QuestionSpec::new(base(QuestionType::Noul, instructions, options.evidence)?))
}
/// Score over 2–10 ordered text levels.
pub fn score_with_options(
    instructions: &str,
    levels: &[&str],
    options: EvidenceOptions,
) -> Result<ScoreQuestion, Error> {
    if !(2..=10).contains(&levels.len()) {
        return Err(Error::InvalidRequest("score requires 2–10 levels"));
    }
    let mut values = Vec::with_capacity(levels.len());
    for &level in levels {
        text(level)?;
        values.push(Value::String(level.into()));
    }
    let mut q = base(QuestionType::Score, instructions, options.evidence)?;
    q.criteria = Some(Value::Array(values));
    Ok(QuestionSpec::new(q))
}
/// Select labels independently; options have no evidence.
pub fn multi_with_options(
    instructions: &str,
    values: &[(&str, &str)],
    options: MultiOptions,
) -> Result<MultiQuestion, Error> {
    let mut q = base(QuestionType::Multi, instructions, false)?;
    q.criteria = Some(criteria(values, 1, usize::MAX)?);
    if let Some(t) = options.threshold {
        q.extra.insert("threshold".into(), finite(t, true)?);
    }
    for (name, limit) in [("min", options.min), ("max", options.max)] {
        if let Some(limit) = limit {
            if limit > values.len() {
                return Err(Error::InvalidRequest("selection limit exceeds labels"));
            }
            q.extra.insert(name.into(), Value::from(limit));
        }
    }
    if options.min.zip(options.max).is_some_and(|(min, max)| min > max) {
        return Err(Error::InvalidRequest("min exceeds max"));
    }
    Ok(QuestionSpec::new(q))
}
/// Select an action from a full cost matrix. Omitted criteria use true/false outcomes.
pub fn act_with_options(instructions: &str, costs: &CostMatrix, options: ActOptions<'_>) -> Result<ActQuestion, Error> {
    if costs.is_empty() {
        return Err(Error::InvalidRequest("act requires costs"));
    }
    let mut q = base(QuestionType::Act, instructions, options.evidence)?;
    if let Some(values) = options.criteria {
        q.criteria = Some(criteria(values, 2, 10)?);
    }
    let outcomes: Vec<&str> =
        options.criteria.map_or_else(|| vec!["true", "false"], |values| values.iter().map(|(key, _)| *key).collect());
    let mut matrix = Map::new();
    for (action, row) in costs {
        text(action)?;
        if row.len() != outcomes.len() || outcomes.iter().any(|key| !row.contains_key(*key)) {
            return Err(Error::InvalidRequest("each cost row must cover every outcome exactly"));
        }
        let mut values = Map::new();
        for (key, value) in row {
            values.insert(key.clone(), finite(*value, false)?);
        }
        matrix.insert(action.clone(), Value::Object(values));
    }
    q.extra.insert("costs".into(), Value::Object(matrix));
    if let Some(action) = options.defer_action {
        if !costs.contains_key(action) {
            return Err(Error::InvalidRequest("unknown defer action"));
        }
        q.extra.insert("defer_action".into(), Value::String(action.into()));
    }
    if let Some(error) = options.max_error {
        q.extra.insert("max_error".into(), finite(error, true)?);
    }
    Ok(QuestionSpec::new(q))
}

type Validator = fn(&Answer) -> Result<(), Error>;
#[derive(Debug, Clone)]
struct Entry {
    data: Arc<Question>,
    validate: Validator,
}
/// Ordered heterogeneous request builder. Request keys identify answers, not definition identity.
#[derive(Debug, Clone, Default)]
pub struct Questions {
    entries: IndexMap<String, Entry>,
}
impl Questions {
    /// Start an empty builder; sending it is an input error.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add a typed question under a unique ID. One definition may be used under several IDs.
    pub fn with<A: AnswerKind>(mut self, id: &str, question: &QuestionSpec<A>) -> Result<Self, Error> {
        text(id)?;
        if self.entries.contains_key(id) {
            return Err(Error::InvalidRequest("duplicate question ID"));
        }
        self.entries.insert(id.into(), Entry { data: question.data.clone(), validate: |a| A::parse(a).map(|_| ()) });
        Ok(self)
    }
    pub(crate) fn request(&self, state: Value, options: DecisionOptions<'_>) -> Result<DecisionRequest, Error> {
        if state.is_null() || self.entries.is_empty() {
            return Err(Error::InvalidRequest("state and questions must not be empty/null"));
        }
        let mut request = DecisionRequest::new(state);
        if let Some(model) = options.model {
            text(model)?;
            request.model = Some(model.into());
        }
        if let Some(facts) = options.facts {
            request.extra.insert(
                "facts".into(),
                Value::String(
                    match facts {
                        FactsMode::Off => "off",
                        FactsMode::Auto => "auto",
                    }
                    .into(),
                ),
            );
        }
        for (id, entry) in &self.entries {
            request.insert_question(id, &entry.data)?;
        }
        Ok(request)
    }
    pub(crate) fn parse<S: AnswerSchema>(
        self,
        raw: DecisionResponse,
        schema: S,
    ) -> Result<TypedDecisionResponse<S>, Error> {
        for (id, entry) in &self.entries {
            let answer = raw.answers.get(id).ok_or(Error::InvalidResponse("missing answer"))?;
            (entry.validate)(answer)?;
            if answer.probabilities.values().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v)) {
                return Err(Error::InvalidResponse("invalid probabilities"));
            }
            if let Some(key) = &answer.choice {
                if entry.data.criteria.as_ref().is_none_or(|v| v.get(key).is_none()) {
                    return Err(Error::InvalidResponse("unknown choice"));
                }
            }
            if let Some(keys) = &answer.selected {
                if keys.iter().any(|key| entry.data.criteria.as_ref().is_none_or(|v| v.get(key).is_none())) {
                    return Err(Error::InvalidResponse("unknown multi label"));
                }
            }
            if let Some(action) = &answer.action {
                if entry.data.extra.get("costs").is_none_or(|v| v.get(action).is_none()) {
                    return Err(Error::InvalidResponse("unknown action"));
                }
            }
        }
        Ok(TypedDecisionResponse { raw, questions: self, schema })
    }
}
/// Checked answer selected by a runtime question key.
#[derive(Debug, Clone, Copy)]
pub enum AnswerResult<'a> {
    /// One selected criterion.
    Choice(ChoiceAnswer<'a>),
    /// Yes/no probability.
    Noul(NoulAnswer<'a>),
    /// Weighted score.
    Score(ScoreAnswer<'a>),
    /// Independently selected labels.
    Multi(MultiAnswer<'a>),
    /// Selected action.
    Act(ActAnswer<'a>),
}

impl<'a> AnswerResult<'a> {
    fn parse(answer: &'a Answer) -> Result<Self, Error> {
        match answer.kind {
            QuestionType::Choice => ChoiceKind::parse(answer).map(Self::Choice),
            QuestionType::Noul => NoulKind::parse(answer).map(Self::Noul),
            QuestionType::Score => ScoreKind::parse(answer).map(Self::Score),
            QuestionType::Multi => MultiKind::parse(answer).map(Self::Multi),
            QuestionType::Act => ActKind::parse(answer).map(Self::Act),
        }
    }
}

/// Constructs borrowed answer fields for a request shape.
///
/// Usually generated by [`crate::questions!`]. Implementations must use request keys to select fields;
/// missing keys or mismatched answer kinds must return an error.
pub trait AnswerSchema: Clone {
    /// Named fields or a map of checked answer variants, borrowed from the response.
    type Answers<'a>;
    /// Build checked views without copying probability maps or borrowing question definitions.
    fn parse<'a>(&self, raw: &'a DecisionResponse, questions: &'a Questions) -> Result<Self::Answers<'a>, Error>;
}

/// Schema for a runtime-keyed [`Questions`] map.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamicSchema;

impl AnswerSchema for DynamicSchema {
    type Answers<'a> = IndexMap<&'a str, AnswerResult<'a>>;

    fn parse<'a>(&self, raw: &'a DecisionResponse, questions: &'a Questions) -> Result<Self::Answers<'a>, Error> {
        questions
            .entries
            .keys()
            .map(|id| {
                let answer = raw.answers.get(id).ok_or(Error::InvalidResponse("missing answer"))?;
                Ok((id.as_str(), AnswerResult::parse(answer)?))
            })
            .collect()
    }
}

/// Request questions accepted by [`crate::Client::decide`].
///
/// Use [`Questions`] for dynamic keys or [`crate::questions!`] for inferred named answer fields.
pub trait QuestionSet: sealed::Sealed {
    /// Response shape retained from these questions.
    type Schema: AnswerSchema;
    /// Ordered, validated question definitions.
    fn as_questions(&self) -> &Questions;
    /// Response field schema, independent of question object identities.
    fn schema(&self) -> Self::Schema;
}

impl sealed::Sealed for Questions {}
impl QuestionSet for Questions {
    type Schema = DynamicSchema;
    fn as_questions(&self) -> &Questions {
        self
    }
    fn schema(&self) -> Self::Schema {
        DynamicSchema
    }
}

/// Ordered question definitions together with their named response shape.
/// Usually constructed by [`crate::questions!`].
#[derive(Debug, Clone)]
pub struct PreparedQuestions<S: AnswerSchema> {
    questions: Questions,
    schema: S,
}

impl<S: AnswerSchema> PreparedQuestions<S> {
    /// Combine definitions with a custom response schema. Prefer [`crate::questions!`].
    pub fn new(questions: Questions, schema: S) -> Self {
        Self { questions, schema }
    }
}

impl<S: AnswerSchema> sealed::Sealed for PreparedQuestions<S> {}
impl<S: AnswerSchema> QuestionSet for PreparedQuestions<S> {
    type Schema = S;
    fn as_questions(&self) -> &Questions {
        &self.questions
    }
    fn schema(&self) -> Self::Schema {
        self.schema.clone()
    }
}

/// Validated response retaining request keys and an optional named answer shape.
#[derive(Debug)]
pub struct TypedDecisionResponse<S: AnswerSchema = DynamicSchema> {
    raw: DecisionResponse,
    questions: Questions,
    schema: S,
}
impl<S: AnswerSchema> TypedDecisionResponse<S> {
    /// Native response, borrowed immutably for extension fields.
    pub fn raw(&self) -> &DecisionResponse {
        &self.raw
    }
    /// Model actually used.
    pub fn model(&self) -> &str {
        &self.raw.model
    }
    /// Native usage.
    pub fn usage(&self) -> &Usage {
        &self.raw.usage
    }
    /// Borrow named fields inferred by [`crate::questions!`], or a map for dynamic questions.
    /// The normal macro-generated schema has already been validated before the client returns.
    pub fn answers(&self) -> Result<S::Answers<'_>, Error> {
        self.schema.parse(&self.raw, &self.questions)
    }
    fn answer_raw(&self, id: &str) -> Result<&Answer, Error> {
        if !self.questions.entries.contains_key(id) {
            return Err(Error::InvalidRequest("question key was not part of this response"));
        }
        self.raw.answers.get(id).ok_or(Error::InvalidResponse("missing answer"))
    }
    /// Look up a checked answer by its request key. Match the enum for dynamic heterogeneous keys.
    pub fn answer(&self, id: &str) -> Result<AnswerResult<'_>, Error> {
        AnswerResult::parse(self.answer_raw(id)?)
    }
    /// Look up a choice answer by key; a different question kind is an error.
    pub fn choice(&self, id: &str) -> Result<ChoiceAnswer<'_>, Error> {
        ChoiceKind::parse(self.answer_raw(id)?)
    }
    /// Look up a yes/no answer by key.
    pub fn noul(&self, id: &str) -> Result<NoulAnswer<'_>, Error> {
        NoulKind::parse(self.answer_raw(id)?)
    }
    /// Look up a score answer by key.
    pub fn score(&self, id: &str) -> Result<ScoreAnswer<'_>, Error> {
        ScoreKind::parse(self.answer_raw(id)?)
    }
    /// Look up a multi answer by key.
    pub fn multi(&self, id: &str) -> Result<MultiAnswer<'_>, Error> {
        MultiKind::parse(self.answer_raw(id)?)
    }
    /// Look up an action answer by key.
    pub fn act(&self, id: &str) -> Result<ActAnswer<'_>, Error> {
        ActKind::parse(self.answer_raw(id)?)
    }
}

/// Build a named request shape with inferred, checked answer fields.
///
/// Field names become wire keys. Expressions are evaluated once, and one definition can be reused.
/// No procedural macro, code generator or runtime dependency is required.
///
/// ```rust,ignore
/// let questions = questions! { department: &routing, second_department: &routing }?;
/// let response = client.decide(state, &questions).await?;
/// let answers = response.answers()?;
/// println!("{} {}", answers.department.choice, answers.second_department.choice);
/// ```
#[macro_export]
macro_rules! questions {
    ($($field:ident : $question:expr),+ $(,)?) => {{
        // The caller's field identifiers also carry inferred answer-kind type parameters.
        #[allow(non_camel_case_types, reason = "macro type parameters reuse caller field names")]
        mod __basal_question_shape {
            pub struct Schema<$($field: $crate::AnswerKind),+> {
                marker: ::std::marker::PhantomData<($($field,)+)>,
            }

            impl<$($field: $crate::AnswerKind),+> ::std::fmt::Debug for self::Schema<$($field),+> {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    f.write_str("QuestionSchema")
                }
            }

            impl<$($field: $crate::AnswerKind),+> ::std::clone::Clone for self::Schema<$($field),+> {
                fn clone(&self) -> Self { Self { marker: ::std::marker::PhantomData } }
            }

            pub struct Answers<'a, $($field: $crate::AnswerKind),+> {
                $(pub $field: <$field as $crate::AnswerKind>::Output<'a>,)+
            }

            impl<$($field: $crate::AnswerKind),+> $crate::AnswerSchema for self::Schema<$($field),+> {
                type Answers<'a> = self::Answers<'a, $($field),+>;

                fn parse<'a>(
                    &self,
                    raw: &'a $crate::DecisionResponse,
                    _: &'a $crate::Questions,
                ) -> ::std::result::Result<Self::Answers<'a>, $crate::Error> {
                    ::std::result::Result::Ok(self::Answers {
                        $($field: {
                            let id = ::std::stringify!($field);
                            let id = id.strip_prefix("r#").unwrap_or(id);
                            let answer = raw.answers.get(id).ok_or($crate::Error::InvalidResponse("missing answer"))?;
                            <$field as $crate::AnswerKind>::parse(answer)?
                        },)+
                    })
                }
            }

            pub fn build<$($field: $crate::AnswerKind),+>(
                $($field: &$crate::QuestionSpec<$field>),+
            ) -> ::std::result::Result<$crate::PreparedQuestions<self::Schema<$($field),+>>, $crate::Error> {
                let questions = $crate::Questions::new();
                $(let id = ::std::stringify!($field);
                  let id = id.strip_prefix("r#").unwrap_or(id);
                  let questions = questions.with(id, $field)?;)+
                ::std::result::Result::Ok($crate::PreparedQuestions::new(questions, self::Schema { marker: ::std::marker::PhantomData }))
            }
        }
        // This binding extends factory temporaries and infers block expressions before coercing references.
        let ($($field,)+) = ($(&$question,)+);
        __basal_question_shape::build($($field),+)
    }};
}

/// Choose from 2–10 options with default display settings.
pub fn choice(instructions: &str, values: &[(&str, &str)]) -> Result<ChoiceQuestion, Error> {
    choice_with_options(instructions, values, ChoiceOptions::default())
}
/// Yes/no probability with default options.
pub fn noul(instructions: &str) -> Result<NoulQuestion, Error> {
    noul_with_options(instructions, EvidenceOptions::default())
}
/// Score over ordered levels with default options.
pub fn score(instructions: &str, levels: &[&str]) -> Result<ScoreQuestion, Error> {
    score_with_options(instructions, levels, EvidenceOptions::default())
}
/// Multi selection with default threshold and no count limits.
pub fn multi(instructions: &str, values: &[(&str, &str)]) -> Result<MultiQuestion, Error> {
    multi_with_options(instructions, values, MultiOptions::default())
}
/// Act over true/false outcomes with default options.
pub fn act(instructions: &str, costs: &CostMatrix) -> Result<ActQuestion, Error> {
    act_with_options(instructions, costs, ActOptions::default())
}
