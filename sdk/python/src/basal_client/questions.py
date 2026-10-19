"""Typed, validated question factories. Wire dictionaries remain available separately."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Generic, Literal, Self, TypeVar, cast, overload

from .task_types import TaskAnswer
from .types import Calibration, DecisionResponse, EvidenceSpan, JsonValue, MultiObjective, Question as WireQuestion, ToolRanking, Usage

OptionKeys = Literal["show", "hide"]


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]
    evidence: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    noul: float
    confidence: float
    probabilities: dict[str, float]
    evidence: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    score: float
    confidence: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    evidence: tuple[EvidenceSpan, ...] = ()


@dataclass(frozen=True, slots=True)
class MultiAnswer:
    selected: tuple[str, ...]
    threshold: float | None
    set_confidence: float
    probabilities: dict[str, float]
    objective: MultiObjective | None = None
    expected_score: float | None = None
    constraints_applied: JsonValue = None


@dataclass(frozen=True, slots=True)
class ToolAnswer:
    call: bool
    probabilities: dict[str, float]
    tool: str | None = None
    confidence: float | None = None
    multi: bool = False
    tools: tuple[str, ...] = ()
    ranking: tuple[ToolRanking, ...] = ()
    selection: MultiAnswer | None = None
    deferred: bool | None = None
    refused: str | None = None
    max_error: str | None = None


@dataclass(frozen=True, slots=True)
class ActAnswer:
    action: str
    answer: str
    expected_costs: dict[str, float]
    confidence: float
    calibration: Calibration
    probabilities: dict[str, float]
    evidence: tuple[EvidenceSpan, ...] = ()
    refused: str | None = None
    max_error: str | None = None
    deferred: bool | None = None


TAnswer = TypeVar("TAnswer")
AnswerResult = ChoiceAnswer | NoulAnswer | ScoreAnswer | MultiAnswer | ActAnswer | ToolAnswer | TaskAnswer
TSchema = TypeVar("TSchema", bound="Questions")
TSchema_co = TypeVar("TSchema_co", bound="Questions", covariant=True)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _criteria(values: Mapping[str, str], minimum: int, maximum: int | None) -> dict[str, JsonValue]:
    if not isinstance(values, Mapping) or len(values) < minimum or (maximum is not None and len(values) > maximum):
        raise ValueError("invalid number of criteria")
    return {_text(key, "criterion key"): _text(value, "criterion description") for key, value in values.items()}


def _number(value: float, name: str, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if probability and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


class QuestionSpec(Generic[TAnswer]):
    """Typed question. to_dict() returns a copy; read() checks its answer's required fields."""

    def __init__(self, kind: str, instructions: str, fields: dict[str, JsonValue]):
        self._payload: dict[str, JsonValue] = {"type": kind, "instructions": _text(instructions, "instructions"), **fields}

    def to_dict(self) -> WireQuestion:
        return cast(WireQuestion, deepcopy(self._payload))

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Questions, owner: type[Any] | None = None) -> TAnswer: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Self: ...

    def __get__(self, instance: object, owner: type[Any] | None = None) -> Self | TAnswer:
        # Non-data descriptor: validated answers stored under individual instance
        # keys shadow the definition. Reusing one definition never merges keys.
        # Outside a Questions schema, definitions remain ordinary values, also
        # when stored in namespace classes, instance fields or dataclasses.
        if not isinstance(instance, Questions):
            return self
        raise AttributeError("Question answers are populated by client.decide")

    def read(self, response: DecisionResponse, question_id: str) -> TAnswer:
        from .client import BasalProtocolError
        answer = response.get("answers", {}).get(question_id)
        if not isinstance(answer, dict) or answer.get("type") != self._payload["type"]:
            raise BasalProtocolError("Basal returned a missing or mismatched answer")
        data = cast(dict[str, Any], answer)
        try:
            probabilities = data["probabilities"]
            if not isinstance(probabilities, dict) or not probabilities or any(not isinstance(k, str) for k in probabilities):
                raise ValueError("invalid probabilities")
            for value in probabilities.values():
                _number(value, "probability", True)
            spans = data.get("evidence", [])
            if not isinstance(spans, list):
                raise ValueError("invalid evidence")
            for span in spans:
                if not isinstance(span, dict) or not isinstance(span.get("text"), str):
                    raise ValueError("invalid span")
                for field in ("start", "end"):
                    if isinstance(span.get(field), bool) or not isinstance(span.get(field), int) or span[field] < 0:
                        raise ValueError("invalid offset")
                _number(span["probability"], "span probability", True)
            evidence = tuple(cast(list[EvidenceSpan], spans))
            kind = data["type"]
            result: AnswerResult
            if kind == "choice":
                selected = _text(data["choice"], "choice")
                if selected not in cast(dict[str, JsonValue], self._payload["criteria"]):
                    raise ValueError("unknown choice")
                result = ChoiceAnswer(selected, _number(data["confidence"], "confidence", True), probabilities, evidence)
            elif kind == "noul":
                result = NoulAnswer(_number(data["noul"], "noul", True), _number(data["confidence"], "confidence", True), probabilities, evidence)
            elif kind == "score":
                legend = data["legend"]
                if not isinstance(legend, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in legend.items()):
                    raise ValueError("invalid legend")
                result = ScoreAnswer(_number(data["score"], "score"), _number(data["confidence"], "confidence", True), legend, probabilities, evidence)
            elif kind == "multi":
                result = _selection(data, "selected", set(cast(dict[str, JsonValue], self._payload["criteria"])), self._payload.get("objective", "threshold"))
            elif kind == "tool":
                if not isinstance(data.get("call"), bool):
                    raise ValueError("invalid tool call flag")
                is_multi = self._payload.get("multi", False)
                if not isinstance(data.get("multi", False), bool) or data.get("multi", False) != is_multi:
                    raise ValueError("mismatched tool selection mode")
                selection = _selection(data, "tools", set(probabilities), self._payload.get("objective", "threshold")) if is_multi else None
                selected_tool = data.get("tool")
                if not is_multi and (selected_tool is not None and selected_tool not in probabilities):
                    raise ValueError("unknown tool")
                if not is_multi and selected_tool is None and data.get("deferred") is not True:
                    raise ValueError("missing tool")
                ranking = data.get("ranking", [])
                if not isinstance(ranking, list):
                    raise ValueError("invalid ranking")
                for entry in ranking:
                    if not isinstance(entry, dict) or entry.get("tool") not in probabilities:
                        raise ValueError("invalid ranking entry")
                    _number(entry["probability"], "ranking probability", True)
                result = ToolAnswer(data["call"], probabilities, tool=selected_tool,
                    confidence=None if is_multi else _number(data["confidence"], "confidence", True),
                    multi=bool(is_multi), tools=selection.selected if selection else (), ranking=tuple(ranking),
                    selection=selection, deferred=_optional_bool(data, "deferred"),
                    refused=_optional_text(data, "refused"), max_error=_optional_text(data, "max_error"))
            else:
                action = _text(data["action"], "action")
                if action not in cast(dict[str, JsonValue], self._payload["costs"]) or data["calibration"] not in ("validated", "unvalidated", "text"):
                    raise ValueError("invalid action or calibration")
                costs = data["expected_costs"]
                if not isinstance(costs, dict) or any(not isinstance(k, str) for k in costs):
                    raise ValueError("invalid expected costs")
                for value in costs.values():
                    _number(value, "expected cost")
                refused, max_error = data.get("refused"), data.get("max_error")
                if any(value is not None and not isinstance(value, str) for value in (refused, max_error)):
                    raise ValueError("invalid act metadata")
                result = ActAnswer(action, _text(data["answer"], "answer"), costs, _number(data["confidence"], "confidence", True), data["calibration"], probabilities, evidence, refused, max_error, _optional_bool(data, "deferred"))
            return cast(TAnswer, result)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise BasalProtocolError("Basal returned an invalid typed answer") from error


def _optional_bool(data: dict[str, Any], key: str) -> bool | None:
    value = data.get(key)
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"invalid {key}")
    return value


def _optional_text(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"invalid {key}")
    return value


def _selection(data: dict[str, Any], field: str, keys: set[str], objective: JsonValue) -> MultiAnswer:
    labels = data[field]
    if not isinstance(labels, list) or any(not isinstance(k, str) or k not in keys for k in labels):
        raise ValueError("invalid selected labels")
    if objective == "threshold":
        threshold = _number(data["threshold"], "threshold", True)
        expected_score = None
    else:
        if objective not in ("exact", "iou", "f1") or data.get("objective") != objective:
            raise ValueError("invalid objective")
        threshold = None
        expected_score = None if data.get("expected_score") is None else _number(data["expected_score"], "expected_score", True)
    return MultiAnswer(tuple(labels), threshold, _number(data["set_confidence"], "set_confidence", True),
        data["probabilities"], cast(MultiObjective, objective), expected_score, data.get("constraints_applied"))


def _selection_options(keys: set[str] | None, *, objective: MultiObjective = "threshold", threshold: float | None = None,
        min: int | None = None, max: int | None = None, exclusive: Sequence[str] | None = None,
        fallback: str | None = None, fallback_threshold: float | None = None) -> dict[str, JsonValue]:
    if objective not in ("threshold", "exact", "iou", "f1"):
        raise ValueError("invalid multi objective")
    fields: dict[str, JsonValue] = {}
    if objective != "threshold":
        fields["objective"] = objective
    if threshold is not None:
        if objective != "threshold":
            raise ValueError("threshold applies only to objective threshold")
        fields["threshold"] = _number(threshold, "threshold", True)
    for name, value in (("min", min), ("max", max)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (keys is not None and value > len(keys)):
                raise ValueError(f"invalid {name}")
            fields[name] = value
    if min is not None and max is not None and min > max:
        raise ValueError("min exceeds max")
    if exclusive is not None:
        if isinstance(exclusive, str):
            raise ValueError("exclusive must be a sequence of keys")
        fields["exclusive"] = [_text(key, "exclusive key") for key in exclusive]
        if keys is not None and any(key not in keys for key in exclusive):
            raise ValueError("unknown exclusive key")
    if fallback is not None:
        fields["fallback"] = _text(fallback, "fallback")
        if keys is not None and fallback not in keys:
            raise ValueError("unknown fallback")
    if fallback_threshold is not None:
        if fallback is None:
            raise ValueError("fallback_threshold requires fallback")
        fields["fallback_threshold"] = _number(fallback_threshold, "fallback_threshold", True)
    return fields


def _display(evidence: bool, option_keys: OptionKeys) -> dict[str, JsonValue]:
    if not isinstance(evidence, bool) or option_keys not in ("show", "hide"):
        raise ValueError("invalid evidence or option_keys")
    fields: dict[str, JsonValue] = {}
    if evidence:
        fields["evidence"] = True
    if option_keys != "show":
        fields["option_keys"] = option_keys
    return fields


class ChoiceQuestion(QuestionSpec[ChoiceAnswer]):
    def __init__(self, instructions: str, criteria: Mapping[str, str], *, evidence: bool = False,
                 option_keys: OptionKeys = "show"):
        super().__init__("choice", instructions, {"criteria": _criteria(criteria, 2, 255), **_display(evidence, option_keys)})


class NoulQuestion(QuestionSpec[NoulAnswer]):
    def __init__(self, instructions: str, *, evidence: bool = False):
        super().__init__("noul", instructions, _display(evidence, "show"))


class ScoreQuestion(QuestionSpec[ScoreAnswer]):
    def __init__(self, instructions: str, levels: Sequence[str], *, evidence: bool = False):
        if isinstance(levels, str) or not 2 <= len(levels) <= 255:
            raise ValueError("score requires 2–255 ordered levels")
        super().__init__("score", instructions, {"criteria": [_text(value, "level") for value in levels], **_display(evidence, "show")})


class MultiQuestion(QuestionSpec[MultiAnswer]):
    def __init__(self, instructions: str, criteria: Mapping[str, str], *, threshold: float | None = None,
                 min: int | None = None, max: int | None = None, objective: MultiObjective = "threshold",
                 exclusive: Sequence[str] | None = None, fallback: str | None = None, fallback_threshold: float | None = None):
        fields: dict[str, JsonValue] = {"criteria": _criteria(criteria, 1, None)}
        fields.update(_selection_options(set(criteria), objective=objective, threshold=threshold, min=min, max=max,
            exclusive=exclusive, fallback=fallback, fallback_threshold=fallback_threshold))
        super().__init__("multi", instructions, fields)


class ToolQuestion(QuestionSpec[ToolAnswer]):
    """Select tools without executing them. description_chars=0 disables server-side description truncation."""

    def __init__(self, instructions: str, tools: Sequence[Mapping[str, JsonValue]], *,
                 extra: Sequence[Literal["answer_directly", "ask_user", "none"]] | None = None, rank: int | None = None,
                 multi: bool = False, max_error: float | None = None, objective: MultiObjective = "threshold",
                 threshold: float | None = None, min: int | None = None, max: int | None = None,
                 exclusive: Sequence[str] | None = None, fallback: str | None = None, fallback_threshold: float | None = None,
                 description_chars: int | None = None):
        if isinstance(tools, (str, bytes)) or not all(isinstance(item, Mapping) for item in tools):
            raise ValueError("tools must contain definition objects")
        if not isinstance(multi, bool):
            raise ValueError("multi must be boolean")
        fields: dict[str, JsonValue] = {"tools": [deepcopy(dict(item)) for item in tools]}
        if description_chars is not None:
            if isinstance(description_chars, bool) or not isinstance(description_chars, int) or description_chars < 0:
                raise ValueError("description_chars must be a non-negative integer")
            fields["description_chars"] = description_chars
        # Omitted extras keep the server default: all non-tool choices for a single tool, none for multi.
        if extra is not None:
            if isinstance(extra, str) or any(item not in ("answer_directly", "ask_user", "none") for item in extra):
                raise ValueError("invalid extra tool option")
            fields["extra"] = list(extra)
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("rank must be a positive integer")
            fields["rank"] = rank
        if max_error is not None:
            fields["max_error"] = _number(max_error, "max_error", True)
        selection = _selection_options(None, objective=objective, threshold=threshold, min=min, max=max,
            exclusive=exclusive, fallback=fallback, fallback_threshold=fallback_threshold)
        if selection and not multi:
            raise ValueError("selection options require multi=True")
        if multi:
            fields["multi"] = True
            fields.update(selection)
        super().__init__("tool", instructions, fields)


class ActQuestion(QuestionSpec[ActAnswer]):
    def __init__(self, instructions: str, costs: Mapping[str, Mapping[str, float]], *,
                 criteria: Mapping[str, str] | None = None, evidence: bool = False,
                 defer_action: str | None = None, max_error: float | None = None):
        outcomes = set(criteria) if criteria is not None else {"true", "false"}
        fields: dict[str, JsonValue] = _display(evidence, "show")
        if criteria is not None:
            fields["criteria"] = _criteria(criteria, 2, 255)
        if not costs:
            raise ValueError("act requires a nonempty cost matrix")
        matrix: dict[str, JsonValue] = {}
        for action, row in costs.items():
            if not isinstance(row, Mapping) or set(row) != outcomes:
                raise ValueError("each cost row must cover every outcome exactly")
            matrix[_text(action, "action")] = {key: _number(value, "cost") for key, value in row.items()}
        fields["costs"] = matrix
        if defer_action is not None:
            if defer_action not in costs:
                raise ValueError("defer_action must identify a cost row")
            fields["defer_action"] = defer_action
        if max_error is not None:
            fields["max_error"] = _number(max_error, "max_error", True)
        super().__init__("act", instructions, fields)


# Factories and constructors share exactly the same typed signatures.
choice = ChoiceQuestion
noul = NoulQuestion
score = ScoreQuestion
multi = MultiQuestion
act = ActQuestion
tool = ToolQuestion
QuestionSet = Mapping[str, QuestionSpec[Any]]


class Questions:
    """Declare questions as class fields; pass the class itself to client.decide.

    The returned response.answers is an instance with inferred answer fields.
    Definitions may be reused or inherited. Constructors are not called when
    populating responses; use this class as a declarative schema.
    """

    def __new__(cls) -> Self:
        raise TypeError("Pass the Questions class to client.decide instead of constructing it")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Question answers are read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Question answers are read-only")


def schema_questions(schema: type[Questions]) -> dict[str, QuestionSpec[Any]]:
    """Snapshot effective definitions in declaration order, respecting Python MRO."""
    if not issubclass(schema, Questions):
        raise TypeError("expected a Questions subclass")
    members: dict[str, Any] = {}
    for base in reversed(schema.__mro__):
        members.update(vars(base))
    fields: dict[str, QuestionSpec[Any]] = {}
    for name, value in members.items():
        if isinstance(value, QuestionSpec):
            if name.startswith("_"):
                raise ValueError("Schema question names must not start with an underscore")
            fields[name] = value
    return fields


def build_questions(questions: QuestionSet) -> dict[str, WireQuestion]:
    from .tasks import ClassifyQuestion, VerifyQuestion, CompressQuestion, ExtractQuestion
    if not questions:
        raise ValueError("questions must not be empty")
    result: dict[str, WireQuestion] = {}
    for key, question in questions.items():
        _text(key, "question ID")
        if not isinstance(question, (ChoiceQuestion, NoulQuestion, ScoreQuestion, MultiQuestion, ActQuestion, ToolQuestion,
                                     ClassifyQuestion, VerifyQuestion, CompressQuestion, ExtractQuestion)):
            raise TypeError("questions must contain typed question specs")
        result[key] = question.to_dict()
    return result


class _DecisionResponseBase:
    def __init__(self, raw: DecisionResponse):
        from .client import BasalProtocolError
        if not isinstance(raw.get("model"), str) or not raw.get("model") or not isinstance(raw.get("usage"), dict) or not isinstance(raw.get("answers"), dict):
            raise BasalProtocolError("Invalid decision envelope")
        for name in ("input_tokens", "output_tokens"):
            value = raw["usage"].get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BasalProtocolError("Invalid usage field")
        self.model: str = raw["model"]
        self.usage: Usage = raw["usage"]
        self.raw = raw


class TypedDecisionResponse(_DecisionResponseBase):
    """Checked answers from a dictionary request, indexed by their request keys.

    Values have a union of answer types. Use a Questions class when each named
    field should have an exact inferred type, or isinstance to narrow this union.
    """

    def __init__(self, raw: DecisionResponse, questions: QuestionSet):
        super().__init__(raw)
        self.answers: Mapping[str, AnswerResult] = MappingProxyType({
            name: question.read(raw, name) for name, question in questions.items()
        })

    def answer(self, key: str) -> AnswerResult:
        """Retrieve by request key; an unknown key raises KeyError."""
        return self.answers[key]


class SchemaDecisionResponse(_DecisionResponseBase, Generic[TSchema_co]):
    """Checked response with the named answer fields inferred from its schema."""

    def __init__(self, raw: DecisionResponse, schema: type[TSchema_co], questions: QuestionSet):
        super().__init__(raw)
        parsed = {name: question.read(raw, name) for name, question in questions.items()}
        # Populate only after all answers pass validation, bypassing user
        # constructors. Instance attributes shadow non-data question descriptors.
        self._answers = object.__new__(schema)
        vars(self._answers).update(parsed)

    @property
    def answers(self) -> TSchema_co:
        """Read-only schema instance with each field's inferred answer type."""
        return self._answers
