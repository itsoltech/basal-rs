"""Wire types for POST /v1/basal. Question and criteria order is significant."""

from typing import Literal, NotRequired, TypedDict

# Python 3.11 syntax keeps these aliases usable on the minimum supported version.
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class Question(TypedDict, total=False):
    """Native question fields; validation and calibration belong to the server."""

    type: Literal["choice", "noul", "score", "multi", "act"]
    instructions: JsonValue
    criteria: dict[str, JsonValue] | list[JsonValue]
    levels: dict[str, JsonValue] | list[JsonValue]
    option_keys: Literal["show", "hide"]
    evidence: bool
    threshold: float
    min: int
    max: int
    base: Literal["noul", "choice", "score"]
    costs: dict[str, float] | dict[str, dict[str, float]]
    defer_action: str
    max_error: float


class DecisionRequest(TypedDict):
    state: JsonValue
    questions: dict[str, Question]
    model: NotRequired[str]
    facts: NotRequired[Literal["off", "auto"]]


class EvidenceSpan(TypedDict):
    text: str
    start: int
    end: int
    probability: float


class Answer(TypedDict):
    type: Literal["choice", "noul", "score", "multi", "act"]
    probabilities: dict[str, float]
    confidence: NotRequired[float]
    choice: NotRequired[str]
    noul: NotRequired[float]
    score: NotRequired[float]
    legend: NotRequired[dict[str, str]]
    selected: NotRequired[list[str]]
    threshold: NotRequired[float]
    set_confidence: NotRequired[float]
    action: NotRequired[str]
    answer: NotRequired[str]
    expected_costs: NotRequired[dict[str, float]]
    calibration: NotRequired[Literal["validated", "unvalidated"]]
    refused: NotRequired[str]
    max_error: NotRequired[str]
    evidence: NotRequired[list[EvidenceSpan]]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    questions: NotRequired[int]
    branches: NotRequired[int]
    latency_ms: NotRequired[float]


class DecisionResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(TypedDict):
    name: str
    description: NotRequired[str]
    release_date: NotRequired[str]
    mode: NotRequired[str]
    early_exit: NotRequired[list[JsonValue]]


class ModelsResponse(TypedDict):
    models: list[ModelMetadata]
