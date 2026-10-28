"""Wire types for POST /v1/basal. Question and criteria order is significant."""

from typing import Literal, NotRequired, TypedDict

# Python 3.11 syntax keeps these aliases usable on the minimum supported version.
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
Calibration = Literal["validated", "unvalidated", "text"]
MultiObjective = Literal["threshold", "exact", "iou", "f1"]


class MediaInput(TypedDict):
    type: Literal["image", "audio"]
    data: str


class MediaUsage(TypedDict):
    images: int
    audio: int
    media_tokens: int
    detail: Literal["default", "high"]


class ToolRanking(TypedDict):
    tool: str
    probability: float


class Question(TypedDict, total=False):
    """Native question fields; validation and calibration belong to the server."""

    type: Literal["choice", "noul", "score", "multi", "act", "tool", "classify", "verify", "compress", "extract"]
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
    max_error: float | list[float]
    objective: MultiObjective
    exclusive: list[str]
    fallback: str
    fallback_threshold: float
    tools: list[dict[str, JsonValue]]
    extra: list[Literal["answer_directly", "ask_user", "none"]]
    rank: int
    description_chars: int
    multi: bool
    labels: dict[str, JsonValue] | list[JsonValue]
    multi_label: bool
    none: bool
    claims: dict[str, JsonValue] | list[JsonValue]
    binary: bool
    unit: Literal["sentence", "line", "paragraph"]
    keep_ratio: float
    min_keep: int
    schema: dict[str, JsonValue]


class DecisionRequest(TypedDict):
    state: JsonValue
    questions: dict[str, Question]
    model: NotRequired[str]
    facts: NotRequired[Literal["off", "auto"]]
    think: NotRequired[Literal["off", "auto", "force"]]
    think_text: NotRequired[bool]
    early_exit: NotRequired[str | None]
    lang: NotRequired[Literal["pl", "en"]]
    media: NotRequired[list[MediaInput]]
    detail: NotRequired[Literal["default", "high"]]


class EvidenceSpan(TypedDict):
    text: str
    start: int
    end: int
    probability: float


class Answer(TypedDict):
    type: Literal["choice", "noul", "score", "multi", "act", "tool", "classify", "verify", "compress", "extract"]
    probabilities: NotRequired[dict[str, float]]
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
    calibration: NotRequired[Calibration]
    refused: NotRequired[str]
    max_error: NotRequired[str]
    evidence: NotRequired[list[EvidenceSpan]]
    objective: NotRequired[MultiObjective]
    expected_score: NotRequired[float | None]
    constraints_applied: NotRequired[JsonValue]
    tool: NotRequired[str | None]
    tools: NotRequired[list[str]]
    call: NotRequired[bool]
    multi: NotRequired[bool]
    ranking: NotRequired[list[ToolRanking]]
    deferred: NotRequired[bool]
    experimental: NotRequired[bool]
    label: NotRequired[str]
    top_label: NotRequired[str]
    labels: NotRequired[list[str]]
    scores: NotRequired[dict[str, float]]
    automated: NotRequired[dict[str, bool | None]]
    claims: NotRequired[dict[str, JsonValue]]
    all_supported: NotRequired[bool]
    not_supported: NotRequired[list[str]]
    compressed: NotRequired[str]
    kept: NotRequired[list[int]]
    units: NotRequired[list[JsonValue]]
    unit: NotRequired[str]
    original_chars: NotRequired[int]
    compressed_chars: NotRequired[int]
    ratio: NotRequired[float]
    data: NotRequired[dict[str, JsonValue]]
    fields: NotRequired[dict[str, JsonValue]]
    # Raw objects retain unknown nested fields; think calibration is a separate identifier.
    think: NotRequired[dict[str, JsonValue]]
    meta: NotRequired[dict[str, JsonValue]]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    questions: NotRequired[int]
    branches: NotRequired[int]
    latency_ms: NotRequired[float]
    longest_prompt_tokens: NotRequired[int]
    second_order_branches: NotRequired[int]
    media: NotRequired[MediaUsage]
    lane: NotRequired[str]
    truncated: NotRequired[dict[str, JsonValue]]
    think_questions: NotRequired[int]


class DecisionResponse(TypedDict):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(TypedDict):
    name: str
    description: NotRequired[str]
    release_date: NotRequired[str]
    mode: NotRequired[str]
    early_exit: NotRequired[list[str]]
    early_exit_default: NotRequired[str]
    layout: NotRequired[str]
    calibration_warning: NotRequired[str]
    think: NotRequired[dict[str, JsonValue]]
    prompt_format: NotRequired[int]
    orders: NotRequired[JsonValue]
    temperature_per_prim: NotRequired[dict[str, float]]
    media: NotRequired[list[Literal["image", "audio"]]]


class ModelsResponse(TypedDict):
    models: list[ModelMetadata]
