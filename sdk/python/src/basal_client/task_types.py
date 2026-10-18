"""Typed results of experimental Basal tasks; source offsets count Unicode code points."""
from dataclasses import dataclass
from typing import Literal, Required, TypedDict
from .types import JsonValue


class ClaimResult(TypedDict):
    verdict: Literal['supported', 'contradicted', 'not_stated', 'not_supported']
    probabilities: dict[str, float]
    confidence: float
    automated: dict[str, bool | None]


class CompressionUnit(TypedDict):
    start: int
    end: int
    probability: float


class ExtractField(TypedDict, total=False):
    value: Required[JsonValue]
    confidence: Required[float]
    source: Required[Literal['choice', 'span']]
    calibration: Required[Literal['validated', 'per decision']]
    probabilities: dict[str, float]
    present_probability: float
    text: str
    start: int
    end: int
    line_probability: float
    span_probability: float
    error: str


@dataclass(frozen=True, slots=True)
class ClassifyAnswer:
    scores: dict[str, float]
    label: str | None = None
    top_label: str | None = None
    confidence: float | None = None
    automated: dict[str, bool | None] | None = None
    labels: tuple[str, ...] | None = None
    set_confidence: float | None = None
    experimental: Literal[True] = True


@dataclass(frozen=True, slots=True)
class VerifyAnswer:
    claims: dict[str, ClaimResult]
    all_supported: bool
    not_supported: tuple[str, ...]
    experimental: Literal[True] = True


@dataclass(frozen=True, slots=True)
class CompressAnswer:
    compressed: str
    kept: tuple[int, ...]
    units: tuple[CompressionUnit, ...]
    unit: Literal['sentence', 'line', 'paragraph']
    original_chars: int
    compressed_chars: int
    ratio: float
    experimental: Literal[True] = True


@dataclass(frozen=True, slots=True)
class ExtractAnswer:
    data: dict[str, JsonValue]
    fields: dict[str, ExtractField]
    experimental: Literal[True] = True


TaskAnswer = ClassifyAnswer | VerifyAnswer | CompressAnswer | ExtractAnswer
