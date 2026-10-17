"""Experimental Basal 2.1 tasks. Offsets count Unicode code points (Python str indices)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Literal, TypedDict, TypeVar, cast

from .questions import QuestionSpec, _number
from .types import DecisionResponse, JsonValue, MultiObjective


from .task_types import ClaimResult, CompressionUnit, ExtractField, ClassifyAnswer, VerifyAnswer, CompressAnswer, ExtractAnswer

T = TypeVar("T")

def _object(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict) or any(not isinstance(k, str) for k in v):
        raise ValueError('expected object')
    return v


def _probabilities(v: Any) -> dict[str, float]:
    result = _object(v)
    for n in result.values():
        _number(n, 'probability', True)
    return result


def _automation(v: Any) -> dict[str, bool | None]:
    result = _object(v)
    if any(n is not None and not isinstance(n, bool) for n in result.values()):
        raise ValueError('invalid automated decision')
    return result


def _strings(v: Any) -> tuple[str, ...]:
    if not isinstance(v, list) or any(not isinstance(k, str) for k in v):
        raise ValueError('expected string list')
    return tuple(v)


def _offset(v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError('invalid code point offset/count')
    return v


def _span(v: dict[str, Any]) -> None:
    if _offset(v['end']) < _offset(v['start']):
        raise ValueError('reversed source span')


def _risks(v: float | Sequence[float] | None) -> dict[str, JsonValue]:
    if v is None:
        return {}
    values = list(v) if isinstance(v, Sequence) else [v]
    if len(values) > 64 or any(not 0 < _number(n, 'max_error') < 1 for n in values):
        raise ValueError('invalid max_error')
    return {'max_error': cast(JsonValue, values)}


class _TaskQuestion(QuestionSpec[T]):
    def __init__(self, kind: str, instructions: str, fields: dict[str, JsonValue]):
        if not isinstance(instructions, str):
            raise ValueError('instructions must be a string')
        self._payload = deepcopy({'type': kind, **({'instructions': instructions} if instructions else {}), **fields})

    def read(self, response: DecisionResponse, question_id: str) -> T:
        return cast(T, self._read(response, question_id))

    def _read(self, response: DecisionResponse, question_id: str) -> Any:
        from .client import BasalProtocolError
        try:
            d = _object(response['answers'][question_id])
            if d.get('type') != self._payload['type'] or d.get('experimental') is not True:
                raise ValueError('mismatched task')
            kind = d['type']
            if kind == 'classify':
                scores = _probabilities(d['scores'])
                if self._payload.get('multi_label', False):
                    labels = _strings(d['labels'])
                    if any(k not in scores for k in labels):
                        raise ValueError('unknown label')
                    return ClassifyAnswer(scores, labels=labels, set_confidence=_number(d['set_confidence'], 'set_confidence', True))
                if not isinstance(d['label'], str) or d['label'] not in scores or d['top_label'] != d['label']:
                    raise ValueError('unknown label')
                return ClassifyAnswer(scores, d['label'], d['top_label'], _number(d['confidence'], 'confidence', True), _automation(d['automated']))
            if kind == 'verify':
                claims = _object(d['claims'])
                for c in claims.values():
                    c = _object(c)
                    probabilities = _probabilities(c['probabilities'])
                    if c['verdict'] not in ('supported', 'contradicted', 'not_stated', 'not_supported') or c['verdict'] not in probabilities:
                        raise ValueError('invalid verdict')
                    _number(c['confidence'], 'confidence', True)
                    _automation(c['automated'])
                flagged = _strings(d['not_supported'])
                if not isinstance(d['all_supported'], bool) or any(k not in claims for k in flagged):
                    raise ValueError('invalid verification summary')
                return VerifyAnswer(cast(dict[str, ClaimResult], claims), d['all_supported'], flagged)
            if kind == 'compress':
                if not isinstance(d['compressed'], str) or d['unit'] not in ('sentence', 'line', 'paragraph') or not isinstance(d['units'], list):
                    raise ValueError('invalid compression')
                for unit in d['units']:
                    _span(_object(unit)); _number(unit['probability'], 'probability', True)
                if not isinstance(d['kept'], list) or any(_offset(i) >= len(d['units']) for i in d['kept']):
                    raise ValueError('invalid kept indices')
                return CompressAnswer(d['compressed'], tuple(d['kept']), tuple(d['units']), d['unit'], _offset(d['original_chars']), _offset(d['compressed_chars']), _number(d['ratio'], 'ratio'))
            data, fields = _object(d['data']), _object(d['fields'])
            for field in fields.values():
                field = _object(field)
                if 'value' not in field or field['source'] not in ('choice', 'span') or field['calibration'] not in ('validated', 'per decision'):
                    raise ValueError('invalid extraction field')
                _number(field['confidence'], 'confidence', True)
                if 'probabilities' in field: _probabilities(field['probabilities'])
                for key in ('present_probability', 'line_probability', 'span_probability'):
                    if key in field: _number(field[key], key, True)
                if 'text' in field:
                    if not isinstance(field['text'], str): raise ValueError('invalid source text')
                    _span(field)
            return ExtractAnswer(data, cast(dict[str, ExtractField], fields))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise BasalProtocolError('Basal returned an invalid task answer') from error


class ClassifyQuestion(_TaskQuestion[ClassifyAnswer]):
    def __init__(self, instructions: str, labels: Mapping[str, JsonValue] | Sequence[JsonValue], *, multi_label: bool = False, none: bool | None = None, max_error: float | Sequence[float] | None = None, option_keys: Literal["show", "hide"] = "show", threshold: float | None = None, min: int | None = None, max: int | None = None, objective: MultiObjective = "threshold"):
        if not isinstance(labels, (Mapping, list, tuple)) or not 1 <= len(labels) <= 256 or not isinstance(multi_label, bool):
            raise ValueError('classify requires 1..256 labels')
        fields: dict[str, JsonValue] = {'labels': dict(labels) if isinstance(labels, Mapping) else list(labels), 'multi_label': multi_label, **_risks(max_error)}
        if none is not None:
            if not isinstance(none, bool): raise ValueError('none must be a boolean')
            fields['none'] = none
        if option_keys not in ('show', 'hide'): raise ValueError('invalid option_keys')
        fields['option_keys'] = option_keys
        from .questions import _selection_options
        selection = _selection_options(None, objective=objective, threshold=threshold, min=min, max=max)
        if selection and not multi_label: raise ValueError('selection options require multi_label')
        fields.update(selection)
        super().__init__('classify', instructions, fields)


class VerifyQuestion(_TaskQuestion[VerifyAnswer]):
    def __init__(self, instructions: str, claims: Mapping[str, JsonValue] | Sequence[JsonValue], *, binary: bool = False, max_error: float | Sequence[float] | None = None):
        if not isinstance(claims, (Mapping, list, tuple)) or not claims or not isinstance(binary, bool):
            raise ValueError('verify requires claims and a boolean binary flag')
        super().__init__('verify', instructions, {'claims': dict(claims) if isinstance(claims, Mapping) else list(claims), 'binary': binary, **_risks(max_error)})


class CompressQuestion(_TaskQuestion[CompressAnswer]):
    def __init__(self, instructions: str = '', *, unit: Literal['sentence', 'line', 'paragraph'] = 'sentence', keep_ratio: float | None = None, threshold: float | None = None, min_keep: int = 1):
        if unit not in ('sentence', 'line', 'paragraph'): raise ValueError('invalid compression unit')
        fields: dict[str, JsonValue] = {'unit': unit, 'min_keep': _offset(min_keep)}
        for name, value in (('keep_ratio', keep_ratio), ('threshold', threshold)):
            if value is not None:
                fields[name] = _number(value, name, True)
                if name == 'keep_ratio' and value == 0: raise ValueError('keep_ratio must be positive')
        super().__init__('compress', instructions, fields)


class ExtractQuestion(_TaskQuestion[ExtractAnswer]):
    def __init__(self, instructions: str, schema: Mapping[str, JsonValue]):
        super().__init__('extract', instructions, {'schema': _object(dict(schema))})


def classify(instructions: str, labels: Mapping[str, JsonValue] | Sequence[JsonValue], *, multi_label: bool = False,
        none: bool | None = None, max_error: float | Sequence[float] | None = None, option_keys: Literal['show', 'hide'] = 'show',
        threshold: float | None = None, min: int | None = None, max: int | None = None, objective: MultiObjective = 'threshold') -> ClassifyQuestion:
    return ClassifyQuestion(instructions, labels, multi_label=multi_label, none=none, max_error=max_error,
        option_keys=option_keys, threshold=threshold, min=min, max=max, objective=objective)


def verify(instructions: str, claims: Mapping[str, JsonValue] | Sequence[JsonValue], *, binary: bool = False,
        max_error: float | Sequence[float] | None = None) -> VerifyQuestion:
    return VerifyQuestion(instructions, claims, binary=binary, max_error=max_error)


def compress(instructions: str = '', *, unit: Literal['sentence', 'line', 'paragraph'] = 'sentence',
        keep_ratio: float | None = None, threshold: float | None = None, min_keep: int = 1) -> CompressQuestion:
    return CompressQuestion(instructions, unit=unit, keep_ratio=keep_ratio, threshold=threshold, min_keep=min_keep)


def extract(instructions: str, schema: Mapping[str, JsonValue]) -> ExtractQuestion:
    return ExtractQuestion(instructions, schema)
