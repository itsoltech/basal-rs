"""Connection-pooled synchronous and asynchronous clients for the Basal API."""

import ipaddress
import json
import math
from types import TracebackType
from typing import Any, Literal, Self, cast, overload
from urllib.parse import urlsplit

import httpx

from .types import DecisionRequest, DecisionResponse, ModelsResponse, JsonValue
from .questions import (
    Questions, QuestionSet, TSchema, TypedDecisionResponse, SchemaDecisionResponse,
    build_questions, schema_questions,
)


class BasalError(Exception):
    """Base SDK error. Transport errors retain their original exception as cause."""


class BasalHTTPError(BasalError):
    """Non-2xx response, retaining the body and unmodified Retry-After header."""

    def __init__(self, status_code: int, body: Any, retry_after: str | None):
        # Keep server bodies accessible explicitly; do not include state or secrets in logs.
        super().__init__(f"Basal HTTP {status_code}")
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


class BasalTransportError(BasalError):
    """Connection, timeout or response-body transport failure."""


class BasalResponseTooLargeError(BasalError):
    """Actual streamed response exceeds the configured byte limit."""


class BasalProtocolError(BasalError):
    """Invalid JSON envelope or unsupported response encoding."""


def _options(base_url: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    url = urlsplit(base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or "?" in base_url
        or "#" in base_url
    ):
        raise ValueError("base_url must be an HTTP(S) server URL without credentials, query or fragment")
    try:
        loopback = ipaddress.ip_address(url.hostname).is_loopback
    except ValueError:
        loopback = url.hostname.lower() == "localhost"
    if url.scheme != "https" and not loopback:
        raise ValueError("base_url requires HTTPS or loopback HTTP (e.g. an SSH tunnel)")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive (seconds)")
    headers = {"Accept": "application/json", "Accept-Encoding": "identity", "User-Agent": "basal-client-python"}
    if api_key is not None:
        if not api_key or "\r" in api_key or "\n" in api_key:
            raise ValueError("api_key must be a nonempty, single-line token")
        headers["Authorization"] = f"Bearer {api_key}"
    return dict(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout,
                follow_redirects=False)


def _decode(response: httpx.Response, raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as error:
        if response.is_success:
            raise BasalProtocolError("Basal returned invalid JSON") from error
        body = raw.decode("utf-8", errors="replace")
    if not response.is_success:
        raise BasalHTTPError(response.status_code, body, response.headers.get("Retry-After"))
    if not isinstance(body, dict):
        raise BasalProtocolError("Basal returned a JSON value other than an object")
    return body


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    return limit


def _check_encoding(response: httpx.Response) -> None:
    # Reject before HTTPX invokes a decompressor, including when a proxy ignores Accept-Encoding.
    if response.headers.get("Content-Encoding", "").strip().lower() not in ("", "identity"):
        raise BasalProtocolError("Basal requires an uncompressed response")


def _decision_request(state: DecisionRequest | JsonValue, questions: QuestionSet | None,
                      model: str | None, facts: Literal["off", "auto"] | None) -> DecisionRequest:
    if questions is None:
        if model is not None or facts is not None:
            raise ValueError("pass model/facts in a raw request, or supply typed questions")
        if not isinstance(state, dict):
            raise TypeError("a raw request must be a dictionary")
        return cast(DecisionRequest, state)
    if state is None:
        raise ValueError("state must not be null")
    request: DecisionRequest = {"state": cast(JsonValue, state), "questions": build_questions(questions)}
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        request["model"] = model
    if facts is not None:
        if facts not in ("off", "auto"):
            raise ValueError("facts must be off or auto")
        request["facts"] = facts
    return request


def _prepare_questions(questions: QuestionSet | type[Questions] | None) -> tuple[type[Questions] | None, QuestionSet | None]:
    if isinstance(questions, type):
        return questions, schema_questions(questions)
    return None, None if questions is None else dict(questions)


class BasalClient:
    """Sync client. Reuse one instance; close it or use a with block.

    base_url is the server root (optionally a proxy prefix), without /v1.
    timeout applies to each HTTPX network phase, not the whole request.
    No automatic retries: the caller decides whether to repeat inference.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *,
                 api_key: str | None = None, timeout: float = 120.0,
                 max_response_bytes: int = 8 * 1024 * 1024, trust_env: bool = False):
        self._max_response_bytes = _validate_limit(max_response_bytes)
        self._http = httpx.Client(**_options(base_url, api_key, timeout), trust_env=trust_env)

    def _request(self, method: str, path: str, body: DecisionRequest | None = None) -> dict[str, Any]:
        try:
            with self._http.stream(method, path, json=body) as response:
                _check_encoding(response)
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(chunk) > self._max_response_bytes - len(raw):
                        raise BasalResponseTooLargeError("Basal response exceeds the configured byte limit")
                    raw.extend(chunk)
                return _decode(response, bytes(raw))
        except httpx.RequestError as error:
            raise BasalTransportError("Basal HTTP transport failed") from error

    @overload
    def decide(self, state: DecisionRequest) -> DecisionResponse: ...

    @overload
    def decide(self, state: JsonValue, questions: QuestionSet, *, model: str | None = None,
               facts: Literal["off", "auto"] | None = None) -> TypedDecisionResponse: ...

    @overload
    def decide(self, state: JsonValue, questions: type[TSchema], *, model: str | None = None,
               facts: Literal["off", "auto"] | None = None) -> SchemaDecisionResponse[TSchema]: ...

    def decide(self, state: DecisionRequest | JsonValue, questions: QuestionSet | type[Questions] | None = None, *,
               model: str | None = None, facts: Literal["off", "auto"] | None = None) -> DecisionResponse | TypedDecisionResponse | SchemaDecisionResponse[Questions]:
        """POST a raw request, or decide(state, a question dictionary or Questions class)."""
        schema, fields = _prepare_questions(questions)
        response = cast(DecisionResponse, self._request("POST", "v1/basal", _decision_request(state, fields, model, facts)))
        if fields is None:
            return response
        return SchemaDecisionResponse(response, schema, fields) if schema is not None else TypedDecisionResponse(response, fields)

    def models(self) -> ModelsResponse:
        """GET /v1/models."""
        return cast(ModelsResponse, self._request("GET", "v1/models"))

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()


class AsyncBasalClient:
    """Native async HTTPX client; use async with or await aclose()."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *,
                 api_key: str | None = None, timeout: float = 120.0,
                 max_response_bytes: int = 8 * 1024 * 1024, trust_env: bool = False):
        self._max_response_bytes = _validate_limit(max_response_bytes)
        self._http = httpx.AsyncClient(**_options(base_url, api_key, timeout), trust_env=trust_env)

    async def _request(self, method: str, path: str, body: DecisionRequest | None = None) -> dict[str, Any]:
        try:
            async with self._http.stream(method, path, json=body) as response:
                _check_encoding(response)
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(chunk) > self._max_response_bytes - len(raw):
                        raise BasalResponseTooLargeError("Basal response exceeds the configured byte limit")
                    raw.extend(chunk)
                return _decode(response, bytes(raw))
        except httpx.RequestError as error:
            raise BasalTransportError("Basal HTTP transport failed") from error

    @overload
    async def decide(self, state: DecisionRequest) -> DecisionResponse: ...

    @overload
    async def decide(self, state: JsonValue, questions: QuestionSet, *, model: str | None = None,
                     facts: Literal["off", "auto"] | None = None) -> TypedDecisionResponse: ...

    @overload
    async def decide(self, state: JsonValue, questions: type[TSchema], *, model: str | None = None,
                     facts: Literal["off", "auto"] | None = None) -> SchemaDecisionResponse[TSchema]: ...

    async def decide(self, state: DecisionRequest | JsonValue, questions: QuestionSet | type[Questions] | None = None, *,
                     model: str | None = None, facts: Literal["off", "auto"] | None = None) -> DecisionResponse | TypedDecisionResponse | SchemaDecisionResponse[Questions]:
        """POST a raw request, a question dictionary, or a declarative Questions class."""
        schema, fields = _prepare_questions(questions)
        response = cast(DecisionResponse, await self._request("POST", "v1/basal", _decision_request(state, fields, model, facts)))
        if fields is None:
            return response
        return SchemaDecisionResponse(response, schema, fields) if schema is not None else TypedDecisionResponse(response, fields)

    async def models(self) -> ModelsResponse:
        return cast(ModelsResponse, await self._request("GET", "v1/models"))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        traceback: TracebackType | None) -> None:
        await self.aclose()
