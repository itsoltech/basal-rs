"""Connection-pooled synchronous and asynchronous clients for the Basal API."""

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence

import ipaddress
import json
import math
from types import TracebackType
from typing import Any, Literal, Self, cast, overload
from urllib.parse import urlsplit

import httpx

from .types import DecisionRequest, DecisionResponse, ModelsResponse, JsonValue, MediaInput
from .batches import (
    AsyncUploadContent, Batch, BatchList, CreateBatchRequest, DeletedFile, ExpiresAfter, FileList, FileObject,
    FilePurpose, Multipart, UploadContent, batch_path, create_body, file_path, idempotency_headers, list_params,
    response_limit,
)
# Errors live in errors.py; the "as" form keeps them exported from basal_client.client as before.
from .errors import (
    BasalError as BasalError, BasalHTTPError as BasalHTTPError, BasalProtocolError as BasalProtocolError,
    BasalResponseTooLargeError as BasalResponseTooLargeError, BasalTransportError as BasalTransportError,
)
from .media import _media
from .resources import AsyncBatches, AsyncFiles, Batches, Files
from .questions import (
    Questions, QuestionSet, TSchema, TypedDecisionResponse, SchemaDecisionResponse,
    build_questions, schema_questions,
)


# Result files are JSONL; error answers stay JSON.
_CONTENT_HEADERS = {"Accept": "application/jsonl, application/json"}


def _options(base_url: str, timeout: float) -> dict[str, Any]:
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


def _body(response: httpx.Response, raw: bytes, binary: bool) -> Any:
    # File content is returned as received; only error answers are decoded as JSON.
    if binary and response.is_success:
        return raw
    return _decode(response, raw)


def _answer_limit(response: httpx.Response, limit: int, default: int) -> int:
    # A per-call download cap applies to file bytes; error answers keep the client limit.
    return limit if response.is_success else default


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    return limit


def _check_encoding(response: httpx.Response) -> None:
    # Reject before HTTPX invokes a decompressor, including when a proxy ignores Accept-Encoding.
    if response.headers.get("Content-Encoding", "").strip().lower() not in ("", "identity"):
        raise BasalProtocolError("Basal requires an uncompressed response")


def _decision_request(state: DecisionRequest | JsonValue, questions: QuestionSet | None,
                      model: str | None, facts: Literal["off", "auto"] | None, lang: Literal["pl", "en"] | None,
                      media: Sequence[MediaInput] | None, detail: Literal["default", "high"] | None,
                      think: Literal["off", "auto", "force"] | None, think_text: bool | None,
                      early_exit: str | None) -> DecisionRequest:
    if questions is None:
        if any(value is not None for value in (model, facts, lang, media, detail, think, think_text, early_exit)):
            raise ValueError("pass request options in a raw request, or supply typed questions")
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
    if think is not None:
        if think not in ("off", "auto", "force"):
            raise ValueError("think must be off, auto or force")
        request["think"] = think
    if think_text is not None:
        if not isinstance(think_text, bool):
            raise ValueError("think_text must be a boolean")
        request["think_text"] = think_text
    if early_exit is not None:
        if not isinstance(early_exit, str):
            raise ValueError("early_exit must be a string or None")
        request["early_exit"] = early_exit
    if lang is not None:
        if lang not in ("pl", "en"):
            raise ValueError("lang must be pl or en")
        request["lang"] = lang
    if detail is not None:
        if detail not in ("default", "high"):
            raise ValueError("detail must be default or high")
        request["detail"] = detail
    if media is not None:
        if isinstance(media, (str, bytes)) or not 1 <= len(media) <= 8:
            raise ValueError("media must contain 1 to 8 inputs")
        request["media"] = [_media(item["type"], item["data"]) for item in media]
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
                 timeout: float = 120.0,
                 max_response_bytes: int = 8 * 1024 * 1024, trust_env: bool = False):
        self._max_response_bytes = _validate_limit(max_response_bytes)
        self._options = _options(base_url, timeout)
        self._trust_env = trust_env
        self._http = httpx.Client(**self._options, trust_env=trust_env)

    @property
    def files(self) -> Files:
        """Files API resource: create, retrieve, list, delete, content."""
        return Files(self)

    @property
    def batches(self) -> Batches:
        """Batch API resource: create, retrieve, list, cancel, wait, results."""
        return Batches(self)

    def _async_twin(self) -> "AsyncBasalClient":
        """A new async client with the same validated URL, authentication, timeout, limit and trust_env.

        batches.wait drives it on its own event loop and closes it there.
        """
        if self._http.is_closed:
            # Same failure as any other call on a closed client.
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return AsyncBasalClient._with(httpx.AsyncClient(**self._options, trust_env=self._trust_env),
                                      self._max_response_bytes)

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None, *,
                 params: dict[str, str] | None = None, headers: dict[str, str] | None = None,
                 content: Iterator[bytes] | None = None, limit: int | None = None, binary: bool = False) -> Any:
        limit = self._max_response_bytes if limit is None else limit
        try:
            with self._http.stream(method, path, json=body, params=params, headers=headers, content=content) as response:
                _check_encoding(response)
                limit = _answer_limit(response, limit, self._max_response_bytes)
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(chunk) > limit - len(raw):
                        raise BasalResponseTooLargeError("Basal response exceeds the configured byte limit")
                    raw.extend(chunk)
                return _body(response, bytes(raw), binary)
        except httpx.RequestError as error:
            raise BasalTransportError("Basal HTTP transport failed") from error

    @overload
    def decide(self, state: DecisionRequest) -> DecisionResponse: ...

    @overload
    def decide(self, state: JsonValue, questions: QuestionSet, *, model: str | None = None,
               facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
               think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> TypedDecisionResponse: ...

    @overload
    def decide(self, state: JsonValue, questions: type[TSchema], *, model: str | None = None,
               facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
               think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> SchemaDecisionResponse[TSchema]: ...

    def decide(self, state: DecisionRequest | JsonValue, questions: QuestionSet | type[Questions] | None = None, *,
               model: str | None = None, facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
               think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> DecisionResponse | TypedDecisionResponse | SchemaDecisionResponse[Questions]:
        """POST a raw request, or decide(state, a question dictionary or Questions class)."""
        schema, fields = _prepare_questions(questions)
        response = cast(DecisionResponse, self._request("POST", "v1/basal", _decision_request(state, fields, model, facts, lang, media, detail, think, think_text, early_exit)))
        if fields is None:
            return response
        return SchemaDecisionResponse(response, schema, fields) if schema is not None else TypedDecisionResponse(response, fields)

    def models(self) -> ModelsResponse:
        """GET /v1/models."""
        return cast(ModelsResponse, self._request("GET", "v1/models"))

    # Batch and Files API: the server must run with batches enabled.

    def upload_file(self, content: UploadContent, *, filename: str = "input.jsonl",
                    purpose: Literal["batch"] = "batch", expires_after: ExpiresAfter | None = None) -> FileObject:
        """POST /v1/files, streaming caller-owned bytes, a binary file object or an iterable of bytes."""
        form = Multipart(filename, purpose, expires_after)
        stream, size = form.stream(content)
        return cast(FileObject, self._request("POST", "v1/files", content=stream, headers=form.headers(size)))

    def list_files(self, *, after: str | None = None, limit: int | None = None,
                   purpose: FilePurpose | None = None) -> FileList:
        """GET /v1/files: one page; pass the previous page's last_id as after."""
        return cast(FileList, self._request("GET", "v1/files", params=list_params(after, limit, purpose)))

    def get_file(self, file_id: str) -> FileObject:
        return cast(FileObject, self._request("GET", file_path(file_id)))

    def delete_file(self, file_id: str) -> DeletedFile:
        return cast(DeletedFile, self._request("DELETE", file_path(file_id)))

    def download_file(self, file_id: str, *, max_bytes: int | None = None) -> bytes:
        """GET /v1/files/{id}/content as unparsed bytes, bounded by max_bytes (default max_response_bytes)."""
        limit = response_limit(max_bytes, self._max_response_bytes)
        return cast(bytes, self._request("GET", file_path(file_id, "/content"), headers=_CONTENT_HEADERS,
                                         limit=limit, binary=True))

    def create_batch(self, input_file_id: str | CreateBatchRequest, *, metadata: Mapping[str, str] | None = None,
                     output_expires_after: ExpiresAfter | None = None, idempotency_key: str | None = None) -> Batch:
        """POST /v1/batches for /v1/basal with the 24h window; returns at once, without waiting for results."""
        body = create_body(input_file_id, metadata, output_expires_after)
        return cast(Batch, self._request("POST", "v1/batches", body, headers=idempotency_headers(idempotency_key)))

    def list_batches(self, *, after: str | None = None, limit: int | None = None) -> BatchList:
        return cast(BatchList, self._request("GET", "v1/batches", params=list_params(after, limit)))

    def get_batch(self, batch_id: str) -> Batch:
        return cast(Batch, self._request("GET", batch_path(batch_id)))

    def cancel_batch(self, batch_id: str) -> Batch:
        """Start cancelling without waiting; a terminal batch is returned unchanged."""
        return cast(Batch, self._request("POST", batch_path(batch_id, "/cancel")))

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
                 timeout: float = 120.0,
                 max_response_bytes: int = 8 * 1024 * 1024, trust_env: bool = False):
        self._max_response_bytes = _validate_limit(max_response_bytes)
        self._http = httpx.AsyncClient(**_options(base_url, timeout), trust_env=trust_env)

    @classmethod
    def _with(cls, http: httpx.AsyncClient, max_response_bytes: int) -> Self:
        # Internal: wraps an already configured HTTPX client (see BasalClient._async_twin).
        client = cls.__new__(cls)
        client._max_response_bytes = max_response_bytes
        client._http = http
        return client

    @property
    def files(self) -> AsyncFiles:
        """Files API resource: create, retrieve, list, delete, content."""
        return AsyncFiles(self)

    @property
    def batches(self) -> AsyncBatches:
        """Batch API resource: create, retrieve, list, cancel, wait, results."""
        return AsyncBatches(self)

    async def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None, *,
                       params: dict[str, str] | None = None, headers: dict[str, str] | None = None,
                       content: AsyncIterator[bytes] | None = None, limit: int | None = None,
                       binary: bool = False) -> Any:
        limit = self._max_response_bytes if limit is None else limit
        try:
            async with self._http.stream(method, path, json=body, params=params, headers=headers,
                                         content=content) as response:
                _check_encoding(response)
                limit = _answer_limit(response, limit, self._max_response_bytes)
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(chunk) > limit - len(raw):
                        raise BasalResponseTooLargeError("Basal response exceeds the configured byte limit")
                    raw.extend(chunk)
                return _body(response, bytes(raw), binary)
        except httpx.RequestError as error:
            raise BasalTransportError("Basal HTTP transport failed") from error

    @overload
    async def decide(self, state: DecisionRequest) -> DecisionResponse: ...

    @overload
    async def decide(self, state: JsonValue, questions: QuestionSet, *, model: str | None = None,
                     facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
                     think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> TypedDecisionResponse: ...

    @overload
    async def decide(self, state: JsonValue, questions: type[TSchema], *, model: str | None = None,
                     facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
                     think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> SchemaDecisionResponse[TSchema]: ...

    async def decide(self, state: DecisionRequest | JsonValue, questions: QuestionSet | type[Questions] | None = None, *,
                     model: str | None = None, facts: Literal["off", "auto"] | None = None, lang: Literal["pl", "en"] | None = None,
                     think: Literal["off", "auto", "force"] | None = None, think_text: bool | None = None, early_exit: str | None = None,
               media: Sequence[MediaInput] | None = None, detail: Literal["default", "high"] | None = None) -> DecisionResponse | TypedDecisionResponse | SchemaDecisionResponse[Questions]:
        """POST a raw request, a question dictionary, or a declarative Questions class."""
        schema, fields = _prepare_questions(questions)
        response = cast(DecisionResponse, await self._request("POST", "v1/basal", _decision_request(state, fields, model, facts, lang, media, detail, think, think_text, early_exit)))
        if fields is None:
            return response
        return SchemaDecisionResponse(response, schema, fields) if schema is not None else TypedDecisionResponse(response, fields)

    async def models(self) -> ModelsResponse:
        return cast(ModelsResponse, await self._request("GET", "v1/models"))

    async def upload_file(self, content: AsyncUploadContent, *, filename: str = "input.jsonl",
                          purpose: Literal["batch"] = "batch", expires_after: ExpiresAfter | None = None) -> FileObject:
        """POST /v1/files, streaming caller-owned bytes or an async iterable of bytes."""
        form = Multipart(filename, purpose, expires_after)
        stream, size = form.astream(content)
        return cast(FileObject, await self._request("POST", "v1/files", content=stream, headers=form.headers(size)))

    async def list_files(self, *, after: str | None = None, limit: int | None = None,
                         purpose: FilePurpose | None = None) -> FileList:
        return cast(FileList, await self._request("GET", "v1/files", params=list_params(after, limit, purpose)))

    async def get_file(self, file_id: str) -> FileObject:
        return cast(FileObject, await self._request("GET", file_path(file_id)))

    async def delete_file(self, file_id: str) -> DeletedFile:
        return cast(DeletedFile, await self._request("DELETE", file_path(file_id)))

    async def download_file(self, file_id: str, *, max_bytes: int | None = None) -> bytes:
        """GET /v1/files/{id}/content as unparsed bytes, bounded by max_bytes (default max_response_bytes)."""
        limit = response_limit(max_bytes, self._max_response_bytes)
        return cast(bytes, await self._request("GET", file_path(file_id, "/content"), headers=_CONTENT_HEADERS,
                                               limit=limit, binary=True))

    async def create_batch(self, input_file_id: str | CreateBatchRequest, *,
                           metadata: Mapping[str, str] | None = None, output_expires_after: ExpiresAfter | None = None,
                           idempotency_key: str | None = None) -> Batch:
        body = create_body(input_file_id, metadata, output_expires_after)
        return cast(Batch, await self._request("POST", "v1/batches", body,
                                               headers=idempotency_headers(idempotency_key)))

    async def list_batches(self, *, after: str | None = None, limit: int | None = None) -> BatchList:
        return cast(BatchList, await self._request("GET", "v1/batches", params=list_params(after, limit)))

    async def get_batch(self, batch_id: str) -> Batch:
        return cast(Batch, await self._request("GET", batch_path(batch_id)))

    async def cancel_batch(self, batch_id: str) -> Batch:
        return cast(Batch, await self._request("POST", batch_path(batch_id, "/cancel")))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        traceback: TracebackType | None) -> None:
        await self.aclose()
