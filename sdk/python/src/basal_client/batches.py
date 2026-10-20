"""Wire types and request builders of the Batch and Files API (contracts/batches/openapi.json).

Responses are plain dictionaries: unknown fields sent by newer servers are preserved. Times are Unix
seconds. Resource ids are opaque strings; the SDK escapes them as one path segment and leaves format
checks to the server, which answers malformed ids like missing resources (404).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Iterator, AsyncIterator, Mapping
import json
import math
import secrets
import unicodedata
from typing import Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict, cast
from urllib.parse import quote

from .errors import BasalBatchResultError
from .types import JsonValue

FilePurpose = Literal["batch", "batch_output"]
BatchStatus = Literal["validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"]


class ExpiresAfter(TypedDict):
    """Lifetime counted from the creation of the file; the server accepts 3600..2592000 seconds."""

    anchor: Literal["created_at"]
    seconds: int


class FileObject(TypedDict):
    id: str
    object: Literal["file"]
    bytes: int
    created_at: int
    expires_at: int | None
    filename: str
    purpose: FilePurpose
    status: Literal["processed"]
    status_details: None


class DeletedFile(TypedDict):
    id: str
    object: Literal["file"]
    deleted: Literal[True]


class FileList(TypedDict):
    object: Literal["list"]
    data: list[FileObject]
    first_id: str | None
    last_id: str | None
    has_more: bool


class RequestCounts(TypedDict):
    total: int
    completed: int
    failed: int


class BatchErrorEntry(TypedDict):
    code: str
    message: str
    param: str | None
    line: int | None


class BatchErrors(TypedDict):
    object: Literal["list"]
    data: list[BatchErrorEntry]


class Batch(TypedDict):
    id: str
    object: Literal["batch"]
    endpoint: str
    model: str | None
    errors: BatchErrors | None
    input_file_id: str
    completion_window: str
    status: BatchStatus
    output_file_id: str | None
    error_file_id: str | None
    created_at: int
    in_progress_at: int | None
    expires_at: int
    finalizing_at: int | None
    completed_at: int | None
    failed_at: int | None
    expired_at: int | None
    cancelling_at: int | None
    cancelled_at: int | None
    request_counts: RequestCounts
    metadata: dict[str, str] | None


class BatchList(TypedDict):
    object: Literal["list"]
    data: list[Batch]
    first_id: str | None
    last_id: str | None
    has_more: bool


class CreateBatchRequest(TypedDict):
    """Raw body of POST /v1/batches. Only /v1/basal and 24h are accepted by the server."""

    input_file_id: str
    endpoint: Literal["/v1/basal"]
    completion_window: Literal["24h"]
    metadata: NotRequired[dict[str, str] | None]
    output_expires_after: NotRequired[ExpiresAfter | None]


class BatchInputLine(TypedDict):
    """One JSONL line of an input file; body is a native /v1/basal request."""

    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/basal"]
    body: dict[str, JsonValue]


class BatchResponse(TypedDict):
    status_code: int
    request_id: str
    body: dict[str, JsonValue]


class BatchRecordError(TypedDict):
    code: str
    message: str


class BatchResultLine(TypedDict):
    """One JSONL line of an output or error file (both have purpose batch_output)."""

    id: str
    custom_id: str
    response: BatchResponse | None
    error: BatchRecordError | None


class SupportsRead(Protocol):
    def read(self, size: int = ..., /) -> bytes: ...


BytesLike: TypeAlias = bytes | bytearray | memoryview
UploadContent: TypeAlias = BytesLike | SupportsRead | Iterable[bytes]
AsyncUploadContent: TypeAlias = BytesLike | AsyncIterable[bytes]

_CHUNK = 64 * 1024
_MAX_FILENAME_BYTES = 255


def _segment(value: str, name: str) -> str:
    """One escaped path segment; dot segments would be normalized away by URL handling."""
    if not isinstance(value, str) or not value or value in (".", ".."):
        raise ValueError(f"{name} must be a nonempty id")
    return quote(value, safe="")


def file_path(file_id: str, suffix: str = "") -> str:
    return f"v1/files/{_segment(file_id, 'file_id')}{suffix}"


def batch_path(batch_id: str, suffix: str = "") -> str:
    return f"v1/batches/{_segment(batch_id, 'batch_id')}{suffix}"


def list_params(after: str | None, limit: int | None, purpose: str | None = None) -> dict[str, str]:
    """Query values; HTTPX percent-encodes them. Ranges are checked by the server."""
    params: dict[str, str] = {}
    if after is not None:
        if not isinstance(after, str) or not after:
            raise ValueError("after must be a nonempty id")
        params["after"] = after
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        params["limit"] = str(limit)
    if purpose is not None:
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("purpose must be a nonempty string")
        params["purpose"] = purpose
    return params


def response_limit(max_bytes: int | None, default: int) -> int:
    if max_bytes is None:
        return default
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return max_bytes


def _expires(value: ExpiresAfter, name: str) -> ExpiresAfter:
    if not isinstance(value, Mapping) or set(value) != {"anchor", "seconds"}:
        raise ValueError(f"{name} must have exactly anchor and seconds")
    seconds = value["seconds"]
    if value["anchor"] != "created_at" or isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise ValueError(f"{name} needs anchor created_at and positive whole seconds")
    return {"anchor": "created_at", "seconds": seconds}


def create_body(input_file_id: str | CreateBatchRequest, metadata: Mapping[str, str] | None,
                output_expires_after: ExpiresAfter | None) -> dict[str, Any]:
    if isinstance(input_file_id, Mapping):
        if metadata is not None or output_expires_after is not None:
            raise ValueError("pass options inside a raw request, or supply an input file id")
        return dict(input_file_id)
    if not isinstance(input_file_id, str) or not input_file_id:
        raise ValueError("input_file_id must be a nonempty id")
    body: dict[str, Any] = {"input_file_id": input_file_id, "endpoint": "/v1/basal", "completion_window": "24h"}
    if metadata is not None:
        if not isinstance(metadata, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError("metadata must map strings to strings")
        body["metadata"] = dict(metadata)
    if output_expires_after is not None:
        body["output_expires_after"] = _expires(output_expires_after, "output_expires_after")
    return body


def idempotency_headers(key: str | None) -> dict[str, str] | None:
    """One header of 1-128 visible ASCII characters; anything else could split or alter the request."""
    if key is None:
        return None
    if not isinstance(key, str) or not 1 <= len(key) <= 128 or any(not "!" <= c <= "~" for c in key):
        raise ValueError("idempotency_key must have 1-128 visible ASCII characters")
    return {"Idempotency-Key": key}


def _quoted(text: str, field: str) -> str:
    # Form field values and names are emitted inside a quoted header parameter.
    if any(c in '"\\' or unicodedata.category(c) == "Cc" for c in text):
        raise ValueError(f"{field} must not contain quotes, backslashes or control characters")
    return text


class Multipart:
    """multipart/form-data framing for POST /v1/files.

    Fields precede the file part: the server reserves the upload session before the first file byte
    and requires expires_after[...] before it. The file is streamed between the head and tail; the
    boundary is 128 random bits.
    """

    def __init__(self, filename: str, purpose: str, expires_after: ExpiresAfter | None):
        if not isinstance(filename, str) or not 1 <= len(filename.encode("utf-8")) <= _MAX_FILENAME_BYTES:
            raise ValueError("filename must have 1-255 UTF-8 bytes")
        _quoted(filename, "filename")
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("purpose must be a nonempty string")
        _quoted(purpose, "purpose")
        boundary = "basal-" + secrets.token_hex(16)
        fields: list[tuple[str, str]] = [("purpose", purpose)]
        if expires_after is not None:
            expires = _expires(expires_after, "expires_after")
            fields += [("expires_after[anchor]", expires["anchor"]), ("expires_after[seconds]", str(expires["seconds"]))]
        head = "".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
                       for name, value in fields)
        head += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 "Content-Type: application/jsonl\r\n\r\n")
        self.head = head.encode("utf-8")
        self.tail = f"\r\n--{boundary}--\r\n".encode("ascii")
        self.content_type = f"multipart/form-data; boundary={boundary}"

    def headers(self, size: int | None) -> dict[str, str]:
        headers = {"Content-Type": self.content_type}
        if size is not None:
            # A known length avoids chunked encoding; HTTPX then omits Transfer-Encoding.
            headers["Content-Length"] = str(len(self.head) + size + len(self.tail))
        return headers

    def stream(self, content: UploadContent) -> tuple[Iterator[bytes], int | None]:
        if isinstance(content, (bytes, bytearray, memoryview)):
            view = memoryview(content).cast("B")
            return self._frame(_slices(view)), view.nbytes
        if isinstance(content, (str, Mapping)):
            raise TypeError("upload content must be bytes, a binary file object or an iterable of bytes")
        if callable(getattr(content, "read", None)):
            return self._frame(_reads(cast(SupportsRead, content))), None
        if isinstance(content, Iterable):
            return self._frame(_checked(content)), None
        raise TypeError("upload content must be bytes, a binary file object or an iterable of bytes")

    def astream(self, content: AsyncUploadContent) -> tuple[AsyncIterator[bytes], int | None]:
        if isinstance(content, (bytes, bytearray, memoryview)):
            view = memoryview(content).cast("B")
            return self._aframe_sync(_slices(view)), view.nbytes
        if isinstance(content, AsyncIterable):
            return self._aframe(content), None
        raise TypeError("async upload content must be bytes or an async iterable of bytes")

    def _frame(self, body: Iterator[bytes]) -> Iterator[bytes]:
        yield self.head
        yield from body
        yield self.tail

    async def _aframe_sync(self, body: Iterator[bytes]) -> AsyncIterator[bytes]:
        for chunk in self._frame(body):
            yield chunk

    async def _aframe(self, body: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        yield self.head
        async for chunk in body:
            yield _chunk(chunk)
        yield self.tail


def _chunk(chunk: object) -> bytes:
    if not isinstance(chunk, (bytes, bytearray, memoryview)):
        raise TypeError("upload content must yield bytes")
    return bytes(chunk)


def _slices(view: memoryview) -> Iterator[bytes]:
    # Copies one bounded chunk at a time, never the whole file.
    for start in range(0, view.nbytes, _CHUNK):
        yield bytes(view[start:start + _CHUNK])


def _reads(source: SupportsRead) -> Iterator[bytes]:
    while chunk := _chunk(source.read(_CHUNK)):
        yield chunk


def _checked(source: Iterable[bytes]) -> Iterator[bytes]:
    for chunk in source:
        yield _chunk(chunk)


# High-level helpers of batches.create(requests=...), wait and results.

class BatchRequest(TypedDict):
    """One request of batches.create(requests=...): becomes one input JSONL line.

    body is a native /v1/basal request; method and url may be omitted (POST, /v1/basal).
    """

    custom_id: str
    body: Mapping[str, Any]
    method: NotRequired[Literal["POST"]]
    url: NotRequired[Literal["/v1/basal"]]


TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "expired", "cancelled"})
MAX_BATCH_RECORDS = 50_000
MAX_BATCH_INPUT_BYTES = 200_000_000
DEFAULT_WAIT_TIMEOUT = 600.0
DEFAULT_POLL_INTERVAL = 5.0
_MAX_CUSTOM_ID_BYTES = 256
_REQUEST_KEYS = frozenset({"custom_id", "body", "method", "url"})


def batch_options(endpoint: str, completion_window: str) -> None:
    # The server accepts only these values; checking them first avoids a pointless upload.
    if endpoint != "/v1/basal":
        raise ValueError("endpoint must be /v1/basal")
    if completion_window != "24h":
        raise ValueError("completion_window must be 24h")


def generated_idempotency_key() -> str:
    """128 random bits; one key per batches.create(requests=...) call that did not supply its own."""
    return "basal-sdk-" + secrets.token_hex(16)


class JsonlBuilder:
    """Validates BatchRequest records and appends them as JSONL into one in-memory buffer.

    The buffer is bounded by MAX_BATCH_INPUT_BYTES and MAX_BATCH_RECORDS: preparation stops at the
    first record that would exceed a bound, before anything is uploaded.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.count = 0
        self._ids: set[str] = set()

    def add(self, request: object) -> None:
        index = self.count
        if not isinstance(request, Mapping):
            raise TypeError(f"requests[{index}] must be a mapping with custom_id and body")
        unknown = set(request) - _REQUEST_KEYS
        if unknown:
            raise ValueError(f"requests[{index}] has unsupported fields: {', '.join(sorted(map(str, unknown)))}")
        custom_id = request.get("custom_id")
        if not isinstance(custom_id, str):
            raise ValueError(f"requests[{index}].custom_id must be a string")
        try:
            id_bytes = len(custom_id.encode("utf-8"))
        except UnicodeEncodeError as error:
            raise ValueError(f"requests[{index}].custom_id must be valid Unicode text") from error
        if not 1 <= id_bytes <= _MAX_CUSTOM_ID_BYTES:
            raise ValueError(f"requests[{index}].custom_id must have 1-256 UTF-8 bytes")
        if custom_id in self._ids:
            raise ValueError(f"requests[{index}].custom_id duplicates an earlier request")
        if request.get("method", "POST") != "POST":
            raise ValueError(f"requests[{index}].method must be POST")
        if request.get("url", "/v1/basal") != "/v1/basal":
            raise ValueError(f"requests[{index}].url must be /v1/basal")
        body = request.get("body")
        if not isinstance(body, Mapping):
            raise ValueError(f"requests[{index}].body must be a JSON object (a native /v1/basal request)")
        if body.get("stream") is True:
            # Batch records are answered as one JSON body; the server rejects streaming requests.
            raise ValueError(f"requests[{index}].body.stream must not be true (batch records do not stream)")
        if self.count >= MAX_BATCH_RECORDS:
            raise ValueError(f"requests must contain at most {MAX_BATCH_RECORDS} records")
        line = {"custom_id": custom_id, "method": "POST", "url": "/v1/basal", "body": dict(body)}
        try:
            # Strict JSON: NaN/Infinity and lone surrogates would be rejected by the server.
            text = json.dumps(line, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError(f"requests[{index}].body is not serializable as strict JSON") from error
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"requests[{index}].body contains a lone surrogate, which is not valid Unicode text") from error
        if len(encoded) + 1 > MAX_BATCH_INPUT_BYTES - len(self.buffer):
            raise ValueError(f"requests exceed {MAX_BATCH_INPUT_BYTES} JSONL bytes")
        self.buffer += encoded
        self.buffer += b"\n"
        self._ids.add(custom_id)
        self.count += 1

    def finish(self) -> bytearray:
        if not self.count:
            raise ValueError("requests must not be empty")
        return self.buffer


def request_records(requests: object) -> Iterator[object]:
    if isinstance(requests, (str, bytes, bytearray, memoryview, Mapping)) or not isinstance(requests, Iterable):
        raise TypeError("requests must be an iterable of {custom_id, body} mappings")
    return iter(requests)


def wait_options(timeout: float, poll_interval: float) -> tuple[float, float]:
    for name, value in (("timeout", timeout), ("poll_interval", poll_interval)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive (seconds)")
    return float(timeout), float(poll_interval)


def is_terminal(batch: Mapping[str, Any]) -> bool:
    # Unknown statuses are not terminal: waiting for them stays bounded by the timeout.
    return batch.get("status") in TERMINAL_STATUSES


def result_file_ids(batch: Mapping[str, Any]) -> list[str]:
    """output_file_id, then error_file_id; one id once if both are equal."""
    ids: list[str] = []
    for field in ("output_file_id", "error_file_id"):
        value = batch.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"batch {field} must be a nonempty id or null")
        if value not in ids:
            ids.append(value)
    return ids


def _reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


def result_lines(raw: bytes, file_id: str) -> Iterator[BatchResultLine]:
    """Parsed records of one downloaded JSONL output or error file, in file order.

    Lines are sliced one at a time from the buffered download; no second list of lines is built.
    A trailing LF ends the last line; CRLF is accepted. The first empty (also in the middle of the
    file), non-UTF-8, non-JSON or malformed line raises BasalBatchResultError with its 1-based number
    and ends the iteration.
    """
    view = memoryview(raw)
    start, number, size = 0, 0, len(raw)
    while start < size:
        number += 1
        end = raw.find(b"\n", start)
        stop = size if end < 0 else end
        next_start = size if end < 0 else end + 1
        if stop > start and raw[stop - 1] == 0x0D:
            stop -= 1
        if stop == start:
            raise BasalBatchResultError(file_id, number, "empty line")
        try:
            text = str(view[start:stop], "utf-8")
        except UnicodeDecodeError as error:
            raise BasalBatchResultError(file_id, number, "invalid UTF-8") from error
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as error:
            raise BasalBatchResultError(file_id, number, "invalid JSON") from error
        reason = _envelope_error(value)
        if reason is not None:
            raise BasalBatchResultError(file_id, number, reason)
        yield cast(BatchResultLine, value)
        start = next_start


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _envelope_error(value: object) -> str | None:
    """Checks the BatchResultLine envelope; response.body stays native and unknown fields are kept.

    Same rules as the Rust SDK: nonempty id and custom_id; both response and error present, exactly
    one of them non-null; response has status_code 100..599, nonempty request_id and an object body;
    error has a nonempty code and a string message.
    """
    if not isinstance(value, dict):
        return "result line is not a JSON object"
    for field in ("id", "custom_id", "response", "error"):
        if field not in value:
            return f"result line lacks {field}"
    if not _nonempty(value["id"]) or not _nonempty(value["custom_id"]):
        return "result id and custom_id must be nonempty strings"
    response, error = value["response"], value["error"]
    if (response is None) == (error is None):
        return "result line must have exactly one non-null response or error"
    if response is not None:
        if not isinstance(response, dict):
            return "result response must be an object or null"
        status = response.get("status_code")
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            return "result response.status_code must be an HTTP status 100..599"
        if not _nonempty(response.get("request_id")) or not isinstance(response.get("body"), dict):
            return "result response needs a nonempty request_id and an object body"
    if error is not None:
        if not isinstance(error, dict):
            return "result error must be an object or null"
        if not _nonempty(error.get("code")) or not isinstance(error.get("message"), str):
            return "result error needs a nonempty code and a string message"
    return None
