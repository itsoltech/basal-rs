"""Wire types and request builders of the Batch and Files API (contracts/batches/openapi.json).

Responses are plain dictionaries: unknown fields sent by newer servers are preserved. Times are Unix
seconds. Resource ids are opaque strings; the SDK escapes them as one path segment and leaves format
checks to the server, which answers malformed ids like missing resources (404).
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Iterator, AsyncIterator, Mapping
import secrets
import unicodedata
from typing import Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict, cast
from urllib.parse import quote

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
