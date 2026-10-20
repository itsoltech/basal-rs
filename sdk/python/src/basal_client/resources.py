"""Resource namespaces client.files and client.batches (OpenAI-style naming over the native Batch API).

They delegate to the flat client methods, so transport, authentication, TLS, proxy prefix and response
limits are the same. The high-level helpers add: batches.create(requests=...), wait and results.
"""

from __future__ import annotations

import asyncio
import builtins
import contextvars
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Mapping
import threading
from typing import TYPE_CHECKING, Literal, NoReturn, TypeVar, cast, overload

from .batches import (
    DEFAULT_POLL_INTERVAL, DEFAULT_WAIT_TIMEOUT, AsyncUploadContent, Batch, BatchList, BatchRequest,
    BatchResultLine, DeletedFile, ExpiresAfter, FileList, FileObject, FilePurpose, JsonlBuilder, UploadContent,
    batch_options, batch_path, create_body, generated_idempotency_key, idempotency_headers, is_terminal,
    request_records, response_limit, result_file_ids, result_lines, wait_options,
)
from .errors import (
    BasalBatchNotReadyError, BasalBatchSubmissionError, BasalBatchTimeoutError, BasalProtocolError,
    _attach_submission,
)

if TYPE_CHECKING:
    from .client import AsyncBasalClient, BasalClient

# Filename of the input file uploaded by batches.create(requests=...).
REQUESTS_FILENAME = "requests.jsonl"
# Async loops yield to the event loop after this many records, so large inputs/results stay cooperative.
_YIELD_EVERY = 256
# The sync wait joins its worker in slices, so KeyboardInterrupt is noticed promptly on every platform.
_JOIN_SLICE = 0.1

_T = TypeVar("_T")


def _create_key(input_file_id: object, requests: object, endpoint: str, completion_window: str,
                metadata: Mapping[str, str] | None, output_expires_after: ExpiresAfter | None,
                idempotency_key: str | None) -> str | None:
    """Checks every create option before I/O; returns the key of the requests path (generated if omitted)."""
    if (input_file_id is None) == (requests is None):
        raise ValueError("pass exactly one of input_file_id or requests")
    if input_file_id is not None and not isinstance(input_file_id, str):
        raise ValueError("input_file_id must be a nonempty id")
    batch_options(endpoint, completion_window)
    create_body(input_file_id if isinstance(input_file_id, str) else "pending", metadata, output_expires_after)
    idempotency_headers(idempotency_key)
    if requests is None:
        return None
    return idempotency_key if idempotency_key is not None else generated_idempotency_key()


def _uploaded(file: FileObject) -> FileObject:
    if not isinstance(file.get("id"), str) or not file["id"]:
        raise BasalProtocolError("Basal upload response lacks a file id")
    return file


def _submission_failed(error: BaseException, file: FileObject, key: str) -> NoReturn:
    """Raised from inside the except block that caught a failure of the create step after the upload."""
    if isinstance(error, Exception):
        raise BasalBatchSubmissionError(file, key) from error
    # Cancellation, KeyboardInterrupt, SystemExit: keep the exact exception, attach the upload context.
    _attach_submission(error, file, key)
    raise error


def _run_owned(client: BasalClient, run: Callable[[AsyncBasalClient], Awaitable[_T]]) -> _T:
    """Runs run(async twin of client) on a private event loop in a worker thread, joined before returning.

    The twin and the loop (including its resolver threads) are closed by the worker. If the caller is
    interrupted (KeyboardInterrupt), the coroutine is cancelled, the worker is joined and the
    interruption propagates; a repeated interrupt during that cleanup is absorbed.
    """
    twin = client._async_twin()
    context = contextvars.copy_context()
    loop = asyncio.new_event_loop()
    tasks: builtins.list[asyncio.Task[_T]] = []
    failure: builtins.list[BaseException] = []

    async def main() -> _T:
        try:
            return await run(twin)
        finally:
            await twin.aclose()

    def cancel() -> None:
        # Runs on the loop, so the task already exists.
        tasks[0].cancel()

    def work() -> None:
        try:
            # Created here, so a worker that fails to start leaves no coroutine behind.
            tasks.append(loop.create_task(main(), context=context))
            loop.run_until_complete(tasks[0])
        except BaseException as error:
            if not tasks or not tasks[0].done():
                failure.append(error)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                loop.close()

    worker = threading.Thread(target=work, name="basal-batches-wait")
    try:
        worker.start()
    except BaseException:
        loop.close()
        raise
    try:
        while worker.is_alive():
            worker.join(_JOIN_SLICE)
    except BaseException:
        try:
            loop.call_soon_threadsafe(cancel)
        except RuntimeError:
            pass  # the loop has already finished
        while worker.is_alive():
            try:
                worker.join(_JOIN_SLICE)
            except BaseException:
                pass  # keep the first interruption; the worker is already cancelled
        if tasks and tasks[0].done() and not tasks[0].cancelled():
            tasks[0].exception()  # retrieved, so asyncio does not log it as unhandled
        raise
    if failure:
        raise failure[0]
    return tasks[0].result()


def _file_ids(batch: Batch) -> builtins.list[str]:
    try:
        return result_file_ids(batch)
    except ValueError as error:
        raise BasalProtocolError("Basal returned an invalid batch result file id") from error


class Files:
    """client.files of BasalClient."""

    def __init__(self, client: BasalClient):
        self._client = client

    def create(self, content: UploadContent, *, filename: str = "input.jsonl", purpose: Literal["batch"] = "batch",
               expires_after: ExpiresAfter | None = None) -> FileObject:
        """POST /v1/files; same as upload_file."""
        return self._client.upload_file(content, filename=filename, purpose=purpose, expires_after=expires_after)

    def retrieve(self, file_id: str) -> FileObject:
        return self._client.get_file(file_id)

    def list(self, *, after: str | None = None, limit: int | None = None,
             purpose: FilePurpose | None = None) -> FileList:
        """One page; pass the previous page's last_id as after."""
        return self._client.list_files(after=after, limit=limit, purpose=purpose)

    def delete(self, file_id: str) -> DeletedFile:
        return self._client.delete_file(file_id)

    def content(self, file_id: str, *, max_bytes: int | None = None) -> bytes:
        """Unparsed bytes, buffered and bounded by max_bytes (default max_response_bytes)."""
        return self._client.download_file(file_id, max_bytes=max_bytes)


class Batches:
    """client.batches of BasalClient."""

    def __init__(self, client: BasalClient):
        self._client = client

    @overload
    def create(self, *, input_file_id: str, endpoint: Literal["/v1/basal"] = "/v1/basal",
               completion_window: Literal["24h"] = "24h", metadata: Mapping[str, str] | None = None,
               output_expires_after: ExpiresAfter | None = None, idempotency_key: str | None = None) -> Batch: ...

    @overload
    def create(self, *, requests: Iterable[BatchRequest], endpoint: Literal["/v1/basal"] = "/v1/basal",
               completion_window: Literal["24h"] = "24h", metadata: Mapping[str, str] | None = None,
               output_expires_after: ExpiresAfter | None = None, idempotency_key: str | None = None) -> Batch: ...

    def create(self, *, input_file_id: str | None = None, requests: Iterable[BatchRequest] | None = None,
               endpoint: Literal["/v1/basal"] = "/v1/basal", completion_window: Literal["24h"] = "24h",
               metadata: Mapping[str, str] | None = None, output_expires_after: ExpiresAfter | None = None,
               idempotency_key: str | None = None) -> Batch:
        """Create a batch from an uploaded input_file_id, or from requests uploaded first as JSONL.

        requests are validated and serialized into one in-memory buffer (at most 50000 records and
        200000000 bytes) before anything is sent. If creation fails after the upload,
        BasalBatchSubmissionError keeps the file id and idempotency key; nothing is retried or deleted.
        A KeyboardInterrupt after the upload propagates unchanged; BasalBatchSubmissionError.find(error)
        returns the same context. An interrupted upload has no known file id (check files.list).
        """
        key = _create_key(input_file_id, requests, endpoint, completion_window, metadata, output_expires_after,
                          idempotency_key)
        if key is None:
            return self._client.create_batch(cast(str, input_file_id), metadata=metadata,
                                             output_expires_after=output_expires_after, idempotency_key=idempotency_key)
        builder = JsonlBuilder()
        for record in request_records(requests):
            builder.add(record)
        file = _uploaded(self._client.upload_file(builder.finish(), filename=REQUESTS_FILENAME))
        del builder
        try:
            return self._client.create_batch(file["id"], metadata=metadata, output_expires_after=output_expires_after,
                                             idempotency_key=key)
        except BaseException as error:
            _submission_failed(error, file, key)

    def retrieve(self, batch_id: str) -> Batch:
        return self._client.get_batch(batch_id)

    def list(self, *, after: str | None = None, limit: int | None = None) -> BatchList:
        return self._client.list_batches(after=after, limit=limit)

    def cancel(self, batch_id: str) -> Batch:
        """Start cancelling without waiting; a terminal batch is returned unchanged."""
        return self._client.cancel_batch(batch_id)

    def wait(self, batch_id: str, *, timeout: float = DEFAULT_WAIT_TIMEOUT,
             poll_interval: float = DEFAULT_POLL_INTERVAL) -> Batch:
        """Poll until completed, failed, expired or cancelled and return that batch.

        The first retrieve is immediate. One monotonic deadline covers every request (headers and
        body included) and every sleep: at the deadline BasalBatchTimeoutError carries the last
        observed batch; the server job is not cancelled. HTTP errors propagate without retries.

        To enforce that deadline the polling runs as AsyncBasalClient.batches.wait on a private event
        loop in a worker thread, with its own connections (same URL, authentication, TLS, timeout and
        trust_env) closed before returning; the caller's client and its pool are not used or closed.
        The calling thread blocks until the worker is joined. KeyboardInterrupt cancels the polling,
        joins the worker and propagates. Name resolution by the operating system cannot be
        interrupted; with a host name it can delay the return beyond the deadline.
        """
        path = batch_path(batch_id)
        timeout, poll_interval = wait_options(timeout, poll_interval)
        return _run_owned(self._client, lambda twin: AsyncBatches(twin)._wait(batch_id, path, timeout, poll_interval))

    def results(self, batch_id: str, *, max_bytes: int | None = None) -> Iterator[BatchResultLine]:
        """Records of the output file, then of the error file, of a terminal batch.

        The batch is retrieved when iteration starts, without waiting: a non-terminal batch raises
        BasalBatchNotReadyError. Each file is downloaded whole into memory, bounded by max_bytes
        (default max_response_bytes), then parsed one line at a time. Expired or cancelled batches may
        have partial results; a completed batch may include failed records.
        """
        path = batch_path(batch_id)
        response_limit(max_bytes, 1)
        return self._results(path, max_bytes)

    def _results(self, path: str, max_bytes: int | None) -> Iterator[BatchResultLine]:
        batch = cast(Batch, self._client._request("GET", path))
        if not is_terminal(batch):
            raise BasalBatchNotReadyError(batch)
        for file_id in _file_ids(batch):
            yield from result_lines(self._client.download_file(file_id, max_bytes=max_bytes), file_id)


class AsyncFiles:
    """client.files of AsyncBasalClient."""

    def __init__(self, client: AsyncBasalClient):
        self._client = client

    async def create(self, content: AsyncUploadContent, *, filename: str = "input.jsonl",
                     purpose: Literal["batch"] = "batch", expires_after: ExpiresAfter | None = None) -> FileObject:
        """POST /v1/files; same as upload_file."""
        return await self._client.upload_file(content, filename=filename, purpose=purpose, expires_after=expires_after)

    async def retrieve(self, file_id: str) -> FileObject:
        return await self._client.get_file(file_id)

    async def list(self, *, after: str | None = None, limit: int | None = None,
                   purpose: FilePurpose | None = None) -> FileList:
        return await self._client.list_files(after=after, limit=limit, purpose=purpose)

    async def delete(self, file_id: str) -> DeletedFile:
        return await self._client.delete_file(file_id)

    async def content(self, file_id: str, *, max_bytes: int | None = None) -> bytes:
        return await self._client.download_file(file_id, max_bytes=max_bytes)


class AsyncBatches:
    """client.batches of AsyncBasalClient. Cancelling the awaiting task stops any call."""

    def __init__(self, client: AsyncBasalClient):
        self._client = client

    @overload
    async def create(self, *, input_file_id: str, endpoint: Literal["/v1/basal"] = "/v1/basal",
                     completion_window: Literal["24h"] = "24h", metadata: Mapping[str, str] | None = None,
                     output_expires_after: ExpiresAfter | None = None, idempotency_key: str | None = None) -> Batch: ...

    @overload
    async def create(self, *, requests: Iterable[BatchRequest], endpoint: Literal["/v1/basal"] = "/v1/basal",
                     completion_window: Literal["24h"] = "24h", metadata: Mapping[str, str] | None = None,
                     output_expires_after: ExpiresAfter | None = None, idempotency_key: str | None = None) -> Batch: ...

    async def create(self, *, input_file_id: str | None = None, requests: Iterable[BatchRequest] | None = None,
                     endpoint: Literal["/v1/basal"] = "/v1/basal", completion_window: Literal["24h"] = "24h",
                     metadata: Mapping[str, str] | None = None, output_expires_after: ExpiresAfter | None = None,
                     idempotency_key: str | None = None) -> Batch:
        """Same contract as BasalClient.batches.create; JSONL preparation yields to the event loop.

        Cancellation after the upload propagates as asyncio.CancelledError (never converted or
        swallowed); BasalBatchSubmissionError.find(error) returns the file id and idempotency key.
        """
        key = _create_key(input_file_id, requests, endpoint, completion_window, metadata, output_expires_after,
                          idempotency_key)
        if key is None:
            return await self._client.create_batch(cast(str, input_file_id), metadata=metadata,
                                                   output_expires_after=output_expires_after,
                                                   idempotency_key=idempotency_key)
        builder = JsonlBuilder()
        for record in request_records(requests):
            builder.add(record)
            if builder.count % _YIELD_EVERY == 0:
                await asyncio.sleep(0)
        file = _uploaded(await self._client.upload_file(builder.finish(), filename=REQUESTS_FILENAME))
        del builder
        try:
            return await self._client.create_batch(file["id"], metadata=metadata,
                                                   output_expires_after=output_expires_after, idempotency_key=key)
        except BaseException as error:
            _submission_failed(error, file, key)

    async def retrieve(self, batch_id: str) -> Batch:
        return await self._client.get_batch(batch_id)

    async def list(self, *, after: str | None = None, limit: int | None = None) -> BatchList:
        return await self._client.list_batches(after=after, limit=limit)

    async def cancel(self, batch_id: str) -> Batch:
        return await self._client.cancel_batch(batch_id)

    async def wait(self, batch_id: str, *, timeout: float = DEFAULT_WAIT_TIMEOUT,
                   poll_interval: float = DEFAULT_POLL_INTERVAL) -> Batch:
        """Same contract as BasalClient.batches.wait; one asyncio deadline covers requests and sleeps.

        Cancelling the awaiting task stops the wait with CancelledError, without cancelling the job.
        """
        path = batch_path(batch_id)
        timeout, poll_interval = wait_options(timeout, poll_interval)
        return await self._wait(batch_id, path, timeout, poll_interval)

    async def _wait(self, batch_id: str, path: str, timeout: float, poll_interval: float) -> Batch:
        last: Batch | None = None
        scope = asyncio.timeout(timeout)
        try:
            async with scope:
                while True:
                    last = cast(Batch, await self._client._request("GET", path))
                    if is_terminal(last):
                        return last
                    await asyncio.sleep(poll_interval)
        except TimeoutError as error:
            if not scope.expired():
                raise
            raise BasalBatchTimeoutError(batch_id, last) from error

    def results(self, batch_id: str, *, max_bytes: int | None = None) -> AsyncIterator[BatchResultLine]:
        """Same contract as BasalClient.batches.results; use async for."""
        path = batch_path(batch_id)
        response_limit(max_bytes, 1)
        return self._results(path, max_bytes)

    async def _results(self, path: str, max_bytes: int | None) -> AsyncIterator[BatchResultLine]:
        batch = cast(Batch, await self._client._request("GET", path))
        if not is_terminal(batch):
            raise BasalBatchNotReadyError(batch)
        for file_id in _file_ids(batch):
            raw = await self._client.download_file(file_id, max_bytes=max_bytes)
            for count, record in enumerate(result_lines(raw, file_id), 1):
                yield record
                if count % _YIELD_EVERY == 0:
                    await asyncio.sleep(0)
