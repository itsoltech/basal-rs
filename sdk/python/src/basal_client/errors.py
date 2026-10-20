"""SDK exceptions. Every error keeps its original exception as __cause__."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .batches import Batch, FileObject


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


class BasalBatchSubmissionError(BasalError):
    """batches.create(requests=...) uploaded the input file, but creating the batch failed.

    Nothing is retried or deleted. Resume with
    batches.create(input_file_id=error.file_id, idempotency_key=error.idempotency_key, ...) and the
    same options; the cause (HTTP, transport or protocol error) is __cause__.

    Cancellation (asyncio.CancelledError) or KeyboardInterrupt after the upload is not converted: the
    original exception propagates, carrying this context in its basal_submission attribute; use find.
    """

    def __init__(self, file: FileObject, idempotency_key: str):
        super().__init__(f"Basal batch creation failed after uploading file {file['id']}")
        self.file = file
        self.file_id = file["id"]
        self.idempotency_key = idempotency_key

    @staticmethod
    def find(error: BaseException) -> BasalBatchSubmissionError | None:
        """Upload context of a failed or interrupted batches.create(requests=...), else None.

        Searches error and its __cause__/__context__ chain, so it also works on the new CancelledError
        that awaiting a cancelled task raises (its __context__ is the original cancellation).
        """
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, BasalBatchSubmissionError):
                return current
            attached = getattr(current, _SUBMISSION_ATTRIBUTE, None)
            if isinstance(attached, BasalBatchSubmissionError):
                return attached
            current = current.__cause__ if current.__cause__ is not None else current.__context__
        return None


_SUBMISSION_ATTRIBUTE = "basal_submission"


def _attach_submission(interruption: BaseException, file: FileObject, idempotency_key: str) -> None:
    """Internal: keeps the upload context on a cancellation/interruption that must propagate unchanged."""
    setattr(interruption, _SUBMISSION_ATTRIBUTE, BasalBatchSubmissionError(file, idempotency_key))
    interruption.add_note(f"Basal input file {file['id']} was uploaded before the batch was created; "
                          "BasalBatchSubmissionError.find(error) returns its id and idempotency key")


class BasalBatchTimeoutError(BasalError):
    """batches.wait reached its deadline. The batch keeps running on the server; it is never cancelled."""

    def __init__(self, batch_id: str, batch: Batch | None):
        super().__init__(f"Basal batch {batch_id} did not reach a terminal state before the timeout")
        self.batch_id = batch_id
        # Last batch observed before the deadline; None when no retrieve completed in time.
        self.batch = batch


class BasalBatchNotReadyError(BasalError):
    """batches.results was called for a batch that has not reached a terminal state."""

    def __init__(self, batch: Batch):
        status = batch.get("status")
        super().__init__(f"Basal batch {batch.get('id')} is not terminal (status {status!r})")
        self.batch_id = batch.get("id")
        self.status = status
        self.batch = batch


class BasalBatchResultError(BasalProtocolError):
    """A line of a batch output or error file is not UTF-8, not JSON or not a result envelope.

    line is 1-based. Records before it were already yielded; iteration stops here.
    """

    def __init__(self, file_id: str, line: int, reason: str):
        super().__init__(f"Basal batch result file {file_id}, line {line}: {reason}")
        self.file_id = file_id
        self.line = line
        self.reason = reason
