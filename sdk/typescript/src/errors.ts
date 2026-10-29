import type { Batch, FileObject } from "./batches.js";

export class BasalError extends Error {}
const authHints: Partial<Record<number, string>> = {
  401: "missing or invalid API key; pass apiKey when the server has auth.enabled",
  403: "the API key has no scope for this route",
};
/** Non-2xx response. Server body and Retry-After are available without logging them. */
export class BasalHTTPError extends BasalError {
  constructor(readonly statusCode: number, readonly body: unknown, readonly retryAfter: string | null) {
    const hint = authHints[statusCode];
    super(hint === undefined ? `Basal HTTP ${statusCode}` : `Basal HTTP ${statusCode}: ${hint}`);
    this.name = "BasalHTTPError";
    // Node's default inspection and JSON.stringify must not print reflected state or credentials.
    Object.defineProperty(this, "body", { enumerable: false });
  }
}
export class BasalTransportError extends BasalError {}
export class BasalProtocolError extends BasalError {}
export class BasalResponseTooLargeError extends BasalError {}

/** `batches.create({requests})` uploaded the input file, but creating the batch failed (HTTP, transport or abort).
 * The SDK neither retries nor deletes the file. The create may still have succeeded on the server, so resume with
 * the same parameters, `{input_file_id: error.fileId}` and `{idempotencyKey: error.idempotencyKey}`; the original
 * failure is in `cause`.
 */
export class BasalBatchSubmissionError extends BasalError {
  readonly fileId: string;
  constructor(readonly file: FileObject, readonly idempotencyKey: string, options: { cause: unknown }) {
    super(`Basal batch creation failed after uploading input file ${file.id}`, options);
    this.name = "BasalBatchSubmissionError";
    this.fileId = file.id;
  }
}
/** `batches.wait` reached its deadline. The batch keeps running on the server; nothing is cancelled.
 * `lastBatch` is the last retrieved state (null if no retrieve finished) and is non-enumerable, like `BasalHTTPError.body`.
 */
export class BasalBatchTimeoutError extends BasalError {
  constructor(readonly batchId: string, readonly lastBatch: Batch | null, readonly timeoutMs: number, options?: { cause: unknown }) {
    super(`Basal batch did not reach a terminal status within ${timeoutMs} ms`, options);
    this.name = "BasalBatchTimeoutError";
    Object.defineProperty(this, "lastBatch", { enumerable: false });
  }
}
/** `batches.results` was called for a batch that is not `completed`, `failed`, `expired` or `cancelled`.
 * `batch` (non-enumerable) is the retrieved state; `status` is its raw status.
 */
export class BasalBatchNotReadyError extends BasalError {
  readonly status: string;
  constructor(readonly batchId: string, readonly batch: Batch) {
    super("Basal batch has not reached a terminal status");
    this.name = "BasalBatchNotReadyError";
    this.status = String(batch.status);
    Object.defineProperty(this, "batch", { enumerable: false });
  }
}
/** A result file line that is not UTF-8, not JSON or not a result envelope. `line` is 1-based; the content is not
 * copied into the error.
 */
export class BasalBatchResultError extends BasalProtocolError {
  constructor(readonly fileId: string, readonly line: number, reason: string, options?: { cause: unknown }) {
    super(`Batch result file ${fileId} line ${line} ${reason}`, options);
    this.name = "BasalBatchResultError";
  }
}
