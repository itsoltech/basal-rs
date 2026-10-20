import {
  BasalBatchNotReadyError, BasalBatchSubmissionError, BasalBatchTimeoutError, BasalProtocolError, BasalTransportError,
} from "./errors.js";
import {
  idempotencyHeader, readBatchResultLines, resourcePath, serializeBatchRequests, TERMINAL_BATCH_STATUSES, type Batch,
  type BatchCreateOptions, type BatchCreateParams, type BatchList, type BatchResultLine, type BatchResultsOptions,
  type BatchWaitOptions, type CreateBatchRequest, type DeletedFile, type DownloadOptions, type ExpiresAfter, type FileList,
  type FileObject, type ListFilesOptions, type ListOptions, type RequestOptions, type TransferOptions, type UploadFileRequest,
} from "./batches.js";
import type { BasalClient } from "./index.js";

const TERMINAL = new Set<string>(TERMINAL_BATCH_STATUSES);
const COMMON_KEYS = new Set(["endpoint", "completion_window", "metadata", "output_expires_after"]);
const MAX_TIMER_MS = 2_147_483_647;

function duration(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  if (!Number.isFinite(value) || value <= 0 || value > MAX_TIMER_MS) throw new TypeError(`${name} must be positive and at most ${MAX_TIMER_MS}`);
  return value;
}

/** Resolves after `ms`, or rejects with `signal.reason` on abort; the timer and listener are always removed. */
function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function expiresAfter(value: ExpiresAfter | null | undefined): void {
  if (value === undefined || value === null) return;
  if (typeof value !== "object" || value.anchor !== "created_at" || !Number.isSafeInteger(value.seconds) || value.seconds < 0) {
    throw new TypeError("output_expires_after needs anchor created_at and whole seconds");
  }
}

/** Common parameters of the `requests` form, checked before the upload so a local mistake leaves no orphan file.
 * Metadata limits stay with the server, as in `createBatch`.
 */
function commonParams(params: Record<string, unknown>): Omit<CreateBatchRequest, "input_file_id"> {
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && key !== "requests" && !COMMON_KEYS.has(key)) throw new TypeError(`unsupported batch parameter ${JSON.stringify(key)}`);
  }
  const { endpoint, completion_window, metadata, output_expires_after } = params;
  if (endpoint !== undefined && endpoint !== "/v1/basal") throw new TypeError("endpoint must be /v1/basal");
  if (completion_window !== undefined && completion_window !== "24h") throw new TypeError("completion_window must be 24h");
  if (metadata !== undefined && metadata !== null) {
    if (typeof metadata !== "object" || Array.isArray(metadata) || Object.values(metadata).some(v => typeof v !== "string")) {
      throw new TypeError("metadata must be an object of string values");
    }
  }
  expiresAfter(output_expires_after as ExpiresAfter | null | undefined);
  const common: Omit<CreateBatchRequest, "input_file_id"> = {};
  for (const key of COMMON_KEYS) if (params[key] !== undefined) Object.assign(common, { [key]: params[key] });
  return common;
}

function randomKey(): string {
  const random = new Uint8Array(16);
  globalThis.crypto.getRandomValues(random);
  return `basal-sdk-${Array.from(random, b => b.toString(16).padStart(2, "0")).join("")}`;
}

/** `client.files`: resource names for the flat Files methods, with the same behavior. */
export class Files {
  readonly #client: BasalClient;
  constructor(client: BasalClient) { this.#client = client; }
  /** `uploadFile`. */
  create(request: UploadFileRequest, options: TransferOptions = {}): Promise<FileObject> { return this.#client.uploadFile(request, options); }
  /** `getFile`. */
  retrieve(fileId: string, options: RequestOptions = {}): Promise<FileObject> { return this.#client.getFile(fileId, options); }
  /** `listFiles`. */
  list(options: ListFilesOptions = {}): Promise<FileList> { return this.#client.listFiles(options); }
  /** `deleteFile`. */
  delete(fileId: string, options: RequestOptions = {}): Promise<DeletedFile> { return this.#client.deleteFile(fileId, options); }
  /** `downloadFile`: raw bytes, buffered up to `maxBytes` (default `maxResponseBytes`). */
  content(fileId: string, options: DownloadOptions = {}): Promise<Uint8Array> { return this.#client.downloadFile(fileId, options); }
}

/** `client.batches`: resource names for the flat Batch methods plus `create({requests})`, `wait` and `results`.
 * Nothing runs in the background and nothing is retried.
 */
export class Batches {
  readonly #client: BasalClient;
  constructor(client: BasalClient) { this.#client = client; }

  /** `{input_file_id, ...}` is `createBatch`. `{requests, ...}` validates and writes the JSONL in memory, uploads it,
   * then creates the batch with `idempotencyKey` (generated when omitted). A create failure after the upload raises
   * `BasalBatchSubmissionError` with the file id and key; the file is kept.
   */
  async create(params: BatchCreateParams, options: BatchCreateOptions = {}): Promise<Batch> {
    if (!params || typeof params !== "object") throw new TypeError("batch parameters must be an object");
    const fromFile = params.input_file_id !== undefined;
    if (fromFile === (params.requests !== undefined)) throw new TypeError("pass exactly one of input_file_id or requests");
    const { uploadTimeoutMs, ...createOptions } = options;
    if (fromFile) {
      if (uploadTimeoutMs !== undefined) throw new TypeError("uploadTimeoutMs applies only to requests");
      return this.#client.createBatch(params as CreateBatchRequest, createOptions);
    }
    const common = commonParams(params as unknown as Record<string, unknown>);
    const idempotencyKey = createOptions.idempotencyKey ?? randomKey();
    idempotencyHeader(idempotencyKey);
    duration(uploadTimeoutMs, 1, "uploadTimeoutMs");
    const content = serializeBatchRequests(params.requests!);
    const transfer: TransferOptions = {};
    if (createOptions.signal !== undefined) transfer.signal = createOptions.signal;
    if (uploadTimeoutMs !== undefined) transfer.timeoutMs = uploadTimeoutMs;
    const file = await this.#client.uploadFile({ file: content, filename: "batch_input.jsonl" }, transfer);
    if (!file || typeof file.id !== "string" || !file.id) throw new BasalProtocolError("Basal upload response has no file id");
    try {
      return await this.#client.createBatch({ ...common, input_file_id: file.id }, { ...createOptions, idempotencyKey });
    } catch (cause) {
      throw new BasalBatchSubmissionError(file, idempotencyKey, { cause });
    }
  }
  /** `getBatch`. */
  retrieve(batchId: string, options: RequestOptions = {}): Promise<Batch> { return this.#client.getBatch(batchId, options); }
  /** `listBatches`. */
  list(options: ListOptions = {}): Promise<BatchList> { return this.#client.listBatches(options); }
  /** `cancelBatch`: requests cancellation and returns without waiting. */
  cancel(batchId: string, options: RequestOptions = {}): Promise<Batch> { return this.#client.cancelBatch(batchId, options); }

  /** Retrieve until the status is `completed`, `failed`, `expired` or `cancelled` and return that batch (check
   * `status`, `errors` and `request_counts`). The first retrieve is immediate. `timeoutMs` bounds the whole wait,
   * aborting an in-flight retrieve, and raises `BasalBatchTimeoutError` without cancelling the batch. HTTP and
   * transport errors propagate without retries; an abort rejects with `BasalTransportError` (cause: the reason).
   */
  async wait(batchId: string, options: BatchWaitOptions = {}): Promise<Batch> {
    resourcePath("v1/batches", batchId);
    const timeoutMs = duration(options.timeoutMs, 600_000, "timeoutMs");
    const pollIntervalMs = duration(options.pollIntervalMs, 5_000, "pollIntervalMs");
    const { signal } = options;
    const controller = new AbortController();
    const forward = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", forward, { once: true });
    if (signal?.aborted) forward();
    const expired = new Error("Basal batch wait timed out");
    // Timers use a monotonic clock, so wall-clock changes do not move the deadline.
    const timer = setTimeout(() => controller.abort(expired), timeoutMs);
    let last: Batch | null = null;
    try {
      for (;;) {
        try {
          last = await this.#client.getBatch(batchId, { signal: controller.signal });
          if (TERMINAL.has(last.status)) return last;
          await sleep(pollIntervalMs, controller.signal);
        } catch (cause) {
          if (controller.signal.reason === expired && (cause === expired || cause instanceof BasalTransportError)) {
            throw new BasalBatchTimeoutError(batchId, last, timeoutMs, { cause });
          }
          if (controller.signal.aborted && cause === controller.signal.reason) {
            throw new BasalTransportError("Basal batch wait aborted", { cause });
          }
          throw cause;
        }
      }
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", forward);
    }
  }

  /** Records of a finished batch: the output file, then the error file (once if both ids are equal). Does not wait:
   * a non-terminal batch raises `BasalBatchNotReadyError`. Expired and cancelled batches yield their partial results;
   * a completed batch can yield failed records. Batch-level validation errors are in `retrieve`/`wait`, not here.
   * Each file is downloaded whole (bounded by `maxBytes`) and parsed one line at a time; a malformed line raises
   * `BasalBatchResultError` after the earlier records were yielded.
   */
  results(batchId: string, options: BatchResultsOptions = {}): AsyncGenerator<BatchResultLine, void, undefined> {
    resourcePath("v1/batches", batchId);
    duration(options.timeoutMs, 1, "timeoutMs");
    if (options.maxBytes !== undefined && (!Number.isSafeInteger(options.maxBytes) || options.maxBytes <= 0)) {
      throw new TypeError("maxBytes must be a positive safe integer");
    }
    return this.#results(batchId, options);
  }
  async *#results(batchId: string, options: BatchResultsOptions): AsyncGenerator<BatchResultLine, void, undefined> {
    const request: RequestOptions = options.signal === undefined ? {} : { signal: options.signal };
    const batch = await this.#client.getBatch(batchId, request);
    if (!TERMINAL.has(batch.status)) throw new BasalBatchNotReadyError(batchId, batch);
    const ids: string[] = [];
    for (const id of [batch.output_file_id, batch.error_file_id]) {
      if (id === null || id === undefined) continue;
      if (typeof id !== "string" || !id) throw new BasalProtocolError("Basal batch has an invalid result file id");
      if (!ids.includes(id)) ids.push(id);
    }
    for (const fileId of ids) {
      const content = await this.#client.downloadFile(fileId, options);
      yield* readBatchResultLines(content, fileId);
    }
  }
}
