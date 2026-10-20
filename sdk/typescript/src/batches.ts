import { BasalProtocolError } from "./errors.js";
import type { DecisionRequest, JsonValue } from "./index.js";

/** Batch and Files wire types (`contracts/batches/openapi.json`). Times are Unix seconds.
 * Unknown response fields are preserved at runtime; `(string & {})` keeps enums open to future server values.
 */
export type FilePurpose = "batch" | "batch_output";
export interface FileObject {
  id: string;
  object: "file";
  bytes: number;
  created_at: number;
  expires_at: number | null;
  /** As uploaded, or `batch_<id>_output.jsonl` / `batch_<id>_error.jsonl` for results; never a server path. */
  filename: string;
  purpose: FilePurpose | (string & {});
  /** Always `processed` (OpenAI compatibility). */
  status: "processed" | (string & {});
  status_details: string | null;
}
export interface DeletedFile { id: string; object: "file"; deleted: boolean }
export interface ListPage<T> { object: "list"; data: T[]; first_id: string | null; last_id: string | null; has_more: boolean }
export type FileList = ListPage<FileObject>;
export type BatchList = ListPage<Batch>;
/** At most 16 pairs, keys of 1–64 and values of up to 512 characters, in insertion order (checked by the server).
 * JavaScript enumerates integer-like keys first; use textual keys when order matters.
 */
export type Metadata = Record<string, string>;
export interface ExpiresAfter { anchor: "created_at"; seconds: number }
export type BatchStatus =
  | "validating" | "failed" | "in_progress" | "finalizing" | "completed" | "expired" | "cancelling" | "cancelled";
export type BatchErrorCode =
  | "empty_file" | "empty_line" | "invalid_json_line" | "line_too_large" | "too_many_records" | "invalid_custom_id"
  | "duplicate_custom_id" | "invalid_method" | "invalid_url" | "stream_unsupported" | "invalid_body"
  | "model_not_found" | "mixed_models" | "budget_exceeded" | "host_budget_exceeded" | "unbounded_feature"
  | "input_changed" | "batch_expired" | "errors_truncated";
export interface RequestCounts { total: number; completed: number; failed: number }
export interface BatchErrorEntry { code: BatchErrorCode | (string & {}); message: string; param: string | null; line: number | null }
export interface BatchErrors { object: "list"; data: BatchErrorEntry[] }
export interface Batch {
  id: string;
  object: "batch";
  endpoint: "/v1/basal" | (string & {});
  model: string | null;
  errors: BatchErrors | null;
  input_file_id: string;
  completion_window: "24h" | (string & {});
  status: BatchStatus | (string & {});
  output_file_id: string | null;
  error_file_id: string | null;
  created_at: number;
  in_progress_at: number | null;
  /** End of the 24h inference window; export may finish later. */
  expires_at: number;
  finalizing_at: number | null;
  completed_at: number | null;
  failed_at: number | null;
  expired_at: number | null;
  cancelling_at: number | null;
  cancelled_at: number | null;
  request_counts: RequestCounts;
  metadata: Metadata | null;
}
/** `POST /v1/batches` body. `endpoint` and `completion_window` default to their only accepted values. */
export interface CreateBatchRequest {
  input_file_id: string;
  endpoint?: "/v1/basal";
  completion_window?: "24h";
  metadata?: Metadata | null;
  /** Lifetime of each result file, anchored at its creation. */
  output_expires_after?: ExpiresAfter | null;
}
/** One line of an input file; `body` is the native `/v1/basal` request (see `buildRequest`). */
export interface BatchInputLine { custom_id: string; method?: "POST"; url?: "/v1/basal"; body: DecisionRequest }
/** One line of a result file. `response.body` is the native answer: a `DecisionResponse` for status 200, otherwise
 * the native error body. `error` is set when the record has no response (e.g. `batch_expired`, `result_too_large`).
 */
export interface BatchResultLine {
  id: string;
  custom_id: string;
  response: { status_code: number; request_id: string; body: { [key: string]: JsonValue } } | null;
  error: { code: string; message: string } | null;
}

/** Common options of Batch and Files calls. */
export interface RequestOptions { signal?: AbortSignal }
/** Upload and download options; `timeoutMs` replaces the client deadline for this call only. */
export interface TransferOptions extends RequestOptions { timeoutMs?: number }
export interface DownloadOptions extends TransferOptions {
  /** Byte cap of this download; defaults to the client's `maxResponseBytes`. The body is buffered in memory. */
  maxBytes?: number;
}
export interface CreateBatchOptions extends RequestOptions {
  /** 1–128 visible ASCII characters; a replay with the same request data returns the existing batch. */
  idempotencyKey?: string;
}
export interface ListOptions extends RequestOptions { after?: string; limit?: number }
export interface ListFilesOptions extends ListOptions { purpose?: FilePurpose }
/** Caller-owned upload content. The SDK never fetches URLs or reads paths; a stream is consumed and locked. */
export type UploadContent = Uint8Array | ArrayBuffer | Blob | ReadableStream<Uint8Array>;
export interface UploadFileRequest {
  file: UploadContent;
  /** 1–255 UTF-8 bytes without control characters or backslashes; stored as given, never used as a path. */
  filename: string;
  purpose?: "batch";
  /** File lifetime (server default 30 days). */
  expires_after?: ExpiresAfter;
}

const encoder = new TextEncoder();

/** Path segment of a resource id. `.` and `..` would be resolved as dot segments by URL parsing. */
export function resourcePath(prefix: string, id: string, suffix = ""): string {
  if (typeof id !== "string" || !id || id === "." || id === "..") throw new TypeError("resource id must be a nonempty string");
  return `${prefix}/${encodeURIComponent(id)}${suffix}`;
}

export function listPath(path: string, options: ListFilesOptions): string {
  const query = new URLSearchParams();
  if (options.after !== undefined) {
    if (typeof options.after !== "string" || !options.after) throw new TypeError("after must be a nonempty id");
    query.set("after", options.after);
  }
  if (options.limit !== undefined) {
    if (!Number.isSafeInteger(options.limit) || options.limit < 1) throw new TypeError("limit must be a positive integer");
    query.set("limit", String(options.limit));
  }
  if (options.purpose !== undefined) {
    if (typeof options.purpose !== "string" || !options.purpose) throw new TypeError("purpose must be a nonempty string");
    query.set("purpose", options.purpose);
  }
  const encoded = query.toString();
  return encoded ? `${path}?${encoded}` : path;
}

export function idempotencyHeader(key: string | undefined): Record<string, string> {
  if (key === undefined) return {};
  if (typeof key !== "string" || !/^[!-~]{1,128}$/.test(key)) {
    throw new TypeError("idempotencyKey must have 1-128 visible ASCII characters");
  }
  return { "Idempotency-Key": key };
}

export function createBody(request: CreateBatchRequest): CreateBatchRequest {
  if (!request || typeof request !== "object" || typeof request.input_file_id !== "string" || !request.input_file_id) {
    throw new TypeError("input_file_id must be a nonempty string");
  }
  // Extra caller fields are forwarded; the server refuses fields it does not know.
  return { endpoint: "/v1/basal", completion_window: "24h", ...request };
}

function expiresFields(expires: ExpiresAfter | undefined): [string, string][] {
  if (expires === undefined) return [];
  if (!expires || expires.anchor !== "created_at" || !Number.isSafeInteger(expires.seconds) || expires.seconds < 0) {
    throw new TypeError("expires_after needs anchor created_at and whole seconds");
  }
  return [["expires_after[anchor]", expires.anchor], ["expires_after[seconds]", String(expires.seconds)]];
}

/** A multipart body; a stream body is created per request and stops its source when `signal` aborts. */
export interface Multipart { contentType: string; body: Blob | ((signal: AbortSignal) => ReadableStream<Uint8Array>) }

/** Multipart form of an upload: `expires_after[...]`, then `purpose`, then the file part last (the server reserves the
 * upload session at the file part, so expiry fields must precede it).
 */
export function uploadForm(request: UploadFileRequest): Multipart {
  if (!request || typeof request !== "object") throw new TypeError("upload request must be an object");
  const { file, filename } = request;
  if (request.purpose !== undefined && request.purpose !== "batch") throw new TypeError("purpose must be batch");
  // Control characters (CR/LF) would split the part header; a backslash is ambiguous in a quoted filename.
  if (typeof filename !== "string" || !filename || encoder.encode(filename).length > 255 || /[\p{Cc}\\]/u.test(filename)) {
    throw new TypeError("filename must have 1-255 UTF-8 bytes without control characters or backslashes");
  }
  const random = new Uint8Array(16);
  globalThis.crypto.getRandomValues(random);
  const boundary = `basal-${Array.from(random, b => b.toString(16).padStart(2, "0")).join("")}`;
  let head = "";
  for (const [name, value] of [...expiresFields(request.expires_after), ["purpose", "batch"] as [string, string]]) {
    head += `--${boundary}\r\nContent-Disposition: form-data; name="${name}"\r\n\r\n${value}\r\n`;
  }
  head += `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="${filename.replaceAll('"', '\\"')}"\r\n`
    + "Content-Type: application/jsonl\r\n\r\n";
  const prefix = encoder.encode(head);
  const suffix = encoder.encode(`\r\n--${boundary}--\r\n`);
  const contentType = `multipart/form-data; boundary=${boundary}`;
  if (file instanceof ReadableStream) {
    if (file.locked) throw new TypeError("upload stream is locked");
    return { contentType, body: signal => framedStream(prefix, file, suffix, signal) };
  }
  let content: BlobPart;
  if (file instanceof Blob || file instanceof ArrayBuffer) content = file;
  else if (file instanceof Uint8Array) content = file.buffer instanceof ArrayBuffer ? file as Uint8Array<ArrayBuffer> : new Uint8Array(file);
  else throw new TypeError("file must be a Uint8Array, ArrayBuffer, Blob or ReadableStream");
  return { contentType, body: new Blob([prefix, content, suffix]) };
}

/** Streams the multipart prefix, the caller's chunks and the suffix without buffering the file.
 * Fetch implementations may keep pulling a request body after abort, so `signal` cancels the source here.
 */
function framedStream(
  prefix: Uint8Array, source: ReadableStream<Uint8Array>, suffix: Uint8Array, signal: AbortSignal,
): ReadableStream<Uint8Array> {
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let stage: "prefix" | "file" | "finished" | "stopped" = "prefix";
  const stop = (reason: unknown): Promise<void> => {
    signal.removeEventListener("abort", onAbort);
    if (stage === "finished" || stage === "stopped") return Promise.resolve();
    stage = "stopped";
    return (reader ? reader.cancel(reason) : source.cancel(reason)).catch(() => {});
  };
  const onAbort = () => { void stop(signal.reason); };
  signal.addEventListener("abort", onAbort, { once: true });
  if (signal.aborted) onAbort();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (stage === "stopped") throw signal.reason ?? new Error("upload stopped");
      if (stage === "prefix") { stage = "file"; controller.enqueue(prefix); return; }
      if (stage === "finished") { controller.close(); return; }
      reader ??= source.getReader();
      const { done, value } = await reader.read();
      // A read cancelled by `stop` ends as done; it must not be framed as a complete file.
      if (stage !== "file") throw signal.reason ?? new Error("upload stopped");
      if (!done) {
        if (!(value instanceof Uint8Array)) {
          const error = new TypeError("upload stream chunks must be Uint8Array");
          await stop(error);
          throw error;
        }
        controller.enqueue(value);
        return;
      }
      reader.releaseLock();
      reader = undefined;
      stage = "finished";
      signal.removeEventListener("abort", onAbort);
      controller.enqueue(suffix);
    },
    cancel(reason) { return stop(reason); },
  });
}

/** JSONL input file of native records (`method` POST and `url` /v1/basal are filled in). Never sends anything. */
export function batchInput(lines: Iterable<BatchInputLine>): Uint8Array {
  const out: string[] = [];
  for (const line of lines) {
    if (!line || typeof line.custom_id !== "string" || !line.custom_id || !line.body || typeof line.body !== "object") {
      throw new TypeError("each batch line needs a custom_id and a native request body");
    }
    out.push(JSON.stringify({ custom_id: line.custom_id, method: line.method ?? "POST", url: line.url ?? "/v1/basal", body: line.body }));
  }
  if (!out.length) throw new TypeError("a batch input needs at least one line");
  return encoder.encode(out.join("\n") + "\n");
}

/** Parse downloaded result bytes (output or error file) into lines. Never fetches anything. */
export function parseBatchResults(content: Uint8Array): BatchResultLine[] {
  let text: string;
  try { text = new TextDecoder("utf-8", { fatal: true }).decode(content); }
  catch (cause) { throw new BasalProtocolError("Batch result file is not UTF-8", { cause }); }
  const lines = text.split("\n");
  if (lines[lines.length - 1] === "") lines.pop();
  return lines.map((line, index) => {
    let value: unknown;
    try { value = JSON.parse(line); }
    catch (cause) { throw new BasalProtocolError(`Batch result line ${index + 1} is not JSON`, { cause }); }
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new BasalProtocolError(`Batch result line ${index + 1} is not a JSON object`);
    }
    return value as BatchResultLine;
  });
}
