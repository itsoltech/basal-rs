import { buildRequest, type QuestionSet, type TypedDecisionResponse, type DecisionOptions } from "./questions.js";
import { BasalError, BasalHTTPError, BasalProtocolError, BasalResponseTooLargeError, BasalTransportError } from "./errors.js";
import {
  createBody, idempotencyHeader, listPath, resourcePath, uploadForm, type Batch, type BatchList, type CreateBatchOptions,
  type CreateBatchRequest, type DeletedFile, type DownloadOptions, type FileList, type FileObject, type ListFilesOptions,
  type ListOptions, type RequestOptions, type TransferOptions, type UploadFileRequest,
} from "./batches.js";
export * from "./questions.js";
export * from "./media.js";
export * from "./tasks.js";
export * from "./errors.js";
export { batchInput, parseBatchResults } from "./batches.js";
export type {
  Batch, BatchErrorCode, BatchErrorEntry, BatchErrors, BatchInputLine, BatchList, BatchResultLine, BatchStatus,
  CreateBatchOptions, CreateBatchRequest, DeletedFile, DownloadOptions, ExpiresAfter, FileList, FileObject, FilePurpose,
  ListFilesOptions, ListOptions, ListPage, Metadata, RequestCounts, RequestOptions, TransferOptions, UploadContent,
  UploadFileRequest,
} from "./batches.js";
import type {TaskAnswer} from "./tasks.js";
/** Native Basal wire types. Criterion order is significant; do not sort keys. */
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };
export type Calibration = "validated" | "unvalidated" | "text";
export type MultiObjective = "threshold" | "exact" | "iou" | "f1";
export interface MediaInput { type: "image" | "audio"; data: string }
export interface MediaUsage { images: number; audio: number; media_tokens: number; detail: "default" | "high" }
export interface ToolRanking { tool: string; probability: number }
export type ToolDefinition = Record<string, JsonValue>;
export interface Question {
  type?: "choice" | "noul" | "score" | "multi" | "act" | "tool" | "classify" | "verify" | "compress" | "extract";
  instructions?: JsonValue;
  criteria?: Record<string, JsonValue> | JsonValue[];
  levels?: Record<string, JsonValue> | JsonValue[];
  option_keys?: "show" | "hide";
  evidence?: boolean;
  threshold?: number;
  min?: number;
  max?: number;
  base?: "noul" | "choice" | "score";
  costs?: Record<string, number> | Record<string, Record<string, number>>;
  defer_action?: string;
  max_error?: number | readonly number[];
  objective?: MultiObjective;
  exclusive?: readonly string[];
  fallback?: string;
  fallback_threshold?: number;
  tools?: readonly ToolDefinition[];
  extra?: readonly ("answer_directly" | "ask_user" | "none")[];
  rank?: number;
  description_chars?: number;
  multi?: boolean;
  labels?: Record<string,JsonValue> | JsonValue[];
  multi_label?: boolean; none?: boolean;
  claims?: Record<string,JsonValue> | JsonValue[]; binary?: boolean;
  unit?: 'sentence'|'line'|'paragraph'; keep_ratio?: number; min_keep?: number;
  schema?: Record<string,JsonValue>;
}
export interface DecisionRequest {
  state: JsonValue;
  questions: Record<string, Question>;
  model?: string;
  facts?: "off" | "auto";
  lang?: "pl" | "en";
  media?: readonly MediaInput[];
  detail?: "default" | "high";
}
export interface EvidenceSpan { text: string; start: number; end: number; probability: number }
interface BaseAnswer<K extends string = string> { probabilities: Record<K, number>; evidence?: EvidenceSpan[]; calibration?: Calibration }
export interface ChoiceAnswer<K extends string = string> extends BaseAnswer<K> { type: "choice"; choice: K; confidence: number }
export interface NoulAnswer extends BaseAnswer<"true" | "false"> { type: "noul"; noul: number; confidence: number }
export interface ScoreAnswer extends BaseAnswer { type: "score"; score: number; legend: Record<string, string>; confidence: number }
export interface SelectionResult<K extends string = string> {
  threshold?: number; objective?: MultiObjective; expected_score?: number | null; constraints_applied?: JsonValue;
  set_confidence: number;
}
export interface MultiAnswer<K extends string = string> extends BaseAnswer<K>, SelectionResult<K> { type: "multi"; selected: K[] }
export interface SingleToolAnswer extends BaseAnswer {
  type: "tool"; multi?: false; tool: string | null; call: boolean; confidence: number;
  ranking?: ToolRanking[]; deferred?: boolean; refused?: string; max_error?: string;
}
export interface MultiToolAnswer extends BaseAnswer, SelectionResult {
  type: "tool"; multi: true; tools: string[]; call: boolean; ranking?: ToolRanking[];
}
export type ToolAnswer = SingleToolAnswer | MultiToolAnswer;
export interface ActAnswer<K extends string = string, A extends string = string> extends BaseAnswer<K> {
  type: "act";
  action: A;
  answer: K;
  expected_costs: Record<A, number>;
  confidence: number;
  calibration: Calibration;
  refused?: string;
  max_error?: string;
  deferred?: boolean;
}
export type Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer | MultiAnswer | ActAnswer | ToolAnswer | TaskAnswer;
export interface Usage { input_tokens: number; output_tokens: number; questions?: number; branches?: number; latency_ms?: number; longest_prompt_tokens?: number; second_order_branches?: number; media?: MediaUsage }
export interface DecisionResponse { model: string; answers: Record<string, Answer>; usage: Usage }
export interface ModelMetadata { name: string; description?: string; release_date?: string; mode?: string; early_exit?: JsonValue[]; prompt_format?: number; orders?: JsonValue; temperature_per_prim?: Record<string,number>; media?: ("image"|"audio")[] }
export interface ModelsResponse { models: ModelMetadata[] }

/** Web Fetch input accepted by browsers and server runtimes. */
export type RequestInput = string | URL | Request;
/** A transport adapter must honor signal/redirect and return a standard Web Response. */
export type Fetch = (input: RequestInput, init?: RequestInit) => Promise<Response>;
export interface BasalClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  maxResponseBytes?: number;
  /** Defaults to the runtime's native fetch; custom HTTP clients can supply a Fetch adapter. */
  fetch?: Fetch;
}

async function readBody(response: Response, limit: number, signal: AbortSignal): Promise<Uint8Array> {
  // Count bytes from the actual stream, including decompressed responses; never trust Content-Length alone.
  const reader = response.body?.getReader();
  if (!reader) return new Uint8Array();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener("abort", cancel, { once: true });
  if (signal.aborted) cancel();
  const parts: Uint8Array[] = [];
  let bytes = 0;
  let complete = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) { complete = true; break; }
      bytes += value.byteLength;
      if (bytes > limit) {
        throw new BasalResponseTooLargeError(`Basal response exceeds ${limit} bytes`);
      }
      parts.push(value);
    }
  } finally {
    signal.removeEventListener("abort", cancel);
    // A read failure must also stop the underlying HTTP body, before releasing ownership.
    if (!complete) cancel();
    reader.releaseLock();
  }
  if (parts.length === 1) return parts[0]!;
  const out = new Uint8Array(bytes);
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.byteLength; }
  return out;
}

/** One HTTP exchange. `json` is serialized; `body` is sent as is (multipart upload). */
interface Call {
  method: string;
  path: string;
  json?: unknown;
  /** A function creates a streamed body bound to the request's abort signal. */
  body?: Blob | ((signal: AbortSignal) => ReadableStream<Uint8Array>);
  headers?: Record<string, string>;
  signal?: AbortSignal | undefined;
  timeoutMs?: number | undefined;
  maxBytes?: number | undefined;
  /** Return the successful body as bytes, without decoding or JSON parsing. */
  bytes?: boolean;
}

function positiveTimeout(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0 || value > 2_147_483_647) {
    throw new TypeError(`${name} must be positive and at most 2147483647`);
  }
  return value;
}
function byteLimit(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive safe integer`);
  return value;
}

/** Reusable fetch client. Server root URL, optional proxy prefix, without /v1.
 * No automatic retries. AbortSignal cancels a request; timeout includes body reading.
 */
export class BasalClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;
  private readonly maxResponseBytes: number;
  private readonly fetch: Fetch;

  constructor(options: BasalClientOptions = {}) {
    const url = new URL(options.baseUrl ?? "http://127.0.0.1:8000");
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.href.includes("?") || url.href.includes("#")) {
      throw new TypeError("baseUrl must be an HTTP(S) server URL without credentials, query or fragment");
    }
    if (url.protocol !== "https:" && !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) {
      throw new TypeError("baseUrl requires HTTPS or loopback HTTP (e.g. an SSH tunnel)");
    }
    this.baseUrl = url.href.replace(/\/+$/, "") + "/";
    this.timeoutMs = options.timeoutMs ?? 120_000;
    if (!Number.isFinite(this.timeoutMs) || this.timeoutMs <= 0 || this.timeoutMs > 2_147_483_647) {
      throw new TypeError("timeoutMs must be positive and at most 2147483647");
    }
    this.headers = { Accept: "application/json" };
    this.maxResponseBytes = options.maxResponseBytes ?? 8 * 1024 * 1024;
    if (!Number.isSafeInteger(this.maxResponseBytes) || this.maxResponseBytes <= 0) {
      throw new TypeError("maxResponseBytes must be a positive safe integer");
    }
    this.fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    if (options.apiKey !== undefined) {
      if (!options.apiKey || /[\r\n]/.test(options.apiKey)) throw new TypeError("apiKey must be a nonempty, single-line token");
      this.headers.Authorization = `Bearer ${options.apiKey}`;
    }
  }

  private request<T>(method: string, path: string, body?: DecisionRequest, signal?: AbortSignal): Promise<T> {
    return this.call({ method, path, json: body, signal }) as Promise<T>;
  }

  private call(call: Call & { bytes: true }): Promise<Uint8Array>;
  private call(call: Call): Promise<object>;
  private async call(call: Call): Promise<object> {
    // Serialization and option errors are local input errors, not failures of the HTTP transport.
    const encoded = call.json === undefined ? undefined : JSON.stringify(call.json);
    const timeoutMs = call.timeoutMs === undefined ? this.timeoutMs : positiveTimeout(call.timeoutMs, "timeoutMs");
    const limit = call.maxBytes === undefined ? this.maxResponseBytes : byteLimit(call.maxBytes, "maxBytes");
    const headers: Record<string, string> = { ...this.headers, ...call.headers };
    if (encoded !== undefined) headers["Content-Type"] = "application/json";
    const { signal } = call;
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    const timer = setTimeout(() => controller.abort(new Error("Basal request timed out")), timeoutMs);
    let onAbort: () => void = () => {};
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(new BasalTransportError("Basal HTTP request aborted", { cause: controller.signal.reason }));
      controller.signal.addEventListener("abort", onAbort, { once: true });
      if (controller.signal.aborted) onAbort();
    });
    const work = async (): Promise<object> => {
      let response: Response;
      let content: Uint8Array;
      let raw = "";
      try {
        const init: RequestInit & { duplex?: "half" } = {
          method: call.method, headers, signal: controller.signal, redirect: "error", credentials: "omit", cache: "no-store",
        };
        if (encoded !== undefined) init.body = encoded;
        else if (call.body instanceof Blob) init.body = call.body;
        else if (call.body !== undefined) {
          // Streamed request bodies require half-duplex fetch (Node.js, Chromium over HTTP/2).
          init.body = call.body(controller.signal);
          init.duplex = "half";
        }
        response = await this.fetch(this.baseUrl + call.path, init);
        if (response.redirected || ([301, 302, 303, 307, 308].includes(response.status) && response.headers.has("Location"))) {
          await response.body?.cancel().catch(() => {});
          throw new BasalTransportError("Basal transport returned a redirect");
        }
        // A per-download cap limits successful artifact bytes. Error envelopes retain the client's normal cap
        // so a small artifact cap does not hide an actionable 404/429 and its Retry-After.
        content = await readBody(response, response.ok ? limit : this.maxResponseBytes, controller.signal);
        if (!call.bytes || !response.ok) raw = new TextDecoder("utf-8", { fatal: true }).decode(content);
      } catch (cause) {
        if (cause instanceof BasalError) throw cause;
        throw new BasalTransportError("Basal HTTP transport failed", { cause });
      }
      if (call.bytes && response.ok) return content;
      let data: unknown;
      try { data = JSON.parse(raw); }
      catch (cause) {
        if (response.ok) throw new BasalProtocolError("Basal returned invalid JSON", { cause });
        data = raw;
      }
      if (!response.ok) throw new BasalHTTPError(response.status, data, response.headers.get("Retry-After"));
      if (data === null || typeof data !== "object" || Array.isArray(data)) {
        throw new BasalProtocolError("Basal returned a JSON value other than an object");
      }
      return data;
    };
    try {
      if (controller.signal.aborted) return await aborted;
      return await Promise.race([work(), aborted]);
    } finally {
      controller.signal.removeEventListener("abort", onAbort);
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      // An answer can arrive before a streamed upload is fully sent (e.g. 413); stop its source.
      if (typeof call.body === "function") controller.abort(new Error("Basal request finished"));
    }
  }

  decide(request: DecisionRequest, signal?: AbortSignal): Promise<DecisionResponse>;
  decide<const Q extends QuestionSet>(state: JsonValue, questions: Q, options?: DecisionOptions): Promise<TypedDecisionResponse<Q>>;
  async decide(state: DecisionRequest | JsonValue, questionsOrSignal?: QuestionSet | AbortSignal, options: DecisionOptions = {}): Promise<DecisionResponse> {
    if (questionsOrSignal === undefined || ("aborted" in questionsOrSignal && typeof questionsOrSignal.addEventListener === "function")) {
      return this.request("POST", "v1/basal", state as DecisionRequest, questionsOrSignal as AbortSignal | undefined);
    }
    const questions = {...questionsOrSignal as QuestionSet};
    const response = await this.request<DecisionResponse>("POST", "v1/basal", buildRequest(state as JsonValue, questions, options), options.signal);
    const parsed = new Map<string, Answer>();
    try {
      if (typeof response.model !== "string" || !response.model || !response.usage || Array.isArray(response.usage)
          || !Number.isSafeInteger(response.usage.input_tokens) || response.usage.input_tokens < 0
          || !Number.isSafeInteger(response.usage.output_tokens) || response.usage.output_tokens < 0
          || !response.answers || Array.isArray(response.answers)) throw new TypeError("invalid decision envelope");
      for (const [id, question] of Object.entries(questions)) parsed.set(id, question.read(response, id));
    }
    catch (cause) { throw new BasalProtocolError("Invalid typed Basal answer", {cause}); }
    const typed = {...response, answers: Object.fromEntries(parsed)};
    Object.defineProperty(typed, "raw", {value: response});
    Object.defineProperty(typed, "answer", {value: (key: string | number): Answer => {
      if (typeof key !== "string" && typeof key !== "number") throw new TypeError("answer key must be a string or number");
      const answer = parsed.get(String(key));
      if (answer === undefined) throw new TypeError("answer key was not part of this response");
      return answer;
    }});
    return typed as DecisionResponse;
  }
  models(signal?: AbortSignal): Promise<ModelsResponse> { return this.request("GET", "v1/models", undefined, signal); }

  /** Upload a JSONL input file (`purpose=batch`) from caller-owned bytes, a Blob or a stream; never fetches URLs. */
  async uploadFile(request: UploadFileRequest, options: TransferOptions = {}): Promise<FileObject> {
    const form = uploadForm(request);
    return await this.call({
      method: "POST", path: "v1/files", body: form.body, headers: { "Content-Type": form.contentType },
      signal: options.signal, timeoutMs: options.timeoutMs,
    }) as FileObject;
  }
  async listFiles(options: ListFilesOptions = {}): Promise<FileList> {
    return await this.call({ method: "GET", path: listPath("v1/files", options), signal: options.signal }) as FileList;
  }
  async getFile(fileId: string, options: RequestOptions = {}): Promise<FileObject> {
    return await this.call({ method: "GET", path: resourcePath("v1/files", fileId), signal: options.signal }) as FileObject;
  }
  async deleteFile(fileId: string, options: RequestOptions = {}): Promise<DeletedFile> {
    return await this.call({ method: "DELETE", path: resourcePath("v1/files", fileId), signal: options.signal }) as DeletedFile;
  }
  /** Raw file bytes (JSONL), buffered up to `maxBytes` (default `maxResponseBytes`); never JSON-parsed. */
  async downloadFile(fileId: string, options: DownloadOptions = {}): Promise<Uint8Array> {
    return await this.call({
      method: "GET", path: resourcePath("v1/files", fileId, "/content"), bytes: true,
      headers: { Accept: "application/jsonl, application/json" },
      signal: options.signal, timeoutMs: options.timeoutMs, maxBytes: options.maxBytes,
    });
  }
  async createBatch(request: CreateBatchRequest, options: CreateBatchOptions = {}): Promise<Batch> {
    const headers = idempotencyHeader(options.idempotencyKey);
    return await this.call({ method: "POST", path: "v1/batches", json: createBody(request), headers, signal: options.signal }) as Batch;
  }
  async listBatches(options: ListOptions = {}): Promise<BatchList> {
    const { after, limit } = options;
    const query = { ...(after === undefined ? {} : { after }), ...(limit === undefined ? {} : { limit }) };
    return await this.call({ method: "GET", path: listPath("v1/batches", query), signal: options.signal }) as BatchList;
  }
  async getBatch(batchId: string, options: RequestOptions = {}): Promise<Batch> {
    return await this.call({ method: "GET", path: resourcePath("v1/batches", batchId), signal: options.signal }) as Batch;
  }
  /** Request cancellation; the batch moves to `cancelling` and later `cancelled`. Does not wait. */
  async cancelBatch(batchId: string, options: RequestOptions = {}): Promise<Batch> {
    return await this.call({ method: "POST", path: resourcePath("v1/batches", batchId, "/cancel"), signal: options.signal }) as Batch;
  }
}
