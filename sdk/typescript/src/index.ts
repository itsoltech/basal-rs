import { buildRequest, type QuestionSet, type TypedDecisionResponse, type DecisionOptions } from "./questions.js";
export * from "./questions.js";
export * from "./media.js";
export * from "./tasks.js";
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

export class BasalError extends Error {}
/** Non-2xx response. Server body and Retry-After are available without logging them. */
export class BasalHTTPError extends BasalError {
  constructor(readonly statusCode: number, readonly body: unknown, readonly retryAfter: string | null) {
    super(`Basal HTTP ${statusCode}`);
    this.name = "BasalHTTPError";
    // Node's default inspection and JSON.stringify must not print reflected state or credentials.
    Object.defineProperty(this, "body", { enumerable: false });
  }
}
export class BasalTransportError extends BasalError {}
export class BasalProtocolError extends BasalError {}
export class BasalResponseTooLargeError extends BasalError {}
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

async function readBody(response: Response, limit: number, signal: AbortSignal): Promise<string> {
  // Count bytes from the actual stream, including decompressed responses; never trust Content-Length alone.
  const reader = response.body?.getReader();
  if (!reader) return "";
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal.addEventListener("abort", cancel, { once: true });
  if (signal.aborted) cancel();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const parts: string[] = [];
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
      parts.push(decoder.decode(value, { stream: true }));
    }
    parts.push(decoder.decode());
    return parts.join("");
  } finally {
    signal.removeEventListener("abort", cancel);
    // A decoder/read failure must also stop the underlying HTTP body, before releasing ownership.
    if (!complete) cancel();
    reader.releaseLock();
  }
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

  private async request<T>(method: string, path: string, body?: DecisionRequest, signal?: AbortSignal): Promise<T> {
    // Serialization errors are local input errors, not failures of the HTTP transport.
    const encoded = body === undefined ? undefined : JSON.stringify(body);
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    const timer = setTimeout(() => controller.abort(new Error("Basal request timed out")), this.timeoutMs);
    let onAbort: () => void = () => {};
    const aborted = new Promise<never>((_, reject) => {
      onAbort = () => reject(new BasalTransportError("Basal HTTP request aborted", { cause: controller.signal.reason }));
      controller.signal.addEventListener("abort", onAbort, { once: true });
      if (controller.signal.aborted) onAbort();
    });
    const work = async (): Promise<T> => {
      let response: Response;
      let raw: string;
      try {
        response = await this.fetch(this.baseUrl + path, {
          method, headers: body === undefined ? this.headers : { ...this.headers, "Content-Type": "application/json" },
          ...(encoded === undefined ? {} : { body: encoded }),
          signal: controller.signal, redirect: "error", credentials: "omit", cache: "no-store",
        });
        if (response.redirected || ([301, 302, 303, 307, 308].includes(response.status) && response.headers.has("Location"))) {
          await response.body?.cancel().catch(() => {});
          throw new BasalTransportError("Basal transport returned a redirect");
        }
        raw = await readBody(response, this.maxResponseBytes, controller.signal);
      } catch (cause) {
        if (cause instanceof BasalError) throw cause;
        throw new BasalTransportError("Basal HTTP transport failed", { cause });
      }
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
      return data as T;
    };
    try {
      if (controller.signal.aborted) return await aborted;
      return await Promise.race([work(), aborted]);
    } finally {
      controller.signal.removeEventListener("abort", onAbort);
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
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
}
