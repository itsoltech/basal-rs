// HTTP client for the console. Every request carries the session cookie and the X-Basal-UI header the server
// requires for cookie-authenticated calls (a custom header forces a CORS preflight, which the server never grants).

let unauthorizedHandler = null;

/** Called once per 401 response (unless the caller opts out), e.g. to return to the login screen. */
export function onUnauthorized(handler) {
  unauthorizedHandler = handler;
}

export class ApiError extends Error {
  constructor(status, data, text, headers) {
    super(errorText(status, data, text));
    this.status = status;
    this.data = data;
    this.text = text;
    this.headers = headers;
  }

  /** Machine code of an OpenAI-style error envelope (`{"error":{"code":…}}`), if any. */
  get code() {
    const e = this.data && this.data.error;
    return e && typeof e === 'object' ? e.code || null : null;
  }
}

/**
 * Low-level request. Never throws on HTTP errors: returns status, headers, raw text and parsed JSON (or null).
 * Throws ApiError(0) on network failures.
 */
export async function send(path, { method = 'GET', json, body, headers = {}, signal, quiet401 = false } = {}) {
  const init = {
    method,
    credentials: 'same-origin',
    cache: 'no-store',
    headers: { 'X-Basal-UI': '1', ...headers },
    signal,
  };
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    init.headers['Content-Type'] = 'application/json';
  } else if (body !== undefined) {
    init.body = body;
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new ApiError(0, null, `Network error: ${e.message}`, new Headers());
  }
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (res.status === 401 && !quiet401 && unauthorizedHandler) unauthorizedHandler();
  return { ok: res.ok, status: res.status, statusText: res.statusText, headers: res.headers, text, data };
}

/** JSON request that throws ApiError on any non-2xx status. Resolves to parsed JSON (null for empty bodies). */
export async function api(path, options) {
  const r = await send(path, options);
  if (!r.ok) throw new ApiError(r.status, r.data, r.text, r.headers);
  return r.data;
}

/** Download a response body as a Blob (used for file content, which needs the header and cookie). */
export async function fetchBlob(path) {
  let res;
  try {
    res = await fetch(path, { credentials: 'same-origin', headers: { 'X-Basal-UI': '1' } });
  } catch (e) {
    throw new ApiError(0, null, `Network error: ${e.message}`, new Headers());
  }
  if (!res.ok) {
    const text = await res.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch {
      /* not JSON */
    }
    if (res.status === 401 && unauthorizedHandler) unauthorizedHandler();
    throw new ApiError(res.status, data, text, res.headers);
  }
  return res.blob();
}

/**
 * Human-readable lines of an error body. Understands the shapes served by basal:
 * OpenAI-style `{"error":{message,code}}`, native `{"error":"…"}`, TypeSafe `{"detail":"…"}`,
 * `{"detail":{"message":…}}` and pydantic-style 422 `{"detail":[{loc,msg,type}]}`.
 */
export function errorLines(data, text) {
  if (data && typeof data === 'object') {
    const { error, detail } = data;
    if (error && typeof error === 'object') {
      return [error.message || 'Error'].concat(error.code ? [`Code: ${error.code}`] : []);
    }
    if (Array.isArray(detail)) {
      return detail.map((d) => {
        if (!d || typeof d !== 'object') return String(d);
        const loc = Array.isArray(d.loc) ? d.loc.join(' › ') : '';
        const msg = d.msg || d.message || JSON.stringify(d);
        return loc ? `${loc}: ${msg}${d.type ? ` (${d.type})` : ''}` : msg;
      });
    }
    if (typeof detail === 'string') return [detail];
    if (detail && typeof detail === 'object') {
      return [detail.message || JSON.stringify(detail)].concat(detail.error_type ? [`Type: ${detail.error_type}`] : []);
    }
    if (typeof error === 'string') return [error];
  }
  if (text) return [text.length > 400 ? `${text.slice(0, 400)}…` : text];
  return [];
}

function errorText(status, data, text) {
  const lines = errorLines(data, text);
  if (status === 0) return lines[0] || 'Network error';
  return `HTTP ${status}${lines.length ? `: ${lines.join('; ')}` : ''}`;
}

/** Message for any thrown value, for toasts and inline errors. */
export function errorMessage(e) {
  if (e instanceof ApiError) return e.message;
  return e && e.message ? e.message : String(e);
}

export const enc = encodeURIComponent;
