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
