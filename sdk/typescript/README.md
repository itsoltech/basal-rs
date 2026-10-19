# @itsoltech/basal-client

TypeScript/JavaScript SDK for the native Basal API (`POST /v1/basal`). ESM and CommonJS, with type declarations.
Node.js 22+ or modern browsers with `fetch` and `AbortController`. No runtime dependencies or GPU requirement.

This package is versioned independently. Release tag: `sdk-typescript-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/typescript/CHANGELOG.md).

```sh
npm i @itsoltech/basal-client
```

```typescript
import { BasalClient, choice, noul } from "@itsoltech/basal-client";

const client = new BasalClient({ baseUrl: "http://127.0.0.1:8000" });
const response = await client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", {
  department: choice("Do którego działu skierować zgłoszenie?", { returns: "Zwroty", support: "Wsparcie techniczne" }),
  urgent: noul("Czy sprawa jest pilna?"),
});
console.log(response.answers.department.choice); // "returns" | "support"
console.log(response.answers.urgent.noul);        // number
```

CommonJS: `const { BasalClient } = require("@itsoltech/basal-client")`.

The transport uses the standard Web Fetch contract, in browsers and on the backend:

```typescript
import { BasalClient, type Fetch, type RequestInput } from "@itsoltech/basal-client";

const transport: Fetch = async (input: RequestInput, init?: RequestInit): Promise<Response> => {
  const request = new Request(input, init);
  return fetch(request);
};
const client = new BasalClient({ baseUrl: "https://basal.example.com", fetch: transport });
```

`RequestInput = string | URL | Request`. An adapter for another HTTP client must return a Web `Response`
with status, headers and a readable body, and honor cancellation, `redirect: "error"`, TLS verification,
and `credentials: "omit"`. Those requirements apply to the adapter's HTTP client too. The default is
native `fetch`; no adapter or polyfill dependency is installed by this package. A pre-aborted signal
rejects before dispatch; the SDK deadline rejects even if an adapter fails to honor cancellation,
but such an adapter may continue its own network I/O.

`models()` reads `/v1/models`. Each method optionally takes an `AbortSignal`.
`timeoutMs` includes response-body reading. `apiKey` optionally sends a bearer token to a reverse proxy.
Use the server root (possibly with a proxy prefix), without `/v1`. HTTPS is required except for loopback
HTTP (localhost, 127.0.0.1, ::1), suitable for a local server or SSH tunnel. URL credentials/query/fragment
are rejected. Requests omit cookies and use `cache: "no-store"`.
`maxResponseBytes` defaults to 8 MiB and limits the actual streamed body, including HTTP errors;
oversized responses raise `BasalResponseTooLargeError`.

`BasalHTTPError` provides `statusCode`, `body` and the raw `retryAfter` header. Transport failures raise
`BasalTransportError`; malformed successful JSON raises `BasalProtocolError`. Original causes are kept
in `cause`. `BasalHTTPError.body` remains directly accessible but is non-enumerable, so default Node
inspection and JSON.stringify do not log it. There are no automatic retries or redirects. Types describe the native contract; successful
responses are checked to be JSON objects without full schema validation. Unknown JSON fields are preserved.

Supports `choice`, `noul`, `score`, `multi`, `act` and `tool`, with model-dependent `facts` and `evidence` capabilities. Native choice/score accepts 2–255
options for format-2 models (older model limits are checked by the server). Criterion insertion order matters; JavaScript enumerates integer-like object keys numerically,
so use textual keys (e.g. `low`, `high`) or an ordered score array. Evidence offsets count Unicode code
points; use the returned text, or `Array.from(state).slice(start, end).join("")` for textual states.

In browsers, the server/proxy must support your origin through CORS. Basal-rs does not enable CORS by default.
Keep private API keys in server-side code.

Factories and concrete classes cover `choice`, `noul`, `score`, `multi`, `act`. The two-argument
`decide(state, questions, {model, facts, signal})` infers exact question IDs, answer types, choice keys
and selected-label keys. Keep the inferred object type instead of widening it to `QuestionSet`.
`response.answer("department")` also infers the answer from its request key. The same question
definition can be reused under different keys; each key retains its own answer. `answer()` takes
a key, not a question object (changed before the initial release). Required fields are
validated before the promise resolves. Typed evidence defaults to an empty array; `response.raw`
retains the native JSON without those defaults. The one-argument raw `decide(request, signal)` remains available.


[Full documentation and publication setup](https://github.com/itsoltech/basal-rs/blob/main/docs/SDK.md).


## Basal 2.0 and Omni

Typed choice/score and act outcomes allow 2–255 options. The server remains
responsible for limits of older models and for calibration/model capabilities.
`tool` accepts OpenAI, MCP and Anthropic definition objects, optional non-tool
choices (`answer_directly`, `ask_user`, `none`), ranking, multi selection and
certified deferral. The SDK returns the decision; it never executes tools.

Multi selection supports `threshold`, `exact`, `iou` and `f1`, count limits,
`exclusive`, `fallback` and `fallback_threshold`. `threshold` is optional in
results because objective-based responses instead contain `objective` and
`expected_score`; constraint diagnostics remain available. Single tool selection
can be null on certified deferral, while multi mode returns a tool list.

Typed request options include `lang` (`pl`/`en`), ordered inline `media` and
`detail` (`default`/`high`). Media helpers accept encoded bytes, base64 or a data
URL; they never fetch URLs. Read files explicitly in your application. Decoding,
model support and resource limits are checked by the server. Omni requires a
CUDA server started with media enabled. `calibration: "text"` means text
calibration applied to media, not validated calibration for images/audio.

Usage exposes `longest_prompt_tokens`, `second_order_branches` and media counts,
soft tokens and detail. Discovery exposes `prompt_format`, `orders`, enabled
media and calibration temperatures. Model names remain strings: use exact names
from model discovery, not a hardcoded model enumeration. Keep old models and
raw JSON access available. The server validates each model’s capabilities.
Basal 2.1 task helpers are described below.

```typescript
import { BasalClient, image, tool, multi } from "@itsoltech/basal-client";
const response = await new BasalClient().decide("Inspect the supplied material.", {
  route: tool("Choose a tool.", [{name:"inspect", description:"Inspect material", inputSchema:{type:"object"}}], {extra:["none"], rank:2, description_chars:120}),
  labels: multi("Select visible objects.", {animal:"An animal", vehicle:"A vehicle"}, {objective:"f1"}),
}, {model:"basal-omni-1.0-mini", lang:"en", media:[image(new Uint8Array(await file.arrayBuffer()))]});
console.log(response.answers.route, response.answers.labels.expected_score);
```

Tool descriptions are normalized by the server. Omit `description_chars`
(`DescriptionChars` in .NET) to keep the default cap of 300 Unicode code points;
use `0` to disable truncation. Descriptions longer than a positive cap are
shortened at a word boundary where possible, with an ellipsis appended. The SDK forwards this option
without changing the supplied tool definitions.

## Experimental Basal 2.1 tasks

Typed classify, verify, compress and extract factories preserve nullable task automation,
Unicode code point offsets and nested JSON extraction values. Task answers do not share
the primitive probabilities field. See [task semantics](../../docs/TASKS.md) and
[SDK examples](../../docs/SDK.md#zadania-basal-21). These source changes require a server
with the experimental task layer; installing an SDK does not update the inference server.
