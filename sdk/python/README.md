# basal-client (Python)

HTTP SDK for the native Basal API (`POST /v1/basal`) and the Batch/Files API. Python 3.11+; no GPU
required on the client.

This package is versioned independently. Release tag: `sdk-python-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/python/CHANGELOG.md).

SDK 0.6.0 removes API-key configuration. Remove `api_key`/`apiKey` from client initialization.
Batch/Files require server 0.1.13+; all callers use its shared storage namespace.

Local 0.7.0 adds `think="off"|"auto"|"force"`, `think_text=bool` and
`early_exit=str` keyword options to every sync/async dictionary/schema call.
Omitted/None options stay absent; raw dictionaries can send explicit `early_exit: None`.
Wire null for `think`/`think_text` is invalid. Missing/null early-exit inherits the
server default; any explicit policy, even `off`, requires a head backend.

Usage adds optional `lane`, raw `truncated` (state/head/tail/dropped token counts,
`head_share`) and `think_questions`. `branches` is not multiplied by orders;
`output_tokens` may include generated notes/stop/EOS even on a skipped second prompt.
Discovery adds optional `layout`, `calibration_warning`, raw `think` (default,
available, limits, calibration provenance), open-string `early_exit` and `early_exit_default`.
No defaults are synthesized. Read raw answer additions from
`response.raw["answers"][id]["think"]` / `["meta"]`; convenience dataclasses omit them.
`think` is skipped-context or successful tokens/ms/calibration/no_think/text data;
its calibration identifier is independent of answer `validated`/`unvalidated`/`text`.
Nested unknown keys remain available; required typed answer validation is unchanged.
Existing call forms remain; discovery policy annotations now describe string inventories.
This local change does not publish a package or establish runtime parity for the new fields.

```sh
pip install basal-client
```

Choose either a normal dictionary or a declarative class. Both use the same factories.
A class gives each response field its exact inferred answer type:

```python
from basal_client import BasalClient, Questions, choice, noul

routing = choice("Do którego działu skierować zgłoszenie?", {"returns": "Zwroty", "support": "Wsparcie techniczne"})

class TicketQuestions(Questions):
    department = routing
    second_department = routing
    urgent = noul("Czy sprawa jest pilna?")

with BasalClient("http://127.0.0.1:8000") as client:
    response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", TicketQuestions)
    print(response.answers.department.choice)         # str, ChoiceAnswer
    print(response.answers.second_department.choice)  # independent answer
    print(response.answers.urgent.noul)               # float, NoulAnswer
```

Pass the class itself, without `()`. No response class, field annotations, casts, type-checker
plugin or code generator is needed. The same definition may appear under several names.

For dynamic question sets, use a dictionary and read answers by their request keys:

```python
from basal_client import BasalClient, choice

routing = choice("Which department?", {"returns": "Returns", "support": "Support"})
with BasalClient() as client:
    response = client.decide("The phone is broken.", {
        "department": routing,
        "second_department": routing,
    })
    print(response.answers["department"])
    print(response.answer("second_department"))
```

Dictionary values are checked answer dataclasses, with the static union `AnswerResult`.
Use `isinstance(answer, ChoiceAnswer)` to narrow that union when needed. Python cannot infer
an individual answer type from an arbitrary string key; the class form supplies that information.
Choice/label keys remain `str` in Python, rather than inferred literal unions.
`answer()` now takes a request key, not a question object (a change before the initial release).

Native asynchronous HTTP with connection pooling supports both forms:

```python
from basal_client import AsyncBasalClient, Questions, noul

class Availability(Questions):
    urgent = noul("Is this urgent?")

async def decide():
    async with AsyncBasalClient(timeout=120) as client:
        response = await client.decide("The service is unavailable.", Availability)
        return response.answers.urgent.noul
```

`models()` reads `/v1/models`. API keys are not supported. The server root URL may include a proxy prefix; do not append `/v1`.
HTTPS is required except for loopback HTTP, suitable for local servers and SSH tunnels.
URL credentials/query/fragment are rejected. `trust_env=False` is the default: environment proxies and
certificate overrides are ignored. Set `trust_env=True` only when you intend to trust that configuration.
`max_response_bytes` defaults to 8 MiB and limits the actual decoded stream, including HTTP errors;
oversized responses raise `BasalResponseTooLargeError`. TLS verification remains enabled.
The client requests `Accept-Encoding: identity` and rejects compressed responses before decoding,
including when a proxy ignores that header. Configure proxies to return uncompressed JSON for this client.
Malformed or excessively nested JSON raises `BasalProtocolError`.
`timeout` is in seconds and applies to connect/read/write/pool phases, not a total wall-clock deadline.

`BasalHTTPError` provides `status_code`, `body` and the raw `retry_after` header. Transport failures raise
`BasalTransportError`; malformed successful JSON raises `BasalProtocolError`. Original exceptions are
available through `__cause__`. There are no automatic retries or redirects. Type hints describe the native
contract; successful responses are checked to be JSON objects, without full schema validation. Unknown
JSON fields are preserved. Server validation errors (422) and overload (529) remain distinct.

Supports `choice`, `noul`, `score`, `multi`, `act` and `tool`, with model-dependent `facts` and `evidence` capabilities. Criterion order matters. Native
choice/score accepts 2–255 options for format-2 models (older model limits are checked by the server); multi has independent probabilities and `set_confidence`. Evidence
offsets count Unicode characters. The SDK does not evaluate model decisions or perform selected actions.

Factories `choice`, `noul`, `score`, `multi`, `act` return concrete generic question classes.
Constructors `ChoiceQuestion(...)` etc. have the same signatures. Required response fields are
validated before returning; evidence defaults to an empty tuple. `response.raw` retains native
JSON and extensions. The one-argument `decide(request)` remains available for full native
JSON dictionaries. `model=` and `facts=` work with either two-argument form.

Schema fields may be inherited and overridden with the same question kind for compatibility
with both mypy and Pyright. Changing a field's kind or removing it with `None` is not supported
by mypy. Field order follows reversed Python MRO, like dataclasses: for `class D(M1, M2)`,
fields from `M2` precede those from `M1`; overriding a field retains its original position.
Names starting with `_` are reserved. Schema instances are populated by the SDK without
calling user constructors, and their fields cannot be reassigned. Define schemas declaratively;
do not mutate class definitions while requests are running. Dictionary question maps are also
snapshotted before I/O. Response dictionary keys are read-only; nested native data is not deeply frozen.

Typed response/answer containers do not support the copy/pickle protocols. For caching or
passing a result between processes, serialize `response.raw` as JSON. Use `copy.deepcopy(response.raw)`
when a detached native result is needed. Question definitions may also be stored in ordinary
classes, `ClassVar` attributes or dataclasses; outside `Questions` they remain question definitions.

[Full documentation and publication setup](https://github.com/itsoltech/basal-rs/blob/main/docs/SDK.md).

Runtime uses HTTPX with seven dependencies in the audited Python 3.11 graph. Building requires only the
hash-locked Flit Core backend; consumers of the wheel do not install the build backend.


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

```python
from basal_client import BasalClient, image, tool, multi
with BasalClient() as client:
    response = client.decide("Inspect the supplied material.", {
        "route": tool("Choose a tool.", [{"name": "inspect", "description": "Inspect the material", "inputSchema": {"type": "object"}}], extra=["none"], rank=2, description_chars=120),
        "labels": multi("Select visible objects.", {"animal": "An animal", "vehicle": "A vehicle"}, objective="f1"),
    }, model="basal-omni-1.0-mini", lang="en", media=[image(open("photo.jpg", "rb").read())], detail="default")
    print(response.answer("route"))
```

Tool descriptions are normalized by the server. Omit `description_chars`
(`DescriptionChars` in .NET) to keep the default cap of 300 Unicode code points;
use `0` to disable truncation. Descriptions longer than a positive cap are
shortened at a word boundary where possible, with an ellipsis appended. The SDK forwards this option
without changing the supplied tool definitions.

## Batch and Files API

Resource helpers (`create`, `wait`, `results`) require SDK 0.5.0+.

Requires basal-rs 0.1.13+ started with batches enabled. All callers share the server's
configured Batch/Files storage namespace without authentication. Both `BasalClient` and
`AsyncBasalClient` provide the same methods:

```python
import json
from basal_client import BasalClient

lines = [{"custom_id": f"ticket-{i}", "method": "POST", "url": "/v1/basal",
          "body": {"state": text, "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}}
         for i, text in enumerate(["The service is down.", "Please send an invoice."])]
jsonl = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()

with BasalClient("http://127.0.0.1:8000") as client:
    file = client.upload_file(jsonl, filename="tickets.jsonl")
    batch = client.create_batch(file["id"], metadata={"job": "tickets"}, idempotency_key="tickets-2026-10-20")
    batch = client.get_batch(batch["id"])          # poll at an interval chosen by the application
    if batch["output_file_id"] is not None:
        results = client.download_file(batch["output_file_id"], max_bytes=256 * 1024 * 1024)
        for line in results.splitlines():
            record = json.loads(line)              # BatchResultLine: custom_id, response.body, error
```

| Method | Route |
|---|---|
| `upload_file(content, *, filename="input.jsonl", purpose="batch", expires_after=None)` | `POST /v1/files` |
| `list_files(*, after=None, limit=None, purpose=None)` / `get_file(id)` / `delete_file(id)` | `/v1/files[/{id}]` |
| `download_file(id, *, max_bytes=None)` | `GET /v1/files/{id}/content` |
| `create_batch(input_file_id, *, metadata=None, output_expires_after=None, idempotency_key=None)` | `POST /v1/batches` |
| `list_batches(*, after=None, limit=None)` / `get_batch(id)` / `cancel_batch(id)` | `/v1/batches[/{id}[/cancel]]` |

Upload content is caller-owned: `bytes`, `bytearray` or `memoryview` (sent with `Content-Length`,
64 KiB at a time, without a full copy; do not modify it during the call), a binary file object or an
iterable of bytes (sent chunked). `AsyncBasalClient.upload_file` accepts bytes-like content or an
async iterable of bytes, so it never blocks the event loop on file reads. The SDK never opens paths
or fetches URLs. The form sends `purpose`, then `expires_after[anchor]`/`expires_after[seconds]`, then
the file, as the server requires. Filenames have 1–255 UTF-8 bytes without control characters,
quotes or backslashes.

`create_batch` sends endpoint `/v1/basal` and window `24h`, the only values the server accepts; a
full raw `CreateBatchRequest` dictionary is also accepted. `ExpiresAfter` is
`{"anchor": "created_at", "seconds": n}` (the server checks 3600–2592000). `idempotency_key` must have
1–128 visible ASCII characters: a replay with the same data returns the same batch, different data is
a 409. Lists return one page; pass `last_id` as `after` while `has_more` is true.

`download_file` returns the file bytes without parsing them. The whole body is buffered, bounded by
`max_bytes`, or `max_response_bytes` (8 MiB) when omitted; a larger body raises
`BasalResponseTooLargeError`. Result files can be much larger than 8 MiB: raise `max_bytes` for that
call to a size your process can hold. Resource ids are escaped as one path segment; `.`/`..` are
rejected. Responses are typed dictionaries (`FileObject`, `Batch`, `FileList`, `BatchList`, ...) that
keep fields added by newer servers; JSONL lines can be typed with `BatchInputLine` and `BatchResultLine`.

Errors, TLS, proxy prefix, redirect refusal, timeouts and `trust_env` behave as for `decide`.
`BasalHTTPError.body` keeps the `{"error": {...}}` answer and `retry_after` the raw header of 429/503.
The SDK has no automatic retries and does not poll: `create_batch` returns at once (`validating`) and
`cancel_batch` does not wait for running work. A server refusing an upload early (413) may surface as
`BasalTransportError` while the body is still being sent. Expired or cancelled batches may have partial
results. See [BATCHES.md](https://github.com/itsoltech/basal-rs/blob/main/docs/BATCHES.md) for the JSONL format.

### Resources: `client.files` and `client.batches`

Both clients also expose resource namespaces named like the OpenAI Batch API. The compatibility is
in resource names and workflow only: records still carry native `/v1/basal` requests and answers, not
OpenAI chat payloads. The flat methods above are unchanged.

| Resource method | Same as / adds |
|---|---|
| `files.create(content, ...)`, `files.retrieve(id)`, `files.list(...)`, `files.delete(id)`, `files.content(id, *, max_bytes=None)` | `upload_file`, `get_file`, `list_files`, `delete_file`, `download_file` |
| `batches.create(*, input_file_id=... \| requests=..., metadata=None, output_expires_after=None, idempotency_key=None)` | `create_batch`, or upload + create |
| `batches.retrieve(id)`, `batches.list(...)`, `batches.cancel(id)` | `get_batch`, `list_batches`, `cancel_batch` |
| `batches.wait(id, *, timeout=600, poll_interval=5)` | polls until a terminal state |
| `batches.results(id, *, max_bytes=None)` | iterates output and error records |

The OpenAI-shaped flow uploads a prepared JSONL file:

```python
with BasalClient("http://127.0.0.1:8000") as client:
    file = client.files.create(jsonl, filename="tickets.jsonl")
    batch = client.batches.create(input_file_id=file["id"], idempotency_key="tickets-2026-10-20")
    batch = client.batches.retrieve(batch["id"])
    if batch["output_file_id"] is not None:
        data = client.files.content(batch["output_file_id"], max_bytes=256 * 1024 * 1024)
```

The simplified flow builds the file from requests, waits and reads the records:

```python
from basal_client import BasalClient, BasalBatchSubmissionError

requests = [{"custom_id": f"ticket-{i}",
             "body": {"state": text, "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}}
            for i, text in enumerate(["The service is down.", "Please send an invoice."])]

with BasalClient("http://127.0.0.1:8000") as client:
    try:
        batch = client.batches.create(requests=requests, metadata={"job": "tickets"})
    except BasalBatchSubmissionError as error:   # uploaded, but creating the batch failed
        batch = client.batches.create(input_file_id=error.file_id, idempotency_key=error.idempotency_key,
                                      metadata={"job": "tickets"})
    batch = client.batches.wait(batch["id"], timeout=3600)
    for record in client.batches.results(batch["id"], max_bytes=256 * 1024 * 1024):
        if record["response"] is not None and record["response"]["status_code"] == 200:
            print(record["custom_id"], record["response"]["body"]["answers"])
        else:
            print(record["custom_id"], "failed:", record["error"] or record["response"])
```

`AsyncBasalClient` has the same resources: `await client.batches.create(...)`, `await client.batches.wait(...)`
and `async for record in client.batches.results(...)`.

`batches.create` takes exactly one of `input_file_id` and `requests`. `endpoint="/v1/basal"` and
`completion_window="24h"` may be passed for OpenAI-style code; other values are rejected. Each request
is a `BatchRequest`: `custom_id` (1–256 UTF-8 bytes, unique) and `body` (a native `/v1/basal`
request, not streaming: `body["stream"] is True` is rejected); `method` and `url` may be omitted or must
be `POST` and `/v1/basal`. Other fields are rejected.
Before anything is sent, the SDK checks the requests, options and `idempotency_key`, and serializes the
requests as strict JSON (no NaN/Infinity, no lone surrogates) into **one in-memory buffer** of at most 50000 records and
200000000 bytes; the server still validates bodies, models and its own, possibly lower, limits. The
buffer is uploaded as `requests.jsonl` with purpose `batch`, then the batch is created. Without an
`idempotency_key` the SDK generates one for the call. The async client yields to the event loop while
preparing large inputs. The uploaded input file remains after the batch; delete it with
`files.delete(batch["input_file_id"])` when no longer needed.

If the upload fails, its error propagates and no batch exists. If creating the batch fails after the
upload, `BasalBatchSubmissionError` keeps `file_id`, `file` and `idempotency_key`, with the original
error as `__cause__`. The SDK neither retries nor deletes the file: resume with `input_file_id` and the
same key and options. If the first attempt did create the batch (e.g. the response was lost), the server
replays it instead of creating another. Interrupting the call after the upload (`KeyboardInterrupt`,
`asyncio.CancelledError`) propagates that exact exception, never converted or swallowed; it carries the
same context, which `BasalBatchSubmissionError.find(error)` returns (also from the `CancelledError` of
an awaited cancelled task). An interruption during the upload has no known file id; check `files.list()`.

`batches.wait` returns the batch once it is `completed`, `failed`, `expired` or `cancelled` (`TERMINAL_STATUSES`); check
`status`, `errors` and `request_counts` yourself. The first retrieve is immediate, then it sleeps
`poll_interval` seconds between retrieves. `timeout` and `poll_interval` must be finite and positive.
Unknown statuses are treated as not terminal. HTTP and transport errors propagate without retries.
At the deadline `BasalBatchTimeoutError` carries `batch_id` and `batch`, the last batch observed (or
`None`); the server job keeps running and is never cancelled by the SDK. One deadline covers every
request, including slow headers and a slowly trickling body, and every sleep. The client `timeout`
still applies to each HTTPX phase; a phase timing out before the deadline is a `BasalTransportError`.
In `AsyncBasalClient` cancelling the awaiting task stops the wait. HTTPX has no total-request timeout
for its sync client, so `BasalClient.batches.wait` runs the async polling on a private event loop in a
worker thread, with **its own connections** (same URL, TLS verification, timeout and
`trust_env`) that are closed before it returns; it does not use or close the caller's connection pool.
The calling thread blocks until the worker is joined; nothing is left running. `KeyboardInterrupt`
cancels the polling, waits for the worker to close its connections and then propagates. Name resolution
by the operating system cannot be interrupted, so with a host name (not an IP address) a slow resolver
can delay the return beyond the deadline.

`batches.results` retrieves the batch when iteration starts, without waiting: a non-terminal batch
raises `BasalBatchNotReadyError` (`batch_id`, `status`, `batch`). It then yields the records of
`output_file_id`, followed by those of `error_file_id` (once if both ids are equal), as
`BatchResultLine` dictionaries with `custom_id`, the native `response` and `error`; unknown fields are
kept. A batch without result files yields nothing; batch-level validation errors are in `batch["errors"]`
from `retrieve`/`wait`. Expired or cancelled batches may have partial results, and a completed batch may
contain failed records — read both files. **Each file is downloaded whole into memory**, bounded by
`max_bytes` (default `max_response_bytes`, 8 MiB; larger files raise `BasalResponseTooLargeError`), and
then parsed one line at a time without building a second list. This is buffered, not unbounded
streaming: set `max_bytes` to what your process can hold. Records are not reordered; match them by
`custom_id`. An empty (also inside the file), non-UTF-8 or non-JSON line, a NaN/Infinity constant, or a
line breaking the result envelope raises `BasalBatchResultError`
(a `BasalProtocolError`) with `file_id`, the 1-based `line` and `reason`; earlier records were already
yielded. The envelope needs nonempty `id` and `custom_id` and both `response` and `error` keys with
exactly one of them non-null; `response` has `status_code` 100–599, a nonempty `request_id` and an
object `body`; `error` has a nonempty `code` and a string `message`.

## Experimental Basal 2.1 tasks

Typed classify, verify, compress and extract factories preserve nullable task automation,
Unicode code point offsets and nested JSON extraction values. Task answers do not share
the primitive probabilities field. See [task semantics](../../docs/TASKS.md) and
[SDK examples](../../docs/SDK.md#zadania-basal-21). These source changes require a server
with the experimental task layer; installing an SDK does not update the inference server.
