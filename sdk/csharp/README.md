# ITSOL.Basal.Client

Asynchronous C# SDK for the native Basal API (`POST /v1/basal`) and its Batch/Files API. .NET 8+; no external runtime dependencies
or GPU requirement.
This package is versioned independently. Release tag: `sdk-csharp-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/csharp/CHANGELOG.md).

SDK 0.6.0 removed API-key configuration; local 0.7.0 restores the optional `apiKey`
constructor parameter for servers with API keys enabled ([API keys](#api-keys)).

Local 0.7.0 adds init properties `DecisionOptions.Think` (`off`/`auto`/`force`),
`ThinkText` (nullable bool) and `EarlyExit` (open string), also on `DecisionRequest`.
The five-parameter positional constructor and `Deconstruct` are unchanged; use
`new DecisionOptions(Model: model) { Think = "off", ThinkText = false }`.
Null options stay absent. Use a raw `JsonObject` for explicit wire `early_exit: null`.
Wire null for `think`/`think_text` is invalid. Missing/null early-exit inherits the
server default; explicit `off` also requires heads.

Usage adds optional `Lane`, raw `Truncated` (state/head/tail/dropped token counts,
`head_share`) and `ThinkQuestions`. `branches` is not multiplied by orders;
`output_tokens` may include generated notes/stop/EOS even on a skipped second prompt.
Discovery adds optional `Layout`, `CalibrationWarning`, raw `Think` (default,
available, limits, calibration provenance), open-string `EarlyExit` and `EarlyExitDefault`.
No defaults are synthesized. Read answer `think`/`meta` from
`response.Raw.Answers[id].Extra`; convenience result records omit them.
`think` is skipped-context or successful tokens/ms/calibration/no_think/text data;
its calibration identifier is independent of answer `validated`/`unvalidated`/`text`.
Nested unknown keys and required typed answer validation remain. Rebuild consumers
against the new package; no general cross-version binary compatibility is promised.
This local change does not publish a package or establish runtime parity for the new fields.

```sh
dotnet add package ITSOL.Basal.Client
```

```csharp
using ITSOL.Basal;

using var client = new BasalClient();
// These criteria may be loaded asynchronously from a database before preparation.
var criteria = new Dictionary<string, string> { ["returns"] = "Zwroty", ["support"] = "Wsparcie techniczne" };
var questions = QuestionBuilder.Prepare(q => new
{
    Department = q.Choice("Do którego działu skierować zgłoszenie?", criteria),
    Urgent = q.Noul("Czy sprawa jest pilna?"),
});
var response = await client.DecideAsync("Klient prosi o zwrot pieniędzy za uszkodzony telefon.",
    new { questions.Department, questions.Urgent });
Console.WriteLine($"Dział: {response.Answers.Department.Choice}, P(tak): {response.Answers.Urgent.Noul}");
```

Load database/enum values first, then call `QuestionBuilder.Prepare` once. Its factory is
synchronous and snapshots each built-in question's configuration. The prepared fields can be
cached and reused; they never receive answers. Each decision creates a fresh answer shape.
An asynchronous `Prepare` lambda is rejected at compile time and may report ambiguous overloads;
load its inputs before calling `Prepare`.

You can pass the whole anonymous prepared object, select a subset, rename a field, or reuse
one definition under several names:

```csharp
var one = await client.DecideAsync(state, new { Department = questions.Department });
var repeated = await client.DecideAsync(state, new
{
    FirstDepartment = questions.Department,
    SecondDepartment = questions.Department,
});
var whole = await client.DecideAsync(state, questions);
```

The two repeated members receive independent results. `response.Answers` has the same inferred
field types as the selected object, without casts or null suppression. Concurrent decisions
can share the prepared object. Reading a prepared field's result itself throws; read the
returned response instead. Fields from a previous response cannot be used as prepared definitions.

The direct-object `DecideAsync` is an extension in `ITSOL.Basal`; keep `using ITSOL.Basal` in scope.
It supports anonymous objects and named shapes with one public constructor whose parameter
names/types match all public answer members. Positional records are a convenient named form:

```csharp
public sealed record TicketQuestions(ChoiceField Department, NoulField Urgent);

// Inside the calling method:
// var prepared = QuestionBuilder.Prepare(q => new TicketQuestions(
//     q.Choice("Which department?", criteria), q.Noul("Is this urgent?")));
// var response = await client.DecideAsync(state, prepared);
// string choice = response.Answers.Department.Choice;
```

A named prepared class built with a builder-dependent constructor can select its members into
`new { questions.Department, questions.Urgent }` instead of being passed whole. Reconstruction
calls the matching public constructor before HTTP; it must preserve the supplied answer fields.
The SDK rejects constructors that swap or replace answer fields. Pass a concrete shape type;
a base/interface-typed variable is rejected instead of dropping derived questions.
Constructor metadata is shallow-copied and passed through the constructor again, so use direct
assignments rather than transformations. Members outside the constructor are not copied.
Answer getters must return stable fields. No private fields or readonly backing fields are modified.
The request order follows the selected shape's constructor parameters. Explicit wire IDs from
`q.Add` remain explicit; inferred IDs come from the selected member names. An explicit ID cannot
be repeated under multiple names. To cancel, use the named argument `cancellationToken: token`.

For questions prepared inline, the existing synchronous factory is also available:

```csharp
var response = await client.DecideAsync(state, q => new
{
    Department = q.Choice("Which department?", criteria),
    Urgent = q.Noul("Is this urgent?"),
});
```

The returned member name is the exact request ID: `Department` and `Urgent` above.
There is no separate key string. The anonymous object preserves each answer field's type.
Factories cover `Choice`, `Noul`, `Score`, `Multi`, `Act`, with the same typed options as
`Questions` factories. Read fields after awaiting
`DecideAsync`. The factory runs once, before HTTP, and the builder closes when it returns.
The factory must be synchronous: finish any asynchronous preparation before calling `DecideAsync`.
Factories returning `Task` or `ValueTask` are rejected at compile time.
An untyped `async` lambda can report ambiguous overloads; use a synchronous factory instead.
All results are validated before any field becomes readable. Failed or cancelled requests
leave captured fields unreadable; early reads throw `InvalidOperationException`.

Reuse a definition by creating one field for each member:

```csharp
var routing = Questions.Choice("Which department?", ("returns", "Returns"), ("support", "Support"));
var response = await client.DecideAsync(state, q => new
{
    Department = q.Choice(routing),
    SecondDepartment = q.Choice(routing),
});
// The two IDs have independent answers, even though the definition is shared.
```

A named class can express the same shape:

```csharp
sealed class TicketQuestions(QuestionBuilder q, ChoiceQuestion routing)
{
    public ChoiceField Department { get; } = q.Choice(routing);
    public ChoiceField SecondDepartment { get; } = q.Choice(routing);
    public NoulField Urgent { get; } = q.Noul("Is this urgent?");
}

// Inside the calling method:
// var response = await client.DecideAsync(state, q => new TicketQuestions(q, routing));
// string selected = response.Answers.Department.Choice;
```

Return inferred fields as initialized, readable public properties or public fields on the
factory's object; inference uses the compile-time returned type and examines only its top-level
answer members. Nested or `object`-typed members do not provide inferred IDs. Question order
follows field creation in the factory, independently of member-discovery order. Names preserve case.
Use an anonymous object, record, class or struct for inferred names; tuples with inferred fields
are rejected because tuple aliases are not available at runtime. Inherited base-class members
are included. Inherited interface members and hidden duplicate fields are not supported;
return a concrete shape with unambiguous members.
Answer property getters are read once before HTTP; do not construct fields inside a getter.

For inline DecideAsync factories, create a fresh object and its fields in each invocation. The SDK rejects fields from
another builder and rejects using one inferred field under two different member names before HTTP.
Reuse definitions through `q.Choice(routing)` etc. instead of caching inline-factory fields in `static` or with `??=`.
Fields created by `QuestionBuilder.Prepare` may be cached and shared as described above.
Do not use builders concurrently or add questions after the factory returns.
Fields are read-only views of checked result records; `.Result` exposes the record itself.
Member names are discovered with built-in .NET reflection; no source generator, dynamic compilation
or external dependency is added. To set a wire ID explicitly, `q.Add("custom_id", definition)` remains
available and overrides the member name; tuples using only explicit IDs also remain available.
Generic custom definitions can use `q.From(definition)` and the field's `.Result`.

For dynamic question sets, a dictionary remains available:

```csharp
var response = await client.DecideAsync("A damaged phone.",
    Questions.Create(("department", routing), ("second_department", routing)));
if (response.Answer("department") is ChoiceResult answer)
    Console.WriteLine(answer.Choice);
// response.Answers["second_department"] is also available.
```

Dictionary values have the common `AnswerResult` type and can be narrowed with pattern matching.
`Answer` now accepts a request ID, not a question object (changed before the initial release).
Use the object factory for exact named field types without pattern matching or null suppression.

For structured states, criteria and cost matrices use `System.Text.Json.Nodes.JsonObject`/`JsonArray`.
Supports `choice`, `noul`, `score`, `multi`, `act` and `tool`, with model-dependent `facts` and `evidence` capabilities. Extra native fields are retained
with `JsonExtensionData`; do not duplicate explicit property names in `Extra`.
For a completely native request, use the `DecideAsync(JsonObject)` overload. It preserves explicit
JSON null fields, including `instructions: null`; null DTO properties are omitted.

`ModelsAsync()` reads `/v1/models`. All methods accept a `CancellationToken`.
The configurable timeout (default 120 seconds) includes response-body reading. The optional `apiKey` is described in [API keys](#api-keys). Use the server root (possibly with a proxy prefix), without `/v1`.

The default constructor owns its HTTP connection pool; dispose the SDK client when finished. The constructor
accepting an existing `HttpClient` leaves ownership to its caller and supports `IHttpClientFactory`. Configure
that client's handler with `AllowAutoRedirect=false`; its own timeout also applies. The owned handler
also sets `UseProxy=false`. An injected client controls its own proxy and TLS settings; keep certificate
verification enabled and configure the client's behavior before sending requests.

HTTPS is required except for loopback HTTP, suitable for a local server or SSH tunnel. URL credentials,
query and fragment are rejected. `maxResponseBytes` defaults to 8 MiB and limits the actual stream,
including HTTP errors. An oversized response raises `BasalException`.

`BasalHttpException` provides `StatusCode`, `Body` and raw `RetryAfter`. Other SDK failures raise
`BasalException` with the original cause in `InnerException`. Caller cancellation remains an
`OperationCanceledException`. No automatic retries. Responses deserialize into typed native DTOs.

Native choice/score accepts 2–255 options for format-2 models (older model limits are checked by the server). Preserve criterion insertion order. Multi probabilities are
independent, with `SetConfidence`. Evidence offsets count Unicode characters, not UTF-16 units; prefer
`EvidenceSpan.Text` when displaying a quote. The SDK does not execute selected actions.

Namespace: `ITSOL.Basal`. Factories in `Questions` and concrete `ChoiceQuestion`, `NoulQuestion`,
`ScoreQuestion`, `MultiQuestion`, `ActQuestion` classes have typed options. `Create` builds the question
map without generic annotations. The object factory retains exact field types in
`response.Answers`; the dictionary overload returns checked records keyed by request ID.
`response.Raw` retains native DTOs/extensions. Add `DecisionOptions(Model: ..., Facts: ...)`
and a cancellation token to either the dictionary or object-factory overload; the raw DTO/JsonObject overloads remain.

## API keys

Optional, SDK 0.7.0+. A server started with `auth.enabled: true` requires a key on every route
except `/health` and `/health/batches` (and `/metrics` by default). Create the key with
`basal keys create` and pass it by name as `apiKey` (the last optional parameter of both
constructors); the client then sends `Authorization: Bearer <key>` on every inference and
Batch/Files request. With an injected `HttpClient` the header is set per request, not in its
`DefaultRequestHeaders`. The key is an opaque string: the SDK only rejects an empty or multi-line
value. The SDK does not read environment variables; the example below reads `BASAL_API_KEY`
explicitly. Without `apiKey` no header is sent, as before. A server without `auth.enabled`
ignores the header.

```csharp
using var client = new BasalClient("https://basal.example.com",
    apiKey: Environment.GetEnvironmentVariable("BASAL_API_KEY"));
```

A missing, unknown, revoked or expired key raises `BasalHttpException` with `StatusCode == 401`
(`Body["error"]["code"]` is `invalid_api_key`); a key without the route's scope gives 403
(`insufficient_scope`). An unreachable key database gives 503. The key is not part of exception
messages or `ToString()`.

## Batch API

Resource helpers (`create`, `wait`, `results`) require SDK 0.5.0+.

Requires server 0.1.13+ with the Batch API enabled ([docs/BATCHES.md](https://github.com/itsoltech/basal-rs/blob/main/docs/BATCHES.md)). The same client,
optional API key, proxy prefix, HTTPS rule, redirect refusal, timeout, cancellation and error types apply.
Batches run native `/v1/basal` records only; each JSONL line is one `BatchInputLine`.

```csharp
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ITSOL.Basal;

using var client = new BasalClient("http://127.0.0.1:8000");
var lines = new[]
{
    new BatchInputLine { CustomId = "ticket-1", Body = JsonNode.Parse("""{"state":"A damaged phone.","questions":{"urgent":{"type":"noul","instructions":"Is it urgent?"}}}""")! },
};
var jsonl = string.Join("\n", lines.Select(line => JsonSerializer.Serialize(line))) + "\n";
var input = await client.UploadFileAsync(Encoding.UTF8.GetBytes(jsonl), "tickets.jsonl",
    new ExpiresAfter { Seconds = 7 * 24 * 3600 });
var batch = await client.CreateBatchAsync(new CreateBatchRequest
{
    InputFileId = input.Id,
    Metadata = new Dictionary<string, string> { ["job"] = "tickets" },
    OutputExpiresAfter = new ExpiresAfter { Seconds = 7 * 24 * 3600 },
}, idempotencyKey: "tickets-2026-10-20");

using var deadline = new CancellationTokenSource(TimeSpan.FromHours(25));
while (!BatchStatus.IsTerminal(batch.Status))
{
    await Task.Delay(TimeSpan.FromSeconds(30), deadline.Token);
    batch = await client.GetBatchAsync(batch.Id, deadline.Token);
}
if (batch.OutputFileId is string outputId)
{
    var bytes = await client.DownloadFileContentAsync(outputId, maxBytes: 64 * 1024 * 1024);
    foreach (var line in Encoding.UTF8.GetString(bytes).Split('\n', StringSplitOptions.RemoveEmptyEntries))
    {
        var result = JsonSerializer.Deserialize<BatchResultLine>(line)!;
        Console.WriteLine($"{result.CustomId}: {result.Response?.StatusCode}");
    }
}
```

- `UploadFileAsync` accepts caller-owned bytes (`ReadOnlyMemory<byte>`/`byte[]`) or a readable `Stream`; it never
  fetches URLs. A stream is read once from its current position and is not disposed or rewound, so the upload is
  never retried. The form sends `expires_after[...]` before the file part and `purpose=batch`. File names need
  1–255 UTF-8 bytes without control characters, `"` or `\`. The client timeout covers the whole upload; use a
  client with a longer timeout for large files.
- `CreateBatchAsync` sends `endpoint=/v1/basal` and `completion_window=24h` by default. Without an
  `idempotencyKey` (1–128 visible ASCII characters) each call creates a new batch; with one, repeating the same
  request after a lost answer returns the existing batch and different data gives HTTP 409.
- `ListFilesAsync(after, limit, purpose)` and `ListBatchesAsync(after, limit)` return one `ListPage<T>`; continue
  with `after: page.LastId` while `page.HasMore`. The SDK does not page or poll on its own: `GetBatchAsync` is one
  request, so use your own interval and deadline. `CancelBatchAsync` requests cancellation and returns the batch.
- `DownloadFileContentAsync(fileId)` returns raw bytes without JSON parsing, limited by `maxResponseBytes`. Raise
  the limit for one call with `maxBytes`, or stream into a caller-owned file with
  `DownloadFileContentAsync(fileId, destination, maxBytes)`; the destination is not disposed and may hold a partial
  copy after a failure. The client timeout also covers reading the whole body. Result lines can be parsed with `BatchResultLine`; a status-200 `Response.Body` deserializes
  as `DecisionResponse`.
- `Batch.Status` is a string (see `BatchStatus`); DTOs keep unknown fields in `Extra`. IDs and query values are
  percent-escaped. Errors are `BasalHttpException` with the `{"error": {...}}` body and raw `Retry-After` (429/503).

### Resources: `client.Files` and `client.Batches`

The same operations are also available under OpenAI-style resource names. The flat methods above remain with
unchanged signatures and behavior. "OpenAI-style" refers only to resource names and the
upload → create → retrieve → content workflow: records are still native `/v1/basal` requests, not chat payloads.

| Resource method | Flat method |
|---|---|
| `Files.CreateAsync(bytes or stream, filename, expiresAfter)` | `UploadFileAsync` |
| `Files.RetrieveAsync`, `ListAsync`, `DeleteAsync` | `GetFileAsync`, `ListFilesAsync`, `DeleteFileAsync` |
| `Files.ContentAsync(fileId, maxBytes)` / `(fileId, destination, maxBytes)` | `DownloadFileContentAsync` |
| `Batches.CreateAsync(CreateBatchRequest, idempotencyKey)` | `CreateBatchAsync` |
| `Batches.RetrieveAsync`, `ListAsync`, `CancelAsync` | `GetBatchAsync`, `ListBatchesAsync`, `CancelBatchAsync` |

```csharp
// 1. Existing input file: files.create -> batches.create -> retrieve -> files.content.
var file = await client.Files.CreateAsync(Encoding.UTF8.GetBytes(jsonl), "tickets.jsonl");
var created = await client.Batches.CreateAsync(new CreateBatchRequest { InputFileId = file.Id }, idempotencyKey: "tickets-1");
var current = await client.Batches.RetrieveAsync(created.Id);
if (current.OutputFileId is string id) { byte[] raw = await client.Files.ContentAsync(id); }

// 2. Records: validate, upload and create in one call; then wait and read the results.
Batch batch;
try
{
    batch = await client.Batches.CreateAsync(
        [new BatchInputLine { CustomId = "ticket-1", Body = JsonNode.Parse("""{"state":"A damaged phone.","questions":{"urgent":{"type":"noul","instructions":"Is it urgent?"}}}""")! }],
        new CreateBatchOptions { Metadata = new Dictionary<string, string> { ["job"] = "tickets" } });
}
catch (Exception error) when (error is IBatchSubmissionFailure failure)
{
    // The input file was uploaded but creation was not confirmed: resume with the same file and key.
    batch = await client.Batches.CreateAsync(failure.Request, failure.IdempotencyKey);
}
batch = await client.Batches.WaitAsync(batch.Id, timeout: TimeSpan.FromMinutes(30));
await foreach (var line in client.Batches.ResultsAsync(batch.Id))
    Console.WriteLine($"{line.CustomId}: {line.Response?.StatusCode.ToString() ?? line.Error?.Code}");
```

- `Batches.CreateAsync(IEnumerable<BatchInputLine>, CreateBatchOptions?)` checks every record before any HTTP:
  at least one and at most 50,000 (`BasalBatches.MaxRequests`), no null record, `custom_id` of 1–256 UTF-8 bytes
  and unique byte for byte, `method` `POST`, `url` `/v1/basal`, a `JsonObject` body and no `Extra` fields; the
  generated JSONL may not exceed 200,000,000 bytes (`MaxInputBytes`). The server may be configured lower. The file
  name, idempotency key and options are also checked first. Invalid input raises `ArgumentException`.
- That JSONL is built **in memory** in one growing buffer (while it grows, transiently up to about three times the
  JSONL size, plus one serialized record) and uploaded as one file
  (`purpose=batch`, `CreateBatchOptions.Filename`/`InputExpiresAfter`). The client timeout covers the whole upload.
  For larger or streamed inputs, upload a file yourself and use the input-file overload.
- `CreateBatchOptions` holds `Endpoint` (`/v1/basal`), `CompletionWindow` (`24h`), `Metadata`,
  `OutputExpiresAfter` and `IdempotencyKey`. Without a key, a fresh `basal-sdk-…` key is generated for the call.
  The key protects only the create request, not the upload.
- An upload failure propagates unchanged. If creation then fails, `BatchSubmissionException` (with the HTTP,
  transport or response error in `InnerException`) is raised; caller cancellation raises
  `BatchSubmissionCanceledException`, which is still an `OperationCanceledException` for your token. Both implement
  `IBatchSubmissionFailure` with `InputFile`, `FileId`, `IdempotencyKey` and the `Request` to resend. The SDK never
  retries, deletes the uploaded file or cancels a batch.
- `Batches.WaitAsync(batchId, timeout, pollInterval)` retrieves immediately, then every `pollInterval` (default
  5 s) until `completed`, `failed`, `expired` or `cancelled`, and returns that `Batch` (a `failed` batch carries its
  `Errors`). One monotonic deadline (default 600 s; positive values up to 4294967294 ms) covers the whole wait,
  including in-flight HTTP requests, through a linked `CancellationToken`. Unknown statuses keep polling until it.
  On timeout `BatchWaitTimeoutException` has `BatchId` and `LastBatch` (null if no retrieval finished); the server
  batch keeps running. Your `cancellationToken` raises `OperationCanceledException`. HTTP errors propagate, no retry.
- `Batches.ResultsAsync(batchId, maxBytesPerFile)` is an `IAsyncEnumerable<BatchResultLine>`. Enumeration first
  retrieves the batch; a non-terminal batch raises `BatchNotReadyException` (`Batch`) without waiting. It then reads
  `OutputFileId` and `ErrorFileId` (once if equal). Each file is **downloaded whole into memory** within
  `maxBytesPerFile` (default `maxResponseBytes`, 8 MiB) and parsed one line at a time; this is not unbounded
  streaming, so raise the cap for large result files or use `Files.ContentAsync` with a destination stream. Lines
  keep `custom_id`, the native `Response` and the record `Error`: completed batches can contain failed records,
  expired or cancelled batches partial results. A batch without result files yields nothing; batch-level validation
  errors are in `Batch.Errors`. Invalid UTF-8, JSON or envelopes raise `BatchResultFormatException` with `FileId` and
  1-based `Line`, never the line content. Each enumeration repeats the requests.

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

```csharp
var response = await client.DecideAsync("Inspect the supplied material.", q => new {
    Route = q.Tool("Choose a tool.", new[] { JsonNode.Parse("""{"name":"inspect","description":"Inspect material","inputSchema":{"type":"object"}}""")!.AsObject() }, new ToolOptions(Extra: ["none"], Rank: 2, DescriptionChars: 120)),
    Labels = q.Multi("Select visible objects.", new Dictionary<string,string> { ["animal"]="An animal", ["vehicle"]="A vehicle" }, new MultiOptions(Objective:"f1")),
}, new DecisionOptions(Model:"basal-omni-1.0-mini", Lang:"en", Media:[Media.Image(File.ReadAllBytes("photo.jpg"))]));
Console.WriteLine(response.Answers.Route.Tool);
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

The raw `Question.MaxError` property is now `JsonNode?` to support both a scalar
and the list accepted by `classify`/`verify`. Scalar initializers still work:

```csharp
var scalar = new Question { Type = "verify", MaxError = 0.05 };
var multiple = new Question { Type = "verify", MaxError = new JsonArray(0.05, 0.1) };
```

When reading a scalar, use `question.MaxError?.GetValue<double>()`; for lists use
`AsArray()`. The typed `ClassifyOptions`/`VerifyOptions` already accept lists.
