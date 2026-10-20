# basal-client (Rust)

Async Rust SDK for the native Basal API (`POST /v1/basal`), using Hyper HTTP/1 with Rustls and the platform certificate verifier.
No dependency on the inference engine or GPU libraries. Rust 1.95+; a Tokio runtime is required.

This package is versioned independently. Release tag: `sdk-rust-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/rust/CHANGELOG.md).

## Batch and Files API (0.5+)

### Resource helpers (0.6+)

The resource facades add the OpenAI-shaped workflow and a simplified create path. This is resource/
workflow compatibility: records are native `/v1/basal` payloads. Existing flat methods below remain available.

```rust,no_run
use basal_client::{BatchInputLine, BatchWaitOptions, Client, CreateBatchFromRequests,
    CreateBatchRequest, FileUploadOptions};
# async fn example(client: &Client, input: &[u8], requests: &[BatchInputLine]) -> Result<(), Box<dyn std::error::Error>> {
// Path A: explicitly upload, then create using the uploaded file id.
let file = client.files().create("input.jsonl", input, FileUploadOptions::default()).await?;
let batch = client.batches().create(&CreateBatchRequest::new(file.id), None).await?;
let current = client.batches().retrieve(&batch.id).await?;
if let Some(output) = &current.output_file_id {
    let bytes = client.files().content(output).await?;
}

// Path B: submit normal native requests wrapped with custom_id; upload is automatic.
let batch = client.batches().create(&CreateBatchFromRequests::new(requests), None).await?;
let done = client.batches().wait(&batch.id, BatchWaitOptions::default()).await?;
let mut results = client.batches().results(&done.id).await?;
while let Some(record) = results.next().await? {
    println!("{}", record.custom_id);
}
# Ok(()) }
```

The complete, compile-checked example is [`examples/batches.rs`](examples/batches.rs).
Configure `BASAL_BATCH_TOKEN` and optionally `BASAL_BASE_URL`, then run
`cargo run --example batches -p basal-client` (simplified path), or append `-- input.jsonl` for path A.
It requires an enabled server and an explicit background execution profile for the selected model.

`wait` defaults to a 600-second whole-operation timeout and a 5-second polling interval. It returns
any terminal state (`completed`, `failed`, `expired`, `cancelled`); inspect `errors` and counters.
HTTP failures propagate without retry. `BatchOperationError::WaitTimeout` retains the id and last
observed batch. Timeout or dropping the future stops local I/O/polling; server work continues.

`results` requires a terminal state, otherwise `BatchOperationError::NotReady` carries the current
batch. Its async `next()` cursor reads output then error files, including partial expired/cancelled
results, retaining native bodies and caller ids. Use `cursor.batch()` for batch validation errors.
Rows are not input ordered. Each file is buffered within the client's byte cap (default 8 MiB),
then parsed one line at a time; this is not an unbounded file stream. Use `results_with_options`
and `BatchResultsOptions { max_bytes: Some(...) }` to explicitly change the per-file cap. Malformed
UTF-8/JSON/result envelopes stop iteration with a contextual `ResultLine` error. No file means an
empty cursor, including failed validation batches whose errors are on the batch itself.

The requests path buffers JSONL (up to 50,000 records/200 MB), plus a multipart copy during upload.
The server can impose stricter limits. It rejects empty lists, duplicate/invalid ids and unsupported
record envelopes before upload. The SDK generates an idempotency key when none was supplied.
If creation fails after upload, `BatchOperationError::Submission` provides `input_file_id`,
`idempotency_key` and the original error. Resume using `CreateBatchRequest::new(input_file_id)` with
the same metadata, retention and key; do not resubmit the requests path, which uploads a different file.
It never retries or deletes a possibly referenced file. Dropping the entire create future cannot
return recovery context; use the explicit Files path when the application must persist each stage.

`files().content_with_limit(id, bytes)` overrides only successful download bounds; HTTP errors
keep the ordinary client-wide byte cap. Resource `list`, `retrieve`, `delete` and `cancel` delegate
to their flat equivalents and share the same authentication, prefix, TLS and timeout policy.

### Existing flat methods

Use a server with batches enabled (basal-rs 0.1.12+) and its Batch bearer token:

```rust,no_run
use basal_client::{Client, ClientOptions, CreateBatchRequest, FileUploadOptions};
# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let token = std::env::var("BASAL_BATCH_TOKEN")?;
let client = Client::with_options(ClientOptions {
    api_key: Some(&token), ..Default::default()
})?;
let input = std::fs::read("input.jsonl")?;
let file = client.upload_file("input.jsonl", &input, FileUploadOptions::default()).await?;
let batch = client.create_batch(&CreateBatchRequest::new(file.id), Some("import-1")).await?;
let state = client.get_batch(&batch.id).await?;
// Call get_batch later at your chosen interval; the SDK does not poll or retry automatically.
if let Some(output_id) = state.output_file_id {
    std::fs::write("results.jsonl", client.download_file(&output_id).await?)?;
}
# Ok(()) }
```

`list_files`, `get_file`, `delete_file`, `list_batches`, `get_batch`, `cancel_batch`
and `batch_health` cover the remaining routes. `BatchListOptions` exposes cursor/limit;
`FileUploadOptions` and `BatchExpiresAfter` expose retention. `BatchInputLine` wraps
a native `DecisionRequest` for one JSONL line, and `BatchResultLine` decodes a result.
Only `/v1/basal` records and a `24h` completion window are supported by the current server.
Inspect output and error file ids, including after expiry or cancellation.

Uploads accept bytes owned by the caller (at most 200 MB) and allocate a multipart copy;
they do not read local files or fetch URLs. Downloads return exact bytes with the same
timeout and `max_response_bytes` bound as other requests (default 8 MiB). Increase that
explicit cap to download larger artifacts. The whole response is buffered within that cap.
No redirects or automatic retries; `Error::Http` retains status/body/Retry-After.
Batch statuses are strings and unknown response fields are retained for compatibility.
See [the server's Batch contract](../../docs/BATCHES.md) for authentication, input format
and the best-effort 24-hour window with online traffic priority.

```sh
cargo add basal-client
cargo add tokio --features macros,rt-multi-thread
```

```rust,no_run
use basal_client::{choice, noul, questions, Client};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let department = choice(
        "Do którego działu skierować zgłoszenie?",
        &[("returns", "Zwroty"), ("support", "Wsparcie techniczne")],
    )?;
    let urgent = noul("Czy sprawa jest pilna?")?;
    let questions = questions! {
        department: &department,
        second_department: &department,
        urgent: &urgent,
    }?;
    let response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", &questions).await?;
    let answers = response.answers()?;
    println!("Dział: {}, P(tak): {}", answers.department.choice, answers.urgent.noul);
    println!("Drugie pytanie: {}", answers.second_department.choice);
    Ok(())
}
```

For choice/score, set `question.criteria` to an ordered `serde_json::Value` object/array. Question and request
`extra` maps accept native extension fields (`evidence`, `costs`, `facts` etc.). Do not duplicate explicit fields
in `extra`. JSON object order is retained through serde_json's `preserve_order` and indexmap. The `float_roundtrip`
feature preserves binary64 values when parsing wire probabilities, without extra dependencies or
`arbitrary_precision` feature unification.
Add `serde_json` directly (`cargo add serde_json --features preserve_order`) to construct structured JSON
with its `json!` macro.

`models()` reads `/v1/models`. Use `Client::with_options(ClientOptions { ... })`
for a bearer token or a different timeout. The timeout (default 120 seconds) includes response-body reading.
Dropping the request future cancels client-side I/O. Clone/reuse the client to share its connection pool.
Use the server root (possibly with a proxy prefix), without `/v1`. HTTPS is required except for loopback
HTTP, suitable for a local server or SSH tunnel. URL credentials/query/fragment are rejected. DNS names
must be ASCII; use punycode for IDN. The client ignores environment proxies, uses a per-client ring
crypto provider, and verifies TLS with platform roots. It does not enable HTTP/2 or install a process-global
crypto provider. ring includes a native build step; this SDK does not depend on AWS-LC or CMake.
`max_response_bytes` defaults to 8 MiB and limits the actual stream, including error bodies;
oversized responses return `Error::ResponseTooLarge`, deadlines return `Error::Timeout`.

`Error::Http` provides `status`, `body` and raw `retry_after`. `Error::Transport`, `Error::Encode` and
`Error::Protocol` preserve their causes. No automatic retries or redirects. Responses deserialize into
native DTOs; extra JSON fields are retained. HTTP 422 and overload 529 remain distinct.
`Display` and `Debug` omit HTTP bodies and nested transport details; access them explicitly when needed.

Native choice/score accepts 2–255 options for format-2 models (older model limits are checked by the server); multi uses independent probabilities and `set_confidence`.
Evidence offsets count Unicode characters, not UTF-8 bytes. The SDK does not execute selected actions.

Factories return distinct `QuestionSpec<...>` types. Default factories take only their required
arguments; `choice_with_options`, `noul_with_options`, `score_with_options`, `multi_with_options`,
`act_with_options` accept type-specific options.

Starting with **0.2.0**, `questions!` infers the answer kind for each named field. Field names become
request keys. A definition or its clone can appear under several keys; each field reads its own
answer. Expressions are evaluated once. `response.answers()?` returns borrowed views with required
fields, so `answers.department.choice` is `&str` and `answers.urgent.noul` is `f64`, without optional
fields or casts. Choice keys remain strings in Rust; the SDK validates them against the requested
criteria at runtime. `?` propagates a validation error, rather than unwrapping a nullable field.

For keys loaded dynamically, use the ordered map builder:

```rust,ignore
let questions = Questions::new()
    .with("department", &department)?
    .with("second_department", &department)?;
let response = client.decide(state, &questions).await?;
println!("{}", response.choice("department")?.choice);
println!("{}", response.choice("second_department")?.choice);
```

Import `basal_client::Questions` for this variant. `response.answer("department")?` returns the
`AnswerResult` enum; match its variants for heterogeneous dynamic keys. The `choice`, `noul`,
`score`, `multi` and `act` accessors return checked views by key and reject a different answer kind.
`response.answers()?` on a dynamic map returns an ordered map of these enum variants. Unknown keys
are errors, including extra answers that were not requested. Duplicate request keys are rejected.

**Migrating from 0.1.0:** replace `response.answer(&question)?` with the named fields from
`questions!`, or a keyed accessor such as `response.choice("department")?`. Question object identity
no longer determines which answer is returned. This is a breaking API change; other SDK package
versions are independent.

Views borrow the response without copying probability maps. Use `decide_with_options` for
model/facts, `response.raw()` for native DTOs, and `decide_request(&request)` for the full raw request
(Rust has no method overloading).


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

```rust,no_run
use basal_client::{tool_with_options, ToolOptions, ToolExtra, DecisionOptions, Language, MediaInput, questions};
let definitions = [serde_json::json!({"name":"inspect","description":"Inspect material","inputSchema":{"type":"object"}})];
let routing = tool_with_options("Choose a tool.", &definitions, ToolOptions { extra: Some(&[ToolExtra::None]), rank: Some(2), description_chars: Some(120), ..Default::default() })?;
let questions = questions! { route: &routing }?;
let media = [MediaInput::image(&std::fs::read("photo.jpg")?)];
let response = client.decide_with_options("Inspect the supplied material.", &questions, DecisionOptions {
    model: Some("basal-omni-1.0-mini"), lang: Some(Language::En), media: Some(&media), ..Default::default()
}).await?;
println!("{:?}", response.answers()?.route.tool);
```

Rust 0.3 adds a `Tool` question/answer variant and fields to request options and
metadata. When constructing option structs use `..Default::default()`; multi
threshold is now `Option<f64>`. `MultiOptions` borrows constraint keys. Existing
request factories and raw response access remain available.

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
