# basal-client (Rust)

Async Rust SDK for the native Basal API (`POST /v1/basal`), using Hyper HTTP/1 with Rustls and the platform certificate verifier.
No dependency on the inference engine or GPU libraries. Rust 1.95+; a Tokio runtime is required.

This package is versioned independently. Release tag: `sdk-rust-vX.Y.Z`.
[Changelog](https://github.com/itsoltech/basal-rs/blob/main/sdk/rust/CHANGELOG.md).

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

Native choice/score accepts 2–10 options; multi uses independent probabilities and `set_confidence`.
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
