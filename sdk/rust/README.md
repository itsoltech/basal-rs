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
use basal_client::{choice, noul, Client, Questions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("http://127.0.0.1:8000")?;
    let department = choice(
        "Do którego działu skierować zgłoszenie?",
        &[("returns", "Zwroty"), ("support", "Wsparcie techniczne")],
    )?;
    let urgent = noul("Czy sprawa jest pilna?")?;
    let questions = Questions::new().with("department", &department)?.with("urgent", &urgent)?;
    let response = client.decide("Klient prosi o zwrot pieniędzy za uszkodzony telefon.", &questions).await?;
    println!("Dział: {}, P(tak): {}", response.answer(&department)?.choice, response.answer(&urgent)?.noul);
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
`act_with_options` accept type-specific options. The ordered
`Questions` builder preserves those question identities. `response.answer(&question)?` returns a
checked view with required fields; `?` propagates an invalid/unrelated question error and does not
unwrap nullable fields. Views borrow the response without copying probability maps. A clone of a
question retains its identity. Use `decide_with_options` for model/facts, `response.raw()` for the
native DTOs, and `decide_request(&request)` for the full raw request (Rust has no method overloading).


[Full documentation and publication setup](https://github.com/itsoltech/basal-rs/blob/main/docs/SDK.md).
