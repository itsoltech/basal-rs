# Rust SDK keyed answers — 2026-10-16

Status: implementation and local verification complete; release candidate 0.2.0, not published.
Base: 09810f4305611cff94e110a1f93a241d95c8e0ec.
Scope: Rust SDK API and documentation. No inference engine or numeric computations changed.

## Problem and behavior

Version 0.1.0 resolved answers by question object identity and rejected reuse of one definition
under two request keys. Version 0.2.0 permits reuse, rejects duplicate keys, and resolves each
answer by the request key. `questions!` infers a named result shape; a runtime-keyed `Questions`
map returns checked enum variants or kind-specific keyed views.

## Evidence

- `cargo fmt --all --check` — passed.
- `cargo clippy --locked --workspace --release --all-targets -- -D warnings` — passed on Linux without GPU.
- `cargo clippy --locked -p basal-client --all-targets -- -D warnings` — passed.
- `RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --release --no-deps --document-private-items` — passed.
- `cargo deny --locked --all-features check --hide-inclusion-graph` — advisories, bans, licenses and sources passed;
  existing duplicate-version warnings remain. No audit exceptions added or changed.
- `bash tools/ci/check-paste-exception.sh` — approved graph and expiry passed.
- `python3 tools/sdk/check-versions.py` — Python/TypeScript/C# 0.1.0, Rust 0.2.0.
- `cargo package --locked -p basal-client --allow-dirty --offline` — packaging and unpacked source verification passed.
- Lockfile comparison excluding the SDK's own package entry — identical dependency graph.
- Scratch consumer compiled under Rust editions 2021 and 2024. Both localized landing snippets
  were exported from the content/translation modules and compiled against the local 0.2.0 candidate.
- Scratch HTTP diagnostic exercised all five answer kinds and identical/cloned definitions under distinct keys.
  Different fixture choices were returned for the two reused keys. Required fields remained non-nullable.
  Unknown keys (including unrequested server extras), wrong-kind access, missing answers, invalid choice
  and missing required confidence were rejected. Request field order and raw identifier mapping were checked.
  Probability views borrow the raw response's maps. See runtime.json.
- Scratch compiler diagnostics reject the old identity API, wrong/unknown named fields, duplicate named fields
  and views that outlive their response. Multiple macro invocations, temporary/borrowed expressions,
  internal-name collisions and reuse without consuming the original definition compile. See compile-cases.json.

Diagnostics lived outside the repository; no test framework, test files or CI gates were added.

## Review and limits

Inline review followed rust-implementation, rust-review and itsol-self-review. Confirmed request-key mapping,
immutable definition ownership, response lifetimes, validation before returning from Client, no unsafe,
no new dependencies, no HTTP policy changes and explicit breaking-version migration guidance.
The macro uses a scoped naming-lint allowance because caller field identifiers also serve as generic type
parameters. Its borrowed expressions are evaluated once and retained for the duration of request assembly.
The generated schema's Clone implementation does not require answer marker types to implement Clone.

No real Basal inference was executed: HTTP responses were synthetic. No model quality or performance claim
is made. CUDA/Metal runtime behavior and registry publication were not verified in this change.
The landing page must not deploy its new Rust example before basal-client 0.2.0 is published.
