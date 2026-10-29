# Changelog

## Unreleased

### 0.8.0 (local, unpublished)

- Restore optional `ClientOptions::api_key`. When set, every inference and Batch/Files request
  sends `Authorization: Bearer <key>` (a sensitive header value); the key is opaque, explicit-only
  (no environment default) and absent from `Debug`/`Display`. Without it no header is sent, as in
  0.7.0. `Error::Http` Display names the missing/invalid key (401) or missing scope (403). Requires
  a server with `auth.enabled` and a key from `basal keys create`. The new public field can break
  exhaustive `ClientOptions` struct literals; use `..Default::default()`.
- Add ergonomic ThinkMode and DecisionOptions think/think_text/early_exit; omit unset
  options. DecisionRequest.extra retains raw fields, including explicit null early_exit.
- Add optional Usage lane/truncated/think_questions and ModelMetadata layout/
  calibration_warning/think/early_exit/early_exit_default. Policies are open strings;
  raw nested JSON and answer extra think/meta retain unknown keys. Required validation remains.
- **Behavior change:** these Usage/ModelMetadata keys, including the always-emitted
  `early_exit`, now bind to the typed fields and no longer appear in `extra`. Lookups such as
  `model.extra.get("early_exit")` still compile but return `None`; read the typed fields.
- New public Usage/ModelMetadata/DecisionOptions fields can break exhaustive struct literals.
  Existing constructors/call forms remain; Option does not guarantee source compatibility.
  No stable Rust ABI or new runtime parity is claimed. No dependencies changed.

## 0.7.0 — 2026-10-21

- **Breaking:** remove API-key configuration and bearer headers from client initialization.
  Remove `api_key`/`apiKey` arguments. Batch/Files now require server 0.1.13+ without authentication;
  all callers share the server-configured storage namespace. Existing create/wait/results flows remain.

## 0.6.0 — 2026-10-20

- Add `files()` and `batches()` resource facades, retaining all existing flat methods.
- Accept uploaded files or `CreateBatchFromRequests` in `batches().create`; the latter prepares
  and uploads bounded JSONL and exposes file/key recovery context on post-upload creation failure.
- Add explicit bounded `wait`, terminal-state `results` cursor (output and error files),
  contextual result errors and per-download success byte limits. No automatic retries or cancellation.

## 0.5.0 — 2026-10-20

- Add typed Batch and Files API: upload JSONL, create/list/retrieve/cancel batches,
  list/retrieve/delete files and download bounded raw result bytes.
- Reuse bearer authentication, proxy prefixes, response bounds and cancellation;
  support owner-scoped idempotency keys and file/output retention options.
- Preserve unknown response fields and future batch statuses. No automatic retries or polling.

## 0.4.0 — 2026-10-19

- Add experimental Basal 2.1 classify, verify, compress and extract tasks with typed results.
- Preserve nullable task automation, source offsets in Unicode code points and nested JSON extraction values.
- Task answers no longer require the primitive-only probabilities field.
- Expose tool `description_chars` in typed factories and raw request types; omission keeps the server default and zero disables truncation.
- Correct outdated documentation about task support.
- `ToolOptions` gains an optional field; use `..Default::default()` in struct literals.

## 0.3.0 — 2026-10-17

- Align typed APIs with Basal 2.0 and Omni: up to 255 choice/score/act outcomes,
  tool routing/ranking/multi/deferral, multi objectives and constraints.
- Add inline media helpers and lang/detail request options, calibration `text`,
  deferral metadata, media usage and model capability discovery.
- Multi result threshold is now optional; use objective and expected score for
  exact/IoU/F1 results. Older model capabilities remain server-validated.
- Registry publication is separate from this source update.

## 0.2.0 — 2026-10-16

- **Breaking:** `response.answer` now takes the request key and returns an `AnswerResult` enum.
  Use `response.choice("department")?` for a checked choice view instead of `response.answer(&question)?`.
- Reuse a question definition or its clone under several keys; each answer is resolved by its key.
  Duplicate keys remain invalid.
- Add `questions!` with inferred named response fields: `response.answers()?.department.choice`.
  Dynamic maps also expose keyed accessors for all five answer kinds.
- Preserve response validation, criterion order and borrowed probability maps. No new dependencies.

## 0.1.0 — 2026-10-15

- Native Basal HTTP client with typed questions and responses.
- Independent package releases; see the SDK release documentation.
