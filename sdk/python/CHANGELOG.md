# Changelog

## Unreleased

### 0.7.0 (local, unpublished)

- Restore optional `api_key` on `BasalClient` and `AsyncBasalClient`. When set, every inference
  and Batch/Files request sends `Authorization: Bearer <key>`; the key is opaque, explicit-only
  (no environment default) and kept out of messages. Without it no header is sent, as in 0.6.0.
  401/403 `BasalHTTPError` messages name the missing/invalid key or missing scope. Requires a
  server with `auth.enabled` and a key from `basal keys create`.
- Add think/think_text/early_exit to raw requests and every sync/async ergonomic overload.
  Omitted options remain absent; raw JSON can express explicit null early_exit.
- Add optional lane/truncated/think_questions usage and layout/calibration_warning/think/
  early_exit_default discovery. Policies are open strings; nested JSON and raw answer
  think/meta retain unknown keys. Required answer validation is unchanged.
- Existing call forms remain. Discovery early_exit is now annotated as a string inventory,
  which may require type-checking fixes for callers using non-string values. No general
  source/binary compatibility or new runtime parity is claimed; convenience dataclasses
  use response.raw for the additions. No dependencies changed.

## 0.6.0 — 2026-10-21

- **Breaking:** remove API-key configuration and bearer headers from client initialization.
  Remove `api_key`/`apiKey` arguments. Batch/Files now require server 0.1.13+ without authentication;
  all callers share the server-configured storage namespace. Existing create/wait/results flows remain.

## 0.5.0 — 2026-10-20

- Add resource namespaces `client.files` (`create`, `retrieve`, `list`, `delete`, `content`) and
  `client.batches` (`create`, `retrieve`, `list`, `cancel`, `wait`, `results`) to `BasalClient` and
  `AsyncBasalClient`. Flat methods are unchanged.
- `batches.create` takes `input_file_id`, or `requests` (`custom_id` and native `body`): the SDK
  validates them, builds the JSONL in memory (at most 50000 records and 200000000 bytes), uploads it and
  creates the batch with a caller or generated `Idempotency-Key`. A failure after the upload raises
  `BasalBatchSubmissionError` with the file id and key for resuming; nothing is retried or deleted.
  Cancellation or `KeyboardInterrupt` after the upload propagates unchanged and
  `BasalBatchSubmissionError.find(error)` returns the same context. `body.stream == true` is rejected.
- `batches.wait` polls (default timeout 600 s, interval 5 s) until a terminal state;
  `BasalBatchTimeoutError` keeps the last observed batch and never cancels the job. One deadline covers
  requests and sleeps; the sync client polls on a private event loop in a joined worker thread with its
  own connections.
- `batches.results` iterates output, then error records of a terminal batch (`BasalBatchNotReadyError`
  otherwise). Each file is buffered within `max_bytes` and parsed line by line; malformed lines and
  envelopes (same rules as the Rust SDK) raise `BasalBatchResultError` with the line number.
- Exceptions now live in `basal_client.errors`; imports from `basal_client` and `basal_client.client`
  keep working.

## 0.4.0 — 2026-10-20

- Add the Batch and Files API to `BasalClient` and `AsyncBasalClient`: `upload_file`, `list_files`,
  `get_file`, `delete_file`, `download_file`, `create_batch`, `list_batches`, `get_batch` and
  `cancel_batch`. Requires basal-rs 0.1.12+ with batches enabled.
- Uploads stream caller-owned bytes, binary file objects or (async) iterables as multipart with
  `purpose` and `expires_after[...]` before the file. The SDK never reads paths or fetches URLs.
- `download_file` returns raw bytes without JSON parsing, bounded by `max_bytes` or `max_response_bytes`.
- Batch creation supports metadata, output file expiry and a validated `Idempotency-Key`. Resource ids
  are escaped as one path segment; header values that could split a request are rejected.
- Typed dictionaries for files, batches, lists, request counts, validation errors and JSONL lines keep
  fields added by newer servers. No automatic retries or polling. The online API is unchanged.

## 0.3.0 — 2026-10-19

- Add experimental Basal 2.1 classify, verify, compress and extract tasks with typed results.
- Preserve nullable task automation, source offsets in Unicode code points and nested JSON extraction values.
- Task answers no longer require the primitive-only probabilities field.
- Expose tool `description_chars` in typed factories and raw request types; omission keeps the server default and zero disables truncation.
- Correct outdated documentation about task support.

## 0.2.0 — 2026-10-17

- Align typed APIs with Basal 2.0 and Omni: up to 255 choice/score/act outcomes,
  tool routing/ranking/multi/deferral, multi objectives and constraints.
- Add inline media helpers and lang/detail request options, calibration `text`,
  deferral metadata, media usage and model capability discovery.
- Multi result threshold is now optional; use objective and expected score for
  exact/IoU/F1 results. Older model capabilities remain server-validated.
- Registry publication is separate from this source update.

## 0.1.0 — 2026-10-15

- Native Basal HTTP client with typed questions and responses.
- Independent package releases; see the SDK release documentation.
