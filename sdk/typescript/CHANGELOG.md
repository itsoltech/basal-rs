# Changelog

## Unreleased

## 0.6.0 — 2026-10-21

- **Breaking:** remove API-key configuration and bearer headers from client initialization.
  Remove `api_key`/`apiKey` arguments. Batch/Files now require server 0.1.13+ without authentication;
  all callers share the server-configured storage namespace. Existing create/wait/results flows remain.

## 0.5.0 — 2026-10-20

- Add `client.files` (`create`, `retrieve`, `list`, `delete`, `content`) and `client.batches` (`create`, `retrieve`,
  `list`, `cancel`, `wait`, `results`) with OpenAI resource names; the flat methods are unchanged.
- `batches.create` takes `input_file_id` or `requests`. With `requests` the SDK checks the records (nonempty, unique
  `custom_id`, envelope, 50 000 records, 200 000 000 bytes) before building JSONL in memory, uploads it and creates the
  batch with a caller or generated idempotency key. A failure after the upload raises `BasalBatchSubmissionError` with
  the file id and key for resuming; nothing is retried or deleted.
- `batches.wait` polls until a terminal status (default timeout 600 s, interval 5 s), bounds in-flight retrieves by its
  deadline and raises `BasalBatchTimeoutError` with the last batch without cancelling it.
- `batches.results` yields records of the output and error files of a terminal batch (`BasalBatchNotReadyError`
  otherwise). Each file is buffered up to `maxBytes` and parsed one line at a time; malformed lines raise
  `BasalBatchResultError` with the file id and line number.
- Export `MAX_BATCH_REQUESTS`, `MAX_BATCH_INPUT_BYTES` and `TERMINAL_BATCH_STATUSES`.

## 0.4.0 — 2026-10-20

- Add the Batch and Files API: `uploadFile` (multipart, `purpose=batch`, optional `expires_after`),
  `listFiles`, `getFile`, `deleteFile`, `downloadFile` (raw bytes), `createBatch` (metadata,
  `output_expires_after`, `Idempotency-Key`), `listBatches`, `getBatch` and `cancelBatch`.
- Typed wire types for files, batches, lists, batch errors and result lines; `batchInput` builds
  JSONL input locally and `parseBatchResults` parses downloaded result bytes.
- Uploads accept caller-owned bytes, Blob or `ReadableStream`; they never fetch URLs or read paths.
  A streamed source is cancelled on abort, timeout or an early server answer.
- Downloads are bounded by `maxResponseBytes` or a per-call `maxBytes`; uploads and downloads take a per-call `timeoutMs`.
- Resource ids and query values are escaped; `.`/`..` ids, multiline or non-ASCII idempotency keys and
  filenames with control characters or backslashes are refused before sending.
- Error classes moved to an internal module; exports are unchanged.

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
