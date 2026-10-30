# Changelog

## Unreleased

### 0.7.0 (local, unpublished)

- Restore optional `apiKey` on both `BasalClient` constructors. When set, every inference and
  Batch/Files request sends `Authorization: Bearer <key>`; the key is opaque, explicit-only (no
  environment default) and kept out of messages. Without it no header is sent, as in 0.6.0.
  401/403 `BasalHttpException` messages name the missing/invalid key or missing scope. Requires a
  server with `auth.enabled` and a key from `basal keys create`. `apiKey` is the last optional
  constructor parameter (after `maxResponseBytes`), so existing positional calls still compile;
  rebuild against the changed assembly. Pass it by name: `apiKey: key`.
- Add Think/ThinkText/EarlyExit init properties to DecisionOptions and low-level requests.
  Preserve the positional constructor and five-value Deconstruct; omit unset options.
- Add optional Lane/Truncated/ThinkQuestions usage and Layout/CalibrationWarning/Think/
  EarlyExit/EarlyExitDefault discovery. Raw nested JSON and answer Extra think/meta retain
  unknown keys. Policies are open strings; required answer validation is unchanged.
- **Behavior change:** these Usage/ModelMetadata keys, including the always-emitted
  `early_exit`, now bind to the typed properties and no longer appear in `Extra`. Code such as
  `model.Extra["early_exit"]` still compiles but throws `KeyNotFoundException`; read the typed properties.
- Existing call forms remain; convenience result records use Raw.Answers[id].Extra for
  additions. Rebuild against the changed assembly version; no general source/binary
  compatibility or new runtime parity is claimed. No dependencies changed.

## 0.6.0 — 2026-10-21

- **Breaking:** remove API-key configuration and bearer headers from client initialization.
  Remove `api_key`/`apiKey` arguments. Batch/Files now require server 0.1.13+ without authentication;
  all callers share the server-configured storage namespace. Existing create/wait/results flows remain.

## 0.5.0 — 2026-10-20

- Add `client.Files` (`CreateAsync`, `RetrieveAsync`, `ListAsync`, `DeleteAsync`, `ContentAsync`) and
  `client.Batches` (`CreateAsync`, `RetrieveAsync`, `ListAsync`, `CancelAsync`) resources delegating to the
  existing flat methods, which keep their signatures and behavior.
- Add `Batches.CreateAsync(IEnumerable<BatchInputLine>, CreateBatchOptions?)`: validates records, bounds and the
  idempotency key before HTTP, builds the JSONL in memory, uploads it and creates the batch. A key is generated when
  omitted. Failures after the upload raise `BatchSubmissionException` or `BatchSubmissionCanceledException`
  (`IBatchSubmissionFailure`: file ID, key, request, original cause); no retry or file deletion.
- Add `Batches.WaitAsync` (default timeout 600 s, poll 5 s, deadline including HTTP, `BatchWaitTimeoutException`
  with the last batch, no server cancellation) and `Batches.ResultsAsync` (`IAsyncEnumerable<BatchResultLine>` over
  output then error file of a terminal batch, per-file in-memory cap, `BatchNotReadyException`,
  `BatchResultFormatException` with file ID and line).
- The C# example compiles both batch flows (`dotnet run -- batch`).

## 0.4.0 — 2026-10-20

- Add the Batch/Files API: `UploadFileAsync` (multipart, purpose `batch`, optional expiry sent before the file part,
  from caller-owned bytes or a caller-owned stream), `ListFilesAsync`, `GetFileAsync`, `DeleteFileAsync`,
  `DownloadFileContentAsync`, `CreateBatchAsync` (metadata, 24h window, result expiry, `Idempotency-Key`),
  `ListBatchesAsync`, `GetBatchAsync` and `CancelBatchAsync`.
- File content downloads return raw bytes without JSON parsing. The in-memory overload is bounded by
  `maxResponseBytes` or an explicit `maxBytes`; the stream overload copies into a caller-owned stream within a
  required byte limit.
- Typed `FileObject`, `Batch`, `ListPage<T>`, `BatchInputLine` and `BatchResultLine` DTOs keep unknown fields in
  `Extra`; batch status stays a string with `BatchStatus` constants.
- Resource IDs and query values are percent-escaped; dot-segment IDs, header-unsafe idempotency keys and
  unframeable file names are refused before HTTP. No automatic retries or polling.

## 0.3.0 — 2026-10-19

- Add experimental Basal 2.1 classify, verify, compress and extract tasks with typed results.
- Preserve nullable task automation, source offsets in Unicode code points and nested JSON extraction values.
- Task answers no longer require the primitive-only probabilities field.
- Expose tool `description_chars` in typed factories and raw request types; omission keeps the server default and zero disables truncation.
- Correct outdated documentation about task support.
- Raw `Question.MaxError` is now `JsonNode?`, allowing scalar or list targets for classify/verify. Scalar initializers remain valid; read scalars with `GetValue<double>()`.

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
