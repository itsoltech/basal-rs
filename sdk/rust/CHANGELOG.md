# Changelog

## Unreleased

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
