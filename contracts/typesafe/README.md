# TypeSafe System One

`openapi.json` is the upstream HTTP contract, fetched on 2026-10-19 from
<https://api.typesafe.ai/openapi.json>.

SHA-256: `ee599512cbea3a198b34af3f60addcd72b381d3521196f93c70ae83185c057c3`.
Upstream still reports `info.version: 0.2.0`; the version alone does not identify
the schema. This revision adds `assets`, multipart uploads, the images header,
and Box questions/answers.

This is a reference contract, not a declaration that every basal-rs model
implements every TypeSafe feature. `/v1/systemone` adapts positioned image
assets to media-enabled Omni backends. Unsupported model/runtime features,
including Box, return HTTP 422 with `error: "unsupported"`.

The adapter supports PNG/JPEG assets in JSON/base64 or multipart, and gzip/zstd
request compression. Its current image budget is eight distinct assets and
32 MiB of expanded source image bytes and text. Native task questions with
assets and `facts: "auto"` with assets are unsupported. The original references
remain in score legends. The native `/v1/basal` contract and the SDKs are separate.

Request behavior and examples: [docs/SYSTEM_ONE.md](../../docs/SYSTEM_ONE.md).
The detailed upstream image contract is at <https://docs.typesafe.ai/images>.
