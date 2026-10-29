# Provenance of the vendored tokenizers 0.23.2

## Release

- Crate: `tokenizers 0.23.2` from crates.io,
  <https://static.crates.io/crates/tokenizers/tokenizers-0.23.2.crate>.
- Archive SHA-256:
  `7afbf6e88718afcc138bad01d6ccc3051dbbc3b2ce9793d8b8a3aeb610969cfc`.
  This is the checksum that `Cargo.lock` recorded for the registry package
  before the local patch.
- `.cargo_vcs_info.json` in the archive: commit
  `88a4498ad4ea1a9487b0a9b0ff881383fd5a06a3`, path `tokenizers`
  (file SHA-256 `937e34c64860adc61c1b9b10f2bc1a06fb84289d26d705a6f7c229aeaaf90f39`).
- Upstream `Cargo.lock` in the archive, not copied: SHA-256
  `dad088d1ed601ee25b2a27d8451d0d409ed344440c7b5ec44034c68f3d921973`.
- License: Apache-2.0, retained unchanged in `LICENSE`.

## Copied files

The 73 files under `src/`, `LICENSE`, `README.md`, the normalized release
`Cargo.toml` and the original `Cargo.toml.orig` were copied from the release.
`UPSTREAM-SHA256SUMS` lists all 77 copied files before any change. The copy
was compared with both the cargo registry source and a separately extracted,
checksum-verified archive.

Omitted from the archive: `benches/` (ten files), `tests/` (nine files),
`Cargo.lock`, `CHANGELOG.md`, `Makefile`, `README.tpl` and
`.cargo_vcs_info.json`. The manifest omits the 8 `[[test]]` and 9 `[[bench]]`
targets, the 5 dev dependencies (`assert_approx_eq`, `criterion`, `tempfile`,
`tracing`, `tracing-subscriber`) and the `profiling`/`release` profiles, which
Cargo ignores outside the workspace root. Its first line records this. Runtime
dependencies, features, edition and library target are unchanged. Inline
`#[cfg(test)]` modules remain unchanged and are not run by this project.

## Local change

`merge-scratch.patch` reproduces the seven modified files from their pinned
preimages in `UPSTREAM-SHA256SUMS` and creates the eighth. Apply it with
`patch -p1` in a directory holding those preimages:

- `Cargo.toml`: the omissions above.
- `src/models/bpe/model.rs`: `BpeBuilder::build` and the new
  `build_with_merge_storage` share one build body. The legacy build still
  allocates `vec![0; max_len]`; the new one takes the buffer from a
  `MergeStorage`, requires the slice to have the requested length and uses
  that one slice for all merge writes and reads.
- `src/models/bpe/serialization.rs`: the BPE visitor hands its configured
  builder to a caller-supplied step; `BpeMergeSeed` and `BpeTrial` reuse it.
- `src/models/mod.rs`: the tagged/legacy helpers move to module scope;
  `ModelMergeSeed` follows `ModelWrapper::deserialize`.
- `src/tokenizer/serialization.rs`: the tokenizer visitor reads `model`
  through an explicit strategy; `TokenizerMergeSeed` uses the merge seed.
  After a refusal it skips the remaining entries syntactically; if that skip
  fails, the refusal is kept in the caller's per-call slot.
- `src/tokenizer/mod.rs`: `Tokenizer::from_str_with_merge_storage`.
- `src/utils/mod.rs` and new `src/utils/merge_storage.rs`: `MergeStorage`
  and `MergeBuildError`.

`SHA256SUMS` covers every file in this directory except itself.
