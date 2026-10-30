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
Cargo ignores outside the workspace root. Its first line records this. The
initial copy preserves runtime dependencies, features, edition and library
target; the added direct dependencies are listed below. Inline
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

`merge-table.patch` then applies, also with `patch -p1`, to the result of
`merge-scratch.patch` and to the pinned upstream `src/models/bpe/word.rs`:

- `Cargo.toml`: direct dependencies `hashbrown = "=0.16.1"` (no default
  features, `allocator-api2`) and `allocator-api2 = "=0.2.21"` (no default
  features, `alloc`). Both versions were already locked; no version changes.
- `src/utils/merge_storage.rs`: `TableAccount`, `TableRefusal`,
  `MergeTableAccount` and the provided `MergeStorage::merge_table` (default
  `None`). The `MergeBuildError::Storage` message is unchanged.
- `src/models/bpe/model.rs`: a crate-private, non-`Clone` charged allocator
  and `MergeTable`. Without an account the table is collected exactly as
  before; with one it is built in a hashbrown map whose every block is
  admitted on the account before allocation and released after it is freed.
  Clones share the charged table through one `Arc`.
- `src/models/bpe/word.rs`: `Word::merge_all` takes a lookup function instead
  of the map type.

`decoder-regex.patch` then applies with `patch -p1` to that result and the
pinned upstream preimages of the four newly modified source files:

- `src/utils/onig.rs`: admitted regexes share an `Arc<Regex>`; legacy clones
  still recompile, and the existing `Debug` text is retained.
- `src/utils/merge_storage.rs`: a provided `MergeStorage::decoder_regex`
  returns no account by default, with crate-private account/error adapters.
- `src/normalizers/replace.rs`: admitted construction for `Replace` uses that
  account, and cloning shares the compiled program and its charge.
- `src/decoders/mod.rs` and `src/decoders/sequence.rs`: the decoder seed reads
  `Replace` and nested sequences with the account and preserves legacy tagged
  and untagged parsing. A refusal ends the read rather than trying an
  uncharged variant.
- `src/tokenizer/serialization.rs`: the decoder strategy uses that seed and
  the existing first-refusal channel.

Admitted clones do not recompile, so they do not observe compile-time global
changes made after construction, unlike legacy clones. The retained native
allocation boundary and the separate `onig`/`onig_sys` patches are documented
in the parent `vendor/README.md` and those crates' provenance files.

`reverse-vocab-table.patch` then applies with `patch -p1` to the result of the
three earlier patches. It changes no further upstream file:

- `src/utils/merge_storage.rs`: a provided `MergeStorage::reverse_vocab_table`
  returns no account by default. Existing signatures and error messages are
  unchanged; only documentation is generalized to every charged table.
- `src/models/bpe/model.rs`: the BPE reverse vocabulary becomes a
  crate-private `VocabR` enum, global as before or charged in a hashbrown map
  with the same key, value and hasher, held in one `Arc`. Without an account it
  is collected exactly as before. With one, the same single
  `RandomState::new()`, a fallible reservation of the vocabulary length and
  the same per-entry inserts build it; its single block is admitted on the
  account through the existing non-`Clone` table allocator, whose code is
  unchanged, before any token is copied. Lookup, indexing, map equality and
  trainer collection call sites are textually unchanged; clones share the
  charged table.
- `src/models/mod.rs`: the ordered vocabulary serializer takes either reverse
  vocabulary through a private trait; the legacy map uses the same calls.

`SHA256SUMS` covers every file in this directory except itself.

The eleven modified upstream Rust source files carry an in-file "Modified for
basal-rs" notice. The new `merge_storage.rs` module is identified above and
included in the patches and checksum manifest. Serialization, trainer and seed
files are not changed by `merge-table.patch` or `reverse-vocab-table.patch`;
the decoder patch changes the files listed above.
