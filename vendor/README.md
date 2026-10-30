# Vendored Candle, tokenizers and Oniguruma patches

`candle-core/` is the library from the official crates.io `candle-core 0.11.0`
release. The workspace patches that package locally because both the published
release and Candle's upstream manifest still require `zip 8.6.0`, affected by
[RUSTSEC-2026-0367](https://rustsec.org/advisories/RUSTSEC-2026-0367.html),
[RUSTSEC-2026-0368](https://rustsec.org/advisories/RUSTSEC-2026-0368.html) and
[RUSTSEC-2026-0369](https://rustsec.org/advisories/RUSTSEC-2026-0369.html).
All three advisories are fixed in `zip >=9.0.0`.

## Provenance and changes

- Source: <https://static.crates.io/crates/candle-core/candle-core-0.11.0.crate>.
- Original crate SHA-256:
  `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706`.
- The 65 files under `src/`, `LICENSE` and the upstream `README.md` originate
  from that verified archive. Of the source files, 61 remain byte for byte
  identical. `src/npy.rs` and `src/pickle.rs` adapt archive names to zip 9's
  fallible API: propagate name-decoding errors instead of treating names as
  infallible. `src/quantized/gguf_file.rs` adds the read validation described
  below. `src/quantized/ggml_file.rs` adds checked materialization and aligned
  typed copies as described below. No dequantization arithmetic, tensor
  operations, block layouts, kernels or model code change.
- The normalized release manifest changes only the runtime dependency
  `zip = "8.6.0"` to `zip = "9.0.0"`, retaining `default-features = false`.
  Upstream example, test and benchmark targets and their dev dependencies are
  omitted together with their files. No test framework is added to this project.
- `Cargo.lock` pins the patched registry version of `zip` and retains the other
  existing runtime dependency versions. No advisory exceptions are added.
- `candle-core/zip-9.patch` records the prior archive compatibility delta from
  the official manifest and source; it is unchanged by the GGUF reader patch.
  Apply `candle-core/gguf-read-validation.patch` afterwards, from the crate
  root, to reproduce the additional GGUF and GGML source deltas. Its GGUF preimage is the
  archive's unchanged GGUF source (SHA-256
  `303d0bc548677b41a572b4c878a79905108775eba840037daf778b051d281822`).
  The unchanged archive GGML preimage has SHA-256
  `dfc48f822e4393bc836efcf782f4a6bb36168ac9d641324a054f13ee862aadac`.
  `candle-core/SHA256SUMS` records the resulting library files and combined patch.

Verify the copied files with:

```sh
cd vendor/candle-core
sha256sum --check SHA256SUMS
```

The original license remains in `candle-core/LICENSE`. These are temporary
vendored patches. Remove the
workspace `[patch.crates-io]` entry, its workspace exclusion and this directory
when an official compatible Candle release uses the fixed `zip` and provides
equivalent GGUF read validation and GGML materialization safety. Review the release diff and repeat the
dependency and build checks before removal.

The patch is included in both Docker build paths, the cloud build source sync
and the CUDA source sync tool. The sync tools retain `sdk/rust` at its workspace
path as required by the Docker builders.
CI runs on changes under `vendor/`. Compilation and an advisory scan do not
establish GPU runtime parity or renew the existing performance measurements.

## Validation of the prior archive compatibility patch

- `cargo deny --locked --all-features check --hide-inclusion-graph`: passes
  advisories, licenses, sources and bans, retaining existing duplicate warnings.
- Workspace formatting, release Clippy and private Rustdoc with warnings denied:
  pass locally on Linux without GPU and with the Intel feature.
- The existing scoped `paste` exception matches its approved dependency graph.
- Applying `zip-9.patch` to the checksum-verified official archive reproduced
  the prior 68 selected source, manifest and license/documentation files
  exactly, before the separate GGUF delta.
- A local one-off comparison reads upstream's three PTH fixtures (plain,
  nested state dictionary and Fortran storage) and NPZ eagerly and by index.
  Names, original dtypes, shapes and the decoded values of seven small integer
  tensors agree before and after the patch. No test framework or CI test gate
  is added. This is an archive-reader check, not model inference parity.
- The pinned cargo-chef 0.1.73 recipe assembles in an isolated directory with
  the vendored dependency and resolves locked metadata successfully.

Full Docker execution was unavailable in the local environment because access
to the daemon was denied. CUDA and Metal compile/lint verification belongs to
CI; no GPU inference was performed for this patch. Existing performance reports
continue to describe their original measured binaries.

## GGUF read validation

The separate GGUF patch rejects duplicate metadata keys and tensor names after
the existing trailing-NUL removal and lossy UTF-8 decoding. Missing alignment
still selects 32. Positive U8/U16/U32 and I8/I16/I32 alignments retain the
reader's prior behavior, including non-power-of-two values; zero, negative and
other typed values now return errors rather than dividing by zero or silently
selecting the default. This is a compatibility policy for the existing reader,
not a claim of full GGUF specification validation.

Dimensions must fit `usize`. Checked element/block/byte calculations and file
spans run both for the parsed inventory and for independently constructed public
`TensorInfo` values. They precede tensor-buffer allocation and GGML decoding.
The last dimension must also satisfy the existing QTensor block constraint.
Scalar inventory entries keep the empty-product element count of one; reading
a scalar still fails because QTensor does not support scalars. Zero-element
shapes remain accepted in the inventory where their ordered intermediate products,
block shape and zero-byte span are valid; reading them now returns an error before
allocation or decoding. Padding and tensor bytes are unchanged. An empty
inventory needs no tensor span, so an aligned data offset past EOF alone is not
an error. The existing writer is unchanged, including its empty-inventory
padding.

Alignment rounding and both span additions are checked. The legacy `Content::read`
keeps its compatibility/error ordering and decoded-name behavior. Existing file-size,
string/count/dimension/depth caps remain. Tensor and string read buffers use
fallible reservation. This does not impose a product memory budget or guarantee
OOM immunity for arbitrary large files: metadata collections, decoded strings,
GGML conversion and device storage still allocate separately. Native model
loading/mapping and a configurable loader memory budget remain future work;
this prerequisite does not close REQ-009.

Static validation for this GGUF delta uses Rust 1.95.0, the locked offline
workspace graph and no runtime probes. Direct `candle-core` release lint via
the pinned `clippy-driver` wrapping `cargo check`, with warnings denied and no
lint cap on Candle, passes. Workspace release Clippy, formatting and the cached offline
dependency-policy check pass. The standalone file is formatted with rustfmt's
100-column/default-heuristics settings to retain its upstream style. Reapplying
the GGUF patch to its pinned preimage reproduces the source byte for byte, and
the complete vendored checksum manifest passes.

Direct private Rustdoc with warnings denied is blocked by ten pre-existing
upstream documentation diagnostics: broken links in `error.rs`, `ggml_file.rs`,
`tensor.rs` and `indexer.rs`, and the bare specification URL in `gguf_file.rs`.
These are not suppressed. Static checks do not establish parser runtime
behavior, model equivalence, GPU behavior or complete input safety.

## Explicitly bounded GGUF header reading

`Content::read_with_limits(reader, &ReadLimits)` adds an opt-in header policy
without changing legacy `Content::read` callers or tensor materialization.
Defaults are 64 MiB of raw header reads, 4096 metadata entries, 8195 tensors,
1 MiB per raw string, 32 MiB of cumulative raw string payload, 1,048,576
metadata `Value` nodes, two enclosing array levels, rank four and a 256 MiB
cumulative allocation ledger. Callers can tighten every allowance; zero means
no allowance, and checked arithmetic rejects accounting overflow. The existing
generic format caps still apply to larger caller limits. Root values and array
containers count as nodes, as does every element of every nested array.

Byte checks cover scalar/length fields as well as strings, arrays and tensor
inventory, before their reads. The budget is relative to the initial reader
position. File-sizing and position seeks do not count file payload as header;
alignment padding and tensor payload are not read. Physical remaining-file
checks still apply. Counts, minimum encoded sizes and storage charges precede
fallible reservations for maps, array slots, strings and dimensions. Maps are
reserved once for their full declared counts, and arrays for all their elements,
so parsing those collections requires no uncharged capacity growth.

Bounded keys and tensor names require valid UTF-8 and reject every NUL byte;
duplicate checks run before insertion without name normalization. Metadata
value strings retain trailing-NUL removal and lossy UTF-8 decoding, including
tokenizer byte-token content. Strict names transfer their raw buffer into the
final string. Value strings precharge both the raw buffer and a separate
fallibly reserved output buffer of three bytes per raw byte, covering lossy
expansion and simultaneous temporary/output storage.

Ledger units are planned requested bytes, accumulated without refunds. Nonempty
buffers add 64 bytes of allowance beyond requested element storage. Nonempty
maps charge `4 * entries + 16` buckets, each with `size_of::<(String, Value)>()`
or `size_of::<(String, TensorInfo)>()` plus a control byte, and 64 extra bytes.
This conservative table allowance assumes the current standard-library hash
table layout, including capacity, control and alignment slack. It is not a
physical memory bound: allocator metadata/rounding, overcommit, stack values,
error diagnostics, reader-internal allocations and unrelated process memory
are outside this ledger. Fallible reservation cannot guarantee OOM immunity.

The shared parser retains generic v1/v2/v3, alignment and checked shape/span
validation. This API does not implement native model loading, exact model
inventory admission, canonical G1 alignment/version rules, immutable-source
leasing, payload quality or device/staging budgets. G0/G0b materialization and
CUDA source lifetime containment are unchanged. REQ-009 remains partial.

G1a uses fresh Rust 1.95.0 offline/locked static checks with a private target:
vendor/workspace formatting, direct uncapped Candle Clippy, workspace release
Clippy and private workspace Rustdoc pass for default and CUDA builds. Direct
Candle Rustdoc retains the same ten upstream diagnostics described above,
without new public-API diagnostics or suppressions. All 69 vendor checksums and
combined-patch reproduction pass; dependency-policy evidence is inherited with
unchanged inputs. No malformed-file replay, model/bitwise comparison, physical
memory measurement or GPU runtime check was performed. Metal was not checked.

## GGML typed materialization prerequisite

The same patch also hardens the public `qtensor_from_ggml` entry point, independent
of GGUF callers. Scalars and zero-element shapes return crate errors. For nonzero
dimensions, a checked total bounds both forward element and reverse contiguous
stride products. The last dimension must be divisible by the dtype's block size;
the block byte count is checked and the input must contain that many bytes.
Trailing bytes remain ignored. Native endian interpretation is preserved; this
does not add cross-endian decoding or any dtype previously unsupported here.
The legacy GGML tensor reader reuses the size checks before its fallible raw
buffer reservation. This is not comprehensive legacy GGML metadata hardening.

Only the existing concrete float and quantized block types are copied with
unaligned reads into an owned, correctly aligned `Vec<T>`. Their fields are
integers, half floats and floats with valid arbitrary bit patterns; the block
size assertions establish no padding. The public safe `GgmlType` trait alone
does not authorize byte reinterpretation. CPU storage takes ownership of this
vector. Metal copies the data before its upload helper returns. CUDA supports
only the default per-thread stream; explicit shared streams are rejected before
upload because another thread could capture the copy between a status query
and synchronization. It also refuses active graph capture and capture-status
errors before upload, and retains the owned vector until an explicit stream
synchronization confirms completion, including after an upload error. If
completion cannot be confirmed, it retains
the host allocation instead of freeing a possibly live source and returns an
error; repeated failed uploads can therefore retain additional memory. No
borrowed typed slice is created from the input bytes.

G0b uses Rust 1.95.0 offline static checks: direct uncapped Candle Clippy via
`clippy-driver`/`cargo check`, workspace Clippy and formatting pass. Workspace
private Rustdoc passes; direct Candle Rustdoc retains the ten inherited
diagnostics listed above. The combined patch reproduces both sources from their
pinned preimages and all vendor checksums pass. Dependency-policy results are
inherited for the unchanged manifests and lockfile. Parent verification also
passes all-features workspace Clippy with the local CUDA 12.9.1 toolchain; the
initial child CUDA attempt lacked `nvcc` in its command environment. Metal
compilation is not available on this Linux host. Neither GPU runtime behavior nor model bitwise
equivalence, performance or general OOM safety was measured. REQ-009 remains
partial; model mapping and resource policy remain open.

## tokenizers 0.23.2: explicit BPE merge scratch and table storage

`tokenizers/` is the library from the official crates.io `tokenizers 0.23.2`
release. It is patched locally so that opt-in core routes can admit the BPE
builder's temporary merge scratch buffer and the built model's merge-table
backing on the existing GGUF source ledger; see
`read_with_bpe_merge_scratch_budget` and `read_with_bpe_merge_table_budget` in
[GGUF.md](../docs/GGUF.md). No
advisory motivates this patch. The `[patch.crates-io]` entry matches only
`0.23.2`; `tokenizers 0.22.2`, required by another dependency, stays on
crates.io.

- Source, archive SHA-256, upstream commit and the exact omissions are in
  `tokenizers/PROVENANCE.md`. `tokenizers/UPSTREAM-SHA256SUMS` lists the 77
  copied release files before the change. Of the 73 copied source files, 66
  remain byte for byte identical after the first two patches and 62 after the
  third, `decoder-regex.patch` (see below); `LICENSE`, `README.md` and
  `Cargo.toml.orig` are unchanged.
- The normalized release manifest omits only the upstream test and benchmark
  targets, their dev dependencies and the profiles that Cargo ignores outside
  the workspace root. Features and edition are unchanged; the merge-table
  patch below adds two already-locked runtime dependencies.
  `Cargo.lock` changes only the `tokenizers 0.23.2` entry, which loses its
  registry source and checksum and gains those two dependency names. No
  dependency version changes.
- `BpeBuilder::build` and the new `build_with_merge_storage` share one build
  body. The legacy build still allocates `vec![0; max_len]` at the same point;
  the new one takes the buffer from a caller's `MergeStorage`, requires the
  slice to have the requested length and uses that one slice for all merge
  writes and reads. Maximum length, maps, merge order, ranks and the returned
  model are computed by the same code. `Tokenizer::from_str_with_merge_storage`
  passes the storage through explicit deserialization seeds for the tokenizer,
  model and BPE; they reuse the existing visitors and tagged/legacy helpers.
  Existing `Deserialize`, `FromStr`, file and builder routes keep their code
  paths and allocations.
- Storage refusal is returned unchanged as `MergeBuildError::Storage`, never
  boxed or formatted; a legacy untagged model trial does not treat it as a
  mismatch. Parsing and model errors keep the existing boxed backend channel.
  After a refusal no later entry or fallback model is constructed; the
  remaining tokenizer JSON is only skipped syntactically, because serde_json
  requires the closing brace. The refusal wins over any error in that skip or
  in trailing input: a skip error leaves it in a private per-call slot, and
  the entry point returns it before checking trailing input. Only the typed
  refusal moves without new formatting or boxing; serde parsing, skipping and
  their syntax diagnostics still allocate outside the admitted buffer.

Verify the copied files with:

```sh
cd vendor/tokenizers
sha256sum --check SHA256SUMS
```

`merge-scratch.patch` applies with `patch -p1` to the seven preimages listed in
`UPSTREAM-SHA256SUMS` and creates `src/utils/merge_storage.rs`; the result is
byte for byte equal to the vendored files. Remove this directory, its workspace
exclusion and patch entry when the admitted provider route no longer needs the
explicit storage hook or upstream provides an equivalent. The existing Docker,
release and sync paths copy the whole `vendor/` directory; no package build was
run for this change.

Static validation with Rust 1.95.0, offline and locked: workspace formatting,
release Clippy and private Rustdoc with warnings denied pass for default,
Intel and CUDA all-features builds. The changed and new tokenizers files are
formatted with rustfmt's 100-column defaults, as upstream. Direct uncapped
Clippy of the vendored crate reports no diagnostics. Its direct private
Rustdoc keeps six upstream diagnostics in unchanged files (`unigram/lattice.rs`,
`normalizers/bert.rs`, `pre_tokenizers/byte_level.rs`,
`tokenizer/added_vocabulary.rs`); they are not suppressed. `cargo deny
--offline` passes with the existing 22 duplicate warnings, using the cached
advisory database only. `tools/ci/check-paste-exception.sh` normalizes only the
checkout path of the pinned local `tokenizers v0.23.2` package before comparing
the exact approved list. The exception's dependents, versions and expiry are
unchanged; new dependents or changed versions still fail the check. No
test, tokenizer execution, model comparison, memory measurement or GPU check
was performed. REQ-009 remains partial.

### Charged BPE merge table

A second local patch, `tokenizers/merge-table.patch`, lets a second opt-in core
route, `read_with_bpe_merge_table_budget` in [GGUF.md](../docs/GGUF.md), admit
the actual backing blocks of the built BPE's merge hash table on the same GGUF
source ledger. It applies with `patch -p1` after `merge-scratch.patch` and also
changes the pinned upstream `src/models/bpe/word.rs`.

- The vendored manifest adds direct dependencies on `hashbrown =0.16.1`
  (`allocator-api2` feature, no defaults) and `allocator-api2 =0.2.21`
  (`alloc`, no defaults). Both were already in `Cargo.lock` at those versions;
  the resolved graph still has one `allocator-api2` and no hashbrown `nightly`
  feature. The basal-core graph now includes hashbrown 0.16.1.
- `MergeStorage` gains a provided `merge_table` method returning `None`, so the
  existing scratch-only storage and every legacy route still collect the merge
  table into the upstream `AHashMap`. With an account, the same validation
  builds a hashbrown map with the same key, value and `ahash::RandomState`. A
  private, non-`Clone` allocator admits each block's exact size before the
  global allocation, returns exactly that length, frees before releasing, and
  keeps its first refusal for the build, which returns it through the
  unchanged `MergeBuildError::Storage` channel and message. Model clones share
  the charged table through an `Arc`.
- Ranks keep upstream's `as u32` cast and serialization keeps its unstable sort,
  so wrapped equal ranks follow the table's iteration order. That order equals
  the global table's only under equal hasher keys, an identical hashbrown
  algorithm and layout in the standard library and the same target; this is
  not otherwise claimed. Serialization, trainer and seed files are unchanged.

Static validation with Rust 1.95.0, offline and locked: workspace formatting,
release Clippy and private Rustdoc with warnings denied pass for default,
Intel and CUDA all-features builds. The changed tokenizers files keep
rustfmt's 100-column upstream style. Direct uncapped Clippy of the vendored
crate reports no diagnostics; its direct private Rustdoc reports only the same
six upstream diagnostics listed above. `cargo deny --offline` passes with the
same 22 duplicate warnings and the cached advisory database only; the paste
exception check passes. Both patches reproduce all nine changed or new files
from the pinned preimages, and all 82 vendored checksums pass. No test,
tokenizer execution, output comparison, memory measurement or GPU check was
performed. REQ-009 remains partial.

### Charged decoder regex backing (onig 6.5.3, onig_sys 69.9.3)

A third opt-in core route, `read_with_decoder_regex_budget` in
[GGUF.md](../docs/GGUF.md), also admits the retained compiled Oniguruma
backing of every `Replace` regex in the tokenizer's decoder tree on the same
GGUF source ledger. It needs the vendored `onig/` and `onig_sys/` copies of the
official crates.io releases; their sources, archive checksums, upstream commit,
omissions and local changes are in `onig/PROVENANCE.md` and
`onig_sys/PROVENANCE.md`. No advisory motivates this patch. The
`[patch.crates-io]` entries match the locked versions, so the one native
Oniguruma of the graph, also used by `tokenizers 0.22.2`, is the patched one.
`Cargo.lock` changes only the `onig` and `onig_sys` entries, which lose their
registry source and checksum. No dependency version changes.

- `onig_sys` adds `onig_basal_new` (new `onig_basal.c`/`onig_basal.h`) and its
  Rust binding. An admitted regex carries a private account pointer in its
  `regex_t`. Only the retained compiled-program blocks dispatch on it:
  `regex_t` itself (in one container with the account), ops/ocs, string and
  class operands, backref numbers, the string pool, repeat ranges and the
  optimizer string. Each block is one global-allocator allocation of a size
  header plus payload, charged for its full size before allocation and
  released after deallocation. Growth charges the new block while the old one
  lives. Admitted-only guards turn signed `int` size overflow into a typed
  `CapacityOverflow`. Compilation and search otherwise run the unchanged
  native code under the same compile lock, so process-wide syntax, callout,
  user-property, case-fold and limit settings are observed exactly where the
  legacy constructor observes them.
- **Legacy effect.** Every legacy `regex_t` in the process, including those of
  `tokenizers 0.22.2`, grows by one pointer, and the listed legacy compile and
  free sites evaluate one NULL branch. Every legacy `SysRegex` in tokenizers
  0.23.2 grows by one enum discriminant. Their algorithms, accepted inputs,
  outputs, errors and `Debug` text are unchanged, and legacy `Replace` clones
  still recompile. Earlier wording that legacy routes keep their exact
  allocation layout no longer holds for these two types. Object sizes were not
  measured.
- `onig` adds `Regex::new_with_account` with the `RegexAccount` trait and
  `RegexAdmissionError` (`Refused`, `Allocation`, `CapacityOverflow`,
  `Compile`). No Rust unwind crosses C: a panic in `try_acquire` is caught,
  kept as the first failure and resumed after native cleanup, the lock guard
  and the account are released; a panic in `release`, in dropping the account
  or in dropping a later, ignored failure aborts the process. The compile lock
  is never poisoned by this route.
- `tokenizers/decoder-regex.patch` applies with `patch -p1` after the two
  earlier patches and changes `src/utils/onig.rs`, `src/utils/merge_storage.rs`,
  `src/normalizers/replace.rs`, `src/decoders/mod.rs`,
  `src/decoders/sequence.rs` and `src/tokenizer/serialization.rs`.
  `MergeStorage` gains a provided `decoder_regex` method returning `None`.
  With an account, the tokenizer's `decoder` entry is read by a seed that
  repeats the legacy decoder body, admits each `Replace` regex and recurses
  into `Sequence` elements; every other variant is read unchanged. The first
  refusal is returned through the unchanged `MergeBuildError::Storage`
  channel, never boxed or formatted, and ends the tokenizer read as for a
  model refusal. A refusal inside a legacy untagged decoder is never treated
  as a variant mismatch. Admitted regexes are shared by `Replace` clones, so
  the charge lasts until the last owner drops it, on any thread.
  `RegexCharge` and `AdmittedRegexError` are crate-private items in
  `merge_storage.rs`, because the `utils::onig` module is private.
- Not admitted: regex parse and compile temporaries, the name table and
  callout (`RegexExt`) data, the process-wide Oniguruma registries allocated
  at the first compilation, match-time state, the `Arc`s and `String`s around
  the regex, JSON values, normalizer and pre-tokenizer regexes and every
  regex built outside this route. Without the `onig` feature the account is
  not requested.
- With `RUSTONIG_SYSTEM_LIBONIG` or `RUSTONIG_DYNAMIC_LIBONIG`, the bundled
  sources are not compiled; a binary that references `onig_basal_new` then
  fails to link. This is reasoned from the build script, not linked.

Verify the copied files with `sha256sum --check SHA256SUMS` in `onig/`,
`onig_sys/` and `tokenizers/`. `onig/regex-account.patch` and
`onig_sys/regex-account.patch` apply with `patch -p1` to the preimages listed in
their `UPSTREAM-SHA256SUMS`; the three tokenizers patches apply in order. The
results are byte for byte equal to the vendored files.

Static validation with Rust 1.95.0, offline and locked: workspace formatting,
release Clippy and private Rustdoc with warnings denied pass for default,
Intel and CUDA all-features builds. These logs now show 16 inherited
`unpredictable_function_pointer_comparisons` warnings from the unchanged
upstream `onig_sys/src/ffi.rs`, because the local path package is no longer
lint-capped; they are not suppressed. Direct uncapped Clippy of `onig_sys`,
`onig` and tokenizers 0.23.2 reports no diagnostic in changed or new code;
the remaining diagnostics are in unchanged upstream lines and are not
suppressed. Their direct private Rustdoc adds only the same six upstream
tokenizers diagnostics listed above. The changed vendored Rust files keep rustfmt's 100-column upstream
style. C compiler warnings are disabled by the upstream build script and no
separate C diagnostic was run. `cargo deny --offline` passes with the same 22
duplicate warnings and the cached advisory database only; the paste exception
check passes. Locked feature trees for Linux, macOS and the CUDA feature show
one local `onig_sys` without `generate`/bindgen, used by both tokenizers
versions through the local `onig`. No test, native execution, tokenizer run,
output comparison, memory measurement or GPU check was performed. REQ-009
remains partial.
