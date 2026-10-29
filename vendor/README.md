# Candle archive dependency and GGUF/GGML reader patches

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

Alignment rounding and both span additions are checked. Existing file-size,
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
vector. GPU upload helpers borrow it until they return, retaining their existing
copy/lifetime contract. No borrowed typed slice is created from the input bytes.

G0b uses Rust 1.95.0 offline static checks: direct uncapped Candle Clippy via
`clippy-driver`/`cargo check`, workspace Clippy and formatting pass. Workspace
private Rustdoc passes; direct Candle Rustdoc retains the ten inherited
diagnostics listed above. The combined patch reproduces both sources from their
pinned preimages and all vendor checksums pass. Dependency-policy results are
inherited for the unchanged manifests and lockfile. CUDA compilation is blocked
locally by the missing CUDA toolkit (`nvcc`); Metal compilation is
not available on this Linux host. Neither GPU runtime behavior nor model bitwise
equivalence, performance or general OOM safety was measured. REQ-009 remains
partial; model mapping and resource policy remain open.
