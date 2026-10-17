# Candle archive dependency security patch

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
- The 65 files under `src/`, `LICENSE` and the upstream `README.md` come from
  that verified archive. Of the source files, 63 remain byte for byte identical.
  Only `src/npy.rs` and `src/pickle.rs` adapt archive names to zip 9's fallible
  API: propagate name-decoding errors instead of treating names as infallible.
  No tensor operations, kernels or model code are changed.
- The normalized release manifest changes only the runtime dependency
  `zip = "8.6.0"` to `zip = "9.0.0"`, retaining `default-features = false`.
  Upstream example, test and benchmark targets and their dev dependencies are
  omitted together with their files. No test framework is added to this project.
- `Cargo.lock` pins the patched registry version of `zip` and retains the other
  existing runtime dependency versions. No advisory exceptions are added.
- `candle-core/zip-9.patch` records the entire delta from the official manifest
  and source. `candle-core/SHA256SUMS` records the patched library files.

Verify the copied files with:

```sh
cd vendor/candle-core
sha256sum --check SHA256SUMS
```

The original license remains in `candle-core/LICENSE`. This is a temporary
archive compatibility patch. Remove the
workspace `[patch.crates-io]` entry, its workspace exclusion and this directory
when an official compatible Candle release uses the fixed `zip`. Review the
release diff and repeat the dependency and build checks before removal.

The patch is included in both Docker build paths and the CUDA source sync tool.
CI runs on changes under `vendor/`. Compilation and an advisory scan do not
establish GPU runtime parity or renew the existing performance measurements.
