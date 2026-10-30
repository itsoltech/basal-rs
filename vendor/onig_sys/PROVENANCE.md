# Provenance of the vendored onig_sys 69.9.3

## Release

- Crate: `onig_sys 69.9.3` from crates.io,
  <https://static.crates.io/crates/onig_sys/onig_sys-69.9.3.crate>.
- Archive SHA-256:
  `1e68317604e77e53b85896388e1a803c1d21b74c899ec9e5e1112db90735edd7`.
  This is the checksum that `Cargo.lock` recorded for the registry package
  before the local patch.
- `.cargo_vcs_info.json` in the archive: commit
  `3572dd210e9bbfbf056c12f3a65bfcc9fede7866`, path `onig_sys`
  (file SHA-256 `aeb366cad8c20c1d8adf7393562c4c45bcdbe6daa25af16fd9709874d28c77a6`).
- Upstream `Cargo.lock` in the archive, not copied: SHA-256
  `545c41a783a49c1d9a9ef55595dba33bec1dbd33538e512387e7c9d51fa697bd`.
- Licenses, retained unchanged: the crate's MIT `LICENSE.md` and the bundled
  Oniguruma's BSD-2-Clause `oniguruma/COPYING`, with `oniguruma/AUTHORS`.

## Copied files

Copied from the release: `build.rs`, the normalized `Cargo.toml`,
`Cargo.toml.orig`, `LICENSE.md`, `README.md`, `src/lib.rs`, `src/ffi.rs`,
`src/bindgened.rs`, `oniguruma/COPYING`, `oniguruma/AUTHORS` and 66 files
of `oniguruma/src`: every `.c` and `.h` file plus the two Windows `config.h`
templates.
`UPSTREAM-SHA256SUMS` lists all 76 copied files before any change. The copy
was compared with the cargo registry source and with a separately extracted
archive whose SHA-256 equals the lockfile checksum.

Omitted from the archive (112 files): `Cargo.lock`, `.cargo_vcs_info.json`
and, under `oniguruma/`, documentation, samples, tests, fuzzing harnesses,
CI files, autotools/CMake/Windows build files, Unicode table generator
scripts and the `oniguruma/src` build inputs `config.h.cmake.in`,
`config.h.windows.in`, `Makefile.am`, `Makefile.windows`, both `.gperf`
files and the `make_*` and `gperf_*` scripts. The build script compiles none
of them; the manifest is unchanged.

## Local change

`regex-account.patch` reproduces the four modified files from their pinned
preimages in `UPSTREAM-SHA256SUMS` and creates three new ones. Apply it with
`patch -p1` in a directory holding those preimages:

- `oniguruma/src/regint.h`: `struct re_pattern_buffer` gains a last private
  member `basal` (an account pointer, NULL for every regex not built by
  `onig_basal_new`), plus the block header, helper declarations and
  dispatch macros. The legacy branch of each macro is the old expression.
- `oniguruma/src/regcomp.c`: only the retained compiled-backing sites dispatch
  on `reg->basal`: ops/ocs (`ops_init`, `ops_resize`, `ops_free`), exact
  string operands, class bitsets and multibyte copies, backref numbers, the
  string pool, repeat ranges, the optimizer exact string and their frees.
  Admitted-only guards record a size overflow before signed `int` arithmetic
  (`ops_new` doubling, `add_compile_string` byte length, string-pool total).
  `onig_free` hands an admitted regex to `onig_basal_free_regex`.
- New `oniguruma/src/onig_basal.h` and `onig_basal.c`: `onig_basal_new` and
  the admitted block helpers.
- `build.rs`: `onig_basal.c` is appended to the bundled file list. The
  pkg-config, dynamic and static selection is unchanged.
- `src/lib.rs` and new `src/basal.rs`: `OnigBasalAccount` and
  `onig_basal_new`, exported with or without `generate`. `ffi.rs` and
  `bindgened.rs` are unchanged.

Every legacy `regex_t` in the process now carries the extra pointer, and the
legacy compile and free sites above evaluate one NULL branch. Algorithms,
accepted patterns, results, error codes and messages of the legacy routes
are unchanged. When `RUSTONIG_SYSTEM_LIBONIG` or `RUSTONIG_DYNAMIC_LIBONIG`
selects a system library, the bundled sources, including `onig_basal.c`, are
not compiled. A binary that references `onig_basal_new` then fails to link,
because the system library lacks it. Legacy-only linking depends on the linker
discarding unused references to that symbol and has not been verified. These
are source conclusions, not link or runtime results.

`SHA256SUMS` covers every file in this directory except itself.
