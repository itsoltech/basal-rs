# Provenance of the vendored onig 6.5.3

## Release

- Crate: `onig 6.5.3` from crates.io,
  <https://static.crates.io/crates/onig/onig-6.5.3.crate>.
- Archive SHA-256:
  `0cc3cbf698f9438986c11a880c90a6d04b9de27575afd28bbf45b154b6c709e2`.
  This is the checksum that `Cargo.lock` recorded for the registry package
  before the local patch.
- `.cargo_vcs_info.json` in the archive: commit
  `3572dd210e9bbfbf056c12f3a65bfcc9fede7866`, path `onig`
  (file SHA-256 `3c8806781165958ff170bbe182f5ff622da80c5728dbf73559e6ece8a311d648`).
- Upstream `Cargo.lock` in the archive, not copied: SHA-256
  `67b3385174f69cc7ca81881f9f0743957a3cb3efb98d32f9e97d2bd3a16d90ba`.
- License: MIT, retained unchanged in `LICENSE.md`.

## Copied files

The 12 files under `src/`, `LICENSE.md`, `README.md`, the normalized release
`Cargo.toml` and the original `Cargo.toml.orig` were copied from the release.
`UPSTREAM-SHA256SUMS` lists all 16 copied files before any change. The copy
was compared with the cargo registry source and with a separately extracted
archive whose SHA-256 equals the lockfile checksum.

Omitted from the archive: `examples/` (ten files), `Cargo.lock` and
`.cargo_vcs_info.json`. The manifest omits the 10 `[[example]]` targets of
those files; its first line records this. Dependencies, features, edition and
library target are unchanged. Inline `#[cfg(test)]` modules remain unchanged
and are not run by this project.

## Local change

`regex-account.patch` reproduces the two modified files from their pinned
preimages in `UPSTREAM-SHA256SUMS` and creates the third. Apply it with
`patch -p1` in a directory holding those preimages:

- `Cargo.toml`: the omission above.
- `src/lib.rs`: the private `admitted` module and the re-export of
  `RegexAccount` and `RegexAdmissionError`.
- New `src/admitted.rs`: `Regex::new_with_account`. It reads the same
  `Syntax::default()`, uses the same options and UTF-8 encoding and holds the
  same compile lock as `Regex::new`, and calls `onig_sys::onig_basal_new`.
  Account callbacks are monomorphized `extern "C"` trampolines. Each runs
  its Rust code inside `catch_unwind`; a panicking `try_acquire` is recorded
  and resumed after the lock and the account are released; a panic in
  `release`, in dropping the owner or in dropping an ignored later failure
  aborts the process. Only the first failure of one construction is kept.
  `Regex`, its `Drop`, `Send`/`Sync`, every existing API, error and message
  are unchanged.

`SHA256SUMS` covers every file in this directory except itself.
