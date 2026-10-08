# Contributing to basal-rs

[Polski](CONTRIBUTING.pl.md)

basal-rs runs the Basal decision models with the answers of the upstream FP32 model, faster. A change is accepted
when it keeps those answers and its effect is measured. Issues and pull requests may be written in Polish or English.

## Before you start

- Bugs, upstream disagreements, performance and ideas: the [issue forms](https://github.com/itsoltech/basal-rs/issues/new/choose).
  Questions: [Discussions](https://github.com/itsoltech/basal-rs/discussions).
- Security vulnerabilities: privately, see [SECURITY.md](SECURITY.md).
- For a larger change (a new kernel, scheduler, API) open an issue first, so the approach and the measurement can be
  agreed before the work.

## Building

Rust stable (checked with 1.95).

```sh
# Metal (macOS, Apple Silicon)
cargo build --release

# CUDA (Linux; image with the toolkit: tools/cuda/Dockerfile)
docker build -t basal-dev:cuda tools/cuda
docker run -d --name basal-dev --gpus all --ipc=host -v "$PWD":/work -w /work basal-dev:cuda sleep infinity
docker exec basal-dev cargo build --release --features basal-cli/cuda
```

Layout of the code: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How changes are verified

The project has no unit, integration or end-to-end tests, and pull requests should not add test frameworks, snapshots
or CI test gates. Correctness is checked by comparing the runtime's outputs with the upstream reference, and speed by
measurements ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

- A change that should not change the computation (packing, caches, scheduling, refactoring) gives output
  bit-identical to `main`: `basal export` with both versions, `basal compare` reports 0.0.
- A change of the numerics (kernels, precision, reductions) is compared with the upstream FP32 reference: the
  decisions stay the same and the logit and probability differences are recorded.
- A performance claim comes with a measurement before and after on the same hardware and precision. Compared with
  upstream, at the same precision too (e.g. basal-rs f16 against MLX f16). Laptops are measured on AC power; the
  conditions (GPU, power limit, model revision) go into the report.

Measurements go into a new directory under [reports/](reports/README.md) with a `README.md` and the data the numbers
come from; existing reports are not overwritten. Reports and code contain no data identifying the machines they were
made on (hostnames, IP addresses, home paths). Upstream checkouts in `.baseline/` are not modified.

`cargo fmt --all --check` and `cargo clippy --locked --workspace --release --all-targets -- -D warnings` pass.
CI runs them on Linux without CUDA and on macOS with Metal, and also checks all features on Linux with CUDA.

Every pull request runs formatting, Clippy, Rustdoc, dependency and workflow checks. See [CI quality gates](docs/CI.md).
CI does not run tests while the project has none. Docker images are built and pushed only for releases or a manual
run with a release tag.

## GEMM tables

`gemm_table: auto` needs a batch-invariant cuBLASLt table per GPU, cuBLASLt version, model and dtype; generating
one takes several to tens of minutes at the first start. Tables in
[crates/basal-cli/gemm-tables/](crates/basal-cli/gemm-tables) are compiled into the binary, and `basal serve` uses a
matching one after cuBLASLt accepts its algorithms on the GPU. A user who generated a table for a GPU or model the
build lacks can send it with `basal gemm-share` (an issue with the `gemm-table` label, after a confirmation; without
`gh` through the "GEMM table" issue form).

Only the CUDA build carries tables, and `build.rs` bounds them: f16, the cuBLASLt version of the CUDA toolkit
(`cublas_api.h`) and the current search version, one table per GPU and weight shapes, the file name
`<gpu>--<model>--f16--cublaslt<version>.json`. It keeps only the fields the server reads (13-15 KB per table instead
of ~34 KB) and stops the build when a table breaks a rule or all of them exceed 1 MiB; CI (Clippy with CUDA) runs
it for every pull request. A new CUDA toolkit or search version therefore means new tables, not more of them.

Before a table is added, a maintainer checks it on the same GPU and cuBLASLt version, with the table as
`gemm_table`: exports of the basal-bench items as single requests, in a tree and in a budgeted batch compared with
each other (0.0), the decision set against upstream FP32, and the time against a table generated there
(`basal bench`, the context ladder). The file name is `<gpu>--<model>--<dtype>--cublaslt<version>.json`, with the
JSON from the issue plus a `source` field naming the report or issue.

## Rust code quality

The implementation and review rules are in [AGENTS.md](AGENTS.md#jakość-kodu-rust), based on
`itsolpowers:rust-implementation`. Review ownership and unnecessary copies, public API invariants, typed errors
and their causes, input validation, async worker lifecycles, and documented safety of FFI and `unsafe`.
Prefer correctness and readability; justify optimization with requirements or measurements.

Every crate inherits the workspace lints, which reject `dbg!`, `todo!`, `unimplemented!`, undocumented `unsafe`
and implicit unsafe operations inside unsafe functions. Keep lint exceptions local and justified with `reason`;
prefer `#[expect(..., reason = "...")]` for an expected lint.
Formatting and Clippy complement `itsolpowers:rust-review` and `itsolpowers:itsol-self-review`.
CI compiles and lints Metal and CUDA code; it does not establish GPU runtime correctness or model parity.

## Pull requests

- One coherent change per pull request; the [template](.github/pull_request_template.md) lists what to state.
- Title in the Angular convention (`feat(cli): …`, `fix(gpu): …`, `perf(metal): …`, `docs: …`): pull requests are
  squash-merged and the title becomes the commit message.
- Documentation (README, `docs/`, `serve.example.yml`) changes together with the behaviour or options it describes.
- A pull request needs one approving review of a maintainer (`@itsoltech/basal`) and a passing `lint` check.

## Licence

Contributions are accepted under the [Apache License 2.0](LICENSE) of the project. This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
