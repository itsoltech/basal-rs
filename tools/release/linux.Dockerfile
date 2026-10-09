# Builder of the Linux release package (tools/release/package-linux.sh, context: the repository root). CUDA 12.9 toolkit
# on Rocky Linux 8, so the binaries need glibc 2.28 or newer (Debian 10+, Ubuntu 20.04+, RHEL 8+).
#
# Stages: chef (toolkit, Rust, cargo-chef) -> planner (dependency recipe) -> build (dependencies of both builds in their
# own layer, cached while Cargo.toml / Cargo.lock do not change; then the workspace, profile dist) -> out (the package
# only, for `--output type=local`).
ARG CUDA_VERSION=12.9.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8 AS chef
RUN dnf install -y gcc gcc-c++ make pkgconf-pkg-config ca-certificates tar gzip \
 && dnf clean all
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
# Rocky Linux 8's curl (7.61) has no --fail-with-body
RUN curl --fail --proto '=https' --tlsv1.2 -sS https://sh.rustup.rs -o /tmp/rustup.sh \
 && sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.95.0 \
 && rm /tmp/rustup.sh \
 && cargo install cargo-chef --locked --version 0.1.73
WORKDIR /src

FROM chef AS planner
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates crates
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS build
# our kernels for compute capability 8.0, 8.9, 9.0; candle's kernels for 8.0 (they run on newer GPUs; there is no GPU
# here to detect one)
ENV CUDA_COMPUTE_CAP=80 CUDA_COMPUTE_CAPS=80,89,90
COPY --from=planner /src/recipe.json recipe.json
RUN cargo chef cook --profile dist --recipe-path recipe.json \
 && cargo chef cook --profile dist --features basal-cli/cuda --target-dir target/cuda --recipe-path recipe.json
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates crates
ARG BASAL_GIT_SHA=unknown
ENV BASAL_GIT_SHA=${BASAL_GIT_SHA}
RUN cargo build --profile dist --locked \
 && cargo build --profile dist --locked --features basal-cli/cuda --target-dir target/cuda
COPY LICENSE NOTICE ./
# bin/basal: commands without a GPU, runs the GPU ones through libexec/basal/basal-cuda
RUN version=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml) \
 && name=basal-$version-x86_64-linux \
 && mkdir -p /pkg/$name/bin /pkg/$name/libexec/basal /out \
 && cp target/dist/basal /pkg/$name/bin/basal \
 && cp target/cuda/dist/basal /pkg/$name/libexec/basal/basal-cuda \
 && cp LICENSE NOTICE /pkg/$name/ \
 && tar -C /pkg -czf /out/$name.tar.gz $name \
 && cd /out && sha256sum $name.tar.gz > $name.tar.gz.sha256

# Development build for tests on rented GPUs (tools/cloud/basal-cloud.py build): the CUDA binary only, profile release,
# kernels for CUDA_COMPUTE_CAPS. `docker buildx build -f tools/release/linux.Dockerfile --target dev --output
# type=local,dest=DIR .` writes DIR/basal-cuda.
FROM chef AS dev-build
ARG CUDA_COMPUTE_CAPS=80,89,90
ENV CUDA_COMPUTE_CAP=80 CUDA_COMPUTE_CAPS=${CUDA_COMPUTE_CAPS}
COPY --from=planner /src/recipe.json recipe.json
RUN cargo chef cook --release --features basal-cli/cuda --recipe-path recipe.json
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates crates
ARG BASAL_GIT_SHA=unknown
ENV BASAL_GIT_SHA=${BASAL_GIT_SHA}
RUN cargo build --release --locked --features basal-cli/cuda && mkdir -p /out && cp target/release/basal /out/basal-cuda

FROM scratch AS dev
COPY --from=dev-build /out/ /

FROM scratch AS out
COPY --from=build /out/ /
