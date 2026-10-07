# Runtime image of basal-rs (CUDA): `basal serve --config /config/serve.yml`.
#
#   docker build -t basal-rs -f docker/Dockerfile .
#   docker run --gpus all -p 8000:8000 -v "$PWD/serve.yml:/config/serve.yml:ro" -v basal-data:/data basal-rs
#
# Models given by `repo` in serve.yml are downloaded into /data/hf (HF_TOKEN for gated or private repositories),
# GEMM tables of `gemm_table: auto` are generated into /data/.cache/gemm on the first start; keep /data on a volume.
# CUDA_COMPUTE_CAPS: GPU architectures of the kernels, one image for all of them (default 80,89,90: A100 / RTX 30xx,
# RTX 6000 Ada / L40S / RTX 40xx, H100; the runtime loads the highest one not above the GPU's, and PTX also runs on
# newer GPUs through the driver's JIT); CUDA_COMPUTE_CAP: architecture of candle's kernels. The build needs no GPU.
#
# Stages: chef (CUDA toolkit, Rust, cargo-chef) -> planner (dependency recipe) -> build (dependencies in their own
# layer, cached while Cargo.toml / Cargo.lock do not change; then the workspace, profile dist) -> runtime (CUDA runtime
# + binary).
ARG CUDA_VERSION=12.9.1
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS chef
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --fail-with-body --proto '=https' --tlsv1.2 -sS https://sh.rustup.rs -o /tmp/rustup.sh \
 && sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.95.0 \
 && rm /tmp/rustup.sh \
 && cargo install cargo-chef --locked --version 0.1.73
WORKDIR /src

FROM chef AS planner
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates crates
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS build
ARG CUDA_COMPUTE_CAP=80
ARG CUDA_COMPUTE_CAPS=80,89,90
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} CUDA_COMPUTE_CAPS=${CUDA_COMPUTE_CAPS}
COPY --from=planner /src/recipe.json recipe.json
RUN cargo chef cook --profile dist --features basal-cli/cuda --recipe-path recipe.json
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates crates
# commit shown by `basal --version` (the build context has no .git); after the dependencies, which it does not change
ARG BASAL_GIT_SHA=unknown
ENV BASAL_GIT_SHA=${BASAL_GIT_SHA}
RUN cargo build --profile dist --locked --features basal-cli/cuda \
 && cp target/dist/basal /usr/local/bin/basal

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime
ARG CUDA_COMPUTE_CAPS=80,89,90
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/basal /usr/local/bin/basal
# BASAL_HOME: GEMM tables of `gemm_table: auto` in /data/.cache/gemm (src/paths.rs)
ENV HF_HOME=/data/hf BASAL_HOME=/data BASAL_ADDR=0.0.0.0:8000
LABEL org.opencontainers.image.title="basal-rs" \
      org.opencontainers.image.description="Rust runtime of the basal decision models (TypeSafe System One API), CUDA" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.basal.cuda-compute-caps="${CUDA_COMPUTE_CAPS}"
WORKDIR /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=2h \
  CMD curl --fail-with-body --silent http://127.0.0.1:8000/health > /dev/null || exit 1
ENTRYPOINT ["basal", "serve", "--config", "/config/serve.yml"]
