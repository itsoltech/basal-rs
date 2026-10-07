#!/bin/bash
# Linux release package, built in the Rocky Linux 8 CUDA builder (tools/release/linux.Dockerfile):
#   tools/release/package-linux.sh [OUT_DIR]        (from the repository root; needs Docker, no GPU)
# Writes OUT_DIR/basal-<version>-x86_64-linux.tar.gz (+ .sha256) with
#   bin/basal                   commands without a GPU (doctor, setup, init, update, ...); runs the GPU ones through
#   libexec/basal/basal-cuda    the CUDA build (kernels for compute capability 8.0, 8.9, 9.0)
# and LICENSE, NOTICE. The CUDA libraries are not in the package: `basal setup` downloads them from NVIDIA.
set -euo pipefail
OUT=${1:-dist}
VERSION=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
SHA=${BASAL_GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
IMAGE=basal-release-linux:cuda12.9
NAME=basal-$VERSION-x86_64-linux
docker build -q -t $IMAGE -f tools/release/linux.Dockerfile tools/release > /dev/null
mkdir -p "$OUT"
# cargo registry and targets in volumes, so a second run only rebuilds what changed
docker run --rm -v "$PWD":/src -v basal-release-cargo:/opt/cargo/registry -v basal-release-target:/target \
  -e BASAL_GIT_SHA="$SHA" -e CARGO_TARGET_DIR=/target -e OUT="$OUT" -e NAME="$NAME" -w /src $IMAGE bash -c '
    set -euo pipefail
    cargo build --release --locked -p basal-cli
    W=$(mktemp -d)
    mkdir -p "$W/$NAME/bin" "$W/$NAME/libexec/basal"
    cp /target/release/basal "$W/$NAME/bin/basal"
    # our kernels for 8.0, 8.9, 9.0; candle'"'"'s kernels for 8.0 (they run on newer GPUs; no GPU to detect here)
    CUDA_COMPUTE_CAP=80 CUDA_COMPUTE_CAPS=80,89,90 \
      cargo build --release --locked -p basal-cli --features cuda --target-dir /target/cuda
    cp /target/cuda/release/basal "$W/$NAME/libexec/basal/basal-cuda"
    strip --strip-debug "$W/$NAME/bin/basal" "$W/$NAME/libexec/basal/basal-cuda"
    cp LICENSE NOTICE "$W/$NAME/"
    tar -C "$W" -czf "$OUT/$NAME.tar.gz" "$NAME"
    chown "$(stat -c %u:%g /src)" "$OUT" "$OUT/$NAME.tar.gz"'
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
ls -l "$OUT/$NAME.tar.gz"
