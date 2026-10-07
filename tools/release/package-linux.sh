#!/bin/bash
# Linux release package (Rocky Linux 8 CUDA builder with cargo-chef, tools/release/linux.Dockerfile):
#   tools/release/package-linux.sh [OUT_DIR]        (from the repository root; needs Docker with buildx, no GPU)
# Writes OUT_DIR/basal-<version>-x86_64-linux.tar.gz (+ .sha256) with
#   bin/basal                   commands without a GPU (doctor, setup, init, update, ...); runs the GPU ones through
#   libexec/basal/basal-cuda    the CUDA build (kernels for compute capability 8.0, 8.9, 9.0)
# and LICENSE, NOTICE. The CUDA libraries are not in the package: `basal setup` downloads them from NVIDIA.
# BASAL_BUILD_CACHE: extra buildx cache options, e.g. "--cache-from type=registry,ref=R --cache-to type=registry,ref=R,
# mode=max" in CI; locally the layers stay in the builder.
set -euo pipefail
OUT=${1:-dist}
SHA=${BASAL_GIT_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}
mkdir -p "$OUT"
# shellcheck disable=SC2086
docker buildx build -f tools/release/linux.Dockerfile --target out --output "type=local,dest=$OUT" \
  --build-arg BASAL_GIT_SHA="$SHA" ${BASAL_BUILD_CACHE:-} .
ls -l "$OUT"/basal-*-x86_64-linux.tar.gz
