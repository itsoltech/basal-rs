# Build and run environment for the CUDA backend and the upstream baseline on Linux (RTX 6000 Ada, sm_89).
#
#   docker build -t basal-dev:cuda tools/cuda
#   docker run -d --name basal-dev --gpus all --ipc=host -v "$PWD":/work -w /work basal-dev:cuda sleep infinity
#   docker exec -it basal-dev bash
#
# The host only needs the NVIDIA driver and the container runtime; the toolkit (nvcc, cuBLAS headers) comes from the
# image. Cargo target and the upstream venv live in the mounted /work.
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config git curl ca-certificates rsync python3 python3-dev \
    cuda-nsight-systems-12-9 \
 && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:/opt/uv:$PATH CUDA_COMPUTE_CAP=89
RUN curl --fail-with-body --proto '=https' --tlsv1.2 -sS https://sh.rustup.rs -o /tmp/rustup.sh \
 && sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.95.0 -c clippy,rustfmt \
 && rm /tmp/rustup.sh
RUN curl --fail-with-body -LsS https://astral.sh/uv/install.sh -o /tmp/uv.sh \
 && env UV_INSTALL_DIR=/opt/uv UV_NO_MODIFY_PATH=1 sh /tmp/uv.sh && rm /tmp/uv.sh
