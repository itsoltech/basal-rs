# Builder of the Linux release package (tools/release/package-linux.sh): CUDA 12.9 toolkit on Rocky Linux 8, so the
# binaries need glibc 2.28 or newer (Debian 10+, Ubuntu 20.04+, RHEL 8+).
ARG CUDA_VERSION=12.9.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8
RUN dnf install -y gcc gcc-c++ make pkgconf-pkg-config ca-certificates tar xz git \
 && dnf clean all
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
# Rocky Linux 8's curl (7.61) has no --fail-with-body
RUN curl --fail --proto '=https' --tlsv1.2 -sS https://sh.rustup.rs -o /tmp/rustup.sh \
 && sh /tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain 1.95.0 \
 && rm /tmp/rustup.sh
WORKDIR /src
