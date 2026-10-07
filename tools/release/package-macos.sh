#!/bin/bash
# macOS release package (Apple Silicon, Metal):  tools/release/package-macos.sh [OUT_DIR]   (from the repository root)
# Writes OUT_DIR/basal-<version>-aarch64-apple-darwin.tar.gz (+ .sha256): bin/basal, LICENSE, NOTICE.
set -euo pipefail
OUT=${1:-dist}
VERSION=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
NAME=basal-$VERSION-aarch64-apple-darwin
[ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || { echo "build on an Apple Silicon Mac" >&2; exit 1; }
cargo build --profile dist --locked -p basal-cli
W=$(mktemp -d)
mkdir -p "$W/$NAME/bin" "$OUT"
cp target/dist/basal "$W/$NAME/bin/basal"
cp LICENSE NOTICE "$W/$NAME/"
tar -C "$W" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
ls -l "$OUT/$NAME.tar.gz"
