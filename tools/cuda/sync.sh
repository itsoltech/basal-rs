#!/bin/sh
# Copy the sources (not models, targets, reports or venvs) to a CUDA host: tools/cuda/sync.sh USER@HOST REMOTE_DIR
set -e
[ $# -eq 2 ] || { echo "usage: $0 USER@HOST REMOTE_DIR" >&2; exit 2; }
HOST=$1
DIR=$2
cd "$(dirname "$0")/../.."
rsync -az --exclude target --exclude .venv --exclude '__pycache__' crates tools Cargo.toml Cargo.lock rustfmt.toml "$HOST:$DIR/"
