#!/bin/bash
# Context ladder on one rented GPU (tools/cloud/basal-cloud.py): whole requests with states of ~0.5k-16k tokens and
# 1 / 5 questions, basal-rs (`basal bench-requests`, tree batching, the batch-invariant GEMM table of this GPU) against
# upstream v1.5.0 fast (tools/reference/bench_requests.py), and the sections of the basal-rs forward per length
# (`basal profile`). Models: CONTEXT_MODELS (default "4.5B max"). Run in /work after `push` and `sync tools/bench
# tools/reference tools/perf .baseline/upstream-1.5`. Writes out/context/.
set -e
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
UP=.baseline/upstream-1.5
if [ ! -x .venv-up/bin/python ]; then
  curl --fail -sSL -o /tmp/uv.tgz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf /tmp/uv.tgz -C /tmp
  (cd $UP && UV_PROJECT_ENVIRONMENT=/work/.venv-up /tmp/uv-x86_64-unknown-linux-gnu/uv sync --frozen --no-dev)
fi
PY=.venv-up/bin/python
O=out/context
mkdir -p $O
B=basal-dev
up() {
  for _ in $(seq 3600); do
    curl --fail-with-body --silent "http://127.0.0.1:$1/health" > /dev/null && return 0
    pgrep -f "$2" > /dev/null || return 1
    sleep 1
  done
  return 1
}
nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit --format=csv,noheader > $O/gpu.txt
snap() { echo /data/hf/hub/models--Remek--$1/snapshots/$2; }
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c
                [mini]=1978d0705ce09cd2d7d8d3e87b304468121e7255)
for s in ${CONTEXT_MODELS:-4.5B max}; do
  m=basal-1.5-$s; dir=$(snap $m ${REV[$s]})
  L=$O/ladder-$s
  (cd tools/bench && python3 make_context_ladder.py --model $m --out /work/$L)
  # the GEMM table of this GPU (gemm_table: auto at the first server start)
  T=/data/basal/.cache/gemm/$m-f16.json
  if [ ! -f $T ]; then
    $B --color never serve --model $dir --addr 127.0.0.1:8100 > $O/serve-$s.log 2>&1 &
    up 8100 "[b]asal-cuda --color never serve" || true
    pkill -INT -f "[b]asal-cuda --color never serve" || true; sleep 5
  fi
  echo "=== $s basal-rs requests $(date +%H:%M:%S)"
  $B --color never bench-requests --model $dir --gemm-table $T --requests $L/requests.jsonl --reps 3 \
    --out $O/requests-rust-$s.json > $O/requests-rust-$s.log 2>&1
  echo "=== $s basal-rs profile $(date +%H:%M:%S)"
  for r in $L/ref-*; do
    n=${r##*-}
    $B --color never profile --model $dir --gemm-table $T --reference $r --n 5 > $O/profile-$s-$n.json 2> $O/profile-$s-$n.log || true
  done
  echo "=== $s upstream requests $(date +%H:%M:%S)"
  BASAL_UPSTREAM=$UP $PY tools/reference/bench_requests.py --mode fast --model $dir --requests $L/requests.jsonl --reps 2 \
    --out $O/requests-upstream-$s.json > $O/requests-upstream-$s.log 2>&1 || true
done
echo DONE
