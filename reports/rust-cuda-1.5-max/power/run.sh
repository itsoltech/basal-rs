#!/bin/bash
# Current defaults (grouped batch-invariant GEMM table, two lanes, HRRN) at the GPU power limit in effect: mixed
# workload (sequential, 32 clients, open-loop arrivals), one-question requests over HTTP and long states.
# Usage: run.sh LABEL (results in reports/rust-cuda-1.5-max/power/LABEL). Run from the repository root in the CUDA
# container.
set -e
O=reports/rust-cuda-1.5-max/power/$1
mkdir -p $O
B=./target/release/basal
PY=.baseline/upstream/.venv/bin/python
M="--model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm-equiv/gemm-algos-f16-invariant-groups.json"
nvidia-smi --query-gpu=name,power.limit,power.default_limit,clocks.max.sm --format=csv > $O/gpu.csv
$B bench-requests $M --requests tools/bench/long_states.jsonl --reps 2 --out $O/long.json 2> /dev/null
$B serve $M --addr 127.0.0.1:8100 > $O/serve.log 2>&1 &
for i in $(seq 300); do curl --fail-with-body --silent http://127.0.0.1:8100/health > /dev/null && break; sleep 1; done
L="$PY tools/bench/loadtest.py --gpu --model basal-1.5-max --url http://127.0.0.1:8100/v1/systemone"
$L --reference reports/reference-1.5-max-fp32 --n-seq 44 --n-conc 300 --concurrency 32 --out $O/short.json > /dev/null 2>&1
$L --requests tools/bench/mixed.jsonl --n-warm 44 --n-seq 400 --n-conc 400 --concurrency 32 \
  --rates 0.9 1.3 1.6 2.0 2.4 --out $O/mixed.json > /dev/null 2>&1
pkill -INT -f "basal serve"; sleep 3
echo POWER-DONE
