#!/bin/bash
# Mixed workload, Rust scheduling variants with one binary: arrival order in one lane, HRRN in one lane, HRRN with
# the long lane (default). Same offered loads (open loop, requests/s) for every variant. Run from the repository root
# inside the CUDA container.
O=reports/rust-cuda-1.5-max/mixed-load
PY=.baseline/upstream/.venv/bin/python
B=./target/release/basal
L="$PY tools/bench/loadtest.py --gpu --model basal-1.5-max --requests tools/bench/mixed.jsonl --n-warm 44 --n-seq 0 --n-conc 400 --concurrency 32 --rates 0.9 1.3 1.6"
for v in fifo hrrn lanes; do
  case $v in
    fifo) X="--schedule fifo --long-tokens 0" ;;
    hrrn) X="--schedule hrrn --long-tokens 0" ;;
    lanes) X="" ;;
  esac
  $B serve --model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json --addr 127.0.0.1:8100 $X > $O/serve-$v.log 2>&1 &
  for i in $(seq 300); do curl --fail-with-body --silent http://127.0.0.1:8100/health >/dev/null && break; sleep 1; done
  $L --url http://127.0.0.1:8100/v1/systemone --out $O/sched-$v.json > $O/sched-$v.log 2>&1
  pkill -INT -f "basal serve"; sleep 3
  echo "$v done"
done
echo SCHED-DONE
