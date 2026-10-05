#!/bin/bash
cd /work
mkdir -p reports/rust-cuda-1.5-max/sweep-concurrency
for table in invariant tuned; do
  T=reports/rust-cuda-1.5-max/gemm/gemm-algos-f16.json; [ $table = invariant ] && T=reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json
  for budget in 2048 4096 8192 16384; do
    ./target/release/basal serve --model .models/basal-1.5-max --gemm-table $T --max-batch-tokens $budget --addr 127.0.0.1:8100 > /tmp/serve.log 2>&1 &
    for i in $(seq 180); do curl --fail-with-body --silent http://127.0.0.1:8100/health >/dev/null && break; sleep 1; done
    .baseline/upstream/.venv/bin/python tools/bench/loadtest.py --gpu --model basal-1.5-max --reference reports/reference-1.5-max-fp32 --n-seq 40 --n-conc 300 --concurrency 1 8 32 64 --url http://127.0.0.1:8100/v1/systemone --out reports/rust-cuda-1.5-max/sweep-concurrency/$table-$budget.json > /dev/null 2>&1
    pkill -INT -f "basal serve"; sleep 3
    echo "$table $budget done"
  done
done
echo SWEEP-DONE
