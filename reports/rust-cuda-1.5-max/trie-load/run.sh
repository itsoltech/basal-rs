#!/bin/bash
# Paired HTTP load test (ABBA) of the three-level row packing (target-old) and the trie packing (target), basal-1.5-max,
# batch-invariant GEMM table, default server options. Run from the repository root inside the CUDA container.
O=reports/rust-cuda-1.5-max/trie-load
T=reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json
PY=.baseline/upstream/.venv/bin/python
k=0
for bin in old new new old; do
  k=$((k+1))
  B=./target/release/basal; [ $bin = old ] && B=./target-old/release/basal
  $B serve --model .models/basal-1.5-max --gemm-table $T --addr 127.0.0.1:8100 > /tmp/serve-$bin.log 2>&1 &
  for i in $(seq 180); do curl --fail-with-body --silent http://127.0.0.1:8100/health >/dev/null && break; sleep 1; done
  L="$PY tools/bench/loadtest.py --gpu --model basal-1.5-max --reference reports/reference-1.5-max-fp32 --url http://127.0.0.1:8100/v1/systemone"
  $L --n-seq 40 --n-conc 200 --concurrency 1 8 32 --out $O/bench-$k-$bin.json > /dev/null 2>&1
  $L --requests tools/reference/requests_same_state.jsonl --n-seq 28 --n-conc 140 --concurrency 1 8 32 --out $O/same_state-$k-$bin.json > /dev/null 2>&1
  $L --requests tools/bench/shared_states.jsonl --n-seq 10 --n-conc 40 --concurrency 1 5 10 --out $O/shared_states-$k-$bin.json > /dev/null 2>&1
  pkill -INT -f "basal serve"; sleep 3
  echo "run $k $bin done"
done
echo LOAD-DONE
