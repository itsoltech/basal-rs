#!/bin/bash
# Batch-invariant GEMM table with one algorithm per M class from a group of bitwise-identical algorithms
# (gemm-algos-f16-invariant-groups.json) against one algorithm per weight shape (gemm-retune ladder table): bitwise
# invariance of the new table, comparison with the old one, agreement with the
# upstream FP32 reference, paired single-decision latency (ABBA), long states and HTTP load. Run from the repository
# root in the CUDA container.
set -e
O=reports/rust-cuda-1.5-max/gemm-equiv
B=./target/release/basal
M="--model .models/basal-1.5-max"
OLD="--gemm-table reports/rust-cuda-1.5-max/gemm-retune/gemm-algos-f16-invariant-ladder.json"
NEW="--gemm-table $O/gemm-algos-f16-invariant-groups.json"
R=reports/reference-1.5-max-fp32
PY=.baseline/upstream/.venv/bin/python
for b in single tree; do $B export $M $NEW --batching $b --inputs $R --out $O/export-new-$b > /dev/null 2>&1; done
$B compare --a $O/export-new-single --b $O/export-new-tree --out $O/compare-new-single-vs-tree.json | tail -1
$B export $M $OLD --inputs $R --out $O/export-old-single > /dev/null 2>&1
$B compare --a $O/export-old-single --b $O/export-new-single --out $O/compare-old-vs-new.json | tail -1
$B compare --a $R --b $O/export-new-single --out $O/compare-upstream-fp32-vs-new.json | tail -1
$PY tools/bench/ab.py --out $O/ab-bench --rounds 2 \
  --a "$B bench $M $OLD --reference $R" --b "$B bench $M $NEW --reference $R" | tail -3
$B bench-requests $M $OLD --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-old.json 2> $O/long-old.log
$B bench-requests $M $NEW --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-new.json 2> $O/long-new.log
k=0
for t in old new new old; do
  k=$((k + 1))
  T=$OLD; [ $t = new ] && T=$NEW
  $B serve $M $T --addr 127.0.0.1:8100 > /tmp/serve-gemm-$t.log 2>&1 &
  for i in $(seq 300); do curl --fail-with-body --silent http://127.0.0.1:8100/health >/dev/null && break; sleep 1; done
  $PY tools/bench/loadtest.py --gpu --model basal-1.5-max --reference $R --n-seq 40 --n-conc 300 --concurrency 32 \
    --url http://127.0.0.1:8100/v1/systemone --out $O/load-$k-$t.json > /dev/null 2>&1
  pkill -INT -f "basal serve"; sleep 3
done
echo AB-DONE
