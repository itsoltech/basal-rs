#!/bin/bash
# Precision variants of the tensor-core attention kernel (BASAL_ATT=tc | tc-pv1 | tc-qk1 | tc-f16) on basal-1.5-max:
# agreement with the upstream FP32 reference (44 basal-bench items, System One requests, basal-1.5 cases) and latency
# and answers on long states against the runtime's FP32 forward. Run from the repository root in the CUDA container.
set -e
O=reports/rust-cuda-1.5-max/attn-precision
B=./target/release/basal
M="--model .models/basal-1.5-max"
T="--gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json"
# runtime FP32 on long states (reference for 1k-16k tokens, no upstream FP32 at these lengths)
$B bench-requests $M --dtype f32 --requests tools/bench/long_states.jsonl --reps 1 --out $O/long-f32.json 2> $O/long-f32.log
for v in tc tc-pv1 tc-qk1 tc-f16; do
  BASAL_ATT=$v $B export $M $T --inputs reports/reference-1.5-max-fp32 --out $O/export-$v > $O/export-$v.log 2>&1
  $B compare --a reports/reference-1.5-max-fp32 --b $O/export-$v --out $O/compare-upstream-fp32-vs-$v.json > /dev/null
  BASAL_ATT=$v $B export $M $T --inputs reports/reference-1.5-max-fp32-cases2 --out $O/export-cases2-$v > /dev/null 2>&1
  $B compare --a reports/reference-1.5-max-fp32-cases2 --b $O/export-cases2-$v --out $O/compare-cases2-upstream-fp32-vs-$v.json > /dev/null
  BASAL_ATT=$v $B bench-requests $M $T --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-$v.json 2> $O/long-$v.log
  echo "$v done"
done
$B compare --a reports/rust-cuda-1.5-max/export-f16-trie-single --b $O/export-tc --out $O/compare-previous-vs-tc.json > /dev/null
echo PRECISION-DONE
