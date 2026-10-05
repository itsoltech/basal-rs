#!/bin/bash
# Tensor-core attention with 32 query tokens (128 rows, 8 warps) per block instead of 16, Q fragments in registers:
# half the K/V reads; bitwise comparison with the previous kernel and latency on long states. Run from the repository root in
# the CUDA container after building the variant.
set -e
O=reports/rust-cuda-1.5-max/attn-bm128
P=reports/rust-cuda-1.5-max/attn-precision
B=./target/release/basal
M="--model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json"
$B export $M --inputs reports/reference-1.5-max-fp32 --out $O/export-tc > /dev/null 2>&1
$B compare --a $P/export-tc --b $O/export-tc --out $O/compare-previous-vs-new.json | tail -1
$B export $M --inputs reports/reference-1.5-max-fp32-cases2 --out $O/export-cases2-tc > /dev/null 2>&1
$B compare --a $P/export-cases2-tc --b $O/export-cases2-tc --out $O/compare-cases2-previous-vs-new.json > /dev/null
$B bench-requests $M --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-tc.json 2> $O/long-tc.log
echo BM128-DONE
