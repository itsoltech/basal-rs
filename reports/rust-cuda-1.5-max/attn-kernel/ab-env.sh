#!/bin/bash
# Kernel A/B within one binary: A = BASAL_ATT=$2, B = BASAL_ATT=$3. Bitwise comparison of the 44 basal-bench items
# and the basal-1.5 cases, agreement with the upstream FP32 reference, then bench-requests on
# tools/bench/long_states.jsonl in ABBA order. Usage: ab-env.sh NAME A B. Run from the repository root in the CUDA
# container.
set -e
O=reports/rust-cuda-1.5-max/attn-kernel/$1
mkdir -p $O
B=./target/release/basal
M="--model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm-retune/gemm-algos-f16-invariant-ladder.json"
for v in $2 $3; do
  BASAL_ATT=$v $B export $M --inputs reports/reference-1.5-max-fp32 --out $O/export-$v > /dev/null 2>&1
  BASAL_ATT=$v $B export $M --inputs reports/reference-1.5-max-fp32-cases2 --out $O/export-cases2-$v > /dev/null 2>&1
done
$B compare --a $O/export-$2 --b $O/export-$3 --out $O/compare-$2-vs-$3.json | tail -1
$B compare --a $O/export-cases2-$2 --b $O/export-cases2-$3 --out $O/compare-cases2-$2-vs-$3.json > /dev/null
$B compare --a reports/reference-1.5-max-fp32 --b $O/export-$3 --out $O/compare-upstream-fp32-vs-$3.json | tail -1
k=0
for v in $2 $3 $3 $2; do
  k=$((k + 1))
  BASAL_ATT=$v $B bench-requests $M --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-$k-$v.json 2> /dev/null
done
python3 - $O $2 $3 <<'PY'
import json, sys, glob, statistics
o, a, b = sys.argv[1:4]
runs = {}
for f in sorted(glob.glob(f"{o}/long-*.json")):
    v = f.split("/long-", 1)[1].split("-", 1)[1][:-5]
    for r in json.load(open(f))["requests"]:
        runs.setdefault(r["id"], {}).setdefault(v, []).append(r["median_ms"])
for rid, d in runs.items():
    x, y = statistics.mean(d[a]), statistics.mean(d[b])
    print(f"{rid:16} {a} {x:8.1f} ms  {b} {y:8.1f} ms  ratio {y / x:.3f}")
PY
