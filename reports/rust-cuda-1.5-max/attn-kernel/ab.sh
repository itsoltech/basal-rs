#!/bin/bash
# Kernel iteration A/B on long states: A = ./target-old (previous commit), B = ./target (working tree). Bitwise
# comparison of the 44 basal-bench items and the basal-1.5 cases, then bench-requests on tools/bench/long_states.jsonl
# in ABBA order. Usage: ab.sh NAME (results in reports/rust-cuda-1.5-max/attn-kernel/NAME). Run from the repository
# root in the CUDA container.
set -e
O=reports/rust-cuda-1.5-max/attn-kernel/$1
mkdir -p $O
M="--model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm-retune/gemm-algos-f16-invariant-ladder.json"
for v in old new; do
  B=./target/release/basal; [ $v = old ] && B=./target-old/release/basal
  $B export $M --inputs reports/reference-1.5-max-fp32 --out $O/export-$v > /dev/null 2>&1
  $B export $M --inputs reports/reference-1.5-max-fp32-cases2 --out $O/export-cases2-$v > /dev/null 2>&1
done
./target/release/basal compare --a $O/export-old --b $O/export-new --out $O/compare-old-vs-new.json | tail -1
./target/release/basal compare --a $O/export-cases2-old --b $O/export-cases2-new --out $O/compare-cases2-old-vs-new.json > /dev/null
./target/release/basal compare --a reports/reference-1.5-max-fp32 --b $O/export-new --out $O/compare-upstream-fp32-vs-new.json | tail -1
k=0
for v in old new new old; do
  k=$((k + 1))
  B=./target/release/basal; [ $v = old ] && B=./target-old/release/basal
  $B bench-requests $M --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-$k-$v.json 2> /dev/null
done
python3 - $O <<'PY'
import json, sys, glob, statistics
o = sys.argv[1]
runs = {}
for f in sorted(glob.glob(f"{o}/long-*.json")):
    v = f.rsplit("-", 1)[1][:-5]
    for r in json.load(open(f))["requests"]:
        runs.setdefault(r["id"], {}).setdefault(v, []).append(r["median_ms"])
for rid, d in runs.items():
    a, b = statistics.mean(d["old"]), statistics.mean(d["new"])
    print(f"{rid:16} old {a:8.1f} ms  new {b:8.1f} ms  ratio {b / a:.3f}")
PY
