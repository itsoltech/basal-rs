#!/bin/bash
# Agreement of basal-1.5-4.5B and basal-1.5-mini with upstream v1.5.0: FP32 reference (eager, CUDA) and the served
# BF16 path of upstream for comparison, prompt/token check, runtime FP16 (default, single and tree batching) and
# FP32 exports compared with the reference. Run from the repository root in the CUDA container.
set -e
O=reports/compat-1.5-small
B=./target/release/basal
UP=.baseline/upstream-1.5
for spec in basal-1.5-4.5B:784a683bfadcc8865238fc0fc74a83b4000269bc basal-1.5-mini:1978d0705ce09cd2d7d8d3e87b304468121e7255; do
  m=${spec%%:*}; rev=${spec#*:}
  R=reports/reference-$m-fp32
  E="--model .models/$m --repo Remek/$m --revision $rev --cases tools/reference/systemone_cases_1.5.jsonl"
  [ -d $R ] || BASAL_UPSTREAM=$UP $UP/.venv/bin/python tools/reference/export_reference.py $E \
    --mode eager --device cuda --dtype float32 --out $R > reports/reference-$m-fp32.log 2>&1
  [ -d reports/reference-$m-bf16 ] || BASAL_UPSTREAM=$UP $UP/.venv/bin/python tools/reference/export_reference.py $E \
    --mode fast --dtype bfloat16 --out reports/reference-$m-bf16 > reports/reference-$m-bf16.log 2>&1
  M="--model .models/$m"
  T="--gemm-table .cache/gemm/$m-f16.json"
  $B check-prompts $M --reference $R --out $O/check-prompts-$m.json | tail -2
  $B export $M $T --inputs $R --out $O/export-$m-f16-single > /dev/null 2>&1
  $B export $M $T --batching tree --inputs $R --out $O/export-$m-f16-tree > /dev/null 2>&1
  $B export $M --dtype f32 --inputs $R --out $O/export-$m-f32 > /dev/null 2>&1
  echo "== $m"
  for v in f32 f16-single f16-tree; do
    printf "%-12s " $v; $B compare --a $R --b $O/export-$m-$v --out $O/compare-upstream-fp32-vs-rust-$m-$v.json | tail -1
  done
  printf "%-12s " up-bf16; $B compare --a $R --b reports/reference-$m-bf16 --out $O/compare-upstream-fp32-vs-upstream-bf16-$m.json | tail -1
done
echo COMPAT-DONE
