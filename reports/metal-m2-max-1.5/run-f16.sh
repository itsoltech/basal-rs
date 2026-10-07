#!/bin/bash
# Upstream MLX in float16 (the same precision as the runtime; bf16 is the served upstream path):
# an export for the agreement with the upstream FP32 reference and the basal-bench methodology paired with the runtime
# f16 in ABBA order (runtime 3, MLX f16 1, MLX f16 2, runtime 4). Upstream 1.5 loads bf16 weights; the model is cast to
# float16 after loading (tools/reference/bench_mlx_dtype.py, export_reference.py --dtype float16). Run from the
# repository root after run.sh:  reports/metal-m2-max-1.5/run-f16.sh basal-1.5-mini
set -e
m=$1
O=reports/metal-m2-max-1.5
B=./target/release/basal
UP=.baseline/upstream-1.5
PY="env BASAL_UPSTREAM=$UP $UP/.venv/bin/python"
PAUSE=${PAUSE:-60}
case $m in
  basal-1.5-max) R=reports/reference-1.5-max-fp32; rev=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c ;;
  basal-1.5-4.5B) R=reports/reference-$m-fp32; rev=784a683bfadcc8865238fc0fc74a83b4000269bc ;;
  basal-1.5-mini) R=reports/reference-$m-fp32; rev=1978d0705ce09cd2d7d8d3e87b304468121e7255 ;;
  *) echo "unknown model $m" >&2; exit 2 ;;
esac
M="--model .models/$m"
cool() {
  sleep "$PAUSE"
  for i in $(seq 60); do pmset -g therm | rg -qv '^(Note: No|$)' || break; sleep 10; done
  echo "$(date +%H:%M:%S) $1: $(pmset -g therm | rg -v '^$' | tr '\n' ' ')" >> $O/therm.log
}

U=$O/upstream-$m-mlx-f16
if [ ! -d $U ]; then
  cool "upstream export $m mlx-f16"
  $PY tools/reference/export_reference.py $M --repo Remek/$m --revision $rev --mode mlx --dtype float16 \
    --cases tools/reference/systemone_cases_1.5.jsonl --out $U > $U.log 2>&1
  printf "%-12s " mlx-f16; $B compare --a $R --b $U --out $O/compare-upstream-fp32-vs-upstream-$m-mlx-f16.json | tail -1
fi
for spec in rust:3 f16:1 f16:2 rust:4; do
  who=${spec%:*}; k=${spec#*:}
  if [ $who = rust ]; then
    f=$O/bench-rust-$m-f16-$k.json
    [ -f $f ] && continue
    cool "bench rust $m $k"
    $B bench $M --reference $R --out $f 2>&1 | tail -1
  else
    f=$O/bench-upstream-$m-mlx-f16-$k.json
    [ -f $f ] && continue
    cool "bench upstream f16 $m $k"
    $PY tools/reference/bench_mlx_dtype.py $M --dtype float16 --out $f 2>&1 | tail -1
  fi
done
echo "F16-DONE $m"
