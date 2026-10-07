#!/bin/bash
# basal-1.5 models on the Apple M2 Max (Metal) against upstream v1.5.0 with MLX: agreement with the upstream FP32
# references (CUDA exports in reports/reference-basal-1.5-*-fp32, reports/reference-1.5-max-fp32), upstream MLX bf16
# and 8-bit (mlx-q8: affine 8-bit, group 64, made from bf16 at load) exports, single-decision latency and throughput
# (basal-bench methodology) of the runtime and of upstream. Run from the repository root:
#   reports/metal-m2-max-1.5/run.sh basal-1.5-mini [steps]
# steps: any of prompts export upstream bench (default: all). Existing outputs are kept; a pause between the GPU runs
# lets the laptop cool down (PAUSE seconds, default 60), `pmset -g therm` is logged before each run.
set -e
m=$1; shift
steps=${*:-prompts export upstream bench}
O=reports/metal-m2-max-1.5
B=./target/release/basal
UP=.baseline/upstream-1.5
PAUSE=${PAUSE:-60}
case $m in
  basal-1.5-max) R=reports/reference-1.5-max-fp32; rev=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c ;;
  basal-1.5-4.5B) R=reports/reference-$m-fp32; rev=784a683bfadcc8865238fc0fc74a83b4000269bc ;;
  basal-1.5-mini) R=reports/reference-$m-fp32; rev=1978d0705ce09cd2d7d8d3e87b304468121e7255 ;;
  *) echo "unknown model $m" >&2; exit 2 ;;
esac
M="--model .models/$m"
# pause; while macOS reports a thermal or performance warning (lines other than "Note: No ..."), wait up to 10 min more
cool() {
  sleep "$PAUSE"
  for i in $(seq 60); do pmset -g therm | rg -qv '^(Note: No|$)' || break; sleep 10; done
  echo "$(date +%H:%M:%S) $1: $(pmset -g therm | rg -v '^$' | tr '\n' ' ')" >> $O/therm.log
}
has() { [[ " $steps " == *" $1 "* ]]; }

if has prompts && [ ! -f $O/check-prompts-$m.json ]; then
  $B check-prompts $M --reference $R --out $O/check-prompts-$m.json | tail -1
fi
if has export; then
  # f32 weights of max (~43 GB) do not fit in 32 GB
  for v in f16-single f16-tree $([ $m = basal-1.5-max ] || echo f32); do
    [ -d $O/export-$m-$v ] && continue
    case $v in f16-single) a="" ;; f16-tree) a="--batching tree" ;; f32) a="--dtype f32" ;; esac
    cool "export $m $v"
    $B export $M $a --inputs $R --out $O/export-$m-$v > $O/export-$m-$v.log 2>&1
    printf "%-12s " $v; $B compare --a $R --b $O/export-$m-$v --out $O/compare-upstream-fp32-vs-rust-$m-$v.json | tail -1
  done
fi
if has upstream; then
  for mode in mlx mlx-q8; do
    U=$O/upstream-$m-$mode  # upstream exports of this machine
    [ -d $U ] && continue
    cool "upstream export $m $mode"
    BASAL_UPSTREAM=$UP $UP/.venv/bin/python tools/reference/export_reference.py $M --repo Remek/$m --revision $rev \
      --mode $mode --dtype bfloat16 --cases tools/reference/systemone_cases_1.5.jsonl --out $U > $U.log 2>&1
    printf "%-12s " $mode; $B compare --a $R --b $U --out $O/compare-upstream-fp32-vs-upstream-$m-$mode.json | tail -1
  done
fi
if has bench; then
  # alternating rounds: runtime f16, upstream mlx + mlx-q8, upstream, runtime
  for k in 1 2; do
    for who in $([ $k = 1 ] && echo "rust upstream" || echo "upstream rust"); do
      if [ $who = rust ]; then
        f=$O/bench-rust-$m-f16-$k.json
        [ -f $f ] && continue
        cool "bench rust $m $k"
        $B bench $M --reference $R --out $f 2>&1 | tail -1
      else
        f=$O/bench-upstream-$m-mlx-$k.json
        [ -f $f ] && continue
        cool "bench upstream $m $k"
        $UP/.venv/bin/basal-bench $M --modes mlx mlx-q8 --n 44 --lat-n 39 --out $f 2>&1 | tail -3
      fi
    done
  done
fi
echo "RUN-DONE $m"
