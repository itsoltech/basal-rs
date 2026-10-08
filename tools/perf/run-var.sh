#!/bin/bash
# timing of builds / attention variants on this GPU, basal-1.5-4.5B, order rotated every repetition (ABBA...):
#   VARIANTS="A=out/basal-A: B=out/basal-B:tc ..." (name=binary:BASAL_ATT)
# exports of the 44 basal-bench items (tree) compared with the first variant, `basal bench`, context ladder;
# GPU clocks and temperature are logged every second. Run in /work of the CUDA dev container (tools/cuda/Dockerfile,
# models in .models, GEMM table GEMM_TABLE); writes out/att/$TAG. Lock the GPU clock for the run (nvidia-smi -lgc,
# then -rgc) and leave out the first repetition (tools/perf/variant_table.py --skip 1).
cd /work
export BASAL_NO_UPDATE_CHECK=1
M=/work/.models/basal-1.5-4.5B
T=${GEMM_TABLE:-out/prof/gemm/basal-1.5-4.5B-f16.json}
REF=reports/reference-basal-1.5-4.5B-fp32
O=out/att/${TAG:-run}
L=${LADDER:-out/prof/ladder-full/requests.jsonl}
mkdir -p $O
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader > $O/gpu.txt
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -l 1 > $O/clocks.csv &
SMI=$!
read -ra VS <<< "$VARIANTS"
run() { local spec=${1#*=}; shift; BASAL_ATT=${spec#*:} ${spec%%:*} --color never "$@"; }
first=${VS[0]%%=*}
for v in "${VS[@]}"; do
  n=${v%%=*}
  run $v export --model $M --gemm-table $T --batching tree --inputs $REF --out $O/export-$n > /dev/null 2>&1
  [ $n = $first ] || { printf "%-8s " $n; target/release/basal compare --a $O/export-$first --b $O/export-$n --out $O/compare-$first-$n.json | tail -1; }
done
for rep in $(seq ${REPS:-4}); do
  order=("${VS[@]}")
  [ $((rep % 2)) = 0 ] && order=($(printf "%s\n" "${VS[@]}" | tac))
  for v in "${order[@]}"; do
    n=${v%%=*}
    echo "$(date +%T) rep $rep $n" >> $O/order.txt
    run $v bench --model $M --gemm-table $T --reference $REF --out $O/bench-$n-$rep.json > /dev/null 2>&1
    run $v bench-requests --model $M --gemm-table $T --requests $L --reps 2 --out $O/requests-$n-$rep.json > /dev/null 2>&1
  done
done
kill $SMI
echo DONE
