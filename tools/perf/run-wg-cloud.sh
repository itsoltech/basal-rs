#!/bin/bash
# Attention kernels of one build (BASAL_ATT values WG_VARIANTS, default "tc wg", the first the reference) on one rented
# GPU (tools/cloud/basal-cloud.py), for the models WG_MODELS (default "4.5B max"):
#   - results: exports of the 44 basal-bench items (tree and single) of every variant compared with the first one's,
#     single against tree, and the first and last variant against the FP32 upstream reference
#   - time: `basal bench` and the context ladder, REPS repetitions in rotating order, SM clock locked when allowed
# GEMM tables: the built-in ones of this GPU (crates/basal-cli/gemm-tables), else generated. Run in /work after pushing
# the build and `sync tools/bench tools/perf crates/basal-cli/gemm-tables reports/reference-basal-1.5-4.5B-fp32
# reports/reference-1.5-max-fp32`. Writes out/wg/.
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
O=out/wg
mkdir -p $O
nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv,noheader > $O/gpu.txt
basal-dev --version > $O/version.txt 2>&1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
read -ra VS <<< "${WG_VARIANTS:-tc wg}"
first=${VS[0]}; last=${VS[-1]}
run() { local v=$1; shift; BASAL_ATT=$v basal-dev --color never "$@"; }
snap() { echo /data/hf/hub/models--Remek--$1/snapshots/$2; }
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A REF=([4.5B]=reports/reference-basal-1.5-4.5B-fp32 [max]=reports/reference-1.5-max-fp32)
slug=$(echo "$GPU" | tr 'A-Z' 'a-z' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
table() {
  local t=crates/basal-cli/gemm-tables/$slug--basal-1.5-$1--f16--cublaslt120901.json
  if [ ! -f $t ]; then
    t=$O/table-$1.json
    [ -f $t ] || basal-dev --color never gemm-search --model $(snap basal-1.5-$1 ${REV[$1]}) --invariant --out $t \
      > $O/gemm-$1.log 2>&1
  fi
  echo $t
}
for s in ${WG_MODELS:-4.5B max}; do
  m=basal-1.5-$s; dir=$(snap $m ${REV[$s]}); T=$(table $s)
  [ -d $O/ladder-$s ] || (cd tools/bench && python3 make_context_ladder.py --model $m --out /work/$O/ladder-$s > /dev/null)
  echo "=== $s results $(date +%T)"
  for v in "${VS[@]}"; do
    for b in tree single; do
      run $v export --model $dir --gemm-table $T --batching $b --inputs ${REF[$s]} --out $O/export-$s-$v-$b > $O/export-$s-$v-$b.log 2>&1
    done
    if [ $v != $first ]; then
      printf "%-24s " "$s $first vs $v"
      basal-dev compare --a $O/export-$s-$first-tree --b $O/export-$s-$v-tree --out $O/compare-$s-$first-$v.json | tail -1
    fi
    printf "%-24s " "$s $v single vs tree"
    basal-dev compare --a $O/export-$s-$v-single --b $O/export-$s-$v-tree --out $O/compare-$s-$v-single-tree.json | tail -1
  done
  for v in $first $last; do
    printf "%-24s " "$s fp32 vs $v"; basal-dev compare --a ${REF[$s]} --b $O/export-$s-$v-tree --out $O/compare-$s-fp32-$v.json | tail -1
  done
done
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
sudo -n nvidia-smi -lgc $((MAXC * 8 / 10)),$((MAXC * 8 / 10)) > $O/lock.txt 2>&1 || true
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -l 1 > $O/clocks.csv &
SMI=$!
for s in ${WG_MODELS:-4.5B max}; do
  m=basal-1.5-$s; dir=$(snap $m ${REV[$s]}); T=$(table $s)
  echo "=== $s time $(date +%T)"
  mkdir -p $O/$s
  for rep in $(seq ${REPS:-4}); do
    order=("${VS[@]}")
    [ $((rep % 2)) = 0 ] && order=($(printf "%s\n" "${VS[@]}" | tac))
    for v in "${order[@]}"; do
      echo "$(date +%T) $s rep $rep $v" >> $O/order.txt
      run $v bench --model $dir --gemm-table $T --reference ${REF[$s]} --out $O/$s/bench-$v-$rep.json > /dev/null 2>&1
      run $v bench-requests --model $dir --gemm-table $T --requests $O/ladder-$s/requests.jsonl --reps 2 \
        --out $O/$s/requests-$v-$rep.json > /dev/null 2>&1
    done
  done
done
kill $SMI
sudo -n nvidia-smi -rgc > /dev/null 2>&1 || true
echo DONE
