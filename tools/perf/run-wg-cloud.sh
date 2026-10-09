#!/bin/bash
# Kernel variants of one build (WG_VARIANTS, default "tc wg", the first the reference) on one rented
# GPU (tools/cloud/basal-cloud.py), for the models WG_MODELS (default "4.5B max"):
#   - results: exports of the 44 basal-bench items (tree and single) of every variant compared with the first one's,
#     single against tree, and the first and last variant against the FP32 upstream reference
#   - time: `basal bench` and the context ladder, REPS repetitions in rotating order, SM clock locked when allowed
# GEMM tables: the built-in ones of this GPU (crates/basal-cli/gemm-tables), else generated. Run in /work after pushing
# the build and `sync tools/bench tools/perf crates/basal-cli/gemm-tables reports/reference-basal-1.5-4.5B-fp32
# reports/reference-1.5-max-fp32`. Writes WG_OUT (default out/wg), which must not exist.
# Variant syntax: attention[:norm[:silu_block[:wgp_load]]], e.g. "auto:::original auto:::coalesced".
# Empty norm/block uses current defaults; use "auto::1024" to pin the original SiLU launch.
set -euo pipefail
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
O=${WG_OUT:-out/wg}
if [ -e "$O" ]; then echo "output already exists: $O" >&2; exit 1; fi
mkdir -p "$O"
SMI=
cleanup() {
  if [ -n "$SMI" ]; then kill "$SMI" 2>/dev/null || true; fi
  { sudo -n nvidia-smi -rgc || true; } > "$O/unlock.txt" 2>&1
}
trap cleanup EXIT
nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit,clocks.max.sm --format=csv,noheader > "$O/gpu.txt"
basal-dev --version > "$O/version.txt" 2>&1
sha256sum /opt/basal-dev/basal-cuda > "$O/binary.sha256"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
read -ra VS <<< "${WG_VARIANTS:-tc wg}"
first=${VS[0]}; last=${VS[-1]}
run() {
  local v=$1 att norm block load; shift
  IFS=: read -r att norm block load <<< "$v"
  BASAL_ATT=$att BASAL_NORM=$norm BASAL_SILU_BLOCK=$block BASAL_WGP_LOAD=$load basal-dev --color never "$@"
}
snap() { echo "/data/hf/hub/models--Remek--$1/snapshots/$2"; }
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A REF=([4.5B]=reports/reference-basal-1.5-4.5B-fp32 [max]=reports/reference-1.5-max-fp32)
slug=$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
table() {
  local t=crates/basal-cli/gemm-tables/$slug--basal-1.5-$1--f16--cublaslt120901.json
  if [ ! -f "$t" ]; then
    t=$O/table-$1.json
    [ -f "$t" ] || basal-dev --color never gemm-search --model "$(snap "basal-1.5-$1" "${REV[$1]}")" --invariant --out "$t" \
      > "$O/gemm-$1.log" 2>&1 || return
  fi
  echo "$t"
}
read -ra MODELS <<< "${WG_MODELS:-4.5B max}"
for s in "${MODELS[@]}"; do
  m=basal-1.5-$s; dir=$(snap "$m" "${REV[$s]}"); T=$(table "$s")
  (cd tools/bench && python3 make_context_ladder.py --model "$m" --out "/work/$O/ladder-$s" > /dev/null)
  echo "=== $s results $(date +%T)"
  for v in "${VS[@]}"; do
    for b in tree single; do
      run "$v" export --model "$dir" --gemm-table "$T" --batching "$b" --inputs "${REF[$s]}" --out "$O/export-$s-$v-$b" > "$O/export-$s-$v-$b.log" 2>&1
    done
    if [ "$v" != "$first" ]; then
      printf "%-24s " "$s $first vs $v"
      basal-dev compare --a "$O/export-$s-$first-tree" --b "$O/export-$s-$v-tree" --out "$O/compare-$s-$first-$v.json" | tail -1
    fi
    printf "%-24s " "$s $v single vs tree"
    basal-dev compare --a "$O/export-$s-$v-single" --b "$O/export-$s-$v-tree" --out "$O/compare-$s-$v-single-tree.json" | tail -1
  done
  refs=("$first")
  if [ "$last" != "$first" ]; then refs+=("$last"); fi
  for v in "${refs[@]}"; do
    printf "%-24s " "$s fp32 vs $v"; basal-dev compare --a "${REF[$s]}" --b "$O/export-$s-$v-tree" --out "$O/compare-$s-fp32-$v.json" | tail -1
  done
done
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
{ sudo -n nvidia-smi -lgc "$((MAXC * 8 / 10)),$((MAXC * 8 / 10))" || true; } > "$O/lock.txt" 2>&1
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw --format=csv,noheader -l 1 > "$O/clocks.csv" &
SMI=$!
for s in "${MODELS[@]}"; do
  m=basal-1.5-$s; dir=$(snap "$m" "${REV[$s]}"); T=$(table "$s")
  echo "=== $s time $(date +%T)"
  mkdir -p "$O/$s"
  for rep in $(seq "${REPS:-4}"); do
    order=("${VS[@]}")
    if [ $((rep % 2)) = 0 ]; then
      order=()
      for ((i=${#VS[@]}-1; i>=0; i--)); do order+=("${VS[i]}"); done
    fi
    for v in "${order[@]}"; do
      echo "$(date +%T) $s rep $rep $v" >> "$O/order.txt"
      run "$v" bench --model "$dir" --gemm-table "$T" --reference "${REF[$s]}" --out "$O/$s/bench-$v-$rep.json" > "$O/$s/bench-$v-$rep.log" 2>&1
      run "$v" bench-requests --model "$dir" --gemm-table "$T" --requests "$O/ladder-$s/requests.jsonl" --reps 2 \
        --out "$O/$s/requests-$v-$rep.json" > "$O/$s/requests-$v-$rep.log" 2>&1
    done
  done
done
echo DONE
