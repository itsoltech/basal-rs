#!/bin/bash
# Generate missing f16 tables with the installed release on a rented GPU.
# Sync this runner, tools/bench, the three 44-case FP32 references and
# .cache/cloud/refs/reference-basal-1.5-{mini,4.5B,max}-fp32 first.
# GEMM_MODELS selects models that fit the GPU; GEMM_OUT must be new.
# Raw exports stay in the ignored local cache after retrieval.
set -euo pipefail
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
export LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -d /usr/local/cuda-12.9/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
fi
BIN=/opt/basal-release/libexec/basal/basal-cuda
O=${GEMM_OUT:-out/gemm-tables}
if [ -e "$O" ]; then echo "output already exists: $O" >&2; exit 1; fi
mkdir -p "$O"
"$BIN" --version > "$O/version.txt"
sha256sum "$BIN" > "$O/binary.sha256"
nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit --format=csv,noheader > "$O/gpu.txt"
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu --format=csv,noheader -l 1 > "$O/clocks.csv" &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(printf '%s' "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
declare -A REV=([mini]=1978d0705ce09cd2d7d8d3e87b304468121e7255 [4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A REF=([mini]=reports/reference-basal-1.5-mini-fp32 [4.5B]=reports/reference-basal-1.5-4.5B-fp32 [max]=reports/reference-1.5-max-fp32)
for s in ${GEMM_MODELS:-mini 4.5B max}; do
  model=basal-1.5-$s
  dir=/data/hf/hub/models--Remek--$model/snapshots/${REV[$s]}
  D=$O/$s
  mkdir -p "$D"
  echo "$(date -u +%FT%TZ) $s search"
  t0=$(date +%s)
  "$BIN" --color never gemm-search --model "$dir" --dtype f16 --invariant --out "$D/raw-table.json" > "$D/search.log" 2>&1
  printf '%s\n' "$(( $(date +%s) - t0 ))" > "$D/search-seconds.txt"
  version=$(jq -r '.cublaslt_version' "$D/raw-table.json")
  table=$D/$slug--$model--f16--cublaslt$version.json
  jq --arg model "$model" --arg source "reports/gemm-tables-2026-10-10/$slug" \
    '. + {model: $model, source: $source}' "$D/raw-table.json" > "$table"
  jq -e '.invariant == true and .gemm_search_version == 2 and .dtype == "f16" and .cublaslt_version == 120901 and (.entries | length) > 0' "$table" > /dev/null
  echo "$(date -u +%FT%TZ) $s invariance"
  for mode in single tree budget; do
    "$BIN" --color never export --model "$dir" --gemm-table "$table" --batching "$mode" \
      --inputs "${REF[$s]}" --out "$D/export-$mode" > "$D/export-$mode.log" 2>&1
    jq -se 'length == 44 and all(.error == null)' "$D/export-$mode/bench.jsonl" > /dev/null
  done
  for mode in tree budget; do
    "$BIN" --color never compare --a "$D/export-single" --b "$D/export-$mode" \
      --out "$D/compare-single-vs-$mode.json" > "$D/compare-single-vs-$mode.log" 2>&1
    for file in bench.jsonl systemone.jsonl; do
      cmp "$D/export-single/$file" "$D/export-$mode/$file"
    done
  done
  echo "$(date -u +%FT%TZ) $s FP32"
  "$BIN" --color never compare --a "${REF[$s]}" --b "$D/export-tree" \
    --out "$D/compare-fp32-44.json" > "$D/compare-fp32-44.log" 2>&1
  R=.cache/cloud/refs/reference-$model-fp32
  "$BIN" --color never export --model "$dir" --gemm-table "$table" --batching tree \
    --inputs "$R" --out "$D/export-decisions" > "$D/export-decisions.log" 2>&1
  jq -se 'length == 900 and all(.error == null)' "$D/export-decisions/bench.jsonl" > /dev/null
  "$BIN" --color never compare --a "$R" --b "$D/export-decisions" \
    --out "$D/compare-fp32-900.json" > "$D/compare-fp32-900.log" 2>&1
  echo "$(date -u +%FT%TZ) $s timing"
  "$BIN" --color never bench --model "$dir" --gemm-table "$table" --reference "${REF[$s]}" \
    --out "$D/bench.json" > "$D/bench.log" 2>&1
  python3 tools/bench/make_context_ladder.py --model "$model" --out "$D/ladder" --sizes 512 1792 4096
  "$BIN" --color never bench-requests --model "$dir" --gemm-table "$table" --requests "$D/ladder/requests.jsonl" \
    --reps 3 --out "$D/requests.json" > "$D/requests.log" 2>&1
  touch "$D/VALIDATED"
  echo "$(date -u +%FT%TZ) $s VALIDATED"
done
touch "$O/DONE"
echo DONE
