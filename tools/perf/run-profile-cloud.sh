#!/bin/bash
# Profile the pushed 4.5B build on a rented GPU, after sync tools/perf tools/bench and GEMM tables/reference.
# Nsight must already be installed in a version compatible with the driver. This script never installs packages.
# PROF_OUT must be new; PROF_LADDER / PROF_TABLE can reuse campaign inputs instead of out/wg.
# Systems is the default. PROF_NCU=1 additionally requires PROF_LAUNCH_SKIP, selected from the actual trace
# (matching attention launches, not all CUDA launches). Raw traces may contain machine metadata: keep them out of git.
set -euo pipefail
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
O=${PROF_OUT:-out/prof}
L=${PROF_LADDER:-out/wg/ladder-4.5B/requests.jsonl}
SECONDS_LIMIT=${PROF_TIMEOUT:-300}
NCU_ENABLED=${PROF_NCU:-0}
[[ "$SECONDS_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "PROF_TIMEOUT must be positive seconds" >&2; exit 1; }
[[ "$NCU_ENABLED" == 0 || "$NCU_ENABLED" == 1 ]] || { echo "PROF_NCU must be 0 or 1" >&2; exit 1; }
for tool in nsys jq timeout nvidia-smi basal-dev; do
  command -v "$tool" > /dev/null || { echo "required tool missing: $tool" >&2; exit 1; }
done
if [ "$NCU_ENABLED" = 1 ]; then
  command -v ncu > /dev/null || { echo "required tool missing: ncu" >&2; exit 1; }
  [[ "${PROF_LAUNCH_SKIP:-}" =~ ^[0-9]+$ ]] || {
    echo "set PROF_LAUNCH_SKIP from a verified trace before enabling Compute" >&2; exit 1;
  }
fi
[ ! -e "$O" ] || { echo "output already exists: $O" >&2; exit 1; }
[ -s "$L" ] || { echo "missing ladder: $L" >&2; exit 1; }
dir=/data/hf/hub/models--Remek--basal-1.5-4.5B/snapshots/784a683bfadcc8865238fc0fc74a83b4000269bc
REF=reports/reference-basal-1.5-4.5B-fp32
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
T=${PROF_TABLE:-crates/basal-cli/gemm-tables/$slug--basal-1.5-4.5B--f16--cublaslt120901.json}
[ -s "$T" ] || { echo "missing GEMM table: $T (set PROF_TABLE)" >&2; exit 1; }
[ -s "$REF/bench.jsonl" ] || { echo "missing reference: $REF" >&2; exit 1; }
[ -d "$dir" ] || { echo "model not downloaded" >&2; exit 1; }
mkdir -p "$(dirname "$O")"
mkdir "$O"
nsys --version > "$O/nsight-versions.txt" 2>&1
if [ "$NCU_ENABLED" = 1 ]; then ncu --version >> "$O/nsight-versions.txt" 2>&1; fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit --format=csv,noheader > "$O/gpu.txt"
sha256sum /opt/basal-dev/basal-cuda > "$O/binary.sha256"
printf 'BASAL_ATT=%s\nBASAL_NORM=%s\nBASAL_SILU_BLOCK=%s\nBASAL_WGP_LOAD=%s\n' \
  "${BASAL_ATT:-auto}" "${BASAL_NORM:-}" "${BASAL_SILU_BLOCK:-}" "${BASAL_WGP_LOAD:-default}" > "$O/variants.txt"
for id in ctx-512-q1 ctx-1792-q1 ctx-4096-q1 ctx-16384-q1; do
  # Exact ID lookup independent of JSON whitespace; missing/duplicate IDs fail before starting the profiler.
  jq -sce --arg id "$id" 'map(select(.id == $id)) | if length == 1 then .[0] else error("expected one ladder ID") end' \
    "$L" > "$O/$id.jsonl"
done
profile() {
  local id=$1; shift
  timeout --kill-after=10 "$SECONDS_LIMIT" nsys profile -t cuda --sample=none --cpuctxsw=none \
    --cuda-memory-usage=false --cuda-event-trace=false -o "$O/$id" basal-dev --color never "$@" > "$O/$id.log" 2>&1
  timeout --kill-after=10 "$SECONDS_LIMIT" nsys export -t sqlite -o "$O/$id.sqlite" "$O/$id.nsys-rep" \
    > "$O/$id-export.log" 2>&1
  python3 tools/perf/nsys_forward.py "$O/$id.sqlite" > "$O/$id.md" 2> "$O/$id-analysis.log"
  # Keep the native timeline/API reports: burst gaps alone do not prove a CPU launch bottleneck.
  timeout --kill-after=10 "$SECONDS_LIMIT" nsys stats --report cuda_gpu_trace,cuda_gpu_kern_sum,cuda_api_sum \
    --format csv --output "$O/$id" "$O/$id.sqlite" > "$O/$id-stats.log" 2>&1
}
profile bench bench --model "$dir" --gemm-table "$T" --reference "$REF" --out "$O/bench.json" --lat-n 20
for id in ctx-512-q1 ctx-1792-q1 ctx-4096-q1 ctx-16384-q1; do
  profile "$id" bench-requests --model "$dir" --gemm-table "$T" --requests "$O/$id.jsonl" --reps 4 \
    --out "$O/$id.json"
done
if [ "$NCU_ENABLED" = 1 ]; then
  NCU=$(command -v ncu)
  read -ra ATT <<< "${PROF_ATT:-auto}"
  for v in "${ATT[@]}"; do
    [[ "$v" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid attention variant" >&2; exit 1; }
    for id in ctx-4096-q1 ctx-16384-q1; do
      { sudo -n env "PATH=$PATH" HF_HOME="$HF_HOME" BASAL_HOME="$BASAL_HOME" BASAL_NO_UPDATE_CHECK=1 \
        BASAL_ATT="$v" BASAL_NORM="${BASAL_NORM:-}" BASAL_SILU_BLOCK="${BASAL_SILU_BLOCK:-}" \
        BASAL_WGP_LOAD="${BASAL_WGP_LOAD:-default}" \
        LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib timeout --kill-after=10 "$SECONDS_LIMIT" "$NCU" \
        --set basic -k regex:attn_tree -c 1 --launch-skip "$PROF_LAUNCH_SKIP" -o "$O/ncu-$id-$v" \
        /opt/basal-dev/basal-cuda --color never bench-requests --model "$dir" --gemm-table "$T" \
        --requests "$O/$id.jsonl" --reps 1 --out "$O/ncu-$id-$v.json"; } > "$O/ncu-$id-$v.log" 2>&1
      timeout --kill-after=10 "$SECONDS_LIMIT" "$NCU" --import "$O/ncu-$id-$v.ncu-rep" --page details \
        > "$O/ncu-$id-$v.txt" 2>&1
    done
  done
fi
echo DONE
