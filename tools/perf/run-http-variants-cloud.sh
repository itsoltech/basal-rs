#!/bin/bash
# HTTP A/B/B/A of attention variants in one build, sequential servers on a rented GPU.
# HTTP_REQUESTS contains JSONL {id, request, class}; loadtest overrides request.model.
# Samples full answers, latency, throughput, power and VRAM. Fails on HTTP errors or
# changed answers within/across runs. Fixed arrival rates are shared by both variants.
# Sync tools/bench, this file, requests and GEMM tables before running in /work.
set -euo pipefail
cd /work
: "${HTTP_REQUESTS:?mixed request file required}"
O=${HTTP_OUT:-out/http-variants}
if [ -e "$O" ]; then echo "output already exists: $O" >&2; exit 1; fi
mkdir -p "$O"
SERVER=
cleanup() {
  if [ -n "$SERVER" ]; then
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
}
trap cleanup EXIT
sha256sum /opt/basal-dev/basal-cuda > "$O/binary.sha256"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader > "$O/gpu.txt"
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A RATE=([4.5B]="${HTTP_RATE_45B:-11}" [max]="${HTTP_RATE_MAX:-5}")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
read -ra MODELS <<< "${HTTP_MODELS:-4.5B max}"
read -ra ORDER <<< "${HTTP_ORDER:-original sw128 sw128 original}"
for s in "${MODELS[@]}"; do
  rep=0
  for variant in "${ORDER[@]}"; do
    rep=$((rep + 1))
    out=$O/$s-$rep-$variant
    mkdir -p "$out"
    echo "$(date +%T) $s $rep $variant" | tee -a "$O/order.txt"
    launch=(env "BASAL_WGP_LOAD=$variant")
    if [ "$variant" = default ]; then launch=(env -u BASAL_WGP_LOAD); fi
    "${launch[@]}" basal-dev --color never serve \
      --model "/data/hf/hub/models--Remek--basal-1.5-$s/snapshots/${REV[$s]}" \
      --gemm-table "crates/basal-cli/gemm-tables/$slug--basal-1.5-$s--f16--cublaslt120901.json" \
      --addr 127.0.0.1:18100 > "$out/server.log" 2>&1 &
    SERVER=$!
    ready=false
    for ((attempt=0; attempt<90; attempt++)); do
      kill -0 "$SERVER"
      if curl --fail --silent http://127.0.0.1:18100/health > /dev/null; then ready=true; break; fi
      sleep 1
    done
    if [ "$ready" != true ]; then echo "server startup timed out" >&2; exit 1; fi
    timeout -k 10 600 python3 tools/bench/loadtest.py --gpu --model "basal-1.5-$s" \
      --requests "$HTTP_REQUESTS" --n-warm 44 --n-seq 44 --n-conc 400 \
      --concurrency 16 32 --rates "${RATE[$s]}" --url http://127.0.0.1:18100/v1/systemone \
      --out "$out/mixed.json" --answers-out "$out/answers.json" > "$out/load.log" 2>&1
    cleanup
    SERVER=
    jq -e '.phases | length == 4 and all(.errors == 0 and .answers_differing_vs_first == 0
      and .max_prob_diff_vs_first_answer == 0)' "$out/mixed.json" > /dev/null
    jq -S 'sort_by(.id)' "$out/answers.json" > "$out/answers-sorted.json"
    if [ "$rep" = 1 ]; then first=$out; fi
    cmp "$first/answers-sorted.json" "$out/answers-sorted.json"
    echo "ANSWERS_EQUAL $s $rep $variant" | tee -a "$O/parity.txt"
  done
done
echo DONE
