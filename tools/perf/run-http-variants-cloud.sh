#!/bin/bash
# HTTP A/B/B/A of attention variants in one build, sequential servers on a rented GPU.
# HTTP_REQUESTS contains JSONL {id, request, class}; loadtest overrides request.model.
# Samples full answers, latency, throughput, power and VRAM. Fails on HTTP errors or
# changed answers within/across runs. Fixed arrival rates are shared by both variants.
# HTTP_ORDER: BASAL_WGP_LOAD values original, coalesced, sw128, or default (unset).
# Sync tools/bench, this file, requests and GEMM tables before running in /work.
set -euo pipefail
cd /work
: "${HTTP_REQUESTS:?mixed request file required}"
O=${HTTP_OUT:-out/http-variants}
NCONC=400
read -ra MODELS <<< "${HTTP_MODELS:-4.5B max}"
read -ra ORDER <<< "${HTTP_ORDER:-original sw128 sw128 original}"
if [ ${#ORDER[@]} -eq 0 ]; then echo "HTTP_ORDER is empty" >&2; exit 1; fi
for variant in "${ORDER[@]}"; do
  case $variant in
    original|coalesced|sw128|default) ;;
    *) echo "HTTP_ORDER: unknown variant '$variant' (original, coalesced, sw128, default)" >&2; exit 1 ;;
  esac
done
# Sorted request IDs. A phase of NCONC requests cycles through the file, so it covers every ID only up to NCONC.
IDS=$(jq -cs --argjson n "$NCONC" '
  if any(.[]; type != "object" or (.id | type) != "string" or (.request | type) != "object") then
    error("every line must be {id: string, request: object, class}")
  elif length == 0 or length > $n then error("\(length) requests, 1 to \($n) supported")
  elif ([.[].id] | unique | length) != length then error("duplicate request IDs")
  else [.[].id] | sort end' "$HTTP_REQUESTS")
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
server_alive() { kill -0 "$SERVER" 2>/dev/null || { echo "server exited, see $out/server.log" >&2; exit 1; }; }
sha256sum /opt/basal-dev/basal-cuda > "$O/binary.sha256"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader > "$O/gpu.txt"
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A RATE=([4.5B]="${HTTP_RATE_45B:-11}" [max]="${HTTP_RATE_MAX:-5}")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
for s in "${MODELS[@]}"; do
  rep=0
  for variant in "${ORDER[@]}"; do
    rep=$((rep + 1))
    out=$O/$s-$rep-$variant
    mkdir -p "$out"
    echo "$(date +%T) $s $rep $variant" | tee -a "$O/order.txt"
    # a leftover server on the port would answer /health instead of this one (curl exit 7: nothing listening)
    probe=0
    curl --silent --output /dev/null --max-time 5 http://127.0.0.1:18100/health || probe=$?
    if [ "$probe" != 7 ]; then echo "127.0.0.1:18100 already in use (curl exit $probe)" >&2; exit 1; fi
    launch=(env "BASAL_WGP_LOAD=$variant")
    if [ "$variant" = default ]; then launch=(env -u BASAL_WGP_LOAD); fi
    "${launch[@]}" basal-dev --color never serve \
      --model "/data/hf/hub/models--Remek--basal-1.5-$s/snapshots/${REV[$s]}" \
      --gemm-table "crates/basal-cli/gemm-tables/$slug--basal-1.5-$s--f16--cublaslt120901.json" \
      --addr 127.0.0.1:18100 > "$out/server.log" 2>&1 &
    SERVER=$!
    ready=false
    for ((attempt=0; attempt<90; attempt++)); do
      server_alive
      if curl --fail --silent --max-time 5 http://127.0.0.1:18100/health > /dev/null; then ready=true; break; fi
      sleep 1
    done
    if [ "$ready" != true ]; then echo "server startup timed out" >&2; exit 1; fi
    server_alive
    timeout -k 10 600 python3 tools/bench/loadtest.py --gpu --model "basal-1.5-$s" \
      --requests "$HTTP_REQUESTS" --n-warm 44 --n-seq 44 --n-conc "$NCONC" \
      --concurrency 16 32 --rates "${RATE[$s]}" --url http://127.0.0.1:18100/v1/systemone \
      --out "$out/mixed.json" --answers-out "$out/answers.json" > "$out/load.log" 2>&1
    server_alive
    cleanup
    SERVER=
    jq -e '.phases | length == 4 and all(.errors == 0 and .answers_differing_vs_first == 0
      and .max_prob_diff_vs_first_answer == 0)' "$out/mixed.json" > /dev/null
    # loadtest stores a 200 response without answers as {}: require a non-empty answer for exactly each request ID
    jq -e --argjson ids "$IDS" 'type == "array" and all(.[]; .answers | type == "object" and length > 0)
      and ([.[].id] | sort) == $ids' "$out/answers.json" > /dev/null \
      || { echo "answers missing, empty or duplicated: $out/answers.json" >&2; exit 1; }
    jq -S 'sort_by(.id)' "$out/answers.json" > "$out/answers-sorted.json"
    if [ "$rep" = 1 ]; then first=$out; fi
    cmp "$first/answers-sorted.json" "$out/answers-sorted.json"
    echo "ANSWERS_EQUAL $s $rep $variant" | tee -a "$O/parity.txt"
  done
done
echo DONE
