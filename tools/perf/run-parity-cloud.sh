#!/bin/bash
# Byte-for-byte export comparison of a baseline and the pushed build on a rented GPU.
# Run in /work after syncing this file, GEMM tables and normalized inputs.
# Required: PARITY_INPUT_45B, PARITY_INPUT_MAX; each has 44 bench and 34 System One cases
# (30 valid and the four explicitly listed invalid cases below). No upstream parity claim.
# PARITY_OUT must be new. PARITY_LOAD=default tests the candidate's compiled default.
set -euo pipefail
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
export LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [ -d /usr/local/cuda-12.9/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
fi
O=${PARITY_OUT:-out/parity}
LOAD=${PARITY_LOAD:-sw128}
: "${PARITY_INPUT_45B:?normalized 4.5B export inputs required}"
: "${PARITY_INPUT_MAX:?normalized max export inputs required}"
if [ -e "$O" ]; then echo "output already exists: $O" >&2; exit 1; fi
mkdir -p "$O"
sha256sum /opt/basal-dev/basal-cuda-base /opt/basal-dev/basal-cuda > "$O/binaries.sha256"
printf '%s\n' "$LOAD" > "$O/candidate-load.txt"
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A INPUT=([4.5B]="$PARITY_INPUT_45B" [max]="$PARITY_INPUT_MAX")
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
run() {
  local v=$1; shift
  if [ "$v" = base ]; then
    BASAL_WGP_LOAD=original /opt/basal-dev/basal-cuda-base --color never "$@"
  elif [ "$LOAD" = default ]; then
    env -u BASAL_WGP_LOAD basal-dev --color never "$@"
  else
    BASAL_WGP_LOAD=$LOAD basal-dev --color never "$@"
  fi
}
for s in 4.5B max; do
  model=/data/hf/hub/models--Remek--basal-1.5-$s/snapshots/${REV[$s]}
  table=crates/basal-cli/gemm-tables/$slug--basal-1.5-$s--f16--cublaslt120901.json
  for v in base candidate; do
    for mode in tree single budget no-prefix cache; do
      args=(--batching tree)
      case $mode in
        single|budget) args=(--batching "$mode") ;;
        no-prefix) args+=(--no-prefix-cache) ;;
        cache) args+=(--state-cache-mb 2048) ;;
      esac
      out=$O/$s-$v-$mode
      echo "$(date +%T) $s $v $mode" | tee -a "$O/order.txt"
      run "$v" export --model "$model" --gemm-table "$table" "${args[@]}" \
        --inputs "${INPUT[$s]}" --out "$out" > "$out.log" 2>&1
      jq -se 'length == 44 and all(.error == null)' "$out/bench.jsonl" > /dev/null
      jq -se 'length == 34 and ([.[] | select(.error != null) | .id] | sort) ==
        ["so-eleven-options", "so-missing-type", "so-null-and-missing-instructions", "so-score-one-level"]' \
        "$out/systemone.jsonl" > /dev/null
      for file in bench.jsonl systemone.jsonl; do
        cmp "$O/$s-base-tree/$file" "$out/$file"
      done
      echo "BITWISE_EQUAL $s $v $mode" | tee -a "$O/parity.txt"
    done
  done
done
echo DONE
