#!/usr/bin/env bash
# Usage: bash examples/client/classify-tickets.sh INPUT.jsonl NEW_OUTPUT_DIR
# INPUT may be - for stdin. BASAL_URL, BASAL_MODEL and BASAL_API_KEY configure HTTP.
set -euo pipefail

if [[ $# != 2 ]]; then
  printf 'Usage: %s INPUT.jsonl NEW_OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

script_dir="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
input_path="$1"
output_dir="$2"
command -v basal >/dev/null
command -v jq >/dev/null
# Require a new directory so previous results cannot be overwritten.
mkdir -- "$output_dir"

status=0
if cat -- "$input_path" |
  basal client --template "$script_dir/routing.json" \
    --input jsonl --state-pointer /text \
    --jobs "${BASAL_JOBS:-8}" --keep-going \
    > "$output_dir/results.jsonl"; then
  status=0
else
  status=$?
fi

jq -c 'select(.ok) | {
  id: .input.id,
  department: .response.answers.department.choice,
  urgent_probability: .response.answers.urgent.noul
}' "$output_dir/results.jsonl" > "$output_dir/routed.jsonl"

jq -c 'select(.ok == false)' "$output_dir/results.jsonl" > "$output_dir/errors.jsonl"

printf 'Routed: %s; errors: %s; pipeline exit: %s\n' \
  "$(wc -l < "$output_dir/routed.jsonl")" \
  "$(wc -l < "$output_dir/errors.jsonl")" "$status" >&2
exit "$status"
