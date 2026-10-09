#!/usr/bin/env bash
# Usage: find documents -type f -name '*.txt' -print0 |
#          xargs -0 -n 1 bash examples/client/classify-file.sh
set -euo pipefail

if [[ $# != 1 ]]; then
  printf 'Usage: %s FILE\n' "$0" >&2
  exit 2
fi

category="$(basal client --state "$1" \
  --ask 'Jaki jest główny temat dokumentu?' \
  --choice 'invoice=Faktura lub rozliczenie' \
  --choice 'contract=Umowa lub warunki współpracy' \
  --choice 'other=Inny temat lub brak informacji' \
  --value)"

jq -nc --arg path "$1" --arg category "$category" '{path:$path,category:$category}'
