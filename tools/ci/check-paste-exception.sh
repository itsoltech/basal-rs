#!/usr/bin/env bash
# Enforce the user-approved scope of RUSTSEC-2024-0436 beyond cargo-deny's id/reason exception.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

readonly REVIEW_BY=2027-01-08
today=$(date -u +%F)
if [[ "$today" > "$REVIEW_BY" || "$today" == "$REVIEW_BY" ]]; then
    echo "The paste maintenance exception expired on $REVIEW_BY; review or remove it."
    exit 1
fi

# Compare package names and exact versions, including paste itself, across all target platforms.
# A changed version, new dependent or removal of paste requires a fresh policy review.
# The pinned local tokenizers patch has the same package/version/dependents;
# normalize only its checkout location, leaving every other identity unchanged.
cargo tree --locked --workspace --all-features --target all --invert paste \
    --depth 1 --prefix none --format '{p}' \
    | sed -E 's#^(tokenizers v0\.23\.2) \(/.*/vendor/tokenizers\)$#\1#' \
    | LC_ALL=C sort -u \
    | diff -u .github/paste-dependents.txt -

echo "The paste exception matches the approved dependency graph and expires on $REVIEW_BY."
