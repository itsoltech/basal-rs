#!/bin/bash
# Tree attention on Metal (attn_tree_f32 in kernels.metal) against the earlier MLX SDPA path (BASAL_ATT=sdpa) on an
# Apple M2 Max, basal-1.5-max, 4.5B and mini. Run from the repository root:
#   1. agreement with the upstream FP32 references and dependence on packing: exports with one question per forward
#      (single), questions in shared-prefix trees (tree) and several rows per forward (budget), both attention paths;
#      f32 exports of mini and 4.5B
#   2. single decision (basal-bench methodology), SDPA vs tree attention in ABBA order (tools/bench/ab.py), 3 rounds
# A pause before every GPU run (PAUSE seconds, default 30).
set -e
O=reports/metal-m2-max-tree
B=./target/release/basal
PAUSE=${PAUSE:-30}
ref() { [ $1 = basal-1.5-max ] && echo reports/reference-1.5-max-fp32 || echo reports/reference-$1-fp32; }
MODELS="basal-1.5-mini basal-1.5-4.5B basal-1.5-max"

for m in $MODELS; do
  R=$(ref $m)
  for att in sdpa tree; do
    for v in single tree budget; do
      E=$O/export-$m-$att-$v
      [ -d $E ] && continue
      sleep $PAUSE
      env BASAL_ATT=$([ $att = sdpa ] && echo sdpa || echo tree) $B export --model .models/$m --batching $v --inputs $R \
        --out $E > $E.log 2>&1
    done
    for v in tree budget; do
      printf "%s %s single vs %s: " $m $att $v
      $B compare --a $O/export-$m-$att-single --b $O/export-$m-$att-$v --out $O/compare-$m-$att-single-vs-$v.json | tail -1
    done
    printf "%s %s vs FP32: " $m $att
    $B compare --a $R --b $O/export-$m-$att-single --out $O/compare-upstream-fp32-vs-$m-$att-single.json | tail -1
  done
  if [ $m != basal-1.5-max ] && [ ! -d $O/export-$m-tree-f32 ]; then
    sleep $PAUSE
    $B export --model .models/$m --dtype f32 --inputs $R --out $O/export-$m-tree-f32 > $O/export-$m-tree-f32.log 2>&1
    printf "%s tree f32 vs FP32: " $m
    $B compare --a $R --b $O/export-$m-tree-f32 --out $O/compare-upstream-fp32-vs-$m-tree-f32.json | tail -1
  fi
done

for m in $MODELS; do
  R=$(ref $m)
  [ -d $O/ab-bench-$m ] && continue
  python3 tools/bench/ab.py --out $O/ab-bench-$m --rounds 3 \
    --a "sleep $PAUSE; env BASAL_ATT=sdpa $B bench --model .models/$m --reference $R" \
    --b "sleep $PAUSE; $B bench --model .models/$m --reference $R" | tail -3
done
echo RUN-DONE
