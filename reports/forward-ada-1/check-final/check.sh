#!/bin/bash
# bitwise checks of the final build (target/release/basal) against out/basal-A: 4.5B f16 and bf16, max f16
cd /work
export BASAL_NO_UPDATE_CHECK=1
O=out/check-b4; mkdir -p $O
cmp() { printf "%-14s " $1; target/release/basal compare --a $O/A-$1 --b $O/B-$1 --out $O/compare-$1.json | tail -1; }
M=/work/.models/basal-1.5-4.5B; T=out/prof/gemm/basal-1.5-4.5B-f16.json; REF=reports/reference-basal-1.5-4.5B-fp32
TB=out/prof/gemm/basal-1.5-4.5B-bf16.json
[ -f $TB ] || target/release/basal --color never gemm-search --model $M --dtype bf16 --invariant --out $TB > $O/gemm-bf16.log 2>&1
for b in A B; do bin=target/release/basal; [ $b = A ] && bin=out/basal-A
  $bin --color never export --model $M --gemm-table $T --batching tree --inputs $REF --out $O/$b-4.5B-f16 > /dev/null 2>&1
  $bin --color never export --model $M --dtype bf16 --gemm-table $TB --batching tree --inputs $REF --out $O/$b-4.5B-bf16 > /dev/null 2>&1
done
cmp 4.5B-f16; cmp 4.5B-bf16
M=/work/.models/basal-1.5-max; REF=reports/reference-1.5-max-fp32; T=out/prof/gemm/basal-1.5-max-f16.json
if [ ! -f $T ]; then
  target/release/basal --color never gemm-search --model $M --invariant --out $T > $O/gemm-max.log 2>&1
fi
for b in A B; do bin=target/release/basal; [ $b = A ] && bin=out/basal-A
  $bin --color never export --model $M --gemm-table $T --batching tree --inputs $REF --out $O/$b-max-f16 > /dev/null 2>&1
done
cmp max-f16
echo DONE
