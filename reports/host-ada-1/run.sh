#!/bin/bash
# host caches A/B on the Ada: exports (4.5B tree) compared, then single decisions (basal bench) and short ladder
# requests in rotating order, SM clock locked
cd /work
export BASAL_NO_UPDATE_CHECK=1
M=/work/.models/basal-1.5-4.5B; T=out/prof/gemm/basal-1.5-4.5B-f16.json; REF=reports/reference-basal-1.5-4.5B-fp32
O=out/host-${TAG:-1}; mkdir -p $O
for v in H C; do out/basal-$v --color never export --model $M --gemm-table $T --batching tree --inputs $REF --out $O/export-$v > /dev/null 2>&1; done
target/release/basal compare --a $O/export-H --b $O/export-C --out $O/compare-H-C.json | tail -1
awk '/"id": "ctx-(512|1024|1792)-q[15]"/' out/prof/ladder-full/requests.jsonl > $O/short.jsonl
for rep in 1 2 3 4 5; do
  order="H C"; [ $((rep % 2)) = 0 ] && order="C H"
  for v in $order; do
    out/basal-$v --color never bench --model $M --gemm-table $T --reference $REF --out $O/bench-$v-$rep.json > /dev/null 2>&1
    out/basal-$v --color never bench-requests --model $M --gemm-table $T --requests $O/short.jsonl --reps 3 --out $O/requests-$v-$rep.json > /dev/null 2>&1
  done
done
echo DONE
