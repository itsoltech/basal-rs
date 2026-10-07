#!/bin/bash
cd /work
O=/work/out; mkdir -p $O
for spec in basal-1.5-mini:/base/.cache/gemm/basal-1.5-mini-f16.json:/base/reports/reference-basal-1.5-mini-fp32 basal-1.5-4.5B:/base/.cache/gemm/basal-1.5-4.5B-f16.json:/base/reports/reference-basal-1.5-4.5B-fp32 basal-1.5-max:/base/reports/rust-cuda-1.5-max/gemm-equiv/gemm-algos-f16-invariant-groups.json:/base/reports/reference-1.5-max-fp32; do
  IFS=: read m t r <<< "$spec"
  for b in main new; do for v in single tree; do
    [ -d $O/$b-$m-$v ] || /work/$b/target/release/basal export --model /base/.models/$m --gemm-table $t --batching $v --inputs $r --out $O/$b-$m-$v > $O/$b-$m-$v.log 2>&1
    echo "$b $m $v $(head -1 $O/$b-$m-$v.log)"
  done; done
  for v in single tree; do printf "%s %s main vs new: " $m $v; /work/new/target/release/basal compare --a $O/main-$m-$v --b $O/new-$m-$v --out $O/cmp-$m-$v.json | tail -1; done
done
echo CMP-DONE
