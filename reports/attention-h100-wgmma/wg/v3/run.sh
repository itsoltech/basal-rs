#!/bin/bash
# wg3 (Q in shared memory, 2 blocks / SM) on the H100: exports against tc, timing of tc, wg2, wg3 (4.5B), and Nsight
# Compute of the attention at 16384 tokens for the three
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
export LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib
B=/opt/basal-dev/basal-wg3
O=out/wg3; mkdir -p $O/4.5B
snap() { echo /data/hf/hub/models--Remek--$1/snapshots/$2; }
declare -A REV=([4.5B]=784a683bfadcc8865238fc0fc74a83b4000269bc [max]=be1b5ee7e7a9755a931262fa7fab4f59be0fd03c)
declare -A REF=([4.5B]=reports/reference-basal-1.5-4.5B-fp32 [max]=reports/reference-1.5-max-fp32)
for s in 4.5B max; do
  dir=$(snap basal-1.5-$s ${REV[$s]}); T=crates/basal-cli/gemm-tables/nvidia-h100-pcie--basal-1.5-$s--f16--cublaslt120901.json
  for b in tree single; do
    BASAL_ATT=wg3 $B --color never export --model $dir --gemm-table $T --batching $b --inputs ${REF[$s]} --out $O/export-$s-$b > $O/export-$s-$b.log 2>&1
  done
  printf "%-20s " "$s tc vs wg3"; $B compare --a out/wg/export-$s-tc-tree --b $O/export-$s-tree --out $O/compare-$s-tc-wg3.json | tail -1
  printf "%-20s " "$s wg3 single/tree"; $B compare --a $O/export-$s-single --b $O/export-$s-tree --out $O/compare-$s-single-tree.json | tail -1
done
s=4.5B; dir=$(snap basal-1.5-$s ${REV[$s]}); T=crates/basal-cli/gemm-tables/nvidia-h100-pcie--basal-1.5-$s--f16--cublaslt120901.json
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
sudo -n nvidia-smi -lgc $((MAXC * 8 / 10)),$((MAXC * 8 / 10)) > /dev/null 2>&1
for rep in 1 2 3; do
  for v in tc wg2 wg3; do
    BASAL_ATT=$v $B --color never bench --model $dir --gemm-table $T --reference ${REF[$s]} --out $O/$s/bench-$v-$rep.json > /dev/null 2>&1
    BASAL_ATT=$v $B --color never bench-requests --model $dir --gemm-table $T --requests out/fwd/ladder-$s/requests.jsonl --reps 2 --out $O/$s/requests-$v-$rep.json > /dev/null 2>&1
  done
done
sudo -n nvidia-smi -rgc > /dev/null 2>&1
echo TIMING-DONE
NCU=$(ls -d /opt/nvidia/nsight-compute/* | tail -1)/ncu
for v in tc wg2 wg3; do
  sudo -n env HF_HOME=$HF_HOME BASAL_HOME=$BASAL_HOME BASAL_NO_UPDATE_CHECK=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH BASAL_ATT=$v \
    timeout 900 $NCU --set full -k regex:attn_tree -c 1 --launch-skip 200 -o $O/ncu-16384-$v -f $B --color never \
    bench-requests --model $dir --gemm-table $T --requests out/prof/ctx-16384-q1.jsonl --reps 1 --out /tmp/ncu-$v-$RANDOM.json \
    > $O/ncu-16384-$v.log 2>&1
  f=$(ls $O/ncu-16384-$v.ncu-rep* | head -1)
  $NCU --import $f --page details > $O/ncu-16384-$v.txt 2>&1
done
echo DONE
