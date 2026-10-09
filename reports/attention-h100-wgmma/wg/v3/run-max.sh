#!/bin/bash
# timing of tc and wg3 on basal-1.5-max (2 repetitions, alternating), after wg3-test.sh
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1 LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib
B=/opt/basal-dev/basal-wg3
O=out/wg3/max; mkdir -p $O
dir=/data/hf/hub/models--Remek--basal-1.5-max/snapshots/be1b5ee7e7a9755a931262fa7fab4f59be0fd03c
T=crates/basal-cli/gemm-tables/nvidia-h100-pcie--basal-1.5-max--f16--cublaslt120901.json
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
sudo -n nvidia-smi -lgc $((MAXC * 8 / 10)),$((MAXC * 8 / 10)) > /dev/null 2>&1
for rep in 1 2 3; do
  order="tc wg3"; [ $((rep % 2)) = 0 ] && order="wg3 tc"
  for v in $order; do
    BASAL_ATT=$v $B --color never bench --model $dir --gemm-table $T --reference reports/reference-1.5-max-fp32 --out $O/bench-$v-$rep.json > /dev/null 2>&1
    BASAL_ATT=$v $B --color never bench-requests --model $dir --gemm-table $T --requests out/fwd/ladder-max/requests.jsonl --reps 1 --out $O/requests-$v-$rep.json > /dev/null 2>&1
  done
done
sudo -n nvidia-smi -rgc > /dev/null 2>&1
echo DONE
