#!/bin/bash
# Profiles of the pushed build (basal-dev) on a rented GPU (tools/cloud/basal-cloud.py), basal-1.5-4.5B: Nsight Systems
# on single decisions and ladder requests (GPU time per kernel category and idle gaps, tools/perf/nsys_forward.py) with
# the default attention, Nsight Compute (full set) on one attention launch of a forward of 4096 and of 16384 tokens for
# the kernels PROF_ATT (BASAL_ATT values, default "tc"). Installs Nsight Systems and Compute from the CUDA apt
# repository when missing (sudo). Run in /work after tools/perf/run-wg-cloud.sh (its ladder and tables). Writes
# out/prof/.
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
O=out/prof
mkdir -p $O
if ! command -v nsys > /dev/null || ! command -v ncu > /dev/null; then
  if ! apt-cache pkgnames 2>/dev/null | awk '/^nsight-systems-20/ {f=1} END {exit !f}'; then
    . /etc/os-release
    wget -q -O /tmp/kr.deb "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb" \
      && sudo -n dpkg -i /tmp/kr.deb > /dev/null
    sudo -n apt-get update -qq > /dev/null 2>&1
  fi
  ns=$(apt-cache pkgnames | awk '/^nsight-systems-20[0-9.]+$/' | sort -V | tail -1)
  nc=$(apt-cache pkgnames | awk '/^nsight-compute-20[0-9.]+$/' | sort -V | tail -1)
  sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $ns $nc > $O/nsight-install.log 2>&1
  export PATH=$PATH:$(dirname "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | tail -1)"):$(ls -d /opt/nvidia/nsight-compute/* | tail -1)
fi
nsys --version > $O/nsight-versions.txt 2>&1; ncu --version >> $O/nsight-versions.txt 2>&1
dir=/data/hf/hub/models--Remek--basal-1.5-4.5B/snapshots/784a683bfadcc8865238fc0fc74a83b4000269bc
REF=reports/reference-basal-1.5-4.5B-fp32
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
slug=$(echo "$GPU" | tr 'A-Z' 'a-z' | tr -cs 'a-z0-9' '-' | sed 's/-$//')
T=crates/basal-cli/gemm-tables/$slug--basal-1.5-4.5B--f16--cublaslt120901.json
[ -f $T ] || T=out/wg/table-4.5B.json
L=out/wg/ladder-4.5B/requests.jsonl
P="nsys profile -t cuda --cuda-memory-usage=false --force-overwrite=true"
$P -o $O/bench basal-dev --color never bench --model $dir --gemm-table $T --reference $REF --out $O/bench.json \
  --lat-n 20 > /dev/null 2>&1
for id in ctx-512-q1 ctx-1792-q1 ctx-4096-q1; do
  awk -v id="\"id\": \"$id\"" 'index($0, id)' $L > $O/$id.jsonl
  $P -o $O/$id basal-dev --color never bench-requests --model $dir --gemm-table $T --requests $O/$id.jsonl --reps 4 \
    --out $O/$id.json > /dev/null 2>&1
done
for r in $O/*.nsys-rep; do
  nsys export -t sqlite --force-overwrite=true -o ${r%.nsys-rep}.sqlite $r > /dev/null 2>&1
  python3 tools/perf/nsys_forward.py ${r%.nsys-rep}.sqlite > ${r%.nsys-rep}.md 2>&1
done
NCU=$(command -v ncu)
for v in ${PROF_ATT:-tc}; do
  for id in ctx-4096-q1 ctx-16384-q1; do
    awk -v id="\"id\": \"$id\"" 'index($0, id)' $L > $O/$id.jsonl
    # launch 200: past the template prefixes and the warm-up, inside a forward of the request
    sudo -n env "PATH=$PATH" HF_HOME=$HF_HOME BASAL_HOME=$BASAL_HOME BASAL_NO_UPDATE_CHECK=1 BASAL_ATT=$v \
      LD_LIBRARY_PATH=/data/basal/.local/cuda/12.9.1/lib timeout 1800 $NCU --set full -k regex:attn_tree -c 1 \
      --launch-skip 200 -o $O/ncu-$id-$v -f /opt/basal-dev/basal-cuda --color never bench-requests --model $dir \
      --gemm-table $T --requests $O/$id.jsonl --reps 1 --out /tmp/ncu-$id-$v.json > $O/ncu-$id-$v.log 2>&1
    $NCU --import "$(ls $O/ncu-$id-$v.ncu-rep* | head -1)" --page details > $O/ncu-$id-$v.txt 2>&1
  done
done
rm -f $O/*.sqlite
echo DONE
