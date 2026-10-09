#!/bin/bash
# Profiles of the pushed build (basal-dev) on a rented GPU (tools/cloud/basal-cloud.py), basal-1.5-4.5B, for the
# attention kernels PROF_ATT (BASAL_ATT values, default "tc"): Nsight Systems on single decisions and ladder requests
# (GPU time per kernel category and idle gaps, tools/perf/nsys_forward.py), Nsight Compute (full set) on one
# attention launch at 4096 and 16384 tokens. Installs Nsight Systems and Compute from the CUDA apt repository when
# missing (sudo). Run in /work after tools/perf/run-forward-cloud.sh (its ladder and tables). Writes out/prof/.
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
T=$(awk '$1 == "4.5B" {print $3}' out/fwd/tables.txt)
L=out/fwd/ladder-4.5B/requests.jsonl
P="nsys profile -t cuda --cuda-memory-usage=false --force-overwrite=true"
for v in ${PROF_ATT:-tc}; do
  BASAL_ATT=$v $P -o $O/bench-$v basal-dev --color never bench --model $dir --gemm-table $T --reference $REF \
    --out $O/bench-$v.json --lat-n 20 > /dev/null 2>&1
  for id in ctx-512-q1 ctx-1792-q1 ctx-4096-q1; do
    awk -v id="\"id\": \"$id\"" 'index($0, id)' $L > $O/$id.jsonl
    BASAL_ATT=$v $P -o $O/$id-$v basal-dev --color never bench-requests --model $dir --gemm-table $T \
      --requests $O/$id.jsonl --reps 4 --out $O/$id-$v.json > /dev/null 2>&1
  done
  for r in $O/*-$v.nsys-rep; do
    nsys export -t sqlite --force-overwrite=true -o ${r%.nsys-rep}.sqlite $r > /dev/null 2>&1
    python3 tools/perf/nsys_forward.py ${r%.nsys-rep}.sqlite > ${r%.nsys-rep}.md 2>&1
  done
  for id in ctx-4096-q1 ctx-16384-q1; do
    awk -v id="\"id\": \"$id\"" 'index($0, id)' $L > $O/$id.jsonl
    sudo -n env "PATH=$PATH" HF_HOME=$HF_HOME BASAL_HOME=$BASAL_HOME BASAL_NO_UPDATE_CHECK=1 BASAL_ATT=$v \
      timeout 1800 ncu --set full -k regex:attn_tree -c 1 --launch-skip 40 -o $O/ncu-$id-$v -f \
      $(command -v basal-dev) --color never bench-requests --model $dir --gemm-table $T --requests $O/$id.jsonl \
      --reps 1 --out /tmp/ncu.json > $O/ncu-$id-$v.log 2>&1
    ncu --import $O/ncu-$id-$v.ncu-rep --page details > $O/ncu-$id-$v.txt 2>&1
  done
done
rm -f $O/*.sqlite
echo DONE
