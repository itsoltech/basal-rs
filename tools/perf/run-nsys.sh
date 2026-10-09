#!/bin/bash
# Nsight Systems profile of basal-1.5-4.5B on this GPU: single decisions (`basal bench`) and every context ladder
# request (LADDER, tools/bench/make_context_ladder.py), one trace each, with the GPU time per kernel category of
# tools/perf/nsys_forward.py. Run in /work of the CUDA dev container (tools/cuda/Dockerfile, models in .models, GEMM
# table GEMM_TABLE); writes out/prof/$TAG.
cd /work
export BASAL_NO_UPDATE_CHECK=1
B=${BIN:-target/release/basal}
M=/work/.models/basal-1.5-4.5B
T=${GEMM_TABLE:-out/prof/gemm/basal-1.5-4.5B-f16.json}
REF=reports/reference-basal-1.5-4.5B-fp32
O=out/prof/${TAG:-base}
mkdir -p $O
P="nsys profile -t cuda --cuda-memory-usage=false --force-overwrite=true"
$P -o $O/bench $B --color never bench --model $M --gemm-table $T --reference $REF --out $O/bench.json --lat-n 20 > $O/bench.log 2>&1
nsys export -t sqlite --force-overwrite=true -o $O/bench.sqlite $O/bench.nsys-rep > /dev/null
python3 tools/perf/nsys_forward.py $O/bench.sqlite > $O/bench.md
while read -r line; do
  id=$(python3 -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$line")
  echo "$line" > $O/$id.jsonl
  $P -o $O/$id $B --color never bench-requests --model $M --gemm-table $T --requests $O/$id.jsonl --reps 4 --out $O/$id.json > $O/$id.log 2>&1
  nsys export -t sqlite --force-overwrite=true -o $O/$id.sqlite $O/$id.nsys-rep > /dev/null
  python3 tools/perf/nsys_forward.py $O/$id.sqlite > $O/$id.md
done < ${LADDER:-out/prof/ladder-4.5B/requests.jsonl}
rm -f $O/*.sqlite
echo DONE
