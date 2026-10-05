#!/bin/bash
# Where the forward time of basal-1.5-max goes: per-section profile (synchronised after every section) on the
# basal-bench items, one question per forward and 16 per forward, and nsys kernel summaries of single requests with
# states of ~2k and ~16k tokens. Run from the repository root inside the CUDA container.
set -e
O=reports/rust-cuda-1.5-max/profile-floor
B=./target/release/basal
G="--model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json"
$B profile $G --reference reports/reference-1.5-max-fp32 --n 44 --batch 1 > $O/profile-batch1.json 2> $O/profile-batch1.log
$B profile $G --reference reports/reference-1.5-max-fp32 --n 44 --batch 16 > $O/profile-batch16.json 2> $O/profile-batch16.log
for id in long-2000-q1 long-16000-q1 long-16000-q5; do
  python3 -c "import json,sys
for l in open('tools/bench/long_states.jsonl'):
    r=json.loads(l)
    if r['id']=='$id': json.dump(r['request'], open('/tmp/$id.json','w'))"
  $B decide $G --request /tmp/$id.json > /dev/null   # warm-up of the request shapes in a separate process
  nsys profile -t cuda -o $O/nsys-$id -f true $B decide $G --request /tmp/$id.json > /dev/null 2>&1
  nsys stats --report cuda_gpu_kern_sum --format csv -o $O/kern-$id $O/nsys-$id.nsys-rep > /dev/null 2>&1
done
echo PROFILE-DONE
