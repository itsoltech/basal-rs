#!/bin/bash
# Decision sets (tools/decision-sets) against upstream v1.5.0 FP32 on a CUDA machine (tools/cloud/basal-cloud.py): references,
# basal-rs exports (f16 with the GEMM tables of .cache/gemm, f32), comparisons. Run in /work after basal-cloud.py push and
# sync of tools/reference, tools/decision-sets, .cache/decision-sets/set.jsonl, .cache/gemm and .baseline/upstream-1.5.
set -e
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
UP=.baseline/upstream-1.5
if [ ! -x .venv-up/bin/python ]; then
  curl --fail -sSL -o /tmp/uv.tgz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf /tmp/uv.tgz -C /tmp
  (cd $UP && UV_PROJECT_ENVIRONMENT=/work/.venv-up /tmp/uv-x86_64-unknown-linux-gnu/uv sync --frozen --no-dev)
fi
O=out/decision-sets
mkdir -p $O
B=basal-dev
for spec in basal-1.5-4.5B:784a683bfadcc8865238fc0fc74a83b4000269bc basal-1.5-mini:1978d0705ce09cd2d7d8d3e87b304468121e7255; do
  m=${spec%%:*}; rev=${spec#*:}
  dir=/data/hf/hub/models--Remek--$m/snapshots/$rev
  R=$O/reference-$m-fp32
  E="--model $dir --repo Remek/$m --revision $rev --questions .cache/decision-sets/set.jsonl --no-system-one"
  echo "=== $m reference fp32 $(date +%H:%M:%S)"
  [ -d $R ] || BASAL_UPSTREAM=$UP .venv-up/bin/python tools/reference/export_reference.py $E \
    --mode eager --device cuda --dtype float32 --out $R > $R.log 2>&1
  echo "=== $m reference bf16 $(date +%H:%M:%S)"
  [ -d $O/reference-$m-bf16 ] || BASAL_UPSTREAM=$UP .venv-up/bin/python tools/reference/export_reference.py $E \
    --mode fast --dtype bfloat16 --out $O/reference-$m-bf16 > $O/reference-$m-bf16.log 2>&1
  echo "=== $m basal-rs $(date +%H:%M:%S)"
  M="--model $dir"
  $B --color never check-prompts $M --reference $R --out $O/check-prompts-$m.json | tail -2
  $B --color never export $M --gemm-table .cache/gemm/$m-f16.json --batching tree --inputs $R --out $O/export-$m-f16 > $O/export-$m-f16.log 2>&1
  $B --color never export $M --dtype f32 --inputs $R --out $O/export-$m-f32 > $O/export-$m-f32.log 2>&1
  for v in f32 f16; do
    printf "%-8s " $v; $B --color never compare --a $R --b $O/export-$m-$v --out $O/compare-fp32-vs-rust-$m-$v.json | tail -1
  done
  printf "%-8s " up-bf16; $B --color never compare --a $R --b $O/reference-$m-bf16 --out $O/compare-fp32-vs-upstream-bf16-$m.json | tail -1
done
echo DONE
