#!/bin/bash
# bundled GEMM table check on the RTX 6000 Ada: fresh BASAL_HOME, serve start, exports single/tree, FP32, gemm-share
cd /work
export BASAL_HOME=/tmp/bh-bundled BASAL_NO_UPDATE_CHECK=1
B=target/release/basal
M=/work/.models/basal-1.5-4.5B
REF=reports/reference-basal-1.5-4.5B-fp32
O=out/bundled-check
mkdir -p $O
t0=$(date +%s)
$B --color never serve --model $M --addr 127.0.0.1:8199 > $O/serve.log 2>&1 &
for _ in $(seq 1800); do
  curl --fail-with-body --silent http://127.0.0.1:8199/health > /dev/null && break
  pgrep -f "[c]olor never serve --model $M --addr 127.0.0.1:8199" > /dev/null || break
  sleep 1
done
echo "ready after $(( $(date +%s) - t0 )) s"
pkill -INT -f "[c]olor never serve --model $M --addr 127.0.0.1:8199"; sleep 5
T=$BASAL_HOME/.cache/gemm/basal-1.5-4.5B-f16.json
echo "table: $T"
for b in single tree; do
  $B --color never export --model $M --gemm-table $T --batching $b --inputs $REF --out $O/export-$b > /dev/null 2>&1
done
printf "single vs tree: "; $B compare --a $O/export-single --b $O/export-tree --out $O/compare-single-tree.json | tail -1
printf "fp32 vs tree:   "; $B compare --a $REF --b $O/export-tree --out $O/compare-fp32-tree.json | tail -1
$B --color never gemm-share --dir $(dirname $T) --dry-run
echo DONE
