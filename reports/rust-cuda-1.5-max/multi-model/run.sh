#!/bin/bash
# Multi-model server test: config.yml (basal-1.5-max with the repository GEMM table, basal-1.5-4.5B and
# basal-1.5-mini with `gemm_table: auto`). Start-up with table generation, routing checks (smoke.py), answers of
# basal-1.5-max against a single-model server, batch invariance of the generated tables, three load tests (one per
# model) at the same time, restart with the cached tables. Run from the repository root in the CUDA container.
set -e
O=reports/rust-cuda-1.5-max/multi-model
B=./target/release/basal
PY=.baseline/upstream/.venv/bin/python
MAXT=reports/rust-cuda-1.5-max/gemm-equiv/gemm-algos-f16-invariant-groups.json
up() { for i in $(seq 7200); do curl --fail-with-body --silent http://127.0.0.1:8100/health > /dev/null && return 0; sleep 1; done; return 1; }

t0=$(date +%s)
$B serve --config $O/config.yml > $O/serve-1.log 2>&1 &
up
echo "start with table generation: $(( $(date +%s) - t0 )) s" > $O/startup.txt
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $O/startup.txt
$PY $O/smoke.py --url http://127.0.0.1:8100 --models basal-1.5-max basal-1.5-4.5B basal-1.5-mini --out $O/smoke-multi.json
# concurrent load on all three models (one client process per model, 16 clients each)
pids=""
for m in basal-1.5-max basal-1.5-4.5B basal-1.5-mini; do
  $PY tools/bench/loadtest.py --gpu --model $m --reference reports/reference-1.5-max-fp32 --n-seq 20 --n-conc 200 \
    --concurrency 16 --url http://127.0.0.1:8100/v1/systemone --out $O/load-$m.json > $O/load-$m.log 2>&1 &
  pids="$pids $!"
done
wait $pids
pkill -INT -f "basal serve"; sleep 5

# single-model server with the same basal-1.5-max options: same answers
$B serve --model .models/basal-1.5-max --gemm-table $MAXT --addr 127.0.0.1:8100 > $O/serve-single.log 2>&1 &
up
$PY $O/smoke.py --url http://127.0.0.1:8100 --models basal-1.5-max --out $O/smoke-single.json > /dev/null
pkill -INT -f "basal serve"; sleep 5

# batch invariance of the generated tables
for m in basal-1.5-4.5B basal-1.5-mini; do
  for b in single tree; do
    $B export --model .models/$m --gemm-table .cache/gemm/$m-f16.json --batching $b --inputs reports/reference-1.5-max-fp32 \
      --out $O/export-$m-$b > /dev/null 2>&1
  done
  $B compare --a $O/export-$m-single --b $O/export-$m-tree --out $O/compare-$m-single-vs-tree.json | tail -1
done

# restart: tables from the cache
t0=$(date +%s)
$B serve --config $O/config.yml > $O/serve-2.log 2>&1 &
up
echo "restart with cached tables: $(( $(date +%s) - t0 )) s" >> $O/startup.txt
pkill -INT -f "basal serve"; sleep 5
echo MULTI-DONE
