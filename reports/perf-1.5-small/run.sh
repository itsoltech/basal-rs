#!/bin/bash
# Performance of basal-1.5-4.5B and basal-1.5-mini, basal-rs against upstream v1.5.0 (`fast`: BF16, torch.compile,
# CUDA graphs) on the same GPU: single decision (basal-bench methodology), long states, HTTP with one-question
# requests (1 / 8 / 32 clients) and the mixed workload. basal-1.5-mini (8192 positions) gets the long states and
# mixed requests with states up to ~4k tokens. Also upstream v1.5.0 long states of basal-1.5-max (the earlier run
# used the upstream 1.0 server code). Run from the repository root in the CUDA container.
set -e
O=reports/perf-1.5-small
B=./target/release/basal
UP=.baseline/upstream-1.5
PY=$UP/.venv/bin/python
LPY=.baseline/upstream/.venv/bin/python
R=reports/reference-1.5-max-fp32
up() { for i in $(seq 1800); do curl --fail-with-body --silent http://127.0.0.1:$1/health > /dev/null && return 0; sleep 1; done; return 1; }
nvidia-smi --query-gpu=name,power.limit --format=csv > $O/gpu.csv

# request sets: long states with the model's name (basal-1.5-mini: states up to ~4k tokens)
python3 - <<'PY'
import json
lines = [json.loads(l) for l in open("tools/bench/long_states.jsonl") if l.strip()]
for m, cap in [("basal-1.5-4.5B", 16000), ("basal-1.5-mini", 4000)]:
    open(f"reports/perf-1.5-small/long_states_{m}.jsonl", "w").write("".join(
        json.dumps(dict(r, request=dict(r["request"], model=m)), ensure_ascii=False) + "\n"
        for r in lines if int(r["id"].split("-")[1]) <= cap))
mixed = [json.loads(l) for l in open("tools/bench/mixed.jsonl") if l.strip()]
open("/tmp/mixed_4k.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mixed
                                                if len(json.dumps(r["request"]["state"], ensure_ascii=False)) < 20000))
PY

[ -f $O/long-upstream-basal-1.5-max.json ] || BASAL_UPSTREAM=$UP $PY tools/reference/bench_requests.py --mode fast --model .models/basal-1.5-max \
  --requests tools/bench/long_states.jsonl --reps 2 --out $O/long-upstream-basal-1.5-max.json > $O/long-upstream-basal-1.5-max.log 2>&1

for m in basal-1.5-4.5B basal-1.5-mini; do
  LONG=$O/long_states_$m.jsonl; MIX=tools/bench/mixed.jsonl
  [ $m = basal-1.5-mini ] && MIX=/tmp/mixed_4k.jsonl
  M="--model .models/$m --gemm-table .cache/gemm/$m-f16.json"
  # basal-rs
  $B bench $M --reference $R --out $O/bench-rust-$m.json > /dev/null 2>&1
  $B bench-requests $M --requests $LONG --reps 2 --out $O/long-rust-$m.json 2> /dev/null
  $B serve $M --addr 127.0.0.1:8100 > $O/serve-rust-$m.log 2>&1 &
  up 8100
  $LPY tools/bench/loadtest.py --gpu --model $m --reference $R --n-seq 44 --n-conc 300 --concurrency 8 32 \
    --url http://127.0.0.1:8100/v1/systemone --out $O/short-rust-$m.json > /dev/null 2>&1
  $LPY tools/bench/loadtest.py --gpu --model $m --requests $MIX --n-warm 44 --n-seq 400 --n-conc 400 --concurrency 32 \
    --rate-fractions 0.5 0.75 0.9 --url http://127.0.0.1:8100/v1/systemone --out $O/mixed-rust-$m.json > /dev/null 2>&1
  pkill -INT -f "basal serve"; sleep 5
  echo "$m rust done"
  # upstream v1.5.0
  $UP/.venv/bin/basal-bench --model .models/$m --modes fast --out $O/bench-upstream-$m.json > $O/bench-upstream-$m.log 2>&1
  BASAL_UPSTREAM=$UP $PY tools/reference/bench_requests.py --mode fast --model .models/$m --requests $LONG --reps 2 \
    --out $O/long-upstream-$m.json > $O/long-upstream-$m.log 2>&1
  (cd $UP && .venv/bin/basal-serve --model /work/.models/$m --name $m --mode fast --host 127.0.0.1 --port 8101 \
    > /work/$O/serve-upstream-$m.log 2>&1 &)
  up 8101
  $LPY tools/bench/loadtest.py --gpu --model $m --reference $R --n-seq 44 --n-conc 300 --concurrency 8 32 \
    --url http://127.0.0.1:8101/v1/systemone --out $O/short-upstream-$m.json > /dev/null 2>&1
  $LPY tools/bench/loadtest.py --gpu --model $m --requests $MIX --n-warm 44 --n-seq 400 --n-conc 200 --concurrency 32 \
    --url http://127.0.0.1:8101/v1/systemone --out $O/mixed-upstream-$m.json > /dev/null 2>&1
  pkill -INT -f basal-serve; sleep 5; pkill -9 -f basal-serve || true; sleep 3
  echo "$m upstream done"
done
echo PERF-DONE
