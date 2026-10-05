#!/bin/bash
# Mixed workload (tools/bench/mixed.jsonl from make_mixed.py): Rust `basal serve` (batch-invariant GEMM table, default
# options), then upstream `basal-serve --mode fast`, same client. Run from the repository root inside the CUDA container.
O=reports/rust-cuda-1.5-max/mixed-load
PY=.baseline/upstream/.venv/bin/python
L="$PY tools/bench/loadtest.py --gpu --model basal-1.5-max --requests tools/bench/mixed.jsonl --n-warm 44 --n-seq 400 --n-conc 400 --concurrency 8 32 --rate-fractions 0.5 0.75 0.9"
wait_up() { for i in $(seq 900); do curl --fail-with-body --silent http://127.0.0.1:$1/health >/dev/null && break; sleep 1; done; }

./target/release/basal serve --model .models/basal-1.5-max --gemm-table reports/rust-cuda-1.5-max/gemm/gemm-algos-f16-invariant.json --addr 127.0.0.1:8100 > $O/serve-rust.log 2>&1 &
wait_up 8100
$L --url http://127.0.0.1:8100/v1/systemone --out $O/rust.json > $O/rust.log 2>&1
pkill -INT -f "basal serve"; sleep 3
echo "rust done"

(cd .baseline/upstream-1.5 && .venv/bin/basal-serve --model /work/.models/basal-1.5-max --name basal-1.5-max --mode fast --host 127.0.0.1 --port 8101 > /work/$O/serve-upstream.log 2>&1 &)
wait_up 8101
$L --url http://127.0.0.1:8101/v1/systemone --out $O/upstream.json > $O/upstream.log 2>&1
pkill -INT -f basal-serve; sleep 3
echo MIXED-DONE
