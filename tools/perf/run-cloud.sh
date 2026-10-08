#!/bin/bash
# Performance of basal-rs against upstream v1.5.0 (`fast`: BF16, torch.compile, CUDA graphs) on one rented GPU
# (tools/cloud/basal-cloud.py), for basal-1.5-mini, 4.5B and max, each in the default server configuration
# (batch-invariant GEMM tables of `gemm_table: auto`, generated on the GPU):
#   - single decision, basal-bench methodology (44 items, both orders, batch 1): `basal bench`, upstream `basal-bench`
#   - HTTP, one question per request (44 items in a loop), 1 / 8 / 32 clients: tools/bench/loadtest.py, both servers
#   - mixed workload (tools/bench/mixed.jsonl; mini: states up to ~4k tokens), sequential and 32 clients: basal-rs
# Run in /work after `push` and `sync tools/bench tools/reference reports/reference-1.5-max-fp32 .baseline/upstream-1.5`.
# Writes out/perf/; the GPU, driver, power limit and CPU model go to out/perf/machine.json (no host names, no
# addresses).
set -e
cd /work
export HF_HOME=/data/hf BASAL_HOME=/data/basal BASAL_NO_UPDATE_CHECK=1
[ -d /usr/local/cuda-13.0/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
UP=.baseline/upstream-1.5
if [ ! -x .venv-up/bin/python ]; then
  curl --fail -sSL -o /tmp/uv.tgz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
  tar -xzf /tmp/uv.tgz -C /tmp
  (cd $UP && UV_PROJECT_ENVIRONMENT=/work/.venv-up /tmp/uv-x86_64-unknown-linux-gnu/uv sync --frozen --no-dev)
fi
PY=.venv-up/bin/python
O=out/perf
mkdir -p $O
# upstream's PyTorch (CUDA 13 wheels) on a driver older than 580 (forward compatibility of 13.0 unavailable): the
# same torch version built for CUDA 12.8, with the forward-compatibility libraries of CUDA 12.9
if ! $PY -c "import torch; torch.zeros(1).cuda()" > /dev/null 2>&1; then
  v=$($PY -c "import importlib.metadata as m; print(m.version('torch'))")
  /tmp/uv-x86_64-unknown-linux-gnu/uv pip install --python $PY --index-url https://download.pytorch.org/whl/cu128 \
    "torch==${v%%+*}" > $O/torch-cu128.log 2>&1 || true
  [ -d /usr/local/cuda-12.9/compat ] && export LD_LIBRARY_PATH=/usr/local/cuda-12.9/compat:$LD_LIBRARY_PATH
  $PY -c "import torch; torch.zeros(1).cuda(); print('torch', torch.__version__, torch.version.cuda)" >> $O/torch-cu128.log 2>&1 || true
fi
R=reports/reference-1.5-max-fp32
B=basal-dev
# up PORT [PATTERN]: wait for /health; give up when no process matches PATTERN any more (the server died)
up() {
  for _ in $(seq 3600); do
    curl --fail-with-body --silent "http://127.0.0.1:$1/health" > /dev/null && return 0
    [ -n "$2" ] && ! pgrep -f "$2" > /dev/null && return 1
    sleep 1
  done
  return 1
}
$PY - > $O/machine.json <<'PY'
import json, platform, subprocess
q = "name,driver_version,compute_cap,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.max"
gpu = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout
cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
print(json.dumps({"gpu_query": q, "gpu": gpu.strip(), "cpu": cpu[0] if cpu else None, "cpus": len(cpu),
                  "kernel": platform.release(), "basal": subprocess.run(["basal-dev", "--version"], capture_output=True,
                  text=True).stdout.strip()}, indent=1))
PY
cat $O/machine.json
$PY - <<'PY'
import json
mixed = [json.loads(l) for l in open("tools/bench/mixed.jsonl") if l.strip()]
open("/tmp/mixed_4k.jsonl", "w").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mixed
                                                if len(json.dumps(r["request"]["state"], ensure_ascii=False)) < 20000))
PY
snap() { echo /data/hf/hub/models--Remek--$1/snapshots/$2; }
for spec in basal-1.5-mini:1978d0705ce09cd2d7d8d3e87b304468121e7255 basal-1.5-4.5B:784a683bfadcc8865238fc0fc74a83b4000269bc \
            basal-1.5-max:be1b5ee7e7a9755a931262fa7fab4f59be0fd03c; do
  m=${spec%%:*}; dir=$(snap $m ${spec#*:})
  MIX=tools/bench/mixed.jsonl
  [ $m = basal-1.5-mini ] && MIX=/tmp/mixed_4k.jsonl
  echo "=== $m basal-rs $(date +%H:%M:%S)"
  # the server generates the GEMM table of this GPU at the first start (gemm_table: auto), then serves
  $B --color never serve --model $dir --addr 127.0.0.1:8100 > $O/serve-rust-$m.log 2>&1 &
  up 8100 "basal-cuda --color never serve" || { echo "$m: basal-rs server did not start"; continue; }
  $PY tools/bench/loadtest.py --gpu --model $m --reference $R --n-seq 44 --n-conc 300 --concurrency 8 32 \
    --url http://127.0.0.1:8100/v1/systemone --out $O/short-rust-$m.json > $O/short-rust-$m.log 2>&1
  $PY tools/bench/loadtest.py --gpu --model $m --requests $MIX --n-warm 44 --n-seq 300 --n-conc 300 --concurrency 32 \
    --url http://127.0.0.1:8100/v1/systemone --out $O/mixed-rust-$m.json > $O/mixed-rust-$m.log 2>&1
  pkill -INT -f "basal-cuda --color never serve" || true; sleep 5
  $B --color never bench --model $dir --gemm-table /data/basal/.cache/gemm/$m-f16.json --reference $R \
    --out $O/bench-rust-$m.json > $O/bench-rust-$m.log 2>&1
  echo "=== $m upstream $(date +%H:%M:%S)"
  .venv-up/bin/basal-bench --model $dir --modes fast --out $O/bench-upstream-$m.json > $O/bench-upstream-$m.log 2>&1 || true
  (cd $UP && /work/.venv-up/bin/basal-serve --model $dir --name $m --mode fast --host 127.0.0.1 --port 8101 \
    > /work/$O/serve-upstream-$m.log 2>&1 &)
  if up 8101 basal-serve; then
    $PY tools/bench/loadtest.py --gpu --model $m --reference $R --n-seq 44 --n-conc 300 --concurrency 8 32 \
      --url http://127.0.0.1:8101/v1/systemone --out $O/short-upstream-$m.json > $O/short-upstream-$m.log 2>&1 || true
  fi
  pkill -INT -f basal-serve || true; sleep 5; pkill -9 -f basal-serve || true; sleep 3
done
echo DONE
