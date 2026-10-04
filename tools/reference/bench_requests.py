"""Latency of whole System One requests on the pinned upstream server logic (Server.decide, no HTTP; MLX backend on
Apple Silicon, --mode fast on CUDA).

  .baseline/upstream/.venv/bin/python tools/reference/bench_requests.py \
      --model .models/basal-1.0-4.5B --requests tools/reference/requests_fanout.jsonl --reps 10 --out FILE.json

One client, requests one after another; each request is warmed up once, then timed `--reps` times. A timing covers
Server.decide: prompt rendering, tokenization, the adaptive-batching worker (all questions of the request in one
run_shared call), the forward and the calibrated answer. The Rust counterpart is `basal bench-requests`.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".baseline" / "upstream"))

from basal.server import Server, parser  # noqa: E402


async def run(srv, reqs, reps):
    worker = asyncio.get_running_loop().create_task(srv.worker())
    rows = []
    for r in reqs:
        body = r["request"]
        await srv.decide(body)  # warm-up
        lat = []
        for _ in range(reps):
            t = time.perf_counter()
            out = await srv.decide(body)
            lat.append((time.perf_counter() - t) * 1000)
        lat.sort()
        rows.append(dict(id=r["id"], questions=len(body["questions"]), input_tokens=out["usage"]["input_tokens"],
                         median_ms=statistics.median(lat), min_ms=lat[0], max_ms=lat[-1], all_ms=lat))
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "all_ms"}), flush=True)
    worker.cancel()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / ".models/basal-1.0-4.5B"))
    ap.add_argument("--requests", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mode", default="mlx", help="upstream server mode (mlx; on CUDA: fast, fast-nocompile, eager)")
    ap.add_argument("--dtype", default="bfloat16", help="dtype of the upstream server (bfloat16, float16)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    reqs = [json.loads(x) for x in Path(a.requests).read_text().splitlines() if x.strip()]
    srv = Server(parser().parse_args(["--model", a.model, "--mode", a.mode, "--name", "basal-1.0-4.5B",
                                       "--dtype", a.dtype]))
    rows = asyncio.run(run(srv, reqs, a.reps))
    Path(a.out).write_text(json.dumps(dict(tool="tools/reference/bench_requests.py", mode=a.mode, dtype=a.dtype,
                                           reps=a.reps, requests=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
