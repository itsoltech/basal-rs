"""basal-bench methodology (upstream bench.measure) for the MLX backend in a chosen dtype.

basal-bench builds the MLX backend in bfloat16 only; this runs the same measure() with MLXBackend(model, dtype) so a
float16 Rust run can be compared with float16 MLX rather than with bfloat16.

  .baseline/upstream/.venv/bin/python tools/reference/bench_mlx_dtype.py --dtype float16 --out FILE.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".baseline" / "upstream"))

from basal.bench import DEFAULT_QUESTIONS, groups_for, load_questions, measure, memory_gb, reset_memory  # noqa: E402
from basal.engine import MLXBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / ".models/basal-1.0-4.5B"))
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=44)
    ap.add_argument("--lat-n", dest="lat_n", type=int, default=39)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    reset_memory()
    t0 = time.time()
    be = MLXBackend(Path(a.model), a.dtype)
    load_s = time.time() - t0
    qs = load_questions(DEFAULT_QUESTIONS, a.n)
    groups = groups_for(be.tok, qs)
    dec, r = measure(be, groups, min(a.lat_n, len(groups) - 5))
    top = [max(range(len(d)), key=d.__getitem__) for d in dec]
    r = dict(mode=f"mlx-{a.dtype}", n=len(groups), load_s=round(load_s, 1), **r, mem_gb=memory_gb(be),
             acc=sum(t == g[0]["gold"] for t, g in zip(top, groups)) / len(groups), decisions=dec)
    Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "decisions"}), flush=True)


if __name__ == "__main__":
    main()
