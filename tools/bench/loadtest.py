"""HTTP load test of a System One server (basal-rs `basal serve` or upstream `basal-serve`), same client for both.

  .baseline/upstream/.venv/bin/python tools/bench/loadtest.py --url http://127.0.0.1:8000/v1/systemone \
      --reference reports/reference-cuda-fp32 --out FILE.json

Requests: every item of `<reference>/bench.jsonl` (the 44 default basal-bench items) as one choice question in a
full System One request (`model`, `state`, `questions`), cycled. With `--requests FILE` (lines {"id", "request"},
e.g. tools/reference/requests_fanout.jsonl) those whole requests are cycled instead.

Phases: warm-up (one pass, or `--n-warm` requests), sequential latency (`--n-seq` requests one after another), then
for every concurrency in `--concurrency` `--n-conc` requests with that many clients in flight (closed loop). With
`--rate-fractions`, open-loop phases follow: `--n-conc` requests arriving as a Poisson process (fixed seed) at the
given fractions of the highest requests/s of the closed-loop phases (or at `--rates` requests/s), whatever the
answers (as independent users).
Request lines may carry a "class" (tools/bench/make_mixed.py); latency is then also reported per class. Reported per phase: p50 / p95 / p99 latency,
requests/s and decisions/s (questions answered per second), errors (non-200 answers). Answers are compared across
phases: the same request must get the same probabilities whatever the batch it ran in (max |difference| reported),
and identical complete `answers` objects (including confidence and evidence); differing answers are counted.
`--answers-out` saves the first complete answers by request ID for comparison across runs.
With `--gpu`, nvidia-smi is sampled every 200 ms: mean power, SM clock, temperature, utilization, J/decision,
and the highest observed memory usage (shorter allocation peaks may be missed);
`mean_batch_requests` is read from the server's `x-basal-batch-requests` header.
"""
import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

import httpx


def requests_from_reference(path, model):
    """(id, body, class) for every bench item."""
    out = []
    for line in (Path(path) / "bench.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = {"type": "choice", "instructions": r["question"],
             "criteria": {f"option_{k + 1}": o for k, o in enumerate(r["options"])}}
        out.append((f"bench-{r['index']}", {"model": model, "state": r["state"], "questions": {"q": q}}, "bench"))
    return out


def quant(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else float("nan")


class Gpu:
    """nvidia-smi sampled every 200 ms (power W, SM MHz, temperature C, utilization %, memory MiB)."""

    def __init__(self):
        self.samples, self.proc = [], None
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-lms", "200"], stdout=subprocess.PIPE, text=True)
        except OSError:
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), *[float(x) for x in line.split(",")]))
            except ValueError:
                pass

    def window(self, t0, t1):
        xs = [s for s in self.samples if t0 <= s[0] <= t1]
        if not xs:
            return {}
        mean = lambda i: statistics.mean(x[i] for x in xs)
        return dict(gpu_power_w=mean(1), gpu_sm_mhz=mean(2), gpu_temp_c=mean(3), gpu_util=mean(4),
                    gpu_memory_peak_sampled_mib=max(x[5] for x in xs))


def probs(resp):
    return {k: v.get("probabilities", {}) for k, v in resp.get("answers", {}).items()}


def per_class(lat):
    out = {}
    for cls in sorted({k for k, _ in lat}):
        xs = [v for k, v in lat if k == cls]
        out[cls] = dict(n=len(xs), p50_ms=statistics.median(xs), p95_ms=quant(xs, 0.95), max_ms=max(xs))
    return out


async def phase(c, url, reqs, n, conc, seen, gpu, rate=None):
    """Closed loop with `conc` clients, or open loop with Poisson arrivals at `rate` requests/s."""
    sem = asyncio.Semaphore(conc if rate is None else n)
    lat, errors, questions, batch = [], 0, 0, []
    diff = 0.0
    answer_mismatches = 0
    done = []

    async def one(i, at=0.0):
        nonlocal errors, questions, diff, answer_mismatches
        rid, body, cls = reqs[i % len(reqs)]
        if at:
            await asyncio.sleep(max(0.0, t_start + at - time.perf_counter()))
        async with sem:
            t0 = time.perf_counter()
            r = await c.post(url, json=body)
            dt = time.perf_counter() - t0
        done.append(time.perf_counter())
        if r.status_code != 200:
            errors += 1
            return
        lat.append((cls, dt * 1000))
        questions += len(body["questions"])
        if "x-basal-batch-requests" in r.headers:
            batch.append(int(r.headers["x-basal-batch-requests"]))
        response = r.json()
        answers = response.get("answers", {})
        p = probs(response)
        if rid in seen:
            first_probs, first_answers = seen[rid]
            if answers != first_answers:
                answer_mismatches += 1
            for qn, d in p.items():
                for k, v in d.items():
                    if qn in first_probs and k in first_probs[qn]:
                        diff = max(diff, abs(v - first_probs[qn][k]))
        else:
            seen[rid] = (p, answers)

    rng = random.Random(1)
    arrivals, t = [], 0.0
    for _ in range(n):
        arrivals.append(t)
        if rate:
            t += rng.expovariate(rate)
    t_start = t0 = time.perf_counter()
    await asyncio.gather(*[one(i, at) for i, at in enumerate(arrivals)])
    t1 = max(done) if done else time.perf_counter()
    wall = t1 - t0
    lat_ms = [v for _, v in lat]
    g = gpu.window(t0, t1) if gpu else {}
    if g and questions:
        g["joules_per_decision"] = g["gpu_power_w"] * wall / questions
    return dict(concurrency=conc if rate is None else None, offered_requests_per_s=rate, requests=n, errors=errors,
                p50_ms=statistics.median(lat_ms) if lat_ms else None, p95_ms=quant(lat_ms, 0.95),
                p99_ms=quant(lat_ms, 0.99), requests_per_s=len(lat) / wall, decisions_per_s=questions / wall,
                max_prob_diff_vs_first_answer=diff, answers_differing_vs_first=answer_mismatches,
                mean_batch_requests=statistics.mean(batch) if batch else None,
                classes=per_class(lat), **g)


async def run(a):
    reqs = ([(r["id"], dict(r["request"], model=a.model), r.get("class", "all"))
             for r in map(json.loads, Path(a.requests).read_text().splitlines()) if r]
            if a.requests else requests_from_reference(a.reference, a.model))
    seen = {}
    rows = []
    gpu = Gpu() if a.gpu else None
    async with httpx.AsyncClient(timeout=1800, limits=httpx.Limits(max_connections=None)) as c:
        n_warm = len(reqs) if a.n_warm is None else a.n_warm
        await phase(c, a.url, reqs, n_warm, 1, seen, None)  # warm-up, first answers
        if a.n_seq:
            rows.append(dict(phase="sequential", **await phase(c, a.url, reqs, a.n_seq, 1, seen, gpu)))
            print(json.dumps(rows[-1]), flush=True)
        for conc in a.concurrency:
            rows.append(dict(phase="concurrent", **await phase(c, a.url, reqs, a.n_conc, conc, seen, gpu)))
            print(json.dumps(rows[-1]), flush=True)
        peak = max((r["requests_per_s"] for r in rows if r["phase"] == "concurrent"), default=None)
        for f in a.rate_fractions or []:
            rows.append(dict(phase="open-loop", rate_fraction=f,
                             **await phase(c, a.url, reqs, a.n_conc, None, seen, gpu, rate=f * peak)))
            print(json.dumps(rows[-1]), flush=True)
        for r in a.rates or []:
            rows.append(dict(phase="open-loop", **await phase(c, a.url, reqs, a.n_conc, None, seen, gpu, rate=r)))
            print(json.dumps(rows[-1]), flush=True)
    out = dict(tool="tools/bench/loadtest.py", url=a.url, source=a.requests or f"{a.reference}/bench.jsonl",
               distinct_requests=len(reqs), phases=rows)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    if a.answers_out:
        answers = [{"id": rid, "answers": answer} for rid, (_, answer) in seen.items()]
        Path(a.answers_out).write_text(json.dumps(answers, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    ap.add_argument("--reference", default="reports/reference-cuda-fp32")
    ap.add_argument("--requests", default=None)
    ap.add_argument("--model", default="basal-1.0-4.5B")
    ap.add_argument("--n-warm", dest="n_warm", type=int, default=None, help="warm-up requests (default: one pass)")
    ap.add_argument("--n-seq", dest="n_seq", type=int, default=100)
    ap.add_argument("--n-conc", dest="n_conc", type=int, default=400)
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 8, 16, 32],
                    help="closed-loop client counts; empty to run only sequential or fixed-rate phases")
    ap.add_argument("--rate-fractions", dest="rate_fractions", type=float, nargs="*", default=None,
                    help="open-loop phases at these fractions of the peak closed-loop requests/s")
    ap.add_argument("--rates", type=float, nargs="*", default=None, help="open-loop phases at these requests/s")
    ap.add_argument("--gpu", action="store_true", help="sample nvidia-smi during every phase (power, clocks, J/decision)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--answers-out", default=None, help="save first complete answers by request ID for cross-run comparison")
    a = ap.parse_args()
    if a.rate_fractions and not a.concurrency:
        raise SystemExit("--rate-fractions requires a closed-loop --concurrency; use --rates for fixed rates")
    if a.out and a.answers_out and Path(a.out).resolve() == Path(a.answers_out).resolve():
        raise SystemExit("--out and --answers-out must be different paths")
    for path in (a.out, a.answers_out):
        if path and Path(path).exists():
            raise SystemExit(f"{path} exists")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
