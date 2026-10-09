"""Time of variants against a base, from `basal bench` / `basal bench-requests` outputs named bench-V-R.json and
requests-V-R.json (variant V, repetition R) in DIR:

  python3 tools/perf/variant_table.py DIR BASE V1 [V2 ...] [--skip N]

Per variant the median over the repetitions (after the first N, warm-up on a cold GPU) of the per-run medians (single
decision: `lat2_ms`; throughput: `dec_s`; ladder: `median_ms` per request), and its ratio to BASE.
"""
import glob
import json
import os
import statistics
import sys

args = sys.argv[1:]
skip = 0
if "--skip" in args:
    i = args.index("--skip")
    skip = int(args[i + 1])
    del args[i:i + 2]
d, base, *vs = args
V = [base] + vs


def best(pattern, key):
    runs = {}
    for f in glob.glob(os.path.join(d, pattern)):
        rep = int(os.path.basename(f).rsplit("-", 1)[1].split(".")[0])
        if rep <= skip:
            continue
        try:
            j = json.load(open(f))
        except (OSError, ValueError):
            continue
        for k, x in key(j):
            runs.setdefault(k, []).append(x)
    return {k: statistics.median(xs) for k, xs in runs.items()}


rows = {}
for v in V:
    r = best(f"bench-{v}-*.json", lambda j: [
        ("single decision (ms)", j["lat2_ms"]), ("throughput (decisions/s)", j["dec_s"]),
    ])
    r.update(best(f"requests-{v}-*.json", lambda j: [(q["id"][4:], q["median_ms"]) for q in j["requests"]]))
    rows[v] = r
print("| | " + " | ".join(V) + " | " + " | ".join(f"{v} / {base}" for v in vs) + " |")
print("|---|" + "---:|" * (2 * len(V) - 1))
for k in rows[base]:
    vals = [rows[v].get(k) for v in V]
    cells = [f"{x:.1f}" if x is not None else "" for x in vals]
    ratios = [f"{x / vals[0]:.3f}" if x else "" for x in vals[1:]]
    print(f"| {k} | " + " | ".join(cells) + " | " + " | ".join(ratios) + " |")
