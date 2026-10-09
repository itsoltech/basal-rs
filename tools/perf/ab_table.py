"""Tables of a tools/perf/run-ab.sh run: python3 tools/perf/ab_table.py DIR [MODEL ...]

Per model: the single decision (`basal bench`: median, decisions/s) and the context ladder (median ms per request)
of A (base build), B (working tree) and B with BASAL_ATT=tc-pipe, with B / A and B+pipe / B.
"""
import json
import os
import sys

d = sys.argv[1]
models = sys.argv[2:] or ["4.5B", "max"]
V = ["A", "B", "Bpipe"]


def load(f):
    try:
        return json.load(open(os.path.join(d, f)))
    except (OSError, ValueError):
        return None


for m in models:
    print(f"\n## basal-1.5-{m}\n")
    print("| | A | B | B+pipe | B / A | B+pipe / B |")
    print("|---|---:|---:|---:|---:|---:|")
    b = {v: load(f"bench-{m}-{v}.json") for v in V}
    if all(b.values()):
        lat = [b[v]["lat2_ms"] for v in V]
        print(f"| single decision, median | {lat[0]:.1f} ms | {lat[1]:.1f} ms | {lat[2]:.1f} ms | "
              f"{lat[1] / lat[0]:.3f} | {lat[2] / lat[1]:.3f} |")
        ds = [b[v]["dec_s"] for v in V]
        print(f"| decisions/s | {ds[0]:.0f} | {ds[1]:.0f} | {ds[2]:.0f} | {ds[1] / ds[0]:.3f} | {ds[2] / ds[1]:.3f} |")
    r = {v: load(f"requests-{m}-{v}.json") for v in V}
    if all(r.values()):
        rows = {v: {q["id"]: q["median_ms"] for q in r[v]["requests"]} for v in V}
        for q in r["A"]["requests"]:
            k = q["id"]
            a, bb, p = rows["A"][k], rows["B"][k], rows["Bpipe"][k]
            print(f"| {k[4:]} | {a:.0f} ms | {bb:.0f} ms | {p:.0f} ms | {bb / a:.3f} | {p / bb:.3f} |")
