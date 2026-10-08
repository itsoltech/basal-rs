"""Tables of a tools/perf/run-context.sh run:

    python3 tools/perf/context_table.py DIR [MODEL ...]

Per model (4.5B, max): whole requests by state length and number of questions, basal-rs against upstream (median ms),
and the sections of the basal-rs forward per length (`basal profile`, one question).
"""
import json
import os
import sys

d = sys.argv[1]
models = sys.argv[2:] or ["4.5B", "max"]


def load(f):
    try:
        return json.load(open(os.path.join(d, f)))
    except (OSError, ValueError):
        return None


def fmt(x, n=0):
    return f"{x:.{n}f}".replace(".", ",") if x is not None else "–"


for m in models:
    r = load(f"requests-rust-{m}.json")
    u = load(f"requests-upstream-{m}.json")
    if not r:
        continue
    ups = {}
    if u:
        for q in u.get("requests", []):
            ups[q["id"]] = q
    print(f"\n## basal-1.5-{m}: whole requests (median ms)\n")
    print("| State (target) | Questions | Prompt tokens (both orders) | basal-rs tree tokens | upstream | basal-rs | ratio |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for q in r["requests"]:
        target = q["id"].split("-")[1]
        uq = ups.get(q["id"])
        um = uq.get("median_ms") if uq else None
        ratio = um / q["median_ms"] if um else None
        print(f"| {target} | {q['questions']} | {q['input_tokens']} | {q['packed_tokens_shared_tree']} | {fmt(um)} | "
              f"{fmt(q['median_ms'])} | {fmt(ratio, 2) + '×' if ratio else '–'} |")
    rows = []
    for f in sorted(os.listdir(d)):
        if f.startswith(f"profile-{m}-") and f.endswith(".json") and f[len(f"profile-{m}-"):-5].isdigit():
            p = load(f)
            if p:
                rows.append((int(f[len(f"profile-{m}-"):-5]), p))
    if rows:
        rows.sort()
        keys = sorted({k for _, p in rows for k in p["sections_ms_per_item"]})
        gemm = [k for k in keys if "proj" in k]
        print(f"\n## basal-1.5-{m}: forward sections, one question (ms, profiled)\n")
        print("| State | Packed tokens | Total | GEMM | Attention | Other | GEMM share | Attention share |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for n, p in rows:
            s = p["sections_ms_per_item"]
            g = sum(s.get(k, 0) for k in gemm)
            a = s.get("attention", 0)
            t = p["ms_per_item_profiled"]
            print(f"| {n} | {fmt(p['mean_packed_tokens_with_template'])} | {fmt(t, 1)} | {fmt(g, 1)} | {fmt(a, 1)} | "
                  f"{fmt(t - g - a, 1)} | {fmt(100 * g / t)}% | {fmt(100 * a / t)}% |")
