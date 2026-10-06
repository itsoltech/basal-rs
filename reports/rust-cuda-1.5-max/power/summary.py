"""Side by side summary of power/<label>/ runs (run.sh): long states, one-question HTTP, mixed workload per class.
Usage: python3 summary.py 250w 300w (from this directory)."""
import json
import sys
from pathlib import Path

labels = sys.argv[1:]
D = Path(__file__).parent


def load(label, name):
    p = D / label / name
    return json.loads(p.read_text()) if p.exists() else None


print("long states, median ms")
rows = {}
for lb in labels:
    d = load(lb, "long.json")
    for r in d["requests"] if d else []:
        rows.setdefault(r["id"], {})[lb] = r["median_ms"]
for rid, v in rows.items():
    print(f"  {rid:16}" + "".join(f"  {lb} {v.get(lb, float('nan')):8.1f}" for lb in labels))

for name in ["short.json", "mixed.json"]:
    print(name)
    for lb in labels:
        d = load(lb, name)
        if not d:
            continue
        for p in d["phases"]:
            load_desc = p.get("offered_requests_per_s") or p.get("concurrency")
            print(f"  {lb} {p['phase']:10} {str(load_desc):5} req/s {p['requests_per_s']:6.2f} dec/s "
                  f"{p['decisions_per_s']:6.2f} p50 {p['p50_ms']:7.0f} p95 {p['p95_ms']:7.0f} "
                  f"W {p.get('gpu_power_w', 0):5.0f} MHz {p.get('gpu_sm_mhz', 0):5.0f} "
                  f"J/dec {p.get('joules_per_decision', 0):5.1f} diff {p['max_prob_diff_vs_first_answer']}")
            if name == "mixed.json":
                for k, v in p["classes"].items():
                    print(f"        {k:12} p50 {v['p50_ms']:7.0f} p95 {v['p95_ms']:7.0f}")
