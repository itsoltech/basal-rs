"""Tables of the report from the compare-*.json and bench-*.json files of this directory:
  python3 reports/metal-m2-max-1.5/summary.py
"""
import json
from pathlib import Path

O = Path(__file__).parent
MODELS = ["basal-1.5-max", "basal-1.5-4.5B", "basal-1.5-mini"]
VARIANTS = [("rust", "f32"), ("rust", "f16-single"), ("rust", "f16-tree"), ("upstream", "mlx"), ("upstream", "mlx-f16"),
            ("upstream", "mlx-q8")]


def pl(x, f):
    return format(x, f).replace(".", ",")


def endpoint(c):
    """/v1/basal of the runtime (answer_upstream) against upstream Server.decide: questions with the same fields."""
    qs = [q["diff"] for case in c.get("upstream_endpoint", []) for q in case.get("questions", [])]
    if not qs:
        return "", ""
    same = sum(q.get("fields_same", False) and q.get("type_same", False) for q in qs)
    return f"{same}/{len(qs)}", pl(max(q["max_abs_prob_diff"] for q in qs), ".4f")


print("| Model | Wariant | Decyzje 44 | Maks. różnica logitu | Maks. różnica p (kal.) | /v1/basal: pytania | Maks. różnica p |")
print("|---|---|---:|---:|---:|---:|---:|")
for m in MODELS:
    for who, v in VARIANTS:
        f = O / f"compare-upstream-fp32-vs-{who}-{m}-{v}.json"
        if not f.exists():
            continue
        c = json.loads(f.read_text())
        b = c["bench"]
        name = {"rust": "basal-rs ", "upstream": "upstream "}[who] + v
        ep, epd = endpoint(c)
        print(f"| {m} | {name} | {b['argmax_agreement_cal']} | {pl(b['letter_logit_diff']['max'], '.3f')} | "
              f"{pl(b['cal_prob_diff']['max'], '.4f')} | {ep} | {epd} |")

print()
print("| Model | Wariant | Przebieg | lat2 mediana | lat2 p95 | lat1 | Decyzje/s |")
print("|---|---|---:|---:|---:|---:|---:|")
for m in MODELS:
    rows = []
    for f in sorted(O.glob(f"bench-rust-{m}-f16-*.json")):
        r = json.loads(f.read_text())
        rows.append(("basal-rs f16", f.stem.rsplit("-", 1)[1], r["lat2_ms"], r["lat2_p95_ms"], r["lat1_ms"], r["dec_s"]))
    for f in sorted(O.glob(f"bench-upstream-{m}-mlx-*.json")):
        rs = json.loads(f.read_text())
        for r in rs if isinstance(rs, list) else [rs]:  # basal-bench: list of modes; bench_mlx_dtype.py: one
            rows.append((f"upstream {r['mode']}", f.stem.rsplit("-", 1)[1], r["lat2_ms"], r["lat2_p95_ms"], r["lat1_ms"],
                         r["dec_s"]))
    for name, k, l2, p95, l1, ds in sorted(rows, key=lambda x: (x[0], x[1])):
        print(f"| {m} | {name} | {k} | {l2:.0f} ms | {p95:.0f} ms | {l1:.0f} ms | {pl(ds, '.2f')} |")
