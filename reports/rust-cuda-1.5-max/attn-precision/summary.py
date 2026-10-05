"""Summary of the attention precision variants (run.sh): agreement with upstream FP32 and with the runtime FP32 on
long states, latency on long states. Prints a JSON object; run from the repository root."""
import json
from pathlib import Path

O = Path("reports/rust-cuda-1.5-max/attn-precision")
VARIANTS = ["tc", "tc-pv1", "tc-qk1", "tc-f16"]


def decision(a):
    """Comparable decision of one answer object (choice key, noul side, score argmax, multi selection)."""
    t = a.get("type")
    if t == "noul":
        return a["noul"] >= 0.5
    if t == "multi":
        return tuple(sorted(a.get("selected", [])))
    if t == "act":
        return a.get("action")
    p = a.get("probabilities", {})
    return max(p, key=p.get) if p else None


def numbers(a):
    out = {}
    if "noul" in a:
        out["noul"] = a["noul"]
    for k, v in a.get("probabilities", {}).items():
        out["p:" + k] = v
    return out


def long_rows(name):
    return {r["id"]: r for r in json.loads((O / f"long-{name}.json").read_text())["requests"]}


ref = long_rows("f32")
summary = {}
for v in VARIANTS:
    c = json.loads((O / f"compare-upstream-fp32-vs-{v}.json").read_text())
    b = c["bench"]
    s = {"bench_decisions_cal": b["argmax_agreement_cal"], "bench_max_logit_diff": b["letter_logit_diff"]["max"],
         "bench_max_cal_prob_diff": b["cal_prob_diff"]["max"], "bench_max_tv_cal": b["tv_cal"]["max"]}
    so = c["systemone"]["numeric_identical_tokens"]
    s["systemone_max_logit_diff"] = so["letter_logit_diff"]["max"]
    for name in ["", "cases2-"]:
        cc = json.loads((O / f"compare-{name}upstream-fp32-vs-{v}.json").read_text())["systemone"]["cases"]
        items = [it for case in cc for it in case["items"]]
        ok = sum(all(x for k, x in it["answer"].items() if k.endswith("_same")) for it in items)
        s[f"systemone_{name}answers_same"] = f"{ok}/{len(items)}"
    rows = long_rows(v)
    lat, maxdiff, same, total = {}, 0.0, 0, 0
    for rid, r in rows.items():
        lat[rid] = round(r["median_ms"], 1)
        for q, a in r["answers"].items():
            fa = ref[rid]["answers"][q]
            total += 1
            same += decision(a) == decision(fa)
            na, nf = numbers(a), numbers(fa)
            for k in na:
                if k in nf:
                    maxdiff = max(maxdiff, abs(na[k] - nf[k]))
    s["long_vs_runtime_f32"] = {"same_decisions": f"{same}/{total}", "max_prob_diff": maxdiff}
    s["long_median_ms"] = lat
    summary[v] = s
print(json.dumps(summary, indent=1))
