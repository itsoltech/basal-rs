"""Tables of a choice-set evaluation: basal eval-choice-set outputs against the labels and against TypeSafe:

    python3 tools/choice-sets/report.py SET.jsonl TYPESAFE.jsonl MODEL=DIR [MODEL=DIR ...]

Per model, strategy and set: accuracy against the labels, agreement with the TypeSafe answer, accuracy on the
questions TypeSafe answers correctly, label in the final group, mean NLL of the label, median ms and prompts.
"""
import json
import math
import statistics
import sys

set_items = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
ts = {r["id"]: r for r in map(json.loads, open(sys.argv[2])) if "error" not in r}
sets = sorted({it["dataset"] for it in set_items.values()})


def row(name, recs):
    cells = []
    for d in sets:
        xs = [r for r in recs if r["dataset"] == d]
        n = len(xs)
        acc = sum(r["top"] == r["gold"] for r in xs)
        agree = sum(r["top"] == ts[r["id"]]["top"] for r in xs if r["id"] in ts)
        jev_ok = [r for r in xs if r["id"] in ts and ts[r["id"]]["top"] == r["gold"]]
        acc_jev_ok = sum(r["top"] == r["gold"] for r in jev_ok)
        fin = [r for r in xs if r.get("gold_in_final") is not None]
        fin_ok = sum(bool(r["gold_in_final"]) for r in fin)
        nll = statistics.mean(-math.log(max(r["p_gold"] if "p_gold" in r else r["p"][r["gold"]], 1e-6)) for r in xs)
        ms = statistics.median(r["ms"] for r in xs) if "ms" in xs[0] else float("nan")
        pr = statistics.median(r["prompts"] for r in xs) if "prompts" in xs[0] else 0
        cells.append(f"{acc}/{n} | {agree}/{n} | {acc_jev_ok}/{len(jev_ok)} | "
                     + (f"{fin_ok}/{len(fin)}" if fin else "") + f" | {nll:.2f} | {ms:.0f} | {pr:.0f}")
    return f"| {name} | " + " | ".join(cells) + " |"


head = "| model, strategia | " + " | ".join(
    f"{d}: trafność | zgodność z jev | trafność gdy jev trafia | etykieta w finale | NLL | ms | prompty" for d in sets) + " |"
print(head)
print("|" + "---|" * (1 + 7 * len(sets)))
print(row("TypeSafe jev", [dict(r, gold=r["gold"]) for r in ts.values()]))
for arg in sys.argv[3:]:
    model, d = arg.split("=", 1)
    summary = json.load(open(f"{d}/summary.json"))
    done = set()
    for s in summary["strategies"]:
        if s["strategy"] in done:
            continue
        done.add(s["strategy"])
        recs = [json.loads(l) for l in open(f"{d}/{s['strategy']}.jsonl")]
        for r in recs:
            r["id"] = r["id"].strip('"') if isinstance(r["id"], str) else r["id"]
        print(row(f"{model} {s['strategy']}", recs))
