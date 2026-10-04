"""Paired A/B comparison of two `basal bench` configurations on the same GPU.

  python3 tools/bench/ab.py --out DIR --rounds 3 \
      --a "./target/release/basal bench --dtype f16 --reference R" \
      --b "env BASAL_ATT_NOSKIP=1 ./target/release/basal bench --dtype f16 --reference R"

Runs the two commands alternately in ABBA order (A B B A A B ...), each with `--out DIR/<variant>-<round>.json`,
so slow drift of the card (power cap, temperature) hits both variants alike. The comparison is paired per question:
for every measured item the median lat2 over the rounds of each variant, then the ratio B/A. Reported: median and
quartiles of the per-item ratio, how many items B is faster, median lat2 and dec_s of each variant and the spread
between rounds. Writes DIR/ab.json. Commands run through the shell; the directory must not exist.
"""
import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


def quantiles(xs):
    xs = sorted(xs)
    q = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return dict(p25=q(0.25), p50=statistics.median(xs), p75=q(0.75), min=xs[0], max=xs[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    order = []
    for r in range(a.rounds):
        order += ["a", "b"] if r % 2 == 0 else ["b", "a"]
    runs = {"a": [], "b": []}
    for i, v in enumerate(order):
        path = out / f"{v}-{len(runs[v])}.json"
        cmd = f"{getattr(a, v)} --out {shlex.quote(str(path))}"
        print(f"[{i + 1}/{len(order)}] {v}: {cmd}", flush=True)
        log = out / f"{v}-{len(runs[v])}.log"
        with open(log, "w") as f:
            if subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                sys.exit(f"command failed, see {log}")
        runs[v].append(json.loads(path.read_text()))
    per = {}
    for v in "ab":
        items = {}
        for run in runs[v]:
            for it in run["per_item"]:
                items.setdefault(it["index"], []).append(it["lat2_ms"])
        per[v] = {k: statistics.median(x) for k, x in items.items()}
    common = sorted(set(per["a"]) & set(per["b"]))
    ratio = [per["b"][k] / per["a"][k] for k in common]
    summary = dict(
        a=a.a, b=a.b, rounds=a.rounds, order="".join(order), items=len(common),
        ratio_b_over_a=quantiles(ratio), b_faster_items=sum(r < 1 for r in ratio),
        lat2_median_ms={v: [r["lat2_ms"] for r in runs[v]] for v in "ab"},
        lat2_p95_ms={v: [r["lat2_p95_ms"] for r in runs[v]] for v in "ab"},
        dec_s={v: [r["dec_s"] for r in runs[v]] for v in "ab"},
        per_item={str(k): dict(a=per["a"][k], b=per["b"][k]) for k in common},
    )
    (out / "ab.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ["items", "ratio_b_over_a", "b_faster_items", "lat2_median_ms", "dec_s"]},
                     indent=1))


if __name__ == "__main__":
    main()
