"""Deterministic mixed System One workload for load tests (latency and throughput, not a quality set).

  python3 tools/bench/make_mixed.py --model basal-1.5-max --out tools/bench/mixed.jsonl

Output lines {"id", "class", "request"}, shuffled with a fixed seed. Classes and shares:

  short-1q     40%  one choice question about a short state (the 44 basal-bench items, 2-10 options, PL and EN)
  short-multi  20%  2-3 questions about a short state (requests_fanout cx-*), sometimes all 14 questions of cx-01
  features     10%  basal-1.5 types and options (systemone_cases_1.5: multi, act, evidence, facts "auto")
  doc-1k-2k    20%  1-5 questions about a document of ~1k or ~2k tokens (6 distinct documents)
  doc-4k-8k     8%  1-5 questions about a document of ~4k or ~8k tokens (4 distinct documents)
  doc-16k       2%  1-3 questions about a document of ~16k tokens (2 distinct documents)

Documents are the synthetic contract of make_long_states.py with paragraph numbers shifted per document, so documents
differ while requests about one document share its state (as several clients asking about the same text).
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_long_states import KEY, QUESTIONS, para  # noqa: E402

SHARES = [("short-1q", 40), ("short-multi", 20), ("features", 10), ("doc-1k-2k", 20), ("doc-4k-8k", 8),
          ("doc-16k", 2)]


def document(n_par, off):
    k = n_par * 2 // 3
    out = []
    for i in range(1, n_par + 1):
        out.append(para(i + off))
        if i == k:
            out.append(KEY.format(i=i + off))
    return "\n".join(out), k + off


def doc_request(rng, model, n_par, off, max_q):
    st, k = document(n_par, off)
    names = rng.sample(list(QUESTIONS), rng.randint(1, max_q))
    qs = {}
    for name in names:
        q = dict(QUESTIONS[name])
        q["instructions"] = q["instructions"].format(k=k)
        qs[name] = q
    return {"model": model, "state": st, "questions": qs}


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="basal-1.5-max")
    ap.add_argument("--reference", default="reports/reference-1.5-max-fp32")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    bench = lines(Path(a.reference) / "bench.jsonl")
    fanout = lines("tools/reference/requests_fanout.jsonl")
    cases = lines("tools/reference/systemone_cases_1.5.jsonl")
    docs = {"doc-1k-2k": [(10, 0), (10, 200), (10, 400), (21, 600), (21, 800), (21, 1000)],
            "doc-4k-8k": [(42, 1200), (42, 1400), (84, 1600), (84, 1800)],
            "doc-16k": [(168, 2000), (168, 2200)]}
    max_q = {"doc-1k-2k": 5, "doc-4k-8k": 5, "doc-16k": 3}
    out = []
    for cls, share in SHARES:
        for j in range(a.n * share // 100):
            if cls == "short-1q":
                r = bench[j % len(bench)]
                q = {"type": "choice", "instructions": r["question"],
                     "criteria": {f"option_{k + 1}": o for k, o in enumerate(r["options"])}}
                req = {"model": a.model, "state": r["state"], "questions": {"q": q}}
            elif cls == "short-multi":
                pool = [f for f in fanout if f["id"].startswith("cx-")]
                f = fanout[5] if j % 10 == 9 else pool[j % len(pool)]  # every 10th: 14 questions about cx-01
                req = dict(f["request"], model=a.model)
            elif cls == "features":
                req = dict(cases[j % len(cases)]["request"], model=a.model)
            else:
                n_par, off = docs[cls][rng.randrange(len(docs[cls]))]
                req = doc_request(rng, a.model, n_par, off, max_q[cls])
            out.append({"id": f"{cls}/{j}", "class": cls, "request": req})
    rng.shuffle(out)
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"{len(out)} requests -> {a.out}")


if __name__ == "__main__":
    main()
