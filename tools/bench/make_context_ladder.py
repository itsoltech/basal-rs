"""Context ladder for latency measurements on long states (not a quality set), from the synthetic contract of
make_long_states.py:

  python3 tools/bench/make_context_ladder.py --model basal-1.5-4.5B --out DIR [--copies 6]

Writes
  DIR/requests.jsonl    System One requests with states of about 512 ... 16384 tokens (plus 1792, the prompt length of
                        the upstream author's speed-up table), 1 and 5 questions: `basal bench-requests`,
                        tools/reference/bench_requests.py
  DIR/ref-N/bench.jsonl `--copies` identical choice questions (4 options) over the state of N tokens, in the format of
                        a reference export: `basal profile --reference DIR/ref-N` (sections of the forward per length)
Token counts are approximate (~95 tokens per paragraph); the measured tools report the real ones.
"""
import argparse
import json
import os

from make_long_states import QUESTIONS, state

SIZES = [512, 1024, 1792, 2048, 4096, 8192, 16384]


def paragraphs(target):
    # the template, question and options take ~130 tokens
    return max(2, round((target - 130) / 95))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="basal-1.5-4.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=6)
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "requests.jsonl"), "w") as f:
        for target in a.sizes:
            n_par = paragraphs(target)
            k = max(1, n_par * 2 // 3)
            st = state(n_par, k)
            for nq in (1, 5):
                qs = {}
                for name, q in list(QUESTIONS.items())[:nq]:
                    q = dict(q)
                    q["instructions"] = q["instructions"].format(k=k)
                    qs[name] = q
                f.write(json.dumps({"id": f"ctx-{target}-q{nq}", "request": {"model": a.model, "state": st,
                                                                             "questions": qs}},
                                   ensure_ascii=False) + "\n")
            d = os.path.join(a.out, f"ref-{target}")
            os.makedirs(d, exist_ok=True)
            item = {"type": "choice", "state": st, "question": "Jaka jest kara umowna za każdy dzień opóźnienia w §1?",
                    "options": ["50 zł", "61 zł", "100 zł", "nie ma kary"], "gold": 1}
            with open(os.path.join(d, "bench.jsonl"), "w") as g:
                for i in range(a.copies):
                    g.write(json.dumps(dict(item, index=i, id=f"ctx-{target}-{i}"), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
