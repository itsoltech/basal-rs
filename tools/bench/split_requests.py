"""Split multi-question System One requests into one request per question (same state), for load tests in which
concurrent clients ask about the same state.

  python3 tools/bench/split_requests.py tools/bench/long_states.jsonl --ids long-2000-q5 long-4000-q5 \
      --out tools/bench/shared_states.jsonl

Output lines {"id", "request"} in the order request 1 question 1, request 1 question 2, ... (consecutive requests of
a cycled load test share the state).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("--ids", nargs="*", default=None, help="only these request ids (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for line in Path(a.requests).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if a.ids and r["id"] not in a.ids:
            continue
        for name, q in r["request"]["questions"].items():
            req = dict(r["request"], questions={name: q})
            out.append({"id": f"{r['id']}/{name}", "request": req})
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    print(f"{len(out)} requests -> {a.out}")


if __name__ == "__main__":
    main()
