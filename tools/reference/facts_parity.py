"""Parity of the Rust port of upstream basal/facts.py (crates/basal-core/src/facts.rs) with upstream `inject`.

  python tools/reference/facts_parity.py [--corpus tools/reference/facts_corpus.jsonl] [--basal target/release/basal]
                                         [--upstream .baseline/upstream-1.5] [--show 5]

Runs `basal facts` on the corpus and upstream `basal.facts.inject` on the same states, then prints how many outputs
are byte-identical and the first differences. A state that makes upstream raise counts as identical when the Rust
port returns the same exception text (`str(e)`, the upstream server's 422 `error`). Non-string states are passed as
`json.dumps(state, ensure_ascii=False)`, as the upstream server does. facts.py needs only the standard library; run
this with the upstream interpreter (Python 3.12, e.g. .baseline/upstream/.venv/bin/python). Exit status 1 on any
difference.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_upstream(root):
    spec = importlib.util.spec_from_file_location("upstream_facts", Path(root) / "basal" / "facts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upstream_result(mod, state):
    try:
        return {"out": mod.inject(state)}
    except Exception as e:  # noqa: BLE001 - the server turns any exception into a 422 with str(e)
        return {"error": str(e)}


def first_diff(a, b):
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return i, a[max(0, i - 60): i + 60], b[max(0, i - 60): i + 60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "tools/reference/facts_corpus.jsonl"))
    ap.add_argument("--basal", default=str(ROOT / "target/release/basal"))
    ap.add_argument("--upstream", default=str(ROOT / ".baseline/upstream-1.5"))
    ap.add_argument("--show", type=int, default=5, help="differences to print")
    a = ap.parse_args()

    mod = load_upstream(a.upstream)
    recs = [json.loads(line) for line in Path(a.corpus).read_text(encoding="utf-8").split("\n") if line.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
        with src.open("w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                f.write(json.dumps({"id": i, "state": r["state"]}, ensure_ascii=False) + "\n")
        subprocess.run([a.basal, "facts", "--input", str(src), "--out", str(out)], check=True)
        rust = {r["id"]: r for r in map(json.loads, filter(None, out.read_text(encoding="utf-8").split("\n")))}

    same, with_facts, errors, diffs = 0, 0, 0, []
    for i, r in enumerate(recs):
        st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
        want = upstream_result(mod, st)
        got = {k: v for k, v in rust.get(i, {}).items() if k in ("out", "error")}
        if got == want:
            same += 1
            with_facts += "out" in want and want["out"] != st
            errors += "error" in want
        else:
            diffs.append((r.get("id", i), want, got))
    print(f"identical: {same}/{len(recs)} (with facts: {with_facts}, upstream errors reproduced: {errors})")
    for rid, want, got in diffs[: a.show]:
        print(f"--- {rid}")
        if "out" in want and "out" in got:
            k, w, g = first_diff(want["out"], got["out"])
            print(f"first difference at char {k}\n  upstream: {w!r}\n  rust:     {g!r}")
        else:
            print(f"  upstream: {want!r}\n  rust:     {got!r}")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
