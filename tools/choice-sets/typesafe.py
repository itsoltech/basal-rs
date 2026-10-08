"""Answers of the TypeSafe API to a choice set (tools/choice-sets/build.py):

    python3 tools/choice-sets/typesafe.py ENV_FILE SET.jsonl OUT.jsonl [--model jev-latest] [--threads 4]

ENV_FILE holds TYPESAFE_API_KEY. Each output line: id, dataset, gold, top (index of the answered choice), p (the
probabilities in option order), confidence, model, ms, usage; or status and error for a failed request.
"""
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

args = sys.argv[1:]
env = dict(l.strip().split("=", 1) for l in open(args[0]) if "=" in l and not l.startswith("#"))
KEY = env["TYPESAFE_API_KEY"].strip().strip('"').strip("'")
MODEL = args[args.index("--model") + 1] if "--model" in args else "jev-latest"
THREADS = int(args[args.index("--threads") + 1]) if "--threads" in args else 4
items = [json.loads(l) for l in open(args[1])]


def ask(it):
    body = {"model": MODEL, "state": it["state"], "questions": {"q": {
        "type": "choice", "instructions": it["question"], "criteria": {o: None for o in it["options"]}}}}
    req = urllib.request.Request("https://api.typesafe.ai/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for attempt in range(6):
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            a = out["answers"]["q"]
            p = [a["probabilities"].get(o, 0.0) for o in it["options"]]
            return {"id": it["id"], "dataset": it["dataset"], "gold": it["gold"], "top": it["options"].index(a["choice"]),
                    "p": p, "confidence": a["confidence"], "model": out["model"], "ms": round((time.time() - t) * 1000),
                    "usage": out["usage"]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            return {"id": it["id"], "dataset": it["dataset"], "status": e.code, "error": e.read().decode()[:500]}


with open(args[2], "w") as f, concurrent.futures.ThreadPoolExecutor(THREADS) as ex:
    for i, rec in enumerate(ex.map(ask, items)):
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(i, file=sys.stderr, flush=True)
