"""Choice questions with many options from public intent sets (test splits, Hugging Face datasets-server):

    python3 tools/choice-sets/build.py OUT.jsonl [--per-set 200] [--seed 0]

- banking77 (legacy-datasets/banking77, CC BY 4.0): 77 banking intents, English
- clinc150 (clinc/clinc_oos, config plus, CC BY 3.0): 150 intents, English; out-of-scope examples and label dropped
- massive-pl (mteb/amazon_massive_intent, config pl, from AmazonScience/massive, CC BY 4.0): 60 intents, Polish

Each line: id, dataset, state (the utterance), question, options (all intent names, underscores as spaces, in the
set's label order), gold (index of the labelled intent). Questions are a seeded random sample of the test split.
"""
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co"


def get(path, **q):
    url = f"{API}/{path}?{urllib.parse.urlencode(q)}"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 7:
                raise
            time.sleep(2 ** attempt)


def rows(dataset, config, split):
    n = get("info", dataset=dataset, config=config)["dataset_info"]["splits"][split]["num_examples"]
    out = []
    for off in range(0, n, 100):
        out += [r["row"] for r in get("rows", dataset=dataset, config=config, split=split, offset=off, length=100)["rows"]]
    return out


def names(dataset, config, field):
    return get("info", dataset=dataset, config=config)["dataset_info"]["features"][field]["names"]


def human(label):
    return label.replace("_", " ").strip().lower()


def main():
    args = sys.argv[1:]
    out = args[0]
    per_set = int(args[args.index("--per-set") + 1]) if "--per-set" in args else 200
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
    sets = []
    labels = names("legacy-datasets/banking77", "default", "label")
    data = [(r["text"], labels[r["label"]]) for r in rows("legacy-datasets/banking77", "default", "test")]
    sets.append(("banking77", labels, data, "What is the customer asking about?"))
    labels = names("clinc/clinc_oos", "plus", "intent")
    data = [(r["text"], labels[r["intent"]]) for r in rows("clinc/clinc_oos", "plus", "test")]
    labels = [l for l in labels if l != "oos"]
    data = [(t, l) for t, l in data if l != "oos"]
    sets.append(("clinc150", labels, data, "What does the user want?"))
    data = [(r["text"], r["label"]) for r in rows("mteb/amazon_massive_intent", "pl", "test")]
    labels = sorted({l for _, l in data})
    sets.append(("massive-pl", labels, data, "Czego chce użytkownik?"))
    rng = random.Random(seed)
    with open(out, "w") as f:
        for name, labels, data, question in sets:
            options = [human(l) for l in labels]
            assert len(set(options)) == len(options), name
            for i, (text, label) in enumerate(rng.sample(data, per_set)):
                rec = {"id": f"{name}-{i}", "dataset": name, "state": text, "question": question,
                       "options": options, "gold": labels.index(label)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{name}: {len(options)} options, {len(data)} test examples, {per_set} sampled", file=sys.stderr)


main()
