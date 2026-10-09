"""Decision questions with at most 10 options from public labelled sets, for the comparison with upstream FP32
(tools/reference/export_reference.py --questions, then basal export / basal compare):

    python3 tools/decision-sets/build.py OUT.jsonl [--per-set 100] [--seed 0]

Each line has the fields of a basal-bench question: id, type (choice, noul, score), state, question, options, gold,
plus dataset and row (index in the source split). Test splits (BoolQ: validation) of Hugging Face datasets-server:

| set                 | language | type   | options | source                                       |
|---------------------|----------|--------|---------|----------------------------------------------|
| polemo2-in          | pl       | choice | 4       | allegro/klej-polemo2-in (KLEJ)               |
| allegro-reviews     | pl       | score  | 5       | allegro/klej-allegro-reviews (KLEJ)          |
| cdsc-e              | pl       | choice | 3       | allegro/klej-cdsc-e (KLEJ)                   |
| dyk                 | pl       | noul   | 2       | allegro/klej-dyk (KLEJ)                      |
| cbd                 | pl       | noul   | 2       | allegro/klej-cbd (KLEJ)                      |
| ag-news             | en       | choice | 4       | fancyzhx/ag_news                             |
| emotion             | en       | choice | 6       | dair-ai/emotion (split)                      |
| boolq               | en       | noul   | 2       | google/boolq                                 |
| sst5                | en       | score  | 5       | SetFit/sst5                                  |

Yes/no sets are sampled half from each label. The texts stay with their licences (some non-commercial); the set is
rebuilt from the sources with the same seed instead of being stored in the repository.
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
            if e.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(2 ** attempt)


def rows(dataset, config, split):
    n = get("info", dataset=dataset, config=config)["dataset_info"]["splits"][split]["num_examples"]
    out = []
    for off in range(0, n, 100):
        out += [r["row"] for r in get("rows", dataset=dataset, config=config, split=split, offset=off, length=100)["rows"]]
    return out


POLEMO = {"__label__meta_plus_m": 0, "__label__meta_minus_m": 1, "__label__meta_zero": 2, "__label__meta_amb": 3}
CDSC = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}
YES_PL, YES_EN = ["Tak", "Nie"], ["Yes", "No"]


def sets():
    """(name, type, question, options, [(state, gold, row)]) for every set."""
    r = rows("allegro/klej-polemo2-in", "default", "test")
    yield ("polemo2-in", "choice", "Jaki jest wydźwięk tej opinii?",
           ["pozytywny", "negatywny", "neutralny", "niejednoznaczny"],
           [(x["sentence"], POLEMO[x["target"]], i) for i, x in enumerate(r)])
    r = rows("allegro/klej-allegro-reviews", "default", "test")
    yield ("allegro-reviews", "score", "Jaką ocenę wystawił autor tej recenzji?",
           ["1 – bardzo zła", "2 – zła", "3 – przeciętna", "4 – dobra", "5 – bardzo dobra"],
           [(x["text"], int(round(float(x["rating"]))) - 1, i) for i, x in enumerate(r)])
    r = rows("allegro/klej-cdsc-e", "default", "test")
    yield ("cdsc-e", "choice", "Jak zdanie B ma się do zdania A?",
           ["zdanie B wynika ze zdania A", "zdanie B przeczy zdaniu A", "zdanie B nie wynika ze zdania A ani mu nie przeczy"],
           [(f"Zdanie A: {x['sentence_A']}\nZdanie B: {x['sentence_B']}", CDSC[x["entailment_judgment"]], i)
            for i, x in enumerate(r)])
    r = rows("allegro/klej-dyk", "default", "test")
    yield ("dyk", "noul", "Czy podana odpowiedź odpowiada na pytanie?", YES_PL,
           [(f"Pytanie: {x['question'].strip(chr(34))}\nOdpowiedź: {x['answer'].strip().strip(chr(34))}",
             0 if int(x["target"]) == 1 else 1, i) for i, x in enumerate(r)])
    r = rows("allegro/klej-cbd", "default", "test")
    yield ("cbd", "noul", "Czy ten wpis zawiera obraźliwe treści lub cyberprzemoc?", YES_PL,
           [(x["sentence"], 0 if int(x["target"]) == 1 else 1, i) for i, x in enumerate(r)])
    r = rows("fancyzhx/ag_news", "default", "test")
    yield ("ag-news", "choice", "What is the topic of this news article?",
           ["World", "Sports", "Business", "Science and technology"],
           [(x["text"], int(x["label"]), i) for i, x in enumerate(r)])
    r = rows("dair-ai/emotion", "split", "test")
    yield ("emotion", "choice", "Which emotion does the author express?",
           ["sadness", "joy", "love", "anger", "fear", "surprise"],
           [(x["text"], int(x["label"]), i) for i, x in enumerate(r)])
    r = rows("google/boolq", "default", "validation")
    yield ("boolq", "noul", None, YES_EN,
           [((x["passage"], x["question"][:1].upper() + x["question"][1:] + "?"), 0 if x["answer"] else 1, i)
            for i, x in enumerate(r)])
    r = rows("SetFit/sst5", "default", "test")
    yield ("sst5", "score", "How positive is this movie review?",
           ["very negative", "negative", "neutral", "positive", "very positive"],
           [(x["text"], int(x["label"]), i) for i, x in enumerate(r)])


def main():
    args = sys.argv[1:]
    out = args[0]
    per_set = int(args[args.index("--per-set") + 1]) if "--per-set" in args else 100
    seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
    rng = random.Random(seed)
    with open(out, "w") as f:
        for name, qtype, question, options, data in sets():
            if qtype == "noul":
                pick = []
                for g in (0, 1):
                    pick += rng.sample([d for d in data if d[1] == g], per_set // 2)
                rng.shuffle(pick)
            else:
                pick = rng.sample(data, per_set)
            for k, (state, gold, row) in enumerate(pick):
                q = question
                if isinstance(state, tuple):  # BoolQ: the question is per row
                    state, q = state
                rec = {"id": f"{name}-{k}", "dataset": name, "row": row, "type": qtype, "state": state,
                       "question": q, "options": options, "gold": gold}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{name}: {qtype}, {len(options)} options, {len(data)} rows, {len(pick)} sampled", file=sys.stderr)


main()
