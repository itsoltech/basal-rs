"""Deterministic long-state System One requests for latency measurements (not a quality set).

  python3 tools/bench/make_long_states.py --model basal-1.5-max --out tools/bench/long_states.jsonl

States of about 1k, 2k, 4k, 8k and 16k tokens are built from numbered paragraphs of a synthetic Polish contract
(every paragraph different: numbers, dates and amounts vary with the paragraph index), with 1 or 5 questions per
request. One key fact is placed at a fixed share of the document so the questions have an answer.
"""
import argparse
import json

PARA = ("§{i}. Strony ustalają, że w okresie rozliczeniowym nr {i} Wykonawca dostarczy {q} sztuk towaru "
        "w cenie {p},{c:02d} zł netto za sztukę, w terminie do {d} {m} 2026 r. Za każdy dzień opóźnienia "
        "Zamawiający może naliczyć karę umowną w wysokości {k} zł. Reklamacje dotyczące okresu {i} "
        "zgłasza się pisemnie w ciągu {r} dni od dostawy.")
MONTHS = ["stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca", "sierpnia", "września",
          "października", "listopada", "grudnia"]
KEY = "§{i}a. Wyjątkowo w tym okresie Zamawiający zrzekł się prawa do naliczania kar umownych."


def para(i):
    return PARA.format(i=i, q=10 + i * 7 % 90, p=100 + i * 37 % 900, c=i * 13 % 100, d=1 + i % 28,
                       m=MONTHS[i % 12], k=50 + i * 11 % 450, r=7 + i % 21)


def state(n_par, key_at):
    out = []
    for i in range(1, n_par + 1):
        out.append(para(i))
        if i == key_at:
            out.append(KEY.format(i=i))
    return "\n".join(out)


QUESTIONS = {
    "kara": {"type": "noul", "instructions": "Czy w okresie wskazanym w §{k}a Zamawiający może naliczyć karę umowną?"},
    "forma": {"type": "choice", "instructions": "W jakiej formie zgłasza się reklamacje?",
              "criteria": {"a": "Pisemnie", "b": "Telefonicznie", "c": "Dowolnie"}, "option_keys": "hide"},
    "ryzyko": {"type": "score", "instructions": "Jak wysokie jest ryzyko sporu o kary umowne?",
               "criteria": ["niskie", "średnie", "wysokie"]},
    "termin": {"type": "noul", "instructions": "Czy reklamacje zgłasza się w terminie dłuższym niż 3 dni?"},
    "tematy": {"type": "multi", "instructions": "Czego dotyczy dokument?",
               "criteria": ["dostawy towaru", "kary umowne", "wynagrodzenie pracowników"]},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="basal-1.5-max")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w") as f:
        # ~95 tokens per paragraph
        for target, n_par in [(1000, 10), (2000, 21), (4000, 42), (8000, 84), (16000, 168)]:
            k = max(1, n_par * 2 // 3)
            st = state(n_par, k)
            for nq in (1, 5):
                qs = {}
                for name, q in list(QUESTIONS.items())[:nq]:
                    q = dict(q)
                    q["instructions"] = q["instructions"].format(k=k)
                    qs[name] = q
                f.write(json.dumps({"id": f"long-{target}-q{nq}", "request": {"model": a.model, "state": st,
                                                                              "questions": qs}},
                                   ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
