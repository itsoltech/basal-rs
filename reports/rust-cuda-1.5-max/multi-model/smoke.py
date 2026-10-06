"""Checks of a multi-model `basal serve` (run.sh): model listing, routing by the "model" field on both endpoints,
default model, unknown and missing model errors. Writes the answers to --out (to compare with a single-model server).

  python3 smoke.py --url http://127.0.0.1:8100 --models basal-1.5-max basal-1.5-4.5B basal-1.5-mini --out FILE
"""
import argparse
import json

import httpx

REQ = {
    "state": "Zgłoszenie: klient prosi o zwrot pieniędzy za uszkodzony telefon, kupiony 3 dni temu.",
    "questions": {
        "dept": {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
                 "criteria": {"returns": "Zwroty", "tech": "Wsparcie techniczne", "sales": "Sprzedaż"}},
        "urgent": {"type": "noul", "instructions": "Czy sprawa jest pilna?"},
        "tone": {"type": "score", "instructions": "Jak bardzo klient jest zdenerwowany?",
                 "criteria": ["spokojny", "zirytowany", "bardzo zdenerwowany"]},
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--default", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    default = a.default or a.models[0]
    c = httpx.Client(base_url=a.url, timeout=600)
    checks, answers = [], {}

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(("ok   " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))

    listed = [m["name"] for m in c.get("/v1/models").json()["models"]]
    check("GET /v1/models lists every model", listed == a.models, str(listed))
    h = c.get("/health").json()
    check("GET /health", h.get("status") == "ok" and h.get("default_model") == default, str(h))
    for m in a.models:
        r = c.post("/v1/systemone", json=dict(REQ, model=m))
        check(f"/v1/systemone {m}", r.status_code == 200 and r.json().get("model") == m, r.text[:200])
        answers[f"systemone/{m}"] = r.json()
        r = c.post("/v1/basal", json=dict(REQ, model=m))
        check(f"/v1/basal model={m}", r.status_code == 200 and r.json().get("model") == m, r.text[:200])
        answers[f"basal/{m}"] = r.json()
    r = c.post("/v1/basal", json=REQ)
    check("/v1/basal without model -> default", r.status_code == 200 and r.json().get("model") == default,
          r.text[:200])
    same = all(r.json()["answers"][q] == answers[f"basal/{default}"]["answers"][q] for q in REQ["questions"])
    check("/v1/basal without model answers as the default model", same)
    r = c.post("/v1/systemone", json=dict(REQ, model="no-such-model"))
    err = r.json().get("detail", [{}])[0]
    check("/v1/systemone unknown model -> 422 unknown_model", r.status_code == 422 and err.get("type") == "unknown_model"
          and all(m in err.get("msg", "") for m in a.models), r.text[:300])
    r = c.post("/v1/basal", json=dict(REQ, model="no-such-model"))
    check("/v1/basal unknown model -> 422 {error}", r.status_code == 422 and "unknown model" in r.json().get("error", ""),
          r.text[:300])
    r = c.post("/v1/systemone", json=REQ)
    err = r.json().get("detail", [{}])[0]
    check("/v1/systemone without model -> 422 Field required", r.status_code == 422 and err.get("loc") == ["body", "model"],
          r.text[:300])
    for k in list(answers):
        for drop in ("usage",):
            answers[k].pop(drop, None)  # latency_ms differs between runs
    json.dump({"checks": checks, "answers": answers}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"{sum(c['ok'] for c in checks)}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
