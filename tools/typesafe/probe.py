"""Edge cases of POST /v1/systemone sent to a server: the TypeSafe API or basal serve, to compare their statuses and
bodies (reports/typesafe-api-edge-cases).

    python3 tools/typesafe/probe.py ENV_FILE BASE_URL MODEL OUT.json

ENV_FILE holds TYPESAFE_API_KEY (sent only to *.typesafe.ai; other servers get a placeholder key). OUT.json keeps
status, time and body per case, without response headers.
"""
import json, sys, time, urllib.request, urllib.error
env = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l and not l.startswith("#"))
KEY = env["TYPESAFE_API_KEY"].strip().strip('"').strip("'")
BASE = sys.argv[2]
MODEL = sys.argv[3]
OUT = sys.argv[4]
AUTH = {"Authorization": f"Bearer {KEY}"} if "typesafe.ai" in BASE else {"Authorization": "Bearer local"}
ST = "Klient pisze: zapłaciłem dwa razy za to samo zamówienie, proszę o zwrot."
opts = lambda n: {f"o{i}": None for i in range(n)}
def Q(q, state=ST, extra=None):
    b = {"model": MODEL, "state": state, "questions": {"q": q} if q is not None else {}}
    if extra: b.update(extra)
    return b
cases = [
 ("choice_0", Q({"type":"choice","instructions":"Temat?","criteria":{}})),
 ("choice_1", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":None}})),
 ("choice_1_desc", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":"płatności"}})),
 ("choice_2_null", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":None,"shipping":None}})),
 ("choice_11", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":"płatności",**opts(10)}})),
 ("choice_255", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":"płatności",**opts(254)}})),
 ("choice_256", Q({"type":"choice","instructions":"Temat?","criteria":{"billing":"płatności",**opts(255)}})),
 ("choice_criteria_list", Q({"type":"choice","instructions":"Temat?","criteria":["a","b"]})),
 ("choice_no_criteria", Q({"type":"choice","instructions":"Temat?"})),
 ("score_0", Q({"type":"score","instructions":"Pilność?","criteria":[]})),
 ("score_1", Q({"type":"score","instructions":"Pilność?","criteria":["pilne"]})),
 ("score_2", Q({"type":"score","instructions":"Pilność?","criteria":["może czekać","pilne"]})),
 ("score_10", Q({"type":"score","instructions":"Pilność?","criteria":[f"poziom {i}" for i in range(10)]})),
 ("score_11", Q({"type":"score","instructions":"Pilność?","criteria":[f"poziom {i}" for i in range(11)]})),
 ("score_null_item", Q({"type":"score","instructions":"Pilność?","criteria":["niska",None,"wysoka"]})),
 ("score_objects", Q({"type":"score","instructions":"Pilność?","criteria":[{"level":"niska"},["średnia"],"wysoka"]})),
 ("score_map", Q({"type":"score","instructions":"Pilność?","criteria":{"niska":"x","wysoka":"y"}})),
 ("noul_no_instr", Q({"type":"noul"})),
 ("noul_criteria_null", Q({"type":"noul","instructions":"Czy chodzi o płatność?","criteria":None})),
 ("noul_only_true", Q({"type":"noul","instructions":"Czy chodzi o płatność?","criteria":{"true":"płatność"}})),
 ("noul_empty_true", Q({"type":"noul","instructions":"Czy chodzi o płatność?","criteria":{"true":"","false":""}})),
 ("noul_instr_null", Q({"type":"noul","instructions":None})),
 ("noul_instr_number", Q({"type":"noul","instructions":5})),
 ("no_type", Q({"instructions":"Czy chodzi o płatność?"})),
 ("type_unknown", Q({"type":"rank","instructions":"x"})),
 ("type_multi", Q({"type":"multi","instructions":"x","criteria":{"a":None,"b":None}})),
 ("questions_empty", Q(None)),
 ("questions_list", {"model":MODEL,"state":ST,"questions":[{"type":"noul","instructions":"x"}]}),
 ("state_number", Q({"type":"noul","instructions":"x"}, state=5)),
 ("state_null", Q({"type":"noul","instructions":"x"}, state=None)),
 ("state_empty", Q({"type":"noul","instructions":"Czy chodzi o płatność?"}, state="")),
 ("no_state", {"model":MODEL,"questions":{"q":{"type":"noul","instructions":"x"}}}),
 ("no_model", {"state":ST,"questions":{"q":{"type":"noul","instructions":"x"}}}),
 ("model_unknown", {"model":"nope-1","state":ST,"questions":{"q":{"type":"noul","instructions":"x"}}}),
 ("extra_field", Q({"type":"noul","instructions":"Czy chodzi o płatność?","foo":1}, extra={"bar":2})),
 ("body_not_json", "not json"),
 ("body_array", []),
 ("multi_errors", {"model":MODEL,"state":None,"questions":{"a":{"type":"score","criteria":[]},"b":{"type":"choice"}}}),
]
results = {}
for name, body in cases:
    data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/systemone", data=data, headers={"Content-Type": "application/json", **AUTH})
    t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=120); code = r.status; raw = r.read(); hdr = dict(r.headers)
    except urllib.error.HTTPError as e:
        code = e.code; raw = e.read(); hdr = dict(e.headers)
    try: out = json.loads(raw)
    except Exception: out = raw.decode(errors="replace")[:500]
    results[name] = {"request": body, "status": code, "ms": round((time.time()-t)*1000), "body": out}
    s = json.dumps(out, ensure_ascii=False)
    print(f"{name:22s} {code} {s[:160]}", flush=True)
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
