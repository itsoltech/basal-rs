"""Verify POST /v1/decisions with the official OpenAI SDK against an equivalent /v1/systemone request.

Usage: python -I tools/openai-decisions/verify.py BASE_URL MODEL OUT_JSON  (requires openai>=3.26)
Numbers are compared as JSON text (bitwise), independently of the Rust translation.
"""

import json
import sys
import urllib.request

import openai

BASE, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
client = openai.OpenAI(base_url=BASE + "/v1", api_key="unused", max_retries=0)


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read(), parse_float=str)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read(), parse_float=str)


def text(parts, kinds=("input_text", "output_text")):
    if isinstance(parts, str):
        return parts
    out = []
    for p in parts:
        if p["type"] in kinds:
            out.append(p["text"])
        elif p["type"] == "input_json":
            out.append(p["data"])
        elif p["type"] == "refusal":
            out.append({"refusal": p["refusal"]})
    return out[0] if len(out) == 1 and isinstance(out[0], str) else out


def state(inp):
    if isinstance(inp, str):
        return inp
    out = []
    for it in inp:
        t = it.get("type", "message")
        if t == "message":
            role = "developer" if it["role"] == "system" else it["role"]
            out.append({"role": role, "content": text(it["content"])})
        elif t in ("function_call", "custom_tool_call"):
            f = "arguments" if t == "function_call" else "input"
            v = {"type": t, "name": it["name"]}
            if it.get("namespace"):
                v["namespace"] = it["namespace"]
            v[f] = it[f]
            v["call_id"] = it["call_id"]
            out.append(v)
        else:
            out.append({"type": t, "call_id": it["call_id"], "output": text(it["output"])})
    return out


def systemone(req):
    qs = {}
    for i, q in enumerate(req["questions"]):
        t = {"predicate": "noul", "choice": "choice", "score": "score"}[q["type"]]
        v = {"type": t, "instructions": text(q["instructions"])}
        if q["type"] == "choice":
            v["criteria"] = {(c["value"] if isinstance(c["value"], str) else json.dumps(c["value"])): c.get("description")
                             for c in q["choices"]}
        if q["type"] == "score":
            v["criteria"] = {lv["label"]: lv.get("description") for lv in q["levels"]}
        qs[f"q{i}"] = v
    return {"model": req["model"], "state": state(req["input"]), "questions": qs}


def expected(req, ts):
    answers = []
    for i, q in enumerate(req["questions"]):
        a = ts["answers"][f"q{i}"]
        name = q.get("name")
        if q["type"] == "predicate":
            answers.append({"type": "predicate", "name": name, "probability": a["noul"]})
        elif q["type"] == "choice":
            key = lambda v: v if isinstance(v, str) else json.dumps(v)
            choice = next(c["value"] for c in q["choices"] if key(c["value"]) == a["choice"])
            answers.append({"type": "choice", "name": name, "choice": choice,
                            "probabilities": [{"value": c["value"], "probability": a["probabilities"][key(c["value"])]}
                                              for c in q["choices"]],
                            "confidence": a["confidence"]})
        else:
            answers.append({"type": "score", "name": name, "score": a["score"],
                            "probabilities": [{"value": j, "label": lv["label"], "probability": a["probabilities"][lv["label"]]}
                                              for j, lv in enumerate(q["levels"])],
                            "confidence": a["confidence"]})
    return answers


CASES = {
    "choice_predicate_score_string_input": {
        "input": "I was charged twice for my order #1234 and the app keeps crashing when I open the invoice.",
        "questions": [
            {"type": "choice", "name": "route", "instructions": "Which team should handle this ticket?",
             "choices": [{"value": "billing", "description": "payments, invoices, refunds"},
                         {"value": "technical"}, {"value": "shipping"}, {"value": "other"}]},
            {"type": "predicate", "name": "duplicate_charge", "instructions": "Was the customer charged twice?"},
            {"type": "score", "name": "severity", "instructions": "How severe is the problem?",
             "levels": [{"label": "Cosmetic"}, {"label": "Workaround available"},
                        {"label": "Fully blocked", "description": "the customer cannot proceed"}]},
            {"type": "predicate", "instructions": "Is the customer asking about shipping?"},
        ],
    },
    "messages_json_tools_bool_choice": {
        "input": [
            {"role": "system", "content": "You triage support conversations."},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Please check my order."},
                {"type": "input_json", "data": {"order": {"id": 1234, "total": 49.9, "charges": [49.9, 49.9]}}}]},
            {"type": "function_call", "call_id": "c1", "name": "get_order", "arguments": "{\"id\":1234}",
             "id": "fc_1", "status": "completed"},
            {"type": "function_call_output", "call_id": "c1", "output": "{\"refundable\": true}"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "I see two charges."}],
             "phase": "final_answer"},
            {"type": "custom_tool_call", "call_id": "c2", "name": "notes", "input": "duplicate charge"},
            {"type": "custom_tool_call_output", "call_id": "c2", "output": [{"type": "input_text", "text": "saved"}]},
        ],
        "questions": [
            {"type": "choice", "name": "refund", "instructions": [{"type": "input_text", "text": "Should we refund?"}],
             "choices": [{"value": True, "description": "issue a refund"}, {"value": False}]},
            {"type": "score", "name": "urgency", "instructions": "How urgent is it?",
             "levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}, {"label": "critical"}]},
        ],
    },
}

ERRORS = {
    "unknown_model": ({"model": "gpt-6-luna", "input": "x", "questions": [{"type": "predicate", "instructions": "x"}]}, 404),
    "https_image": ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}],
                     "questions": [{"type": "predicate", "instructions": "x"}]}, 400),
    "bad_type": ({"input": "x", "questions": [{"type": "box", "instructions": "x"}]}, 400),
    "one_choice": ({"input": "x", "questions": [{"type": "choice", "instructions": "x", "choices": [{"value": "a"}]}]}, 400),
    "dangling_output": ({"input": [{"type": "function_call_output", "call_id": "nope", "output": "x"}],
                         "questions": [{"type": "predicate", "instructions": "x"}]}, 400),
    "unknown_field": ({"input": "x", "questions": [{"type": "predicate", "instructions": "x"}],
                       "extra_body": {"temperature": 0}}, 400),
    "png_on_text_model": ({"input": [{"role": "user", "content": [{"type": "input_image", "image_url":
                           "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}]}],
                           "questions": [{"type": "predicate", "instructions": "x"}]}, 400),
}

report = {"model": MODEL, "openai_sdk": openai.__version__, "cases": {}, "errors": {}}
ok = True
for name, case in CASES.items():
    req = {"model": MODEL, **case}
    raw = client.decisions.with_raw_response.create(**req)
    parsed = raw.parse()  # SDK model parsing must succeed
    body = json.loads(raw.text, parse_float=str)
    status, ts = post("/v1/systemone", systemone(req))
    same = status == 200 and body["answers"] == expected(req, ts)
    usage = body["usage"]["input_tokens"] == ts["usage"]["input_tokens"]
    ok &= same and usage
    report["cases"][name] = {"bitwise_equal_to_systemone": same, "usage_equal": usage,
                             "sdk_types": [type(a).__name__ for a in parsed.answers], "response": body}
    print(name, "equal" if same and usage else "DIFF")
for name, (case, want) in ERRORS.items():
    req = {"model": MODEL, **case}
    try:
        client.decisions.create(**req)
        got, err = 200, None
    except openai.APIStatusError as e:
        got, err = e.status_code, e.body
    good = got == want and isinstance(err, dict) and {"message", "type", "param", "code"} <= err.keys()
    ok &= good
    report["errors"][name] = {"status": got, "expected": want, "envelope_ok": good, "error": err}
    print(name, got, "ok" if good else "FAIL", err and err.get("code"))
json.dump(report, open(OUT, "w"), indent=2)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
