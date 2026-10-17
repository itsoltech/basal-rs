#!/usr/bin/env python3
"""Offline audit of decoded SDK decisions against direct native HTTP observations."""
import argparse
import json
from pathlib import Path
import struct


def project(a):
    kind = a["type"]
    common = {"type": kind, "probabilities": a["probabilities"]}
    if kind in ("choice", "noul", "score", "act"):
        common["confidence"] = a["confidence"]
    if kind == "choice": common["choice"] = a["choice"]
    elif kind == "noul": common["noul"] = a["noul"]
    elif kind == "score": common.update(score=a["score"], legend=a["legend"])
    elif kind == "multi":
        common.update(selected=a["selected"], threshold=a.get("threshold"), objective=a.get("objective") or "threshold",
                      expected_score=a.get("expected_score"), constraints_applied=a.get("constraints_applied"), set_confidence=a["set_confidence"])
    elif kind == "act":
        common.update(action=a["action"], answer=a["answer"], expected_costs=a["expected_costs"], calibration=a["calibration"],
                      deferred=a.get("deferred"), refused=a.get("refused"), max_error=a.get("max_error"))
    elif kind == "tool":
        multi = a.get("multi", False)
        selection = a.get("selection")
        if multi and selection is None:
            selection = {**a, "type": "multi", "selected": a["tools"]}
        common.update(tool=a.get("tool"), call=a["call"], multi=multi, tools=a.get("tools", []), confidence=a.get("confidence"),
                      ranking=a.get("ranking", []), selection=project(selection) if selection else None,
                      deferred=a.get("deferred"), refused=a.get("refused"), max_error=a.get("max_error"))
    else: raise ValueError("unknown answer type")
    return common


def bits(value):
    # All numeric fields compared here are finite probabilities/scores or small counts.
    # Normalize JSON's 1 versus 1.0 spelling while retaining the sign of zero.
    if isinstance(value, bool) or value is None or isinstance(value, str): return value
    if isinstance(value, (int, float)): return {"f64": struct.pack(">d", float(value)).hex()}
    if isinstance(value, list): return [bits(x) for x in value]
    if isinstance(value, dict): return {k: bits(v) for k, v in sorted(value.items())}
    raise ValueError("unexpected projection value")


def usage(raw):
    u = raw["usage"]
    return {key: u.get(key) for key in ("input_tokens", "output_tokens", "questions", "branches", "longest_prompt_tokens", "second_order_branches", "media")}


def audit(directory):
    references = {x["case"]: x for x in map(json.loads, (directory / "native.jsonl").read_text().splitlines())}
    observations = []
    for path in sorted(directory.glob("sdk-*.jsonl")):
        observations.extend(json.loads(line) for line in path.read_text().splitlines())
    method=json.loads((directory / "method.json").read_text())
    expected_ids={(sdk,case) for sdk in method["required_sdks"] for case in method["case_ids"]}
    expected_ids.update((sdk,"named-api") for sdk in method["named_sdks"])
    if set(references)!=set(method["case_ids"]):raise ValueError("incomplete native references")
    if not expected_ids: raise ValueError("empty measurement method")
    issues, counts, seen = [], {}, set()
    discovery = json.loads((directory / "native-models.json").read_text())["models"]
    discovery_seen = set()
    check_seen = set()
    def model_projection(models):
        if isinstance(models, dict): models = models["models"]
        return [{key: m.get(key) for key in ("name", "prompt_format", "orders", "media", "temperature_per_prim")} for m in models]
    for record in observations:
        if "check" in record:
            identity=(record["sdk"],record["check"])
            if identity in check_seen:issues.append({"sdk":record["sdk"],"reason":"duplicate error check"})
            check_seen.add(identity)
            status={"unknown-model":422,"empty-media":"local-rejection"}.get(record["check"])
            if record["status"]!=status:issues.append({"sdk":record["sdk"],"reason":"incorrect error result"})
            continue
        if "case" not in record:
            key = record["sdk"]
            if key == "rust": key = "rust-map"
            if key in discovery_seen: issues.append({"sdk":key,"reason":"duplicate discovery"})
            discovery_seen.add(key)
            if bits(model_projection(record["models"])) != bits(model_projection(discovery)):
                issues.append({"sdk":key,"reason":"discovery differs"})
            continue
        key = record["sdk"]
        identity=(key,record["case"])
        if identity in seen:issues.append({"sdk":key,"case":record["case"],"reason":"duplicate observation"})
        seen.add(identity)
        counts[key] = counts.get(key, 0) + 1
        reference = references[record["case"]]["raw"]
        try:
            expected = {k: project(v) for k, v in reference["answers"].items()}
            actual = {k: project(v) for k, v in record["answers"].items()}
            if bits(expected) != bits(actual): issues.append({"sdk": key, "case": record["case"], "reason": "decoded answer differs", "expected": expected, "actual": actual})
            if bits(usage(reference)) != bits(usage(record["raw"])): issues.append({"sdk": key, "case": record["case"], "reason": "usage differs"})
            if reference["model"] != record["raw"]["model"]: issues.append({"sdk": key, "case": record["case"], "reason": "model differs"})
        except (KeyError, TypeError, ValueError) as error:
            issues.append({"sdk": key, "case": record["case"], "reason": "incomplete result", "error": type(error).__name__})
    if method.get("error_checks"):
        required_checks={(sdk,check) for sdk in method["required_sdks"] for check in method["error_checks"]}
        for sdk,check in sorted(required_checks-check_seen):issues.append({"sdk":sdk,"check":check,"reason":"missing error check"})
    for sdk in sorted(set(method["required_sdks"])-discovery_seen):issues.append({"sdk":sdk,"reason":"missing discovery"})
    for sdk,case in sorted(expected_ids-seen):issues.append({"sdk":sdk,"case":case,"reason":"missing observation"})
    for sdk,case in sorted(seen-expected_ids):issues.append({"sdk":sdk,"case":case,"reason":"unexpected observation"})
    return {"status": "PASS" if not issues else "FAIL", "counts": counts, "observations": sum(counts.values()), "discovery_observations":len(discovery_seen),"error_checks":len(check_seen),"issues": issues}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-artifacts", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.measurement_artifacts)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
