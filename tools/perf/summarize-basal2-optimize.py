"""Audit and summarize basal2-optimize.py HTTP observations without a GPU or network.

Reads one fresh screen/confirm output directory via --measurement-artifacts.
A screen is SIGNAL only. Confirmation retains r2..r4 and requires every scheduled
phase and valid responses; structural variants match the independent baseline,
while numerical variants retain their own complete answers across phases.
No latency or throughput result automatically promotes a runtime default.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics

# The sibling runner contains pure bit/finite helpers; main is guarded on import.
import importlib.util
spec = importlib.util.spec_from_file_location("basal2_optimize", Path(__file__).with_name("basal2-optimize.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement-artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--http-reference", type=Path, help="frozen answers matching the recorded SHA; otherwise discover adjacent archived inputs")
    a = ap.parse_args()
    if a.out.exists():
        ap.error("output must be new")
    root = a.measurement_artifacts
    method = json.loads((root / "method.json").read_text())
    if method["phase"] == "quality":
        ap.error("quality outputs are preserved export/FP32 comparisons; use the recorded per-mode verdicts")
    progress = json.loads((root / "progress.json").read_text())
    issues = [x for x in progress if x["stage"] == "failed"]
    incomplete = []
    errors = []
    references = {}
    reference_sha = method.get("http_reference_sha256")
    if reference_sha is not None:
        if a.http_reference is None:
            # Both private and public archives keep inputs next to the run directory.
            # Discovery is local and must match the recorded content hash exactly.
            inputs = root.parent.parent / "inputs"
            matches = sorted(path for path in inputs.rglob("http-reference-answers.json")
                             if hashlib.sha256(path.read_bytes()).hexdigest() == reference_sha)
            if matches:
                a.http_reference = matches[0]
        if a.http_reference is None or hashlib.sha256(a.http_reference.read_bytes()).hexdigest() != reference_sha:
            ap.error("matching frozen HTTP reference required for this run")
        for row in json.loads(a.http_reference.read_text()):
            key = (row["model"], row["id"])
            if row.get("variant") != "baseline-compatible" or key in references or not isinstance(row.get("answers"), dict) or not row["answers"] or not runner.finite(row["answers"]):
                raise ValueError("invalid or duplicate frozen HTTP reference")
            references[key] = row["answers"]
    elif a.http_reference is not None:
        ap.error("run does not identify a frozen HTTP reference")
    samples = collections.defaultdict(list)
    phases = {}
    for event in progress:
        if event["stage"] == "phase_finished":
            scope = event["ids"][0] if event["phase"].startswith("ladder") else "all"
            key = (event["tag"], event["phase"], event["clients"], scope)
            if key in phases:
                raise ValueError("duplicate phase: " + str(key))
            phases[key] = event
    runs = []
    for suffix in method["models"]:
        model = "basal-2.0-" + suffix
        counts = collections.Counter()
        expected_ladder = set(method["expected_ladder"][model])
        mixed_ids = method["corpora"]["mixed-mini" if suffix == "mini" else "mixed"]
        if method.get("baseline_control"):
            control = [json.loads(x) for x in (root / (model + "-baseline-control-observations.jsonl")).read_text().splitlines()]
            expected_ids = set(method["corpora"]["short"]) | set(mixed_ids) | expected_ladder
            if (len(control) != len(expected_ids) or {x["id"] for x in control} != expected_ids
                    or {x["index"] for x in control} != set(range(len(control)))):
                raise ValueError("independent baseline control coverage differs")
            event = phases.get((model + "-baseline-control", "baseline-control", max(method["short_clients"]), "all"))
            if event is None or event["samples"] != len(control) or set(event["ids"]) != expected_ids:
                raise ValueError("independent baseline control completion marker differs")
            for item in control:
                value = item.get("response", {}); answers = value.get("answers", {})
                if (item.get("phase") != "baseline-control" or item.get("valid") is not True
                        or item.get("clients") != max(method["short_clients"])
                        or item.get("http_status") != 200 or not isinstance(answers, dict) or not answers
                        or not all(isinstance(v, dict) and isinstance(v.get("type"), str) for v in answers.values())
                        or not runner.finite(value) or value.get("usage", {}).get("questions") != len(answers)
                        or value.get("usage", {}).get("truncated")):
                    raise ValueError("invalid independent baseline control response")
                key = (model, item["id"])
                if key in references and runner.bits(references[key]) != runner.bits(answers):
                    raise ValueError("baseline control differs from frozen HTTP reference")
                references[key] = answers
        for variant in method["schedule"]:
            counts[variant] += 1
            rep = counts[variant]
            tag = f"{model}-{variant}-r{rep}"
            path = root / (tag + "-observations.jsonl")
            if not path.exists():
                incomplete.append({"tag": tag, "reason": "missing observations"})
                continue
            observations = [json.loads(x) for x in path.read_text().splitlines()]
            for item in observations:
                value = item.get("response", {})
                answers = value.get("answers", {})
                valid = (item.get("valid") is True and item.get("http_status") == 200
                         and isinstance(answers, dict) and bool(answers) and runner.finite(value))
                reference_key = ((model, variant, item["id"]) if variant in method.get("numerical_variants", []) + method.get("baseline_comparison_only_variants", [])
                                 else (model, item["id"]))
                if variant not in method.get("invariance_exempt_variants", []) and reference_key in references and runner.bits(answers) != runner.bits(references[reference_key]):
                    valid = False
                references.setdefault(reference_key, answers)
                if not valid:
                    errors.append({"tag": tag, "phase": item["phase"], "id": item["id"],
                                   "index": item["index"], "http_status": item.get("http_status"),
                                   "response_error": value.get("error"),
                                   "error": item.get("detail", "invalid or changed answer")})
                scope = item["id"] if item["phase"].startswith("ladder") else "all"
                samples[(tag, item["phase"], item["clients"], scope)].append(item)
            observed_ladder = {key[3] for key in samples if key[0] == tag and key[1] == "ladder"}
            if observed_ladder != expected_ladder:
                incomplete.append({"tag": tag, "reason": "ladder coverage"})
            expected = [("ladder", 1, rid, method["ladder_samples_per_shape"]) for rid in sorted(expected_ladder)]
            expected += [("ladder-warmup", 1, rid, 2) for rid in sorted(expected_ladder)]
            expected += [("short-warmup", 1, "all", len(method["corpora"]["short"]))]
            expected += [("mixed-warmup", 1, "all", min(16, len(mixed_ids)))]
            expected += [("short", c, "all", method["short_samples_per_phase"]) for c in method.get("short_clients", [1, 8, 32])]
            for clients in method["mixed_clients"]:
                expected.append(("mixed", clients, "all", len(mixed_ids)))
            for workload, clients, scope, n in expected:
                key = (tag, workload, clients, scope)
                data = samples[key]
                event = phases.get(key)
                if not n or len(data) != n or event is None or event["samples"] != n:
                    incomplete.append({"tag": tag, "phase": workload, "clients": clients, "id": scope,
                                       "reason": "sample count or completion marker"})
                    continue
                if len({x["index"] for x in data}) != n:
                    incomplete.append({"tag": tag, "reason": "duplicate request indices"})
                    continue
                if workload == "mixed" and {x["id"] for x in data} != set(mixed_ids):
                    incomplete.append({"tag": tag, "phase": workload, "reason": "mixed ID coverage"})
                if workload == "short" and {x["id"] for x in data} != set(method["corpora"]["short"]):
                    incomplete.append({"tag": tag, "phase": workload, "reason": "short ID coverage"})
                if workload.endswith("-warmup"):
                    continue
                if rep in method["excluded_repetitions"]:
                    continue
                times = [x["elapsed_ms"] for x in data]
                class_times = collections.defaultdict(list)
                for x in data:
                    class_times[x["class"]].append(x["elapsed_ms"])
                runs.append({"model": model, "variant": variant, "repetition": rep,
                             "workload": workload, "clients": clients, "id": scope, "samples": n,
                             "parity_reference": "own variant" if variant in method.get("numerical_variants", []) else "upstream differences recorded" if variant == "upstream-fast" else "baseline",
                             "baseline_answer_differences": sum(x.get("baseline_bitwise_equal") is False for x in data),
                             "within_variant_answer_differences": sum(x.get("answer_bitwise_invariant") is False for x in data),
                             "median_ms": statistics.median(times), "requests_per_s": event["requests_per_s"],
                             "latencies_ms": times,
                             "classes": {c: {"samples": len(t), "median_ms": statistics.median(t),
                                              "p95_ms": sorted(t)[min(len(t)-1, int(.95*len(t)))]}
                                         for c, t in class_times.items()}})
    groups = collections.defaultdict(list)
    for run in runs:
        groups[(run["model"], run["variant"], run["workload"], run["clients"], run["id"])].append(run)
    rows = []
    for (model, variant, workload, clients, rid), group in groups.items():
        medians = [x["median_ms"] for x in group]
        rates = [x["requests_per_s"] for x in group]
        pooled = sorted(t for x in group for t in x["latencies_ms"])
        expected_reps = 3 if method["phase"] == "confirm" else 1
        complete = len(group) == expected_reps
        if not complete:
            incomplete.append({"model": model, "variant": variant, "workload": workload,
                               "clients": clients, "id": rid, "reason": "repetition count"})
        rows.append({"model": model, "variant": variant, "workload": workload, "clients": clients, "id": rid,
                     "complete": complete, "repetitions": len(group), "samples": len(pooled),
                     "parity_reference": group[0]["parity_reference"],
                     "baseline_answer_differences": sum(x["baseline_answer_differences"] for x in group),
                     "within_variant_answer_differences": sum(x["within_variant_answer_differences"] for x in group),
                     "median_ms": statistics.median(medians), "repeat_median_min_ms": min(medians),
                     "repeat_median_max_ms": max(medians), "pooled_p95_ms": pooled[min(len(pooled)-1, int(.95*len(pooled)))],
                     "median_requests_per_s": statistics.median(rates)})
    complete = (root / "complete.json").exists() and not (errors or incomplete or issues)
    # Screen data never becomes PASS, even if every exploratory request was valid.
    status = "FAIL" if errors or issues else "GAP" if not complete else "SIGNAL" if method["phase"] == "screen" else "PASS"
    result = {"status": status, "complete": complete, "method": method, "errors": errors,
              "incomplete": incomplete, "execution_issues": issues, "rows": rows, "runs": runs,
              "limits": "median of retained run medians; range is not a confidence interval; pooled p95 exploratory; mixed per-class details in runs; numerical variants require separate FP32 quality review; no automatic speedup/default promotion"}
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": status, "rows": len(rows), "errors": len(errors), "incomplete": len(incomplete)}))


if __name__ == "__main__":
    main()
