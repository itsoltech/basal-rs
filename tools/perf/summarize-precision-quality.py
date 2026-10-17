"""Recompute decision agreement and probability distances to FP32 from retained HTTP answers."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

FIELDS = ["type", "choice", "selected", "action", "refused", "answer", "deferred"]


def compare(reference, candidate):
    if set(reference) != set(candidate):
        raise ValueError("answer names differ from FP32")
    changed, ordering, argmax, gaps = [], [], [], []
    errors, scalar_errors = [], defaultdict(list)
    question_agreement = {}
    categorical_agreement = {}
    product_agreement = {}
    margins = {}
    for name, ref in reference.items():
        cur = candidate[name]
        differences = {f: {"fp32": ref.get(f), "candidate": cur.get(f)} for f in FIELDS if ref.get(f) != cur.get(f)}
        value_differences = {}
        for field, values in differences.items():
            order_only = (field == "selected" and isinstance(values["fp32"], list) and isinstance(values["candidate"], list)
                          and Counter(values["fp32"]) == Counter(values["candidate"]))
            if order_only:
                ordering.append({"question": name, "field": field, **values})
            else:
                value_differences[field] = values
        if value_differences:
            changed.append({"question": name, "fields": value_differences})
        pa, pb = ref.get("probabilities", {}), cur.get("probabilities", {})
        if set(pa) != set(pb) or not pa:
            gaps.append({"question": name, "issue": "probability labels missing/different"})
            continue
        if not all(isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 1 for x in [*pa.values(), *pb.values()]):
            raise ValueError("nonfinite/invalid/out-of-range probabilities")
        errors.extend(abs(pa[k]-pb[k]) for k in pa)
        # Multi-label decisions use the selected set; argmax alone is not their product decision.
        top_changed = False
        if "selected" not in ref:
            ra = {k for k, v in pa.items() if v == max(pa.values())}
            rb = {k for k, v in pb.items() if v == max(pb.values())}
            top_changed = ra != rb
            if top_changed:
                argmax.append({"question": name, "fp32": sorted(ra), "candidate": sorted(rb)})
        ps = sorted(pa.values(), reverse=True)
        margins[name] = ps[0]-ps[1] if len(ps) > 1 else None
        question_agreement[name] = not value_differences and not top_changed
        # score returns an expected value, not the most probable ordinal level. Keep its
        # argmax in the original strict diagnostic, and evaluate the returned value by MAE.
        product_agreement[name] = not value_differences and (not top_changed or ref.get("type") == "score")
        if ref.get("type") != "score":
            categorical_agreement[name] = product_agreement[name]
        for f in ["noul", "score", "confidence", "set_confidence"]:
            if isinstance(ref.get(f), (int, float)) and isinstance(cur.get(f), (int, float)):
                scalar_errors[f].append(abs(ref[f]-cur[f]))
    return dict(decision_agreement=all(question_agreement.values()) and not gaps,
                product_decision_agreement=all(product_agreement.values()) and not gaps,
                categorical_questions=categorical_agreement,
                questions=question_agreement, decision_field_changes=changed, selected_order_changes=ordering,
                argmax_changes=argmax, probability_errors=errors, scalar_errors=dict(scalar_errors),
                fp32_top_two_margins=margins, gaps=gaps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-artifacts", type=Path, required=True, help="one precision-quality run directory")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error("output must be new")
    method = json.loads((a.measurement_artifacts / "method.json").read_text())
    complete = json.loads((a.measurement_artifacts / "complete.json").read_text())
    if not complete.get("execution_finished"):
        raise ValueError("quality execution incomplete")
    rows = defaultdict(dict)
    for row in map(json.loads, (a.measurement_artifacts / "observations.jsonl").read_text().splitlines()):
        variant, key = row["variant"], row["id"]
        if key in rows[variant] or not row.get("valid") or row.get("http_status") != 200:
            raise ValueError("duplicate/invalid observation")
        rows[variant][key] = row
    reference_variant = method.get("reference_variant", "upstream-fp32")
    gold = rows[reference_variant]
    if len(gold) != method["cases"] or set(rows) != set(method["variants"]):
        raise ValueError("variant/FP32 coverage mismatch")
    details, summaries, bf16_agreement = {}, [], {}
    for variant, observations in rows.items():
        if set(observations) != set(gold):
            raise ValueError("candidate coverage differs from FP32")
        details[variant] = {}
        for key, row in observations.items():
            if row["response"]["usage"]["input_tokens"] != gold[key]["response"]["usage"]["input_tokens"]:
                raise ValueError("input tokens differ from FP32")
            details[variant][key] = compare(gold[key]["response"]["answers"], row["response"]["answers"])
        if variant == "upstream-fast":
            bf16_agreement = {k: d["decision_agreement"] for k, d in details[variant].items()}
    for variant, result in details.items():
        cohorts = sorted({x["cohort"] for x in rows[variant].values()})
        for cohort in [None, *cohorts]:
            items = [(k, d) for k, d in result.items() if cohort is None or rows[variant][k]["cohort"] == cohort]
            errors = [e for _, d in items for e in d["probability_errors"]]
            scalar_errors = defaultdict(list)
            for _, detail in items:
                for field, values in detail["scalar_errors"].items():
                    scalar_errors[field].extend(values)
            scalar_distances = {field: dict(count=len(values), mae=statistics.mean(values), max_abs=max(values),
                                           rmse=math.sqrt(statistics.mean(x*x for x in values)))
                                for field, values in sorted(scalar_errors.items())}
            summaries.append(dict(variant=variant, cohort=cohort or "all", responses=len(items),
                                  responses_with_fp32_decisions=sum(d["decision_agreement"] for _, d in items),
                                  questions=sum(len(d["questions"]) for _, d in items),
                                  questions_with_fp32_decisions=sum(sum(d["questions"].values()) for _, d in items),
                                  categorical_questions=sum(len(d["categorical_questions"]) for _, d in items),
                                  categorical_questions_with_fp32_decisions=sum(sum(d["categorical_questions"].values()) for _, d in items),
                                  categorical_changed_ids=[k for k, d in items if not all(d["categorical_questions"].values())],
                                  changed_ids=[k for k, d in items if not d["decision_agreement"]],
                                  order_only_ids=[k for k, d in items if d["selected_order_changes"] and d["decision_agreement"]],
                                  gap_ids=[k for k, d in items if d["gaps"]],
                                  probability_mae=statistics.mean(errors), probability_max_abs=max(errors),
                                  probability_rmse=math.sqrt(statistics.mean(x*x for x in errors)),
                                  scalar_distances=scalar_distances,
                                  wins_over_bf16=sum(d["decision_agreement"] and not bf16_agreement[k] for k, d in items) if bf16_agreement else None,
                                  losses_to_bf16=sum(not d["decision_agreement"] and bf16_agreement[k] for k, d in items) if bf16_agreement else None))
    a.out.write_text(json.dumps(dict(model=method["model"], reference_variant=reference_variant,
                                    bf16_reference_available=bool(bf16_agreement), summaries=summaries, details=details,
                                    interpretation="FP32 agreement is a reference target, not semantic ground-truth accuracy. Original strict fields retain score argmax as a diagnostic; categorical counts exclude continuous score, whose returned expected value is evaluated by scalar_distances.score. Noul retains its binary argmax/threshold criterion. Selected ordering is separate. Cohorts overlap no IDs; coverage is bounded."),
                               ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
