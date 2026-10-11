#!/usr/bin/env python3
"""Prepare the measured subsets from existing FP32 references, without changing them.

Run from the repository root, with a NEW output directory as the only argument.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)


def read(directory, file):
    return [json.loads(line) for line in Path(directory, file).read_text().splitlines() if line.strip()]


def write(name, bench, cases, model):
    output = args.output / name
    output.mkdir()
    for case in cases:
        case["request"]["model"] = model
    for file, rows in [("bench.jsonl", bench), ("systemone.jsonl", cases)]:
        with (output / file).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


mini = "reports/reference-basal-1.5-mini-fp32"
write("mini-normalized", read(mini, "bench.jsonl"), read(mini, "systemone.jsonl"), "basal-1.5-mini")
write("mini-bench", read(mini, "bench.jsonl"), [], "basal-1.5-mini")
write("4.5B-small", read("reports/reference-basal-1.5-4.5B-fp32", "bench.jsonl")[:3], [], "basal-1.5-4.5B")
write("max-bench", read("reports/reference-1.5-max-fp32", "bench.jsonl"), [], "basal-1.5-max")
max_cases = {row["id"]: row for row in read("reports/reference-1.5-max-fp32-cases2", "systemone.jsonl")}
selected = [max_cases[key] for key in [
    "m15-facts-amounts", "m15-mixed", "m15-evidence-choice-en", "cx-01-ticket-fan-out"
]]
write("max-recovery", read("reports/reference-1.5-max-fp32", "bench.jsonl")[:3], selected, "basal-1.5-max")
