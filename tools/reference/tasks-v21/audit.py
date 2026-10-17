#!/usr/bin/env python3
"""Compare host task observations by ID and binary64 values; no inference quality claim."""
import argparse
import json
from pathlib import Path
import struct


def bits(v):
    if isinstance(v, bool) or v is None or isinstance(v, str): return v
    if isinstance(v, (int, float)): return ('binary64', struct.pack('>d', float(v)).hex())
    if isinstance(v, list): return [bits(x) for x in v]
    if isinstance(v, dict): return {k: bits(x) for k, x in v.items()}
    raise ValueError('unexpected observation')


def rows(path):
    result = {}
    for line in path.read_text().splitlines():
        v = json.loads(line)
        if not isinstance(v.get('id'), str) or v['id'] in result: raise ValueError('missing/duplicate observation ID')
        result[v['id']] = v
    if not result: raise ValueError('empty observations')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--candidate', required=True, help='Observation filename within artifact directory')
    a = p.parse_args()
    if Path(a.candidate).name != a.candidate: p.error('candidate must be a filename')
    expected = rows(a.measurement_artifacts/'upstream.jsonl')
    actual = rows(a.measurement_artifacts/a.candidate)
    inputs = rows(a.measurement_artifacts/'host-inputs.jsonl')
    if set(expected) != set(inputs): raise ValueError('reference IDs do not match input corpus')
    issues = [{'id': i, 'reason': 'missing or extra observation'} for i in sorted(set(expected)^set(actual))]
    for i in sorted(set(expected)&set(actual)):
        if bits(expected[i]) != bits(actual[i]): issues.append({'id': i, 'reason': 'decoded value/binary64 differs', 'expected': expected[i], 'actual': actual[i]})
    print(json.dumps({'status': 'FAIL' if issues else 'PASS', 'reference_rows':len(expected), 'candidate_rows':len(actual), 'mismatches':issues}, ensure_ascii=False, indent=2))
    if issues: raise SystemExit(1)


if __name__ == '__main__': main()
