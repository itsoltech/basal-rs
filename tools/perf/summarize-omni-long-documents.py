"""Independently replay long-document HTTP, bitwise references and resource windows."""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('resource_audit', Path(__file__).with_name('summarize-omni-resources.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
protocol = audit.protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new')
    method = json.loads((a.measurement_artifacts / 'method.json').read_text())
    manifest = json.loads((a.inputs / 'provenance.json').read_text())
    if method['protocol'] != 'omni-long-documents-v1' or method['inputs'] != manifest:
        p.error('protocol or provenance mismatch')
    if method['phase'] == 'confirm':
        starts = method.get('confirm_starts', 4)
        if (starts not in [3, 4] or method['schedule'] !=
                (['baseline', 'candidate', 'upstream-eager', 'upstream-eager', 'candidate', 'baseline'] * 2)[:3 * starts]
                or method['excluded_repetitions'] != [1]):
            p.error('confirmation schedule/exclusions differ')
    elif method['phase'] != 'screen' or method['schedule'][0] != 'baseline':
        p.error('invalid screen')
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        if protocol.digest(a.inputs / name) != manifest['files'][name]:
            p.error('input hash mismatch')
    short = audit.records(a.inputs / 'benchmark-short.jsonl')
    ladder = [r for r in audit.records(a.inputs / 'text-ladder.jsonl')
              if r['model'] == method['model'] and r['target_tokens'] == 8192]
    if len(short) != 32 or len(ladder) != 2:
        p.error('frozen corpus must contain32 short and2 long inputs')
    definitions = {r['id']: r for r in short + ladder}
    mixed_ids = []
    for index, row in enumerate(short):
        mixed_ids.append(row['id'])
        if (index + 1) % 16 == 0:
            mixed_ids.append(ladder[(index // 16) % 2]['id'])
    events = audit.records(a.measurement_artifacts / 'progress.jsonl')
    complete = json.loads((a.measurement_artifacts / 'complete.json').read_text())
    issues = list(complete['failures'])
    if not complete['execution_finished']:
        issues.append({'kind': 'execution_incomplete'})
    reference, upstream_reference, phases, counts = {}, {}, [], collections.Counter()
    compared, total, native_differences, upstream_variations = 0, 0, 0, 0
    for variant in method['schedule']:
        counts[variant] += 1
        tag = f'{variant}-r{counts[variant]}'
        try:
            observations = audit.records(a.measurement_artifacts / (tag + '-observations.jsonl'))
            samples = audit.records(a.measurement_artifacts / (tag + '-resources.jsonl'))
        except FileNotFoundError:
            issues.append({'kind': 'missing_run', 'tag': tag})
            continue
        stops = [e for e in events if e['stage'] == 'server_stopped' and e['tag'] == tag]
        if len(stops) != 1:
            issues.append({'kind': 'server_not_stopped', 'tag': tag})
        elif stops[0]['exit_code'] not in ([0, -2, 130] if variant == 'upstream-eager' else [0]):
            issues.append({'kind': 'unexpected_server_exit', 'tag': tag,
                           'exit_code': stops[0]['exit_code']})
        starts = [e for e in events if e['stage'] == 'server_start' and e['tag'] == tag]
        command = starts[0]['command'] if len(starts) == 1 else []
        expected_flags = ({'--mode': 'eager', '--dtype': 'bfloat16', '--orders': '1',
                           '--state-cache-gb': '0', '--context-overflow': 'refuse'}
                          if variant == 'upstream-eager' else
                          {'--dtype': 'auto', '--inference-mode': 'auto', '--orders': '1',
                           '--state-cache-mb': '0', '--gemm-table': 'auto',
                           '--max-batch-tokens': '16384',
                           '--schedule': method['candidate_schedule'] if variant == 'candidate' else 'fifo',
                           '--long-tokens': str(method['candidate_long_tokens'] if variant == 'candidate' else 0),
                           '--long-slice-ms': str(method['candidate_long_slice_ms'])})
        for flag, value in expected_flags.items():
            if command.count(flag) != 1 or command[command.index(flag) + 1:command.index(flag) + 2] != [value]:
                issues.append({'kind': 'server_config', 'tag': tag, 'flag': flag})
        if variant != 'upstream-eager' and '--no-prefix-cache' not in command:
            issues.append({'kind': 'prefix_cache_enabled', 'tag': tag})
        for row in observations:
            total += 1
            source = definitions.get(row['id'])
            response = row.get('response', {})
            valid = (source is not None and row['http_status'] == 200 and row['valid']
                     and protocol.valid_answers(response, source['request']['questions'])
                     and protocol.finite(response) and response.get('model') == method['model']
                     and not response.get('usage', {}).get('truncated')
                     and math.isfinite(row.get('elapsed_ms', -1)) and row.get('elapsed_ms', -1) > 0)
            if source and 'target_tokens' in source:
                valid = valid and response.get('usage', {}).get('input_tokens') == source['target_tokens']
            if not valid:
                issues.append({'kind': 'invalid_response', 'tag': tag, 'index': row['index']})
                continue
            value = protocol.bits(response['answers'])
            if variant == 'upstream-eager':
                upstream_variations += row['id'] in upstream_reference and upstream_reference[row['id']] != value
                upstream_reference.setdefault(row['id'], value)
            elif tag == 'baseline-r1' and row['phase'] == 'warmup':
                reference[row['id']] = value
            else:
                compared += 1
                if row['id'] not in reference or reference[row['id']] != value:
                    native_differences += 1
                    issues.append({'kind': 'native_reference_mismatch', 'tag': tag, 'id': row['id'], 'index': row['index']})
        expected = [('warmup', 1)]
        if 'tasks' in method['workloads']:
            expected.append(('tasks-c32', 32))
        if 'mixed' in method['workloads']:
            expected += [(f'mixed-c{c}', c) for c in method['clients']]
        finished_events = [e for e in events if e['stage'] == 'phase_finished' and e['tag'] == tag]
        if [(e['name'], e['clients']) for e in finished_events] != expected:
            issues.append({'kind': 'phase_coverage', 'tag': tag})
        for name, clients in expected:
            rows = [r for r in observations if r['phase'] == name and r['clients'] == clients]
            ev = next((e for e in finished_events if e['name'] == name and e['clients'] == clients), None)
            corpus = [r['id'] for r in short] if name == 'tasks-c32' else mixed_ids
            required_passes = (1 if name == 'warmup' else method['short_passes'] if name == 'tasks-c32'
                               else method['mixed_passes_c32'] if clients == 32 else method['mixed_passes_c8'])
            if (not ev or not rows or [r['index'] for r in rows] != list(range(len(rows)))
                    or len(rows) % len(corpus) or len(rows) < required_passes * len(corpus)
                    or [r['id'] for r in rows] != [corpus[i % len(corpus)] for i in range(len(rows))]):
                issues.append({'kind': 'corpus_or_count', 'tag': tag, 'phase': name})
                continue
            points = sorted([(r['started_monotonic_s'], 1) for r in rows]
                            + [(r['finished_monotonic_s'], -1) for r in rows])
            active, peak = 0, 0
            for _, change in points:
                active += change
                peak = max(peak, active)
            if active != 0 or peak != clients:
                issues.append({'kind': 'concurrency', 'tag': tag, 'phase': name, 'peak': peak})
            duration = ev['end'] - ev['begin']
            if duration <= 0 or min(r['started_monotonic_s'] for r in rows) < ev['begin'] or max(r['finished_monotonic_s'] for r in rows) > ev['end']:
                issues.append({'kind': 'phase_bounds', 'tag': tag, 'phase': name})
                continue
            if name != 'warmup' and duration < method['minimum_seconds']:
                issues.append({'kind': 'phase_too_short', 'tag': tag, 'phase': name})
            metrics = audit.resources(samples, name, clients, method['sample_interval_s'])
            if metrics['status'] != 'PASS':
                issues.append({'kind': 'resource_gap', 'tag': tag, 'phase': name, 'details': metrics})
            classes = {}
            for label in ['short', 'long']:
                values = [r['elapsed_ms'] for r in rows if ('target_tokens' in definitions[r['id']]) == (label == 'long')
                          and r.get('valid')]
                if values:
                    classes[label] = {'samples': len(values), 'median_ms': statistics.median(values),
                                      'p95_ms': audit.percentile(values, .95), 'max_ms': max(values)}
            phases.append({'tag': tag, 'variant': variant, 'repetition': counts[variant],
                           'retained': name != 'warmup' and counts[variant] not in method['excluded_repetitions'],
                           'phase': name, 'clients': clients, 'requests': len(rows), 'duration_s': duration,
                           'successful_requests_per_s': sum(r.get('valid', False) for r in rows) / duration,
                           'classes': classes, 'resources': metrics, 'peak_clients': peak})
    if len(reference) != 34:
        issues.append({'kind': 'incomplete_native_reference', 'count': len(reference)})
    result = {'status': 'FAIL' if issues else 'AUDITED', 'issues': issues, 'method': method,
              'total_responses': total, 'native_reference_inputs': len(reference),
              'native_compared_responses': compared, 'native_differences': native_differences,
              'upstream_probability_or_answer_variations': upstream_variations, 'phases': phases,
              'scope': 'Finite frozen corpus on one GPU; no general tail guarantee. '
                       'Upstream variation is retained, no bitwise FP32 equivalence claim.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'issues': len(issues), 'responses': total,
                      'native_compared': compared, 'native_differences': native_differences}))
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
