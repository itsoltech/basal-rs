"""Replay Gemma loader startup, server-tree resources and bitwise C1 controls."""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


protocol = module('protocol', Path(__file__).with_name('omni-optimize.py'))
resource_audit = module('resource_audit', Path(__file__).with_name('summarize-omni-resources.py'))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new')
    root = a.measurement_artifacts
    method = json.loads((root / 'method.json').read_text())
    if method['protocol'] != 'gemma-loader-v1' or method['excluded_repetitions'] != [1]:
        p.error('unknown startup protocol')
    schedule = ['baseline', 'candidate', 'candidate', 'baseline'] * 2
    if method['schedule'] not in [schedule, schedule + ['upstream-eager'] * 4]:
        p.error('unexpected schedule')
    manifest = json.loads((a.inputs / 'provenance.json').read_text())
    if method['inputs'] != manifest:
        p.error('input provenance differs')
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        if protocol.digest(a.inputs / name) != manifest['files'][name]:
            p.error('input checksum mismatch')
    definitions = {r['id']: r for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']
                   for r in records(a.inputs / name)}
    expected = records(a.inputs / 'benchmark-short.jsonl') + [r for r in records(a.inputs / 'text-ladder.jsonl')
                                                             if r['model'] == method['model']
                                                             and r['target_tokens'] == 8192]
    expected_ids = [r['id'] for r in expected]
    if expected_ids != method['corpus_ids']:
        p.error('corpus coverage differs')
    complete = json.loads((root / 'complete.json').read_text())
    events = records(root / 'progress.jsonl')
    issues, runs, reference, counts = list(complete['failures']), [], {}, collections.Counter()
    if not complete['execution_finished']:
        issues.append({'kind': 'incomplete_execution'})
    for variant in method['schedule']:
        counts[variant] += 1
        tag = f'{variant}-r{counts[variant]}'
        ready = [e for e in events if e['stage'] == 'server_ready' and e['tag'] == tag]
        starts = [e for e in events if e['stage'] == 'server_start' and e['tag'] == tag]
        finished = [e for e in events if e['stage'] == 'validation_finished' and e['tag'] == tag]
        observations = root / (tag + '-observations.jsonl')
        samples_file = root / (tag + '-resources.jsonl')
        if (len(ready) != 1 or len(starts) != 1 or len(finished) != 1
                or not observations.exists() or not samples_file.exists()):
            issues.append({'kind': 'run_coverage', 'tag': tag})
            continue
        e = ready[0]
        samples = records(samples_file)
        startup = [r for r in samples if r['scope']['name'] == 'startup']
        loaded = [r for r in samples if r['scope']['name'] == 'loaded-idle'
                  and r['kind'] == 'boundary' and r['scope'].get('marker') == 'begin']
        if (not startup or len(loaded) != 1 or any(r['errors'] or r.get('departed_processes')
                                                for r in startup + loaded)):
            issues.append({'kind': 'startup_resource_gap', 'tag': tag})
            continue
        if (not math.isfinite(e['startup_wall_s']) or e['startup_wall_s'] <= 0
                or e['startup_wall_s'] > method['startup_timeout'] + 2 or e['resource_errors']
                or not any(x['stage'] == 'server_stopped' and x['tag'] == tag for x in events)):
            issues.append({'kind': 'startup_or_lifecycle', 'tag': tag})
        row = {'tag': tag, 'variant': variant, 'repetition': counts[variant],
               'excluded_startup': counts[variant] == 1, 'startup_wall_s': e['startup_wall_s'],
               'aggregate_cpu_seconds_to_ready': loaded[0]['cpu_seconds'],
               'root_peak_rss_at_ready_bytes': loaded[0]['root_lifetime_peak_rss_bytes'],
               'sampled_startup_peak_pss_bytes': max(x['pss_bytes'] for x in startup + loaded),
               'root_counters_at_ready': e['counters'], 'resources': {}}
        for scope, clients in [('loaded-idle', None), ('validation', 1), ('drained-idle', None)]:
            usage = resource_audit.resources(samples, scope, clients, 1)
            if (usage['status'] == 'PASS' and scope != 'validation'
                    and usage['duration_s'] < method['idle_seconds']):
                usage = {'status': 'GAP', 'issues': ['idle shorter than scheduled']}
            if usage['status'] != 'PASS':
                issues.append({'kind': 'resource_gap', 'tag': tag, 'scope': scope, 'details': usage})
            row['resources'][scope] = usage
        rows = records(observations)
        if [r['id'] for r in rows] != expected_ids:
            issues.append({'kind': 'http_coverage', 'tag': tag})
        changed, valid_count = 0, 0
        for r in rows:
            source, value = definitions.get(r['id']), r.get('response', {})
            valid = (source is not None and r['http_status'] == 200 and r['valid']
                     and protocol.valid_answers(value, source['request']['questions']) and protocol.finite(value)
                     and not value.get('usage', {}).get('truncated') and value.get('model') == method['model']
                     and math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0)
            if source and 'target_tokens' in source:
                valid = valid and value.get('usage', {}).get('input_tokens') == source['target_tokens']
            if not valid:
                issues.append({'kind': 'invalid_http', 'tag': tag, 'id': r['id']})
                continue
            valid_count += 1
            if variant != 'upstream-eager':
                key, answer = r['id'], protocol.bits(value['answers'])
                if variant == 'baseline':
                    reference.setdefault(key, answer)
                equal = key in reference and answer == reference[key]
                changed += not equal
                if not equal or r['native_bits_equal'] != equal:
                    issues.append({'kind': 'native_bit_mismatch', 'tag': tag, 'id': key})
        if finished[0]['requests'] != len(rows) or finished[0]['native_bit_mismatches'] != changed:
            issues.append({'kind': 'validation_counter_mismatch', 'tag': tag})
        row.update(requests=len(rows), valid_requests=valid_count, native_bit_mismatches=changed)
        runs.append(row)
    summary = []
    for variant in counts:
        selected = [r for r in runs if r['variant'] == variant and not r['excluded_startup']]
        if len(selected) != 3:
            issues.append({'kind': 'retained_startup_count', 'variant': variant, 'count': len(selected)})
        if selected:
            metrics = {}
            for key in ['startup_wall_s', 'aggregate_cpu_seconds_to_ready', 'root_peak_rss_at_ready_bytes',
                        'sampled_startup_peak_pss_bytes']:
                values = [r[key] for r in selected]
                metrics[key] = dict(median=statistics.median(values), min=min(values), max=max(values))
            summary.append({'variant': variant, 'retained_starts': len(selected), 'metrics': metrics})
    result = {'status': 'FAIL' if issues else 'AUDITED', 'issues': issues, 'runs': runs, 'summary': summary,
              'scope': method['scope'], 'warning': 'RSS/PSS include file pages; not minimum viable RAM. '
              'Observed startup has uncontrolled filesystem/library/JIT caches. HTTP is validation, not loadtest.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'issues': len(issues), 'runs': len(runs)}))
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
