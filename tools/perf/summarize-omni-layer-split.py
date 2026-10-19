"""Replay Gemma layer-split HTTP outputs and resource windows without a GPU."""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replicas = load('split_replica_audit', 'summarize-omni-replicas.py')
protocol = replicas.protocol
resources = replicas.resource_audit.resources


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('output must be new')
    root = args.measurement_artifacts
    method = json.loads((root / 'method.json').read_text())
    if method['protocol'] != 'omni-layer-split-v1' or protocol.digest(root / 'inputs.jsonl') != method['inputs_sha256']:
        p.error('unsupported protocol or input checksum mismatch')
    definitions = records(root / 'inputs.jsonl')
    ids = [row['id'] for row in definitions]
    if not ids or len(set(ids)) != len(ids):
        p.error('nonunique inputs')
    sources = {row['id']: row for row in definitions}
    complete = json.loads((root / 'complete.json').read_text())
    events = records(root / 'progress.jsonl')
    reference_tag = method['variants'][0]['tag']
    references = {row['id']: row.get('response', {}).get('answers') for row in records(root / (reference_tag + '-observations.jsonl'))
                  if row['clients'] == 1 and row['valid']}
    issues, phases, differences = [], [], []
    if not complete['execution_finished'] or complete['failures']:
        issues.append(dict(kind='incomplete_execution', failures=complete['failures']))
    if set(references) != set(ids):
        issues.append(dict(kind='incomplete_sequential_reference'))
    for variant in method['variants']:
        tag = variant['tag']
        file = root / (tag + '-observations.jsonl')
        if not file.exists():
            issues.append(dict(kind='missing_variant', tag=tag))
            continue
        observations = records(file)
        samples = records(root / (tag + '-resources.jsonl'))
        stopped = [event for event in events if event['stage'] == 'server_stopped' and event['tag'] == tag]
        if len(stopped) != 1 or stopped[0]['exit_code']:
            issues.append(dict(kind='server_shutdown', tag=tag))
        phase_events = [event for event in events if event['stage'] == 'phase_finished' and event['tag'] == tag]
        if sorted(event['clients'] for event in phase_events) != method['clients']:
            issues.append(dict(kind='missing_or_duplicate_phases', tag=tag))
        for event in phase_events:
            clients = event['clients']
            rows = sorted((row for row in observations if row['clients'] == clients), key=lambda row: row['index'])
            expected_ids = ids * (1 if clients == 1 else method['load_repeats'])
            if [row['index'] for row in rows] != list(range(len(expected_ids))) or [row['id'] for row in rows] != expected_ids:
                issues.append(dict(kind='phase_coverage', tag=tag, clients=clients))
            valid = []
            for row in rows:
                source = sources.get(row['id'])
                response = row.get('response', {})
                ok = (source is not None and row.get('http_status') == 200 and row['valid']
                      and protocol.valid_answers(response, source['request']['questions'])
                      and protocol.finite(response) and response.get('model') == method['model']
                      and not response.get('usage', {}).get('truncated'))
                if source and 'target_tokens' in source:
                    ok = ok and response.get('usage', {}).get('input_tokens') == source['target_tokens']
                if row['id'].startswith('image-'):
                    media = response.get('usage', {}).get('media', {})
                    ok = ok and media.get('images') == len(source['request']['media']) and media.get('media_tokens', 0) > 0
                if not ok:
                    issues.append(dict(kind='invalid_response', tag=tag, clients=clients, id=row['id']))
                    continue
                if (not event['begin'] <= row['started_monotonic_s'] < row['finished_monotonic_s'] <= event['end']
                        or not math.isclose(row['elapsed_ms'], 1000*(row['finished_monotonic_s']-row['started_monotonic_s']), rel_tol=1e-10)):
                    issues.append(dict(kind='timing_disagrees', tag=tag, clients=clients, id=row['id']))
                valid.append(row)
                same = protocol.bits(response['answers']) == protocol.bits(references.get(row['id']))
                if same != row.get('reference_bitwise_equal'):
                    issues.append(dict(kind='online_comparison_disagrees', tag=tag, clients=clients, id=row['id']))
                if not same:
                    differences.append(dict(tag=tag, clients=clients, id=row['id'], index=row['index'],
                        decision_projection_equal=replicas.decision_projection(response['answers']) ==
                        replicas.decision_projection(references.get(row['id']))))
            metrics = (root / (tag + f'-c{clients}-metrics-after.txt')).read_text()
            inflight = replicas.runner.metric_values(metrics, 'basal_replica_inflight_requests')
            available = replicas.runner.metric_values(metrics, 'basal_replica_available')
            if not inflight or any(inflight.values()) or not available or any(value != 1 for value in available.values()):
                issues.append(dict(kind='queue_drain_or_availability', tag=tag, clients=clients))
            usage = resources(samples, f'c{clients}', clients, 1)
            gpus = {}
            for device in [0, 1]:
                per_device = [dict(sample, **next((gpu for gpu in sample.get('gpus', []) if gpu['device'] == device), {}))
                              for sample in samples]
                gpus[str(device)] = resources(per_device, f'c{clients}', clients, 1)
            if usage['status'] != 'PASS' or any(value['status'] != 'PASS' for value in gpus.values()):
                issues.append(dict(kind='resource_gap', tag=tag, clients=clients))
            duration = event['end'] - event['begin']
            qps = len(valid)/duration if duration > 0 else 0
            if not math.isclose(qps, event['successful_requests_per_s'], rel_tol=1e-10):
                issues.append(dict(kind='qps_disagrees', tag=tag, clients=clients))
            classes = {}
            for name, selected in [('text', [row for row in valid if not row['id'].startswith('image-')]),
                                   ('images', [row for row in valid if row['id'].startswith('image-')])]:
                if selected:
                    classes[name] = dict(samples=len(selected), **replicas.latency(selected))
            phases.append(dict(tag=tag, placement=variant['placement'], clients=clients, samples=len(rows),
                successful_requests_per_s=qps, duration_s=duration, resources=usage, per_gpu_resources=gpus,
                latency=replicas.latency(valid) if valid else {}, classes=classes))
    output = dict(audit_status='FAIL_OR_GAP' if issues else 'PASS', issues=issues, phases=phases,
                  exact_answer_status='FAIL' if differences else 'PASS' if not issues else 'GAP',
                  decision_projection_status='FAIL' if any(not row['decision_projection_equal'] for row in differences)
                  else 'PASS' if not issues else 'GAP', differences=differences,
                  method=method, comparisons=collections.Counter(row['tag'] for row in differences),
                  quality_scope='Native placement equivalence; not a new FP32 quality evaluation.')
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: output[key] for key in ['audit_status', 'exact_answer_status', 'decision_projection_status']}))


if __name__ == '__main__':
    main()
