"""Replay Omni replica observations and resource windows without a GPU.

All input paths are explicit. Does not provision, change artifacts or publish.
Strict native answer equality and functional projections are separate checks.
"""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('omni_replicas_runner', 'omni-replicas.py')
resource_audit = load('omni_resource_audit', 'summarize-omni-resources.py')
protocol = runner.protocol


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def decision_projection(value):
    """Keep extracted numeric values, offsets and decisions; omit confidence diagnostics."""
    diagnostics = {'probabilities', 'scores', 'confidence', 'set_confidence', 'expected_score',
                   'probability', 'present_probability', 'line_probability', 'span_probability'}
    if isinstance(value, list):
        return [decision_projection(item) for item in value]
    if isinstance(value, dict):
        return {key: protocol.bits(item) if key in {'data', 'value'} else decision_projection(item)
                for key, item in value.items() if key not in diagnostics}
    return protocol.bits(value)


def latency(rows):
    values = [row['elapsed_ms'] for row in rows]
    return {name: percentile(values, fraction) for name, fraction in [('p50_ms', .5), ('p95_ms', .95), ('p99_ms', .99)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-artifacts', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output must be new')
    root = args.measurement_artifacts
    method = json.loads((root / 'method.json').read_text())
    if method['protocol'] != 'omni-replicas-v1':
        parser.error('unsupported measurement protocol')
    manifest, workloads = runner.corpus(args.inputs, method['model'])
    if method['inputs'] != manifest:
        parser.error('frozen input manifest differs')
    complete = json.loads((root / 'complete.json').read_text())
    events = records(root / 'progress.jsonl')
    definitions = {row['id']: row for row in workloads['mixed']}
    references = {}
    for row in records(root / 'preflight-single0-observations.jsonl'):
        if row['id'] in references or not row['valid'] or row['phase'] != 'reference-c1':
            parser.error('invalid independent GPU0 reference')
        references[row['id']] = row['response']['answers']
    if set(references) != set(definitions):
        parser.error('incomplete independent sequential reference')
    issues, phases, idles, differences = [], [], [], []
    if not complete['execution_finished'] or complete['failures']:
        issues.append(dict(kind='execution_incomplete_or_failed', details=complete['failures']))
    tags = ['preflight-single0', 'preflight-single1']
    variants = {tag: tag.removeprefix('preflight-') for tag in tags}
    for workload in method['workloads']:
        counts = collections.Counter()
        for variant in method['schedule']:
            counts[variant] += 1
            tag = f'{workload}-{variant}-r{counts[variant]}'
            tags.append(tag)
            variants[tag] = variant
    observations_count = comparisons = exact_changes = decision_changes = 0
    for tag in tags:
        observation_file = root / (tag + '-observations.jsonl')
        sample_file = root / (tag + '-resources.jsonl')
        if not observation_file.exists() or not sample_file.exists():
            issues.append(dict(kind='missing_run', tag=tag))
            continue
        observations, samples = records(observation_file), records(sample_file)
        observations_count += len(observations)
        ends = [event for event in events if event['stage'] == 'server_stopped' and event['tag'] == tag]
        if len(ends) != 1 or ends[0]['exit_code'] != 0:
            issues.append(dict(kind='server_shutdown', tag=tag))
        discovery = json.loads((root / (tag + '-models.json')).read_text())
        if [item.get('name') for item in discovery.get('models', [])] != [method['model']]:
            issues.append(dict(kind='model_discovery', tag=tag))
        for row in observations:
            source = definitions.get(row['id'])
            response = row.get('response', {})
            valid = (source is not None and row['http_status'] == 200 and row['valid']
                     and protocol.valid_answers(response, source['request']['questions'])
                     and protocol.finite(response) and response.get('model') == method['model']
                     and not response.get('usage', {}).get('truncated'))
            if source and 'target_tokens' in source:
                valid = valid and response.get('usage', {}).get('input_tokens') == source['target_tokens']
            if not valid:
                issues.append(dict(kind='invalid_response', tag=tag, id=row['id']))
                continue
            same = protocol.bits(response['answers']) == protocol.bits(references[row['id']])
            if row.get('reference_bitwise_equal') is not same:
                issues.append(dict(kind='online_answer_check_disagrees', tag=tag, id=row['id']))
            if tag != 'preflight-single0':
                comparisons += 1
                if not same:
                    functional_same = decision_projection(response['answers']) == decision_projection(references[row['id']])
                    exact_changes += 1
                    decision_changes += not functional_same
                    differences.append(dict(tag=tag, phase=row['phase'], index=row['index'], id=row['id'],
                                            decision_projection_equal=functional_same))
        for name in ['loaded-idle', 'drained-idle']:
            usage = resource_audit.resources(samples, name, None, method['sample_interval_s'])
            if usage['status'] != 'PASS':
                issues.append(dict(kind='idle_resource_gap', tag=tag, name=name))
            idles.append(dict(tag=tag, variant=variants[tag], name=name, resources=usage))
        for event in [event for event in events if event['stage'] == 'phase_finished'
                      and event['tag'] == tag and event['measured']]:
            name, clients = event['name'], event['clients']
            workload = name.rsplit('-c', 1)[0]
            rows = sorted((row for row in observations if row['phase'] == name), key=lambda row: row['index'])
            corpus = workloads[workload]
            minimum = len(corpus) * max(4, math.ceil(4 * clients / len(corpus)))
            coverage = (len(rows) >= minimum and len(rows) % len(corpus) == 0
                        and [row['index'] for row in rows] == list(range(len(rows)))
                        and [row['id'] for row in rows] == [corpus[i % len(corpus)]['id'] for i in range(len(rows))])
            valid = [row for row in rows if row['valid'] and row['http_status'] == 200]
            duration = event['end'] - event['begin']
            if (not coverage or len(valid) != len(rows) or duration < method['minimum_seconds']
                    or any(row['clients'] != clients or row['started_monotonic_s'] < event['begin']
                           or row['finished_monotonic_s'] > event['end'] for row in rows)):
                issues.append(dict(kind='phase_coverage_or_window', tag=tag, name=name))
            if duration <= 0 or not valid:
                continue
            qps = len(valid) / duration
            if not math.isclose(qps, event['successful_requests_per_s'], rel_tol=1e-10):
                issues.append(dict(kind='qps_disagrees', tag=tag, name=name))
            before = runner.metric_values((root / (tag + '-' + name + '-metrics-before.txt')).read_text(),
                                          'basal_replica_routed_requests_total')
            after_text = (root / (tag + '-' + name + '-metrics-after.txt')).read_text()
            after = runner.metric_values(after_text, 'basal_replica_routed_requests_total')
            routed = {key: value - before.get(key, 0) for key, value in after.items()}
            inflight = runner.metric_values(after_text, 'basal_replica_inflight_requests')
            if (sum(routed.values()) != len(rows) or routed != event['replica_routed']
                    or not inflight or any(inflight.values())):
                issues.append(dict(kind='replica_counters_disagree', tag=tag, name=name))
            usage = resource_audit.resources(samples, name, clients, method['sample_interval_s'])
            if usage['status'] != 'PASS':
                issues.append(dict(kind='resource_gap', tag=tag, name=name))
            gpu_usage = {}
            for device in [0, 1]:
                per_device = [dict(sample, **next((gpu for gpu in sample.get('gpus', [])
                                                  if gpu['device'] == device), {})) for sample in samples]
                gpu_usage[str(device)] = resource_audit.resources(per_device, name, clients, method['sample_interval_s'])
                if gpu_usage[str(device)]['status'] != 'PASS':
                    issues.append(dict(kind='per_gpu_resource_gap', tag=tag, name=name, device=device))
            classes = {kind: dict(samples=len(selected), **latency(selected))
                       for kind in ['short', 'long']
                       if (selected := [row for row in valid if row['workload_class'] == kind])}
            phases.append(dict(tag=tag, variant=variants[tag], workload=workload, clients=clients,
                               samples=len(rows), duration_s=duration, successful_requests_per_s=qps,
                               latency=latency(valid), classes=classes, resources=usage,
                               per_gpu_resources=gpu_usage, replica_routed=routed,
                               cpu_seconds_per_request=usage.get('cpu_core_seconds', 0) / len(valid)
                               if usage['status'] == 'PASS' else None))
    expected_phases = len(method['workloads']) * len(method['schedule']) * len(method['clients'])
    if len(phases) != expected_phases:
        issues.append(dict(kind='missing_phases', expected=expected_phases, actual=len(phases)))
    comparisons_table = []
    for workload in method['workloads']:
        for clients in method['clients']:
            groups = {variant: [phase for phase in phases if phase['workload'] == workload
                                and phase['clients'] == clients and phase['variant'] == variant]
                      for variant in ['single0', 'replicas']}
            if any(len(group) != len(method['schedule']) // 2 for group in groups.values()):
                continue
            metrics = {}
            for variant, group in groups.items():
                data = {'successful_requests_per_s': statistics.median(p['successful_requests_per_s'] for p in group)}
                data.update({key: statistics.median(p['latency'][key] for p in group)
                             for key in ['p50_ms', 'p95_ms', 'p99_ms']})
                good = [p for p in group if p['resources']['status'] == 'PASS']
                if len(good) == len(group):
                    data['cpu_seconds_per_request'] = statistics.median(p['cpu_seconds_per_request'] for p in good)
                    for key in ['pss_bytes', 'rss_bytes', 'process_vram_bytes', 'gpu_power_w', 'gpu_util_percent']:
                        data[key] = statistics.median(p['resources']['metrics'][key]['time_weighted_mean'] for p in good)
                metrics[variant] = data
            a, b = metrics['single0'], metrics['replicas']
            comparisons_table.append(dict(workload=workload, clients=clients, retained_starts_per_variant=len(groups['single0']),
                                          single=a, replicas=b,
                                          qps_scaling=b['successful_requests_per_s'] / a['successful_requests_per_s'],
                                          scaling_efficiency=b['successful_requests_per_s'] / a['successful_requests_per_s'] / 2))
    result = dict(protocol='omni-replicas-audit-v1',
                  status='PASS' if not issues and not exact_changes else 'FAIL_OR_GAP',
                  audit_status='PASS' if not issues else 'FAIL_OR_GAP', issues=issues,
                  observed_responses=observations_count, compared_to_independent_reference=comparisons,
                  exact_answer_changes=exact_changes, exact_answer_status='FAIL' if exact_changes else 'PASS',
                  decision_projection_changes=decision_changes,
                  decision_projection_status='FAIL' if decision_changes else 'PASS',
                  decision_projection_scope='Confidence/probability diagnostics omitted; extracted data/value, '
                  'decisions, offsets, automation and remaining metadata retained. This is not FP32 quality.',
                  differences=differences, phases=phases, idle_windows=idles, comparisons=comparisons_table,
                  complete=complete, method=method)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ['method', 'phases', 'comparisons', 'idle_windows', 'differences']}))


if __name__ == '__main__':
    main()
