"""Replay actual HTTP and Linux resource observations; no model or rental execution."""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('omni_http', Path(__file__).with_name('omni-optimize.py'))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * len(values)))]


def resources(samples, name, clients, interval):
    selected = [row for row in samples if row['scope']['name'] == name
                and row['scope'].get('clients') == clients]
    begins = [s for s in selected if s['kind'] == 'boundary' and s['scope'].get('marker') == 'begin']
    ends = [s for s in selected if s['kind'] == 'boundary' and s['scope'].get('marker') == 'end']
    issues = []
    if len(begins) != 1 or len(ends) != 1:
        return {'status': 'GAP', 'issues': ['missing or duplicate resource boundaries']}
    begin, end = begins[0], ends[0]
    rows = [s for s in selected if begin['monotonic_s'] <= s['monotonic_s'] <= end['monotonic_s']]
    if any(s['errors'] for s in rows):
        issues.append('sampler errors in resource window')
    if any(s.get('departed_processes') for s in rows):
        issues.append('departed server child: CPU after last sample may be missing')
    times = [s['monotonic_s'] for s in rows]
    gaps = [right - left for left, right in zip(times, times[1:])]
    duration = end['monotonic_s'] - begin['monotonic_s']
    if duration <= 0 or len(rows) < 3 or not gaps or min(gaps) < 0 or max(gaps) > 2.5 * interval:
        issues.append('insufficient sample coverage or nonmonotonic timestamps')
    if issues:
        return {'status': 'GAP', 'issues': issues, 'samples': len(rows)}
    cpu = end['cpu_seconds'] - begin['cpu_seconds']
    if cpu < 0:
        return {'status': 'GAP', 'issues': ['decreasing aggregate CPU counter']}
    metrics = {}
    keys = ['rss_bytes', 'pss_bytes', 'uss_bytes', 'swap_bytes', 'process_vram_bytes',
            'device_vram_bytes', 'gpu_util_percent', 'gpu_power_w', 'gpu_temperature_c', 'gpu_sm_mhz']
    for key in keys:
        values = [s[key] for s in rows]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in values):
            return {'status': 'GAP', 'issues': ['invalid resource value: ' + key]}
        mean = sum((left[key] + right[key]) / 2 * dt
                   for left, right, dt in zip(rows, rows[1:], gaps)) / duration
        metrics[key] = {'time_weighted_mean': mean, 'min': min(values), 'sampled_max': max(values)}
    # Closely adjacent control/periodic samples can share one scheduler tick;
    # estimate peak CPU only over windows at least half a sampling interval.
    cpu_rates, anchor = [], rows[0]
    for row in rows[1:]:
        dt = row['monotonic_s'] - anchor['monotonic_s']
        if dt >= .5 * interval:
            cpu_rates.append((row['cpu_seconds'] - anchor['cpu_seconds']) / dt)
            anchor = row
    return {'status': 'PASS', 'issues': [], 'samples': len(rows), 'duration_s': duration,
            'max_sample_gap_s': max(gaps), 'cpu_core_seconds': cpu,
            'cpu_mean_cores': cpu / duration, 'cpu_sampled_peak_cores': max(cpu_rates, default=0),
            'root_lifetime_peak_rss_bytes': max(s['root_lifetime_peak_rss_bytes'] for s in rows),
            'metrics': metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--native-reference', type=Path,
                   help='frozen native answers and binary/input provenance from previous validated run')
    p.add_argument('--ladder-choice')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new')
    method = json.loads((a.measurement_artifacts / 'method.json').read_text())
    if (method['protocol'] != 'omni-process-resources-v1'
            or not method['clients'] or method['clients'] != sorted(set(method['clients']))
            or not all(1 <= c <= 128 for c in method['clients'])
            or method['schedule'] not in [['candidate', 'upstream-eager', 'upstream-eager', 'candidate'],
                                          ['upstream-eager', 'candidate', 'candidate', 'upstream-eager'],
                                          ['candidate', 'upstream-eager'], ['upstream-eager', 'candidate']]):
        p.error('invalid resource protocol schedule')
    complete = json.loads((a.measurement_artifacts / 'complete.json').read_text())
    manifest = json.loads((a.inputs / 'provenance.json').read_text())
    if manifest != method['inputs']:
        p.error('input provenance differs')
    # This campaign uses text only; the parent manifest also names unused media.
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        sha = manifest['files'][name]
        file = (a.inputs / name).resolve()
        if not file.is_relative_to(a.inputs.resolve()) or protocol.digest(file) != sha:
            p.error('input checksum mismatch')
    short = records(a.inputs / 'benchmark-short.jsonl')
    ladder = records(a.inputs / 'text-ladder.jsonl')
    definitions = {row['id']: row for row in short + ladder}
    reference = json.loads(a.native_reference.read_text()) if a.native_reference else None
    if reference and (reference['candidate_sha256'] != method['candidate_sha256']
                      or reference['inputs'] != manifest):
        p.error('native reference provenance differs')
    events = records(a.measurement_artifacts / 'progress.jsonl')
    seen, phases, idles, startups, failed_points, issues = {}, [], [], [], [], list(complete['failures'])
    if not complete['execution_finished']:
        issues.append({'kind': 'incomplete_execution'})
    for workload in method['workloads']:
        long_rows = [row for row in ladder if row['model'] == method['model'] and row['target_tokens'] == 8192]
        corpus = list(short)
        if workload == 'mixed':
            corpus = []
            for i, row in enumerate(short):
                corpus.append(row)
                if (i + 1) % 16 == 0:
                    corpus.append(long_rows[(i // 16) % len(long_rows)])
        counts = collections.Counter()
        for variant in method['schedule']:
            counts[variant] += 1
            tag = f'{workload}-{variant}-r{counts[variant]}'
            observation_file = a.measurement_artifacts / (tag + '-observations.jsonl')
            sample_file = a.measurement_artifacts / (tag + '-resources.jsonl')
            if not observation_file.exists() or not sample_file.exists():
                issues.append({'kind': 'missing_run', 'tag': tag})
                continue
            observations, samples = records(observation_file), records(sample_file)
            startup_rows = [r for r in samples if r['scope']['name'] == 'startup' and not r['errors']]
            loaded = next((r for r in samples if r['scope']['name'] == 'loaded-idle'
                           and r['kind'] == 'boundary' and r['scope'].get('marker') == 'begin'), None)
            if loaded and not loaded['errors']:
                startups.append({'tag': tag, 'root_peak_rss_at_ready_bytes': loaded['root_lifetime_peak_rss_bytes'],
                                 'sampled_startup_peak_pss_bytes': max([r['pss_bytes'] for r in startup_rows]
                                                                      + [loaded['pss_bytes']]),
                                 'scope': 'Root lifetime RSS high-water at server ready; startup PSS is sampled.'})
            else:
                issues.append({'kind': 'startup_resource_gap', 'tag': tag})
            if not any(e['stage'] == 'server_stopped' and e['tag'] == tag for e in events):
                issues.append({'kind': 'server_not_stopped', 'tag': tag})
            for row in observations:
                source, response = definitions.get(row['id']), row.get('response', {})
                valid = (source is not None and row['http_status'] == 200 and row['valid']
                         and protocol.valid_answers(response, source['request']['questions'])
                         and protocol.finite(response) and not response.get('usage', {}).get('truncated')
                         and response.get('model') == method['model']
                         and isinstance(response.get('usage', {}).get('input_tokens'), int)
                         and response['usage']['input_tokens'] > 0)
                if source and 'target_tokens' in source:
                    valid = valid and response.get('usage', {}).get('input_tokens') == source['target_tokens']
                    if a.ladder_choice:
                        valid = valid and len(response['answers']) == 1 and next(
                            iter(response['answers'].values())).get('choice') == a.ladder_choice
                dt = row['finished_monotonic_s'] - row['started_monotonic_s']
                valid = valid and dt > 0 and math.isclose(dt * 1000, row.get('elapsed_ms', math.nan),
                                                        rel_tol=1e-10, abs_tol=1e-8)
                if not valid:
                    issues.append({'kind': 'invalid_response', 'tag': tag, 'id': row['id']})
                    continue
                key, value = (variant, row['id']), protocol.bits(response['answers'])
                if variant == 'candidate':
                    if key in seen and seen[key] != value:
                        issues.append({'kind': 'native_answer_changed', 'tag': tag, 'id': row['id']})
                    if reference and protocol.bits(reference['answers'].get(row['id'])) != value:
                        issues.append({'kind': 'native_reference_mismatch', 'tag': tag, 'id': row['id']})
                seen.setdefault(key, value)
            for name in ['loaded-idle', 'warm-idle'] + [f'drained-idle-c{c}' for c in method['clients']]:
                usage = resources(samples, name, None, method['sample_interval_s'])
                if usage['status'] == 'PASS' and usage['duration_s'] < method['idle_seconds']:
                    usage = {'status': 'GAP', 'issues': ['idle interval shorter than scheduled']}
                if usage['status'] != 'PASS':
                    issues.append({'kind': 'idle_resource_gap', 'tag': tag, 'name': name})
                idles.append(dict(tag=tag, workload=workload, variant=variant, name=name, resources=usage))
            for clients in method['clients']:
                name = f'load-c{clients}'
                rows = [r for r in observations if r['phase'] == name]
                matches = [e for e in events if e['stage'] == 'phase_finished' and e['tag'] == tag and e['name'] == name]
                minimum = len(corpus) * max(method['minimum_corpus_passes'], math.ceil(
                    method['minimum_requests_per_client'] * clients / len(corpus)))
                point_usage = resources(samples, name, clients, method['sample_interval_s'])
                if (len(matches) != 1 or not rows or len(rows) < minimum or len(rows) % len(corpus)
                        or [r['index'] for r in rows] != list(range(len(rows)))
                        or [r['id'] for r in rows] != [corpus[i % len(corpus)]['id'] for i in range(len(rows))]):
                    issues.append({'kind': 'phase_coverage', 'tag': tag, 'clients': clients})
                    failed_points.append({'tag': tag, 'clients': clients, 'status': 'FAIL',
                                          'observed_requests': len(rows), 'resources': point_usage})
                    continue
                event = matches[0]
                valid = [r for r in rows if r.get('valid') and r.get('http_status') == 200]
                wall = event['end'] - event['begin']
                if (event['samples'] != len(rows) or event['invalid'] or len(valid) != len(rows)
                        or wall < method['minimum_seconds'] or event['minimum_samples'] != minimum):
                    issues.append({'kind': 'invalid_phase', 'tag': tag, 'clients': clients})
                    failed_points.append({'tag': tag, 'clients': clients, 'status': 'FAIL',
                                          'observed_requests': len(rows), 'resources': point_usage})
                    continue
                if any(r['clients'] != clients or r['started_monotonic_s'] < event['begin']
                       or r['finished_monotonic_s'] > event['end'] for r in rows):
                    issues.append({'kind': 'request_window', 'tag': tag, 'clients': clients})
                qps = len(valid) / wall
                if not math.isclose(qps, event['successful_requests_per_s'], rel_tol=1e-10):
                    issues.append({'kind': 'phase_qps_mismatch', 'tag': tag, 'clients': clients})
                timeline = sorted([(r['started_monotonic_s'], 1) for r in rows]
                                  + [(r['finished_monotonic_s'], -1) for r in rows])
                active, peak = 0, 0
                for _, delta in timeline:
                    active += delta
                    peak = max(peak, active)
                if peak != clients or active != 0:
                    issues.append({'kind': 'observed_concurrency', 'tag': tag, 'clients': clients, 'peak': peak})
                usage = point_usage
                if usage['status'] != 'PASS':
                    issues.append({'kind': 'phase_resource_gap', 'tag': tag, 'clients': clients, 'details': usage['issues']})
                else:
                    usage['cpu_core_seconds_per_request'] = usage['cpu_core_seconds'] / len(valid)
                phases.append({'tag': tag, 'workload': workload, 'variant': variant, 'clients': clients,
                               'samples': len(rows), 'wall_s': wall, 'successful_requests_per_s': qps,
                               'median_ms': statistics.median(r['elapsed_ms'] for r in valid),
                               'p95_ms': percentile([r['elapsed_ms'] for r in valid], .95),
                               'observed_peak_clients': peak, 'mean_active_clients': sum(
                                   r['finished_monotonic_s'] - r['started_monotonic_s'] for r in rows) / wall,
                               'resources': usage})
    groups = collections.defaultdict(list)
    for row in phases:
        if row['resources']['status'] == 'PASS':
            groups[row['workload'], row['variant'], row['clients']].append(row)
    summary = []
    for (workload, variant, clients), rows in sorted(groups.items()):
        metrics = {key: [r[key] for r in rows] for key in ['successful_requests_per_s', 'median_ms', 'p95_ms']}
        for key in ['cpu_mean_cores', 'cpu_sampled_peak_cores', 'cpu_core_seconds_per_request']:
            metrics[key] = [r['resources'][key] for r in rows]
        for key in ['rss_bytes', 'pss_bytes', 'uss_bytes', 'process_vram_bytes', 'device_vram_bytes']:
            for statistic in ['time_weighted_mean', 'sampled_max']:
                metrics[key + '_' + statistic] = [r['resources']['metrics'][key][statistic] for r in rows]
        summary.append({'workload': workload, 'variant': variant, 'clients': clients, 'repetitions': len(rows),
                        'metrics': {k: {'median': statistics.median(v), 'min': min(v), 'max': max(v)}
                                    for k, v in metrics.items()}})
    resource_gaps = {'idle_resource_gap', 'phase_resource_gap', 'startup_resource_gap'}
    status = ('FAIL' if any(x.get('kind') not in resource_gaps for x in issues) else 'GAP' if issues
              else 'SIGNAL' if method['phase'] == 'screen' else 'AUDITED')
    result = {'status': status,
              'issues': issues, 'phases': phases, 'idle': idles, 'startup': startups,
              'failed_points': failed_points, 'summary': summary,
              'scope': method['resource_scope'], 'quality': 'Unchanged binary; native wire answers audited '
              'across clients/repetitions and optional frozen native reference. FP32 quality provenance is separate.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'issues': len(issues), 'phases': len(phases)}))
    if issues:
        raise SystemExit(3 if status == 'GAP' else 2)


if __name__ == '__main__':
    main()
