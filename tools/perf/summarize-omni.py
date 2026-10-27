"""Audit one frozen Omni HTTP run locally and summarize retained repetitions.

Confirmation excludes each engine's first repetition. Screens remain SIGNAL.
Every scheduled measured phase is checked against the SHA-verified inputs;
answers, HTTP validity, counts, token lengths and native bit invariance are
recomputed from observations. Quality against FP32 is a separate gate.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('omni_runner', Path(__file__).with_name('omni-optimize.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latency_metrics(rows):
    values = sorted(row['elapsed_ms'] for row in rows)
    return dict(median_ms=statistics.median(values),
                p95_ms=values[min(len(values)-1, int(.95*len(values)))])


def summarize_groups(groups, labels):
    return [dict(zip(labels, key), repetitions=len(values),
                 metrics={metric: dict(median=statistics.median(v[metric] for v in values),
                                       min=min(v[metric] for v in values), max=max(v[metric] for v in values))
                          for metric in values[0]}) for key, values in sorted(groups.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='one screen/confirm result directory')
    p.add_argument('--inputs', type=Path, required=True, help='restored frozen corpus used by that run')
    p.add_argument('--ladder-choice', help='Optional known correct option key for every single-question text ladder')
    p.add_argument('--surfaces', nargs='+', choices=['text', 'media'],
                   help='Audit independently complete surfaces from an interrupted combined run; retains parent status')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new')
    root = a.measurement_artifacts
    method = json.loads((root / 'method.json').read_text())
    complete_path = root / 'complete.json'
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text())
    elif a.surfaces:
        complete = {'execution_finished': False, 'failures': []}
    else:
        p.error('run has no complete.json; --surfaces may select independently completed surfaces')
    surfaces = a.surfaces or method['surfaces']
    if not set(surfaces).issubset(method['surfaces']):
        p.error('selected surfaces were not scheduled by this run')
    provenance = json.loads((a.inputs / 'provenance.json').read_text())
    if provenance != method['inputs']:
        p.error('frozen input manifest differs from the run')
    for rel, expected in provenance['files'].items():
        path = (a.inputs / rel).resolve()
        if not path.is_relative_to(a.inputs.resolve()) or runner.digest(path) != expected:
            p.error('input checksum mismatch: ' + rel)
    text_phases = method.get('text_phases', ['ladder', 'short'] + (['mixed'] if method.get('mixed') else []))
    short_clients = method.get('short_clients', [1, 8, 32])
    if (not isinstance(short_clients, list) or not short_clients
            or any(type(c) is not int or c <= 0 for c in short_clients)
            or len(set(short_clients)) != len(short_clients)):
        p.error('invalid scheduled short client counts')
    ladder = read_rows(a.inputs / 'text-ladder.jsonl') if 'text' in surfaces and any(x in text_phases for x in ['ladder', 'mixed']) else []
    short = read_rows(a.inputs / 'benchmark-short.jsonl') if 'text' in surfaces else []
    media = read_rows(a.inputs / 'cases-real-media.jsonl') if 'media' in surfaces else []
    events = read_rows(root / 'progress.jsonl')
    issues = [dict(kind='run_failure', **x) for x in complete['failures']
              if not a.surfaces or x.get('surface') is None or x['surface'] in surfaces]
    if not complete.get('execution_finished') and not a.surfaces:
        issues.append(dict(kind='execution_incomplete'))
    seen, phases, aggregated = {}, [], collections.defaultdict(list)
    token_counts = {}
    class_groups, case_groups, short_case_groups = (collections.defaultdict(list) for _ in range(3))
    confirmed = method['phase'] == 'confirm'
    if (not text_phases or len(set(text_phases)) != len(text_phases)
            or not set(text_phases).issubset({'ladder', 'short', 'mixed'})
            or bool(method.get('mixed')) != ('mixed' in text_phases)):
        p.error('invalid scheduled text phases')
    for model in method.get('model_names', ['basal-omni-1.0-' + suffix for suffix in method['models']]):
        for surface in surfaces:
            counts = collections.Counter()
            for variant in method['schedule']:
                if surface == 'media' and variant == 'upstream-fast':
                    continue
                counts[variant] += 1
                repetition = counts[variant]
                tag = f'{model}-{surface}-{variant}-r{repetition}'
                if a.surfaces and not any(e.get('stage') == 'server_stopped' and e.get('tag') == tag for e in events):
                    issues.append(dict(kind='selected_surface_server_not_stopped', tag=tag))
                path = root / (tag + '-observations.jsonl')
                if not path.exists():
                    issues.append(dict(kind='missing_observations', tag=tag))
                    continue
                rows = read_rows(path)
                selected_ladder = [r for r in ladder if r['model'] == model and r['target_tokens'] <= method['text_limit']]
                expected = {}
                if surface == 'text':
                    if 'ladder' in text_phases:
                        for row in selected_ladder:
                            expected[('ladder', 1, row['id'])] = ([row], method.get('latency_samples', 10 if confirmed else 4))
                    repeats = max(1, math.ceil((method.get('requested_load_samples') or (512 if confirmed else 64)) / len(short)))
                    if 'short' in text_phases:
                        for clients in short_clients:
                            expected[('short', clients, 'all')] = (short, repeats)
                    if 'mixed' in text_phases:
                        longs = [r for r in ladder if r['model'] == model and r['target_tokens'] == 8192]
                        mixed = []
                        for i, row in enumerate(short):
                            mixed.append(row)
                            if (i + 1) % 16 == 0:
                                mixed.append(longs[(i // 16) % len(longs)])
                        if len(short) < 16:
                            mixed.append(longs[0])
                        for clients in [8, 32]:
                            expected[('mixed', clients, 'all')] = (mixed, 2 if confirmed else 1)
                else:
                    expected[('media-sequential', 1, 'all')] = (media, method.get('media_sequential_repeats', 3 if confirmed else 1))
                    expected[('media-load', 4, 'all')] = (media, method.get('media_load_repeats', 8 if confirmed else 2))
                definitions = {r['id']: r for r in selected_ladder + short + media +
                               [r for r in ladder if r['model'] == model and r['target_tokens'] == 8192]}
                for row in rows:
                    source = definitions.get(row['id'])
                    response = row.get('response', {})
                    valid = (source is not None and row.get('http_status') == 200 and runner.finite(response)
                             and runner.valid_answers(response, source['request']['questions'])
                             and not response.get('usage', {}).get('truncated'))
                    if source and 'target_tokens' in source:
                        valid = valid and response.get('usage', {}).get('input_tokens') == source['target_tokens']
                        if a.ladder_choice:
                            answers = response.get('answers', {})
                            valid = valid and len(answers) == 1 and next(iter(answers.values())).get('choice') == a.ladder_choice
                    if not valid or not row.get('valid') or not math.isfinite(row.get('elapsed_ms', math.nan)):
                        issues.append(dict(kind='invalid_response', tag=tag, id=row['id'], phase=row['phase']))
                        continue
                    token_key = (model, row['id'])
                    tokens = response.get('usage', {}).get('input_tokens')
                    task = any(q.get('type') in ('classify','verify','compress','extract') for q in source['request']['questions'].values())
                    # A task's dependent extraction prompt is chosen by the model. Keep the response's work
                    # accounting; compare HTTP requests rather than requiring identical internal work across engines.
                    allow_task_work = task and method.get('task_accounting') is not None
                    if not isinstance(tokens, int) or tokens <= 0 or (not allow_task_work and token_key in token_counts and token_counts[token_key] != tokens):
                        issues.append(dict(kind='input_token_count_mismatch', tag=tag, id=row['id']))
                    token_counts.setdefault(token_key, tokens)
                    key = (model, variant, row['id'])
                    answer = runner.bits(response['answers'])
                    if key in seen and seen[key] != answer and not variant.startswith('upstream-'):
                        issues.append(dict(kind='native_answer_changed', tag=tag, id=row['id'], phase=row['phase']))
                    seen.setdefault(key, answer)
                for key, (corpus, repeats) in expected.items():
                    name, clients, scope = key
                    observed = [r for r in rows if r['phase'] == name and r['clients'] == clients
                                and (scope == 'all' or r['id'] == scope)]
                    matching = [e for e in events if e['stage'] == 'phase_finished' and e['tag'] == tag
                                and e['phase'] == name and e['clients'] == clients
                                and (scope == 'all' or e['ids'] == [scope])]
                    wanted = [r['id'] for r in corpus] * repeats
                    if (len(matching) != 1 or [r['id'] for r in observed] != wanted
                            or [r['index'] for r in observed] != list(range(len(wanted)))):
                        issues.append(dict(kind='phase_coverage', tag=tag, phase=key))
                        continue
                    event = matching[0]
                    if event['invalid'] or len(observed) != event['samples'] or event['wall_s'] <= 0:
                        issues.append(dict(kind='phase_invalid', tag=tag, phase=key))
                        continue
                    latencies = sorted(r['elapsed_ms'] for r in observed if r.get('valid'))
                    metrics = dict(median_ms=statistics.median(latencies),
                                   p95_ms=latencies[min(len(latencies)-1, int(.95*len(latencies)))],
                                   successful_requests_per_s=len(latencies)/event['wall_s'])
                    if any(not math.isclose(metrics[k], event[k], rel_tol=1e-10, abs_tol=1e-10) for k in metrics):
                        issues.append(dict(kind='phase_metric_mismatch', tag=tag, phase=key))
                    classes = collections.defaultdict(list)
                    for row in observed:
                        classes[row['workload_class']].append(row)
                    for label, members in classes.items():
                        recorded = event.get('classes', {}).get(label)
                        if recorded is not None and (recorded['samples'] != len(members)
                                or any(not math.isclose(value, recorded[metric], rel_tol=1e-10, abs_tol=1e-10)
                                       for metric, value in latency_metrics(members).items())):
                            issues.append(dict(kind='class_metric_mismatch', tag=tag, phase=key, label=label))
                    retained = repetition not in method.get('excluded_repetitions', [])
                    phases.append(dict(tag=tag, phase=name, clients=clients, scope=scope, retained=retained,
                                       samples=len(observed), **metrics))
                    if retained:
                        aggregated[(model, surface, name, clients, scope, variant)].append(metrics)
                        if name == 'short':
                            for case in corpus:
                                members = [row for row in observed if row['id'] == case['id']]
                                short_case_groups[(model, clients, case['id'], variant)].append(latency_metrics(members))
                        if name == 'mixed':
                            short_members = [row for row in observed if row['workload_class'].startswith('distinct-short-')]
                            if short_members:
                                classes['short-all-languages'] = short_members
                            for label, members in classes.items():
                                class_groups[(model, name, clients, label, variant)].append(latency_metrics(members))
                        if name == 'media-sequential':
                            for case in corpus:
                                members = [row for row in observed if row['id'] == case['id']]
                                case_groups[(model, case['id'], variant)].append(latency_metrics(members))
    summaries = summarize_groups(aggregated, ['model', 'surface', 'phase', 'clients', 'scope', 'variant'])
    result = dict(status='FAIL' if issues else 'AUDITED' if confirmed else 'SIGNAL',
                  quality_gate='separate upstream FP32 audit required', issues=issues, phases=phases, summary=summaries,
                  checked_input_token_counts=len(token_counts),
                  class_summary=summarize_groups(class_groups, ['model', 'phase', 'clients', 'workload_class', 'variant']),
                  media_case_summary=summarize_groups(case_groups, ['model', 'id', 'variant']),
                  short_case_summary=summarize_groups(short_case_groups, ['model', 'clients', 'id', 'variant']),
                  summary_method='median of retained repetition metrics; ranges retain between-run variation',
                  inputs_sha256=hashlib.sha256((a.inputs/'provenance.json').read_bytes()).hexdigest())
    if a.ladder_choice:
        result['ladder_expected_choice'] = a.ladder_choice
    if 'text_phases' in method and 'text' in surfaces:
        result['audited_text_phases'] = text_phases
    if a.surfaces:
        result['audited_surfaces'] = surfaces
        result['parent_execution_finished'] = complete.get('execution_finished', False)
        result['scope'] = 'Selected surfaces only; each scheduled phase and server stop is required. Parent execution status is preserved.'
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], issues=len(issues), summarized_phases=len(summaries))))
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
