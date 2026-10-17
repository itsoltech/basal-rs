"""Audit omni-quality.py outputs; CLI compare exit zero alone is not acceptance.

Require complete coverage, identical token sequences, categorical agreement with
FP32, endpoint field/error compatibility and bit invariance across batching modes.
Optional MODEL=FILE comparisons measure the upstream BF16 baseline on the same
corpus. Aggregate TV uses unique heads, not repeated batching-mode observations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def metrics(comparison):
    bench = comparison['bench']
    system = comparison['systemone']['numeric_identical_tokens']
    count = bench['items'] + system['items']
    total = sum(section['items'] * section['tv_cal']['mean'] for section in [bench, system] if section['items'])
    return dict(heads=count, total_tv_cal=total, mean_tv_cal=total/count if count else None)


def categorical_issues(comparison):
    issues = []
    if comparison['missing'] or comparison['tokens']['mismatches'] or comparison['tokens'].get('known_deviations'):
        issues.append('missing records, token mismatches or known deviations')
    for name, section in [('bench', comparison['bench']),
                          ('API', comparison['systemone']['numeric_identical_tokens'])]:
        same, total = map(int, section['argmax_agreement_cal'].split('/'))
        if same != total or total != section['items']:
            issues.append(name + ': calibrated argmax or coverage differs')
        for key in ['mean', 'max']:
            value = section['tv_cal'][key]
            if section['items'] and (value is None or not math.isfinite(value) or not 0 <= value <= 1):
                issues.append(name + ': invalid TV distance')
    for case in comparison['upstream_endpoint']:
        if not case['errors_agree']:
            issues.append('endpoint errors differ: ' + str(case['id']))
        for question in case.get('questions', []):
            for key, value in question['diff'].items():
                if key.endswith('_same') and value is False:
                    issues.append('endpoint ' + key + ' differs: ' + str(case['id']) + '/' + question['question'])
    return issues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='one complete omni-quality output')
    p.add_argument('--bf16-comparison', action='append', default=[], help='mini=FILE or max=FILE, same combined corpus')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error('output must be new')
    root = a.measurement_artifacts.resolve()
    method = json.loads((root/'method.json').read_text())
    complete = json.loads((root/'complete.json').read_text())
    issues = []
    if not complete['execution_finished'] or not complete['batching_bitwise_equal']:
        issues.append(dict(scope='batching', issue='incomplete execution or non-identical export records'))
    expected = {('basal-omni-1.0-'+model, corpus, mode)
                for model in method['models'] for corpus in method['corpora'] for mode in method['modes']}
    actual = [(r['model'], r['corpus'], r['mode']) for r in complete['runs']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        issues.append(dict(scope='coverage', issue='scheduled export coverage differs'))
    rows, unique = [], {}
    preferred = 'single' if 'single' in method['modes'] else method['modes'][0]
    for run in complete['runs']:
        path = (root/run['fp32_comparison']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('comparison path leaves the artifact directory')
        comparison = json.loads(path.read_text())
        problems = categorical_issues(comparison)
        summary = metrics(comparison)
        rows.append(dict(model=run['model'], corpus=run['corpus'], mode=run['mode'], **summary,
                         fp32_decisions_and_contract_equal=not problems))
        issues.extend(dict(scope=run['model']+'/'+run['corpus']+'/'+run['mode'], issue=problem) for problem in problems)
        if run['mode'] == preferred:
            unique[(run['model'], run['corpus'])] = summary
    models = {}
    for model in method['models']:
        key = 'basal-omni-1.0-'+model
        selected = [value for (name, _), value in unique.items() if name == key]
        heads = sum(value['heads'] for value in selected)
        distance = sum(value['total_tv_cal'] for value in selected)
        models[model] = dict(unique_heads=heads, mean_tv_cal=distance/heads if heads else None)
    baselines = {}
    for item in a.bf16_comparison:
        name, separator, filename = item.partition('=')
        if not separator or name not in models or name in baselines:
            p.error('BF16 comparisons must have unique mini=FILE or max=FILE labels')
        comparison = json.loads(Path(filename).read_text())
        baseline = metrics(comparison)
        native = models[name]
        # The comparison's token/record coverage must also match FP32. BF16 categorical
        # differences are retained, since they are part of the baseline being evaluated.
        coverage_ok = not (comparison['missing'] or comparison['tokens']['mismatches']
                           or comparison['tokens'].get('known_deviations'))
        closer = (coverage_ok and baseline['heads'] == native['unique_heads'] and native['mean_tv_cal'] is not None
                  and math.isfinite(baseline['mean_tv_cal']) and native['mean_tv_cal'] < baseline['mean_tv_cal'])
        baselines[name] = dict(**baseline, native_closer_to_fp32=closer,
                               fp32_decision_and_contract_differences=categorical_issues(comparison))
        if not closer:
            issues.append(dict(scope=name, issue='same-corpus mean TV superiority over upstream BF16 not established'))
    if baselines and set(baselines) != set(models):
        issues.append(dict(scope='BF16 coverage', issue='baseline missing for a selected model'))
    result = dict(status='FAIL' if issues else 'PASS_FP32_AND_BETTER_TV_THAN_BF16' if baselines else 'PASS_FP32_DECISIONS_AND_BATCHING',
                  issues=issues, models=models, runs=rows, upstream_bf16=baselines,
                  binary_sha256=method['binary_sha256'],
                  method_sha256=hashlib.sha256((root/'method.json').read_bytes()).hexdigest(),
                  limitation='Finite frozen corpus; equal FP32 choices do not establish semantic correctness or full-model accuracy.')
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], issues=len(issues), models=models)))
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
