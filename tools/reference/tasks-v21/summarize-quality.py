#!/usr/bin/env python3
"""Audit complete frozen exports against one FP32 reference. CPU-only analysis, no CI gate.

Retains raw CLI comparisons, categorical choices, task source spans, automation changes,
and mean TV. Agreement with FP32 is not a semantic accuracy benchmark.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def selected(a, keys):return {key:a[key] for key in keys if key in a}


def decision(a):
    kind=a['type'];base=selected(a,['type','experimental'])
    if kind=='classify':return base|selected(a,['label','top_label','labels'])
    if kind=='verify':return base|selected(a,['all_supported','not_supported'])|{'claims':{k:v['verdict'] for k,v in a['claims'].items()}}
    if kind=='compress':return base|selected(a,['compressed','kept','unit','original_chars','compressed_chars','ratio'])|{'units':[selected(u,['start','end']) for u in a['units']]}
    if kind=='extract':return base|{'data':a['data'],'fields':{k:selected(v,['value','text','start','end','source','calibration','error']) for k,v in a['fields'].items()}}
    base|=selected(a,['choice','legend','selected','action','answer','tool','tools','call','multi','deferred','refused','max_error','threshold','objective','constraints_applied','calibration'])
    if kind=='noul':base['positive']=a['noul']>=.5
    if a.get('ranking') is not None:base['ranking']=[selected(v,['tool','name']) for v in a['ranking']]
    if a.get('selection') is not None:base['selection']=decision(dict(a['selection'],type='multi'))
    return base


def automation(a):
    if a['type']=='classify':return a.get('automated')
    if a['type']=='verify':return {k:v.get('automated') for k,v in a['claims'].items()}
    return None


def answer(row):return row.get('answer_upstream',row.get('upstream_answer',row.get('answer')))


def records(directory, file, key):
    rows=[json.loads(x) for x in (directory/file).read_text().splitlines() if x.strip()]
    result={str(r[key]):r for r in rows}
    if not rows or len(result)!=len(rows):raise ValueError('empty corpus or duplicate record IDs')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts',type=Path,required=True)
    p.add_argument('--reference',required=True,help='Complete FP32 export relative to supplied root')
    p.add_argument('--candidate',action='append',required=True,help='LABEL=complete export relative to root')
    p.add_argument('--basal',type=Path,required=True,help='CLI binary for CPU-only basal compare')
    p.add_argument('--bf16-label',default='upstream-bf16')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.measurement_artifacts.resolve()
    def path(value):
        result=(root/value).resolve()
        if not result.is_relative_to(root) or not (result/'manifest.json').is_file():raise ValueError('complete export required within artifact root')
        return result
    reference=path(a.reference);ref_bench=records(reference,'bench.jsonl','index');ref_api=records(reference,'systemone.jsonl','id')
    a.out.mkdir(parents=True,exist_ok=False);variants={}
    for argument in a.candidate:
        label,sep,relative=argument.partition('=')
        if not sep or not label or len(label)>128 or any(not(c.isascii() and (c.isalnum() or c in '-_')) for c in label) or label in variants:p.error('unique simple LABEL=export required')
        candidate=path(relative);bench=records(candidate,'bench.jsonl','index');api=records(candidate,'systemone.jsonl','id')
        if set(bench)!=set(ref_bench) or set(api)!=set(ref_api):raise ValueError('reference/candidate coverage differs: '+label)
        comparison_path=a.out/(label+'-comparison.json')
        subprocess.run([str(a.basal),'compare','--a',str(reference),'--b',str(candidate),'--out',str(comparison_path)],check=True)
        comp=json.loads(comparison_path.read_text());sections=[comp['bench'],comp['systemone']['numeric_identical_tokens']]
        heads=sum(s['items'] for s in sections)
        tv=sum(s['items']*s['tv_cal']['mean'] for s in sections if s['items'])/heads
        if not heads or not math.isfinite(tv):raise ValueError('invalid numerical comparison')
        coverage=not (comp['missing'] or comp['tokens']['mismatches'] or comp['tokens'].get('known_deviations'))
        choices=all(s['argmax_agreement_cal']==f"{s['items']}/{s['items']}" for s in sections)
        changes=[];automation_changes=[];error_changes=[];questions=0
        for key,expected in ref_api.items():
            actual=api[key];left,right=answer(expected),answer(actual)
            ea=expected.get('upstream_error') or (left or {}).get('error')
            eb=actual.get('upstream_error') or (right or {}).get('error')
            if ea or eb:
                if bool(ea)!=bool(eb):error_changes.append({'id':key,'reference':ea,'candidate':eb})
                continue
            if not left or not right or set(left['answers'])!=set(right['answers']):raise ValueError('missing served answer')
            for name,la in left['answers'].items():
                rb=right['answers'][name];questions+=1
                if decision(la)!=decision(rb):changes.append({'id':key,'question':name,'reference':decision(la),'candidate':decision(rb)})
                if automation(la)!=automation(rb):automation_changes.append({'id':key,'question':name,'media':bool(expected['request'].get('media')),'reference':automation(la),'candidate':automation(rb)})
        variants[label]={'export':relative,'manifest_sha256':hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest(),
                         'comparison':comparison_path.name,'unique_first_round_heads':heads,'mean_tv_cal':tv,
                         'head_calibrated_argmax_agreement':[s['argmax_agreement_cal'] for s in sections],
                         'tokens_and_coverage_equal':coverage,'served_questions':questions,'served_decision_changes':changes,
                         'automation_changes':automation_changes,'error_changes':error_changes,'gold':comp['quality'],
                         'categorical_status':'PASS' if coverage and choices and not changes and not error_changes else 'FAIL'}
    baseline=variants.get(a.bf16_label)
    for label,value in variants.items():
        if label!=a.bf16_label and baseline:
            value['closer_to_fp32_than_upstream_bf16']=(value['tokens_and_coverage_equal'] and baseline['tokens_and_coverage_equal'] and value['unique_first_round_heads']==baseline['unique_first_round_heads'] and value['mean_tv_cal']<baseline['mean_tv_cal'])
    result={'status':'AUDITED','reference':a.reference,'reference_manifest_sha256':hashlib.sha256((reference/'manifest.json').read_bytes()).hexdigest(),
            'variants':variants,'scope':'First-round head TV/argmax and served categorical answers, source spans and parsed data. Automation reported separately; media automation policy may intentionally suppress text certificates. Continuous scores/probabilities need the retained numerical comparisons. Finite FP32 agreement does not establish semantic accuracy.'}
    (a.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['categorical_status','mean_tv_cal','unique_first_round_heads']} for k,v in variants.items()}))


if __name__=='__main__':main()
