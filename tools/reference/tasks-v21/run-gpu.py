#!/usr/bin/env python3
"""Finite native A/B and upstream FP32/BF16 comparison. Run only inside an independently supervised GPU session.

Inputs are supplied measurement artifacts (primitives/ and tasks/ export inputs). No downloads, rentals or CI gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def rows(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--basal', type=Path, required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--native-library-path', help='CUDA libraries for native exports only; reference keeps its own environment')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--media', action='store_true')
    p.add_argument('--dtypes', nargs='+', choices=['auto','bf16','f16','f32'], default=['auto','bf16','f32'])
    p.add_argument('--half-gemm-table', type=Path, help='Fixed f16 table for native auto/f16 exports; BF16/F32 run without it')
    p.add_argument('--baseline-tasks', action='store_true', help='Baseline already supports tasks; compare the entire input corpus')
    p.add_argument('--baseline-f32', action='store_true')
    p.add_argument('--reference-python', type=Path)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--reference-cpu-threads', type=int, default=8)
    p.add_argument('--reference-dtypes', nargs='+', choices=['float32','bfloat16'], default=['float32','bfloat16'])
    p.add_argument('--fp32-device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--command-timeout', type=int, default=900)
    p.add_argument('--stage', choices=['native','reference','both'], default='both')
    a = p.parse_args()
    if a.reference_cpu_threads <= 0 or a.command_timeout <= 0:p.error('thread count and command timeout must be positive')
    if Path(a.run_name).name != a.run_name or a.run_name in ('.','..'): p.error('run-name must be a new directory name')
    root = a.measurement_artifacts.resolve(); out = root/'runs'/a.run_name; out.mkdir(parents=True,exist_ok=False)
    name = json.loads((a.model_dir/'basal.json').read_text())['name']
    primary = rows(root/'primitives/systemone.jsonl')
    tasks = rows(root/'tasks/systemone.jsonl'); task_ids = {r['id'] for r in primary}
    tasks = [r for r in tasks if r['id'] not in task_ids]
    def admitted(records):
        result=[]
        for r in records:
            if not a.media and r['request'].get('media'): continue
            r['request']['model'] = name; result.append(r)
        return result
    primary, tasks = admitted(primary), admitted(tasks)
    bench = (root/'primitives/bench.jsonl').read_bytes()
    for group, data in [('primitives',primary),('all',primary+tasks)]:
        dest=out/'inputs'/group;dest.mkdir(parents=True)
        (dest/'bench.jsonl').write_bytes(bench)
        (dest/'systemone.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    statuses=[]
    method={'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'baseline_tasks':a.baseline_tasks,'model':name,'media':a.media,'native_library_path_explicit':bool(a.native_library_path),'orders':1,'prefix_cache':False,'state_cache_mib':0,
            'candidate_sha256':hashlib.sha256(a.basal.read_bytes()).hexdigest(),
            'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest() if a.baseline else None,
            'half_gemm_table_sha256':hashlib.sha256(a.half_gemm_table.read_bytes()).hexdigest() if a.half_gemm_table else None,
            'bench_inputs_sha256':hashlib.sha256(bench).hexdigest(),'primitive_ids':[r['id'] for r in primary],
            'task_ids':[r['id'] for r in tasks],'dtypes':a.dtypes,'reference_cpu_threads':a.reference_cpu_threads,'reference_dtypes':a.reference_dtypes,'fp32_device':a.fp32_device,'command_timeout_s':a.command_timeout,'stage':a.stage,'runs':statuses,'status_scope':'execution only; numerical quality and task decisions require separate comparison'}
    def save(): (out/'method.json').write_text(json.dumps(method,indent=2)+'\n')
    def run(tag,argv,env=None):
        if env is None and a.native_library_path:
            env=os.environ|{'LD_LIBRARY_PATH':a.native_library_path+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else '')}
        with (out/(tag+'-private.log')).open('x') as log:
            try: code=subprocess.run([str(x) for x in argv],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.command_timeout).returncode
            except subprocess.TimeoutExpired: code=124
        statuses.append({'tag':tag,'exit_code':code});save();print(tag,code,flush=True)
        return code==0
    common=['export','--model',a.model_dir,'--orders','1','--no-prefix-cache','--batching','single','--upstream']
    if a.media: common+=['--media']
    half_table=['--gemm-table',a.half_gemm_table] if a.half_gemm_table else []
    if a.stage in ('native','both'):
        if a.baseline:
            run('baseline-auto',[a.baseline,*common,*half_table,'--dtype','auto','--inputs',out/('inputs/all' if a.baseline_tasks else 'inputs/primitives'),'--out',out/'baseline-auto'])
            if a.baseline_f32:run('baseline-f32',[a.baseline,*common,'--dtype','f32','--inputs',out/('inputs/all' if a.baseline_tasks else 'inputs/primitives'),'--out',out/'baseline-f32'])
        for dtype in a.dtypes:
            run('candidate-'+dtype,[a.basal,*common,*(half_table if dtype in ('auto','f16') else []),'--dtype',dtype,'--inputs',out/'inputs/all','--out',out/('candidate-'+dtype)])
        for dtype in ['auto']+(['f32'] if a.baseline_f32 else []):
            old,new=out/('baseline-'+dtype),out/('candidate-'+dtype)
            if old.is_dir() and new.is_dir() and all((d/f).is_file() for d in [old,new] for f in ['bench.jsonl','systemone.jsonl','manifest.json']):
                mismatch=[];counts={}
                for file,key in [('bench.jsonl','index'),('systemone.jsonl','id')]:
                    old_rows={json.loads(line)[key]:line for line in (old/file).read_bytes().splitlines()}
                    new_rows={json.loads(line)[key]:line for line in (new/file).read_bytes().splitlines()}
                    counts[file]=len(old_rows)
                    mismatch += [{'file':file,'id':key} for key,line in old_rows.items() if new_rows.get(key)!=line]
                (out/('bit-ab-'+dtype+'.json')).write_text(json.dumps({'status':'FAIL' if mismatch else 'PASS','rows':counts,'mismatches':mismatch},indent=2)+'\n')
    if a.stage in ('reference','both'):
        if not a.reference_python or not a.upstream:p.error('reference requires --reference-python and --upstream')
        exporter=Path(__file__).resolve().parents[1]/'export_reference_v2.py'
        env=os.environ|{'BASAL_UPSTREAM':str(a.upstream.resolve()),'OMP_NUM_THREADS':str(a.reference_cpu_threads),'MKL_NUM_THREADS':str(a.reference_cpu_threads)}
        for dtype in a.reference_dtypes:
            tag='upstream-'+dtype
            argv=[a.reference_python,exporter,'--model',a.model_dir,'--out',out/tag,'--dtype',dtype,'--orders','1',
                  '--questions',out/'inputs/all/bench.jsonl','--n','0','--cases',out/'inputs/all/systemone.jsonl','--exact-cases','--device',a.fp32_device if dtype=='float32' else 'cuda']
            if a.media:argv+=['--media']
            run(tag,argv,env)
    method['status']='completed' if statuses and all(r['exit_code']==0 for r in statuses) else 'partial';save()
    # Failures are preserved in method.json. A supervising session decides whether remaining safe stages can continue.
    if method['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
