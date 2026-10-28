#!/usr/bin/env python3
"""Run finite sequential calls through installed SDKs and direct HTTP on a live server."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='Contains inputs/ and receives runs/')
    p.add_argument('--model', required=True)
    p.add_argument('--media', action='store_true')
    p.add_argument('--run-name', required=True, help='New, unused relative directory under runs/')
    p.add_argument('--url', default='http://127.0.0.1:19439')
    p.add_argument('--python', required=True)
    p.add_argument('--typescript', required=True, help='Compiled .mjs runner installed alongside the packed npm SDK')
    p.add_argument('--dotnet', required=True, help='Compiled Live.dll')
    p.add_argument('--rust', required=True, help='Compiled live_basal2_omni example')
    a = p.parse_args()
    root = a.measurement_artifacts.resolve()
    if Path(a.run_name).name != a.run_name or a.run_name in ('.', '..'): p.error('run-name must be a directory name')
    out = root / 'runs' / a.run_name
    out.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in (root/'inputs/cases.jsonl').read_text().splitlines()]
    cases = [c for c in cases if a.media or 'media' not in c]
    method = {'model':a.model, 'media':a.media, 'error_checks':['unknown-model','empty-media'], 'case_ids':[c['id'] for c in cases],
              'required_sdks':['python-sync','python-async','python-schema','typescript-esm','typescript-cjs','csharp-map','rust-map'],
              'named_sdks':['typescript-esm-named','typescript-cjs-named','csharp-inline','csharp-prepared','csharp-explicit','rust-named'],
              'inputs_sha256':hashlib.sha256((root/'inputs/cases.jsonl').read_bytes()).hexdigest(),
              'comparison':'decoded answers, optional thinking/meta, usage and discovery presence; binary64 equality excluding latency/ms; sequential requests, caches disabled on server',
              'request_options':{c['id']:{k:c[k] for k in ('facts','think','think_text','early_exit') if k in c} for c in cases}}
    (out/'method.json').write_text(json.dumps(method,indent=2)+'\n')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def request(path, body=None):
        req = urllib.request.Request(a.url+path, data=None if body is None else json.dumps(body,ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=300) as response: return json.load(response)
    (out/'native-models.json').write_text(json.dumps(request('/v1/models'),ensure_ascii=False)+'\n')
    with (out/'native.jsonl').open('x') as stream:
        for c in cases:
            body = {k:v for k,v in c.items() if k not in ('id','media')}
            body['model'] = a.model
            if 'media' in c:
                body['media'] = []
                for m in c['media']:
                    data = (root/m['file']).read_bytes()
                    if hashlib.sha256(data).hexdigest()!=m['sha256']: raise ValueError('media checksum mismatch')
                    encoded = base64.b64encode(data).decode()
                    if m['encoding']=='data-url': encoded=f"data:{m['mime']};base64,{encoded}"
                    body['media'].append({'type':m['type'],'data':encoded})
            stream.write(json.dumps({'case':c['id'],'raw':request('/v1/basal',body)},ensure_ascii=False)+'\n');stream.flush()
            print('native',c['id'],flush=True)
    here=Path(__file__).resolve().parent
    common=['--measurement-artifacts',str(root)]
    commands=[(f'python-{mode}',[a.python,str(here/'python.py'),*common,'--mode',mode]) for mode in ('sync','async','schema')]
    commands += [('typescript-esm',['node',a.typescript,*common]),('typescript-cjs',['node',a.typescript,*common,'--cjs']),
                 ('csharp-map',['dotnet',a.dotnet,*common]),('rust-map',[a.rust,*common])]
    env=os.environ|{'BASAL_MODEL':a.model,'BASAL_MEDIA':'1' if a.media else '0','BASAL_URL':a.url,'DOTNET_ROLL_FORWARD':'Major'}
    env.pop('PYTHONPATH',None)
    codes={}
    for name,cmd in commands:
        with (out/f'sdk-{name}.jsonl').open('x') as stdout, (out/f'{name}-stderr.log').open('x') as stderr:
            result=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,timeout=900)
        codes[name]=result.returncode
        print(name,result.returncode,flush=True)
    (out/'processes.json').write_text(json.dumps(codes,indent=2)+'\n')
    result=subprocess.run([sys.executable,str(here/'audit.py'), '--measurement-artifacts',str(out)],capture_output=True,text=True)
    (out/'audit.json').write_text(result.stdout)
    if result.stderr: (out/'audit-stderr.log').write_text(result.stderr)
    print(result.stdout,flush=True)
    if result.returncode or any(codes.values()): raise SystemExit(1)


if __name__=='__main__': main()
