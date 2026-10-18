#!/usr/bin/env python3
"""Freeze exact-token text ladders, distinct task documents and real media for omni-optimize.py.

Requires the pinned upstream Python environment and already downloaded immutable model directories.
No inference or provisioning. The supplied artifact root contains inputs/cases.jsonl and its media assets.
"""
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--models-root', type=Path, required=True)
    p.add_argument('--model-names', nargs='+', required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--image-only', action='store_true')
    p.add_argument('--tokens', nargs='+', type=int, default=[128,512,2048,8192])
    a = p.parse_args()
    if any(Path(m).name != m or m in ('.','..') for m in a.model_names) or len(set(a.model_names)) != len(a.model_names):
        p.error('models must be unique plain directory names')
    if not a.tokens or any(n < 128 or n > 32768 for n in a.tokens) or len(set(a.tokens)) != len(a.tokens):
        p.error('unique token lengths from 128 through 32768 required')
    if subprocess.check_output(['git','-C',str(a.upstream),'status','--porcelain'], text=True).strip():
        p.error('upstream checkout must be immutable')
    sys.path.insert(0, str(a.upstream.resolve()))
    from basal.prompt import model_format, render, max_options
    from basal.server import to_items
    from transformers import AutoTokenizer
    source = a.measurement_artifacts.resolve()
    cases = [json.loads(x) for x in (source/'inputs/cases.jsonl').read_text().splitlines() if x.strip()]
    a.out.mkdir(parents=True, exist_ok=False)
    ladder = []
    for model in a.model_names:
        md = a.models_root/model
        tok = AutoTokenizer.from_pretrained(md)
        fmt = model_format(md)
        for lang in ['en','pl']:
            state = 'The document says BLUE. Additional data:' if lang == 'en' else 'Dokument zawiera informację BLUE. Dodatkowe dane:'
            question = 'Which colour occurs in the document?' if lang == 'en' else 'Jaki kolor występuje w dokumencie?'
            options = ['BLUE','RED']
            for target in a.tokens:
                n = max(0, target-len(tok(render(tok,state,question,options,lang,fmt),add_special_tokens=False).input_ids))
                for _ in range(20):
                    text = state+' alpha'*n
                    count = len(tok(render(tok,text,question,options,lang,fmt),add_special_tokens=False).input_ids)
                    if count == target:break
                    n += target-count
                    if n < 0:raise ValueError('target shorter than prompt')
                else:raise ValueError('cannot construct exact token length')
                ladder.append({'id':f'{model}-{lang}-{target}','class':'text-ladder','model':model,'target_tokens':target,
                               'request':{'state':text,'lang':lang,'questions':{'q':{'type':'choice','instructions':question,'criteria':dict(zip(['blue','red'],options)),'option_keys':'hide'}}}})
    # Validate the final wire question through upstream's parser as well as the prompt renderer.
    # In particular, option_keys changes the text that appears next to each option code.
    tokenizers = {model: AutoTokenizer.from_pretrained(a.models_root/model) for model in a.model_names}
    for row in ladder:
        body = row['request']; model = row['model']; fmt = model_format(a.models_root/model)
        item = to_items(body['state'], body['questions'], max_options(fmt), force_lang=body['lang'])[0]
        prompt = render(tokenizers[model], item['state'], item['question'], item['options'], item['lang'], fmt)
        if len(tokenizers[model](prompt, add_special_tokens=False).input_ids) != row['target_tokens']:
            raise ValueError('final wire request differs from planned token length')
    text_cases = [x for x in cases if 'media' not in x and x['id'] != 'named-api']
    if not text_cases:raise ValueError('no task text cases')
    short = []
    for i in range(32):
        case = copy.deepcopy(text_cases[i % len(text_cases)])
        case.pop('id');case['state'] = f'Document identifier {i:04d}.\n'+case['state']
        short.append({'id':f'distinct-task-{i:03d}','class':'distinct-task-document','request':case})
    media = []
    for case in cases:
        if 'media' not in case or (a.image_only and any(m['type'] != 'image' for m in case['media'])):continue
        body = {k:copy.deepcopy(v) for k,v in case.items() if k not in ('id','media')}
        body['media'] = []
        for item in case['media']:
            path = (source/item['file']).resolve()
            if not path.is_relative_to(source):raise ValueError('media outside supplied artifacts')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:raise ValueError('media checksum mismatch')
            encoded = base64.b64encode(data).decode()
            if item['encoding'] == 'data-url':encoded = f'data:{item["mime"]};base64,{encoded}'
            body['media'].append({'type':item['type'],'data':encoded})
        media.append({'id':case['id'],'class':'task-media','request':body})
    if not media:raise ValueError('no supported real media cases')
    files = {}
    for name, rows in [('text-ladder.jsonl',ladder),('benchmark-short.jsonl',short),('cases-real-media.jsonl',media)]:
        path = a.out/name
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'upstream_commit':subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip(),
                'files':files,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'sdk_cases_sha256':hashlib.sha256((source/'inputs/cases.jsonl').read_bytes()).hexdigest(),
                'model_names':a.model_names,'exact_token_lengths':a.tokens,'distinct_short_documents':len(short),
                'method':'Pinned tokenizer and upstream prompt.render, one choice question; task load has 32 distinct states; media from checksum-verified supplied assets'}
    (a.out/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'PREPARED','ladder':len(ladder),'short':len(short),'media':len(media)}))


if __name__ == '__main__':main()
