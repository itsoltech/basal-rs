#!/usr/bin/env python3
"""Export deterministic task expansion/assembly observations from an explicitly supplied pinned upstream checkout."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--measurement-artifacts',type=Path,required=True)
a=p.parse_args()
# Loading the two source modules directly avoids importing the GPU engine; no source modification or mock server.
package=types.ModuleType('basal');package.__path__=[str(a.upstream/'basal')];sys.modules['basal']=package
for name in ['prompt','tasks']:
 spec=importlib.util.spec_from_file_location('basal.'+name,a.upstream/'basal'/f'{name}.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
tasks=sys.modules['basal.tasks']
for line in (a.measurement_artifacts/'host-inputs.jsonl').read_text().splitlines():
 item=json.loads(line)
 try:
  body,plans=tasks.expand(item['request'],255)
  second=tasks.second_round(item['first'],item['request'],plans,255)
  response=tasks.finish(item['first'],item['request'],plans,item.get('calibration',{}),item.get('second',{}).get('answers'))
  result={'id':item['id'],'expanded':body,'second_request':second,'response':response}
 except (ValueError,TypeError,KeyError) as error:result={'id':item['id'],'error':str(error)}
 print(json.dumps(result,ensure_ascii=False,allow_nan=False))
