#!/usr/bin/env python3
"""Restore two pinned assets from the earlier Omni campaign's reconstructed corpus."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--measurement-artifacts',type=Path,required=True,help='This campaign directory containing inputs/media-sources.json')
p.add_argument('--source-cases',type=Path,required=True,help='Reconstructed cases-real-media.jsonl from the pinned earlier campaign')
a=p.parse_args()
manifest=json.loads((a.measurement_artifacts/'inputs/media-sources.json').read_text())
source={row['id']:row for row in map(json.loads,a.source_cases.read_text().splitlines())}
for asset in manifest['assets']:
    media=source[asset['source_case']]['request']['media'][0]
    data=base64.b64decode(media['data'].split(',',1)[-1],validate=True)
    if hashlib.sha256(data).hexdigest()!=asset['sha256']: raise ValueError('asset checksum mismatch')
    target=a.measurement_artifacts/asset['file']
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(data)
    print(asset['file'],asset['sha256'])
