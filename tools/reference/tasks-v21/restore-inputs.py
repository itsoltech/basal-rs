#!/usr/bin/env python3
"""Restore one frozen input directory from public observations and separately supplied media.

No network access. Assets must match the hashes recorded in media-placeholders.json;
their pinned sources are recorded in inputs/prior-media-source-manifest.json.
Writes a new directory and verifies the original byte hashes before any inference.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='Campaign directory in the extracted archive')
    p.add_argument('--input-subdir', type=Path, required=True, help='Relative frozen input directory, e.g. ada/perf-inputs')
    p.add_argument('--assets', type=Path, required=True, help='Separately downloaded or previously reconstructed media files')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = a.measurement_artifacts.resolve()
    inputs = (root / a.input_subdir).resolve()
    if a.input_subdir.is_absolute() or not inputs.is_relative_to(root) or not inputs.is_dir():
        p.error('input-subdir must be a directory within measurement-artifacts')
    registry = json.loads((root / 'media-placeholders.json').read_text())
    source_map = {x['file']: x for x in json.loads((root / 'public-source-map.json').read_text())['files']}
    files = []
    wanted = set()
    for path in sorted(inputs.rglob('*')):
        if not path.is_file():
            continue
        data = path.read_bytes()
        record = source_map[path.relative_to(root).as_posix()]
        if digest(data) != record['public_sha256']:
            raise ValueError('public input checksum mismatch: ' + path.name)
        text = data.decode()
        tokens = [token for token in registry if token in text]
        wanted.update(registry[token]['sha256'] for token in tokens)
        files.append((path.relative_to(inputs), text, tokens, record['original_sha256']))
    payloads = {}
    for asset in sorted(a.assets.rglob('*')):
        if asset.is_file() and asset.stat().st_size <= 64 * 1024 * 1024:
            data = asset.read_bytes()
            sha = digest(data)
            if sha in wanted:
                payloads[sha] = data
    missing = wanted - payloads.keys()
    if missing:
        raise ValueError('missing checksum-verified media: ' + ', '.join(sorted(missing)))
    restored = []
    for relative, text, tokens, expected in files:
        for token in tokens:
            asset = registry[token]
            encoded = base64.b64encode(payloads[asset['sha256']]).decode()
            if asset['uri_prefix']:
                encoded = asset['uri_prefix'] + ',' + encoded
            text = text.replace(token, encoded)
        data = text.encode()
        if digest(data) != expected:
            raise ValueError('original input checksum mismatch: ' + relative.as_posix())
        restored.append((relative, data))
    a.out.mkdir(parents=True, exist_ok=False)
    for relative, data in restored:
        target = a.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(json.dumps({'status': 'PASS', 'files': len(restored), 'media_assets': len(wanted)}))


if __name__ == '__main__':
    main()
