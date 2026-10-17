"""Restore archived Omni inputs from separately supplied, SHA-256 verified media.

The public archive contains media-registry.json and curation.json. Assets must be
downloaded from the sources recorded in the input media-source-manifest.json.
Pinned FLEURS dev tarballs are accepted. When codec-replay.json is present, FFmpeg
can recreate missing codec variants, including the frozen Ogg serial/CRC fields.
There is no network access; temporary codec files are removed automatically.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import tempfile
import zlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ogg_serial(data, serial):
    """Rewrite only Ogg page serials and their standard, non-reflected CRC-32."""
    data = bytearray(data)
    table = []
    for value in range(256):
        crc = value << 24
        for _ in range(8):
            crc = ((crc << 1) ^ (0x04C11DB7 if crc & 0x80000000 else 0)) & 0xffffffff
        table.append(crc)
    start = 0
    while start < len(data):
        if data[start:start+4] != b'OggS' or start + 27 > len(data):
            raise ValueError('invalid Ogg page header')
        header = start + 27 + data[start+26]
        end = header + sum(data[start+27:header])
        if header > len(data) or end > len(data):
            raise ValueError('truncated Ogg page')
        struct.pack_into('<I', data, start+14, serial)
        data[start+22:start+26] = bytes(4)
        crc = 0
        for byte in data[start:end]:
            crc = ((crc << 8) ^ table[((crc >> 24) ^ byte) & 255]) & 0xffffffff
        struct.pack_into('<I', data, start+22, crc)
        start = end
    return bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='extracted archive root')
    p.add_argument('--input-subdir', type=Path, required=True, help='relative frozen input directory in the archive')
    p.add_argument('--assets', type=Path, required=True, help='original image/audio files or pinned FLEURS dev tarballs')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = a.measurement_artifacts.resolve()
    inputs = (root / a.input_subdir).resolve()
    if a.input_subdir.is_absolute() or not inputs.is_relative_to(root) or not inputs.is_dir():
        p.error('input-subdir must identify an input directory inside the extracted archive')
    registry = json.loads((root / 'media-registry.json').read_text())
    records = {x['path']: x for x in json.loads((root / 'curation.json').read_text())['files']}
    replay_path = root / 'codec-replay.json'
    replay = json.loads(replay_path.read_text()) if replay_path.exists() else {}
    payloads = {}
    required = set()
    files = []
    marker = re.compile(r'REDACTED_MEDIA_SHA256:[0-9a-f]{64}')
    for path in sorted(inputs.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            record = records[path.relative_to(root).as_posix()]
            if digest(data) != record['published_sha256']:
                raise ValueError('published input checksum mismatch: ' + path.name)
            text = data.decode()
            required.update(marker.findall(text))
            files.append((path.relative_to(inputs), text, record['private_original_sha256']))
    wanted = {registry[token]['raw_sha256'] for token in required}
    sources = replay.get('sources', {})
    variants = replay.get('variants', {})
    wanted.update(variants[key]['source_sha256'] for key in list(wanted) if key in variants)
    assets = {}
    # Historical quality cases also contain our generated red PNG and sine tone.
    # These small synthetic fixtures are archived losslessly; real media stay external.
    for key, fixture in replay.get('synthetic_assets', {}).items():
        if key not in wanted:
            continue
        size = fixture['bytes']
        if not isinstance(size, int) or not 0 < size <= 20 << 20:
            raise ValueError('invalid synthetic fixture size')
        decoder = zlib.decompressobj()
        data = decoder.decompress(bytes.fromhex(fixture['zlib_hex']), size + 1)
        if len(data) != size or not decoder.eof or decoder.unused_data or digest(data) != key:
            raise ValueError('synthetic fixture checksum or compressed size mismatch')
        assets[key] = data
    paths = sorted(path for path in a.assets.rglob('*') if path.is_file())
    for path in paths:
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.wav', '.flac', '.ogg', '.mp3', '.bin'] and path.stat().st_size <= 20 << 20:
            data = path.read_bytes()
            key = digest(data)
            if key in wanted:
                assets.setdefault(key, data)
    names = {entry['filename'] for key, entry in sources.items() if key in wanted}
    if wanted - assets.keys() and names:
        for path in paths:
            if path.name.endswith('.tar.gz'):
                # Read selected members only; never extract archive paths to the filesystem.
                with tarfile.open(path, 'r|gz') as archive:
                    for member in archive:
                        if member.isfile() and Path(member.name).name in names and member.size <= 20 << 20:
                            stream = archive.extractfile(member)
                            if stream is None:
                                raise ValueError('missing FLEURS archive member')
                            data = stream.read((20 << 20) + 1)
                            key = digest(data)
                            if key in wanted:
                                assets.setdefault(key, data)
    derived = 0
    for token in sorted(required):
        key = registry[token]['raw_sha256']
        if key in assets or key not in variants:
            continue
        variant = variants[key]
        source = assets.get(variant['source_sha256'])
        if source is None:
            raise ValueError('missing source WAV SHA-256: ' + variant['source_sha256'])
        if subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0] != replay['ffmpeg_version']:
            raise ValueError('exact codec replay requires the recorded FFmpeg version')
        if variant['format'] not in ['flac', 'ogg', 'mp3']:
            raise ValueError('unsupported replay codec')
        with tempfile.TemporaryDirectory(prefix='basal-codec-') as temp:
            src, dest = Path(temp)/'source.wav', Path(temp)/('encoded.' + variant['format'])
            src.write_bytes(source)
            subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(src), str(dest)], check=True, timeout=30)
            data = dest.read_bytes()
        if variant['format'] == 'ogg':
            data = ogg_serial(data, variant['ogg_serial'])
        if digest(data) != key:
            raise ValueError('codec replay checksum mismatch; use the recorded codec build or supply the exact asset')
        assets[key] = data
        derived += 1
    for token in sorted(required):
        entry = registry[token]
        data = assets.get(entry['raw_sha256'])
        if data is None:
            raise ValueError('missing media asset SHA-256: ' + entry['raw_sha256'])
        if digest(data) != entry['raw_sha256'] or len(data) != entry['bytes']:
            raise ValueError('media asset changed during restoration')
        encoded = entry['prefix'] + base64.b64encode(data).decode('ascii')
        if digest(encoded.encode()) != entry['encoded_sha256']:
            raise ValueError('media encoding differs from the frozen input')
        payloads[token] = encoded
    restored = []
    for rel, text, expected in files:
        data = marker.sub(lambda m: payloads[m[0]], text).encode()
        if digest(data) != expected:
            raise ValueError('restored original input checksum mismatch: ' + str(rel))
        restored.append((rel, data))
    # Validate all inputs before creating the destination. Existing directories are never overwritten.
    a.out.mkdir(parents=True, exist_ok=False)
    for rel, data in restored:
        dest = a.out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    provenance = json.loads((a.out / 'provenance.json').read_text())
    for rel, expected in provenance['files'].items():
        path = (a.out / rel).resolve()
        if not path.is_relative_to(a.out.resolve()) or digest(path.read_bytes()) != expected:
            raise ValueError('restored runner provenance mismatch: ' + rel)
    print(json.dumps(dict(status='PASS', files=len(restored), restored_media_payloads=len(payloads), replayed_codecs=derived)))


if __name__ == '__main__':
    main()
