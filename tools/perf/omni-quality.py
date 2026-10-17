"""Export Omni against frozen FP32 references and check exact equality across batching modes.

Inputs: provenance.json with a files mapping (relative path -> SHA-256), and
basal-omni-1.0-{mini,max}/CORPUS/{bench,systemone}.jsonl. Restore media payloads
before running; sanitized public references alone cannot execute media forwards.
No provisioning, GPU-setting changes or publication. Run under a resource guard.
Numerical distances and FP32 decisions require a separate quality audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits(value):
    if isinstance(value, float):
        return ('float64', struct.pack('!d', value).hex())
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bits(v) for v in value]
    return value


def exact_exports(left, right):
    result = []
    for filename, key in [('bench.jsonl', 'index'), ('systemone.jsonl', 'id')]:
        records = []
        for root in [left, right]:
            rows = [json.loads(line) for line in (root / filename).read_text().splitlines()]
            mapped = {str(row[key]): row for row in rows}
            if len(mapped) != len(rows):
                raise ValueError('duplicate exported record ID')
            records.append(mapped)
        a, b = records
        result.append(dict(file=filename, records=len(a), missing=sorted(a.keys() - b.keys()),
                           extra=sorted(b.keys() - a.keys()),
                           different=[key for key in sorted(a.keys() & b.keys()) if bits(a[key]) != bits(b[key])]))
    return dict(files=result, bitwise_equal=all(not (r['missing'] or r['extra'] or r['different']) for r in result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--models-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=['mini', 'max'], default=['mini', 'max'])
    p.add_argument('--corpora', nargs='+', default=['base', 'real-media'])
    p.add_argument('--modes', nargs='+', choices=['single', 'budget', 'tree'], default=['single', 'budget', 'tree'])
    p.add_argument('--gemm-table', help='invariant table path; supports {model} and {dtype}')
    p.add_argument('--dtype', choices=['auto', 'f16', 'bf16', 'f32'], required=True)
    p.add_argument('--inference-mode', choices=['auto', 'accurate'], default='auto')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    if a.timeout <= 0 or len(set(a.models)) != len(a.models) or len(set(a.modes)) != len(a.modes):
        p.error('timeout must be positive; model and mode lists must be unique')
    if len(set(a.corpora)) != len(a.corpora) or any(not c or Path(c).name != c or c in ['.', '..'] for c in a.corpora):
        p.error('corpora must be unique directory names')
    root = a.measurement_artifacts.resolve()
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('input provenance mismatch: ' + name)
    for suffix in a.models:
        for corpus in a.corpora:
            inputs = root / ('basal-omni-1.0-' + suffix) / corpus
            reference_records = 0
            for filename in ['bench.jsonl', 'systemone.jsonl']:
                if str((inputs / filename).relative_to(root)) not in provenance['files']:
                    raise ValueError('required reference is absent from input provenance')
                reference_records += len((inputs / filename).read_text().splitlines())
            if not reference_records:
                raise ValueError('reference corpus must be nonempty')
    tables = ({'basal-omni-1.0-' + suffix: Path(a.gemm_table.format(model='basal-omni-1.0-' + suffix, dtype=a.dtype)) for suffix in a.models} if a.gemm_table else {})
    a.out.mkdir(parents=True, exist_ok=False)
    method = dict(runner_sha256=digest(Path(__file__)), binary_sha256=digest(a.binary), dtype=a.dtype,
                  inference_mode=a.inference_mode,
                  gemm_tables={m: digest(p) for m, p in tables.items()}, models=a.models, corpora=a.corpora, modes=a.modes, inputs=provenance,
                  orders=1, media=True, state_cache=False, prefix_cache=False,
                  acceptance='Exact batching equality plus separately audited upstream FP32 quality; CLI compare exit alone is insufficient')
    (a.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    runs, invariance = [], []

    def run(cmd, log):
        with log.open('x') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=a.timeout)

    for suffix in a.models:
        model = 'basal-omni-1.0-' + suffix
        for corpus in a.corpora:
            inputs = root / model / corpus
            exports = []
            for mode in a.modes:
                tag = f'{model}-{corpus}-{mode}'
                dest = a.out / tag
                started = time.monotonic()
                run([str(a.binary), '--color', 'never', 'export', '--upstream', '--model', str(a.models_root / model),
                     '--dtype', a.dtype, '--inference-mode', a.inference_mode,
                     '--media', '--orders', '1', '--state-cache-mb', '0', '--no-prefix-cache',
                     *(['--gemm-table', str(tables[model])] if tables else []), '--batching', mode, '--inputs', str(inputs), '--out', str(dest)], a.out / (tag + '.log'))
                comparison = a.out / (tag + '-fp32.json')
                run([str(a.binary), 'compare', '--a', str(inputs), '--b', str(dest), '--out', str(comparison)],
                    a.out / (tag + '-fp32.log'))
                runs.append(dict(model=model, corpus=corpus, mode=mode, elapsed_s=time.monotonic() - started,
                                 fp32_comparison=comparison.name))
                if exports:
                    result = exact_exports(exports[0], dest)
                    invariance.append(dict(model=model, corpus=corpus, modes=[a.modes[0], mode], **result))
                exports.append(dest)
                (a.out / 'progress.json').write_text(json.dumps(dict(runs=runs, invariance=invariance), indent=2) + '\n')
                print(json.dumps(runs[-1]), flush=True)
    equal = len(a.modes) > 1 and all(x['bitwise_equal'] for x in invariance)
    (a.out / 'complete.json').write_text(json.dumps(dict(execution_finished=True, runs=runs, invariance=invariance,
        batching_bitwise_equal=equal, status='FP32_AUDIT_REQUIRED' if equal else 'BATCHING_GAP_OR_FAIL'), indent=2) + '\n')
    if not equal:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
