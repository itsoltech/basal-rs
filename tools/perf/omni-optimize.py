"""Compare Omni HTTP engines on frozen local inputs; screen is exploratory, confirm uses ABBA.

No provisioning, GPU-setting changes or publication. An external own-server guard is required.
Every request/response and startup failure is retained. Media use upstream eager because
upstream explicitly rejects fast+media. No failed startup is silently relabelled eager.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import struct
import subprocess
import time
import urllib.error
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bits(value):
    if isinstance(value, float):
        return ('float', struct.pack('!d', value).hex())
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bits(v) for v in value]
    return value


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return True


def valid_answers(value, questions):
    answers = value.get('answers')
    return (value.get('error') is None and isinstance(answers, dict) and bool(answers)
            and set(answers) == set(questions)
            and all(isinstance(answer, dict) and answer.get('error') is None
                    and answer.get('type') == questions[name]['type'] for name, answer in answers.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--native-library-path', help='CUDA libraries for native servers only')
    p.add_argument('--candidate', type=Path)
    p.add_argument('--candidate-gemm-table', help='auto, none, or invariant table path; supports {model} and {dtype}')
    p.add_argument('--cpu-threads', type=int, help='OMP, MKL and Rayon thread budget for both engines')
    p.add_argument('--candidate-cpu-threads', type=int, help='optional candidate override within the same external CPU quota')
    p.add_argument('--candidate-dtype', choices=['auto', 'bf16', 'f16', 'f32'], default='bf16')
    p.add_argument('--reference-python', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--models-root', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=['mini', 'max'], default=['mini', 'max'])
    p.add_argument('--model-names', nargs='+', help='Full local model directory names; overrides legacy --models suffixes')
    p.add_argument('--variants', nargs='+', choices=['baseline', 'candidate', 'upstream-fast', 'upstream-eager'],
                   default=['baseline', 'upstream-fast', 'upstream-eager'])
    p.add_argument('--surfaces', nargs='+', choices=['text', 'media'], default=['text', 'media'])
    p.add_argument('--phase', choices=['screen', 'confirm'], default='screen')
    p.add_argument('--max-text-tokens', type=int, default=2048)
    p.add_argument('--warmup-seconds', type=float, help='minimum initial text-engine warmup; confirm defaults to 10 seconds')
    p.add_argument('--native-long-tokens', type=int, default=0, help='native urgent/background lane threshold; 0 disables splitting')
    p.add_argument('--native-long-slice-ms', type=int, default=100, help='minimum background GPU slice before handing over to short requests')
    p.add_argument('--native-schedule', choices=['hrrn', 'fifo'], default='hrrn', help='native request scheduling policy')
    p.add_argument('--latency-samples', type=int, help='measured N=1 samples per ladder point and engine repetition')
    p.add_argument('--latency-warmup-samples', type=int, default=2, help='discarded N=1 requests before each ladder point')
    p.add_argument('--load-warmup-samples', type=int, help='short warmup requests, capped at one corpus pass; default: two complete passes')
    p.add_argument('--mixed', action='store_true', help='also interleave one 8192-token document per 16 short documents at N=8/32')
    p.add_argument('--load-samples', type=int, help='minimum requests per load phase, rounded up to complete corpus repetitions')
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--request-timeout', type=int, default=180)
    p.add_argument('--deadline', type=float, required=True, help='absolute Unix deadline')
    a = p.parse_args()
    if a.phase == 'confirm' and len(a.variants) != 2:
        p.error('confirmation requires exactly two engines; compare text and media separately')
    if len(set(a.variants)) != len(a.variants) or ('candidate' in a.variants and a.candidate is None):
        p.error('duplicate variants or missing candidate')
    if len(set(a.models)) != len(a.models) or len(set(a.surfaces)) != len(a.surfaces):
        p.error('models and surfaces must be unique')
    if not math.isfinite(a.deadline):
        p.error('deadline must be finite')
    if min(a.max_text_tokens, a.startup_timeout, a.request_timeout, a.native_long_slice_ms) <= 0 or a.native_long_tokens < 0:
        p.error('limits must be positive; the lane threshold can be zero')
    if a.cpu_threads is not None and a.cpu_threads <= 0:
        p.error('CPU thread count must be positive')
    if a.candidate_cpu_threads is not None and a.candidate_cpu_threads <= 0:
        p.error('candidate CPU thread count must be positive')
    if a.load_samples is not None and a.load_samples <= 0:
        p.error('load samples must be positive')
    if a.latency_samples is not None and a.latency_samples <= 0:
        p.error('latency samples must be positive')
    if a.latency_warmup_samples <= 0 or (a.load_warmup_samples is not None and a.load_warmup_samples <= 0):
        p.error('warmup sample counts must be positive')
    if a.warmup_seconds is not None and (not math.isfinite(a.warmup_seconds) or a.warmup_seconds < 0):
        p.error('warmup seconds must be finite and nonnegative')
    initial_warmup_s = a.warmup_seconds if a.warmup_seconds is not None else (10 if a.phase == 'confirm' else 0)
    selected_models = a.model_names or ['basal-omni-1.0-' + suffix for suffix in a.models]
    if len(set(selected_models)) != len(selected_models) or any(Path(m).name != m or m in ('.', '..') for m in selected_models):
        p.error('model names must be unique plain directory names')
    root = a.measurement_artifacts
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('input provenance mismatch: ' + name)
    upstream_commit = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(a.upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('upstream checkout must be clean')
    if upstream_commit != provenance['upstream_commit']:
        raise ValueError('upstream commit differs from frozen manifest')
    table_template = a.candidate_gemm_table if a.candidate_gemm_table not in [None, 'auto', 'none'] else None
    gemm_policy = 'explicit' if table_template else a.candidate_gemm_table or 'none'
    candidate_tables = ({model: Path(table_template.format(model=model, dtype=a.candidate_dtype)) for model in selected_models} if table_template else {})
    a.out.mkdir(parents=True, exist_ok=False)
    ladder = [json.loads(x) for x in (root / 'text-ladder.jsonl').read_text().splitlines()]
    media = [json.loads(x) for x in (root / 'cases-real-media.jsonl').read_text().splitlines()]
    short = [json.loads(x) for x in (root / 'benchmark-short.jsonl').read_text().splitlines()]
    if not ladder or not media or not short:
        raise ValueError('each required input corpus must be nonempty')
    for rows in [ladder, media, short]:
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('input corpus has duplicate IDs')
    for model in selected_models:
        if not any(row['model'] == model
                   and row['target_tokens'] <= a.max_text_tokens for row in ladder):
            raise ValueError('no ladder input in the requested model/token range')
    if a.mixed and any(not any(row['model'] == model and row['target_tokens'] == 8192
                               for row in ladder) for model in selected_models):
        raise ValueError('mixed phase needs frozen 8192-token documents for each selected model')
    schedule = (a.variants if a.phase == 'screen' else [a.variants[i] for i in [0, 1, 1, 0, 0, 1, 1, 0]])
    method = dict(phase=a.phase, schedule=schedule, models=a.models, model_names=selected_models, surfaces=a.surfaces, runner_sha256=digest(Path(__file__)),
                  candidate_dtype=a.candidate_dtype, native_library_path_explicit=bool(a.native_library_path), candidate_gemm_tables={m: digest(p) for m, p in candidate_tables.items()}, cpu_threads=a.cpu_threads, candidate_cpu_threads=a.candidate_cpu_threads, binaries={'baseline': digest(a.binary), **({'candidate': digest(a.candidate)} if a.candidate else {})},
                  candidate_gemm_policy=gemm_policy,
                  upstream_commit=upstream_commit, inputs=provenance, text_limit=a.max_text_tokens,
                  distinct_short_documents=len(short), requested_load_samples=a.load_samples, initial_warmup_s=initial_warmup_s,
                  latency_samples=a.latency_samples or (4 if a.phase == 'screen' else 10),
                  latency_warmup_samples=a.latency_warmup_samples,
                  load_warmup_samples=min(a.load_warmup_samples, len(short)) if a.load_warmup_samples else 2 * len(short),
                  media_load_repeats=max(8 if a.phase == 'confirm' else 2, math.ceil(4 / len(media))) if media else 0,
                  timing='before JSON serialization through complete HTTP body read; parse after timing',
                  excluded_repetitions=[1] if a.phase == 'confirm' else [],
                  media_engine='upstream-eager; fast rejects media',
                  orders=1, state_cache=False, prefix_cache=False,
                  native_long_tokens=a.native_long_tokens, native_long_slice_ms=a.native_long_slice_ms,
                  native_schedule=a.native_schedule, mixed=a.mixed,
                  native_media_admission=4, native_max_large_inflight=128,
                  task_accounting='Same HTTP requests; extraction continuation depends on model choices. Retain per-response tokens/branches; task token counts may differ between engines or upstream batches.',
                  native_timing_headers=['x-basal-batch-requests', 'x-basal-queue-ms', 'x-basal-compute-ms'],
                  startup_timeout_s=a.startup_timeout, request_timeout_s=a.request_timeout,
                  screen_is_signal=a.phase == 'screen')
    (a.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    events, seen, failed = [], {}, []
    env = os.environ | {'BASAL_NO_UPDATE_CHECK': '1', 'PYTHONPATH': str(a.upstream),
                        'TORCHINDUCTOR_CACHE_DIR': str(a.out.parent / 'upstream-compile-cache')}
    if a.cpu_threads is not None:
        env.update({key: str(a.cpu_threads) for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']})
    for key in list(env):
        if key.startswith('BASAL_') and key != 'BASAL_NO_UPDATE_CHECK':
            env.pop(key)

    def event(stage, **extra):
        row = dict(stage=stage, time=time.time(), **extra)
        events.append(row)
        with (a.out / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def remaining():
        left = a.deadline - time.time() - 20
        if left < 1:
            raise TimeoutError('campaign deadline')
        return left

    @contextlib.contextmanager
    def server(variant, model, with_media, tag):
        try:
            with socket.create_connection(('127.0.0.1', 18765), timeout=.5):
                raise RuntimeError('measurement port already occupied')
        except ConnectionRefusedError:
            pass
        if variant.startswith('upstream-'):
            cmd = [str(a.reference_python), '-m', 'basal.server', '--model', str(a.models_root / model),
                   '--mode', variant.removeprefix('upstream-'), '--dtype', 'bfloat16', '--device', 'cuda',
                   '--orders', '1', '--state-cache-gb', '0', '--context-overflow', 'refuse',
                   '--host', '127.0.0.1', '--port', '18765']
            if with_media:
                cmd += ['--media', 'on']
        else:
            cmd = [str(a.candidate if variant == 'candidate' else a.binary), '--color', 'never', 'serve',
                   '--model', str(a.models_root / model), '--dtype', a.candidate_dtype if variant == 'candidate' else 'bf16', '--orders', '1',
                   '--state-cache-mb', '0', '--no-prefix-cache', '--gemm-table', (str(candidate_tables[model]) if candidate_tables else gemm_policy) if variant == 'candidate' else 'none', '--long-tokens', str(a.native_long_tokens),
                   '--long-slice-ms', str(a.native_long_slice_ms),
                   '--schedule', a.native_schedule,
                   '--max-large-inflight', '128', '--addr', '127.0.0.1:18765']
            if with_media:
                cmd += ['--media']
        event('server_start', tag=tag, command=cmd)
        server_env = env
        if not variant.startswith('upstream-') and a.native_library_path:
            server_env = env | {'LD_LIBRARY_PATH': a.native_library_path + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')}
        if variant == 'candidate' and gemm_policy == 'auto':
            server_env = server_env | {'XDG_CACHE_HOME': str(a.out.resolve() / 'native-cache')}
        if variant == 'candidate' and a.candidate_cpu_threads is not None:
            server_env = server_env | {key: str(a.candidate_cpu_threads)
                                      for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']}
        with (a.out / (tag + '-server.log')).open('x') as log:
            process = subprocess.Popen(cmd, env=server_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                until = time.monotonic() + min(a.startup_timeout, remaining())
                while process.poll() is None and time.monotonic() < until:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18765/health', timeout=.5) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError('server did not become ready; exit=' + str(process.poll()))
                event('server_ready', tag=tag)
                if variant == 'candidate' and gemm_policy == 'auto':
                    tables = []
                    for path in sorted((a.out/'native-cache').rglob('*.json')):
                        try:
                            table = json.loads(path.read_text())
                        except (ValueError, UnicodeError):
                            continue
                        if isinstance(table, dict) and table.get('invariant') is True and table.get('model') == model:
                            tables.append(dict(file=path.relative_to(a.out).as_posix(), sha256=digest(path),
                                               dtype=table['dtype'], entries=len(table['entries']),
                                               gpu=table['gpu'], cublaslt_version=table['cublaslt_version']))
                    event('native_auto_gemm_tables', tag=tag, tables=tables)
                yield
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                event('server_stopped', tag=tag, exit_code=process.returncode)

    def phase(tag, variant, model, rows, clients, repeats, name):
        def one(i):
            row = rows[i % len(rows)]
            body = dict(row['request'], model=model)
            # One connection per closed-loop worker request, identically for all engines.
            conn = http.client.HTTPConnection('127.0.0.1', 18765, timeout=min(a.request_timeout, remaining()))
            started = time.perf_counter()
            try:
                data = json.dumps(body, ensure_ascii=False).encode()
                conn.request('POST', '/v1/basal', body=data, headers={'Content-Type': 'application/json'})
                response = conn.getresponse()
                raw = response.read()
                elapsed = (time.perf_counter() - started) * 1000
                scheduler = {name: response.getheader(name) for name in
                             ['x-basal-batch-requests', 'x-basal-queue-ms', 'x-basal-compute-ms']}
                value = json.loads(raw)
                valid = (response.status == 200 and valid_answers(value, body['questions'])
                         and finite(value) and not value.get('usage', {}).get('truncated'))
                if 'target_tokens' in row:
                    valid = valid and value.get('usage', {}).get('input_tokens') == row['target_tokens']
                return dict(id=row['id'], index=i, workload_class=row.get('class', 'tokens-' + str(row['target_tokens']) if 'target_tokens' in row else 'media'), http_status=response.status, elapsed_ms=elapsed,
                            valid=valid, scheduler=scheduler, response=value)
            except (OSError, http.client.HTTPException, ValueError) as error:
                return dict(id=row['id'], index=i, workload_class=row.get('class', 'tokens-' + str(row['target_tokens']) if 'target_tokens' in row else 'media'), http_status=None, valid=False, error=str(error))
            finally:
                conn.close()
        count = len(rows) * repeats
        wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            results = list(pool.map(one, range(count)))
        wall = time.perf_counter() - wall_start
        invalid, changes = 0, 0
        with (a.out / (tag + '-observations.jsonl')).open('a') as f:
            for row in results:
                key = (model, variant, row['id'])
                if row['valid']:
                    answer = bits(row['response']['answers'])
                    equal = key not in seen or seen[key] == answer
                    seen.setdefault(key, answer)
                    row['answer_bitwise_invariant'] = equal
                    changes += not equal
                invalid += not row['valid']
                f.write(json.dumps(dict(phase=name, clients=clients, **row), ensure_ascii=False) + '\n')
        latency = sorted(x['elapsed_ms'] for x in results if x['valid'])
        classes = {}
        for row in results:
            if row['valid']:
                classes.setdefault(row['workload_class'], []).append(row['elapsed_ms'])
        classes = {key: dict(samples=len(values), median_ms=statistics.median(values),
                            p95_ms=sorted(values)[min(len(values)-1, int(.95*len(values)))])
                   for key, values in classes.items()}
        event('phase_finished', tag=tag, phase=name, clients=clients, samples=count, invalid=invalid, classes=classes,
              ids=[row['id'] for row in rows],
              answers_changed=changes, successful_requests_per_s=len(latency) / wall, wall_s=wall,
              median_ms=statistics.median(latency) if latency else None,
              p95_ms=latency[min(len(latency)-1, int(.95*len(latency)))] if latency else None)
        if invalid or (changes and not variant.startswith('upstream-')):
            raise RuntimeError('invalid response or native bitwise invariance failure')

    for model in selected_models:
        for surface in a.surfaces:
            with_media = surface == 'media'
            counts = {}
            for variant in schedule:
                if with_media and variant == 'upstream-fast':
                    event('unsupported_combination', model=model, variant=variant, media=True,
                          reason='upstream requires --mode eager with --media on')
                    continue
                counts[variant] = counts.get(variant, 0) + 1
                tag = f'{model}-{"media" if with_media else "text"}-{variant}-r{counts[variant]}'
                try:
                    with server(variant, model, with_media, tag):
                        if not with_media:
                            warm_until = time.monotonic() + initial_warmup_s
                            while time.monotonic() < warm_until:
                                phase(tag, variant, model, short, 1, 1, 'engine-warmup')
                            for row in ladder:
                                if row['model'] == model and row['target_tokens'] <= a.max_text_tokens:
                                    phase(tag, variant, model, [row], 1, a.latency_warmup_samples, 'ladder-warmup')
                                    phase(tag, variant, model, [row], 1, a.latency_samples or (4 if a.phase == 'screen' else 10), 'ladder')
                            if a.load_warmup_samples is None:
                                phase(tag, variant, model, short, 1, 2, 'short-warmup')
                            else:
                                phase(tag, variant, model, short[:a.load_warmup_samples], 1, 1, 'short-warmup')
                            for clients in [1, 8, 32]:
                                phase(tag, variant, model, short, clients,
                                      max(1, math.ceil((a.load_samples or (64 if a.phase == 'screen' else 512)) / len(short))), 'short')
                            if a.mixed:
                                long_rows = [row for row in ladder if row['model'] == model and row['target_tokens'] == 8192]
                                if not long_rows:
                                    raise ValueError('mixed phase needs frozen 8192-token documents')
                                mixed = []
                                for index, row in enumerate(short):
                                    mixed.append(row)
                                    if (index + 1) % 16 == 0:
                                        mixed.append(long_rows[(index // 16) % len(long_rows)])
                                if len(short) < 16:
                                    mixed.append(long_rows[0])
                                for clients in [8, 32]:
                                    phase(tag, variant, model, mixed, clients, 1 if a.phase == 'screen' else 2, 'mixed')
                        else:
                            # All 34 preselected real-media payloads; no speed-based selection.
                            phase(tag, variant, model, media, 1, 1, 'media-warmup')
                            phase(tag, variant, model, media, 1, 1 if a.phase == 'screen' else 3, 'media-sequential')
                            phase(tag, variant, model, media, 4,
                                  max(8 if a.phase == 'confirm' else 2, math.ceil(4 / len(media))), 'media-load')
                except (RuntimeError, OSError, TimeoutError) as error:
                    failed.append(dict(tag=tag, error=str(error)))
                    event('run_failed', **failed[-1])
                    # The final report must retain this GAP/FAIL; other engine/model observations remain useful.
    (a.out / 'complete.json').write_text(json.dumps(dict(execution_finished=True, failures=failed,
         status='FAIL' if failed else 'SIGNAL' if a.phase == 'screen' else 'AUDIT_REQUIRED')) + '\n')
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
