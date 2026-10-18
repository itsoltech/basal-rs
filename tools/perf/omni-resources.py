"""Measure unchanged Omni API servers and their Linux process-tree resources.

Run under an external GPU/resource/deadline guard. No provisioning, publication,
upstream checkout mutation, allocator clearing or GPU-setting changes.
"""
import argparse
import concurrent.futures
import contextlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.request

from api_resources import ServerResources

spec = importlib.util.spec_from_file_location('omni_http', Path(__file__).with_name('omni-optimize.py'))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def corpus(root, model, workload):
    short = [json.loads(x) for x in (root / 'benchmark-short.jsonl').read_text().splitlines()]
    if workload == 'tasks':
        return short
    ladder = [json.loads(x) for x in (root / 'text-ladder.jsonl').read_text().splitlines()]
    longs = [row for row in ladder if row['model'] == model and row['target_tokens'] == 8192]
    if len(longs) != 2:
        raise ValueError('two frozen long documents required')
    mixed = []
    for i, row in enumerate(short):
        mixed.append(row)
        if (i + 1) % 16 == 0:
            mixed.append(longs[(i // 16) % len(longs)])
    return mixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--reference-python', type=Path, required=True)
    p.add_argument('--native-library-path', required=True)
    p.add_argument('--clients', type=int, nargs='+', default=[1, 2, 8, 16, 32, 128])
    p.add_argument('--workloads', nargs='+', choices=['tasks', 'mixed'], default=['tasks', 'mixed'])
    p.add_argument('--variants', nargs=2, choices=['candidate', 'upstream-eager'],
                   default=['candidate', 'upstream-eager'])
    p.add_argument('--phase', choices=['screen', 'confirm'], default='confirm')
    p.add_argument('--minimum-seconds', type=float, default=60)
    p.add_argument('--idle-seconds', type=float, default=15)
    p.add_argument('--sample-interval', type=float, default=1)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--request-timeout', type=float, default=300)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    if (not a.clients or min(a.clients) < 1 or max(a.clients) > 128 or len(set(a.clients)) != len(a.clients)
            or a.clients != sorted(a.clients) or len(set(a.variants)) != 2
            or len(set(a.workloads)) != len(a.workloads)):
        p.error('unique ascending clients in 1..128 and two distinct engines required')
    numbers = [a.minimum_seconds, a.idle_seconds, a.sample_interval, a.startup_timeout,
               a.request_timeout, a.deadline]
    if (not all(math.isfinite(x) and x > 0 for x in numbers) or a.minimum_seconds < 30
            or a.idle_seconds < 5 or not .5 <= a.sample_interval <= 5 or a.cpu_threads < 1):
        p.error('finite positive limits, minimum phase 30s, idle 5s, sample interval .5..5s required')
    root, model = a.measurement_artifacts, a.model_dir.name
    provenance = json.loads((root / 'provenance.json').read_text())
    # This campaign uses text only; the parent manifest also names unused media.
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        expected = provenance['files'][name]
        file = (root / name).resolve()
        if not file.is_relative_to(root.resolve()) or protocol.digest(file) != expected:
            raise ValueError('frozen input checksum mismatch')
    commit = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != provenance['upstream_commit'] or subprocess.check_output(
            ['git', '-C', str(a.upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('clean pinned upstream required')
    a.out.mkdir(parents=True, exist_ok=False)
    schedule = a.variants if a.phase == 'screen' else [a.variants[i] for i in [0, 1, 1, 0]]
    method = {'protocol': 'omni-process-resources-v1', 'phase': a.phase, 'schedule': schedule,
              'clients': a.clients, 'workloads': a.workloads, 'model': model, 'inputs': provenance,
              'candidate_sha256': protocol.digest(a.candidate), 'upstream_commit': commit,
              'runner_sha256': protocol.digest(Path(__file__)),
              'sampler_sha256': protocol.digest(Path(__file__).with_name('api_resources.py')),
              'http_helper_sha256': protocol.digest(Path(__file__).with_name('omni-optimize.py')),
              'logical_cpu_count': os.cpu_count(),
              'available_logical_cpus': len(os.sched_getaffinity(0)),
              'cpu_pinning': False, 'thread_env_is_hard_cpu_quota': False,
              'minimum_seconds': a.minimum_seconds, 'minimum_corpus_passes': 4,
              'minimum_requests_per_client': 4, 'sample_interval_s': a.sample_interval,
              'idle_seconds': a.idle_seconds, 'cpu_thread_env': a.cpu_threads,
              'native': {'dtype': 'auto', 'inference_mode': 'auto', 'gemm_table': 'auto',
                         'schedule': 'fifo', 'long_tokens': 0, 'batch_tokens': 16384,
                         'max_inflight': 1024, 'max_large_inflight': 1024},
              'upstream': {'mode': 'eager', 'dtype': 'bfloat16', 'batch_tokens': 4096},
              'orders': 1, 'state_cache': False, 'prefix_cache': False,
              'excluded_repetitions': [], 'request_timeout_s': a.request_timeout,
              'resource_scope': 'Server root and descendants only; client/sampler/guard excluded. '
                                'CPU is core-seconds, RSS may double-count shared pages; PSS does not. '
                                'VRAM is sampled process allocation plus exclusive-device usage. '
                                'No allocator cache clearing; ascending clients retain allocation history. '
                                'Loaded idle includes resident model/runtime; not pure HTTP overhead.'}
    (a.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    seen, failures = {}, []

    def event(stage, **fields):
        row = dict(stage=stage, unix_s=time.time(), monotonic_s=time.monotonic(), **fields)
        with (a.out / 'progress.jsonl').open('a') as out:
            out.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def remaining():
        seconds = a.deadline - time.time() - 20
        if seconds <= 0:
            raise TimeoutError('campaign deadline')
        return seconds

    @contextlib.contextmanager
    def server(variant, tag):
        with socket.socket() as port:
            if port.connect_ex(('127.0.0.1', 18765)) == 0:
                raise RuntimeError('measurement port occupied')
        env = os.environ | {key: str(a.cpu_threads) for key in
                            ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']}
        for key in list(env):
            if key.startswith('BASAL_'):
                env.pop(key)
        env.update(BASAL_NO_UPDATE_CHECK='1', PYTHONPATH=str(a.upstream),
                   PYTHONDONTWRITEBYTECODE='1', TORCHINDUCTOR_CACHE_DIR=str(a.out / 'compile-cache'))
        if variant == 'candidate':
            command = [str(a.candidate), '--color', 'never', 'serve', '--model', str(a.model_dir),
                       '--dtype', 'auto', '--inference-mode', 'auto', '--gemm-table', 'auto',
                       '--orders', '1', '--state-cache-mb', '0', '--no-prefix-cache',
                       '--schedule', 'fifo', '--long-tokens', '0', '--max-batch-tokens', '16384',
                       '--max-inflight', '1024', '--max-large-inflight', '1024',
                       '--addr', '127.0.0.1:18765']
            env.update(LD_LIBRARY_PATH=a.native_library_path,
                       XDG_CACHE_HOME=str(a.out.resolve() / 'native-cache'))
        else:
            launcher = ('import runpy;from basal.engine import EagerBackend;'
                        'EagerBackend.MAX_BATCH_TOKENS=4096;'
                        'runpy.run_module("basal.server",run_name="__main__")')
            command = [str(a.reference_python), '-c', launcher, '--model', str(a.model_dir),
                       '--mode', 'eager', '--dtype', 'bfloat16', '--device', 'cuda', '--orders', '1',
                       '--state-cache-gb', '0', '--context-overflow', 'refuse',
                       '--host', '127.0.0.1', '--port', '18765']
        active_gpu = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
        if any(Path('/proc', pid.strip()).exists() for pid in active_gpu.splitlines() if pid.strip().isdigit()):
            raise RuntimeError('GPU already has a compute process before server startup')
        event('server_start', tag=tag, command=command)
        sampler = None
        with (a.out / (tag + '-server.log')).open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                sampler = ServerResources(process.pid, a.out / (tag + '-resources.jsonl'), a.sample_interval)
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
                    raise RuntimeError('server failed startup')
                event('server_ready', tag=tag)
                yield sampler
            finally:
                try:
                    if sampler:
                        sampler.close()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    event('server_stopped', tag=tag, exit_code=process.returncode)

    def idle(sampler, tag, name):
        start = sampler.boundary(name, marker='begin')
        time.sleep(min(a.idle_seconds, remaining()))
        end = sampler.boundary(name, marker='end')
        event('idle_finished', tag=tag, name=name, begin=start['monotonic_s'], end=end['monotonic_s'])

    def phase(sampler, tag, variant, rows, clients, name, measured):
        minimum = len(rows) * max(4, math.ceil(4 * clients / len(rows))) if measured else len(rows)
        lock, index, stopping = threading.Lock(), 0, False
        sampler.boundary(name, marker='begin', clients=clients)
        started = time.monotonic()
        stop_at = started + (a.minimum_seconds if measured else 0)

        def worker():
            nonlocal index, stopping
            results = []
            while True:
                with lock:
                    if (index >= minimum and index % len(rows) == 0
                            and time.monotonic() >= stop_at):
                        stopping = True
                    if stopping:
                        return results
                    i = index
                    index += 1
                row = rows[i % len(rows)]
                request_start = time.monotonic()
                connection = None
                result = dict(index=i, id=row['id'], workload_class=row.get('class', 'text-ladder'),
                              started_monotonic_s=request_start)
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', 18765,
                                                            timeout=min(a.request_timeout, remaining()))
                    body = dict(row['request'], model=model)
                    connection.request('POST', '/v1/basal', json.dumps(body, ensure_ascii=False).encode(),
                                       {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    raw = response.read()
                    end = time.monotonic()
                    value = json.loads(raw)
                    valid = (response.status == 200 and protocol.valid_answers(value, body['questions'])
                             and protocol.finite(value) and not value.get('usage', {}).get('truncated'))
                    valid = valid and value.get('model') == model
                    if 'target_tokens' in row:
                        valid = valid and value.get('usage', {}).get('input_tokens') == row['target_tokens']
                    result.update(http_status=response.status, response=value, valid=valid,
                                  finished_monotonic_s=end, elapsed_ms=(end - request_start) * 1000)
                except (OSError, http.client.HTTPException, ValueError) as error:
                    result.update(http_status=None, valid=False, error=str(error),
                                  finished_monotonic_s=time.monotonic())
                finally:
                    if connection:
                        connection.close()
                results.append(result)
                if not result['valid']:
                    with lock:
                        stopping = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            results = [row for group in pool.map(lambda _: worker(), range(clients)) for row in group]
        finished = time.monotonic()
        sampler.boundary(name, marker='end', clients=clients)
        results.sort(key=lambda row: row['index'])
        invalid, changed = 0, 0
        with (a.out / (tag + '-observations.jsonl')).open('a') as out:
            for row in results:
                if row['valid']:
                    key = (variant, row['id'])
                    value = protocol.bits(row['response']['answers'])
                    changed += key in seen and seen[key] != value
                    seen.setdefault(key, value)
                invalid += not row['valid']
                out.write(json.dumps(dict(phase=name, clients=clients, **row), ensure_ascii=False) + '\n')
        event('phase_finished', tag=tag, name=name, clients=clients, measured=measured,
              begin=started, end=finished, samples=len(results), invalid=invalid,
              answers_changed=changed, minimum_samples=minimum,
              successful_requests_per_s=sum(row['valid'] for row in results) / (finished - started))
        if invalid or (changed and variant == 'candidate'):
            raise RuntimeError('HTTP failure or native answer changed; observations retained')

    for workload in a.workloads:
        rows = corpus(root, model, workload)
        counts = {}
        for variant in schedule:
            counts[variant] = counts.get(variant, 0) + 1
            tag = f'{workload}-{variant}-r{counts[variant]}'
            try:
                with server(variant, tag) as sampler:
                    idle(sampler, tag, 'loaded-idle')
                    phase(sampler, tag, variant, rows, 1, 'warmup', False)
                    idle(sampler, tag, 'warm-idle')
                    for clients in a.clients:
                        phase(sampler, tag, variant, rows, clients, f'load-c{clients}', True)
                        idle(sampler, tag, f'drained-idle-c{clients}')
            except (OSError, RuntimeError, TimeoutError) as error:
                failures.append(dict(tag=tag, error=str(error)))
                event('run_failed', **failures[-1])
    (a.out / 'complete.json').write_text(json.dumps({'execution_finished': True, 'failures': failures,
                                                   'status': 'FAIL' if failures else 'AUDIT_REQUIRED'}) + '\n')
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
