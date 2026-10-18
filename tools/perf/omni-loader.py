"""Compare unchanged inference through old/new Gemma loaders and observed startup.

No provisioning, GPU setting changes, page-cache clearing or upstream edits.
An external resource/deadline guard and rental cleanup are required.
"""
import argparse
import contextlib
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from api_resources import ServerResources, process_stat

spec = importlib.util.spec_from_file_location('protocol', Path(__file__).with_name('omni-optimize.py'))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def proc_counters(pid):
    """Root counters since birth, separate from aggregate server-tree CPU samples."""
    stat = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    ticks = os.sysconf('SC_CLK_TCK')
    io = {key: int(value) for key, value in
          (line.split(':') for line in Path('/proc', str(pid), 'io').read_text().splitlines())}
    return {'cpu_user_seconds': int(stat[11]) / ticks, 'cpu_system_seconds': int(stat[12]) / ticks,
            'minor_faults': int(stat[7]), 'major_faults': int(stat[9]), 'io': io,
            'scope': 'Server root only; counters since process birth, including loading.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True, help='frozen text inputs with provenance.json')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--dtype', choices=['auto', 'bf16', 'f16', 'f32'], default='auto')
    p.add_argument('--native-library-path', required=True)
    p.add_argument('--upstream', type=Path)
    p.add_argument('--reference-python', type=Path)
    p.add_argument('--idle-seconds', type=float, default=15)
    p.add_argument('--startup-timeout', type=float, default=300)
    p.add_argument('--request-timeout', type=float, default=300)
    p.add_argument('--cpu-threads', type=int, default=8)
    p.add_argument('--deadline', type=float, required=True, help='absolute Unix time, bounded externally')
    a = p.parse_args()
    if (not all(math.isfinite(x) and x > 0 for x in
                [a.idle_seconds, a.startup_timeout, a.request_timeout, a.deadline])
            or a.idle_seconds < 5 or a.cpu_threads < 1
            or bool(a.upstream) != bool(a.reference_python)):
        p.error('positive finite limits, idle >=5s and paired upstream/python paths required')
    provenance = json.loads((a.measurement_artifacts / 'provenance.json').read_text())
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        if protocol.digest(a.measurement_artifacts / name) != provenance['files'][name]:
            p.error('frozen input checksum mismatch')
    model = a.model_dir.name
    rows = [json.loads(x) for x in (a.measurement_artifacts / 'benchmark-short.jsonl').read_text().splitlines()]
    long = [json.loads(x) for x in (a.measurement_artifacts / 'text-ladder.jsonl').read_text().splitlines()]
    rows += [r for r in long if r['model'] == model and r['target_tokens'] == 8192]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        p.error('nonempty corpus with unique ids required')
    if a.upstream:
        commit = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != provenance['upstream_commit'] or subprocess.check_output(
                ['git', '-C', str(a.upstream), 'status', '--porcelain'], text=True).strip():
            p.error('clean pinned upstream required')
    a.out.mkdir(parents=True, exist_ok=False)
    schedule = ['baseline', 'candidate', 'candidate', 'baseline'] * 2
    if a.upstream:
        schedule += ['upstream-eager'] * 4
    method = {'protocol': 'gemma-loader-v1', 'schedule': schedule, 'excluded_repetitions': [1],
              'model': model, 'dtype': a.dtype, 'inputs': provenance,
              'baseline_sha256': protocol.digest(a.baseline), 'candidate_sha256': protocol.digest(a.candidate),
              'runner_sha256': protocol.digest(Path(__file__)),
              'sampler_sha256': protocol.digest(Path(__file__).with_name('api_resources.py')),
              'http_helper_sha256': protocol.digest(Path(__file__).with_name('omni-optimize.py')),
              'idle_seconds': a.idle_seconds, 'startup_timeout': a.startup_timeout,
              'request_timeout': a.request_timeout, 'cpu_thread_env': a.cpu_threads,
              'logical_cpu_count': os.cpu_count(), 'available_logical_cpus': len(os.sched_getaffinity(0)),
              'corpus_ids': [r['id'] for r in rows], 'inference_mode': 'auto', 'gemm_table': 'auto',
              'orders': 1, 'state_cache': False, 'prefix_cache': False, 'media': False,
              'native_schedule': 'fifo', 'native_batch_tokens': 16384, 'upstream_batch_tokens': 4096,
              'scope': 'One model; separate process per start. Full corpus C1 after loaded idle. '
                       'First start per engine retained but excluded from startup aggregate. '
                       'CPU/RAM/VRAM server-tree only. No OS/library/JIT cache clearing or CPU pinning. '
                       'HTTP local fresh TCP connection per request, no TLS/proxy. '
                       'Upstream follows native runs; startup comparison is descriptive, not randomized.'}
    (a.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    failures, counts, native_reference = [], {}, {}

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

    env = os.environ | {key: str(a.cpu_threads) for key in
                        ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']}
    for key in list(env):
        if key.startswith('BASAL_'):
            env.pop(key)
    env.update(BASAL_NO_UPDATE_CHECK='1', PYTHONDONTWRITEBYTECODE='1')
    for variant in schedule:
        counts[variant] = counts.get(variant, 0) + 1
        tag = f'{variant}-r{counts[variant]}'
        process, sampler, identity = None, None, None
        try:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', 18765)) == 0:
                    raise RuntimeError('measurement port occupied')
            active = subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
            if any(Path('/proc', pid.strip()).exists() for pid in active.splitlines() if pid.strip().isdigit()):
                raise RuntimeError('GPU already occupied')
            server_env = dict(env)
            if variant == 'upstream-eager':
                launcher = ('import runpy;from basal.engine import EagerBackend;'
                            'EagerBackend.MAX_BATCH_TOKENS=4096;'
                            'runpy.run_module("basal.server",run_name="__main__")')
                argv = [str(a.reference_python), '-c', launcher, '--model', str(a.model_dir), '--mode', 'eager',
                        '--dtype', 'bfloat16', '--device', 'cuda', '--orders', '1', '--state-cache-gb', '0',
                        '--context-overflow', 'refuse', '--host', '127.0.0.1', '--port', '18765']
                server_env['PYTHONPATH'] = str(a.upstream)
            else:
                binary = a.baseline if variant == 'baseline' else a.candidate
                argv = [str(binary), '--color', 'never', 'serve', '--model', str(a.model_dir), '--dtype', a.dtype,
                        '--inference-mode', 'auto', '--gemm-table', 'auto', '--orders', '1', '--state-cache-mb', '0',
                        '--no-prefix-cache', '--schedule', 'fifo', '--long-tokens', '0', '--max-batch-tokens', '16384',
                        '--addr', '127.0.0.1:18765']
                server_env.update(LD_LIBRARY_PATH=a.native_library_path,
                                  XDG_CACHE_HOME=str(a.out.resolve() / 'native-cache'))
            with (a.out / (tag + '-server.log')).open('x') as log:
                event('server_start', tag=tag, command_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest())
                started = time.monotonic()
                process = subprocess.Popen(argv, env=server_env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                identity = process_stat(process.pid)['start_ticks']
                sampler = ServerResources(process.pid, a.out / (tag + '-resources.jsonl'))
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
                wall = time.monotonic() - started
                loaded = sampler.boundary('loaded-idle', marker='begin')
                counters = proc_counters(process.pid)
                event('server_ready', tag=tag, startup_wall_s=wall, counters=counters,
                      resource_errors=loaded['errors'], aggregate_cpu_seconds=loaded.get('cpu_seconds'),
                      root_peak_rss_bytes=loaded.get('root_lifetime_peak_rss_bytes'))
                time.sleep(min(a.idle_seconds, remaining()))
                sampler.boundary('loaded-idle', marker='end')
                sampler.boundary('validation', marker='begin', clients=1)
                mismatches = 0
                with (a.out / (tag + '-observations.jsonl')).open('x') as out:
                    for source in rows:
                        began = time.monotonic()
                        connection = http.client.HTTPConnection('127.0.0.1', 18765,
                                                                timeout=min(a.request_timeout, remaining()))
                        try:
                            body = dict(source['request'], model=model)
                            connection.request('POST', '/v1/basal', json.dumps(body, ensure_ascii=False).encode(),
                                               {'Content-Type': 'application/json'})
                            response = connection.getresponse()
                            raw = response.read()
                            elapsed = time.monotonic() - began
                            value = json.loads(raw)
                        finally:
                            connection.close()
                        valid = (response.status == 200 and protocol.valid_answers(value, body['questions'])
                                 and protocol.finite(value) and not value.get('usage', {}).get('truncated')
                                 and value.get('model') == model)
                        if 'target_tokens' in source:
                            valid = valid and value.get('usage', {}).get('input_tokens') == source['target_tokens']
                        equal = None
                        if valid and variant != 'upstream-eager':
                            answer = protocol.bits(value['answers'])
                            native_reference.setdefault(source['id'], answer)
                            equal = answer == native_reference[source['id']]
                            mismatches += not equal
                        out.write(json.dumps({'id': source['id'], 'http_status': response.status, 'valid': valid,
                                              'elapsed_ms': elapsed * 1000, 'native_bits_equal': equal,
                                              'response': value}, ensure_ascii=False) + '\n')
                        out.flush()
                        if not valid:
                            raise RuntimeError('invalid HTTP validation response')
                sampler.boundary('validation', marker='end', clients=1)
                sampler.boundary('drained-idle', marker='begin')
                time.sleep(min(a.idle_seconds, remaining()))
                sampler.boundary('drained-idle', marker='end')
                event('validation_finished', tag=tag, requests=len(rows), native_bit_mismatches=mismatches)
                if mismatches:
                    raise RuntimeError('native answer differs bitwise')
        except (OSError, RuntimeError, ValueError, TimeoutError, http.client.HTTPException,
                subprocess.SubprocessError) as error:
            failures.append(dict(tag=tag, error=str(error)))
            event('run_failed', **failures[-1])
        finally:
            try:
                if sampler:
                    sampler.close()
            finally:
                if process and process.poll() is None:
                    if identity is not None and process_stat(process.pid)['start_ticks'] != identity:
                        raise RuntimeError('server PID identity changed')
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                event('server_stopped', tag=tag, exit_code=process.returncode if process else None)
    (a.out / 'complete.json').write_text(json.dumps({'status': 'FAIL' if failures else 'AUDIT_REQUIRED',
                                                   'execution_finished': True, 'failures': failures}) + '\n')
    if failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
