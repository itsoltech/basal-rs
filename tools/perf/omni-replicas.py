"""Manual one-vs-two-GPU Omni replica measurements on frozen inputs.

No provisioning, GPU settings, upstream modifications or CI test gates. Run
under an external resource/deadline guard. Raw files contain process metadata
and local paths; sanitize them before publication. Numerical differences are
retained as FAIL, independently of HTTP/performance completion.
"""
import argparse
import collections
import contextlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from api_resources import ServerResources

spec = importlib.util.spec_from_file_location('omni_protocol', Path(__file__).with_name('omni-optimize.py'))
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


def gpu_rows():
    query = ['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw,'
             'temperature.gpu,clocks.current.sm', '--format=csv,noheader,nounits']
    rows = {}
    for line in subprocess.check_output(query, text=True, timeout=5).splitlines():
        index, uuid, *values = [part.strip() for part in line.split(',')]
        memory, util, power, temperature, sm = map(float, values)
        rows[uuid] = dict(device=int(index), device_vram_bytes=memory * (1 << 20),
                          gpu_util_percent=util, gpu_power_w=power,
                          gpu_temperature_c=temperature, gpu_sm_mhz=sm, process_vram_bytes=0)
    if sorted(row['device'] for row in rows.values()) != [0, 1]:
        raise RuntimeError('exactly two visible physical GPUs required')
    return rows


class ReplicaResources(ServerResources):
    """Keep CPU/RAM accounting identical to the existing single-GPU runner."""

    def __init__(self, pid, path, active_devices, interval):
        self.active_devices = active_devices
        super().__init__(pid, path, interval)

    def _gpu_resources(self, pids):
        devices = gpu_rows()
        apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
             '--format=csv,noheader,nounits'], text=True, timeout=5)
        seen = set()
        for line in apps.splitlines():
            uuid, pid, memory = [part.strip() for part in line.split(',')]
            pid = int(pid)
            if pid not in pids:
                raise RuntimeError('foreign GPU process during resource sampling')
            if (uuid, pid) in seen:
                raise RuntimeError('duplicate per-device process allocation')
            seen.add((uuid, pid))
            devices[uuid]['process_vram_bytes'] += float(memory) * (1 << 20)
        rows = sorted(devices.values(), key=lambda row: row['device'])
        active = [row for row in rows if row['device'] in self.active_devices]
        return dict(gpus=rows, active_devices=list(self.active_devices),
                    process_vram_bytes=sum(row['process_vram_bytes'] for row in rows),
                    device_vram_bytes=sum(row['device_vram_bytes'] for row in rows),
                    gpu_power_w=sum(row['gpu_power_w'] for row in rows),
                    gpu_util_percent=sum(row['gpu_util_percent'] for row in active) / len(active),
                    gpu_temperature_c=max(row['gpu_temperature_c'] for row in active),
                    gpu_sm_mhz=sum(row['gpu_sm_mhz'] for row in active) / len(active))


def corpus(root, model):
    manifest = json.loads((root / 'provenance.json').read_text())
    if model not in manifest['model_names']:
        raise ValueError('model absent from frozen input manifest')
    for name in ['benchmark-short.jsonl', 'text-ladder.jsonl']:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or protocol.digest(path) != manifest['files'][name]:
            raise ValueError('input checksum mismatch')
    short = [json.loads(line) for line in (root / 'benchmark-short.jsonl').read_text().splitlines()]
    ladder = [json.loads(line) for line in (root / 'text-ladder.jsonl').read_text().splitlines()]
    long = [row for row in ladder if row['model'] == model and row['target_tokens'] == 8192]
    if len(short) != 32 or len(long) != 2 or len({row['id'] for row in short + long}) != 34:
        raise ValueError('expected 32 distinct short tasks and two long documents')
    mixed = []
    for index, row in enumerate(short):
        mixed.append(row)
        if (index + 1) % 16 == 0:
            mixed.append(long[index // 16])
    return manifest, {'tasks': short, 'mixed': mixed}


def metric_values(text, name):
    values = {}
    for line in text.splitlines():
        if line.startswith(name + '{'):
            replica = re.search(r'replica="([^"]+)"', line)
            if replica is None or replica[1] in values:
                raise RuntimeError('missing or duplicate replica label')
            values[replica[1]] = float(line.rsplit(' ', 1)[1])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement-artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--expect-sha256', required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--model', default='basal-omni-1.0-ultra')
    parser.add_argument('--native-library-path', required=True)
    parser.add_argument('--clients', type=int, nargs='+', default=[1, 2, 8, 16, 32, 128])
    parser.add_argument('--workloads', nargs='+', choices=['tasks', 'mixed'], default=['tasks', 'mixed'])
    parser.add_argument('--minimum-seconds', type=float, default=30)
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('--sample-interval', type=float, default=1)
    parser.add_argument('--cpu-threads', type=int, default=16)
    parser.add_argument('--request-timeout', type=float, default=300)
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--batch-tokens', type=int, default=16384)
    parser.add_argument('--schedule', choices=['fifo', 'hrrn'], default='fifo')
    parser.add_argument('--long-tokens', type=int, default=0)
    parser.add_argument('--long-slice-ms', type=int, default=5000)
    parser.add_argument('--rounds', type=int, choices=[1, 2], default=1)
    parser.add_argument('--port', type=int, default=18765)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    if (args.clients != sorted(set(args.clients)) or not args.clients or min(args.clients) < 1
            or max(args.clients) > 128 or len(set(args.workloads)) != len(args.workloads)
            or not args.workloads or args.cpu_threads < 1 or args.batch_tokens < 1
            or args.long_tokens < 0 or args.long_slice_ms < 1 or not 1024 <= args.port <= 65535):
        parser.error('invalid bounded client, workload or server options')
    numbers = [args.minimum_seconds, args.idle_seconds, args.sample_interval,
               args.request_timeout, args.startup_timeout, args.deadline]
    if (not all(math.isfinite(value) and value > 0 for value in numbers)
            or args.minimum_seconds < 30 or args.idle_seconds < 5
            or not .5 <= args.sample_interval <= 5 or args.deadline <= time.time() + 60):
        parser.error('invalid duration, sample interval or deadline')
    if protocol.digest(args.binary) != args.expect_sha256:
        parser.error('binary checksum mismatch')
    manifest, workloads = corpus(args.measurement_artifacts, args.model)
    devices = gpu_rows()
    identity = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader,nounits'],
        text=True, timeout=5).splitlines()
    if len(identity) != 2 or identity[0].strip() != identity[1].strip():
        parser.error('two identical GPUs required')
    if len(devices) != 2:
        parser.error('two GPUs required')
    args.out.mkdir(parents=True, exist_ok=False)
    url = f'http://127.0.0.1:{args.port}'
    schedule = ['single0', 'replicas', 'replicas', 'single0']
    if args.rounds == 2:
        schedule += ['replicas', 'single0', 'single0', 'replicas']
    method = dict(protocol='omni-replicas-v1', model=args.model, inputs=manifest,
                  binary_sha256=args.expect_sha256, schedule=schedule, clients=args.clients,
                  workloads=args.workloads, minimum_seconds=args.minimum_seconds,
                  minimum_corpus_passes=4, minimum_requests_per_client=4,
                  idle_seconds=args.idle_seconds, sample_interval_s=args.sample_interval,
                  cpu_threads=args.cpu_threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  request_timeout_s=args.request_timeout, schedule_policy=args.schedule,
                  long_tokens=args.long_tokens, long_slice_ms=args.long_slice_ms,
                  batch_tokens_per_replica=args.batch_tokens, dtype='f16', inference_mode='auto',
                  gemm_table='auto', orders=1, state_cache=False, prefix_cache=False,
                  max_inflight=1024, max_large_inflight=1024, metrics=True,
                  active_devices={'single0': [0], 'single1': [1], 'replicas': [0, 1]},
                  preflight=['single0', 'single1'], excluded_repetitions=[],
                  resource_scope='Server process tree. CPU/RAM excludes generator/sampler. Per-device VRAM '
                  'and sum are both recorded. Power sum includes both installed cards, including inactive card. '
                  'GPU utilization/SM aggregate is the mean of active cards. Peaks are sampled. '
                  'Ascending client ladder retains allocator history; no allocator clearing.',
                  runner_sha256=protocol.digest(Path(__file__)),
                  sampler_sha256=protocol.digest(Path(__file__).with_name('api_resources.py')),
                  protocol_helper_sha256=protocol.digest(Path(__file__).with_name('omni-optimize.py')))
    (args.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    references, failures = {}, []
    differences = collections.Counter()
    interrupted = threading.Event()

    def on_signal(signum, frame):
        interrupted.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def remaining():
        value = args.deadline - time.time() - 20
        if value <= 0 or interrupted.is_set():
            raise TimeoutError('campaign deadline or interruption')
        return value

    def event(stage, **fields):
        row = dict(stage=stage, unix_s=time.time(), monotonic_s=time.monotonic(), **fields)
        with (args.out / 'progress.jsonl').open('a') as out:
            out.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def get(path):
        with urllib.request.urlopen(url + path, timeout=min(10, remaining())) as response:
            return response.read().decode()

    @contextlib.contextmanager
    def server(variant, tag):
        remaining()
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError('measurement port occupied')
        apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
        if any(Path('/proc', pid.strip()).exists() for pid in apps.splitlines() if pid.strip().isdigit()):
            raise RuntimeError('GPU compute process present before startup')
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('BASAL_') and key != 'CUDA_VISIBLE_DEVICES'}
        env.update({key: str(args.cpu_threads) for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']})
        env.update(BASAL_NO_UPDATE_CHECK='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   LD_LIBRARY_PATH=args.native_library_path, XDG_CACHE_HOME=str(args.out.resolve() / 'gemm-cache'))
        command = [str(args.binary), '--color', 'never', 'serve', '--model', str(args.model_dir),
                   '--dtype', 'f16', '--inference-mode', 'auto', '--gemm-table', 'auto',
                   '--orders', '1', '--state-cache-mb', '0', '--no-prefix-cache',
                   '--schedule', args.schedule, '--long-tokens', str(args.long_tokens),
                   '--long-slice-ms', str(args.long_slice_ms), '--max-batch-tokens', str(args.batch_tokens),
                   '--max-inflight', '1024', '--max-large-inflight', '1024', '--metrics',
                   '--addr', f'127.0.0.1:{args.port}']
        active = method['active_devices'][variant]
        command += ['--replica-devices', '0,1'] if variant == 'replicas' else ['--device', str(active[0])]
        event('server_start', tag=tag, variant=variant, command=command)
        sampler = None
        with (args.out / (tag + '-server.log')).open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                sampler = ReplicaResources(process.pid, args.out / (tag + '-resources.jsonl'), active, args.sample_interval)
                until = time.monotonic() + min(args.startup_timeout, remaining())
                while process.poll() is None and time.monotonic() < until:
                    try:
                        health = json.loads(get('/health'))
                        if health.get('status') == 'ok' and not health.get('degraded_models'):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError('server startup failed')
                discovery = json.loads(get('/v1/models'))
                (args.out / (tag + '-models.json')).write_text(json.dumps(discovery) + '\n')
                if [model.get('name') for model in discovery.get('models', [])] != [args.model]:
                    raise RuntimeError('discovery must advertise exactly one model name')
                event('server_ready', tag=tag, variant=variant)
                yield sampler
            finally:
                try:
                    if sampler is not None:
                        sampler.close()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    event('server_stopped', tag=tag, exit_code=process.returncode)
                    if process.returncode != 0:
                        failures.append(dict(tag=tag, error='unclean server exit', exit_code=process.returncode))

    def idle(sampler, tag, name):
        sampler.boundary(name, marker='begin')
        time.sleep(min(args.idle_seconds, remaining()))
        sample = sampler.boundary(name, marker='end')
        if name == 'loaded-idle':
            if sample['errors']:
                raise RuntimeError('loaded GPU placement could not be observed')
            for gpu in sample['gpus']:
                loaded = gpu['process_vram_bytes'] > 10 * (1 << 30)
                if loaded != (gpu['device'] in sampler.active_devices):
                    raise RuntimeError('observed Ultra weight placement differs from configured devices')
        event('idle_finished', tag=tag, name=name)

    def phase(sampler, tag, rows, clients, name, measured, establish=False):
        minimum = len(rows) * max(4, math.ceil(4 * clients / len(rows))) if measured else len(rows)
        lock, counter, stopping = threading.Lock(), 0, False
        before = get('/metrics')
        (args.out / (tag + '-' + name + '-metrics-before.txt')).write_text(before)
        begin = sampler.boundary(name, marker='begin', clients=clients)
        start = time.monotonic()
        driver_cpu_start = time.process_time()
        stop_at = start + (args.minimum_seconds if measured else 0)

        def worker(client):
            nonlocal counter, stopping
            while True:
                with lock:
                    if stopping:
                        return
                    if counter >= minimum and counter % len(rows) == 0 and time.monotonic() >= stop_at:
                        stopping = True
                        return
                    index = counter
                    counter += 1
                row = rows[index % len(rows)]
                started = time.monotonic()
                connection = None
                result = dict(index=index, client=client, id=row['id'],
                              workload_class='long' if 'target_tokens' in row else 'short',
                              started_monotonic_s=started)
                try:
                    connection = http.client.HTTPConnection('127.0.0.1', args.port,
                                                            timeout=min(args.request_timeout, remaining()))
                    body = dict(row['request'], model=args.model)
                    connection.request('POST', '/v1/basal', json.dumps(body, ensure_ascii=False).encode(),
                                       {'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    value = json.loads(response.read())
                    finished = time.monotonic()
                    valid = (response.status == 200 and protocol.valid_answers(value, body['questions'])
                             and protocol.finite(value) and value.get('model') == args.model
                             and not value.get('usage', {}).get('truncated'))
                    if 'target_tokens' in row:
                        valid = valid and value.get('usage', {}).get('input_tokens') == row['target_tokens']
                    result.update(http_status=response.status, response=value, valid=valid,
                                  finished_monotonic_s=finished, elapsed_ms=(finished - started) * 1000,
                                  scheduler={key: response.getheader(key) for key in
                                             ['x-basal-batch-requests', 'x-basal-queue-ms', 'x-basal-compute-ms']})
                    if valid:
                        actual = protocol.bits(value['answers'])
                        if establish:
                            if row['id'] in references:
                                raise RuntimeError('duplicate sequential reference')
                            references[row['id']] = actual
                        if row['id'] not in references:
                            raise RuntimeError('missing sequential reference')
                        result['reference_bitwise_equal'] = actual == references[row['id']]
                except (OSError, http.client.HTTPException, ValueError, RuntimeError) as error:
                    result.update(http_status=result.get('http_status'), valid=False, error=str(error),
                                  finished_monotonic_s=time.monotonic())
                finally:
                    if connection:
                        connection.close()
                with lock:
                    output.write(json.dumps(dict(phase=name, clients=clients, **result), ensure_ascii=False) + '\n')
                    output.flush()
                    counts['total'] += 1
                    counts['valid'] += bool(result['valid'])
                    counts['different'] += result.get('reference_bitwise_equal') is False
                    if not result['valid']:
                        stopping = True

        counts = collections.Counter()
        with (args.out / (tag + '-observations.jsonl')).open('a') as output:
            with ThreadPoolExecutor(max_workers=clients) as pool:
                list(pool.map(worker, range(clients)))
        end = time.monotonic()
        driver_cpu = time.process_time() - driver_cpu_start
        finish = sampler.boundary(name, marker='end', clients=clients)
        after = get('/metrics')
        (args.out / (tag + '-' + name + '-metrics-after.txt')).write_text(after)
        inflight = metric_values(after, 'basal_replica_inflight_requests')
        available = metric_values(after, 'basal_replica_available')
        routed_before = metric_values(before, 'basal_replica_routed_requests_total')
        routed_after = metric_values(after, 'basal_replica_routed_requests_total')
        routed = {key: value - routed_before.get(key, 0) for key, value in routed_after.items()}
        if not inflight or any(inflight.values()):
            raise RuntimeError('missing replica inflight or nonzero after drain')
        if (len(routed) != len(sampler.active_devices) or sum(routed.values()) != counts['total']
                or any(value <= 0 for value in routed.values())
                or len(available) != len(routed) or any(value != 1 for value in available.values())):
            raise RuntimeError('replica routing/availability counters disagree with completed phase')
        differences[tag] += counts['different']
        event('phase_finished', tag=tag, name=name, clients=clients, measured=measured,
              begin=start, end=end, resource_begin=begin['monotonic_s'], resource_end=finish['monotonic_s'],
              samples=counts['total'], invalid=counts['total'] - counts['valid'],
              answers_changed=counts['different'], minimum_samples=minimum,
              successful_requests_per_s=counts['valid'] / (end - start),
              driver_python_cpu_seconds=driver_cpu, driver_python_mean_cores=driver_cpu / (end - start),
              replica_inflight_after=inflight, replica_routed=routed, replica_available=available)
        if counts['valid'] != counts['total'] or counts['total'] < minimum:
            raise RuntimeError('HTTP/input failure or incomplete phase; raw observations retained')

    finished = False
    try:
        for variant in ['single0', 'single1']:
            tag = 'preflight-' + variant
            with server(variant, tag) as sampler:
                idle(sampler, tag, 'loaded-idle')
                phase(sampler, tag, workloads['mixed'], 1, 'reference-c1', False, establish=variant == 'single0')
                idle(sampler, tag, 'drained-idle')
        if differences['preflight-single1']:
            raise RuntimeError('GPU1 sequential answers differ from GPU0; investigate before scaling')
        for workload in args.workloads:
            counts = collections.Counter()
            for variant in schedule:
                counts[variant] += 1
                tag = f'{workload}-{variant}-r{counts[variant]}'
                with server(variant, tag) as sampler:
                    idle(sampler, tag, 'loaded-idle')
                    phase(sampler, tag, workloads['mixed'], 1, 'warmup', False)
                    for clients in args.clients:
                        phase(sampler, tag, workloads[workload], clients, f'{workload}-c{clients}', True)
                    idle(sampler, tag, 'drained-idle')
        finished = True
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        failures.append(dict(error=type(error).__name__ + ': ' + str(error)))
        event('run_failed', **failures[-1])
    finally:
        status = dict(execution_finished=finished, failures=failures, differences=dict(differences),
                      exact_answer_status='FAIL' if any(differences.values()) else 'PASS',
                      status='AUDIT_REQUIRED' if finished and not failures else 'FAIL')
        (args.out / 'complete.json').write_text(json.dumps(status, indent=2) + '\n')
    if not finished or failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
