"""Manual Gemma text/image layer-placement verification and resource measurements.

Run under an external deadline/resource guard. This tool never rents hardware,
changes GPU settings or publishes results. Strict answer differences remain FAIL.
Media files are restored separately from checksum-pinned public source manifests.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import contextlib
import http.client
import hashlib
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

spec = importlib.util.spec_from_file_location('omni_replica_protocol', Path(__file__).with_name('omni-replicas.py'))
replicas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replicas)
protocol = replicas.protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--measurement-artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--baseline-sha256')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--native-library-path', required=True)
    p.add_argument('--native-loader', type=Path, help='optional private ELF loader for an older host libc')
    p.add_argument('--native-runtime-library-path', help='library path of the explicitly selected ELF loader')
    p.add_argument('--placements', nargs='+', default=['single', '0:30,1:30', '1:30,0:30'])
    p.add_argument('--clients', type=int, nargs='+', default=[1, 2, 8])
    p.add_argument('--load-repeats', type=int, default=2)
    p.add_argument('--port', type=int, default=18766)
    p.add_argument('--long-tokens', type=int, default=0)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= time.time() + 60:
        p.error("finite future deadline required")
    if bool(args.native_loader) != bool(args.native_runtime_library_path):
        p.error('private loader and its library path must be supplied together')
    if args.clients != sorted(set(args.clients)) or not args.clients or min(args.clients) != 1 or max(args.clients) > 128:
        p.error('client ladder must start at one, be unique, ascending and at most 128')
    if args.load_repeats < 1 or args.long_tokens < 0 or not 1024 <= args.port <= 65535:
        p.error('invalid load, lane or port bounds')
    if protocol.digest(args.candidate) != args.candidate_sha256:
        p.error('candidate checksum mismatch')
    if args.baseline and (not args.baseline_sha256 or protocol.digest(args.baseline) != args.baseline_sha256):
        p.error('baseline checksum mismatch')
    manifest, corpora = replicas.corpus(args.measurement_artifacts, 'basal-omni-1.0-ultra')
    media_manifest = json.loads((args.measurement_artifacts / 'image-sources.json').read_text())
    images = []
    for entry in media_manifest:
        image = args.measurement_artifacts / 'images' / (entry['id'] + '.jpg')
        if protocol.digest(image) != entry['sha256']:
            p.error('image checksum mismatch')
        images.append(dict(id='image-' + entry['id'], request=dict(
            state='Inspect the supplied image.', lang='en', detail='default',
            media=[dict(type='image', data=base64.b64encode(image.read_bytes()).decode())],
            questions=dict(category=dict(type='classify', labels=dict(cats='Cats on furniture', living='A room with tables and chairs',
                bear='A bear outdoors', bedroom='A bedroom')),
                claims=dict(type='verify', claims=['The image contains an animal.', 'The image shows an indoor room.']),
                fields=dict(type='extract', schema=dict(type='object', properties=dict(scene=dict(enum=['cats', 'living room', 'bear', 'bedroom']))))))))
    paired = json.loads(json.dumps(images[0]))
    paired['id'] = 'image-pair-high'
    paired['request']['state'] = 'Inspect the supplied images.'
    paired['request']['detail'] = 'high'
    paired['request']['media'].append(images[1]['request']['media'][0])
    cases = corpora['mixed'] + images + [paired]
    args.out.mkdir(parents=True, exist_ok=False)
    variants = ([('baseline-single', args.baseline, 'single')] if args.baseline else [])
    variants += [('candidate-' + placement.replace(':', '_').replace(',', '-'), args.candidate, placement)
                 for placement in args.placements]
    method = dict(protocol='omni-layer-split-v1', model='basal-omni-1.0-ultra',
                  candidate_sha256=args.candidate_sha256, baseline_sha256=args.baseline_sha256,
                  private_loader_sha256=protocol.digest(args.native_loader) if args.native_loader else None,
                  variants=[dict(tag=tag, placement=placement) for tag, _, placement in variants],
                  clients=args.clients, load_repeats=args.load_repeats, long_tokens=args.long_tokens,
                  inference_mode='auto', dtype='f16', media=True, orders=1, gemm_table='auto',
                  client_media_limit=4,
                  client_media_admission='At most four simultaneous image HTTP requests, matching the existing server limit. Client permit wait recorded separately; request latency starts after its acquisition. Phase throughput includes permit waits.',
                  input_provenance=manifest, image_sources=media_manifest,
                  control='First sequential response for each input. Full answer equality including floating point.',
                  resource_scope='Server process tree; per-GPU VRAM and summed VRAM. Sampled memory maxima, not allocation peaks.',
                  runner_sha256=protocol.digest(Path(__file__)))
    published_cases = []
    for row in cases:
        safe = json.loads(json.dumps(row))
        if safe['id'].startswith('image-'):
            for entry in safe['request']['media']:
                sha = hashlib.sha256(base64.b64decode(entry['data'])).hexdigest()
                if not any(item['sha256'] == sha for item in media_manifest):
                    raise RuntimeError('image absent from source manifest')
                entry['data'] = 'MEDIA_SHA256:' + sha
        published_cases.append(safe)
    (args.out / 'inputs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in published_cases))
    method['inputs_sha256'] = protocol.digest(args.out / 'inputs.jsonl')
    (args.out / 'method.json').write_text(json.dumps(method, indent=2) + '\n')
    references, differences, failures = {}, {}, []
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def remaining():
        if interrupted or time.time() > args.deadline - 20:
            raise TimeoutError('measurement deadline or interruption')
        return args.deadline - time.time() - 20

    def event(stage, **fields):
        row = dict(stage=stage, unix_s=time.time(), monotonic_s=time.monotonic(), **fields)
        with (args.out / 'progress.jsonl').open('a') as out:
            out.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    @contextlib.contextmanager
    def server(tag, binary, placement):
        remaining()
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError('measurement port occupied')
        replicas.gpu_rows()  # Require exactly two visible physical GPUs.
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True, timeout=5)
        if any(x.strip().isdigit() for x in apps.splitlines()):
            raise RuntimeError('foreign GPU process present')
        active = [0] if placement == 'single' else [int(pair.split(':')[0]) for pair in placement.split(',')]
        env = {k: v for k, v in os.environ.items() if not k.startswith('BASAL_') and k != 'CUDA_VISIBLE_DEVICES'}
        env.update(LD_LIBRARY_PATH=args.native_library_path, BASAL_NO_UPDATE_CHECK='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', RAYON_NUM_THREADS='16',
                   XDG_CACHE_HOME=str(args.out.resolve() / 'gemm-cache'))
        command = [str(binary), '--color', 'never', 'serve', '--model', str(args.model_dir), '--media',
                   '--dtype', 'f16', '--inference-mode', 'auto', '--gemm-table', 'auto', '--orders', '1',
                   '--state-cache-mb', '0', '--no-prefix-cache', '--schedule', 'fifo',
                   '--long-tokens', str(args.long_tokens), '--long-slice-ms', '100', '--max-batch-tokens', '16384',
                   '--max-inflight', '1024', '--max-large-inflight', '1024', '--metrics',
                   '--addr', f'127.0.0.1:{args.port}']
        command += ['--device', '0'] if placement == 'single' else ['--layer-split', placement]
        if args.native_loader:
            command = [str(args.native_loader), '--library-path', args.native_runtime_library_path] + command
        event('server_start', tag=tag, command=command)
        sampler = None
        with (args.out / (tag + '-server.log')).open('x') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                sampler = replicas.ReplicaResources(process.pid, args.out / (tag + '-resources.jsonl'), active, 1)
                ready = time.monotonic() + min(300, remaining())
                while process.poll() is None and time.monotonic() < ready:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                            health = json.load(response)
                        if health.get('status') == 'ok' and not health.get('degraded_models'):
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError('server startup failed: ' + tag)
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
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=10)
                    event('server_stopped', tag=tag, exit_code=process.returncode)
                    if process.returncode:
                        failures.append(dict(tag=tag, error='unclean server exit', exit_code=process.returncode))

    media_clients = threading.BoundedSemaphore(4)

    def one(row, clients, index):
        body = dict(row['request'], model='basal-omni-1.0-ultra')
        queued = time.monotonic()
        media_permit = row['id'].startswith('image-')
        if media_permit and not media_clients.acquire(timeout=min(300, remaining())):
            raise TimeoutError('client media admission wait')
        started = time.monotonic()
        conn = None
        try:
            conn = http.client.HTTPConnection('127.0.0.1', args.port, timeout=min(300, remaining()))
            conn.request('POST', '/v1/basal', json.dumps(body).encode(), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            value = json.loads(response.read())
            end = time.monotonic()
            valid = response.status == 200 and protocol.valid_answers(value, body['questions']) and protocol.finite(value)
            valid = valid and value.get('model') == body['model'] and not value.get('usage', {}).get('truncated')
            if 'target_tokens' in row:
                valid = valid and value.get('usage', {}).get('input_tokens') == row['target_tokens']
            if row['id'].startswith('image-'):
                usage = value.get('usage', {}).get('media', {})
                valid = valid and usage.get('images') == len(body['media']) and usage.get('media_tokens', 0) > 0
            return dict(index=index, id=row['id'], clients=clients, http_status=response.status,
                        response=value, valid=valid, elapsed_ms=(end-started)*1000,
                        client_media_wait_ms=(started-queued)*1000,
                        started_monotonic_s=started, finished_monotonic_s=end)
        except (OSError, ValueError, http.client.HTTPException) as error:
            return dict(index=index, id=row['id'], clients=clients, valid=False,
                        error=type(error).__name__ + ': ' + str(error),
                        started_monotonic_s=started, finished_monotonic_s=time.monotonic())
        finally:
            if conn:
                conn.close()
            if media_permit:
                media_clients.release()

    finished = False
    try:
        for variant_index, (tag, binary, placement) in enumerate(variants):
            with server(tag, binary, placement) as sampler:
                sampler.boundary('loaded-idle', marker='begin')
                time.sleep(min(5, remaining()))
                sampler.boundary('loaded-idle', marker='end')
                for clients in args.clients:
                    rows = cases if clients == 1 else cases * args.load_repeats
                    begin = sampler.boundary(f'c{clients}', marker='begin', clients=clients)
                    start = time.monotonic()
                    lock = threading.Lock()
                    def recorded(item):
                        result = one(item[1], clients, item[0])
                        with lock:
                            if result['valid'] and variant_index == 0 and clients == 1 and result['id'] not in references:
                                references[result['id']] = result['response']['answers']
                            result['reference_bitwise_equal'] = (result['valid'] and result['id'] in references and
                                protocol.bits(result['response']['answers']) == protocol.bits(references[result['id']]))
                            with (args.out / (tag + '-observations.jsonl')).open('a') as output:
                                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                        return result
                    with ThreadPoolExecutor(max_workers=clients) as pool:
                        results = list(pool.map(recorded, enumerate(rows)))
                    end = time.monotonic()
                    sample = sampler.boundary(f'c{clients}', marker='end', clients=clients)
                    changed = sum(result['valid'] and result['reference_bitwise_equal'] is False for result in results)
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/metrics', timeout=min(10, remaining())) as response:
                        metrics = response.read().decode()
                    (args.out / (tag + f'-c{clients}-metrics-after.txt')).write_text(metrics)
                    inflight = replicas.metric_values(metrics, 'basal_replica_inflight_requests')
                    available = replicas.metric_values(metrics, 'basal_replica_available')
                    if not inflight or any(inflight.values()) or not available or any(value != 1 for value in available.values()):
                        raise RuntimeError('missing/drained inflight or model availability check failed')
                    differences[tag + f'-c{clients}'] = changed
                    event('phase_finished', tag=tag, clients=clients, samples=len(results),
                          invalid=sum(not row['valid'] for row in results), answers_changed=changed,
                          begin=start, end=end, resource_begin=begin['monotonic_s'], resource_end=sample['monotonic_s'],
                          successful_requests_per_s=sum(row['valid'] for row in results)/(end-start))
                    if not all(row['valid'] for row in results):
                        raise RuntimeError('invalid responses; retained as FAIL')
                sampler.boundary('drained-idle', marker='begin')
                time.sleep(min(5, remaining()))
                sampler.boundary('drained-idle', marker='end')
        finished = True
    except (OSError, RuntimeError, TimeoutError, ValueError, http.client.HTTPException) as error:
        failures.append(dict(error=type(error).__name__ + ': ' + str(error)))
        event('run_failed', **failures[-1])
    finally:
        (args.out / 'complete.json').write_text(json.dumps(dict(execution_finished=finished,
            failures=failures, differences=differences,
            exact_answer_status='FAIL' if any(differences.values()) else 'PASS' if finished else 'GAP'), indent=2) + '\n')
    if not finished or failures:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
