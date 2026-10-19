"""Linux server-tree resource observations, excluding the load generator.

No provisioning or GPU settings. Memory maxima are sampled, not allocation peaks.
An exited child may have CPU work after its last sample: keep that gap explicit.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def process_stat(pid):
    fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'parent': int(fields[1]), 'group': int(fields[2]),
            'start_ticks': int(fields[19]), 'cpu_ticks': int(fields[11]) + int(fields[12]),
            'state': fields[0]}


def process_memory(pid):
    values = {}
    for line in Path('/proc', str(pid), 'smaps_rollup').read_text().splitlines():
        key, _, rest = line.partition(':')
        if rest.strip().endswith('kB'):
            values[key] = int(rest.split()[0]) * 1024
    hwm = next(int(line.split()[1]) * 1024
               for line in Path('/proc', str(pid), 'status').read_text().splitlines()
               if line.startswith('VmHWM:'))
    return {'rss_bytes': values['Rss'], 'pss_bytes': values['Pss'],
            'uss_bytes': sum(values.get(k, 0) for k in ['Private_Clean', 'Private_Dirty', 'Private_Hugetlb']),
            'swap_bytes': values.get('Swap', 0), 'lifetime_peak_rss_bytes': hwm}


class ServerResources:
    def __init__(self, pid, path, interval=1.0):
        self.pid, self.path, self.interval = pid, Path(path), interval
        self.clock_ticks = os.sysconf('SC_CLK_TCK')
        self.identity = process_stat(pid)['start_ticks']
        self.scope = {'name': 'startup'}
        self.previous, self.active, self.cpu_total = {}, set(), 0.0
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.output = self.path.open('x')
        self.thread = threading.Thread(target=self._run, name='server-resource-sampler', daemon=True)
        self.thread.start()

    def _tree(self):
        processes = {}
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit():
                try:
                    row = process_stat(int(entry.name))
                    if row['state'] != 'Z':
                        processes[row['pid']] = row
                except (OSError, ValueError, IndexError):
                    pass  # Unrelated short-lived processes are not measurement gaps.
        root = processes.get(self.pid)
        if root is None or root['start_ticks'] != self.identity:
            raise RuntimeError('server root is gone or its PID was reused')
        owned = {self.pid}
        while True:
            more = {pid for pid, row in processes.items()
                    if row['parent'] in owned or row['group'] == self.pid}
            if more.issubset(owned):
                break
            owned |= more
        return [processes[pid] for pid in sorted(owned)]

    def _capture(self, kind):
        row = {'kind': kind, 'scope': dict(self.scope), 'unix_s': time.time(),
               'monotonic_s': time.monotonic(), 'errors': []}
        try:
            processes = self._tree()
            identities = {(p['pid'], p['start_ticks']) for p in processes}
            row['departed_processes'] = [list(key) for key in sorted(self.active - identities)]
            self.active = identities
            memory = {key: 0 for key in ['rss_bytes', 'pss_bytes', 'uss_bytes', 'swap_bytes']}
            row['processes'] = []
            for process in processes:
                key = (process['pid'], process['start_ticks'])
                current = process['cpu_ticks'] / self.clock_ticks
                previous = self.previous.get(key, 0.0)
                if current < previous:
                    raise RuntimeError('server CPU counter decreased')
                self.cpu_total += current - previous
                self.previous[key] = current
                usage = process_memory(process['pid'])
                for metric in memory:
                    memory[metric] += usage[metric]
                row['processes'].append(dict(process, **usage))
            row.update(memory, cpu_seconds=self.cpu_total)
            row['root_lifetime_peak_rss_bytes'] = next(
                p['lifetime_peak_rss_bytes'] for p in row['processes'] if p['pid'] == self.pid)
            row.update(self._gpu_resources({p['pid'] for p in processes}))
        except (OSError, ValueError, KeyError, StopIteration, RuntimeError, subprocess.SubprocessError) as error:
            row['errors'].append(type(error).__name__ + ': ' + str(error))
        row['sample_finished_monotonic_s'] = time.monotonic()
        self.output.write(json.dumps(row) + '\n')
        self.output.flush()
        return row

    def _gpu_resources(self, pids):
        commands = [
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm',
             '--format=csv,noheader,nounits']]
        app_lines, device_lines = [subprocess.check_output(cmd, text=True, timeout=5).strip().splitlines()
                                   for cmd in commands]
        if len(device_lines) != 1:
            raise RuntimeError('resource campaign requires exactly one visible GPU')
        gpu = {}
        for line in app_lines:
            pid, memory_mib = line.split(',')
            pid = int(pid)
            if pid in pids:
                if pid in gpu:
                    raise RuntimeError('duplicate GPU process entry')
                gpu[pid] = float(memory_mib) * (1 << 20)
        memory_mib, util, power, temperature, sm = map(float, device_lines[0].split(','))
        return dict(process_vram_bytes=sum(gpu.values()), device_vram_bytes=memory_mib * (1 << 20),
                    gpu_util_percent=util, gpu_power_w=power, gpu_temperature_c=temperature,
                    gpu_sm_mhz=sm)

    def _run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            with self.lock:
                self._capture('periodic')
            self.stop.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def boundary(self, name, **fields):
        with self.lock:
            self.scope = dict(name=name, **fields)
            return self._capture('boundary')

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('resource sampler did not stop')
        self.output.close()
