#!/usr/bin/env python3
"""Hold real CUDA device memory and sample the free memory of each GPU (capacity control of the layer-split
measurements). Not a test and not a CI gate: a measurement helper the layer-split runner starts and stops.

  python3 tools/perf/cuda-reserve.py --help
  python3 tools/perf/cuda-reserve.py query
  python3 tools/perf/cuda-reserve.py monitor --series S.csv --ready R.json --result X.json --max-seconds 900
  python3 tools/perf/cuda-reserve.py reserve --leave 0=8192 --leave 1=8192 --series S.csv --ready R.json \\
      --result X.json --max-seconds 900

Python standard library only: ctypes on the CUDA driver library libcuda.so.1 (loaded only by a mode, never by
--help). It touches no model, data or other process; it holds device memory of its own process.

Modes:
  query    the free and total memory of every CUDA device this process sees (JSON on stdout), exit 0.
  monitor  retains the primary context of each device (--device, default: every device), then samples
           cuMemGetInfo of every device every --interval-ms (50) into --series (CSV: t_unix, device, free_bytes,
           total_bytes; one line per device and sample, flushed per line) until SIGTERM, SIGINT or SIGHUP; the
           ready file appears once every context exists and the first sample of every device is written.
  reserve  as monitor for the devices of --leave DEVICE=MIB, but before the ready file it allocates real device
           memory (cuMemAlloc_v2 in chunks of at most --chunk-mib, each set with cuMemsetD8_v2, then
           cuCtxSynchronize) until the free memory the driver reports for that device is DEVICE's MIB within
           --tolerance-mib (64); at most --adjust rounds of correction (allocate the excess, or free the last
           chunk and allocate again). The ready file records per device the free memory before and after, the
           bytes held, the target, the deviation and whether it is within the tolerance (the caller judges it;
           out of tolerance the memory is still held until the stop). Then it samples like monitor.

Ending: SIGTERM, SIGINT or SIGHUP set a flag (no exception inside a driver call); then every allocation is freed
with its own context current (cuMemFree_v2), the primary contexts are released and --result is written.
--max-seconds (one absolute end, counted from the helper's start; finite and positive) ends it as well, also
without a signal, and so does a change of the parent process (--parent-pid: the caller died): the memory is freed
in both cases, but the exit status is 3, never 0. The three conditions are checked from the start on: before each
device's context is retained, before every allocation and correction of the reservation, before the ready file is
written (a stop before it leaves no ready file) and after every sampling round. A synchronous CUDA driver call
already in progress (an allocation with its memset and synchronisation, a free) cannot be interrupted by these
checks; the caller keeps its own bounded wait and stop of the helper's process group.

Exit status: 0 stopped by a signal with every allocation freed (also before the ready file); 1 a CUDA driver error
(its name in stderr and in --result; what was allocated is freed when possible), also a failed release after any
mode, query included; 2 usage; 3 ended by --max-seconds or a parent change.
The files hold no host name, address, PCI identifier or UUID: device ordinals, names and byte counts only.
"""
import argparse
import ctypes
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

MIB = 1 << 20
CUDA_SUCCESS = 0
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


class CudaError(Exception):
    """A CUresult other than CUDA_SUCCESS: the call and the driver's error name."""


class Driver:
    """The few CUDA driver calls this helper needs (cuda.h of CUDA 12: CUdevice is int, CUcontext a pointer,
    CUdeviceptr a 64-bit integer)."""

    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise CudaError(f"loading libcuda.so.1: {e}")
        lib = self.lib
        size_p, ptr_p = ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_uint64)
        sigs = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int],
            "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
            "cuCtxSetCurrent": [ctypes.c_void_p],
            "cuCtxSynchronize": [],
            "cuMemGetInfo_v2": [size_p, size_p],
            "cuMemAlloc_v2": [ptr_p, ctypes.c_size_t],
            "cuMemsetD8_v2": [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t],
            "cuMemFree_v2": [ctypes.c_uint64],
            "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
        }
        for name, args in sigs.items():
            try:
                fn = getattr(lib, name)
            except AttributeError:
                raise CudaError(f"libcuda.so.1 has no {name}")
            fn.argtypes, fn.restype = args, ctypes.c_int

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code != CUDA_SUCCESS:
            text = ctypes.c_char_p()
            got = self.lib.cuGetErrorName(code, ctypes.byref(text))
            label = text.value.decode() if got == CUDA_SUCCESS and text.value else f"CUresult {code}"
            raise CudaError(f"{name}: {label}")

    def init(self):
        self.call("cuInit", 0)
        n = ctypes.c_int()
        self.call("cuDeviceGetCount", ctypes.byref(n))
        return n.value

    def device(self, ordinal):
        d = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(d), ordinal)
        buf = ctypes.create_string_buffer(256)
        self.call("cuDeviceGetName", buf, len(buf), d.value)
        return d.value, buf.value.decode(errors="replace")

    def retain(self, dev):
        ctx = ctypes.c_void_p()
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
        return ctx

    def info(self, ctx):
        """(free, total) bytes of the device of `ctx` (made current first)."""
        self.call("cuCtxSetCurrent", ctx)
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.call("cuMemGetInfo_v2", ctypes.byref(free), ctypes.byref(total))
        return free.value, total.value

    def alloc(self, ctx, size):
        """`size` bytes on the device of `ctx`, every byte set (memset completed): the device pointer."""
        self.call("cuCtxSetCurrent", ctx)
        p = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", ctypes.byref(p), size)
        try:
            self.call("cuMemsetD8_v2", p.value, 0, size)
            self.call("cuCtxSynchronize")
        except CudaError:
            self.free(ctx, p.value)
            raise
        return p.value

    def free(self, ctx, ptr):
        self.call("cuCtxSetCurrent", ctx)
        self.call("cuMemFree_v2", ptr)


class State:
    """What this process holds: retained contexts and allocations per device ordinal."""

    def __init__(self):
        self.stop_signal, self.contexts, self.handles, self.blocks, self.names = None, {}, {}, {}, {}


def on_signal(state):
    def handler(signum, _frame):
        if state.stop_signal is None:
            state.stop_signal = signum
    return handler


def write_atomic(path, value):
    """JSON to `path` through a temporary file in the same directory (a reader never sees a partial file)."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    os.replace(tmp, path)


def device_list(text, count):
    if text is None:
        return list(range(count))
    out = []
    for part in text.split(","):
        d = int(part)
        if not 0 <= d < count or d in out:
            raise ValueError(f"device {d}: not one of the {count} visible devices, or listed twice")
        out.append(d)
    return out


def leave_map(items, count):
    """--leave DEVICE=MIB ... -> {device: target free bytes}; each device once, each a visible ordinal."""
    out = {}
    for item in items:
        dev, sep, mib = item.partition("=")
        if not sep:
            raise ValueError(f"--leave {item!r}: expected DEVICE=MIB")
        d, m = int(dev), int(mib)
        if not 0 <= d < count or d in out or m < 0:
            raise ValueError(f"--leave {item!r}: an unknown or repeated device, or a negative size")
        out[d] = m * MIB
    if not out:
        raise ValueError("reserve needs at least one --leave DEVICE=MIB")
    return out


def stop_reason(state, end, parent):
    """Why the helper must stop now: a stop signal, its own --max-seconds (`end`, monotonic) or a parent change
    (--parent-pid); None to go on."""
    if state.stop_signal is not None:
        return f"signal {state.stop_signal}"
    if time.monotonic() >= end:
        return "max-seconds"
    if parent is not None and os.getppid() != parent:
        return "parent changed"
    return None


def reserve_device(drv, state, d, target, tol, chunk, rounds, stop):
    """Allocate on device `d` until its free memory is `target` bytes within `tol`; returns the record. `stop()` is
    checked before every allocation and correction: once it names a reason nothing more is allocated (the record
    then carries it as stopped_by)."""
    ctx = state.contexts[d]
    blocks = state.blocks.setdefault(d, [])
    before, total = drv.info(ctx)
    attempts, error, stopped = 0, None, None
    try:
        free = before
        while attempts <= rounds:
            stopped = stop()
            if stopped or abs(free - target) <= tol:
                break
            if free > target:
                want = free - target
                while want > 0:
                    stopped = stop()
                    if stopped:
                        break
                    size = min(chunk, want)
                    blocks.append((drv.alloc(ctx, size), size))
                    want -= size
                if stopped:
                    break
            elif blocks:  # below the target: give the last chunk back and allocate the remainder again
                ptr, size = blocks.pop()
                drv.free(ctx, ptr)
            else:
                break  # below the target without holding anything: nothing to give back
            attempts += 1
            free, _ = drv.info(ctx)
    except CudaError as e:
        error = str(e)
    after, _ = drv.info(ctx)
    held = sum(s for _, s in blocks)
    record = {"name": state.names[d], "total_bytes": total, "free_before_bytes": before, "target_free_bytes": target,
              "free_after_bytes": after, "held_bytes": held, "blocks": len(blocks), "rounds": attempts,
              "deviation_bytes": after - target, "within_tolerance": error is None and abs(after - target) <= tol,
              "tolerance_bytes": tol, "error": error}
    if stopped:
        record["stopped_by"] = stopped
    return record


def release_all(drv, state):
    """Free every allocation (its context current) and release every retained context; the errors met."""
    errors = []
    for d, blocks in state.blocks.items():
        while blocks:
            ptr, _ = blocks.pop()
            try:
                drv.free(state.contexts[d], ptr)
            except CudaError as e:
                errors.append(f"device {d}: {e}")
    for d in list(state.contexts):
        try:
            drv.call("cuDevicePrimaryCtxRelease_v2", state.handles[d])
        except CudaError as e:
            errors.append(f"device {d}: {e}")
        del state.contexts[d]
    return errors


def sample_loop(drv, state, devices, series, interval, end, stop):
    """Sample every device every `interval` s into `series` until `stop()` names a reason (a stop signal, `end`
    (monotonic), a parent change); returns (why, samples, max_gap_s per device, min free per device)."""
    gaps, last, lows, n = {d: 0.0 for d in devices}, {}, {}, 0
    while True:
        tick = time.monotonic()
        for d in devices:
            free, total = drv.info(state.contexts[d])
            t = time.time()
            series.write(f"{t:.4f},{d},{free},{total}\n")
            if d in last:
                gaps[d] = max(gaps[d], t - last[d])
            last[d] = t
            lows[d] = min(lows.get(d, free), free)
        n += 1
        why = stop()
        if why:
            return why, n, {str(d): round(g, 4) for d, g in gaps.items()}, {str(d): v for d, v in lows.items()}
        time.sleep(max(0.0, min(interval - (time.monotonic() - tick), end - time.monotonic())))


def query(drv, state, count, env):
    """The free and total memory of every visible device as JSON on stdout (contexts retained for the call)."""
    devs = []
    for d in range(count):
        state.handles[d], name = drv.device(d)
        state.contexts[d] = drv.retain(state.handles[d])
        free, total = drv.info(state.contexts[d])
        devs.append({"device": d, "name": name, "free_bytes": free, "total_bytes": total})
    print(json.dumps({"devices": devs, "env": env}, indent=1))


def hold(drv, state, args, count, end, result):
    """monitor / reserve: retain the contexts, reserve (reserve), write the ready file after the first sample of
    every device, then sample until a stop. Every step checks the stop conditions first (stop_reason); a stop before
    the ready file returns at once without writing it. Returns the reason it ended."""
    def stop():
        return stop_reason(state, end, args.parent_pid)

    def stopped_early(why):
        result.update(ended_by=why, ready_written=False)
        return why

    targets = leave_map(args.leave, count) if args.mode == "reserve" else None
    devices = sorted(targets) if targets else device_list(args.device, count)
    for d in devices:
        why = stop()
        if why:
            return stopped_early(why)
        state.handles[d], state.names[d] = drv.device(d)
        state.contexts[d] = drv.retain(state.handles[d])
    ready = {"mode": args.mode, "pid": os.getpid(), "interval_ms": args.interval_ms, "env": result["env"],
             "devices": {}}
    result["ready"] = ready  # what is held so far, also when it stops before the ready file
    if targets:
        for d in devices:
            why = stop()
            if why:
                return stopped_early(why)
            ready["devices"][str(d)] = reserve_device(drv, state, d, targets[d], args.tolerance_mib * MIB,
                                                      args.chunk_mib * MIB, args.adjust, stop)
    else:
        for d in devices:
            free, total = drv.info(state.contexts[d])
            ready["devices"][str(d)] = {"name": state.names[d], "free_bytes": free, "total_bytes": total}
    with open(args.series, "w", buffering=1) as series:
        series.write("t_unix,device,free_bytes,total_bytes\n")
        for d in devices:  # one sample of every device before the ready file
            free, total = drv.info(state.contexts[d])
            series.write(f"{time.time():.4f},{d},{free},{total}\n")
        why = stop()
        if why:
            return stopped_early(why)
        ready["t_unix"] = time.time()
        write_atomic(args.ready, ready)
        why, n, gaps, lows = sample_loop(drv, state, devices, series, args.interval_ms / 1000.0, end, stop)
    result.update(ended_by=why, ready_written=True, samples=n, max_gap_s=gaps, min_free_bytes=lows)
    return why


def run(args):
    state = State()
    for sig in STOP_SIGNALS:
        signal.signal(sig, on_signal(state))
    # one absolute end for the whole helper, setup included (query has no --max-seconds)
    end = None if args.mode == "query" else time.monotonic() + args.max_seconds
    result = {"mode": args.mode, "pid": os.getpid(), "env": {k: os.environ.get(k) for k in
                                                              ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")}}
    status, drv = 1, None
    try:
        why = None if end is None else stop_reason(state, end, args.parent_pid)  # before loading the driver library
        if why:
            result.update(ended_by=why, ready_written=False)
        else:
            drv = Driver()
            count = drv.init()
            if args.mode == "query":
                query(drv, state, count, result["env"])
                why = None
            else:
                why = hold(drv, state, args, count, end, result)
        status = 0 if why is None or why.startswith("signal") else 3
    except CudaError as e:
        result["error"] = str(e)
        print(f"cuda-reserve: {e}", file=sys.stderr)
        status = 1
    except (OSError, ValueError) as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"cuda-reserve: {result['error']}", file=sys.stderr)
        status = 2 if isinstance(e, ValueError) else 1
    finally:
        if drv is not None:
            errors = release_all(drv, state)
            result["release_errors"] = errors
            if errors:
                print(f"cuda-reserve: release: {errors}", file=sys.stderr)
                status = 1 if status in (0, 3) else status
        result["exit_status"] = status
        if args.mode != "query" and args.result:
            write_atomic(args.result, result)
    return status


def parse_args():
    head, _, rest = __doc__.partition("\n\n")
    ap = argparse.ArgumentParser(description=head, epilog=rest, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("query", "monitor", "reserve"))
    ap.add_argument("--leave", action="append", default=[], metavar="DEVICE=MIB",
                    help="reserve: free memory to leave on DEVICE (MiB), repeatable, one per device")
    ap.add_argument("--device", default=None, help="monitor: comma-separated ordinals (default: every device)")
    ap.add_argument("--series", type=Path, help="monitor/reserve: new CSV of the samples")
    ap.add_argument("--ready", type=Path, help="monitor/reserve: new JSON written once sampling runs")
    ap.add_argument("--result", type=Path, help="monitor/reserve: new JSON written at the end")
    ap.add_argument("--interval-ms", type=int, default=50, help="sampling interval (ms)")
    ap.add_argument("--tolerance-mib", type=int, default=64, help="reserve: allowed deviation from the target")
    ap.add_argument("--chunk-mib", type=int, default=1024, help="reserve: largest single allocation")
    ap.add_argument("--adjust", type=int, default=4, help="reserve: correction rounds after the first pass")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="monitor/reserve: required; the helper frees everything and ends (exit 3) after it")
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="monitor/reserve: end (exit 3, memory freed) when the parent process is no longer it")
    args = ap.parse_args()
    if args.mode != "query":
        missing = [k for k in ("series", "ready", "result", "max_seconds") if getattr(args, k) is None]
        if missing:
            ap.error(f"{args.mode} needs --{', --'.join(m.replace('_', '-') for m in missing)}")
        for p in (args.series, args.ready, args.result):
            if p.exists():
                ap.error(f"{p}: must not exist")
        if not (math.isfinite(args.max_seconds) and args.max_seconds > 0) or not 1 <= args.interval_ms <= 1000 or \
                args.tolerance_mib < 0 or args.chunk_mib < 2 or args.adjust < 0:
            ap.error("--max-seconds finite and > 0, --interval-ms 1..1000, --tolerance-mib >= 0, --chunk-mib >= 2, "
                     "--adjust >= 0")
    if args.mode != "reserve" and args.leave:
        ap.error("--leave is for reserve")
    if args.mode != "monitor" and args.device is not None:
        ap.error("--device is for monitor")
    return args


if __name__ == "__main__":
    sys.exit(run(parse_args()))
