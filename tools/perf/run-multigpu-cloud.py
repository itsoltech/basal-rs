#!/usr/bin/env python3
"""Manual evidence run for models on separate GPUs (`device` per model) on a rented Linux host with two identical
NVIDIA GPUs (RTX 6000 Ada). Not a test and not a CI gate: it collects raw evidence for a report the operator writes.

  cd /work && python3 tools/perf/run-multigpu-cloud.py \
      --base /opt/basal-dev/basal-cuda-base --candidate /opt/basal-dev/basal-cuda \
      --inputs reports/reference-basal-1.5-4.5B-fp32 --out out/multigpu [--models "4.5B mini"]

Prerequisites (synced to /work, prefetched to /data): this file, tools/bench/long_states.jsonl, the bundled GEMM
tables crates/basal-cli/gemm-tables (f16, batch-invariant; no GEMM search is run), crates/basal-cli/src/config.rs
(pinned revisions of the known models), an --inputs directory with bench.jsonl and systemone.jsonl (only the inputs
are read, so one directory serves every model), the weights of --models in HF_HOME=/data/hf at the pinned
revisions, the CUDA libraries of `basal setup` in BASAL_HOME=/data/basal, and both binaries. The baseline binary has
no --device: it runs on CUDA device 0, or with CUDA_VISIBLE_DEVICES=1 on the second GPU. CUDA_DEVICE_ORDER=PCI_BUS_ID
is set for every process, so CUDA ordinals before CUDA_VISIBLE_DEVICES are the nvidia-smi indices.

Steps (each failure is recorded and the next independent step runs; exit status 1 when a check failed):
  doctor     candidate `doctor --json` with a two-GPU configuration: default visibility, CUDA_VISIBLE_DEVICES=1,0,
             and CUDA_VISIBLE_DEVICES=1 (device 1 must fail); no weights are loaded. Identical cards have the
             same name, so the doctor shows the device count, not which card a number is: the physical card of
             each number is read from GPU memory of the `reversed` server below
  invalid    candidate `serve` that must exit before serving: device 2, device 1 under CUDA_VISIBLE_DEVICES=1, the
             same model twice, --device with --config
  exports    per model and batching mode: baseline on device 0 (reference), candidate default (device 0), candidate
             --device 1, baseline under CUDA_VISIBLE_DEVICES=1; bench.jsonl and systemone.jsonl must be byte-equal
             to the reference and `basal compare` must report no difference; manifest.json (timings, paths) is not
             compared; the physical GPU of each process is read from nvidia-smi --query-compute-apps
  serve      one candidate API per configuration (temporary YAML, explicit GEMM tables, long lanes, /metrics):
               local-<model>  the model alone on its GPU, sequential requests: the reference answers
               dual           first model on device 0, second on device 1: sequential and concurrent short and long
                              traffic of both models, routing, unknown model, shutdown with a request in flight
               shared         both models on device 0 (one gate): the same workload, timings descriptive only
               reversed       dual under CUDA_VISIBLE_DEVICES=1,0: sequential requests, placement swapped
             every answer (status and body without usage.latency_ms) must equal the local reference; nvidia-smi is
             sampled per physical index; per-request queue/compute headers give approximate server intervals

Outputs (all under --out, which must not exist; raw, may contain local paths, keep outside Git): checks.txt (one
PASS/FAIL/GAP line per check), summary.json, commands.jsonl, gpus.csv, binaries.sha256, per step directories with
logs, exports, compare JSON, telemetry CSV (time, physical index, bus id, ...), request records and metrics. Timings
are observations only: the run makes no speed or parallelism claim from them.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MODES = {
    "tree": ["--batching", "tree"],
    "single": ["--batching", "single"],
    "budget": ["--batching", "budget"],
    "tree-noprefix": ["--batching", "tree", "--no-prefix-cache"],
    "single-noprefix": ["--batching", "single", "--no-prefix-cache"],
}
LIVE = set()  # spawned processes (own sessions), stopped on any exit


class StepFailed(Exception):
    pass


# ---------------------------------------------------------------- records


class Record:
    def __init__(self, out):
        self.out = out
        self.checks = []
        self.lock = threading.Lock()

    def check(self, name, status, detail=""):
        assert status in ("PASS", "FAIL", "GAP")
        with self.lock:
            self.checks.append({"check": name, "status": status, "detail": detail})
            with open(self.out / "checks.txt", "a") as f:
                f.write(f"{status} {name}{': ' + detail if detail else ''}\n")
        print(f"{time.strftime('%H:%M:%S')} {status} {name}{': ' + detail if detail else ''}", flush=True)

    def expect(self, name, ok, detail=""):
        self.check(name, "PASS" if ok else "FAIL", detail)
        return ok

    def command(self, entry):
        with self.lock, open(self.out / "commands.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def line_count(path):
    return sum(1 for line in Path(path).read_text().splitlines() if line.strip())


# ---------------------------------------------------------------- processes


def spawn(argv, env, log_path):
    log = open(log_path, "wb")
    try:
        p = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    finally:
        log.close()
    LIVE.add(p)
    return p


def stop(p, timeout=60):
    """SIGTERM to the process group, SIGKILL after `timeout` s; returns the exit status."""
    if p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
    LIVE.discard(p)
    return p.returncode


def run_cmd(rec, name, argv, env, log_path, timeout, env_delta):
    """Run to completion (bounded); returns (exit status or None on timeout, seconds)."""
    t0 = time.time()
    p = spawn(argv, env, log_path)
    try:
        code = p.wait(timeout)
    except subprocess.TimeoutExpired:
        code = None
    finally:
        stop(p)
    dt = time.time() - t0
    rec.command({"step": name, "argv": [str(a) for a in argv], "env": env_delta, "exit": code, "seconds": round(dt, 3)})
    return code, dt


# ---------------------------------------------------------------- nvidia-smi


def smi(args, timeout=30):
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise StepFailed(f"nvidia-smi {' '.join(args)}: {r.stderr.strip() or r.returncode}")
    return [[c.strip() for c in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def bus(b):
    """PCI bus id as the two queries print it (domain width differs between fields)."""
    b = b.strip().upper()
    return b[-12:] if len(b) >= 12 else b


class Telemetry:
    """nvidia-smi every 200 ms, one row per physical GPU, prefixed with the local read time."""

    FIELDS = "index,pci.bus_id,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"

    def __init__(self, path):
        self.samples = []
        self.file = open(path, "w")
        self.file.write("t_unix," + self.FIELDS + "\n")
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        LIVE.add(self.proc)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            t = time.time()
            self.file.write(f"{t:.3f},{line.strip()}\n")
            c = [x.strip() for x in line.split(",")]
            try:
                self.samples.append((t, int(c[0]), float(c[2]), float(c[3])))
            except (ValueError, IndexError):
                pass

    def close(self):
        stop(self.proc, timeout=5)
        self.thread.join(5)
        self.file.close()

    def window(self, t0, t1, indices):
        """Per physical index: samples, mean utilization, peak memory; share of ~200 ms ticks with every index busy."""
        per = {i: [s for s in self.samples if s[1] == i and t0 <= s[0] <= t1] for i in indices}
        out = {str(i): {"samples": len(xs),
                        "util_mean": round(sum(x[2] for x in xs) / len(xs), 1) if xs else None,
                        "mem_peak_mib": max((x[3] for x in xs), default=None)} for i, xs in per.items()}
        if len(indices) > 1 and all(per.values()):
            first, rest = per[indices[0]], [per[i] for i in indices[1:]]
            busy = 0
            for t, _, util, _ in first:
                near = [min(xs, key=lambda s: abs(s[0] - t)) for xs in rest]
                busy += util > 0 and all(abs(n[0] - t) < 0.15 and n[2] > 0 for n in near)
            out["all_busy_tick_share"] = round(busy / len(first), 3)
        return out


def session_placement(root_pid, by_bus):
    """Physical GPU index -> MiB of the processes in the session of `root_pid` (None: not observable)."""
    rows = smi(["--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader,nounits"])
    seen, out = False, {}
    for row in rows:
        try:
            pid, b, mib = int(row[0]), bus(row[1]), float(row[2])
        except (ValueError, IndexError):
            continue
        seen = True
        try:
            if os.getsid(pid) != root_pid:
                continue
        except (ProcessLookupError, PermissionError):
            continue
        idx = by_bus.get(b)
        out[idx] = out.get(idx, 0.0) + mib
    return out if (seen or not rows) else None


class PlacementWatch:
    """Polls `session_placement` until closed; keeps the peak MiB per physical index."""

    def __init__(self, root_pid, by_bus):
        self.root, self.by_bus, self.peak, self.stopped = root_pid, by_bus, {}, threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self):
        while not self.stopped.wait(0.5):
            try:
                for i, m in (session_placement(self.root, self.by_bus) or {}).items():
                    self.peak[i] = max(self.peak.get(i, 0.0), m)
            except (StepFailed, subprocess.SubprocessError, OSError):
                pass

    def close(self):
        self.stopped.set()
        self.thread.join(10)
        return self.peak


def placement_check(rec, name, observed, expected):
    """`observed`: index -> MiB; `expected`: set of physical indices that should hold memory."""
    if not observed:
        rec.check(name, "GAP", "no compute process of this run seen by nvidia-smi (container PID namespace?)")
        return
    got = {i for i, m in observed.items() if m > 0}
    rec.expect(name, got == set(expected), f"physical GPU MiB {json.dumps({str(k): v for k, v in observed.items()})}, "
                                           f"expected only {sorted(expected)}")


# ---------------------------------------------------------------- environment and inputs


def environment(args):
    env = dict(os.environ)
    for k in ("CUDA_VISIBLE_DEVICES", "BASAL_WGP_LOAD", "BASAL_CONFIG", "BASAL_ACCESS_LOG"):
        env.pop(k, None)
    env.update(HF_HOME=args.hf_home, BASAL_HOME=args.basal_home, BASAL_NO_UPDATE_CHECK="1",
               CUDA_DEVICE_ORDER="PCI_BUS_ID")
    libs = [f"{args.basal_home}/.local/cuda/12.9.1/lib"]
    if Path("/usr/local/cuda-12.9/compat").is_dir():
        libs.insert(0, "/usr/local/cuda-12.9/compat")
    if env.get("LD_LIBRARY_PATH"):
        libs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env


def with_env(env, delta):
    e = dict(env)
    e.update(delta)
    return e


def known_revisions(root):
    """short name -> (repo, revision) from KNOWN_MODELS of the synced crates/basal-cli/src/config.rs."""
    src = (root / "crates/basal-cli/src/config.rs").read_text()
    found = re.findall(r'short:\s*"([^"]+)",\s*repo:\s*"([^"]+)",\s*revision:\s*"([0-9a-f]{40})"', src)
    if not found:
        raise StepFailed("no KNOWN_MODELS revisions in crates/basal-cli/src/config.rs")
    return {s: (repo, rev) for s, repo, rev in found}


def resolve_model(spec, known, hf_home):
    """`mini`, `4.5B`, `max` (pinned revision in HF_HOME) or `NAME=DIRECTORY`."""
    if "=" in spec:
        short, d = spec.split("=", 1)
        path = Path(d)
    else:
        if spec not in known:
            raise StepFailed(f"unknown model {spec!r} (known: {', '.join(known)}; or NAME=DIRECTORY)")
        repo, rev = known[spec]
        short = spec
        path = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    if not (path / "basal.json").is_file():
        raise StepFailed(f"{short}: no model at {path} (prefetch the pinned revision)")
    name = json.loads((path / "basal.json").read_text())["name"]
    return {"short": short, "dir": path.resolve(), "name": name, "slug": f"basal-1.5-{short}"}


# ---------------------------------------------------------------- doctor and invalid configurations


def yaml_config(path, port, cache, models):
    """Serve configuration; strings as JSON (valid YAML double-quoted scalars)."""
    lines = [f"addr: 127.0.0.1:{port}", "metrics: true", "max_inflight: 1024", f"gemm_cache: {json.dumps(str(cache))}",
             "models:"]
    for m in models:
        lines.append(f"  - path: {json.dumps(str(m['dir']))}")
        for k in ("gemm_table", "dtype", "device", "long_tokens"):
            if k in m:
                lines.append(f"    {k}: {json.dumps(m[k]) if isinstance(m[k], str) else m[k]}")
    path.write_text("\n".join(lines) + "\n")
    return path


def entry(m, table, device, long_tokens):
    return {"dir": m["dir"], "gemm_table": str(table), "dtype": "f16", "device": device, "long_tokens": long_tokens}


def step_doctor(rec, ctx):
    d = ctx.out / "doctor"
    d.mkdir()
    a, b = ctx.models[0], ctx.models[1]
    cfg = yaml_config(d / "dual.yml", ctx.port, d / "gemm-cache",
                      [entry(a, ctx.tables[a["short"]], 0, ctx.long_tokens),
                       entry(b, ctx.tables[b["short"]], 1, ctx.long_tokens)])
    for label, visible, want in (("default", None, 2), ("reversed", "1,0", 2), ("only1", "1", 1)):
        delta = {"CUDA_VISIBLE_DEVICES": visible} if visible else {}
        log = d / f"{label}.json"
        code, _ = run_cmd(rec, f"doctor-{label}",
                          [ctx.candidate, "--color", "never", "doctor", "--config", cfg, "--json"],
                          with_env(ctx.env, delta), log, 180, delta)
        if code is None:
            rec.check(f"doctor {label}", "FAIL", "timed out")
            continue
        text = log.read_text(errors="replace")
        try:
            report, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
        except ValueError:
            rec.check(f"doctor {label}", "FAIL", f"no JSON report (exit {code}), see {log.name}")
            continue
        devices = [c["what"] for c in report["checks"] if c["area"] == "gpu" and c["what"].startswith("CUDA device ")]
        fails = [c["what"] for c in report["checks"] if c["level"] == "fail" and c["area"] == "models"]
        rec.expect(f"doctor {label}: CUDA devices listed", len(devices) == want, f"{devices}")
        if visible == "1":
            rec.expect(f"doctor {label}: device 1 refused", any("device 1:" in f for f in fails), f"{fails}")
        else:
            rec.expect(f"doctor {label}: no model failure", not fails, f"{fails}; report ok={report.get('ok')}")


def step_invalid(rec, ctx):
    d = ctx.out / "invalid"
    d.mkdir()
    m = ctx.models[0]
    t = ctx.tables[m["short"]]
    cases = [
        ("device-2", {}, yaml_config(d / "device-2.yml", ctx.port, d / "gemm-cache", [entry(m, t, 2, 0)]), [],
         "does not exist"),
        ("visible1-device-1", {"CUDA_VISIBLE_DEVICES": "1"},
         yaml_config(d / "visible1-device-1.yml", ctx.port, d / "gemm-cache", [entry(m, t, 1, 0)]), [],
         "does not exist"),
        ("duplicate", {}, yaml_config(d / "duplicate.yml", ctx.port, d / "gemm-cache",
                                      [entry(m, t, 0, 0), entry(m, t, 1, 0)]), [], "configured twice"),
        ("device-with-config", {}, yaml_config(d / "plain.yml", ctx.port, d / "gemm-cache", [entry(m, t, 0, 0)]),
         ["--device", "1"], "cannot be used with"),
    ]
    for label, delta, cfg, extra, needle in cases:
        port_free_or_fail(ctx.port)
        log = d / f"{label}.log"
        code, dt = run_cmd(rec, f"invalid-{label}",
                           [ctx.candidate, "--color", "never", "serve", "--config", cfg, *extra],
                           with_env(ctx.env, delta), log, 120, delta)
        text = log.read_text(errors="replace")
        rec.expect(f"invalid {label}: refused before serving",
                   code not in (0, None) and needle in text and "serving on http" not in text,
                   f"exit {code} after {dt:.1f} s, expected message {needle!r}")


# ---------------------------------------------------------------- exports


def compare_problems(v, path=""):
    """Every nonzero difference, false equality flag, flip, mismatch or missing item in `basal compare` output.
    Skipped: quality (scores, not differences), paths, and `upstream_endpoint`, which compares a's System One answer
    with b's upstream-dialect answer (meant for an upstream reference); the byte comparison covers those answers."""
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            p = f"{path}/{k}"
            if k in ("quality", "a", "b", "known_deviations", "evidence_a", "evidence_b", "sign", "upstream_endpoint"):
                continue
            if k in ("flips_avg", "flips_cal", "mismatches", "missing") and x:
                out.append(f"{p}: {len(x)}")
            elif "diff" in k or k in ("tv_avg", "tv_cal"):
                if isinstance(x, dict) and "max" in x:
                    if x["max"] not in (0, 0.0, None):
                        out.append(f"{p}.max={x['max']}")
                elif isinstance(x, (int, float)) and not isinstance(x, bool) and x != 0:
                    out.append(f"{p}={x}")
                elif isinstance(x, (dict, list)):
                    out += compare_problems(x, p)
            elif (k.endswith("_same") or k in ("tokens_identical", "errors_agree")) and x is False:
                out.append(f"{p}=false")
            elif k == "error_a" and v.get("error_a") != v.get("error_b"):
                out.append(f"{p}!=error_b")
            elif isinstance(x, (dict, list)):
                out += compare_problems(x, p)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            out += compare_problems(x, f"{path}[{i}]")
    return out


def step_exports(rec, ctx):
    d = ctx.out / "exports"
    d.mkdir()
    n_bench, n_so = line_count(ctx.inputs / "bench.jsonl"), line_count(ctx.inputs / "systemone.jsonl")
    # (label, binary, environment delta, extra arguments, physical GPU)
    variants = [("base-dev0", ctx.base, {}, [], 0), ("cand-dev0", ctx.candidate, {}, [], 0),
                ("cand-dev1", ctx.candidate, {}, ["--device", "1"], 1)]
    if not ctx.args.no_base_visible1:
        variants.append(("base-visible1", ctx.base, {"CUDA_VISIBLE_DEVICES": "1"}, [], 1))
    for m in ctx.models:
        for mode in ctx.modes:
            done = {}
            for label, binary, delta, extra, phys in variants:
                name = f"{m['short']}-{mode}-{label}"
                out = d / name
                watch = None
                try:
                    p_env = with_env(ctx.env, delta)
                    argv = [binary, "--color", "never", "export", "--model", m["dir"],
                            "--gemm-table", ctx.tables[m["short"]], *MODES[mode], *extra,
                            "--inputs", ctx.inputs, "--out", out]
                    t0 = time.time()
                    p = spawn(argv, p_env, d / f"{name}.log")
                    watch = PlacementWatch(p.pid, ctx.by_bus)
                    try:
                        code = p.wait(ctx.args.export_timeout)
                    except subprocess.TimeoutExpired:
                        code = None
                    finally:
                        stop(p)
                    rec.command({"step": name, "argv": [str(a) for a in argv], "env": delta, "exit": code,
                                 "seconds": round(time.time() - t0, 3)})
                    peak = watch.close()
                    watch = None
                    if not rec.expect(f"export {name}: completed", code == 0, f"exit {code}, see {name}.log"):
                        continue
                    counts = (line_count(out / "bench.jsonl"), line_count(out / "systemone.jsonl"))
                    if not rec.expect(f"export {name}: all inputs exported", counts == (n_bench, n_so),
                                      f"{counts} lines, inputs {(n_bench, n_so)}"):
                        continue
                    placement_check(rec, f"export {name}: on physical GPU {phys}", peak, {phys})
                    done[label] = out
                finally:
                    if watch:
                        watch.close()
            ref = done.get("base-dev0")
            if ref is None:
                rec.check(f"export {m['short']}-{mode}: comparisons", "FAIL", "no baseline reference export")
                continue
            for label, out in done.items():
                if label == "base-dev0":
                    continue
                pair = f"{m['short']}-{mode} base-dev0 vs {label}"
                hashes = {f: (sha256(ref / f), sha256(out / f)) for f in ("bench.jsonl", "systemone.jsonl")}
                write_json(d / f"{m['short']}-{mode}-{label}.sha256.json", hashes)
                rec.expect(f"bitwise {pair}", all(a == b for a, b in hashes.values()),
                           ", ".join(f"{f} {'equal' if a == b else 'DIFFERENT'}" for f, (a, b) in hashes.items()))
                cmp_out = d / f"compare-{m['short']}-{mode}-{label}.json"
                code, _ = run_cmd(rec, f"compare-{pair}",
                                  [ctx.candidate, "--color", "never", "compare", "--a", ref, "--b", out,
                                   "--out", cmp_out], ctx.env, d / f"compare-{m['short']}-{mode}-{label}.log", 600, {})
                if code != 0 or not cmp_out.is_file():
                    rec.check(f"basal compare {pair}", "FAIL", f"exit {code}")
                    continue
                problems = compare_problems(json.loads(cmp_out.read_text()))
                rec.expect(f"basal compare {pair}: no difference", not problems, "; ".join(problems[:5]))


# ---------------------------------------------------------------- HTTP


def port_free_or_fail(port):
    with socket.socket() as s:
        s.settimeout(2)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise StepFailed(f"127.0.0.1:{port} already in use")


def http(method, url, body=None, timeout=30):
    """(status, headers, bytes); network errors raise."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def normalized(raw):
    """Answer body without the server wall time (usage.latency_ms), canonical JSON."""
    try:
        v = json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")
    if isinstance(v, dict) and isinstance(v.get("usage"), dict):
        v["usage"].pop("latency_ms", None)
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def workload(inputs, long_file, model):
    """(id, class, body) of every bench item (one choice question), System One case and long document request."""
    out = []
    for line in (inputs / "bench.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            q = {"type": "choice", "instructions": r["question"],
                 "criteria": {f"option_{k + 1}": o for k, o in enumerate(r["options"])}}
            out.append((f"bench-{r['index']}", "short", {"model": model, "state": r["state"], "questions": {"q": q}}))
    for path, cls in ((inputs / "systemone.jsonl", "short"), (long_file, "long")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if isinstance(r.get("request"), dict):
                    body = copy.deepcopy(r["request"])
                    body["model"] = model
                    out.append((f"{cls}:{r['id']}", cls, body))
    return out


class Server:
    def __init__(self, rec, ctx, name, cfg, delta):
        self.rec, self.ctx, self.name, self.delta = rec, ctx, name, delta
        self.dir = ctx.out / "serve" / name
        self.dir.mkdir(parents=True)
        self.url = f"http://127.0.0.1:{ctx.port}"
        self.records, self.phases = [], []
        port_free_or_fail(ctx.port)
        self.telemetry = Telemetry(self.dir / "telemetry.csv")
        self.argv = [ctx.candidate, "--color", "never", "serve", "--config", cfg]
        try:
            self.proc = spawn(self.argv, with_env(ctx.env, delta), self.dir / "server.log")
        except OSError:
            self.telemetry.close()
            raise
        rec.command({"step": f"serve-{name}", "argv": [str(a) for a in self.argv], "env": delta})

    def alive(self):
        if self.proc.poll() is not None:
            raise StepFailed(f"server {self.name} exited ({self.proc.returncode}), see {self.dir / 'server.log'}")

    def wait_ready(self):
        deadline = time.time() + self.ctx.args.ready_timeout
        while time.time() < deadline:
            self.alive()
            try:
                status, _, body = http("GET", self.url + "/health", timeout=5)
                if status == 200:
                    return json.loads(body)
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise StepFailed(f"server {self.name} not ready after {self.ctx.args.ready_timeout} s")

    def send(self, phase, short, rid, cls, body):
        t0 = time.time()
        rec = {"phase": phase, "model": short, "id": rid, "class": cls, "t_send": round(t0, 4)}
        try:
            status, headers, raw = http("POST", self.url + "/v1/systemone", body, self.ctx.args.http_timeout)
            h = {k.lower(): v for k, v in headers.items()}
            rec.update(status=status, answer=normalized(raw),
                       queue_ms=float(h["x-basal-queue-ms"]) if "x-basal-queue-ms" in h else None,
                       compute_ms=float(h["x-basal-compute-ms"]) if "x-basal-compute-ms" in h else None,
                       batch=int(h["x-basal-batch-requests"]) if "x-basal-batch-requests" in h else None)
        except OSError as e:
            rec.update(status=None, error=f"{type(e).__name__}: {e}")
        rec["t_done"] = round(time.time(), 4)
        self.records.append(rec)
        return rec

    def phase(self, phase, items, clients):
        """`items`: (short, id, class, body); clients 1 = in order, one at a time."""
        t0 = time.time()
        if clients == 1:
            for it in items:
                self.alive()
                self.send(phase, *it)
        else:
            with concurrent.futures.ThreadPoolExecutor(clients) as pool:
                list(pool.map(lambda it: self.send(phase, *it), items))
        self.phases.append({"phase": phase, "t0": t0, "t1": time.time(), "requests": len(items), "clients": clients})

    def metrics(self):
        status, _, raw = http("GET", self.url + "/metrics", timeout=30)
        text = raw.decode(errors="replace")
        (self.dir / "metrics.txt").write_text(text)
        lanes = {}
        for line in text.splitlines():
            mt = re.match(r'basal_batch_requests_count\{([^}]*)\}\s+([0-9.e+]+)', line)
            if mt:
                labels = dict(re.findall(r'(\w+)="([^"]*)"', mt.group(1)))
                lanes[(labels.get("model"), labels.get("lane"))] = float(mt.group(2))
        return status, lanes

    def shutdown(self, inflight=None):
        """SIGTERM (optionally with a request in flight); the server must exit 0 within the timeout."""
        result = {}
        thread = None
        if inflight:
            thread = threading.Thread(target=lambda: result.update(self.send("shutdown-inflight", *inflight)))
            thread.start()
            time.sleep(self.ctx.args.inflight_delay)
        os.kill(self.proc.pid, signal.SIGTERM)
        try:
            code = self.proc.wait(self.ctx.args.shutdown_timeout)
        except subprocess.TimeoutExpired:
            code = None
        if thread:
            thread.join(self.ctx.args.http_timeout)
        return code, result

    def close(self):
        stop(self.proc)
        self.telemetry.close()
        with open(self.dir / "requests.jsonl", "w") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_json(self.dir / "phases.json", self.phases)


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap_s(xs, ys):
    total, i, j = 0.0, 0, 0
    while i < len(xs) and j < len(ys):
        total += max(0.0, min(xs[i][1], ys[j][1]) - max(xs[i][0], ys[j][0]))
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return total


def observe(srv, phase, physical, shorts):
    """Descriptive: per-GPU telemetry in the phase window and approximate server compute intervals per model."""
    ph = next(p for p in srv.phases if p["phase"] == phase)
    recs = [r for r in srv.records if r["phase"] == phase and r.get("compute_ms") is not None]
    # server compute interval ~ [send + queue, send + queue + compute] on the client clock (HTTP time not subtracted)
    iv = {s: merged([(r["t_send"] + r["queue_ms"] / 1e3, r["t_send"] + (r["queue_ms"] + r["compute_ms"]) / 1e3)
                     for r in recs if r["model"] == s]) for s in shorts}
    obs = {"phase": phase, "wall_s": round(ph["t1"] - ph["t0"], 3), "requests": ph["requests"],
           "clients": ph["clients"], "gpus": srv.telemetry.window(ph["t0"], ph["t1"], physical),
           "compute_interval_union_s": {s: round(sum(b - a for a, b in v), 3) for s, v in iv.items()}}
    if len(shorts) == 2:
        obs["compute_interval_overlap_s"] = round(overlap_s(iv[shorts[0]], iv[shorts[1]]), 3)
    for s in shorts:
        lat = sorted(r["t_done"] - r["t_send"] for r in srv.records if r["phase"] == phase and r["model"] == s)
        if lat:
            obs[f"latency_s_{s}"] = {"p50": round(lat[len(lat) // 2], 4), "max": round(lat[-1], 4), "n": len(lat)}
    return obs


def answers_equal(rec, srv, phase, reference, label):
    recs = [r for r in srv.records if r["phase"] == phase]
    bad = [f"{r['model']}/{r['id']}" for r in recs
           if (r.get("status"), r.get("answer")) != reference.get((r["model"], r["id"]))]
    errors = [f"{r['model']}/{r['id']}: {r.get('status')}" for r in recs
              if r.get("status") is None or r["status"] >= 500]
    if bad:
        write_json(srv.dir / f"mismatches-{phase}.json", bad)
    rec.expect(f"serve {label} {phase}: answers equal to the model-local server", not bad and recs,
               f"{len(recs)} requests, {len(bad)} differing{': ' + ', '.join(bad[:5]) if bad else ''}")
    rec.expect(f"serve {label} {phase}: no transport or 5xx error", not errors, ", ".join(errors[:5]))


def run_server(rec, ctx, label, models, delta, phases, physical_of, check_lanes=False, inflight=False):
    """Start, check routing/placement, run `phases` [(phase, clients, rounds)], scrape metrics, shut down.
    `models`: [(model, device)]; `physical_of`: short -> expected physical index."""
    cfg = yaml_config(ctx.out / "serve" / f"{label}.yml", ctx.port, ctx.out / "serve" / "gemm-cache",
                      [entry(m, ctx.tables[m["short"]], dev, ctx.long_tokens) for m, dev in models])
    srv = Server(rec, ctx, label, cfg, delta)
    observations = []
    try:
        health = srv.wait_ready()
        names = sorted(m["name"] for m, _ in models)
        rec.expect(f"serve {label}: ready with the configured models", sorted(health.get("models", [])) == names,
                   f"{health}")
        _, _, raw = http("GET", srv.url + "/v1/models", timeout=30)
        listed = sorted(x["name"] for x in json.loads(raw)["models"])
        rec.expect(f"serve {label}: /v1/models", listed == names, f"{listed}")
        status, _, raw = http("POST", srv.url + "/v1/systemone",
                              {"model": "no-such-model", "state": "x", "questions": {
                                  "q": {"type": "noul", "instructions": "?"}}}, timeout=30)
        rec.expect(f"serve {label}: unknown model refused", 400 <= status < 500 and b"no-such-model" in raw,
                   f"status {status}")
        time.sleep(2)
        placement = session_placement(srv.proc.pid, ctx.by_bus)
        write_json(srv.dir / "placement-after-load.json", {str(k): v for k, v in (placement or {}).items()})
        expected = {physical_of[m["short"]] for m, _ in models}
        placement_check(rec, f"serve {label}: memory only on physical GPUs {sorted(expected)}", placement, expected)
        if placement and len(expected) == 2:
            big = max((m for m, _ in models), key=lambda m: m["dir"].joinpath("model.safetensors").stat().st_size)
            small = next(m for m, _ in models if m is not big)
            rec.expect(f"serve {label}: larger model on physical GPU {physical_of[big['short']]}",
                       placement.get(physical_of[big["short"]], 0) > placement.get(physical_of[small["short"]], 0),
                       f"MiB by physical GPU {json.dumps({str(k): v for k, v in placement.items()})}")
        shorts = [m["short"] for m, _ in models]
        work = {m["short"]: workload(ctx.inputs, ctx.long_file, m["name"]) for m, _ in models}
        for phase, clients, rounds in phases:
            items = [(s, rid, cls, body) for s in shorts for rid, cls, body in work[s]] * rounds
            if clients > 1:
                random.Random(f"{label}-{phase}").shuffle(items)
            srv.phase(phase, items, clients)
            observations.append(observe(srv, phase, sorted(set(ctx.physical)), shorts))
        status, lanes = srv.metrics()
        write_json(srv.dir / "lanes.json", {f"{k[0]}/{k[1]}": v for k, v in lanes.items()})
        if check_lanes:
            for m, _ in models:
                rec.expect(f"serve {label}: {m['short']} used its main and long lane",
                           lanes.get((m["name"], "main"), 0) > 0 and lanes.get((m["name"], "long"), 0) > 0,
                           f"batches main {lanes.get((m['name'], 'main'))}, long {lanes.get((m['name'], 'long'))}")
        flight = None
        if inflight:
            m0 = models[0][0]
            longest = max((x for x in work[m0["short"]] if x[1] == "long"), key=lambda x: len(json.dumps(x[2])))
            flight = (m0["short"], *longest)
        code, result = srv.shutdown(flight)
        log = (srv.dir / "server.log").read_text(errors="replace")
        rec.expect(f"serve {label}: graceful shutdown", code == 0 and "shutting down" in log, f"exit {code}")
        if flight:
            # observation: whether the request reached the server before SIGTERM is not controlled
            rec.check(f"serve {label}: request in flight at SIGTERM", "PASS" if result.get("status") == 200 else "GAP",
                      f"status {result.get('status')}{' ' + result['error'] if result.get('error') else ''}")
        after = session_placement(srv.proc.pid, ctx.by_bus)
        # an empty list proves nothing when the process was not visible to nvidia-smi while it ran
        rec.check(f"serve {label}: no GPU memory left after exit",
                  "FAIL" if after else "PASS" if after == {} and placement else "GAP", f"{after}")
    finally:
        srv.close()
        write_json(srv.dir / "observations.json", observations)
    return srv


def step_serve(rec, ctx):
    a, b = ctx.models[0], ctx.models[1]
    (ctx.out / "serve").mkdir()
    seq = ("sequential", 1, 1)
    conc = ("concurrent", ctx.args.clients, ctx.args.rounds)
    reference = {}
    for m, dev in ((a, 0), (b, 1)):
        srv = run_server(rec, ctx, f"local-{m['short']}", [(m, dev)], {}, [seq], {m["short"]: dev})
        local = [r for r in srv.records if r["phase"] == "sequential"]
        errors = [r["id"] for r in local if r.get("status") is None or r["status"] >= 500]
        rec.expect(f"serve local-{m['short']}: reference answers", local and not errors,
                   f"{sum(r.get('status') == 200 for r in local)} of {len(local)} with 200, transport/5xx: "
                   f"{errors[:5]}")
        for r in srv.records:
            if r["phase"] == "sequential":
                reference[(r["model"], r["id"])] = (r.get("status"), r.get("answer"))
    write_json(ctx.out / "serve" / "reference-statuses.json",
               {f"{k[0]}/{k[1]}": v[0] for k, v in reference.items()})
    plans = [("dual", [(a, 0), (b, 1)], {}, [seq, conc], {a["short"]: 0, b["short"]: 1}, True),
             ("shared", [(a, 0), (b, 0)], {}, [seq, conc], {a["short"]: 0, b["short"]: 0}, False),
             ("reversed", [(a, 0), (b, 1)], {"CUDA_VISIBLE_DEVICES": "1,0"}, [seq],
              {a["short"]: 1, b["short"]: 0}, False)]
    for label, models, delta, phases, physical_of, inflight in plans:
        try:
            srv = run_server(rec, ctx, label, models, delta, phases, physical_of,
                             check_lanes=len(phases) > 1, inflight=inflight)
            for phase, _, _ in phases:
                answers_equal(rec, srv, phase, reference, label)
        except (StepFailed, OSError, ValueError, KeyError) as e:
            rec.check(f"serve {label}", "FAIL", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path, help="baseline CUDA binary (no --device)")
    ap.add_argument("--candidate", required=True, type=Path, help="candidate CUDA binary")
    ap.add_argument("--inputs", required=True, type=Path, help="directory with bench.jsonl and systemone.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="new output directory")
    ap.add_argument("--models", default="4.5B mini",
                    help="two or more of mini, 4.5B, max, or NAME=DIRECTORY (exports: each); serving uses the first "
                         "two: the first on device 0, the second on device 1 in `dual`")
    ap.add_argument("--modes", default="tree single tree-noprefix", help=f"export modes: {', '.join(MODES)}")
    ap.add_argument("--steps", default="doctor invalid exports serve", help="subset of doctor invalid exports serve")
    ap.add_argument("--no-base-visible1", action="store_true", help="skip baseline exports under CUDA_VISIBLE_DEVICES=1")
    ap.add_argument("--long-requests", type=Path, default=Path("tools/bench/long_states.jsonl"))
    ap.add_argument("--long-tokens", type=int, default=3000, help="long lane threshold of every served model")
    ap.add_argument("--clients", type=int, default=16, help="concurrent clients in the concurrent phase")
    ap.add_argument("--rounds", type=int, default=2, help="passes over the workload in the concurrent phase")
    ap.add_argument("--port", type=int, default=18200)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--export-timeout", type=int, default=1800)
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--http-timeout", type=int, default=300)
    ap.add_argument("--shutdown-timeout", type=int, default=120)
    ap.add_argument("--inflight-delay", type=float, default=0.5, help="seconds between the in-flight request and SIGTERM")
    args = ap.parse_args()

    steps = args.steps.split()
    modes = args.modes.split()
    unknown = [s for s in steps if s not in ("doctor", "invalid", "exports", "serve")] + \
              [m for m in modes if m not in MODES]
    if unknown:
        sys.exit(f"unknown steps or modes: {unknown}")
    if args.out.exists():
        sys.exit(f"output already exists: {args.out}")
    for p in (args.base, args.candidate):
        if not os.access(p, os.X_OK):
            sys.exit(f"not an executable: {p}")
    for f in ("bench.jsonl", "systemone.jsonl"):
        if not (args.inputs / f).is_file():
            sys.exit(f"missing {args.inputs / f}")
    if "serve" in steps and not args.long_requests.is_file():
        sys.exit(f"missing {args.long_requests}")

    root = Path.cwd()
    try:
        known = known_revisions(root)
        models = [resolve_model(s, known, args.hf_home) for s in args.models.split()]
        gpus = smi(["--query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit",
                    "--format=csv,noheader,nounits"])
    except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        sys.exit(f"preflight: {type(e).__name__}: {e}")
    if len(models) < 2 and ({"doctor", "invalid", "serve"} & set(steps)):
        sys.exit("--models needs two models for doctor, invalid and serve")
    if len(gpus) < 2:
        sys.exit(f"{len(gpus)} GPU(s); this run needs two")
    if len({g[1] for g in gpus}) != 1:
        sys.exit(f"GPUs differ ({sorted({g[1] for g in gpus})}); the bundled GEMM tables and the device 1 parity "
                 "need identical cards")
    slug = re.sub(r"[^a-z0-9]+", "-", gpus[0][1].lower()).rstrip("-")
    tables = {}
    for m in models:
        t = root / "crates/basal-cli/gemm-tables" / f"{slug}--{m['slug']}--f16--cublaslt{args.cublaslt}.json"
        if not t.is_file():
            sys.exit(f"missing bundled GEMM table {t} (no GEMM search is run)")
        tables[m["short"]] = t

    args.out.mkdir(parents=True)
    out = args.out.resolve()
    rec = Record(out)
    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w\n" +
                                  "".join(",".join(g) + "\n" for g in gpus))
    (out / "binaries.sha256").write_text(f"{sha256(args.base)}  base\n{sha256(args.candidate)}  candidate\n")
    env = environment(args)
    for label, b in (("base", args.base), ("candidate", args.candidate)):
        r = subprocess.run([b, "--version"], capture_output=True, text=True, env=env, timeout=60)
        (out / f"{label}-version.txt").write_text(r.stdout + r.stderr)
    write_json(out / "run.json", {
        "models": [{"short": m["short"], "name": m["name"], "snapshot": m["dir"].name} for m in models],
        "gemm_tables": {k: {"file": v.name, "sha256": sha256(v)} for k, v in tables.items()},
        "inputs": {f: sha256(args.inputs / f) for f in ("bench.jsonl", "systemone.jsonl")},
        "modes": modes, "steps": steps, "long_tokens": args.long_tokens, "clients": args.clients,
        "rounds": args.rounds, "env": {k: env[k] for k in ("HF_HOME", "BASAL_HOME", "BASAL_NO_UPDATE_CHECK",
                                                          "CUDA_DEVICE_ORDER")},
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args, ctx.out, ctx.env, ctx.models, ctx.tables, ctx.modes = args, out, env, models, tables, modes
    ctx.base, ctx.candidate = args.base.resolve(), args.candidate.resolve()
    ctx.inputs, ctx.long_file = args.inputs.resolve(), args.long_requests.resolve()
    ctx.port, ctx.long_tokens = args.port, args.long_tokens
    ctx.by_bus = {bus(g[2]): int(g[0]) for g in gpus}
    ctx.physical = [int(g[0]) for g in gpus]

    runners = {"doctor": step_doctor, "invalid": step_invalid, "exports": step_exports, "serve": step_serve}
    for s in steps:
        print(f"{time.strftime('%H:%M:%S')} step {s}", flush=True)
        try:
            runners[s](rec, ctx)
        except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
            rec.check(f"step {s}", "FAIL", f"{type(e).__name__}: {e}")
    counts = {k: sum(c["status"] == k for c in rec.checks) for k in ("PASS", "FAIL", "GAP")}
    write_json(out / "summary.json", {"counts": counts, "checks": rec.checks,
                                      "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    print(f"DONE {counts}", flush=True)
    return 1 if counts["FAIL"] else 0


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    try:
        sys.exit(main())
    finally:
        for p in list(LIVE):
            stop(p, timeout=30)
