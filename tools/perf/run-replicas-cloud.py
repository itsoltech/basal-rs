#!/usr/bin/env python3
"""Manual evidence run for replicas of one model (`replica_devices`) on a rented Linux host with exactly two identical
NVIDIA GPUs. Not a test and not a CI gate: it collects raw evidence for a report the operator writes.

  cd /work && python3 tools/perf/run-replicas-cloud.py \
      --base /opt/basal-dev/basal-cuda-placement --expect-base-sha256 SHA \
      --candidate /opt/basal-dev/basal-cuda --expect-candidate-sha256 SHA --out out/replicas
  python3 tools/perf/run-replicas-cloud.py --plan-only        # phases, schedule sizes and hashes; no GPU, no binary

Prerequisites (synced to /work, prefetched to /data): this file, tools/perf/run-isolation-cloud.py and
tools/perf/run-multigpu-cloud.py (their process, nvidia-smi, model, HTTP, schedule, telemetry and statistics helpers
are reused unchanged), tools/bench/long_states.jsonl, reports/reference-basal-1.5-4.5B-fp32 (bench.jsonl and
systemone.jsonl: the export and HTTP inputs), reports/reference-basal-1.5-mini-fp32/systemone.jsonl (performance
requests), crates/basal-cli/gemm-tables (bundled f16 batch-invariant tables of this card; no GEMM search is run), the
Rust sources (crates/, Cargo.toml, Cargo.lock: read and hashed only, for the fingerprint), the weights of mini and
4.5B in HF_HOME=/data/hf at the revisions pinned in crates/basal-cli/src/config.rs, the CUDA libraries of `basal setup`
in BASAL_HOME=/data/basal, and both binaries: --base, the archived placement build (`device` per model, no replicas),
and --candidate, the replica build. A real run refuses binaries whose SHA-256 differs from --expect-*-sha256.
CUDA_DEVICE_ORDER=PCI_BUS_ID is set for every process, so CUDA ordinals (without CUDA_VISIBLE_DEVICES) are the
nvidia-smi indices.

Campaign `correctness` (steps run in this order; a failure is recorded and the next independent step runs):
  doctor    candidate `doctor --json` with `replica_devices: [0, 1]`: no model failure and the group reported; with
            CUDA_VISIBLE_DEVICES=1 device 1 must fail
  invalid   candidate `serve` that must exit before serving and before loading weights: empty list, duplicate
            ordinal, `device: 0` with replica_devices, ordinal 2, `gemm_table: none` with two replicas, the same model
            twice (one entry with replicas), --replica-devices with --config, --replica-devices with --device, an
            ambient configuration (BASAL_CONFIG) with --replica-devices, [0, 1] under CUDA_VISIBLE_DEVICES=1 (ordinal 2
            does not exist because the host must have exactly two GPUs). A mixed
            GPU group cannot be built from identical cards: GAP (static review only)
  exports   per model (--models) and batching mode (--modes): base and candidate `export --device 0` and `--device 1`
            (physical GPU from nvidia-smi); candidate files must be byte-equal to the base file of the same card and
            the card-1 candidate to the card-0 base; `basal compare` must report no difference. Every export waits at
            most --export-timeout and never past the run deadline; an export past --export-timeout is a FAIL that stops
            the run (suspected hung GPU: no further export, server or case), one stopped at the deadline is a GAP
  serve     mini (first of --models) servers, each fresh, /metrics on:
              ref-base-dev0   base, `device: 0`: the HTTP reference (status and canonical body without usage.latency_ms)
                              of every bench item, System One case (multi, evidence, expected 400/422), long document
                              and malformed request of both endpoints, plus raw /health, /v1/models and metric HELP/TYPE
              base-dev1       base, `device: 1`: answers equal to the reference (cross-card on the base)
              cand-dev0       candidate, `device: 0`: answers, /health and /v1/models bytes, metric families unchanged
              cand-one-dev1   candidate, `replica_devices: [1]` (one element: single-instance path): as cand-dev0
              replicas        candidate, `replica_devices: [0, 1]`: one model in /health and /v1/models (bytes equal
                              to the base), memory on both GPUs, replica metrics; sequential and concurrent answers
                              equal to the reference; routed counters of both replicas > 0 and summing to the
                              requests sent; nvidia-smi activity on both GPUs; main and long lanes; reservations back
                              to 0 after the traffic
              pressure        replicas with max_inflight --pressure-inflight: --pressure-requests long requests at once;
                              every status counted, only 200 (equal to the reference) and 529 (TypeSafe body,
                              Retry-After: 1) allowed, at least one of each; reservations back to 0; later requests ok
              cancel          replicas: --cancel-requests long requests whose clients disconnect after --cancel-after s
                              while queued or running; an answer received before the disconnect must equal the
                              reference (status and canonical body) and any other transport error (reset, refused,
                              closed without an answer) is a FAIL that stops the server; reservations seen after the
                              disconnect (GAP when everything finished first), back to 0 within --cancel-drain-timeout;
                              later requests ok
              visible-10      replicas under CUDA_VISIBLE_DEVICES=1,0: memory on both GPUs, answers equal
              visible-1-one   `replica_devices: [0]` under CUDA_VISIBLE_DEVICES=1: physical GPU 1 only, answers equal
              shutdown        replicas: --shutdown-requests long requests in flight on both GPUs, then SIGTERM after
                              --inflight-delay s (default 0.05; reservations on both replicas at SIGTERM, otherwise a
                              GAP): exit 0, every request answered 200 with the reference answer, no GPU memory of the
                              process left
            The long requests of pressure, cancel and shutdown are --long-ids (default long-4000-q1 long-4000-q5). Each
            must be answered 200 in the HTTP reference of the served model (its context limit decides), otherwise the
            reference check fails; nothing is selected automatically. The number in an id is the nominal size of
            make_long_states.py, not the token count the model sees: on mini, long-8000 and long-16000 are answered 422
            and stay in the HTTP reference as expected 422 cases. Custom --models or --long-ids must name requests the
            first model answers 200.
            A replica made unavailable (worker panic, device loss) and a start that fails after a replica loaded are
            not provoked (no fault injection in production code): GAP, static review only.

Campaign `performance` (cases one after another, never overlapping; each a fresh candidate server; order A B B A per
--repeats): A `one`: mini with `device: 0`; B `two`: mini with `replica_devices: [0, 1]`. Same payloads (the
--perf-requests answered 200 in the first warm-up) and schedules in every case. Every case: warm-up (each request
once in order, then twice with 16 clients; every answer equal to the reference of the first warm-up), then per
--closed `CLIENTS:REQUESTS` a closed loop and per --rates an open loop (Poisson, fixed seed, --open-seconds; latency
from the intended arrival, client lag and backlog included, at most --max-outstanding on the wire, nothing dropped;
an open loop must drain within --drain-timeout). A phase is a valid observation only when every scheduled request
was answered 200 with the reference answer, the open loop drained, /metrics, timing headers and telemetry of the
expected GPUs are complete (iso.judge_phase), the replica routed counters sum to the scheduled requests, every replica
of B received requests and was active in nvidia-smi, the reservations returned to 0, and the CPU observation is
complete (wall time and CPU count > 0, finite non-negative CPU seconds per wall second of server and client, host busy
share in [0, 1]; otherwise a GAP). Per phase: p50/p95/p99 of
latency, server queue/compute headers, generator lag, throughput, all statuses, per-replica routed requests, batches
and mean batch size (main/long), per-GPU utilization/memory/power/clocks/temperature, CPU time / wall of the server
process and of this client process, and the host CPU busy share (/proc). Only valid phases enter the A/B medians.

Exit status: 0 all PASS, 1 a FAIL, 3 no FAIL but a GAP (missing evidence), 2 preflight or usage error. A transport
error stops its phase and, in the correctness campaign, its server; every subprocess and HTTP call is bounded; no step
retries a hung GPU. --deadline-minutes stops starting new steps, exports and cases and caps the export waits (the rest
is a GAP: not run). A run stopped early (a fatal error, an unexpected exception, SIGTERM, SIGHUP or Ctrl-C) closes its
processes, writes every record already answered once to requests.jsonl (also when stopping a server fails; signals
after the first are ignored), and writes summary.json with complete false, the reason (`stopped`), a FAIL `run` check
and a GAP for every step, case and coverage not run; after a signal the process exits with 128 + signal, after an
unexpected exception with Python's status, never 0. complete is true only when every requested step and case ran.

Recorded under --out (must not exist; raw, contains local paths and PCI topology, keep outside Git): checks.txt,
summary.json (portable: fingerprint without paths or bus IDs, parameters, schedule hashes, per case and phase results,
A/B medians; local paths and PCI IDs in check details are replaced by <local> with a pointer to checks.txt; review it
before copying it into the repository), run.json, commands.jsonl, gpus.csv, source-files.sha256, per step/case
configs, server logs, requests.jsonl, /metrics snapshots, telemetry.csv, exports and compare outputs.

Limitations: one host and one card type (no mixed group, no cross-architecture claim); the client runs on the same
host (its CPU time and lag are recorded, not removed); nvidia-smi utilization is a coarse busy share; header
intervals are server batch measurements, not kernel time; the routing balance under concurrency is approximate by
design. The run makes no claim of a 2x speed-up or of hardware failure isolation; the operator interprets the data.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import importlib.util
import json
import math
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
ISOLATION = HERE.with_name("run-isolation-cloud.py")


def load_isolation():
    """The committed isolation runner as a module; its `mg` is the committed placement runner, so both share one
    registry of spawned processes (LIVE)."""
    spec = importlib.util.spec_from_file_location("basal_isolation_runner", ISOLATION)
    if spec is None or not ISOLATION.is_file():
        sys.exit(f"missing {ISOLATION} (sync tools/perf)")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


iso = load_isolation()
mg = iso.mg
StepFailed = iso.StepFailed
now = iso.now

MODES = mg.MODES
SOURCES = ("Cargo.toml", "Cargo.lock", "crates/*/Cargo.toml", "crates/*/build.rs", "crates/*/src/**/*.rs",
           "crates/*/src/**/*.cu", "crates/*/src/**/*.metal", "crates/*/src/**/*.wgsl")
OVERLOADED = {"detail": {"error_type": "overloaded_error",
                         "message": "Too many requests in flight. Retry after a short delay."}}


class Deadline(Exception):
    """The run deadline passed: no new step or case starts."""


# ---------------------------------------------------------------- small helpers


def past_deadline(ctx):
    return time.monotonic() >= ctx.deadline


def guard(ctx, rec, what):
    if past_deadline(ctx):
        ctx.deadline_hit = True
        rec.check(what, "GAP", f"not run: --deadline-minutes {ctx.args.deadline_minutes} passed")
        raise Deadline(what)


def source_files(root):
    files = set()
    for pattern in SOURCES:
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def source_digest(root, out):
    """SHA-256 of every source file (read only) into source-files.sha256 and one digest over that list."""
    lines = [f"{mg.sha256(p)}  {p.relative_to(root)}" for p in source_files(root)]
    (out / "source-files.sha256").write_text("".join(x + "\n" for x in lines))
    return {"files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest() if lines else None}


def yaml_config(path, port, cache, models, max_inflight=4096):
    """Serve configuration: strings and lists as JSON (valid YAML flow scalars and sequences)."""
    lines = [f"addr: 127.0.0.1:{port}", "metrics: true", f"max_inflight: {max_inflight}",
             f"gemm_cache: {json.dumps(str(cache))}", "models:"]
    for m in models:
        lines.append(f"  - path: {json.dumps(str(m['dir']))}")
        for k in ("gemm_table", "dtype", "device", "replica_devices", "long_tokens"):
            if k in m:
                v = str(m[k]) if isinstance(m[k], Path) else m[k]
                lines.append(f"    {k}: {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def entry(ctx, m, **placement):
    e = {"dir": m["dir"], "gemm_table": ctx.tables[m["short"]], "dtype": "f16", "long_tokens": ctx.args.long_tokens}
    e.update({k: v for k, v in placement.items() if v is not None})
    return e


SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)')


def parse_metrics(text):
    """[(name, labels, value)] of a Prometheus text exposition."""
    out = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        mt = SAMPLE.match(line)
        if not mt:
            continue
        try:
            value = float(mt.group(3))
        except ValueError:
            continue
        out.append((mt.group(1), dict(re.findall(r'(\w+)="([^"]*)"', mt.group(2) or "")), value))
    return out


def total(samples, name, **labels):
    """Sum of the series of `name` whose labels include `labels` (None when there is none)."""
    xs = [v for n, ls, v in samples if n == name and all(ls.get(k) == str(x) for k, x in labels.items())]
    return sum(xs) if xs else None


def scrape(url, path=None):
    """GET /metrics: (status, parsed samples); the body is written to `path`."""
    try:
        status, _, raw = mg.http("GET", url + "/metrics", timeout=30)
    except (OSError, http.client.HTTPException) as e:
        if path:
            path.write_text(f"# {type(e).__name__}: {e}\n")
        return None, []
    if path:
        path.write_bytes(raw)
    return status, parse_metrics(raw.decode(errors="replace"))


def replica_stats(samples, model, replicas):
    """Per replica index: routed, reserved (inflight), available, batches and batched requests per lane."""
    out = {}
    for r in range(replicas):
        x = {k: total(samples, f"basal_replica_{k}", model=model, replica=r)
             for k in ("routed_requests_total", "inflight_requests", "available")}
        for lane in ("main", "long"):
            x[f"batches_{lane}"] = total(samples, "basal_replica_batch_requests_count", model=model, replica=r, lane=lane)
            x[f"batched_{lane}"] = total(samples, "basal_replica_batch_requests_sum", model=model, replica=r, lane=lane)
        out[str(r)] = x
    return out


def delta(after, before):
    out = {}
    for r, a in after.items():
        b = before.get(r, {})
        out[r] = {k: (None if a.get(k) is None else a[k] - (b.get(k) or 0))
                  for k in ("routed_requests_total", "batches_main", "batched_main", "batches_long", "batched_long")}
        for lane in ("main", "long"):
            n = out[r][f"batches_{lane}"]
            out[r][f"mean_batch_{lane}"] = round(out[r][f"batched_{lane}"] / n, 3) if n else None
        out[r]["inflight_after"] = a.get("inflight_requests")
    return out


def wait_released(url, model, replicas, timeout):
    """Polls /metrics until every replica of `model` has no reservation; (released, seconds, last stats)."""
    t0 = time.monotonic()
    last = None
    while True:
        status, samples = scrape(url)
        if status == 200:
            last = replica_stats(samples, model, replicas)
            if all(x["inflight_requests"] == 0 for x in last.values()):
                return True, round(time.monotonic() - t0, 3), last
        if time.monotonic() - t0 > timeout:
            return False, round(time.monotonic() - t0, 3), last
        time.sleep(0.25)


# ---------------------------------------------------------------- CPU


def proc_cpu_s(pid):
    """User + system CPU seconds of process `pid` (all its threads), None when unreadable."""
    try:
        rest = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(rest[11]) + int(rest[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def host_cpu():
    """(busy, total) jiffies of all CPUs from /proc/stat, None when unreadable."""
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        idle = f[3] + (f[4] if len(f) > 4 else 0)
        return sum(f) - idle, sum(f)
    except (OSError, IndexError, ValueError):
        return None


def cpu_snapshot(pid):
    return {"t": time.monotonic(), "server": proc_cpu_s(pid), "client": time.process_time(), "host": host_cpu()}


def cpu_usage(a, b):
    """CPU seconds per wall second of the server and client processes, host busy share, between two snapshots."""
    wall = b["t"] - a["t"]
    out = {"wall_s": round(wall, 3)}
    for k in ("server", "client"):
        out[f"{k}_cpu_per_wall"] = round((b[k] - a[k]) / wall, 3) if None not in (a[k], b[k]) and wall > 0 else None
    if a["host"] and b["host"] and b["host"][1] > a["host"][1]:
        out["host_busy_share"] = round((b["host"][0] - a["host"][0]) / (b["host"][1] - a["host"][1]), 4)
    out["cpu_count"] = os.cpu_count()
    return out


def cpu_problems(cpu):
    """Why a phase's CPU observation (cpu_usage) is incomplete; [] when every required value is present and sane.
    0 CPU seconds are legitimate; nothing is estimated in place of a missing value."""
    if not isinstance(cpu, dict):
        return ["no CPU observation"]

    def number(k):
        v = cpu.get(k)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None

    out = []
    for k, ok in (("wall_s", lambda v: v > 0), ("cpu_count", lambda v: v > 0),
                  ("server_cpu_per_wall", lambda v: v >= 0), ("client_cpu_per_wall", lambda v: v >= 0),
                  ("host_busy_share", lambda v: 0 <= v <= 1)):
        v = number(k)
        if v is None or not ok(v):
            out.append(f"{k} {cpu.get(k)!r}")
    return out


def judge_cpu(rec, name, phase, cpu):
    """GAP (and an invalid phase) when the CPU observation of the phase is incomplete; True when it is complete."""
    problems = cpu_problems(cpu)
    if problems:
        rec.check(f"{name} {phase}: CPU observation (server, client, host)", "GAP",
                  "missing or invalid: " + ", ".join(problems))
    return not problems


# ---------------------------------------------------------------- HTTP


def send_raw(url, path, raw, timeout):
    """One POST of `raw` bytes: status, canonical answer and its hash, Retry-After, timing headers, shared-clock
    times. A transport error has status None."""
    r = {"t_start": now()}
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request("POST", path, body=raw, headers={"content-type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        h = {k.lower(): v for k, v in resp.getheaders()}
        answer = mg.normalized(body)
        r.update(status=resp.status, answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                 retry_after=h.get("retry-after"), queue_ms=iso.header_number(h, "x-basal-queue-ms", float),
                 compute_ms=iso.header_number(h, "x-basal-compute-ms", float),
                 batch=iso.header_number(h, "x-basal-batch-requests", int))
    except (OSError, http.client.HTTPException, ValueError) as e:
        r.update(status=None, error=f"{type(e).__name__}: {e}")
    finally:
        conn.close()
    r["t_done"] = now()
    return r


def http_items(ctx, model_name):
    """(id, class, path, raw) of every bench item, System One case and long document (model field set), plus
    malformed requests and a /v1/basal request without "model" (the default model)."""
    items = [(rid, cls, "/v1/systemone", json.dumps(body).encode())
             for rid, cls, body in mg.workload(ctx.inputs, ctx.long_file, model_name)]
    q = {"q": {"type": "noul", "instructions": "Czy klient prosi o zwrot?"}}
    state = "Klient pisze, że chce oddać uszkodzony telefon i odzyskać pieniądze."
    items += [
        ("err-systemone-json", "error", "/v1/systemone", b'{"model": '),
        ("err-systemone-unknown-model", "error", "/v1/systemone",
         json.dumps({"model": "no-such-model", "state": state, "questions": q}).encode()),
        ("err-basal-json", "error", "/v1/basal", b"{"),
        ("err-basal-non-object", "error", "/v1/basal", b"[1, 2]"),
        ("basal-default-model", "short", "/v1/basal", json.dumps({"state": state, "questions": q}).encode()),
    ]
    return items


def run_items(url, items, clients, timeout, phase, out):
    """Every item once (clients 1: in order); the record of item k goes to out[k] as soon as it is answered (`out` is
    pre-sized and owned by the caller, so an interruption keeps the records already answered). The first transport
    error stops sending (no retry; the items not sent stay None). Returns whether it occurred."""
    stop = threading.Event()

    def one(k):
        if stop.is_set():
            return
        rid, cls, path, raw = items[k]
        r = send_raw(url, path, raw, timeout)
        r.update(phase=phase, id=rid, cls=cls)
        out[k] = r
        if r["status"] is None:
            stop.set()

    if clients == 1:
        for k in range(len(items)):
            one(k)
            if stop.is_set():
                break
        return stop.is_set()
    with concurrent.futures.ThreadPoolExecutor(clients) as pool:
        for _ in pool.map(one, range(len(items))):
            pass
    return stop.is_set()


def parity(rec, name, records, reference, expected):
    """Every record equal (status, canonical body) to the reference and `expected` requests answered."""
    bad = [r["id"] for r in records if r["status"] is None or (r["status"], r["answer"]) != reference.get(r["id"])]
    errors = [f"{r['id']}: {r.get('status') or r.get('error')}" for r in records
              if r["status"] is None or r["status"] >= 500]
    rec.expect(f"{name}: answers equal to the reference", not bad and len(records) == expected,
               f"{len(records)}/{expected} answered, {len(bad)} differing {bad[:5]}")
    rec.expect(f"{name}: no transport error or 5xx", not errors, iso.scrub(", ".join(errors[:5])))
    return not bad


STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def write_records(path, records):
    """requests.jsonl. In the main thread SIGTERM, SIGHUP and SIGINT are held back while writing and raised again after
    it (with the handler restored), so a signal never truncates the file."""
    held = []
    main = threading.current_thread() is threading.main_thread()
    old = {sig: signal.signal(sig, lambda signum, _frame: held.append(signum)) for sig in STOP_SIGNALS} if main else {}
    try:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if held:
            signal.raise_signal(held[0])


def settle_clients(timeout=10):
    """After the server stopped: waits at most `timeout` s in total for the HTTP client threads still running (the
    only non-daemon threads besides the main one; the collectors are daemons), so their records are final before they
    are written. Returns the number still running (their records, if any arrive later, are not written)."""
    deadline = time.monotonic() + timeout
    me = threading.current_thread()
    for t in threading.enumerate():
        if t is not me and not t.daemon:
            t.join(max(0.0, deadline - time.monotonic()))
    return sum(1 for t in threading.enumerate() if t is not me and not t.daemon and t.is_alive())


def close_and_write(rec, name, close, path, records):
    """`close()` (stops the server), then the client threads settle and every record of `records()` is written once,
    even when `close()`, the settling or the GAP check raises or is interrupted."""
    try:
        close()
    finally:
        try:
            left = settle_clients()
            if left:
                rec.check(f"{name}: client threads finished before requests.jsonl was written", "GAP",
                          f"{left} still running after the server stopped; records they add later are not written")
        finally:
            write_records(path, records())


def long_items(ctx, n):
    """`n` long requests (cycled over --long-ids), with their reference answers."""
    base = [it for it in ctx.items if it[0] in {f"long:{i}" for i in ctx.args.long_ids}]
    return [base[k % len(base)] for k in range(n)] if base else []


# ---------------------------------------------------------------- correctness: doctor and invalid configurations


def doctor_report(text):
    report, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
    return report


def step_doctor(rec, ctx):
    d = ctx.out / "correctness" / "doctor"
    d.mkdir(parents=True)
    cfg = yaml_config(d / "replicas.yml", ctx.args.port, d / "gemm-cache", [entry(ctx, ctx.mini, replica_devices=[0, 1])])
    for label, visible in (("default", None), ("only1", "1")):
        delta_env = {"CUDA_VISIBLE_DEVICES": visible} if visible else {}
        log = d / f"{label}.json"
        code, _ = mg.run_cmd(rec, f"doctor-{label}", [ctx.candidate, "--color", "never", "doctor", "--config", cfg,
                                                      "--json"], mg.with_env(ctx.env, delta_env), log, 180, delta_env)
        try:
            report = doctor_report(log.read_text(errors="replace"))
        except ValueError:
            rec.check(f"doctor {label}", "FAIL", f"no JSON report (exit {code})")
            continue
        models = [c for c in report["checks"] if c["area"] == "models"]
        fails = [c["what"] for c in models if c["level"] == "fail"]
        if visible:
            rec.expect(f"doctor {label}: device 1 refused", any("device 1:" in f for f in fails), iso.scrub(f"{fails}"))
        else:
            group = [c["what"] for c in models if "2 replicas on devices [0, 1]" in c["what"]]
            rec.expect(f"doctor {label}: replica group reported without failure", group and not fails,
                       iso.scrub(f"group {group}, failures {fails}"))


def step_invalid(rec, ctx):
    d = ctx.out / "correctness" / "invalid"
    d.mkdir(parents=True)
    m, port, cache = ctx.mini, ctx.args.port, d / "gemm-cache"

    def cfg(label, entries):
        return yaml_config(d / f"{label}.yml", port, cache, entries)

    plain = cfg("plain", [entry(ctx, m)])
    cases = [  # (label, environment delta, arguments after `serve`, expected message)
        ("empty", {}, ["--config", cfg("empty", [entry(ctx, m, replica_devices=[])])], "is empty"),
        ("duplicate-ordinal", {}, ["--config", cfg("duplicate-ordinal", [entry(ctx, m, replica_devices=[0, 0])])],
         "twice"),
        ("device-0-and-replicas", {},
         ["--config", cfg("device-0-and-replicas", [entry(ctx, m, device=0, replica_devices=[0, 1])])], "not both"),
        ("ordinal-2", {}, ["--config", cfg("ordinal-2", [entry(ctx, m, replica_devices=[0, 2])])], "does not exist"),
        ("gemm-none", {}, ["--config", cfg("gemm-none", [dict(entry(ctx, m, replica_devices=[0, 1]),
                                                             gemm_table="none")])], "not `none`"),
        ("same-model-twice", {}, ["--config", cfg("same-model-twice", [entry(ctx, m, replica_devices=[0, 1]),
                                                                       entry(ctx, m, device=1)])], "configured twice"),
        ("cli-with-config", {}, ["--config", plain, "--replica-devices", "0,1"], "cannot be used with"),
        ("cli-with-device", {}, ["--model", m["dir"], "--device", "0", "--replica-devices", "0,1"],
         "cannot be used with"),
        ("ambient-config-with-cli", {"BASAL_CONFIG": str(plain)}, ["--replica-devices", "0,1"],
         "--replica-devices applies to --model"),
        ("visible1-group", {"CUDA_VISIBLE_DEVICES": "1"},
         ["--config", cfg("visible1-group", [entry(ctx, m, replica_devices=[0, 1])])], "does not exist"),
    ]
    for label, delta_env, extra, needle in cases:
        mg.port_free_or_fail(port)
        log = d / f"{label}.log"
        code, dt = mg.run_cmd(rec, f"invalid-{label}", [ctx.candidate, "--color", "never", "serve", *extra],
                              mg.with_env(ctx.env, delta_env), log, 120, delta_env)
        text = log.read_text(errors="replace")
        # refused before any model weights or GEMM search (`loaded on GPU`, `generating the batch-invariant GEMM`)
        early = "loaded on GPU" not in text and "generating the batch-invariant" not in text
        rec.expect(f"invalid {label}: refused before serving and loading",
                   code not in (0, None) and needle in text and "serving on http" not in text and early,
                   f"exit {code} after {dt:.1f} s, expected message {needle!r}, before load {early}")
    rec.check("invalid mixed GPU group (name, compute capability, SM count)", "GAP",
              "identical cards: not provoked; static review of config::check_replica_gpus and doctor only")


# ---------------------------------------------------------------- correctness: exports


def step_exports(rec, ctx):
    """Every export waits at most --export-timeout and never past the run deadline. Stopped at the deadline: a GAP
    and Deadline. Past --export-timeout: a FAIL and Fatal (a suspected hung GPU; no further export, server or case).
    Either way the exports not started and the comparisons not judged are recorded as a GAP."""
    d = ctx.out / "correctness" / "exports"
    d.mkdir(parents=True)
    n_lines = (mg.line_count(ctx.inputs / "bench.jsonl"), mg.line_count(ctx.inputs / "systemone.jsonl"))
    variants = [("base-dev0", ctx.base, 0), ("base-dev1", ctx.base, 1), ("cand-dev0", ctx.candidate, 0),
                ("cand-dev1", ctx.candidate, 1)]
    pairs = [("base-dev0", "cand-dev0"), ("base-dev1", "cand-dev1"), ("base-dev0", "cand-dev1")]
    combos = [(m, mode, f"{m['short']}-{mode}") for m in ctx.models for mode in ctx.modes]
    attempted, judged = set(), set()
    try:
        for m, mode, combo in combos:
            done = {}
            for label, binary, device in variants:
                name = f"{combo}-{label}"
                attempted.add(name)
                guard(ctx, rec, f"export {name}")
                budget = ctx.deadline - time.monotonic()
                if budget <= 0:
                    guard(ctx, rec, f"export {name}")  # the deadline passed since the guard: GAP, Deadline
                out = d / name
                argv = [binary, "--color", "never", "export", "--model", m["dir"], "--gemm-table",
                        ctx.tables[m["short"]], *MODES[mode], "--device", str(device), "--inputs", ctx.inputs,
                        "--out", out]
                t0 = time.time()
                p = mg.spawn(argv, ctx.env, d / f"{name}.log")
                watch = mg.PlacementWatch(p.pid, ctx.by_bus)
                try:
                    try:
                        code = p.wait(min(ctx.args.export_timeout, budget))
                    except subprocess.TimeoutExpired:
                        code = None
                    finally:
                        mg.stop(p)
                finally:
                    peak = watch.close()
                seconds = round(time.time() - t0, 3)
                rec.command({"step": name, "argv": [str(a) for a in argv], "exit": code, "seconds": seconds})
                if code is None and budget < ctx.args.export_timeout:
                    ctx.deadline_hit = True
                    rec.check(f"export {name}: completed", "GAP", f"stopped after {seconds:.0f} s at "
                              f"--deadline-minutes {ctx.args.deadline_minutes} (incomplete, not judged)")
                    raise Deadline(name)
                if code is None:
                    rec.check(f"export {name}: completed", "FAIL",
                              f"no exit within --export-timeout {ctx.args.export_timeout} s (stopped)")
                    raise iso.Fatal(f"export {name} exceeded --export-timeout {ctx.args.export_timeout} s: "
                                    "suspected hung GPU, no further export, server or case")
                if not rec.expect(f"export {name}: completed", code == 0, f"exit {code}"):
                    continue
                counts = (mg.line_count(out / "bench.jsonl"), mg.line_count(out / "systemone.jsonl"))
                if not rec.expect(f"export {name}: all inputs exported", counts == n_lines,
                                  f"{counts} lines, inputs {n_lines}"):
                    continue
                mg.placement_check(rec, f"export {name}: on physical GPU {device}", peak, {device})
                done[label] = out
            judged.add(combo)
            for a, b in pairs:
                pair = f"{combo} {a} vs {b}"
                if a not in done or b not in done:
                    rec.check(f"bitwise {pair}", "FAIL", "export missing")
                    continue
                hashes = {f: (mg.sha256(done[a] / f), mg.sha256(done[b] / f)) for f in ("bench.jsonl", "systemone.jsonl")}
                mg.write_json(d / f"{combo}-{a}-{b}.sha256.json", hashes)
                rec.expect(f"bitwise {pair}", all(x == y for x, y in hashes.values()),
                           ", ".join(f"{f} {'equal' if x == y else 'DIFFERENT'}" for f, (x, y) in hashes.items()))
                cmp_out = d / f"compare-{combo}-{a}-{b}.json"
                code, _ = mg.run_cmd(rec, f"compare-{pair}", [ctx.candidate, "--color", "never", "compare", "--a",
                                                              done[a], "--b", done[b], "--out", cmp_out],
                                     ctx.env, d / f"compare-{combo}-{a}-{b}.log", 600, {})
                if code != 0 or not cmp_out.is_file():
                    rec.check(f"basal compare {pair}", "FAIL", f"exit {code}")
                    continue
                problems = mg.compare_problems(json.loads(cmp_out.read_text()))
                rec.expect(f"basal compare {pair}: no difference", not problems, "; ".join(problems[:5]))
    except (Deadline, iso.Fatal):
        planned = [f"{combo}-{label}" for _, _, combo in combos for label, _, _ in variants]
        not_run = [x for x in planned if x not in attempted]
        if not_run:
            rec.check("exports not run", "GAP", f"{len(not_run)} of {len(planned)}: {', '.join(not_run)}")
        unjudged = [combo for _, _, combo in combos if combo not in judged]
        if unjudged:
            rec.check("bitwise and basal compare not judged", "GAP", ", ".join(unjudged))
        raise


# ---------------------------------------------------------------- correctness: serve


class Started:
    """A fresh server of the correctness campaign: config, iso.Server (telemetry, log), readiness, placement."""

    def __init__(self, rec, ctx, label, binary, models, delta_env=None, max_inflight=4096):
        guard(ctx, rec, f"serve {label}")
        self.rec, self.ctx, self.label = rec, ctx, label
        self.dir = ctx.out / "correctness" / "serve" / label
        self.dir.mkdir(parents=True)
        busy = iso.wait_gpus_idle()
        if busy:
            raise StepFailed(f"compute processes still on the GPUs before {label}: {len(busy)}")
        cfg = yaml_config(self.dir / "serve.yml", ctx.args.port, self.dir / "gemm-cache", models, max_inflight)
        self.srv = iso.Server(rec, ctx, self.dir, binary, cfg, delta_env or {})
        self.url = self.srv.url
        self.batches, self.closed = [], False  # per phase the records, filled while it runs; written once by close
        try:
            self.srv.wait_ready()
            time.sleep(2)
            self.placement = mg.session_placement(self.srv.proc.pid, ctx.by_bus)
            _, _, self.health_raw = mg.http("GET", self.url + "/health", timeout=30)
            _, _, self.models_raw = mg.http("GET", self.url + "/v1/models", timeout=30)
        except BaseException:
            self.close()
            raise

    def expect_placement(self, physical):
        mg.placement_check(self.rec, f"serve {self.label}: memory only on physical GPUs {sorted(physical)}",
                           self.placement, set(physical))

    def run(self, items, clients, phase):
        """The records of `items`; a transport error keeps the records, is a FAIL and raises StepFailed, so the
        caller's `finally` closes this server and no later phase runs against it."""
        slots = [None] * len(items)
        self.batches.append(slots)
        aborted = run_items(self.url, items, clients, self.ctx.args.http_timeout, phase, slots)
        records = [r for r in slots if r]
        if aborted:
            failed = [r for r in records if r["status"] is None]
            self.rec.check(f"serve {self.label} {phase}: transport", "FAIL",
                           iso.scrub(f"{len(records)} of {len(items)} requests sent ({len(failed)} without an "
                                     f"answer, the rest not sent), first error {failed[0]['id']}: "
                                     f"{failed[0].get('error')}; phase and server stopped"))
            raise StepFailed(f"{phase}: transport error, server {self.label} stopped")
        return records

    def shutdown_check(self):
        code = self.srv.shutdown()
        log = (self.dir / "server.log").read_text(errors="replace")
        self.rec.expect(f"serve {self.label}: graceful shutdown", code == 0 and "shutting down" in log, f"exit {code}")
        after = mg.session_placement(self.srv.proc.pid, self.ctx.by_bus)
        self.rec.check(f"serve {self.label}: no GPU memory left after exit",
                       "FAIL" if after else "PASS" if after == {} and self.placement else "GAP", f"{after}")

    def close(self):
        """Stops the server, then writes every record of its phases once (also when stopping raises or a signal
        arrives; see close_and_write)."""
        if self.closed:
            return
        self.closed = True
        close_and_write(self.rec, f"serve {self.label}", self.srv.close, self.dir / "requests.jsonl",
                        lambda: [r for batch in self.batches for r in batch if r])


def metric_families(text):
    return sorted({line for line in text.splitlines() if line.startswith(("# HELP ", "# TYPE "))})


def check_group_metrics(rec, label, samples, name, devices):
    ok = total(samples, "basal_model_replicas", model=name) == len(devices) and \
        total(samples, "basal_model_replicas_available", model=name) == len(devices)
    for r, dev in enumerate(devices):
        ok &= total(samples, "basal_replica_info", model=name, replica=r, device=dev) == 1
        ok &= total(samples, "basal_replica_available", model=name, replica=r) == 1
    rec.expect(f"serve {label}: replica metrics (count, info with device, available)", ok, f"devices {devices}")


def single_instance_server(rec, ctx, label, binary, placement, physical, delta_env=None):
    s = Started(rec, ctx, label, binary, [entry(ctx, ctx.mini, **placement)], delta_env)
    try:
        s.expect_placement({physical})
        recs = s.run(ctx.items, 1, "sequential")
        parity(rec, f"serve {label} sequential", recs, ctx.http_ref, len(ctx.items))
        if binary == ctx.candidate:
            rec.expect(f"serve {label}: /health bytes equal to the base", s.health_raw == ctx.ref_health,
                       s.health_raw.decode(errors="replace")[:200])
            rec.expect(f"serve {label}: /v1/models bytes equal to the base", s.models_raw == ctx.ref_models)
            status, samples = scrape(s.url, s.dir / "metrics.txt")
            text = (s.dir / "metrics.txt").read_text(errors="replace")
            missing = [f for f in ctx.ref_families if f not in metric_families(text)]
            rec.expect(f"serve {label}: every base metric family with the same HELP and TYPE", status == 200 and
                       not missing, f"missing or changed {missing[:3]}")
            check_group_metrics(rec, label, samples, ctx.mini["name"], [placement.get("device", physical)]
                                if "device" in placement else placement["replica_devices"])
        s.shutdown_check()
    finally:
        s.close()


def step_serve(rec, ctx):
    (ctx.out / "correctness" / "serve").mkdir(parents=True)
    name = ctx.mini["name"]
    ctx.items = http_items(ctx, name)
    rec.check("serve replica unavailable (worker panic or reported device loss), partial start after a load", "GAP",
              "not provoked (no fault injection); static review of serve.rs readiness and lifecycle only")
    singles = (("base-dev1", ctx.base, {"device": 1}, 1, None),
               ("cand-dev0", ctx.candidate, {"device": 0}, 0, None),
               ("cand-one-dev1", ctx.candidate, {"replica_devices": [1]}, 1, None),
               ("visible-1-one", ctx.candidate, {"replica_devices": [0]}, 1, {"CUDA_VISIBLE_DEVICES": "1"}))
    groups = (("replicas", serve_replicas), ("pressure", serve_pressure), ("cancel", serve_cancel),
              ("visible-10", serve_visible), ("shutdown", serve_shutdown))
    attempted = ["ref-base-dev0"]
    try:
        # reference: the base binary, one instance on device 0
        s = Started(rec, ctx, "ref-base-dev0", ctx.base, [entry(ctx, ctx.mini, device=0)])
        try:
            s.expect_placement({0})
            recs = s.run(ctx.items, 1, "sequential")
            errors = [r["id"] for r in recs if r["status"] is None or r["status"] >= 500]
            ok = rec.expect("serve ref-base-dev0: reference answers", len(recs) == len(ctx.items) and not errors,
                            f"{Counter(str(r['status']) for r in recs)}, transport/5xx {errors[:5]}")
            ctx.http_ref = {r["id"]: (r["status"], r["answer"]) for r in recs}
            ctx.ref_health, ctx.ref_models = s.health_raw, s.models_raw
            scrape(s.url, s.dir / "metrics.txt")
            ctx.ref_families = metric_families((s.dir / "metrics.txt").read_text(errors="replace"))
            mg.write_json(ctx.out / "correctness" / "serve" / "reference-statuses.json",
                          {k: v[0] for k, v in ctx.http_ref.items()})
            long_ok = [i for i in ctx.args.long_ids if ctx.http_ref.get(f"long:{i}", (None,))[0] == 200]
            rec.expect("serve reference: --long-ids answered 200", len(long_ok) == len(ctx.args.long_ids),
                       f"200: {long_ok} of {ctx.args.long_ids}")
            s.shutdown_check()
        finally:
            s.close()
        if not ok:
            raise StepFailed("no HTTP reference")

        for label, binary, placement, physical, delta_env in singles:
            attempted.append(label)
            try:
                single_instance_server(rec, ctx, label, binary, placement, physical, delta_env)
            except Deadline:
                raise
            except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
                rec.check(f"serve {label}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))

        for label, fn in groups:
            attempted.append(label)
            try:
                fn(rec, ctx, label)
            except Deadline:
                raise
            except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
                rec.check(f"serve {label}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
    except (Deadline, StepFailed) as e:
        not_run = [x[0] for x in (*singles, *groups) if x[0] not in attempted]
        if not_run:
            rec.check("serve servers not run", "GAP", f"{', '.join(not_run)} ({type(e).__name__}: "
                      f"{iso.scrub(str(e))})")
        raise


def replicas_server(rec, ctx, label, delta_env=None, max_inflight=4096):
    return Started(rec, ctx, label, ctx.candidate, [entry(ctx, ctx.mini, replica_devices=[0, 1])], delta_env,
                   max_inflight)


def serve_replicas(rec, ctx, label):
    name = ctx.mini["name"]
    s = replicas_server(rec, ctx, label)
    try:
        s.expect_placement({0, 1})
        rec.expect(f"serve {label}: /health bytes equal to the base (one model, no degradation)",
                   s.health_raw == ctx.ref_health, s.health_raw.decode(errors="replace")[:200])
        rec.expect(f"serve {label}: /v1/models bytes equal to the base (one logical model)",
                   s.models_raw == ctx.ref_models, s.models_raw.decode(errors="replace")[:200])
        _, samples = scrape(s.url, s.dir / "metrics-start.txt")
        check_group_metrics(rec, label, samples, name, [0, 1])
        recs = s.run(ctx.items, 1, "sequential")
        parity(rec, f"serve {label} sequential", recs, ctx.http_ref, len(ctx.items))
        work = [it for it in ctx.items if it[2] == "/v1/systemone" and not it[0].startswith("err-")]
        items = work * ctx.args.rounds
        _, before_samples = scrape(s.url, s.dir / "metrics-before-concurrent.txt")
        before = replica_stats(before_samples, name, 2)
        t0 = now()
        recs = s.run(items, ctx.args.clients, "concurrent")
        t1 = now()
        parity(rec, f"serve {label} concurrent ({ctx.args.clients} clients)", recs, ctx.http_ref, len(items))
        released, seconds, _ = wait_released(s.url, name, 2, 30)
        rec.expect(f"serve {label}: reservations back to 0 after the traffic", released, f"after {seconds} s")
        _, after_samples = scrape(s.url, s.dir / "metrics-after-concurrent.txt")
        d = delta(replica_stats(after_samples, name, 2), before)
        mg.write_json(s.dir / "replicas-concurrent.json", d)
        routed = [d[r]["routed_requests_total"] or 0 for r in ("0", "1")]
        rec.expect(f"serve {label}: both replicas routed requests, summing to the requests sent",
                   all(x > 0 for x in routed) and sum(routed) == len(items), f"routed {routed}, sent {len(items)}")
        rec.expect(f"serve {label}: main lane batches on both replicas",
                   all((d[r]["batches_main"] or 0) > 0 for r in ("0", "1")),
                   f"{[(r, d[r]['batches_main'], d[r]['batches_long']) for r in ('0', '1')]}")
        long_batches = sum(d[r]["batches_long"] or 0 for r in ("0", "1"))
        rec.expect(f"serve {label}: long lane used", long_batches > 0, f"long batches per replica "
                   f"{[d[r]['batches_long'] for r in ('0', '1')]}")
        window = s.srv.telemetry.window(t0, t1)
        mg.write_json(s.dir / "telemetry-concurrent.json", window)
        active = {i: (window.get(str(i)) or {}).get("util_max") for i in (0, 1)}
        if any(v is None for v in active.values()):
            rec.check(f"serve {label}: nvidia-smi activity on both GPUs", "GAP", f"utilization max {active}")
        else:
            rec.expect(f"serve {label}: nvidia-smi activity on both GPUs", all(v > 0 for v in active.values()),
                       f"utilization max {active}, memory peak "
                       f"{ {i: (window.get(str(i)) or {}).get('mem_peak_mib') for i in (0, 1)} }")
        s.shutdown_check()
    finally:
        s.close()


def serve_pressure(rec, ctx, label):
    name = ctx.mini["name"]
    items = long_items(ctx, ctx.args.pressure_requests)
    if not items:
        raise StepFailed("no --long-ids request in the workload")
    s = replicas_server(rec, ctx, label, max_inflight=ctx.args.pressure_inflight)
    try:
        recs = s.run(items, len(items), "burst")
        statuses = Counter(str(r["status"]) for r in recs)
        overloaded = json.dumps(OVERLOADED, sort_keys=True, ensure_ascii=False)
        bad = [r["id"] for r in recs if not (
            (r["status"] == 200 and (r["status"], r["answer"]) == ctx.http_ref.get(r["id"])) or
            (r["status"] == 529 and r["answer"] == overloaded and r.get("retry_after") == "1"))]
        rec.expect(f"serve {label}: {len(items)} requests at once with max_inflight {ctx.args.pressure_inflight}: "
                   f"only 200 (reference) and 529 (TypeSafe body, Retry-After 1), both seen",
                   len(recs) == len(items) and not bad and statuses.get("200") and statuses.get("529"),
                   f"statuses {dict(statuses)}, unexpected {bad[:5]}")
        released, seconds, last = wait_released(s.url, name, 2, 60)
        rec.expect(f"serve {label}: reservations back to 0", released, f"after {seconds} s, {last}")
        recs = s.run(items[:4], 1, "after")
        parity(rec, f"serve {label} after the burst", recs, ctx.http_ref, 4)
        scrape(s.url, s.dir / "metrics-end.txt")
        s.shutdown_check()
    finally:
        s.close()


def abandon(url, k, item, after):
    """POST `item`, then disconnect when no answer arrived within `after` s (the socket timeout of each step). The
    record's outcome: "answered" (status and canonical answer as in send_raw), "disconnected" (the planned disconnect;
    `status` is set when the status line arrived but the body did not) or "transport_error" (refused, reset, closed
    without an answer or another error: not planned)."""
    rid, cls, path, raw = item
    r = {"phase": "cancel", "k": k, "id": rid, "cls": cls, "status": None, "t_start": now()}
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=after)
    try:
        conn.request("POST", path, body=raw, headers={"content-type": "application/json"})
        resp = conn.getresponse()
        r["status"] = resp.status
        answer = mg.normalized(resp.read())
        r.update(outcome="answered", answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                 retry_after=resp.getheader("retry-after"))
    except (socket.timeout, TimeoutError):
        r["outcome"] = "disconnected"
    except (OSError, http.client.HTTPException, ValueError) as e:
        r.update(outcome="transport_error", error=f"{type(e).__name__}: {e}")
    finally:
        conn.close()
    r["t_done"] = now()
    return r


def serve_cancel(rec, ctx, label):
    name = ctx.mini["name"]
    items = long_items(ctx, ctx.args.cancel_requests)
    if not items:
        raise StepFailed("no --long-ids request in the workload")
    s = replicas_server(rec, ctx, label)
    try:
        _, before = scrape(s.url, s.dir / "metrics-before.txt")
        cancelled_before = total(before, "basal_http_requests_total", status="cancelled") or 0
        results = [None] * len(items)
        s.batches.append(results)

        def cancel_one(k):
            results[k] = abandon(s.url, k, items[k], ctx.args.cancel_after)

        with concurrent.futures.ThreadPoolExecutor(len(items)) as pool:
            for _ in pool.map(cancel_one, range(len(items))):
                pass
        _, right_after = scrape(s.url, s.dir / "metrics-after-disconnect.txt")
        held = replica_stats(right_after, name, 2)
        reserved = sum(x["inflight_requests"] or 0 for x in held.values())
        by = {o: [r for r in results if r["outcome"] == o] for o in ("answered", "disconnected", "transport_error")}
        cut = [r for r in by["disconnected"] if r["status"] is not None]
        mg.write_json(s.dir / "cancel.json", {"results": results, "reserved_after_disconnect": held})
        rec.check(f"serve {label}: reservations kept after the clients disconnected (queued or running jobs)",
                  "PASS" if reserved > 0 else "GAP",
                  f"{reserved} reserved right after {len(by['disconnected'])} disconnects "
                  f"({len(by['answered'])} answered before --cancel-after); per replica "
                  f"{ {r: x['inflight_requests'] for r, x in held.items()} }")
        wrong = [r["id"] for r in by["answered"] if (r["status"], r["answer"]) != ctx.http_ref.get(r["id"])]
        wrong += [r["id"] for r in cut if r["status"] != 200]
        rec.expect(f"serve {label}: answers received before the disconnect equal to the reference", not wrong,
                   f"{len(by['answered'])} answered {dict(Counter(str(r['status']) for r in by['answered']))}, "
                   f"{len(cut)} disconnected after the status line (body not compared) "
                   f"{dict(Counter(str(r['status']) for r in cut))}, differing or not 200 {wrong[:5]}")
        broken = by["transport_error"]
        rec.expect(f"serve {label}: no transport error other than the planned disconnects", not broken,
                   iso.scrub(f"{len(broken)} of {len(items)}: " + ", ".join(f"{r['id']}: {r['error']}"
                                                                           for r in broken[:5])))
        if broken:
            raise StepFailed(f"cancel: {len(broken)} unexpected transport errors, server {label} stopped")
        released, seconds, last = wait_released(s.url, name, 2, ctx.args.cancel_drain_timeout)
        rec.expect(f"serve {label}: reservations back to 0 after the disconnects", released,
                   f"after {seconds} s, last {last}")
        _, end = scrape(s.url, s.dir / "metrics-released.txt")
        cancelled = (total(end, "basal_http_requests_total", status="cancelled") or 0) - cancelled_before
        rec.check(f"serve {label}: cancelled handlers counted", "PASS" if cancelled > 0 else "GAP",
                  f"basal_http_requests_total cancelled +{cancelled:g} (observation)")
        sample = [it for it in ctx.items if it[1] == "short" and it[2] == "/v1/systemone"][:10]
        recs = s.run(sample, 1, "after")
        parity(rec, f"serve {label} after the disconnects", recs, ctx.http_ref, len(sample))
        s.shutdown_check()
    finally:
        s.close()


def serve_visible(rec, ctx, label):
    s = replicas_server(rec, ctx, label, {"CUDA_VISIBLE_DEVICES": "1,0"})
    try:
        s.expect_placement({0, 1})
        recs = s.run(ctx.items, 1, "sequential")
        parity(rec, f"serve {label} sequential", recs, ctx.http_ref, len(ctx.items))
        s.shutdown_check()
    finally:
        s.close()


def serve_shutdown(rec, ctx, label):
    name = ctx.mini["name"]
    items = long_items(ctx, ctx.args.shutdown_requests)
    if not items:
        raise StepFailed("no --long-ids request in the workload")
    s = replicas_server(rec, ctx, label)
    try:
        s.expect_placement({0, 1})
        results = []  # in the order answered
        s.batches.append(results)
        threads = [threading.Thread(target=lambda it=it, sink=results: sink.append(
            dict(send_raw(s.url, it[2], it[3], ctx.args.http_timeout), id=it[0]))) for it in items]
        for t in threads:
            t.start()
        time.sleep(ctx.args.inflight_delay)
        _, held = scrape(s.url, s.dir / "metrics-at-sigterm.txt")
        at_term = replica_stats(held, name, 2)
        code = s.srv.shutdown()
        for t in threads:
            t.join(ctx.args.http_timeout)
        judged = list(results)  # snapshot; a request answered after the joins still goes to `results` and is written
        log = (s.dir / "server.log").read_text(errors="replace")
        busy = {r: x["inflight_requests"] for r, x in at_term.items()}
        rec.check(f"serve {label}: requests reserved on both replicas at SIGTERM",
                  "PASS" if all((v or 0) > 0 for v in busy.values()) else "GAP", f"reserved {busy}")
        rec.expect(f"serve {label}: exit 0 and shutdown logged after SIGTERM", code == 0 and "shutting down" in log,
                   f"exit {code} (in-flight evidence: the reserved check above)")
        bad = [r["id"] for r in judged if (r["status"], r.get("answer")) != ctx.http_ref.get(r["id"])]
        rec.expect(f"serve {label}: every in-flight request answered with the reference answer",
                   len(judged) == len(items) and not bad,
                   f"{len(judged)}/{len(items)}, statuses {dict(Counter(str(r['status']) for r in judged))}")
        after = mg.session_placement(s.srv.proc.pid, ctx.by_bus)
        rec.check(f"serve {label}: no GPU memory left after exit",
                  "FAIL" if after else "PASS" if after == {} and s.placement else "GAP", f"{after}")
    finally:
        s.close()


# ---------------------------------------------------------------- performance


def perf_plan(args):
    """[(phase, kind, value, requests)] of the measured phases in run order."""
    plan = [(f"closed-c{c}", "closed", c, n) for c, n in args.closed]
    plan += [(f"open-{r:g}rps", "open", r, None) for r in args.rates]
    return plan


def perf_schedules(args, ids):
    out = {}
    for phase, kind, value, n in perf_plan(args):
        if kind == "closed":
            out[phase] = {"kind": kind, "clients": value, "ids": iso.request_order(ids, n, f"{args.seed}-{phase}-order")}
        else:
            offsets = [round(x, 6) for x in iso.arrival_offsets(value, args.open_seconds, f"{args.seed}-{phase}",
                                                                args.arrivals)]
            out[phase] = {"kind": kind, "rate": value, "seconds": args.open_seconds, "arrivals": args.arrivals,
                          "offsets": offsets,
                          "ids": iso.request_order(ids, len(offsets), f"{args.seed}-{phase}-order")}
            out[phase]["arrivals_sha256"] = iso.canonical_sha(offsets)
        if not out[phase]["ids"]:
            raise StepFailed(f"{phase}: empty schedule")
        out[phase]["sha256"] = iso.canonical_sha({k: v for k, v in out[phase].items() if k != "sha256"})
    return out


def perf_specs(args):
    variants = {"A": {"variant": "one", "placement": {0}, "replicas": [0], "config": {"device": 0}},
                "B": {"variant": "two", "placement": {0, 1}, "replicas": [0, 1], "config": {"replica_devices": [0, 1]}}}
    out = []
    for n, ab in enumerate("ABBA" * args.repeats):
        spec = dict(variants[ab], ab=ab, campaign="performance", background=False)
        spec["label"] = f"{n + 1:02d}-{spec['variant']}"
        out.append(spec)
    return out


def perf_warm_up(rec, ctx, srv, records, warm_runs, name):
    """Sequential warm-up records go to `records`, the concurrent run to `warm_runs` (both owned by perf_case, which
    writes them, so an interruption keeps the records already answered)."""
    first = "mini" not in ctx.reference
    try:
        seq = iso.run_sequential(srv, ctx, "mini", ctx.perf_items, records, "warm-seq", first)
    except StepFailed as e:
        if first:
            raise iso.Fatal(f"reference warm-up: {e}") from e
        raise
    if first:
        broken = {x["id"]: x.get("status") for x in seq if x.get("status") is None or x["status"] >= 500}
        if broken:
            rec.check("performance reference: warm-up without transport or server errors", "FAIL", json.dumps(broken))
            raise iso.Fatal("server errors in the reference warm-up")
        ctx.mini_valid = [x["id"] for x in seq if x.get("status") == 200]
        ctx.mini_excluded = {x["id"]: x.get("status") for x in seq if x.get("status") != 200}
        if not rec.expect("performance reference: enough requests answered 200",
                          len(ctx.mini_valid) >= ctx.args.min_perf_requests,
                          f"{len(ctx.mini_valid)} selected, excluded {json.dumps(ctx.mini_excluded)}"):
            raise iso.Fatal("too few requests answered 200 in the reference warm-up")
        ctx.schedules = perf_schedules(ctx.args, ctx.mini_valid)
        mg.write_json(ctx.out / "performance" / "schedules.json", ctx.schedules)
    else:
        bad = [x["id"] for x in seq if not x["match"]]
        rec.expect(f"{name}: warm-up equals the reference", not bad, f"{len(seq)} requests, differing {bad[:5]}")
    iso.run_closed(srv, ctx, "warm-conc", iso.request_order(ctx.mini_valid, 2 * len(ctx.mini_valid),
                                                            f"{ctx.args.seed}-warm"), 16, warm_runs)
    run = warm_runs[-1]
    conc = [r for r in run["records"] if r]
    if "aborted" in run:
        raise StepFailed(f"warm-conc: transport error on {run['aborted']['id']}")
    bad = [r["id"] for r in run["records"] if not r or not r["match"]]
    rec.expect(f"{name}: concurrent warm-up equals the reference", not bad, f"{len(conc)} requests, differing {bad[:5]}")


def judge_replicas(rec, ctx, spec, name, phase, s):
    """Replica accounting of a phase: routed requests sum to the scheduled ones, every replica of the case received
    requests and was active in nvidia-smi, reservations back to 0. Returns validity."""
    reps = s["replicas"]
    routed = [reps[str(r)]["routed_requests_total"] or 0 for r in range(len(spec["replicas"]))]
    ok = rec.expect(f"{name} {phase}: replica routed counters sum to the scheduled requests"
                    + (", every replica used" if len(routed) > 1 else ""),
                    sum(routed) == s["scheduled"] and all(x > 0 for x in routed),
                    f"routed {routed}, scheduled {s['scheduled']}")
    released = s.get("released")
    ok &= rec.expect(f"{name} {phase}: reservations back to 0", bool(released and released["released"]),
                     f"{released}")
    util = {i: (s["gpus"].get(str(i)) or {}).get("util_max") for i in sorted(spec["placement"])}
    if any(v is None for v in util.values()):
        rec.check(f"{name} {phase}: nvidia-smi activity on GPUs {sorted(util)}", "GAP", f"utilization max {util}")
        return False
    ok &= rec.expect(f"{name} {phase}: nvidia-smi activity on GPUs {sorted(util)}", all(v > 0 for v in util.values()),
                     f"utilization max {util}")
    return ok


def perf_case(rec, ctx, spec):
    case_dir = ctx.out / "performance" / spec["label"]
    case_dir.mkdir(parents=True)
    name = f"performance/{spec['label']}"
    result = {"label": spec["label"], "variant": spec["variant"], "ab": spec["ab"], "replicas": spec["replicas"],
              "placement_expected": sorted(spec["placement"]), "phases": {}, "valid": {}}
    print(f"{time.strftime('%H:%M:%S')} case {name}", flush=True)
    records, warm_runs, runs, srv = [], [], [], None
    try:
        guard(ctx, rec, f"{name}")
        busy = iso.wait_gpus_idle()
        if rec.expect(f"{name}: no compute process on the GPUs before start", not busy, f"{len(busy)} processes"):
            cfg = yaml_config(case_dir / "serve.yml", ctx.args.port, case_dir / "gemm-cache",
                              [entry(ctx, ctx.mini, **spec["config"])], ctx.args.max_inflight)
            srv = iso.Server(rec, ctx, case_dir, ctx.candidate, cfg, {})
    except (StepFailed, OSError, subprocess.SubprocessError) as e:
        rec.check(f"{name}: server started", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
    if srv is None:
        return perf_not_run(rec, ctx, case_dir, name, result)
    ctx.case_in_progress = result  # main records it as stopped when the run ends before this case returns
    model = ctx.mini["name"]
    replicas = len(spec["replicas"])
    try:
        srv.wait_ready()
        time.sleep(2)
        placement = mg.session_placement(srv.proc.pid, ctx.by_bus)
        result["placement_after_load_mib"] = {str(k): v for k, v in (placement or {}).items()}
        mg.placement_check(rec, f"{name}: memory only on physical GPUs {sorted(spec['placement'])}", placement,
                           spec["placement"])
        perf_warm_up(rec, ctx, srv, records, warm_runs, name)
        for phase, kind, _, _ in perf_plan(ctx.args):
            time.sleep(ctx.args.phase_gap)
            srv.alive()
            before = srv.metrics(f"{phase}-before")
            _, before_samples = scrape(srv.url)
            cpu0 = cpu_snapshot(srv.proc.pid)
            if kind == "closed":
                iso.run_closed(srv, ctx, phase, ctx.schedules[phase]["ids"], ctx.schedules[phase]["clients"], runs)
            else:
                iso.run_open(srv, ctx, phase, ctx.schedules[phase], runs)
            run = runs[-1]
            run["cpu"] = cpu_usage(cpu0, cpu_snapshot(srv.proc.pid))
            released, seconds, _ = wait_released(srv.url, model, replicas, 30)
            run["released"] = {"released": released, "seconds": seconds}
            run["metrics"] = {"before": before, "after": srv.metrics(f"{phase}-after")}
            _, after_samples = scrape(srv.url)
            run["replicas"] = delta(replica_stats(after_samples, model, replicas),
                                    replica_stats(before_samples, model, replicas))
            if "aborted" in run:
                raise StepFailed(f"{phase}: transport error on {run['aborted']['id']}, later phases not run")
        srv.alive()
        code = srv.shutdown()
        log = (case_dir / "server.log").read_text(errors="replace")
        rec.expect(f"{name}: graceful shutdown", code == 0 and "shutting down" in log, f"exit {code}")
        after = mg.session_placement(srv.proc.pid, ctx.by_bus)
        rec.check(f"{name}: no GPU memory left after exit",
                  "FAIL" if after else "PASS" if after == {} and placement else "GAP", f"{after}")
    except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        rec.check(f"{name}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
    finally:
        def close():
            result["telemetry_collector"] = srv.telemetry.status()
            srv.close()

        close_and_write(rec, name, close, case_dir / "requests.jsonl",
                        lambda: records + [r for run in (*warm_runs, *runs) for r in run["records"] if r])
    for run in runs:
        s = iso.phase_summary(run, srv.telemetry, [], ctx, spec)
        s.update(cpu=run.get("cpu"), replicas=run.get("replicas", {}), released=run.get("released"))
        result["phases"][run["phase"]] = s
        valid = iso.judge_phase(rec, ctx, spec, name, run["phase"], s)
        valid = judge_replicas(rec, ctx, spec, name, run["phase"], s) and valid
        result["valid"][run["phase"]] = judge_cpu(rec, name, run["phase"], s["cpu"]) and valid
    return perf_not_run(rec, ctx, case_dir, name, result)


def perf_not_run(rec, ctx, case_dir, name, result):
    for phase, _, _, _ in perf_plan(ctx.args):
        if phase not in result["phases"]:
            rec.check(f"{name} {phase}: measured", "FAIL" if not past_deadline(ctx) else "GAP",
                      "phase not run (see above)")
            result["valid"][phase] = False
    mg.write_json(case_dir / "summary.json", result)
    return result


PERF_METRICS = [("latency_ms", "p50"), ("latency_ms", "p95"), ("latency_ms", "p99"), ("service_ms", "p99"),
                ("queue_ms", "p50"), ("queue_ms", "p99"), ("compute_ms", "p50"), ("batch_requests", "mean"),
                ("gen_lag_ms", "p99"), ("ok_per_s", None)]


def phase_value(s, metric, q):
    if metric == "cpu":
        return (s.get("cpu") or {}).get(q)
    v = s.get(metric)
    return v.get(q) if isinstance(v, dict) and q else v


def perf_comparisons(results, args):
    out = {}
    metrics = PERF_METRICS + [("cpu", "server_cpu_per_wall"), ("cpu", "client_cpu_per_wall"),
                              ("cpu", "host_busy_share")]
    for phase, _, _, _ in perf_plan(args):
        row = {}
        for metric, q in metrics:
            per = {}
            for ab in "AB":
                vals = [phase_value(r["phases"][phase], metric, q) for r in results
                        if r["ab"] == ab and r["valid"].get(phase)]
                vals = [v for v in vals if v is not None]
                per[ab] = {"values": vals, "median": statistics.median(vals) if vals else None}
            a, b = per["A"]["median"], per["B"]["median"]
            row[f"{metric}.{q}" if q else metric] = {**per, "b_over_a": round(b / a, 4) if a and b else None}
        out[phase] = row
    return out


def perf_coverage(rec, results, args):
    for phase, _, _, _ in perf_plan(args):
        for ab in "AB":
            n = sum(1 for r in results if r["ab"] == ab and r["valid"].get(phase))
            want = 2 * args.repeats
            rec.check(f"coverage performance {ab} {phase}", "PASS" if n >= want else "GAP", f"{n} valid of {want} cases")


# ---------------------------------------------------------------- fingerprint and main


def fingerprint(args, ctx, gpus, root):
    lib = Path(args.basal_home) / ".local/cuda/12.9.1/lib"
    versions = {}
    for label, b in (("base", ctx.base), ("candidate", ctx.candidate)):
        try:
            r = subprocess.run([b, "--version"], capture_output=True, text=True, env=ctx.env, timeout=60)
            versions[label] = iso.scrub((r.stdout + r.stderr).strip())
        except (OSError, subprocess.SubprocessError) as e:
            versions[label] = type(e).__name__
    cpu = None
    if Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), None)
    inputs = {f: mg.sha256(ctx.inputs / f) for f in ("bench.jsonl", "systemone.jsonl")}
    return {
        "runner_sha256": mg.sha256(HERE), "isolation_runner_sha256": mg.sha256(ISOLATION),
        "placement_runner_sha256": mg.sha256(iso.HELPER), "sources": source_digest(root, ctx.out),
        "binaries": {"base": {"sha256": mg.sha256(ctx.base), "version": versions["base"]},
                     "candidate": {"sha256": mg.sha256(ctx.candidate), "version": versions["candidate"]}},
        "models": {m["short"]: {"name": m["name"], "revision": m["dir"].name,
                                "safetensors_bytes": (m["dir"] / "model.safetensors").stat().st_size
                                if (m["dir"] / "model.safetensors").is_file() else None} for m in ctx.models},
        "gemm_tables": {k: {"file": v.name, "sha256": mg.sha256(v)} for k, v in ctx.tables.items()},
        "inputs": {"exports_and_http": {"dir": ctx.inputs.name, **inputs},
                   "long_requests": {"file": ctx.long_file.name, "sha256": mg.sha256(ctx.long_file)},
                   "performance_requests": {"file": args.perf_requests.name, "sha256": mg.sha256(args.perf_requests)},
                   "config_rs_sha256": mg.sha256(root / "crates/basal-cli/src/config.rs")},
        "gpus": [{"index": g[0], "name": g[1], "driver": g[3], "memory_total_mib": g[4], "power_limit_w": g[5],
                  "clocks_max_sm_mhz": g[6], "clocks_max_mem_mhz": g[7], "persistence_mode": g[8]} for g in gpus],
        "cuda_libraries": sorted(f"{p.name} {p.stat().st_size}" for p in lib.iterdir() if p.is_file())
        if lib.is_dir() else None,
        "cublaslt_tables": args.cublaslt, "python": platform.python_version(), "kernel": platform.release(),
        "cpu": cpu, "cpu_count": os.cpu_count()}


def closed_list(text):
    out = []
    for part in text.split():
        c, n = part.split(":")
        out.append((int(c), int(n)))
    return out


def print_plan(args):
    print("correctness:")
    exports = len(args.models.split()) * len(args.modes.split()) * 4
    print(f"  doctor (2 runs), invalid (10 refused starts), exports ({exports}: base and candidate on devices 0 and 1, "
          f"{len(args.models.split())} models x {len(args.modes.split())} modes)")
    print("  serve: ref-base-dev0, base-dev1, cand-dev0, cand-one-dev1, visible-1-one, replicas "
          f"({args.clients} clients x {args.rounds} rounds), pressure ({args.pressure_requests} at max_inflight "
          f"{args.pressure_inflight}), cancel ({args.cancel_requests} disconnects after {args.cancel_after:g} s), "
          f"visible-10, shutdown ({args.shutdown_requests} in flight)")
    print(f"performance: {4 * args.repeats} cases (A one replica, B two; A B B A x {args.repeats}), phases per case:")
    for phase, kind, value, n in perf_plan(args):
        if kind == "closed":
            print(f"  {phase}: {n} requests, {value} clients")
        else:
            offs = iso.arrival_offsets(value, args.open_seconds, f"{args.seed}-{phase}", args.arrivals)
            print(f"  {phase}: {len(offs)} arrivals in {args.open_seconds:g} s, arrival sha256 "
                  f"{iso.canonical_sha([round(x, 6) for x in offs])[:16]}")
    fixed = len(args.rates) * args.open_seconds + len(perf_plan(args)) * args.phase_gap
    print(f"open-loop schedules and gaps: {fixed:g} s per case, plus loading, warm-up, closed loops and drains; "
          f"run deadline {args.deadline_minutes} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", type=Path, help="archived placement CUDA binary (device per model, no replicas)")
    ap.add_argument("--candidate", type=Path, help="replica CUDA binary")
    ap.add_argument("--expect-base-sha256", help="required: SHA-256 the base binary must have")
    ap.add_argument("--expect-candidate-sha256", help="required: SHA-256 the candidate binary must have")
    ap.add_argument("--out", type=Path, help="new output directory")
    ap.add_argument("--campaigns", type=str.split, default=["correctness", "performance"],
                    help="subset of: correctness performance")
    ap.add_argument("--steps", type=str.split, default=["doctor", "invalid", "exports", "serve"],
                    help="correctness steps: subset of doctor invalid exports serve")
    ap.add_argument("--models", default="mini 4.5B", help="export models; the first is served")
    ap.add_argument("--modes", default="single tree tree-noprefix", help=f"export modes: {', '.join(MODES)}")
    ap.add_argument("--inputs", type=Path, default=Path("reports/reference-basal-1.5-4.5B-fp32"),
                    help="bench.jsonl and systemone.jsonl for exports and HTTP parity")
    ap.add_argument("--long-requests", type=Path, default=Path("tools/bench/long_states.jsonl"))
    ap.add_argument("--long-ids", type=str.split, default=["long-4000-q1", "long-4000-q5"],
                    help="long requests of the pressure, cancel and shutdown steps (must be answered 200)")
    ap.add_argument("--long-tokens", type=int, default=3000, help="long lane threshold of every served model")
    ap.add_argument("--clients", type=int, default=16, help="replicas step: concurrent clients")
    ap.add_argument("--rounds", type=int, default=4, help="replicas step: passes over the workload")
    ap.add_argument("--pressure-requests", type=int, default=48)
    ap.add_argument("--pressure-inflight", type=int, default=8, help="max_inflight of the pressure server")
    ap.add_argument("--cancel-requests", type=int, default=48)
    ap.add_argument("--cancel-after", type=float, default=0.5, help="seconds before a cancel client disconnects")
    ap.add_argument("--cancel-drain-timeout", type=float, default=180)
    ap.add_argument("--shutdown-requests", type=int, default=16)
    ap.add_argument("--inflight-delay", type=float, default=0.05, help="seconds between the requests and SIGTERM")
    ap.add_argument("--perf-requests", type=Path, default=Path("reports/reference-basal-1.5-mini-fp32/systemone.jsonl"),
                    help="performance requests ({id, request}); those answered 200 in the first warm-up are used")
    ap.add_argument("--min-perf-requests", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=1, help="A B B A blocks of the performance campaign")
    ap.add_argument("--closed", type=closed_list, default=closed_list("1:300 16:2000 32:3000"),
                    help="closed loops as CLIENTS:REQUESTS, space separated")
    ap.add_argument("--rates", type=lambda s: [float(x) for x in s.split()], default=[50.0, 100.0, 150.0],
                    help="open-loop arrival rates (requests/s), space separated")
    ap.add_argument("--open-seconds", type=float, default=30)
    ap.add_argument("--arrivals", choices=["poisson", "uniform"], default="poisson")
    ap.add_argument("--seed", default="20261012")
    ap.add_argument("--max-outstanding", type=int, default=256)
    ap.add_argument("--late-ms", type=float, default=5)
    ap.add_argument("--drain-timeout", type=float, default=120)
    ap.add_argument("--phase-gap", type=float, default=2)
    ap.add_argument("--telemetry-min-share", type=float, default=0.5)
    ap.add_argument("--max-inflight", type=int, default=4096, help="max_inflight of the performance servers")
    ap.add_argument("--port", type=int, default=18400)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--export-timeout", type=int, default=1800)
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--http-timeout", type=int, default=300)
    ap.add_argument("--shutdown-timeout", type=int, default=120)
    ap.add_argument("--deadline-minutes", type=float, default=100, help="no new step or case after this")
    ap.add_argument("--plan-only", action="store_true", help="print the plan; needs no GPU or binary")
    args = ap.parse_args()

    bad = [c for c in args.campaigns if c not in ("correctness", "performance")] + \
        [s for s in args.steps if s not in ("doctor", "invalid", "exports", "serve")] + \
        [m for m in args.modes.split() if m not in MODES]
    if bad or not args.campaigns:
        ap.error(f"unknown or empty campaigns, steps or modes: {bad}")
    if args.repeats < 1 or args.max_outstanding < 1 or args.open_seconds <= 0 or not perf_plan(args) or \
            any(c < 1 or n < 1 for c, n in args.closed) or any(r <= 0 for r in args.rates) or \
            len(set(args.rates)) != len(args.rates) or min(args.pressure_requests, args.cancel_requests,
                                                           args.shutdown_requests, args.clients, args.rounds) < 1:
        ap.error("counts and durations must be positive, rates distinct, at least one phase")
    if not 0 < args.telemetry_min_share <= 1:
        ap.error("--telemetry-min-share must be in (0, 1]")
    empty = [r for r in args.rates if not iso.arrival_offsets(r, args.open_seconds, f"{args.seed}-open-{r:g}rps",
                                                             args.arrivals)]
    if empty:
        ap.error(f"no arrival in the schedule of rates {empty}")
    if args.plan_only:
        print_plan(args)
        return 0

    if not (args.out and args.base and args.candidate and args.expect_base_sha256 and args.expect_candidate_sha256):
        ap.error("--out, --base, --candidate, --expect-base-sha256 and --expect-candidate-sha256 are required")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    for label, b in binaries.items():
        if not os.access(b, os.X_OK):
            ap.error(f"not an executable: {b}")
        got, want = mg.sha256(b), getattr(args, f"expect_{label}_sha256").lower()
        if got != want:
            ap.error(f"{label} binary SHA-256 is {got}, expected {want}")
    if binaries["base"] == binaries["candidate"] or args.expect_base_sha256.lower() == args.expect_candidate_sha256.lower():
        ap.error("--base and --candidate must be different binaries")
    for f in (args.inputs / "bench.jsonl", args.inputs / "systemone.jsonl", args.long_requests, args.perf_requests):
        if not f.is_file():
            ap.error(f"missing {f}")

    root = Path.cwd()
    try:
        known = mg.known_revisions(root)
        models = [mg.resolve_model(s, known, args.hf_home) for s in args.models.split()]
        gpus = mg.smi(["--query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,"
                       "clocks.max.mem,persistence_mode", "--format=csv,noheader,nounits"])
        mg.smi([f"--query-gpu={iso.Telemetry.FIELDS}", "--format=csv,noheader,nounits"])
        perf_items = iso.load_requests(args.perf_requests, models[0]["name"])
        long_ids = {json.loads(x)["id"] for x in args.long_requests.read_text().splitlines() if x.strip()}
    except (StepFailed, iso.Fatal, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        print(f"preflight: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if [i for i in args.long_ids if i not in long_ids]:
        ap.error(f"--long-ids not in {args.long_requests}")
    if len(gpus) != 2 or len({g[1] for g in gpus}) != 1:
        ap.error(f"needs exactly two identical GPUs, found {[g[1] for g in gpus]}")
    slug = re.sub(r"[^a-z0-9]+", "-", gpus[0][1].lower()).rstrip("-")
    tables = {}
    for m in models:
        t = root / "crates/basal-cli/gemm-tables" / f"{slug}--{m['slug']}--f16--cublaslt{args.cublaslt}.json"
        if not t.is_file():
            ap.error(f"missing bundled GEMM table {t} (no GEMM search is run)")
        tables[m["short"]] = t

    sys.setswitchinterval(0.0005)
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    rec = mg.Record(out)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args, ctx.out, ctx.env = args, out, mg.environment(args)
    ctx.base, ctx.candidate = binaries["base"], binaries["candidate"]
    ctx.models, ctx.mini, ctx.tables, ctx.modes = models, models[0], tables, args.modes.split()
    ctx.inputs, ctx.long_file = args.inputs.resolve(), args.long_requests.resolve()
    ctx.by_bus = {mg.bus(g[2]): int(g[0]) for g in gpus}
    ctx.deadline = time.monotonic() + args.deadline_minutes * 60
    ctx.perf_items, ctx.mini_bodies = perf_items, dict(perf_items)
    ctx.reference, ctx.mini_valid, ctx.mini_excluded, ctx.schedules = {}, None, None, None
    ctx.http_ref, ctx.items = {}, []
    ctx.deadline_hit, ctx.case_in_progress = False, None

    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w,"
                                  "clocks.max.sm_mhz,clocks.max.mem_mhz,persistence_mode\n" +
                                  "".join(",".join(g) + "\n" for g in gpus))
    fp = fingerprint(args, ctx, gpus, root)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("base", "candidate", "out", "hf_home", "basal_home", "plan_only", "inputs",
                           "long_requests", "perf_requests")}
    params["telemetry"] = {"fields": iso.Telemetry.FIELDS, "interval_ms": 200}
    params["env"] = {k: ctx.env[k] for k in ("BASAL_NO_UPDATE_CHECK", "CUDA_DEVICE_ORDER")}
    mg.write_json(out / "run.json", {"fingerprint": fp, "parameters": params,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    results, complete, stopped, current = [], False, None, None
    steps = args.steps if "correctness" in args.campaigns else []
    specs = perf_specs(args) if "performance" in args.campaigns else []
    started_steps, started_cases = [], []
    try:
        for s in steps:
            print(f"{time.strftime('%H:%M:%S')} step {s}", flush=True)
            current = f"step {s}"
            started_steps.append(s)
            try:
                guard(ctx, rec, f"step {s}")
                {"doctor": step_doctor, "invalid": step_invalid, "exports": step_exports, "serve": step_serve}[s](
                    rec, ctx)
            except Deadline:
                pass
            except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
                rec.check(f"step {s}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
        if specs:
            (out / "performance").mkdir()
            print(f"{time.strftime('%H:%M:%S')} {len(specs)} cases: " + " ".join(s["label"] for s in specs), flush=True)
            for k, spec in enumerate(specs):
                current = f"performance/{spec['label']}"
                started_cases.append(spec["label"])
                try:
                    results.append(perf_case(rec, ctx, spec))
                    ctx.case_in_progress = None
                except Deadline:
                    for later in specs[k + 1:]:
                        rec.check(f"performance/{later['label']}", "GAP",
                                  f"not run: --deadline-minutes {args.deadline_minutes} passed")
                    break
        complete = not ctx.deadline_hit
    except iso.Fatal as e:
        stopped = f"Fatal during {current or 'setup'}: {e}"
    except BaseException as e:  # an interruption (SIGTERM, SIGHUP, Ctrl-C) or an unexpected error: re-raised
        for sig in STOP_SIGNALS:  # the bounded cleanup below is not cut short
            signal.signal(sig, signal.SIG_IGN)
        reason = f" (exit {e.code})" if isinstance(e, SystemExit) else f": {e}" if str(e) else ""
        stopped = f"{type(e).__name__}{reason} during {current or 'setup'}"
        raise
    finally:
        if stopped:
            rec.check("run", "FAIL", iso.scrub(f"stopped: {stopped}; partial records kept, nothing later run"))
            for s in steps:
                if s not in started_steps:
                    rec.check(f"step {s}", "GAP", "not run: the run stopped")
            for spec in specs:
                if spec["label"] not in started_cases:
                    rec.check(f"performance/{spec['label']}", "GAP", "not run: the run stopped")
        if ctx.case_in_progress is not None:  # stopped inside a case: its raw records are written, nothing is valid
            partial = dict(ctx.case_in_progress, stopped=iso.scrub(stopped),
                           valid={phase: False for phase, _, _, _ in perf_plan(args)})
            mg.write_json(out / "performance" / partial["label"] / "summary.json", partial)
            results.append(partial)
        if specs:
            perf_coverage(rec, results, args)
        if ctx.reference:
            mg.write_json(out / "performance" / "reference-answers.json",
                          {m: {i: [r["status"], r["answer"]] for i, r in refs.items()}
                           for m, refs in ctx.reference.items()})
        counts = {k: sum(c["status"] == k for c in rec.checks) for k in ("PASS", "FAIL", "GAP")}
        status = 1 if counts["FAIL"] or stopped else 3 if counts["GAP"] or not complete else 0
        mg.write_json(out / "summary.json", {
            "complete": complete, "stopped": iso.scrub(stopped), "exit_status": status, "counts": counts,
            "fingerprint": fp, "parameters": params,
            "correctness": {"http_reference_statuses": dict(Counter(str(v[0]) for v in ctx.http_ref.values())),
                            "http_reference_sha256": iso.canonical_sha(ctx.http_ref) if ctx.http_ref else None},
            "performance": {
                "selected": ctx.mini_valid, "excluded_status": ctx.mini_excluded,
                "selected_sha256": iso.canonical_sha([[i, ctx.mini_bodies[i]] for i in ctx.mini_valid])
                if ctx.mini_valid else None,
                "schedules": {p: {k: v for k, v in s.items() if k not in ("ids", "offsets")}
                              for p, s in (ctx.schedules or {}).items()},
                "reference_sha256": {m: iso.canonical_sha({i: r["answer_sha256"] for i, r in refs.items()})
                                     for m, refs in ctx.reference.items()},
                "cases": results, "comparisons": perf_comparisons(results, args) if results else None},
            "checks": iso.portable_checks(rec.checks), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"DONE {counts} exit {status}" + (f" (stopped: {stopped})" if stopped else ""), flush=True)
    return status


def terminate(signum, _frame):
    """SystemExit(128 + signal) once; later SIGTERM, SIGHUP and SIGINT (Ctrl-C) are ignored so the cleanup (process
    stops, each bounded), requests.jsonl and summary.json are not cut short (SIGKILL still ends the runner)."""
    for sig in STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    for sig in STOP_SIGNALS:
        signal.signal(sig, terminate)
    try:
        sys.exit(main())
    finally:
        for p in list(mg.LIVE):
            mg.stop(p, timeout=30)
