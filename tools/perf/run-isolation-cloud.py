#!/usr/bin/env python3
"""Manual matched-traffic evidence run: does a busy model on one GPU change the latency of a model on another GPU of
the same `basal serve` process? Rented Linux host with two identical NVIDIA GPUs. Not a test and not a CI gate: it
collects raw evidence for a report the operator writes.

  cd /work && python3 tools/perf/run-isolation-cloud.py \
      --base /opt/basal-dev/basal-cuda-base --candidate /opt/basal-dev/basal-cuda --out out/isolation

Prerequisites (synced to /work, prefetched to /data): this file and tools/perf/run-multigpu-cloud.py (its process,
nvidia-smi, model and HTTP helpers are reused), tools/bench/long_states.jsonl, reports/reference-basal-1.5-mini-fp32
(systemone.jsonl), crates/basal-cli/gemm-tables (bundled f16 batch-invariant tables of this card; no GEMM search is
run), crates/basal-cli/src/config.rs (pinned model revisions), the weights of mini and 4.5B in HF_HOME=/data/hf, the
CUDA libraries of `basal setup` in BASAL_HOME=/data/basal, and both binaries. CUDA_DEVICE_ORDER=PCI_BUS_ID is set for
every process, so CUDA ordinals (without CUDA_VISIBLE_DEVICES) are the nvidia-smi indices.

Campaigns (cases run one after another, never overlapping; each case is a fresh server; order A B B A per --repeats):
  isolation  candidate binary. A `alone`: mini on device 1 only. B `loaded`: the same mini on device 1 and the
             background model (4.5B) on device 0 of the same server, kept busy by a separate client process with
             --bg-clients closed-loop clients sending the --bg-ids long documents. The background starts after the
             warm-up and before the first measured mini request and stops after the last mini phase has drained.
  overhead   mini alone on one physical GPU (CUDA_VISIBLE_DEVICES=--overhead-gpu, identical YAML without `device`):
             A `base` binary (no --device; per-device kernel selection absent) vs B `cand`. No background.

Every case: warm-up and validation (every mini candidate request once in order, every selected request twice with 16
clients; for loaded cases every background request once in order), then the measured mini phases in this order:
  closed-c16   --closed-requests requests with --closed-clients clients in a closed loop; the throughput is a
               consequence of the server's speed, not a controlled arrival rate
  open-<R>rps  open loop for each --rates R: one arrival schedule per rate (Poisson, fixed seed, --open-seconds),
               fixed after the first warm-up, before any measured phase, identical in every case; requests are
               released at their intended times whatever the answers; latency is measured from the intended arrival,
               so client-side lateness and backlog are included (no coordinated omission); at most --max-outstanding
               requests are on the wire, later ones wait in the client queue and the wait is recorded, never dropped
Mini traffic is identical in every case and campaign: the same selected requests (the System One requests answered
200 in the first warm-up; the rest are listed as excluded with their status), the same closed-loop order and the
same open-loop (time, request) lists; their SHA-256 are in summary.json.

Validation (each a PASS/FAIL/GAP line in checks.txt): the reference answer of each request (status and body without
usage.latency_ms, canonical JSON) is taken from the first warm-up of its model and every later answer of every case,
phase and background request must equal it; every scheduled mini request must be answered (any non-200, 529 global
admission refusal, transport error, cancelled or never sent request fails the phase; an empty schedule is refused
before the run); an open loop must be answered within --drain-timeout after its schedule (later answers are kept
in the records but the phase is a FAIL and no valid observation); the process memory must be on the expected
physical GPUs; the server must exit 0 on SIGTERM. A transport error stops its phase and case at once (the clients
finish only the requests in flight, bounded by --http-timeout; the requests not sent fail the phase), so a hung
server costs minutes, not hours. A phase is a valid observation only with /metrics valid before and after it, the
timing headers on every 200 answer and, for every expected GPU, enough usable telemetry samples (all of
utilization, memory, power, SM clock and temperature present; --telemetry-min-share of the 200 ms intervals, at
least five, at most 2 s without one); otherwise a GAP, separate from inference errors. A loaded phase is only
evidence of a busy neighbour when the background kept on average >= 90% of its clients in flight with at most 1%
of the phase window idle, its overlapping requests matched the reference and the mean nvidia-smi utilization of
its GPU was >= --busy-util (GAP without that GPU's telemetry). Only valid phases enter the A/B medians; coverage
needs a valid phase in all 2 x --repeats cases of each variant (fewer is a GAP). The alone case of the isolation
campaign idles --bg-lead seconds before its first phase, as the loaded case does while its background starts.
Exit status: 0 all PASS, 1 at least one FAIL, 3 no FAIL but a GAP (missing evidence), 2 preflight or usage error.

Recorded (all under --out, which must not exist; raw, may contain local paths and PCI topology, keep outside Git):
checks.txt, summary.json (meant to be portable: fingerprint, parameters, workload and schedule hashes, per case and
phase p50/p95/p99 of latency, server queue/compute headers, generator lag, backlog, throughput, outcomes, metrics
and telemetry coverage, per-GPU utilization/memory/power/clocks/temperature, background in-flight profile, A/B
medians; local paths and PCI bus IDs in check details and error texts are replaced by <local> with a pointer to the
raw checks.txt entry; review it before copying it into the repository), run.json, commands.jsonl, schedules.json,
reference-answers.json, per case: server config and log, requests.jsonl (every request with intended, release,
start and end times on one monotonic clock, pending and in-flight counts, status, answer hash, headers, outcome),
background.jsonl, /metrics before and after every phase, telemetry.csv (nvidia-smi every 200 ms) and placement.

Limitations: one host and one card model; the client runs on the same host (its lag is recorded, not removed); the
A/B difference of a loaded case also includes the second model's HTTP handling and memory in the same process and
the host CPU load of the background client; nvidia-smi utilization is a coarse busy share; header intervals are server
measurements of batching, not GPU kernel time. Quantiles: nearest rank over the requests of one phase of one case.
The run makes no claim of complete isolation or of a speed-up; the operator interprets the medians of A and B.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
HELPER = HERE.with_name("run-multigpu-cloud.py")


def load_helper():
    """The committed placement runner as a module: Record, spawn/stop/LIVE, smi, bus, environment, with_env,
    known_revisions, resolve_model, port_free_or_fail, http, normalized, session_placement, sha256, write_json."""
    spec = importlib.util.spec_from_file_location("basal_multigpu_runner", HELPER)
    if spec is None or not HELPER.is_file():
        sys.exit(f"missing {HELPER} (sync tools/perf)")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mg = load_helper()
StepFailed = mg.StepFailed

# One clock for all records: CLOCK_MONOTONIC (shared by the processes of the host) shifted to Unix time at start.
CLOCK_OFFSET = time.time() - time.monotonic()


def now():
    return time.monotonic() + CLOCK_OFFSET


class Fatal(Exception):
    """Stops the whole run (no case after it would be comparable)."""


# Absolute paths (after a start, space, quote or bracket) and PCI bus IDs in free-form text.
LOCAL = re.compile(r"(?:^|(?<=[\s'\"(\[=:,]))/[^\s'\"\])},]+"
                   r"|\b[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\b")


def scrub(text):
    """`text` with local paths and PCI bus IDs replaced by <local> (for summary.json; raw files keep them)."""
    return None if text is None else LOCAL.sub("<local>", str(text))


def portable_checks(checks):
    """The checks for summary.json: a detail with a local path or PCI bus ID is scrubbed and points to the raw
    entry of checks.txt (the n-th check, in order)."""
    out = []
    for n, c in enumerate(checks, 1):
        x = {"check": scrub(c["check"]), "status": c["status"], "detail": scrub(c["detail"])}
        if x["detail"] != c["detail"] or x["check"] != c["check"]:
            x["raw"] = f"checks.txt, check {n}"
        out.append(x)
    return out


# ---------------------------------------------------------------- inputs and schedules


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_requests(path, model_name):
    """[(id, body)] of the lines {"id", "request": {...}}, `model` replaced; duplicate IDs refused."""
    out, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not isinstance(r.get("request"), dict):
            continue
        if r["id"] in seen:
            raise Fatal(f"{path}: duplicate request id {r['id']}")
        seen.add(r["id"])
        body = json.loads(json.dumps(r["request"]))
        body["model"] = model_name
        out.append((r["id"], body))
    return out


def arrival_offsets(rate, seconds, seed, kind):
    """Intended arrival times (s from the phase start) of an open loop: Poisson or evenly spaced."""
    if kind == "uniform":
        return [k / rate for k in range(int(rate * seconds))]
    rng, t, out = random.Random(seed), 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= seconds:
            return out
        out.append(t)


def request_order(ids, n, seed):
    """`n` request IDs: shuffled passes over `ids`, deterministic for the seed."""
    rng, out = random.Random(seed), []
    while len(out) < n:
        p = list(ids)
        rng.shuffle(p)
        out += p
    return out[:n]


def phase_plan(args):
    """[(phase, kind, parameter)] of the measured mini phases, in run order."""
    plan = []
    if args.closed_requests > 0:
        plan.append((f"closed-c{args.closed_clients}", "closed", args.closed_clients))
    plan += [(f"open-{r:g}rps", "open", r) for r in args.rates]
    return plan


def empty_open_phases(args):
    """Open-loop phases whose arrival schedule has no arrival (rate x seconds too small); they are refused."""
    return [phase for phase, kind, value in phase_plan(args)
            if kind == "open" and not arrival_offsets(value, args.open_seconds, f"{args.seed}-{phase}", args.arrivals)]


def make_schedules(args, valid_ids):
    """Every measured phase: the request IDs in order and, for open loops, the intended arrival offsets."""
    out = {}
    for phase, kind, value in phase_plan(args):
        if kind == "closed":
            ids = request_order(valid_ids, args.closed_requests, f"{args.seed}-{phase}-order")
            out[phase] = {"kind": kind, "clients": value, "ids": ids}
        else:
            offsets = [round(x, 6) for x in arrival_offsets(value, args.open_seconds, f"{args.seed}-{phase}",
                                                            args.arrivals)]
            out[phase] = {"kind": kind, "rate": value, "seconds": args.open_seconds, "arrivals": args.arrivals,
                          "offsets": offsets,
                          "ids": request_order(valid_ids, len(offsets), f"{args.seed}-{phase}-order")}
        if not out[phase]["ids"]:
            raise Fatal(f"{phase}: empty schedule")
        out[phase]["sha256"] = canonical_sha({k: v for k, v in out[phase].items() if k != "sha256"})
        if "offsets" in out[phase]:
            out[phase]["arrivals_sha256"] = canonical_sha(out[phase]["offsets"])
    return out


# ---------------------------------------------------------------- HTTP


def header_number(h, key, cast):
    try:
        return cast(h[key]) if key in h else None
    except ValueError:
        return None


def post(url, body, timeout):
    """One POST /v1/systemone: times on the shared clock, status, answer hash and `_answer` (canonical JSON without
    usage.latency_ms; the caller drops it), the server's timing headers. A transport error has status None."""
    r = {"t_start": now()}
    try:
        status, headers, raw = mg.http("POST", url + "/v1/systemone", body, timeout)
        h = {k.lower(): v for k, v in headers.items()}
        answer = mg.normalized(raw)
        r.update(status=status, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(), _answer=answer,
                 queue_ms=header_number(h, "x-basal-queue-ms", float),
                 compute_ms=header_number(h, "x-basal-compute-ms", float),
                 batch=header_number(h, "x-basal-batch-requests", int))
    except (OSError, http.client.HTTPException, ValueError) as e:
        r.update(status=None, error=f"{type(e).__name__}: {e}")
    r["t_done"] = now()
    return r


def judged(r, ref):
    """`r` with `match` (status and answer equal to the reference) and the answer kept only when it differs."""
    out = dict(r)
    answer = out.pop("_answer", None)
    out["match"] = ref is not None and out.get("status") is not None and \
        (out["status"], out.get("answer_sha256")) == (ref["status"], ref["answer_sha256"])
    if not out["match"] and answer is not None:
        out["answer"] = answer
    return out


# ---------------------------------------------------------------- telemetry


TELEMETRY_MAX_GAP_S = 2.0  # longest time without a usable sample of an expected GPU in a valid phase


class Telemetry:
    """nvidia-smi every 200 ms, one row per physical GPU, prefixed with the local read time (shared clock)."""

    FIELDS = ("index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,"
              "temperature.gpu,pstate")

    # values a sample must have to count toward the coverage of a phase
    REQUIRED = ("util", "mem_mib", "power_w", "sm_mhz", "temp_c")
    INTERVAL_S = 0.2

    def __init__(self, path):
        self.samples, self.rows, self.unparsed = [], 0, 0
        self.file = open(path, "w")
        self.file.write("t," + self.FIELDS + "\n")
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits", "-lms", "200"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        except OSError:
            self.file.close()
            raise
        mg.LIVE.add(self.proc)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            t = now()
            self.file.write(f"{t:.3f},{line.strip()}\n")
            c = [x.strip() for x in line.split(",")]

            def num(i):
                try:
                    return float(c[i])
                except (ValueError, IndexError):
                    return None

            self.rows += 1
            if num(0) is None:
                self.unparsed += 1
                continue
            self.samples.append({"t": t, "index": int(num(0)), "util": num(1), "mem_util": num(2),
                                 "mem_mib": num(3), "power_w": num(4), "sm_mhz": num(5), "mem_mhz": num(6),
                                 "temp_c": num(7), "pstate": c[8] if len(c) > 8 else None})

    def status(self):
        """Collector state: rows read, rows without a GPU index, nvidia-smi exit status (None: running)."""
        return {"rows": self.rows, "unparsed": self.unparsed, "exit": self.proc.poll()}

    def close(self):
        mg.stop(self.proc, timeout=5)
        self.thread.join(5)
        self.file.close()

    def window(self, t0, t1):
        """Per physical index in [t0, t1]: samples, usable samples (every REQUIRED value present) and the longest
        time without one (window edges included), mean/max utilization, peak memory, power, clocks, temperature."""
        out = {}
        for i in sorted({s["index"] for s in self.samples}):
            xs = [s for s in self.samples if s["index"] == i and t0 <= s["t"] <= t1]
            usable = [s["t"] for s in xs if all(s[k] is not None for k in self.REQUIRED)]
            edges = [t0, *usable, t1]

            def vals(k):
                return [s[k] for s in xs if s[k] is not None]

            def mean(k):
                v = vals(k)
                return round(sum(v) / len(v), 1) if v else None

            out[str(i)] = {"samples": len(xs), "usable_samples": len(usable),
                           "max_gap_s": round(max(b - a for a, b in zip(edges, edges[1:])), 3),
                           "util_mean": mean("util"), "util_max": max(vals("util"), default=None),
                           "mem_util_mean": mean("mem_util"), "mem_peak_mib": max(vals("mem_mib"), default=None),
                           "power_mean_w": mean("power_w"), "power_max_w": max(vals("power_w"), default=None),
                           "sm_mean_mhz": mean("sm_mhz"), "sm_min_mhz": min(vals("sm_mhz"), default=None),
                           "mem_clock_mean_mhz": mean("mem_mhz"), "temp_max_c": max(vals("temp_c"), default=None),
                           "pstates": sorted({s["pstate"] for s in xs if s["pstate"]})}
        return out


# ---------------------------------------------------------------- statistics


def dist(xs):
    """n, mean and nearest-rank p50/p95/p99 and max of the values (None when empty)."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None

    def rank(p):
        return round(xs[max(0, math.ceil(p / 100 * len(xs)) - 1)], 3)

    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 3), "p50": rank(50), "p95": rank(95), "p99": rank(99),
            "max": round(xs[-1], 3)}


def in_flight_profile(intervals, w0, w1):
    """Requests in flight within [w0, w1] from their [start, end] intervals: time-weighted mean, minimum, seconds
    with none in flight."""
    if w1 <= w0:
        return None
    level, covered, events = 0, 0.0, []
    for a, b in intervals:
        if b <= w0 or a >= w1:
            continue
        covered += min(b, w1) - max(a, w0)
        if a <= w0:
            level += 1
        else:
            events.append((a, 1))
        if b < w1:
            events.append((b, -1))
    events.sort()
    minimum, idle, cur = level, 0.0, w0
    for t, d in events:
        if level == 0:
            idle += t - cur
        cur = t
        level += d
        minimum = min(minimum, level)
    if level == 0:
        idle += w1 - cur
    return {"mean": round(covered / (w1 - w0), 3), "min": minimum, "idle_s": round(idle, 4)}


# ---------------------------------------------------------------- server


def yaml_config(path, port, max_inflight, cache, models):
    """Serve configuration; `device` only when given (the baseline binary does not know the field)."""
    lines = [f"addr: 127.0.0.1:{port}", "metrics: true", f"max_inflight: {max_inflight}",
             f"gemm_cache: {json.dumps(str(cache))}", "models:"]
    for m in models:
        lines += [f"  - path: {json.dumps(str(m['dir']))}", f"    gemm_table: {json.dumps(str(m['table']))}",
                  '    dtype: "f16"', f"    long_tokens: {m['long_tokens']}"]
        if m.get("device") is not None:
            lines.append(f"    device: {m['device']}")
    path.write_text("\n".join(lines) + "\n")
    return path


class Server:
    def __init__(self, rec, ctx, case_dir, binary, cfg, delta):
        self.ctx, self.dir = ctx, case_dir
        self.url = f"http://127.0.0.1:{ctx.args.port}"
        mg.port_free_or_fail(ctx.args.port)
        self.telemetry = Telemetry(case_dir / "telemetry.csv")
        argv = [binary, "--color", "never", "serve", "--config", cfg]
        try:
            self.proc = mg.spawn(argv, mg.with_env(ctx.env, delta), case_dir / "server.log")
        except OSError:
            self.telemetry.close()
            raise
        rec.command({"step": f"serve-{case_dir.parent.name}-{case_dir.name}", "argv": [str(a) for a in argv],
                     "env": delta})

    def alive(self):
        if self.proc.poll() is not None:
            raise StepFailed(f"server exited ({self.proc.returncode}), see server.log")

    def wait_ready(self):
        deadline = time.monotonic() + self.ctx.args.ready_timeout
        while time.monotonic() < deadline:
            self.alive()
            try:
                status, _, body = mg.http("GET", self.url + "/health", timeout=5)
                if status == 200:
                    return json.loads(body)
            except (OSError, http.client.HTTPException, ValueError):
                pass
            time.sleep(1)
        raise StepFailed(f"server not ready after {self.ctx.args.ready_timeout} s")

    def metrics(self, name):
        """GET /metrics into metrics-<name>.txt: {"status", "valid"}; valid when 200 with the inference request
        counter (basal_http_requests_total, present in both binaries) in the body."""
        path = self.dir / f"metrics-{name}.txt"
        try:
            status, _, raw = mg.http("GET", self.url + "/metrics", timeout=30)
        except (OSError, http.client.HTTPException) as e:
            path.write_text(f"# {type(e).__name__}: {e}\n")
            return {"status": None, "valid": False}
        path.write_bytes(raw)
        series = any(line.startswith("basal_http_requests_total{")
                     for line in raw.decode(errors="replace").splitlines())
        return {"status": status, "valid": status == 200 and series}

    def shutdown(self):
        os.kill(self.proc.pid, signal.SIGTERM)
        try:
            return self.proc.wait(self.ctx.args.shutdown_timeout)
        except subprocess.TimeoutExpired:
            return None

    def close(self):
        mg.stop(self.proc)
        self.telemetry.close()


# ---------------------------------------------------------------- mini traffic


def wait_all(futures, deadline=None, abort=None):
    """Waits in steps of at most 1 s, never past `deadline` (signals stay deliverable); returns the futures not done
    at `deadline` or when `abort` is set. A late wake-up can still return after `deadline`, so callers judging
    a deadline also compare the answer times with it."""
    pending = set(futures)
    while pending and not (abort and abort.is_set()):
        step = 1.0 if deadline is None else min(1.0, deadline - time.monotonic())
        if step <= 0:
            break
        _, pending = concurrent.futures.wait(pending, timeout=step)
    return pending


def transport_failure(r):
    """The first transport error of a phase: index, ID and error text (local paths scrubbed)."""
    return {"k": r["k"], "id": r["id"], "error": scrub(r.get("error")), "t": r["t_done"]}


def run_closed(srv, ctx, phase, ids, clients, runs):
    """`ids` in order by `clients` closed-loop clients; latency from the send. Appended to `runs` before it starts so
    an interrupted phase keeps its partial records. The first transport error (server hung, crashed or the
    connection lost) stops the clients: each finishes its request in flight (bounded by --http-timeout), the IDs
    not sent stay None (not released) and the run keeps the error as `aborted`."""
    run = {"phase": phase, "kind": "closed", "clients": clients, "scheduled": len(ids), "records": [None] * len(ids),
           "complete": False}
    runs.append(run)
    lock, stop, counter = threading.Lock(), threading.Event(), itertools.count()
    state = {"inflight": 0, "max_inflight": 0}

    def client(c):
        while not stop.is_set():
            with lock:
                k = next(counter)
                if k >= len(ids):
                    return
                state["inflight"] += 1
                state["max_inflight"] = max(state["max_inflight"], state["inflight"])
                inflight = state["inflight"]
            rid = ids[k]
            r = post(srv.url, ctx.mini_bodies[rid], ctx.args.http_timeout)
            with lock:
                state["inflight"] -= 1
            out = judged(r, ctx.reference["mini"].get(rid))
            out.update(phase=phase, k=k, id=rid, client=c, inflight_at_start=inflight,
                       latency_ms=round((out["t_done"] - out["t_start"]) * 1e3, 3))
            out["service_ms"] = out["latency_ms"]
            run["records"][k] = out
            if out["status"] is None:
                with lock:
                    run.setdefault("aborted", transport_failure(out))
                stop.set()

    pool = concurrent.futures.ThreadPoolExecutor(clients)
    run["t0"] = now()
    try:
        futures = [pool.submit(client, c) for c in range(clients)]
        wait_all(futures)
        for f in futures:
            f.result()
    except BaseException:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    run.update(t1=now(), complete=True, max_inflight=state["max_inflight"])
    return run


def run_open(srv, ctx, phase, sched, runs):
    """Releases request k at t0 + offsets[k] whatever the answers. At most --max-outstanding requests are sent at
    once; later ones queue in the client (recorded as pending, not dropped). After the schedule, waits for the
    answers until --drain-timeout (`t_drain_deadline`). Past it the phase is not drained (`drain_timed_out`, a
    FAIL): queued requests are cancelled (outcome "cancelled"), the ones in flight are still awaited for the record
    (bounded by --http-timeout; `after_drain_deadline`), and `t1` is the end of that cleanup. Any answer after
    the deadline also sets `drain_timed_out`, even when every request was done by the time the wait returned.
    The first transport error stops the release (later arrivals stay None, not released; the error is kept as
    `aborted`) and the drain."""
    offsets, ids = sched["offsets"], sched["ids"]
    n = len(ids)
    run = {"phase": phase, "kind": "open", "rate": sched["rate"], "scheduled": n, "records": [None] * n,
           "complete": False, "max_outstanding": ctx.args.max_outstanding}
    runs.append(run)
    lock, abort = threading.Lock(), threading.Event()
    cnt = {"submitted": 0, "started": 0, "done": 0, "max_pending": 0, "max_inflight": 0}

    def work(k, pre):
        with lock:
            cnt["started"] += 1
            pending, inflight = cnt["submitted"] - cnt["started"], cnt["started"] - cnt["done"]
            cnt["max_pending"] = max(cnt["max_pending"], pending)
            cnt["max_inflight"] = max(cnt["max_inflight"], inflight)
        r = post(srv.url, ctx.mini_bodies[pre["id"]], ctx.args.http_timeout)
        with lock:
            cnt["done"] += 1
        out = judged(r, ctx.reference["mini"].get(pre["id"]))
        out.update(pre, pending_at_start=pending, inflight_at_start=inflight,
                   gen_lag_ms=round((out["t_start"] - pre["t_intended"]) * 1e3, 3),
                   latency_ms=round((out["t_done"] - pre["t_intended"]) * 1e3, 3),
                   service_ms=round((out["t_done"] - out["t_start"]) * 1e3, 3))
        run["records"][k] = out
        if out["status"] is None:
            with lock:
                run.setdefault("aborted", transport_failure(out))
            abort.set()

    pool = concurrent.futures.ThreadPoolExecutor(ctx.args.max_outstanding)
    t0 = now() + 0.2
    run["t0"] = t0
    futures = []
    try:
        for k in range(n):
            if abort.is_set():
                break
            target = t0 + offsets[k]
            delay = target - now()
            if delay > 0:
                time.sleep(delay)
            t_release = now()
            with lock:
                cnt["submitted"] += 1
                outstanding = cnt["submitted"] - cnt["done"] - 1
            pre = {"phase": phase, "k": k, "id": ids[k], "t_intended": round(target, 6),
                   "t_release": t_release, "release_lag_ms": round((t_release - target) * 1e3, 3),
                   "outstanding_at_arrival": outstanding}
            run["records"][k] = dict(pre, outcome="released")
            futures.append(pool.submit(work, k, pre))
        run["t_schedule_end"] = now()
        deadline = time.monotonic() + ctx.args.drain_timeout
        run["t_drain_deadline"] = deadline + CLOCK_OFFSET
        left = wait_all(futures, deadline, abort)
        timed_out = bool(left) and time.monotonic() >= deadline
        cancelled = [k for k, f in enumerate(futures) if f.cancel()]
        for k in cancelled:
            run["records"][k] = dict(run["records"][k], outcome="cancelled")
        pool.shutdown(wait=True, cancel_futures=True)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    run.update(t1=now(), complete=True, drain_timed_out=timed_out, cancelled=len(cancelled),
               in_flight_at_drain_end=len(left) - len(cancelled), max_pending=cnt["max_pending"],
               max_inflight=cnt["max_inflight"])
    late = 0
    for r in run["records"]:
        if r and r.get("t_done", 0) > run["t_drain_deadline"]:
            r["after_drain_deadline"] = True
            late += 1
    run["drain_timed_out"] = timed_out or late > 0
    return run


def run_sequential(srv, ctx, model, items, records, phase, establish):
    """Every item once in order; with `establish` the answers become the reference of `model`. A transport error
    stops at once (StepFailed after recording it): later requests would only wait on a hung or lost server."""
    out = []
    for rid, body in items:
        srv.alive()
        r = post(srv.url, body, ctx.args.http_timeout)
        if r["status"] is None:
            x = judged(r, None)
            x.update(phase=phase, model=model, id=rid)
            records.append(x)
            raise StepFailed(f"{phase}: transport error on {rid}: {scrub(r['error'])}")
        if establish:
            ref = {"status": r.get("status"), "answer_sha256": r.get("answer_sha256"), "answer": r.get("_answer")}
            ctx.reference.setdefault(model, {})[rid] = ref
        x = judged(r, ctx.reference[model].get(rid))
        x.update(phase=phase, model=model, id=rid)
        records.append(x)
        out.append(x)
    return out


# ---------------------------------------------------------------- background load


def background_worker(cfg_path):
    """Child process: `clients` closed-loop clients sending the background requests until SIGTERM, then finishing
    their requests in flight. One JSON line per request; the answer is written for the first (id, hash) pair only.
    A client stops after a transport error (no retry storm against a stopped or hung server); the parent sees the
    error and the lower in-flight count."""
    global CLOCK_OFFSET
    cfg = json.loads(Path(cfg_path).read_text())
    CLOCK_OFFSET = cfg["clock_offset"]
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGHUP, lambda *_: stop.set())
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent stops it
    items, clients, lock, seen = cfg["requests"], cfg["clients"], threading.Lock(), set()
    out = open(cfg["out"], "a")

    def write(line):
        with lock:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()

    def client(c):
        n = 0
        while not stop.is_set():
            rid, body = items[(c + n) % len(items)]
            r = post(cfg["url"], body, cfg["http_timeout"])
            answer = r.pop("_answer", None)
            line = {"client": c, "n": n, "id": rid, **r}
            with lock:
                if answer is not None and (rid, r["answer_sha256"]) not in seen:
                    seen.add((rid, r["answer_sha256"]))
                    line["answer"] = answer
            write(line)
            n += 1
            if r["status"] is None:
                write({"event": "client-stopped", "stopped_client": c, "t": now(), "reason": "transport error"})
                return

    write({"event": "started", "t": now(), "clients": clients})
    threads = [threading.Thread(target=client, args=(c,), daemon=True) for c in range(clients)]
    for t in threads:
        t.start()
    while not stop.wait(0.5):
        pass
    t_stop = now()
    for t in threads:
        t.join(cfg["http_timeout"] + 30)
    write({"event": "stopped", "t_stop": t_stop, "t_end": now(), "threads_alive": sum(t.is_alive() for t in threads)})
    out.close()
    return 0


class Background:
    def __init__(self, rec, ctx, case_dir, url):
        self.rec, self.ctx, self.dir = rec, ctx, case_dir
        self.path = case_dir / "background.jsonl"
        cfg = {"url": url, "clients": ctx.args.bg_clients, "http_timeout": ctx.args.http_timeout,
               "clock_offset": CLOCK_OFFSET, "out": str(self.path), "requests": ctx.bg_items}
        self.cfg = case_dir / "background-config.json"
        mg.write_json(self.cfg, cfg)
        self.proc, self.code = None, None

    def start(self):
        argv = [sys.executable, "-I", str(HERE), "--background-worker", str(self.cfg)]
        self.proc = mg.spawn(argv, self.ctx.env, self.dir / "background.log")
        self.t_spawn = now()
        self.rec.command({"step": f"background-{self.dir.parent.name}-{self.dir.name}",
                          "argv": [str(a) for a in argv], "env": {}})

    def records(self):
        out = []
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a line cut by a kill
        return out

    def wait_ready(self):
        """--bg-lead seconds after the start and at least one background request answered 200."""
        deadline = time.monotonic() + self.ctx.args.bg_ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise StepFailed(f"background client exited ({self.proc.returncode}), see background.log")
            if now() - self.t_spawn >= self.ctx.args.bg_lead and any(r.get("status") == 200 for r in self.records()):
                return
            time.sleep(0.5)
        raise StepFailed(f"no background answer within {self.ctx.args.bg_ready_timeout} s")

    def stop(self):
        """SIGTERM: no new requests, the ones in flight are finished (bounded by the HTTP timeout)."""
        if self.proc is not None and self.code is None:
            self.code = mg.stop(self.proc, timeout=self.ctx.args.http_timeout + 30)
        return self.code


# ---------------------------------------------------------------- cases


def compute_apps():
    return mg.smi(["--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader,nounits"])


def wait_gpus_idle(timeout=30):
    """[] when no compute process holds a GPU (a previous server may still be releasing memory)."""
    deadline = time.monotonic() + timeout
    while True:
        rows = compute_apps()
        if not rows or time.monotonic() > deadline:
            return rows
        time.sleep(1)


def case_specs(args, models, tables):
    """Run order of the cases: per campaign, A B B A repeated."""
    mini, bg = models["mini"], models["bg"]

    def entry(m, device):
        return {"dir": m["dir"], "table": tables[m["short"]], "long_tokens": args.long_tokens, "device": device}

    variants = {
        ("isolation", "A"): {"variant": "alone", "binary": "candidate", "delta": {}, "models": [entry(mini, 1)],
                             "background": False, "placement": {1}},
        ("isolation", "B"): {"variant": "loaded", "binary": "candidate", "delta": {},
                             "models": [entry(bg, 0), entry(mini, 1)], "background": True, "placement": {0, 1},
                             "background_gpu": 0},
        ("overhead", "A"): {"variant": "base", "binary": "base",
                            "delta": {"CUDA_VISIBLE_DEVICES": str(args.overhead_gpu)}, "models": [entry(mini, None)],
                            "background": False, "placement": {args.overhead_gpu}},
        ("overhead", "B"): {"variant": "cand", "binary": "candidate",
                            "delta": {"CUDA_VISIBLE_DEVICES": str(args.overhead_gpu)}, "models": [entry(mini, None)],
                            "background": False, "placement": {args.overhead_gpu}},
    }
    out = []
    for campaign in args.campaigns:
        for n, ab in enumerate("ABBA" * args.repeats):
            spec = dict(variants[(campaign, ab)], campaign=campaign, ab=ab)
            spec["label"] = f"{n + 1:02d}-{spec['variant']}"
            out.append(spec)
    return out


def establish_selection(rec, ctx, warm):
    """After the first mini warm-up: selected requests (200), excluded ones (4xx: refused input, listed with their
    status), frozen schedules. A transport error or 5xx in the reference warm-up stops the run."""
    broken = {x["id"]: x.get("status") or x.get("error") for x in warm
              if x.get("status") is None or x["status"] >= 500 or x["status"] in range(201, 400)}
    if broken:
        rec.check("mini reference: warm-up without transport or server errors", "FAIL", json.dumps(broken))
        raise Fatal("transport or server errors in the reference warm-up")
    valid = [x["id"] for x in warm if x.get("status") == 200]
    excluded = {x["id"]: x.get("status") for x in warm if x.get("status") != 200}
    ok = len(valid) >= ctx.args.min_mini_requests
    rec.expect("mini reference: enough requests answered 200", ok,
               f"{len(valid)} selected, {len(excluded)} excluded {json.dumps(excluded)}")
    if not ok:
        raise Fatal("too few mini requests answered 200 in the reference warm-up")
    ctx.mini_valid, ctx.mini_excluded = valid, excluded
    ctx.schedules = make_schedules(ctx.args, valid)
    mg.write_json(ctx.out / "schedules.json", ctx.schedules)


def warm_up(rec, ctx, srv, spec, records, name):
    first = "mini" not in ctx.reference
    try:
        seq = run_sequential(srv, ctx, "mini", ctx.mini_items, records, "warm-seq", first)
    except StepFailed as e:
        if first:  # as in establish_selection: no reference, no comparable case after it
            raise Fatal(f"reference warm-up: {e}") from e
        raise
    if first:
        ctx.reference_source["mini"] = f"{spec['campaign']}/{spec['label']}"
        establish_selection(rec, ctx, seq)
    else:
        bad = [x["id"] for x in seq if not x["match"]]
        rec.expect(f"{name}: mini warm-up equals the reference", not bad, f"{len(seq)} requests, differing {bad[:5]}")
    items = request_order(ctx.mini_valid, len(ctx.mini_valid) * 2, f"{ctx.args.seed}-warm")
    runs = []
    run_closed(srv, ctx, "warm-conc", items, ctx.args.closed_clients, runs)
    conc = runs[0]["records"]
    records += [r for r in conc if r]
    if "aborted" in runs[0]:
        raise StepFailed(f"warm-conc: transport error on {runs[0]['aborted']['id']}: {runs[0]['aborted']['error']}")
    bad = [r["id"] for r in conc if not r or not r["match"]]
    rec.expect(f"{name}: concurrent mini warm-up equals the reference", not bad,
               f"{len(conc)} requests, differing or missing {bad[:5]}")
    if spec["background"]:
        first = "bg" not in ctx.reference
        try:
            seq = run_sequential(srv, ctx, "bg", ctx.bg_items, records, "warm-bg", first)
        except StepFailed:
            if first:  # the next loaded case establishes it again
                ctx.reference.pop("bg", None)
            raise
        if first:
            ctx.reference_source["bg"] = f"{spec['campaign']}/{spec['label']}"
            refused = {x["id"]: x.get("status") for x in seq if x.get("status") != 200}
            if refused:
                del ctx.reference["bg"]
                raise StepFailed(f"background requests not answered 200 (context limit?): {refused}")
        bad = [x["id"] for x in seq if not x["match"] or x.get("status") != 200]
        if bad:
            raise StepFailed(f"background warm-up differs from the reference: {bad}")


def phase_summary(run, tel, bg_records, ctx, spec):
    recs = run["records"]
    done = [r for r in recs if r and "t_done" in r]
    ok = [r for r in done if r.get("status") == 200 and r["match"]]
    t0 = run["t0"]
    t1 = run.get("t1") or max((r["t_done"] for r in done), default=t0)
    last = max((r["t_done"] for r in done), default=t0)
    s = {"kind": run["kind"], "scheduled": run["scheduled"], "complete_run": run["complete"],
         "not_released": sum(r is None for r in recs),
         "released_not_answered": sum(1 for r in recs if r and "t_done" not in r),
         "answered": len(done), "ok": len(ok), "statuses": dict(Counter(str(r.get("status")) for r in done)),
         "transport_errors": sum(r.get("status") is None for r in done),
         "admission_rejects_529": sum(r.get("status") == 529 for r in done),
         "mismatches": sum(1 for r in done if r.get("status") is not None and not r["match"]),
         "window_s": round(t1 - t0, 3),
         "latency_ms": dist(r["latency_ms"] for r in ok), "service_ms": dist(r["service_ms"] for r in ok),
         "queue_ms": dist(r.get("queue_ms") for r in ok), "compute_ms": dist(r.get("compute_ms") for r in ok),
         "batch_requests": dist(r.get("batch") for r in ok),
         "ok_per_s": round(len(ok) / (last - t0), 3) if last > t0 else None,
         "max_inflight": run.get("max_inflight"), "aborted": run.get("aborted"),
         "missing_timing_headers": sum(1 for r in ok if None in (r.get("queue_ms"), r.get("compute_ms"),
                                                                  r.get("batch"))),
         "metrics": run.get("metrics", {})}
    if run["kind"] == "closed":
        s["clients"] = run["clients"]
    else:
        released = [r for r in recs if r]
        timed_out = run.get("drain_timed_out")
        s.update(rate=run["rate"], offered_per_s=round(run["scheduled"] / ctx.args.open_seconds, 3),
                 gen_lag_ms=dist(r.get("gen_lag_ms") for r in done),
                 release_lag_ms=dist(r.get("release_lag_ms") for r in released),
                 late=sum(1 for r in done if r.get("gen_lag_ms", 0) > ctx.args.late_ms),
                 max_pending=run.get("max_pending"),
                 max_outstanding_at_arrival=max((r["outstanding_at_arrival"] for r in released), default=None),
                 drain_timed_out=timed_out, max_outstanding=run["max_outstanding"],
                 cancelled=run.get("cancelled"), in_flight_at_drain_end=run.get("in_flight_at_drain_end"),
                 answered_after_drain_deadline=sum(1 for r in done if r.get("after_drain_deadline")),
                 cleanup_after_drain_deadline_s=round(t1 - run["t_drain_deadline"], 3)
                 if timed_out and "t_drain_deadline" in run else None)
    s["gpus"] = tel.window(t0, t1)
    if spec["background"]:
        answered = [r for r in bg_records if "t_done" in r and "client" in r]
        overlap = [r for r in answered if r["t_start"] < t1 and r["t_done"] > t0]
        bad = [r["id"] for r in overlap
               if r.get("status") != 200 or (ctx.reference.get("bg", {}).get(r["id"]) or {}).get("answer_sha256")
               != r.get("answer_sha256")]
        s["background"] = {"clients": ctx.args.bg_clients,
                           "in_flight": in_flight_profile([(r["t_start"], r["t_done"]) for r in answered], t0, t1),
                           "overlapping": len(overlap), "completed_in_window": sum(t0 <= r["t_done"] <= t1
                                                                                   for r in answered),
                           "not_matching": len(bad), "not_matching_ids": sorted(set(bad))[:10],
                           "latency_ms": dist((r["t_done"] - r["t_start"]) * 1e3 for r in overlap
                                              if r.get("status") == 200),
                           "statuses": dict(Counter(str(r.get("status")) for r in overlap))}
    return s


def observability_problems(ctx, spec, s):
    """Missing evidence of a phase that is not an inference error: /metrics before or after not valid, a 200
    answer without the timing headers, or too few usable telemetry samples (or too long without one) of an
    expected physical GPU. {physical index or "metrics"/"headers": problem}."""
    problems = {}
    bad = [w for w in ("before", "after") if not (s["metrics"].get(w) or {}).get("valid")]
    if bad:
        problems["metrics"] = f"/metrics {' and '.join(bad)} not 200 with basal_http_requests_total"
    if s["missing_timing_headers"]:
        problems["headers"] = f"{s['missing_timing_headers']} answers without x-basal timing headers"
    need = max(5, math.ceil(ctx.args.telemetry_min_share * s["window_s"] / Telemetry.INTERVAL_S))
    for i in sorted(spec["placement"]):
        g = s["gpus"].get(str(i)) or {}
        usable, gap = g.get("usable_samples") or 0, g.get("max_gap_s")
        if usable < need or gap is None or gap > TELEMETRY_MAX_GAP_S:
            problems[str(i)] = (f"GPU {i}: {usable} usable telemetry samples in {s['window_s']} s (need {need}), "
                                f"longest gap {gap} s (most {TELEMETRY_MAX_GAP_S})")
    return problems


def judge_phase(rec, ctx, spec, name, phase, s):
    """Checks of one phase; returns whether it is a valid observation of its variant: something scheduled and every
    request answered with the reference (FAIL otherwise), an open loop drained in time (FAIL), complete metrics,
    headers and telemetry (GAP), and for a loaded case a busy background (FAIL; GAP without its telemetry)."""
    abort = f", stopped after a transport error on {s['aborted']['id']}" if s["aborted"] else ""
    valid = rec.expect(f"{name} {phase}: every scheduled mini request answered 200 with the reference answer",
                       s["scheduled"] > 0 and s["answered"] > 0 and s["ok"] == s["scheduled"] and s["complete_run"],
                       f"{s['ok']}/{s['scheduled']} ok, statuses {s['statuses']}, mismatches {s['mismatches']}, "
                       f"transport {s['transport_errors']}, 529 {s['admission_rejects_529']}, not released "
                       f"{s['not_released']}, released not answered {s['released_not_answered']}{abort}")
    if s["kind"] == "open":
        drained = rec.expect(f"{name} {phase}: answered within --drain-timeout after the schedule",
                             not s["drain_timed_out"] and not s["answered_after_drain_deadline"],
                             f"cancelled {s['cancelled']}, in flight at the deadline {s['in_flight_at_drain_end']}, "
                             f"answered after it {s['answered_after_drain_deadline']} (not valid), cleanup "
                             f"{s['cleanup_after_drain_deadline_s']} s"
                             if s["drain_timed_out"] or s["answered_after_drain_deadline"] else "")
        valid = valid and drained
    problems = observability_problems(ctx, spec, s)
    rec.check(f"{name} {phase}: metrics, timing headers and telemetry of GPUs {sorted(spec['placement'])}",
              "GAP" if problems else "PASS", "; ".join(problems.values()))
    valid = valid and not problems
    if not spec["background"]:
        return valid
    bg, busy_gpu = s["background"], str(spec["background_gpu"])
    gpu = s["gpus"].get(busy_gpu, {})
    profile = bg["in_flight"] or {"mean": 0, "idle_s": s["window_s"], "min": 0}
    if busy_gpu in problems:
        rec.check(f"{name} {phase}: background busy on physical GPU {busy_gpu}", "GAP", problems[busy_gpu])
        return False
    busy = (profile["mean"] >= 0.9 * ctx.args.bg_clients and profile["idle_s"] <= 0.01 * s["window_s"]
            and bg["not_matching"] == 0 and bg["overlapping"] > 0 and gpu["util_mean"] is not None
            and gpu["util_mean"] >= ctx.args.busy_util)
    rec.expect(f"{name} {phase}: background busy on physical GPU {busy_gpu}", busy,
               f"in flight mean {profile['mean']} (min {profile['min']}, idle {profile['idle_s']} s) of "
               f"{ctx.args.bg_clients}, {bg['overlapping']} overlapping, {bg['not_matching']} not matching, "
               f"utilization mean {gpu['util_mean']}% (threshold {ctx.args.busy_util}%)")
    return valid and busy


def run_case(rec, ctx, spec):
    case_dir = ctx.out / spec["campaign"] / spec["label"]
    case_dir.mkdir(parents=True)
    name = f"{spec['campaign']}/{spec['label']}"
    result = {"campaign": spec["campaign"], "label": spec["label"], "variant": spec["variant"], "ab": spec["ab"],
              "binary": spec["binary"], "env": spec["delta"],
              "placement_expected": sorted(spec["placement"]), "phases": {}, "valid": {}}
    print(f"{time.strftime('%H:%M:%S')} case {name}", flush=True)
    records, runs, bg, srv = [], [], None, None
    try:
        busy = wait_gpus_idle()
        if rec.expect(f"{name}: no compute process on the GPUs before start", not busy, f"{busy}"):
            cfg = yaml_config(case_dir / "serve.yml", ctx.args.port, ctx.args.max_inflight, case_dir / "gemm-cache",
                              spec["models"])
            srv = Server(rec, ctx, case_dir, ctx.binaries[spec["binary"]], cfg, spec["delta"])
    except (StepFailed, OSError, subprocess.SubprocessError) as e:
        rec.check(f"{name}: server started", "FAIL", f"{type(e).__name__}: {e}")
    if srv is None:
        return not_run(rec, ctx, case_dir, name, result)
    try:
        health = srv.wait_ready()
        names = sorted(ctx.names[str(m["dir"])] for m in spec["models"])
        rec.expect(f"{name}: ready with the configured models", sorted(health.get("models", [])) == names,
                   f"{health}")
        time.sleep(2)
        placement = mg.session_placement(srv.proc.pid, ctx.by_bus)
        result["placement_after_load_mib"] = {str(k): v for k, v in (placement or {}).items()}
        mg.placement_check(rec, f"{name}: memory only on physical GPUs {sorted(spec['placement'])}", placement,
                           spec["placement"])
        if placement and spec["background"]:
            rec.expect(f"{name}: larger model (background) on physical GPU 0, mini on 1",
                       placement.get(0, 0) > placement.get(1, 0) > 0, f"{result['placement_after_load_mib']}")
        warm_up(rec, ctx, srv, spec, records, name)
        m = result["metrics_after_warm_up"] = srv.metrics("after-warm-up")
        rec.check(f"{name}: /metrics after the warm-up", "PASS" if m["valid"] else "GAP", f"status {m['status']}")
        t_lead = now()
        if spec["background"]:
            bg = Background(rec, ctx, case_dir, srv.url)
            bg.start()
            bg.wait_ready()
        elif spec["campaign"] == "isolation":
            time.sleep(ctx.args.bg_lead)  # the same idle lead as the loaded case before its background answers
        result["lead_s"] = round(now() - t_lead, 3)
        for phase, kind, _ in phase_plan(ctx.args):
            time.sleep(ctx.args.phase_gap)
            srv.alive()
            before = srv.metrics(f"{phase}-before")
            if kind == "closed":
                run_closed(srv, ctx, phase, ctx.schedules[phase]["ids"], ctx.schedules[phase]["clients"], runs)
            else:
                run_open(srv, ctx, phase, ctx.schedules[phase], runs)
            runs[-1]["metrics"] = {"before": before, "after": srv.metrics(f"{phase}-after")}
            if "aborted" in runs[-1]:  # a hung or lost server: later phases would only wait on it
                raise StepFailed(f"{phase}: transport error on {runs[-1]['aborted']['id']}, later phases not run")
        if bg:
            code = bg.stop()
            rec.expect(f"{name}: background client stopped after the mini phases", code == 0, f"exit {code}")
        srv.alive()
        code = srv.shutdown()
        log = (case_dir / "server.log").read_text(errors="replace")
        rec.expect(f"{name}: graceful shutdown", code == 0 and "shutting down" in log, f"exit {code}")
        after = mg.session_placement(srv.proc.pid, ctx.by_bus)
        rec.check(f"{name}: no GPU memory left after exit",
                  "FAIL" if after else "PASS" if after == {} and placement else "GAP", f"{after}")
    except (StepFailed, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        rec.check(f"{name}", "FAIL", f"{type(e).__name__}: {e}")
    finally:
        result["telemetry_collector"] = srv.telemetry.status()
        srv.close()  # first: on an abort the background requests then end at once
        if bg:
            bg.stop()
        with open(case_dir / "requests.jsonl", "w") as f:
            for r in records + [r for run in runs for r in run["records"] if r]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    bg_records = bg.records() if bg else []
    if bg:
        answered = [r for r in bg_records if "client" in r]
        bad = [r["id"] for r in answered if r.get("status") != 200 or r.get("answer_sha256") !=
               (ctx.reference.get("bg", {}).get(r["id"]) or {}).get("answer_sha256")]
        result["background"] = {"answered": len(answered), "not_matching": len(bad),
                                "statuses": dict(Counter(str(r.get("status")) for r in answered)),
                                "client_exit": bg.code}
        rec.expect(f"{name}: every background answer 200 with the reference answer", answered and not bad,
                   f"{len(answered)} answered, statuses {result['background']['statuses']}, "
                   f"{len(bad)} not matching {sorted(set(bad))[:5]}")
    for run in runs:
        s = phase_summary(run, srv.telemetry, bg_records, ctx, spec)
        result["phases"][run["phase"]] = s
        result["valid"][run["phase"]] = judge_phase(rec, ctx, spec, name, run["phase"], s)
    return not_run(rec, ctx, case_dir, name, result)


def not_run(rec, ctx, case_dir, name, result):
    """FAIL for every phase the case did not measure; writes the case summary."""
    for phase, _, _ in phase_plan(ctx.args):
        if phase not in result["phases"]:
            rec.check(f"{name} {phase}: measured", "FAIL", "phase not run (see the failure above)")
            result["valid"][phase] = False
    mg.write_json(case_dir / "summary.json", result)
    return result


# ---------------------------------------------------------------- summary


METRICS = [("latency_ms", "p50"), ("latency_ms", "p95"), ("latency_ms", "p99"), ("service_ms", "p99"),
           ("queue_ms", "p50"), ("queue_ms", "p99"), ("compute_ms", "p50"), ("gen_lag_ms", "p99"), ("ok_per_s", None)]


def comparisons(results, args):
    """Per campaign and phase: each metric of the valid cases of A and B in run order, their medians and B/A."""
    out = {}
    for campaign in args.campaigns:
        cases = [r for r in results if r["campaign"] == campaign]
        for phase, _, _ in phase_plan(args):
            row = {}
            for metric, q in METRICS:
                per = {}
                for ab in "AB":
                    vals = []
                    for r in cases:
                        if r["ab"] != ab or not r["valid"].get(phase):
                            continue
                        v = r["phases"][phase].get(metric)
                        v = v.get(q) if isinstance(v, dict) and q else v
                        if v is not None:
                            vals.append(v)
                    per[ab] = {"values": vals, "median": statistics.median(vals) if vals else None}
                a, b = per["A"]["median"], per["B"]["median"]
                row[f"{metric}.{q}" if q else metric] = {**per, "b_over_a": round(b / a, 4) if a and b else None}
            out[f"{campaign}/{phase}"] = row
    return out


def coverage(rec, results, args):
    for campaign in args.campaigns:
        for phase, _, _ in phase_plan(args):
            for ab in "AB":
                n = sum(1 for r in results if r["campaign"] == campaign and r["ab"] == ab and r["valid"].get(phase))
                want = 2 * args.repeats
                rec.check(f"coverage {campaign} {ab} {phase}", "PASS" if n >= max(2, want) else "GAP",
                          f"{n} valid of {want} cases")


def fingerprint(args, ctx, gpus, models, tables):
    env = ctx.env
    lib = Path(args.basal_home) / ".local/cuda/12.9.1/lib"
    versions = {}
    for label, b in ctx.binaries.items():
        try:
            r = subprocess.run([b, "--version"], capture_output=True, text=True, env=env, timeout=60)
            versions[label] = scrub((r.stdout + r.stderr).strip())
        except (OSError, subprocess.SubprocessError) as e:
            versions[label] = f"{type(e).__name__}"
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), None) if Path("/proc/cpuinfo").is_file() else None
    return {
        "runner_sha256": mg.sha256(HERE), "helper_sha256": mg.sha256(HELPER),
        "binaries": {k: {"sha256": mg.sha256(v), "version": versions[k]} for k, v in ctx.binaries.items()},
        "models": {k: {"short": m["short"], "name": m["name"], "revision": m["dir"].name,
                       "safetensors_bytes": (m["dir"] / "model.safetensors").stat().st_size
                       if (m["dir"] / "model.safetensors").is_file() else None} for k, m in models.items()},
        "gemm_tables": {k: {"file": v.name, "sha256": mg.sha256(v)} for k, v in tables.items()},
        "inputs": {"mini_requests": {"file": args.mini_requests.name, "sha256": mg.sha256(args.mini_requests)},
                   "background_requests": {"file": args.bg_requests.name, "sha256": mg.sha256(args.bg_requests)},
                   "config_rs_sha256": mg.sha256(Path("crates/basal-cli/src/config.rs"))},
        "gpus": [{"index": g[0], "name": g[1], "driver": g[3], "memory_total_mib": g[4], "power_limit_w": g[5],
                  "clocks_max_sm_mhz": g[6], "persistence_mode": g[7]} for g in gpus],
        "cuda_libraries": sorted(f"{p.name} {p.stat().st_size}" for p in lib.iterdir() if p.is_file())
        if lib.is_dir() else None,
        "cublaslt_tables": args.cublaslt, "python": platform.python_version(), "kernel": platform.release(),
        "cpu": cpu, "cpu_count": os.cpu_count()}


# ---------------------------------------------------------------- main


def numbers(text, cast):
    return [cast(x) for x in text.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", type=Path, help="baseline CUDA binary (87da786, no --device); overhead campaign")
    ap.add_argument("--candidate", type=Path, help="candidate CUDA binary (ae15ad9)")
    ap.add_argument("--out", type=Path, help="new output directory")
    ap.add_argument("--expect-base-sha256", help="refuse to run when the baseline binary has another SHA-256")
    ap.add_argument("--expect-candidate-sha256", help="refuse to run when the candidate binary has another SHA-256")
    ap.add_argument("--campaigns", type=str.split, default=["isolation", "overhead"],
                    help="subset of: isolation overhead")
    ap.add_argument("--repeats", type=int, default=1, help="A B B A blocks per campaign")
    ap.add_argument("--mini", default="mini", help="measured model: mini, 4.5B, max or NAME=DIRECTORY")
    ap.add_argument("--bg-model", default="4.5B", help="background model on device 0 of the loaded case")
    ap.add_argument("--mini-requests", type=Path, default=Path("reports/reference-basal-1.5-mini-fp32/systemone.jsonl"),
                    help="mini candidate requests ({id, request}); those answered 200 in the first warm-up are used")
    ap.add_argument("--min-mini-requests", type=int, default=10, help="fewest selected mini requests")
    ap.add_argument("--bg-requests", type=Path, default=Path("tools/bench/long_states.jsonl"))
    ap.add_argument("--bg-ids", type=str.split, default=["long-4000-q1", "long-4000-q5", "long-8000-q1",
                                                         "long-8000-q5"],
                    help="background request IDs; each must be answered 200 (within the model's context)")
    ap.add_argument("--bg-clients", type=int, default=16, help="closed-loop background clients")
    ap.add_argument("--bg-lead", type=float, default=10, help="seconds of background load before the first phase")
    ap.add_argument("--bg-ready-timeout", type=float, default=180)
    ap.add_argument("--busy-util", type=float, default=80,
                    help="least mean nvidia-smi utilization (%%) of the background GPU in a loaded phase")
    ap.add_argument("--closed-requests", type=int, default=2000, help="requests of the closed-loop phase (0: none)")
    ap.add_argument("--closed-clients", type=int, default=16)
    ap.add_argument("--rates", type=lambda s: numbers(s, float), default=[50.0, 100.0],
                    help="open-loop arrival rates (requests/s), space separated")
    ap.add_argument("--open-seconds", type=float, default=30, help="length of each open-loop arrival schedule")
    ap.add_argument("--arrivals", choices=["poisson", "uniform"], default="poisson")
    ap.add_argument("--seed", default="20261011", help="seed of the request orders and arrival schedules")
    ap.add_argument("--max-outstanding", type=int, default=256, help="open loop: most mini requests on the wire")
    ap.add_argument("--late-ms", type=float, default=5, help="open loop: start later than intended counted as late")
    ap.add_argument("--drain-timeout", type=float, default=120, help="open loop: wait after the schedule ends")
    ap.add_argument("--phase-gap", type=float, default=2, help="idle seconds of mini traffic between phases")
    ap.add_argument("--telemetry-min-share", type=float, default=0.5,
                    help="least usable nvidia-smi samples of each expected GPU in a valid phase, as a share of the "
                         "200 ms intervals of its window (at least 5; at most 2 s without one)")
    ap.add_argument("--max-inflight", type=int, default=4096, help="server max_inflight (global admission limit)")
    ap.add_argument("--long-tokens", type=int, default=3000, help="long lane threshold of every served model")
    ap.add_argument("--overhead-gpu", type=int, default=1, help="physical GPU of the overhead campaign")
    ap.add_argument("--port", type=int, default=18300)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--ready-timeout", type=int, default=300)
    ap.add_argument("--http-timeout", type=int, default=300)
    ap.add_argument("--shutdown-timeout", type=int, default=120)
    ap.add_argument("--plan-only", action="store_true",
                    help="print the phases, schedule sizes and hashes of the arrival times; needs no GPU or binary")
    args = ap.parse_args()

    unknown = [c for c in args.campaigns if c not in ("isolation", "overhead")]
    if unknown or not args.campaigns:
        ap.error(f"--campaigns: unknown or empty {unknown}")
    if args.repeats < 1 or args.closed_clients < 1 or args.bg_clients < 1 or args.max_outstanding < 1 or \
            args.closed_requests < 0 or args.open_seconds <= 0 or any(r <= 0 for r in args.rates) or \
            len(set(args.rates)) != len(args.rates):
        ap.error("counts and durations must be positive and rates distinct")
    if not phase_plan(args):
        ap.error("no measured phase (--closed-requests 0 and no --rates)")
    if not 0 < args.telemetry_min_share <= 1:
        ap.error("--telemetry-min-share must be in (0, 1]")
    empty = empty_open_phases(args)
    if empty:
        ap.error(f"no arrival in the schedule of {empty} (raise the rate or --open-seconds)")
    if args.plan_only:
        cases = len(args.campaigns) * 4 * args.repeats
        print(f"{cases} cases ({', '.join(args.campaigns)}; A B B A x {args.repeats}), phases per case:")
        for phase, kind, value in phase_plan(args):
            if kind == "closed":
                print(f"  {phase}: {args.closed_requests} requests, {value} clients")
            else:
                offs = arrival_offsets(value, args.open_seconds, f"{args.seed}-{phase}", args.arrivals)
                gaps = [b - a for a, b in zip([0.0] + offs, offs)]
                print(f"  {phase}: {len(offs)} arrivals in {args.open_seconds:g} s ({len(offs) / args.open_seconds:.1f}"
                      f"/s), largest gap {max(gaps, default=0) * 1e3:.1f} ms, arrival sha256 "
                      f"{canonical_sha([round(x, 6) for x in offs])[:16]}")
        fixed = len(args.rates) * args.open_seconds + len(phase_plan(args)) * args.phase_gap
        print(f"open-loop schedules and gaps: {fixed:g} s per case, plus model loading, warm-up, the closed loop, "
              f"drains and (isolation cases) {args.bg_lead:g} s lead")
        return 0

    if not args.out or not args.candidate or ("overhead" in args.campaigns and not args.base):
        ap.error("--out, --candidate and (for overhead) --base are required")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    binaries = {"candidate": args.candidate.resolve()}
    if "overhead" in args.campaigns:
        binaries["base"] = args.base.resolve()
    for label, b in binaries.items():
        if not os.access(b, os.X_OK):
            ap.error(f"not an executable: {b}")
        want = getattr(args, f"expect_{label}_sha256")
        if want and mg.sha256(b) != want.lower():
            ap.error(f"{label} binary SHA-256 is {mg.sha256(b)}, expected {want}")
    for f in (args.mini_requests, args.bg_requests):
        if not f.is_file():
            ap.error(f"missing {f}")

    root = Path.cwd()
    try:
        known = mg.known_revisions(root)
        models = {"mini": mg.resolve_model(args.mini, known, args.hf_home),
                  "bg": mg.resolve_model(args.bg_model, known, args.hf_home)}
        gpus = mg.smi(["--query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,"
                       "persistence_mode", "--format=csv,noheader,nounits"])
        mg.smi([f"--query-gpu={Telemetry.FIELDS}", "--format=csv,noheader,nounits"])
        mini_items = load_requests(args.mini_requests, models["mini"]["name"])
        bg_all = dict(load_requests(args.bg_requests, models["bg"]["name"]))
    except (StepFailed, Fatal, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        print(f"preflight: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    missing = [i for i in args.bg_ids if i not in bg_all]
    if missing:
        ap.error(f"--bg-ids not in {args.bg_requests}: {missing}")
    if models["mini"]["dir"] == models["bg"]["dir"]:
        ap.error("--mini and --bg-model must differ")
    if len(gpus) < 2 or len({g[1] for g in gpus}) != 1:
        ap.error(f"needs two identical GPUs, found {[g[1] for g in gpus]}")
    if args.overhead_gpu >= len(gpus):
        ap.error(f"--overhead-gpu {args.overhead_gpu}: {len(gpus)} GPUs")
    slug = re.sub(r"[^a-z0-9]+", "-", gpus[0][1].lower()).rstrip("-")
    tables = {}
    for m in models.values():
        t = root / "crates/basal-cli/gemm-tables" / f"{slug}--{m['slug']}--f16--cublaslt{args.cublaslt}.json"
        if not t.is_file():
            ap.error(f"missing bundled GEMM table {t} (no GEMM search is run)")
        tables[m["short"]] = t

    # a sleeping dispatcher waits for the GIL at most this long behind answer-parsing threads (default 5 ms)
    sys.setswitchinterval(0.0005)
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    rec = mg.Record(out)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args, ctx.out, ctx.env, ctx.binaries = args, out, mg.environment(args), binaries
    ctx.by_bus = {mg.bus(g[2]): int(g[0]) for g in gpus}
    ctx.names = {str(m["dir"]): m["name"] for m in models.values()}
    ctx.mini_items, ctx.mini_bodies = mini_items, dict(mini_items)
    ctx.bg_items = [[i, bg_all[i]] for i in args.bg_ids]
    ctx.reference, ctx.reference_source = {}, {}
    ctx.mini_valid, ctx.mini_excluded, ctx.schedules = None, None, None

    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w,"
                                  "clocks.max.sm_mhz,persistence_mode\n" + "".join(",".join(g) + "\n" for g in gpus))
    fp = fingerprint(args, ctx, gpus, models, tables)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("base", "candidate", "out", "hf_home", "basal_home", "plan_only")}
    params["mini_requests"], params["bg_requests"] = args.mini_requests.name, args.bg_requests.name
    params["telemetry"] = {"fields": Telemetry.FIELDS, "interval_ms": 200}
    params["python_switch_interval_s"] = sys.getswitchinterval()
    params["env"] = {k: ctx.env[k] for k in ("BASAL_NO_UPDATE_CHECK", "CUDA_DEVICE_ORDER")}
    mg.write_json(out / "run.json", {"fingerprint": fp, "parameters": params,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    specs = case_specs(args, models, tables)
    print(f"{time.strftime('%H:%M:%S')} {len(specs)} cases: " + " ".join(f"{s['campaign']}/{s['label']}"
                                                                          for s in specs), flush=True)

    results, complete = [], False
    try:
        for spec in specs:
            results.append(run_case(rec, ctx, spec))
        complete = True
    except Fatal as e:
        rec.check("run", "FAIL", str(e))
    finally:
        if ctx.reference:
            mg.write_json(out / "reference-answers.json",
                          {m: {i: [r["status"], r["answer"]] for i, r in refs.items()}
                           for m, refs in ctx.reference.items()})
        if complete:
            coverage(rec, results, args)
        counts = {k: sum(c["status"] == k for c in rec.checks) for k in ("PASS", "FAIL", "GAP")}
        status = 1 if counts["FAIL"] or not complete else 3 if counts["GAP"] else 0
        mg.write_json(out / "summary.json", {
            "complete": complete, "exit_status": status, "counts": counts, "fingerprint": fp, "parameters": params,
            "workload": {
                "mini_selected": ctx.mini_valid, "mini_excluded_status": ctx.mini_excluded,
                "mini_selected_sha256": canonical_sha([[i, ctx.mini_bodies[i]] for i in ctx.mini_valid])
                if ctx.mini_valid else None,
                "background": args.bg_ids, "background_sha256": canonical_sha(ctx.bg_items),
                "schedules": {p: {k: v for k, v in s.items() if k not in ("ids", "offsets")}
                              for p, s in (ctx.schedules or {}).items()},
                "reference_source": ctx.reference_source,
                "reference_sha256": {m: canonical_sha({i: r["answer_sha256"] for i, r in refs.items()})
                                     for m, refs in ctx.reference.items()}},
            "cases": results, "comparisons": comparisons(results, args), "checks": portable_checks(rec.checks),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"DONE {counts} exit {status}", flush=True)
    return status


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--background-worker":
        sys.exit(background_worker(sys.argv[2]))
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    try:
        sys.exit(main())
    finally:
        for p in list(mg.LIVE):
            mg.stop(p, timeout=30)
