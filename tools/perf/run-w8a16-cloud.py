#!/usr/bin/env python3
"""Manual evidence run for optional W8A16 weights (`quantization: q8_0`) on a rented Linux host with two identical
NVIDIA GPUs. Not a test and not a CI gate: it collects raw evidence for a report the operator writes.

  python3 tools/perf/run-w8a16-cloud.py --help
  python3 tools/perf/run-w8a16-cloud.py --plan-only                   # counts, 900-input check, rough estimates
  python3 tools/perf/run-w8a16-cloud.py --prepare-900 .cache/w8a16-inputs   # the 900-question inputs (no GPU)
  cd /work && python3 tools/perf/run-w8a16-cloud.py \\
      --base /opt/basal-dev/basal-cuda-phase02 --expect-base-sha256 SHA \\
      --candidate /opt/basal-dev/basal-cuda --expect-candidate-sha256 SHA --deadline-minutes 100 --out out/w8a16

--plan-only and --prepare-900 read only repository files: no cloud, SSH, GPU, binary, network or secret.

Binaries: --base is the phase-02 CUDA binary (replicas, source committed as 4894c5e); its SHA-256 must be
289a2eaca3ac96762b7acd309f9dbf39f43041c7dab37ae89eb968125f82dfef (the version text of a build is not its identity).
--candidate is the W8A16 build. Both --expect-*-sha256 are required and checked before anything runs.

Prerequisites (synced to /work, prefetched to /data): this file, tools/perf/run-replicas-cloud.py,
run-isolation-cloud.py and run-multigpu-cloud.py (process, nvidia-smi, model, HTTP, telemetry and record helpers,
reused unchanged), tools/bench/ab.py (unchanged: the A B B A order of `basal bench`), tools/bench/long_states.jsonl,
tools/decision-sets/build_benchmark.py (its `public_request` checks the 900 inputs), the tracked
crates/basal-cli/benchmark-data/v2/sequential.jsonl and tools/decision-sets/benchmark-v2.lock.json,
reports/decision-sets/decisions-basal-1.5-{mini,4.5B}.jsonl, reports/reference-basal-1.5-{mini,4.5B}-fp32 and
-mlx-q8, crates/basal-cli/gemm-tables (bundled f16 batch-invariant tables of this card, passed explicitly to every
f16 and q8_0 command; no GEMM search runs, a log announcing one stops that command as a FAIL), the Rust sources
(read and hashed for the fingerprint), the weights of mini and 4.5B in HF_HOME=/data/hf at the revisions pinned in
crates/basal-cli/src/config.rs and the CUDA libraries in BASAL_HOME=/data/basal. Every process gets
CUDA_DEVICE_ORDER=PCI_BUS_ID and HF_HUB_OFFLINE=1; the Hub client of basal does not honour the latter, so it is no
guarantee: models are passed as local directories (no command of the campaign needs the Hub), and the invalid cases
use an absent repository and check that the validation error comes before the first Hub message.

Campaign `correctness` (two identical cards):
  none     candidate exports without --quantization and with --quantization none, mini and 4.5B, tree and single on
           GPU 0, and tree on GPU 1, each byte-equal (bench.jsonl, systemone.jsonl) to the base export of the same
           mode on GPU 0 and without any `basal compare` difference; the backend description of the manifest equal to
           the base's in every field, its `device` naming the ordinal the export was given (0, or 1 on GPU 1)
  q8       candidate q8_0 exports, mini and 4.5B: tree, single and tree --no-prefix-cache on GPU 0, tree on GPU 1, tree
           under CUDA_VISIBLE_DEVICES=1 with --device 0, each byte-equal to the tree export on GPU 0; q8_0 manifest
           (quantization object, 4 matrices per layer) and physical GPU of every export
  invalid  q8_0 with bf16 / f32, --kernels candle, an unknown value, --quantization with a configuration file, through
           serve, export, bench, benchmark, decide and profile, for a model absent from the cache: a validation error,
           no download, load or GEMM search, no GPU memory of the process; doctor reports the same error as a model
           failure and a valid q8_0 entry with its weight estimate
  serve    mini servers (YAML with the explicit bundled table): base (HTTP reference), candidate without the field
           (every answer, /health and /v1/models bytes, metric families), candidate q8_0 on GPU 0 (every System One
           request: status and answer, multi, evidence and 400/422 included, equal to a native q8_0 reference of the
           same request body: a q8_0 tree export of derived inputs, serve-inputs/ under --out, whose requests are the
           exact HTTP bodies with `model` set to mini, its bench.jsonl a copy of the reference's; for a case that
           export refuses as Unsupported, as its prepare_request does for what serve answers, one `basal decide` of
           the body with the export's flags; the reference inputs and their exports stay unchanged; that export's
           bench.jsonl and the lines of unchanged requests byte-equal to the q8_0 tree export), candidate q8_0 with
           replica_devices [0, 1] (answers equal to the single q8_0 server sequentially and with --clients clients,
           memory and routed requests on both GPUs, SIGTERM exit 0 and no GPU memory left)
Campaign `quality` (GPU 0), mini and 4.5B: f16 and q8_0 tree exports on the FP32 references (44 + 34) with
  `basal compare` FP32 -> f16, FP32 -> q8_0, f16 -> q8_0 and, as context, FP32 -> upstream MLX 8-bit. Decisions:
  bench flips, System One items with identical tokens (argmax p_cal), the typed answer fields (`*_same`) and the
  decision (argmax of `probabilities`, keys matched by name) of every question of every System One case of the
  upstream-compatible endpoint (/v1/basal against Server.decide), so the 8 items whose prompts differ by the known
  export format are still judged by their answers; a missing case, answer, question or distribution is a FAIL, not
  an empty pass. A decision difference of q8_0 to FP32 is a FAIL (criterion unchanged), of the f16 control to FP32
  and of q8_0 to f16 a SIGNAL. The 900 questions (inputs rebuilt from sequential.jsonl, see --prepare-900): f32
  control (`export --dtype f32`, the flags of the stored basal_rs_f32 run: single, fused, no GEMM table), f16 and
  q8_0 tree with the table; every export 900 unique indices and a finite p_cal per option; `basal compare` f32 ->
  f16, f32 -> q8_0, f16 -> q8_0 (nothing missing, 900 compared). Analysis: f32 control 900/900 decisions equal
  to the stored upstream_fp32 (else the comparison with the stored FP32 is a GAP and nothing is attributed to it),
  every f16 decision difference to the stored FP32 and basal_rs_f16 with its FP32 margin, q8_0 flips to FP32 (new,
  fixed, shared) and to f16 of the session, |dp| and TV distributions, per dataset and type, accuracy against gold.
  New q8_0 flips or a lower accuracy than f16 of the session are a SIGNAL (explanation needed; possibly M2).
Campaign `perf-none` (GPU 0, 4.5B): base (A) and candidate without the option (B), identical arguments and table:
  `ab.py --rounds 4` of `basal bench --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16` and four
  `basal benchmark --scenario all --concurrency 1,16` runs A B B A. Per adjacent pair (actual order): every metric,
  B-A and B/A; `basal bench` per question from per_item. A metric with B worse in every pair is a signal (not a
  proof, recorded as a diagnostic of the primary round); then one confirmation round of the affected command and one
  `basal profile` A and B. A signal again in every pair of both rounds is a persistent SIGNAL (parent analysis;
  unexplained: M2); a signal not repeated in the confirmation gives conclusion (a): no repeatable slowdown shown at
  the measured resolution (never "no regression"). A conclusion needs complete evidence: every scheduled pair, every
  metric with a finite value in A and B, every phase complete, per question exactly the items `basal bench` measures
  (n and lat_n of the report as expected; per_item indices 5..43: 39 items after 5 warm-up items of 44, default
  --lat-n 39; unique, a finite positive lat2_ms each); otherwise a GAP without a conclusion (a missing value never
  counts as "B not worse"; every value is kept). No p-value, tolerance, cohort, re-pairing or exclusion; clocks,
  power, temperature and batches describe, only a foreign compute process or missing samples are recorded as an
  outside disturbance. Different errors, completeness or answer changes are a FAIL.
Campaign `perf-q8` (GPU 0, 4.5B): candidate f16 (A) and q8_0 (B), the same scheme without a signal rule: latency,
  throughput, queue, load time, pool memory (load and runtime peaks) and sampled VRAM, power and clocks (a memory/time
  trade-off, no speed gate).
Campaign `profile` (GPU 0, 4.5B): `basal profile` f16 and q8_0 at --batch 1 and --profile-batch; effective
  bandwidth of the dequantization (3.0625 B per value) and of the f16 GEMM weight reads (2 B per value, the pure
  GEMM sections qkv_proj and gate_up_proj). nsys is not run (GAP).
Campaign `memory-mini` (GPU 0): one `basal benchmark --scenario all --concurrency 1,16` each for mini f16 and q8_0
  (pool load / resident / runtime peaks, used and reserved separately).
Campaign `revalidate` (after a code fix only; needs --previous-run): q8_0 tree exports of mini and 4.5B byte-equal to
  those of the previous run; when they differ the q8 correctness step and the quality campaign run again. Together
  with `correctness` (none step), `perf-none`, `perf-q8`, `profile` and `memory-mini` it is the revalidation session.
  --previous-run is checked before anything runs: an existing raw directory of another session (not --out, neither
  inside the other) whose run.json and summary.json carry the same fingerprint, the phase-02 base, a known candidate
  SHA-256 different from this run's, the model revisions, GEMM tables and inputs of this run, and for each model a
  completed q8_0 tree export on GPU 0 (command, checks and manifest of that session). Its candidate SHA-256, source
  digest and the SHA-256 of both sessions' exports are recorded; without the weights' SHA-256 in both sessions the
  comparison is a GAP, not a PASS.

Exit status: 0 all PASS, 1 a FAIL, 4 no FAIL but a SIGNAL, 3 only a GAP (missing evidence), 2 preflight or usage
error; a run stopped by a signal exits 128 + signal, never 0. SIGNAL (analysis needed, possibly M2) is: a persistent
perf-none signal, a decision difference of the f16 control or of q8_0 to f16 on the 44 + 34, new q8_0 flips or a
lower accuracy on the 900, changed q8_0 exports in `revalidate`; a primary-round perf-none signal is not one.
Every subprocess and HTTP request is bounded by its timeout and by --deadline-minutes (required for a run): after
the deadline no step, command or request starts, a command or request in flight is cut at it, and what did not run
is a GAP (never a FAIL; the records already answered are kept). Both ends are absolute: an HTTP request (headers and
body together, also /health, /v1/models, /metrics and readiness) or a command lasts at most until the earlier of
them, and an answer or exit seen only after it is not timely, a status 200 included (kept as evidence, judged as
the end it missed: GAP at the deadline, FAIL past its own timeout). A GPU command (export, decide, basal bench,
benchmark, profile; each basal bench of ab.py separately) or an HTTP request past its own timeout is a FAIL that
stops the run: a suspected hung GPU, no later GPU work in any campaign. A stopped run (that, a signal, Ctrl-C, an
unexpected error) closes its processes, writes every HTTP record already answered, the interrupted command to
commands.jsonl (once, also when the first signal arrives while its process group is being stopped: then SIGKILL and
a bounded reap) and summary.json with complete false; later signals are ignored.

Recorded under --out (must not exist; raw, contains local paths: keep it outside Git, e.g. in .cache/cloud): checks.txt,
commands.jsonl, run.json, summary.json (portable: fingerprint, parameters, results, scrubbed checks), gpus.csv,
source-files.sha256, inputs-900/, exports and compare outputs, ab and benchmark runs with logs and telemetry, servers
with requests.jsonl, correctness/serve-inputs/ (the served request bodies as export inputs, their SHA-256, the
q8_0 HTTP comparison table) and serve-decide/ (request, JSON and log of each basal decide), quality-900-*.json (ids,
distributions, decisions, labels; no texts).
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import importlib.util
import json
import math
import os
import platform
import re
import resource
import select
import shlex
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
ROOT_GUESS = HERE.parents[2]
REPLICAS = HERE.with_name("run-replicas-cloud.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or not Path(path).is_file():
        sys.exit(f"missing {path} (sync tools/perf, tools/bench and tools/decision-sets)")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rp = load_module("basal_replicas_runner", REPLICAS)  # loads the isolation and placement runners (one LIVE registry)
iso, mg = rp.iso, rp.mg
StepFailed, Fatal = iso.StepFailed, iso.Fatal

PHASE02_BASE_SHA256 = "289a2eaca3ac96762b7acd309f9dbf39f43041c7dab37ae89eb968125f82dfef"
SEQUENTIAL = "crates/basal-cli/benchmark-data/v2/sequential.jsonl"
SEQUENTIAL_SHA256 = "81756119f5fe3a4e44aefa1c1cc56e5cd3153001141fb715ec502bb3ee193374"
LOCK = "tools/decision-sets/benchmark-v2.lock.json"
BUILD_BENCHMARK = "tools/decision-sets/build_benchmark.py"
AB = "tools/bench/ab.py"
LONG = "tools/bench/long_states.jsonl"
DECISIONS = "reports/decision-sets/decisions-basal-1.5-{}.jsonl"
REFERENCE = "reports/reference-basal-1.5-{}-fp32"
MLX_Q8 = "reports/reference-basal-1.5-{}-mlx-q8"
SHORTS = ("mini", "4.5B")
SEARCH = "generating the batch-invariant GEMM table"
ABSENT = "Remek/basal-w8a16-absent"  # never in a cache: validation must fail before any download
MODES = mg.MODES
STATUSES = ("PASS", "FAIL", "GAP", "SIGNAL")
# the phases of `basal benchmark --scenario all --concurrency 1,16` (sequential runs at concurrency 1 only)
PHASES = ("sequential-c1", "mixed-c1", "mixed-c16")
# close FP32 ties known from README (reported only for the matching model and card; never an exception list)
KNOWN_TIES = {("4.5B", "RTX 6000 Ada"): ["dyk-15"], ("4.5B", "A100"): ["polemo2-in-85"],
              ("mini", "RTX 6000 Ada"): ["allegro-reviews-37"]}
# rough estimates for --plan-only: tracked manifests, read at plan time (never a measurement of this campaign)
TRACKED_EXPORTS = "reports/compat-1.5-small/export-basal-1.5-{}-{}/manifest.json"
TRACKED_BENCH = "reports/perf-1.5-small/bench-rust-basal-1.5-{}.json"


class Deadline(Exception):
    """The run deadline passed: no new step starts."""


class Record(mg.Record):
    """mg.Record with a fourth status: SIGNAL (a measured signal or difference that needs analysis, not a defect)."""

    def check(self, name, status, detail=""):
        assert status in STATUSES
        with self.lock:
            self.checks.append({"check": name, "status": status, "detail": detail})
            with open(self.out / "checks.txt", "a") as f:
                f.write(f"{status} {name}{': ' + detail if detail else ''}\n")
        print(f"{time.strftime('%H:%M:%S')} {status} {name}{': ' + detail if detail else ''}", flush=True)


def guard(ctx, rec, what):
    if time.monotonic() >= ctx.deadline:
        ctx.deadline_hit = True
        rec.check(what, "GAP", f"not run: --deadline-minutes {ctx.args.deadline_minutes} passed")
        raise Deadline(what)


def stop_after(label, r):
    """Deadline or Fatal for a command stopped at the run deadline or past its own timeout (judge_run recorded the
    GAP or FAIL); called after the caller stored its partial results."""
    if r.get("stopped") == "deadline":
        raise Deadline(label)
    if r.get("stopped") == "timeout":
        raise Fatal(f"{label}: no exit within its timeout: suspected hung GPU, no later GPU work runs")


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def scrubbed(v):
    """`v` with local paths and PCI IDs in every string replaced (for summary.json)."""
    if isinstance(v, str):
        return iso.scrub(v)
    if isinstance(v, dict):
        return {k: scrubbed(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [scrubbed(x) for x in v]
    return v


def argmax(p):
    """First index of the largest value (basal_core::decision::argmax)."""
    best = 0
    for i, v in enumerate(p):
        if v > p[best]:
            best = i
    return best


def margin(p):
    """Gap between the two largest probabilities (how close a decision is)."""
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else None


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def dist(xs):
    """n, mean, nearest-rank p50/p90/p99 and max (None when empty)."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None

    def rank(p):
        return xs[max(0, math.ceil(p * len(xs)) - 1)]

    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": rank(0.5), "p90": rank(0.9), "p99": rank(0.99),
            "max": xs[-1]}


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- the 900-question inputs (no GPU)


def build_900(root, short):
    """The `basal export` input of the 900 public questions for `short`: every request of the tracked sequential.jsonl
    turned back into the record `public_request` (build_benchmark.py) was made from, plus `dataset` and `gold` of the
    stored decisions; index = line position 0..899. Raises Fatal on any mismatch (nothing is repaired)."""
    data = (root / SEQUENTIAL).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SEQUENTIAL_SHA256:
        raise Fatal(f"{SEQUENTIAL}: SHA-256 {digest}, expected {SEQUENTIAL_SHA256}")
    lock = json.loads((root / LOCK).read_text())
    pinned = json.dumps(lock, sort_keys=True)
    if SEQUENTIAL_SHA256 not in pinned:
        raise Fatal(f"{LOCK} does not pin {SEQUENTIAL_SHA256}")
    rows = [json.loads(x) for x in data.decode("utf-8").split("\n") if x.strip()]
    decisions = {}
    for d in read_jsonl(root / DECISIONS.format(short)):
        if not isinstance(d.get("id"), str) or d["id"] in decisions:
            raise Fatal(f"{DECISIONS.format(short)}: null or duplicate id {d.get('id')!r}")
        decisions[d["id"]] = d
    bb = load_module("basal_build_benchmark", root / BUILD_BENCHMARK)
    out = []
    for i, row in enumerate(rows):
        rid, req = row.get("id"), row.get("request")
        if not isinstance(rid, str) or not isinstance(req, dict) or set(req) != {"state", "questions"}:
            raise Fatal(f"sequential line {i + 1}: unexpected row")
        qs = req["questions"]
        if not isinstance(qs, dict) or list(qs) != ["q"]:
            raise Fatal(f"{rid}: expected one question `q`")
        q = qs["q"]
        kind, crit = q.get("type"), q.get("criteria")
        if kind == "choice" and isinstance(crit, dict):
            keys = [f"option_{k}" for k in range(len(crit))]
            if list(crit) != keys:
                raise Fatal(f"{rid}: choice criteria keys {list(crit)}")
            options = [crit[k] for k in keys]
        elif kind == "score" and isinstance(crit, list):
            options = list(crit)
        elif kind == "noul" and isinstance(crit, dict) and list(crit) == ["true", "false"]:
            options = [crit["true"], crit["false"]]
        else:
            raise Fatal(f"{rid}: unsupported question {kind!r}")
        state, question = req["state"], q.get("instructions")
        if not isinstance(state, str) or not isinstance(question, str) or \
                not all(isinstance(o, str) for o in options):
            raise Fatal(f"{rid}: state, question and options must be text (basal export reads them as strings)")
        public = {"id": rid, "type": kind, "state": state, "question": question, "options": options}
        if bb.public_request(public) != req:
            raise Fatal(f"{rid}: public_request(rebuilt record) differs from the request")
        d = decisions.get(rid)
        if d is None:
            raise Fatal(f"{rid}: not in {DECISIONS.format(short)}")
        if d.get("type") != kind:
            raise Fatal(f"{rid}: type {kind} vs stored {d.get('type')}")
        for k in ("upstream_fp32", "basal_rs_f16", "basal_rs_f32"):
            if not isinstance(d.get(k), list) or len(d[k]) != len(options):
                raise Fatal(f"{rid}: {k} has {len(d.get(k) or [])} values for {len(options)} options")
        out.append({"index": i, "source": "benchmark-data/v2/sequential.jsonl", "line": i + 1, "id": rid,
                    "type": kind, "state": state, "question": question, "options": options,
                    "gold": d.get("gold"), "dataset": d.get("dataset")})
    ids = [r["id"] for r in out]
    if [r["index"] for r in out] != list(range(900)) or len(set(ids)) != 900 or set(ids) != set(decisions):
        raise Fatal(f"{short}: expected indices 0..899 and 900 unique ids equal to the decisions file "
                    f"({len(out)} rows, {len(set(ids))} unique ids, {len(decisions)} decisions)")
    return out, decisions


def write_900(root, out_dir, shorts):
    """inputs-900-<short>/{bench.jsonl, systemone.jsonl (empty)} under `out_dir` (new); returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"sequential_sha256": SEQUENTIAL_SHA256, "models": {}}
    for short in shorts:
        rows, decisions = build_900(root, short)
        d = out_dir / f"inputs-900-{short}"
        d.mkdir()
        (d / "bench.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                       encoding="utf-8")
        (d / "systemone.jsonl").write_text("")
        summary["models"][short] = {
            "dir": d.name, "records": len(rows), "bench_sha256": mg.sha256(d / "bench.jsonl"),
            "systemone_sha256": mg.sha256(d / "systemone.jsonl"),
            "decisions_sha256": mg.sha256(root / DECISIONS.format(short)),
            "types": dict(Counter(r["type"] for r in rows)), "datasets": len({r["dataset"] for r in rows}),
            "roundtrip": "public_request(rebuilt) == request for 900/900"}
    shas = {s["bench_sha256"] for s in summary["models"].values()}
    summary["identical_across_models"] = len(shas) == 1
    mg.write_json(out_dir / "inputs-900.json", summary)
    return summary


# ---------------------------------------------------------------- bounded processes


REAP_S = 10  # after SIGKILL to a process group: the longest wait for its leader (else it stays in LIVE)


def run_bounded(rec, ctx, name, argv, delta, log, timeout, watch=None, physical=None, stdout=None, progress=None):
    """Runs `argv` in its own session until the absolute end min(start + `timeout`, run deadline); a log (or any *.log
    under `watch`) announcing a GEMM search stops it, and so does `progress()` returning "timeout" (a sub-step past its
    own limit). Waiting never goes past the end (pidfd, else a sleep of at most the time left); an exit seen only after
    the end is not timely: stopped like a process still running ("deadline" when the run deadline is the earlier end,
    else "timeout"), its exit status and CPU time kept (exit_observed, observed_after_budget). Returns a dict: exit
    (None: stopped), seconds, wall times, the reason it was stopped, CPU seconds of the process and its waited
    descendants (wait4), peak GPU memory of its session per physical GPU (`physical` set: PlacementWatch) and the
    foreign compute processes seen meanwhile. Exactly one entry goes to commands.jsonl. An interruption (signal,
    Ctrl-C, error) stops the process group (mg.stop: SIGTERM, SIGKILL after 60 s); one arriving during that cleanup
    ends it at once with SIGKILL to the group and a reap bounded by REAP_S. The entry records the interruption, then it
    is re-raised."""
    env = mg.with_env(ctx.env, delta)
    t0, w0 = time.monotonic(), time.time()
    own_end = t0 + timeout
    end = min(own_end, ctx.deadline)
    cause = "deadline" if ctx.deadline < own_end else "timeout"
    if end <= t0:
        ctx.deadline_hit = True
        return {"exit": None, "stopped": "deadline", "seconds": 0.0, "t_start_unix": round(w0, 3),
                "t_end_unix": round(w0, 3), "cpu_s": None, "placement_mib": None, "foreign_compute_processes": None}
    p = placement = foreign = pidfd = None
    stopped, usage, late, interrupted = None, None, False, None
    try:
        if stdout is None:
            p = mg.spawn(argv, env, log)
        else:  # stdout to its own file (JSON), stderr to the log
            with open(log, "wb") as err, open(stdout, "wb") as out:
                p = subprocess.Popen(argv, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            mg.LIVE.add(p)
        placement = mg.PlacementWatch(p.pid, ctx.by_bus) if physical is not None else None
        foreign = ForeignWatch(p.pid) if ctx.watch_foreign else None
        pidfd = exit_fd(p.pid)
        while True:
            pid, status, ru = os.wait4(p.pid, os.WNOHANG)
            now = time.monotonic()
            if pid:
                usage = ru.ru_utime + ru.ru_stime
                p.returncode = os.waitstatus_to_exitcode(status)
                if now > end:  # seen only after the end: not a timely exit
                    stopped, late = cause, True
                break
            if now >= end:
                stopped = cause
                break
            if searched(log, watch):
                stopped = "gemm-search"
                break
            if progress is not None and progress() == "timeout":
                stopped = "timeout"
                break
            wait_exit(pidfd, min(0.5, end - time.monotonic()))
    except BaseException as e:
        interrupted = e
        raise
    finally:
        graceful = p is not None and bool(stopped or interrupted or p.returncode is None)
        (code, killed, peak, seen, _, search), pending = settled([
            lambda: mg.stop(p) if graceful else None if p is None else p.returncode,
            lambda: kill_group(p) if p is not None else None,
            lambda: placement.close() if placement else None,
            lambda: foreign.close() if foreign else None,
            lambda: os.close(pidfd) if pidfd is not None else None,
            lambda: searched(log, watch)])
        code = killed if code is None else code
        seconds = round(time.monotonic() - t0, 3)
        if stopped == "deadline":
            ctx.deadline_hit = True
        if search:
            stopped = stopped or "gemm-search"
        result = {"exit": None if stopped else code, "stopped": stopped, "seconds": seconds,
                  "t_start_unix": round(w0, 3), "t_end_unix": round(w0 + seconds, 3),
                  "cpu_s": None if usage is None else round(usage, 3), "placement_mib": peak,
                  "foreign_compute_processes": seen}
        if late:
            result.update(exit_observed=code, observed_after_budget=True)
        entry = {"step": name, "argv": [str(a) for a in argv], "env": delta, **result}
        stop = interrupted or pending
        if stop is not None:
            entry.update(exit=None, stopped=f"interrupted ({type(stop).__name__})", stop_reason=stopped,
                         exit_after_stop=code, process_left=p is not None and p.returncode is None)
        with holding_signals():
            rec.command(entry)
        if pending is not None and interrupted is None:
            raise pending
    return result


def exit_fd(pid):
    """A pidfd of `pid` (readable once it exits), or None where the kernel or Python has none (then polling)."""
    try:
        return os.pidfd_open(pid)
    except (AttributeError, OSError):
        return None


def wait_exit(pidfd, seconds):
    """Waits at most `seconds` (never negative), returning early when the process of `pidfd` exits."""
    seconds = max(0.0, seconds)
    if pidfd is None:
        time.sleep(seconds)
        return
    poll = select.poll()
    poll.register(pidfd, select.POLLIN)
    poll.poll(seconds * 1000)


def kill_group(p):
    """SIGKILL to the process group of `p` (descendants left behind by a leader that exited, or every process after an
    interrupted graceful stop), then a reap of the leader bounded by REAP_S; `p` leaves LIVE only once reaped (else
    the exit handler stops it again). Its exit status, or None."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        p.wait(REAP_S)
    except subprocess.TimeoutExpired:
        return None
    mg.LIVE.discard(p)
    return p.returncode


def settled(steps):
    """Runs every step of a cleanup in order even when one raises or is interrupted (the first signal arriving during
    it: terminate ignores the later ones); (the results, None for a step that raised; the first exception or None)."""
    results, first = [], None
    for step in steps:
        try:
            results.append(step())
        except BaseException as e:
            results.append(None)
            first = first or e
    return results, first


@contextlib.contextmanager
def holding_signals():
    """In the main thread SIGTERM, SIGHUP and SIGINT are held back during the block and raised again after it (with the
    handlers restored), as rp.write_records does: a record is never cut or lost by the first signal."""
    held = []
    main = threading.current_thread() is threading.main_thread()
    old = {sig: signal.signal(sig, lambda signum, _frame: held.append(signum))
           for sig in rp.STOP_SIGNALS} if main else {}
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if held:
            signal.raise_signal(held[0])


def searched(log, watch):
    """A GEMM search was announced in `log` or in a *.log under `watch`."""
    paths = [Path(log)] + (sorted(Path(watch).glob("*.log")) if watch else [])
    for path in paths:
        try:
            if SEARCH in path.read_text(errors="replace"):
                return True
        except OSError:
            pass
    return False


class ForeignWatch:
    """Compute processes on any GPU outside the session of `root_pid`, sampled every 2 s (an objective outside
    disturbance); records also the samples nvidia-smi could not give."""

    def __init__(self, root_pid):
        self.root, self.seen, self.samples, self.failed, self.gone = root_pid, {}, 0, 0, 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self):
        while not self.stopped.wait(2.0):
            try:
                rows = iso.compute_apps()
            except (StepFailed, subprocess.SubprocessError, OSError):
                self.failed += 1
                continue
            self.samples += 1
            for row in rows:
                try:
                    pid = int(row[0])
                    sid = os.getsid(pid)
                except ProcessLookupError:  # ended since the query: not attributable to either side
                    self.gone += 1
                    continue
                except (ValueError, IndexError, PermissionError):  # another user's or an unreadable process
                    sid, pid = None, row[0] if row else "?"
                if sid != self.root:
                    self.seen[str(pid)] = self.seen.get(str(pid), 0) + 1

    def close(self):
        self.stopped.set()
        self.thread.join(10)
        return {"samples": self.samples, "failed_samples": self.failed, "foreign_pids": self.seen,
                "ended_before_attribution": self.gone}


def judge_run(rec, label, r, what="completed"):
    """FAIL for a non-zero exit, a timeout (never a pass), a GEMM search; GAP at the deadline. True when it passed."""
    late = f"; exit {r.get('exit_observed')} seen only after it (not timely)" if r.get("observed_after_budget") else ""
    if r["stopped"] == "deadline":
        rec.check(f"{label}: {what}", "GAP", f"stopped at the run deadline after {r['seconds']:.0f} s{late}")
        return False
    if r["stopped"] == "gemm-search":
        rec.check(f"{label}: {what}", "FAIL", "the log announces a GEMM search (an explicit bundled table was given)")
        return False
    if r["stopped"] == "timeout":
        rec.check(f"{label}: {what}", "FAIL", f"no exit within the timeout ({r['seconds']:.0f} s, stopped){late}")
        return False
    return rec.expect(f"{label}: {what}", r["exit"] == 0, f"exit {r['exit']}")


# ---------------------------------------------------------------- exports and comparisons


def layers_of(m):
    return json.loads((m["dir"] / "config.json").read_text())["num_hidden_layers"]


def export(rec, ctx, name, short, binary, inputs, mode="tree", quant=None, dtype=None, table=True, device=None,
           visible=None):
    """One `basal export` (cached by `name` within the run): bounded by --export-timeout (past it: FAIL and Fatal, a
    suspected hung GPU), every input exported, memory only on the expected physical GPU, the manifest's
    quantization. Returns the output directory, or None when it did not pass."""
    if name in ctx.exports:
        return ctx.exports[name]
    guard(ctx, rec, f"export {name}")
    m = ctx.model[short]
    out = ctx.out / "exports" / name
    out.parent.mkdir(exist_ok=True)
    argv = [ctx.binary[binary], "--color", "never", "export", "--model", m["dir"]]
    if dtype:
        argv += ["--dtype", dtype]
    if table:
        argv += ["--gemm-table", ctx.tables[short]]
    argv += MODES[mode] if mode else []
    if quant is not None:
        argv += ["--quantization", quant]
    if device is not None:
        argv += ["--device", str(device)]
    argv += ["--inputs", inputs, "--out", out]
    delta = {"CUDA_VISIBLE_DEVICES": visible} if visible else {}
    physical = int(visible) if visible else (device or 0)
    r = run_bounded(rec, ctx, f"export-{name}", argv, delta, out.parent / f"{name}.log", ctx.args.export_timeout,
                    physical={physical})
    ctx.exports[name] = None
    ctx.export_runs[name] = {k: r[k] for k in ("exit", "stopped", "seconds", "cpu_s")}
    if r["stopped"] == "timeout":
        rec.check(f"export {name}: completed", "FAIL", f"no exit within --export-timeout {ctx.args.export_timeout} s")
        raise Fatal(f"export {name} exceeded --export-timeout: suspected hung GPU, nothing later runs")
    if not judge_run(rec, f"export {name}", r):
        if r["stopped"] == "deadline":
            raise Deadline(name)
        return None
    want = (mg.line_count(Path(inputs) / "bench.jsonl"), mg.line_count(Path(inputs) / "systemone.jsonl"))
    got = (mg.line_count(out / "bench.jsonl"), mg.line_count(out / "systemone.jsonl"))
    if not rec.expect(f"export {name}: every input exported", got == want, f"{got} lines, inputs {want}"):
        return None
    mg.placement_check(rec, f"export {name}: memory only on physical GPU {physical}", r["placement_mib"], {physical})
    manifest = json.loads((out / "manifest.json").read_text())
    ctx.export_runs[name]["load_s"] = manifest.get("run", {}).get("load_s")
    q = manifest.get("backend", {}).get("quantization")
    if binary == "candidate":
        want_q = quant or "none"
        ok = manifest.get("run", {}).get("quantization") == want_q and \
            (q is None if want_q == "none" else isinstance(q, dict) and q.get("matrices") == 4 * layers_of(m))
        rec.expect(f"export {name}: manifest quantization {want_q}", ok,
                   f"run.quantization {manifest.get('run', {}).get('quantization')!r}, backend.quantization "
                   f"{json.dumps(q)[:300] if q else None}")
        if isinstance(q, dict):
            ctx.q8_diagnostics[name] = q
    ctx.exports[name] = out
    return out


def compare(rec, ctx, label, a, b, out_name):
    """`basal compare --a a --b b` (candidate binary, no GPU work); its JSON or None."""
    d = ctx.out / "compare"
    d.mkdir(exist_ok=True)
    cmp_out = d / f"{out_name}.json"
    if cmp_out.exists():
        return json.loads(cmp_out.read_text())
    r = run_bounded(rec, ctx, f"compare-{out_name}", [ctx.binary["candidate"], "--color", "never", "compare", "--a",
                                                      a, "--b", b, "--out", cmp_out], {}, d / f"{out_name}.log", 900)
    if r["stopped"] == "deadline":
        rec.check(f"basal compare {label}", "GAP", "stopped at the run deadline")
        raise Deadline(label)
    if r["exit"] != 0 or not cmp_out.is_file():
        rec.check(f"basal compare {label}", "FAIL", f"exit {r['exit']} {r['stopped'] or ''}".strip())
        return None
    return json.loads(cmp_out.read_text())


def same_bytes(rec, label, a, b):
    hashes = {f: (mg.sha256(a / f), mg.sha256(b / f)) for f in ("bench.jsonl", "systemone.jsonl")}
    return rec.expect(f"bitwise {label}", all(x == y for x, y in hashes.values()),
                      ", ".join(f"{f} {'equal' if x == y else 'DIFFERENT'}" for f, (x, y) in hashes.items()))


def identical(rec, ctx, label, a, b, out_name):
    """Byte-equal exports and a `basal compare` without any difference."""
    same_bytes(rec, label, a, b)
    v = compare(rec, ctx, label, a, b, out_name)
    if v is not None:
        problems = mg.compare_problems(v)
        rec.expect(f"basal compare {label}: no difference", not problems, "; ".join(problems[:5]))


def reference(ctx, short):
    return ctx.root / REFERENCE.format(short)


# ---------------------------------------------------------------- correctness


def step_none(rec, ctx):
    """The default path: candidate without the option and with --quantization none byte-equal to the phase-02 base."""
    for short in ctx.shorts:
        ref = reference(ctx, short)
        for mode in ("tree", "single"):
            base = export(rec, ctx, f"{short}-{mode}-base-dev0", short, "base", ref, mode)
            variants = [(f"{short}-{mode}-none-dev0", None, None), (f"{short}-{mode}-none-explicit-dev0", "none", None)]
            if mode == "tree":
                variants.append((f"{short}-tree-none-dev1", None, 1))
            for name, quant, device in variants:
                cand = export(rec, ctx, name, short, "candidate", ref, mode, quant=quant, device=device)
                if base is None or cand is None:
                    rec.check(f"none {name} vs base", "FAIL", "export missing")
                    continue
                identical(rec, ctx, f"{name} vs {short}-{mode}-base-dev0", base, cand, f"{name}-vs-base")
                db = json.loads((base / "manifest.json").read_text())["backend"]
                dc = json.loads((cand / "manifest.json").read_text())["backend"]
                ok, detail = same_backend(db, dc, device or 0)
                rec.expect(f"none {name}: backend description equal to the base (device: ordinal {device or 0})", ok,
                           detail)


CUDA_DEVICE = re.compile(r"^(NVIDIA GPU through candle Device::new_cuda\()(\d+)(\))$")


def same_backend(base, cand, ordinal):
    """The candidate's backend description equal to the base's (an export on GPU 0) in every field except `device`,
    which must name `ordinal` in the base's wording: (ok, detail)."""
    mb, mc = CUDA_DEVICE.match(str(base.get("device"))), CUDA_DEVICE.match(str(cand.get("device")))
    if not mb or not mc or mb.group(2) != "0" or mc.group(2) != str(ordinal) or mb.group(1, 3) != mc.group(1, 3):
        return False, iso.scrub(f"device {base.get('device')!r} (base) / {cand.get('device')!r}, expected ordinal 0 "
                                f"/ {ordinal}")
    others = [k for k in sorted(set(base) | set(cand))
              if k != "device" and ((k in base) != (k in cand) or base.get(k) != cand.get(k))]
    return not others, f"fields differing {others}" if others else f"device ordinal {ordinal}, every other field equal"


def step_q8(rec, ctx):
    """q8_0 results do not depend on the batching mode, prefix cache, card or CUDA_VISIBLE_DEVICES numbering."""
    for short in ctx.shorts:
        ref_inputs = reference(ctx, short)
        ref = export(rec, ctx, f"{short}-tree-q8-dev0", short, "candidate", ref_inputs, "tree", quant="q8_0")
        for name, mode, device, visible in ((f"{short}-single-q8-dev0", "single", None, None),
                                            (f"{short}-tree-noprefix-q8-dev0", "tree-noprefix", None, None),
                                            (f"{short}-tree-q8-dev1", "tree", 1, None),
                                            (f"{short}-tree-q8-visible1-device0", "tree", 0, "1")):
            out = export(rec, ctx, name, short, "candidate", ref_inputs, mode, quant="q8_0", device=device,
                         visible=visible)
            if ref is None or out is None:
                rec.check(f"q8 {name} vs {short}-tree-q8-dev0", "FAIL", "export missing")
                continue
            identical(rec, ctx, f"{name} vs {short}-tree-q8-dev0", ref, out, f"{name}-vs-tree-q8-dev0")


def serve_yaml(path, port, cache, entries):
    """Serve configuration; strings as JSON (valid YAML scalars); keys only when given (the base binary rejects
    `quantization`)."""
    lines = [f"addr: 127.0.0.1:{port}", "metrics: true", "max_inflight: 4096", f"gemm_cache: {json.dumps(str(cache))}",
             "models:"]
    for e in entries:
        first = "path" if "dir" in e else "repo"
        lines.append(f"  - {first}: {json.dumps(str(e.get('dir', e.get('repo'))))}")
        for k in ("gemm_table", "dtype", "quantization", "device", "replica_devices", "long_tokens"):
            if k in e:
                v = str(e[k]) if isinstance(e[k], Path) else e[k]
                lines.append(f"    {k}: {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def step_invalid(rec, ctx):
    """Refused weight-format configurations: a validation message, before any download, load or GEMM search."""
    d = ctx.out / "correctness" / "invalid"
    d.mkdir(parents=True)
    port, cache, ref = ctx.args.port, d / "gemm-cache", reference(ctx, "mini")
    absent = {"repo": ABSENT}
    request = d / "request.json"
    request.write_text(json.dumps({"state": "Klient pisze, że chce oddać telefon.",
                                   "questions": {"q": {"type": "noul", "instructions": "Czy klient prosi o zwrot?"}}}))

    def cfg(label, **kw):
        return serve_yaml(d / f"{label}.yml", port, cache, [dict(absent, **kw)])

    plain = cfg("plain", dtype="f16")
    absent_ref = ["--model", ABSENT]
    cases = [  # (label, arguments after the binary, expected message)
        ("serve-yaml-bf16", ["serve", "--config", cfg("yaml-bf16", dtype="bf16", quantization="q8_0")],
         "needs dtype f16"),
        ("serve-yaml-f32", ["serve", "--config", cfg("yaml-f32", dtype="f32", quantization="q8_0")],
         "needs dtype f16"),
        ("serve-yaml-unknown", ["serve", "--config", cfg("yaml-unknown", dtype="f16", quantization="q4_0")],
         "unknown quantization"),
        ("serve-cli-with-config", ["serve", "--config", plain, "--quantization", "q8_0"],
         "--quantization applies to --model"),
        ("serve-cli-candle", ["serve", *absent_ref, "--quantization", "q8_0", "--kernels", "candle"],
         "needs kernels fused"),
        ("serve-cli-bf16", ["serve", *absent_ref, "--quantization", "q8_0", "--dtype", "bf16"], "needs dtype f16"),
        ("export-unknown", ["export", *absent_ref, "--quantization", "Q8_0", "--inputs", ref, "--out",
                            d / "export-unknown-out"], "unknown quantization"),
        ("bench-f32", ["bench", *absent_ref, "--quantization", "q8_0", "--dtype", "f32", "--reference", ref, "--out",
                       d / "bench-f32.json"], "needs dtype f16"),
        ("benchmark-candle", ["benchmark", *absent_ref, "--quantization", "q8_0", "--kernels", "candle", "--out",
                              d / "benchmark-candle.json"], "needs kernels fused"),
        ("decide-bf16", ["decide", *absent_ref, "--quantization", "q8_0", "--dtype", "bf16", "--request", request],
         "needs dtype f16"),
        ("profile-candle", ["profile", *absent_ref, "--quantization", "q8_0", "--kernels", "candle", "--reference",
                            ref], "needs kernels fused"),
    ]
    for label, extra, needle in cases:
        guard(ctx, rec, f"invalid {label}")
        mg.port_free_or_fail(port)
        log = d / f"{label}.log"
        r = run_bounded(rec, ctx, f"invalid-{label}", [ctx.binary["candidate"], "--color", "never", *extra],
                        {"HF_HUB_OFFLINE": "1"}, log, 120, physical=set())
        if r["stopped"] == "deadline":
            judge_run(rec, f"invalid {label}", r)
            raise Deadline(label)
        text = log.read_text(errors="replace")
        early = not any(x in text for x in ("fetching the model files", "loaded on GPU", SEARCH, "serving on http"))
        rec.expect(f"invalid {label}: refused before download and load",
                   r["exit"] not in (0, None) and needle in text and early,
                   f"exit {r['exit']} after {r['seconds']:.1f} s, expected message {needle!r}, early {early}")
        if r["placement_mib"]:
            rec.check(f"invalid {label}: no GPU memory of the process", "FAIL", f"{r['placement_mib']}")
        elif r["seconds"] < 1.0:
            rec.check(f"invalid {label}: no GPU memory of the process", "GAP",
                      f"ended after {r['seconds']:.2f} s, before a 0.5 s placement sample (log shows no load)")
        else:
            rec.check(f"invalid {label}: no GPU memory of the process", "PASS", "none in the 0.5 s samples")
    # doctor: the same check as a model failure; a valid entry with its estimate
    mini = ctx.model["mini"]
    for label, entries, want_fail in (
            ("doctor-bf16", [dict(absent, dtype="bf16", quantization="q8_0")], True),
            ("doctor-q8-mini", [{"dir": mini["dir"], "dtype": "f16", "quantization": "q8_0",
                                 "gemm_table": ctx.tables["mini"]}], False)):
        guard(ctx, rec, f"invalid {label}")
        conf = serve_yaml(d / f"{label}.yml", port, cache, entries)
        log = d / f"{label}.json"
        r = run_bounded(rec, ctx, label, [ctx.binary["candidate"], "--color", "never", "doctor", "--config", conf,
                                          "--json"], {"HF_HUB_OFFLINE": "1"}, log, 180)
        if r["stopped"] == "deadline":
            judge_run(rec, label, r)
            raise Deadline(label)
        if r["stopped"]:  # past its own timeout (also an exit seen only after it): no report is judged
            judge_run(rec, label, r)
            continue
        try:
            report = rp.doctor_report(log.read_text(errors="replace"))
        except ValueError:
            rec.check(label, "FAIL", f"no JSON report (exit {r['exit']})")
            continue
        models = [c for c in report.get("checks", []) if c.get("area") == "models"]
        fails = [c["what"] for c in models if c.get("level") == "fail"]
        if want_fail:
            rec.expect(f"{label}: q8_0 with bf16 reported as a model failure",
                       any("needs dtype f16" in f for f in fails), iso.scrub(f"{fails}"))
        else:
            est = [c["what"] for c in models if "q8_0 weights" in c.get("what", "")]
            rec.expect(f"{label}: valid q8_0 entry without failure, weight estimate reported", est and not fails,
                       iso.scrub(f"estimate {est}, failures {fails}"))
    rec.check("invalid q8_0 on Metal or Intel", "GAP", "no such device in this run; static review and the "
              "Intel/Metal compile checks only")


def bounded_request(method, url, path, raw, end, active=None):
    """One HTTP request (http.client; headers as rp.send_raw for a body, none else) bounded as a whole by the absolute
    monotonic time `end`: connecting, sending, the headers and the body together. Each socket operation has at most
    the time left at the start, and a watchdog shuts the socket down at `end`, so no answer arriving piecemeal keeps
    it open longer (up to the scheduling of the watchdog thread). The socket is kept from the connect to the end:
    getresponse() drops conn.sock for a response that closes the connection (HTTP/1.0, Connection: close) while the
    response still reads from it; a cut before the socket is kept stops the request right after its connect (itself
    bounded by the time left). `active` (a set): the request's cut function is in
    it while it runs (send_items cuts every request in flight with "interrupted" when the run is interrupted). Returns
    a dict: status (None: no complete response; a response cut by the watchdog or an interruption before it was read
    to its end is never complete), headers, body (also the bytes of an incomplete response), received_status, error
    and late (the response was not complete by `end`, judged on the monotonic clock: whatever arrived, a complete
    status 200 included, is not timely)."""
    out = {"status": None, "headers": {}, "body": b"", "received_status": None, "error": None, "late": False}
    left = end - time.monotonic()
    if left <= 0:
        out.update(error="not sent: no time left", late=True)
        return out
    parts = urllib.parse.urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=left)
    lock, state, resp = threading.Lock(), {"closed": False, "cut": None, "sock": None}, None

    def cut(by="end"):
        with lock:
            if state["closed"]:
                return
            state["cut"] = by
            if state["sock"] is not None:
                try:
                    state["sock"].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    timer = threading.Timer(left, cut)
    timer.daemon = True
    if active is not None:
        active.add(cut)
    try:
        timer.start()
        conn.connect()
        with lock:
            state["sock"] = conn.sock
            if state["cut"] is not None:
                raise TimeoutError(f"cut ({state['cut']}) while connecting")
        conn.request(method, path, body=raw, headers={"content-type": "application/json"} if raw is not None else {})
        resp = conn.getresponse()
        out["received_status"] = resp.status
        headers = {k.lower(): v for k, v in resp.getheaders()}
        body = resp.read()
        with lock:  # read to its end: complete unless cut before
            complete, state["closed"] = state["cut"] is None, True
        out.update(body=body)
        if complete:
            out.update(status=resp.status, headers=headers)
        else:
            out["error"] = f"incomplete: cut after {len(body)} bytes of the body"
    except http.client.IncompleteRead as e:
        out.update(body=e.partial, error=f"IncompleteRead: {len(e.partial)} bytes")
    except (OSError, http.client.HTTPException, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        timer.cancel()
        with lock:
            state["closed"] = True
            if resp is not None:
                resp.close()
            conn.close()
            if state["sock"] is not None:
                state["sock"].close()
        if active is not None:
            active.discard(cut)
    if state["cut"] == "interrupted":
        out["error"] = f"cut by the interruption of the run ({out['error']})"
    elif state["cut"] == "end" or time.monotonic() > end:
        out["late"] = True
        out["error"] = out["error"] or "complete only after the time allowed"
    return out


def send_bounded(url, path, raw, end, active=None):
    """rp.send_raw (the same record and canonical answer) with the whole request bounded by `end` (bounded_request).
    A response not complete by `end` gets late true, its error and the bytes received (`received`, canonical like an
    answer); a complete late one keeps its status and answer as evidence, the caller treats it as not answered."""
    r = {"t_start": rp.now()}
    x = bounded_request("POST", url, path, raw, end, active)
    if x["status"] is not None:
        try:
            h = x["headers"]
            answer = mg.normalized(x["body"])
            r.update(status=x["status"], answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                     retry_after=h.get("retry-after"), queue_ms=iso.header_number(h, "x-basal-queue-ms", float),
                     compute_ms=iso.header_number(h, "x-basal-compute-ms", float),
                     batch=iso.header_number(h, "x-basal-batch-requests", int))
        except ValueError as e:
            r.update(status=None, error=f"{type(e).__name__}: {e}")
    else:
        r.update(status=None, error=x["error"])
        if x["received_status"] is not None or x["body"]:
            r.update(received_status=x["received_status"], received_bytes=len(x["body"]),
                     received=mg.normalized(x["body"]))
    if x["late"]:
        r.update(late=True, error=x["error"])
    r["t_done"] = rp.now()
    return r


def send_items(ctx, url, items, clients, phase, out):
    """rp.run_items (same order, records and slots: out[k] as soon as item k is answered) with every request bounded as
    a whole (send_bounded) by min(start + --http-timeout, run deadline): no request starts after the deadline, none
    lasts past its end. The first failed request stops sending (no retry); an interruption cuts the requests in flight
    (the pool joins at once). Returns None, "transport" (an error), "timeout" (no complete answer within
    --http-timeout: suspected hung GPU) or "deadline" (stopped at the run deadline: not a failure); a request whose
    answer was not complete by its end has late "timeout" or "deadline" in its record."""
    stop, lock, why, active = threading.Event(), threading.Lock(), [], set()

    def halt(reason):
        with lock:
            if not why:
                why.append(reason)
        stop.set()

    def one(k):
        if stop.is_set():
            return
        t = time.monotonic()
        own = t + ctx.args.http_timeout
        end = min(own, ctx.deadline)
        cause = "deadline" if ctx.deadline < own else "timeout"
        if end <= t:
            halt("deadline")
            return
        rid, cls, path, raw = items[k]
        r = send_bounded(url, path, raw, end, active)
        r.update(phase=phase, id=rid, cls=cls, timeout_s=round(end - t, 3))
        if r.get("late"):
            r["late"] = cause
        out[k] = r
        if r.get("late"):
            halt(cause)
        elif r["status"] is None:
            halt("transport")

    if clients == 1:
        for k in range(len(items)):
            one(k)
            if stop.is_set():
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(clients) as pool:
            try:
                for _ in pool.map(one, range(len(items))):
                    pass
            except BaseException:  # before the pool joins its threads: nothing new starts, the requests are cut
                stop.set()
                for cut in list(active):
                    cut("interrupted")
                raise
    return why[0] if why else None


class Served:
    """A fresh server (YAML with the explicit bundled table): iso.Server (telemetry, log), readiness, placement,
    records written once by close (rp.close_and_write). Readiness, /health, /v1/models, every request and /metrics
    scrape are bounded as a whole by the run deadline as well as by their own timeouts (bounded_request); the graceful
    shutdown by --shutdown-timeout."""

    def __init__(self, rec, ctx, label, binary, entries, delta=None):
        guard(ctx, rec, f"serve {label}")
        self.rec, self.ctx, self.label = rec, ctx, label
        self.dir = ctx.out / "correctness" / "serve" / label
        self.dir.mkdir(parents=True)
        busy = iso.wait_gpus_idle()
        if busy:
            raise StepFailed(f"compute processes still on the GPUs before {label}: {len(busy)}")
        cfg = serve_yaml(self.dir / "serve.yml", ctx.args.port, self.dir / "gemm-cache", entries)
        self.srv = iso.Server(rec, ctx, self.dir, ctx.binary[binary], cfg, delta or {})
        self.url = self.srv.url
        self.batches, self.closed = [], False
        try:
            self.wait_ready()
            time.sleep(min(2.0, max(0.0, ctx.deadline - time.monotonic())))
            self.placement = mg.session_placement(self.srv.proc.pid, ctx.by_bus)
            _, self.health_raw = self.get("/health", 30, "GET /health")
            _, self.models_raw = self.get("/v1/models", 30, "GET /v1/models")
            log = (self.dir / "server.log").read_text(errors="replace")
            if SEARCH in log:
                raise StepFailed(f"server {label} announced a GEMM search")
        except BaseException:
            self.close()
            raise

    def budget(self, limit, what):
        """min(limit, time left before the run deadline); past it a GAP and Deadline (nothing is sent)."""
        left = self.ctx.deadline - time.monotonic()
        if left <= 0:
            self.ctx.deadline_hit = True
            self.rec.check(f"serve {self.label}: {what}", "GAP", "not sent: the run deadline passed")
            raise Deadline(f"{self.label} {what}")
        return min(limit, left)

    def get(self, path, limit, what):
        """GET `path` bounded as a whole by min(now + `limit`, run deadline): (status, body). Not complete when the run
        deadline (the earlier end) passes, before or during it: a GAP and Deadline, also for a status 200 arriving
        after it (not timely); past `limit` (also a complete answer seen only after it) or another transport error:
        OSError, as mg.http raises (readiness retries, a scrape returns no status, else a step FAIL). The bytes of a
        response that was late or incomplete are kept in late-<what>.bin."""
        self.budget(limit, what)  # nothing sent after the run deadline: GAP and Deadline
        t = time.monotonic()
        own = t + limit
        end = min(own, self.ctx.deadline)
        x = bounded_request("GET", self.url, path, None, end)
        if x["late"] or x["status"] is None:
            slug = re.sub(r"[^a-z0-9]+", "-", what.lower()).strip("-")
            if x["body"]:
                (self.dir / f"late-{slug}.bin").write_bytes(x["body"])
            detail = (f"status {x['received_status']}, {len(x['body'])} bytes received after {time.monotonic() - t:.3f}"
                      f" s (end {end - t:.3f} s): {x['error']}")
            if x["late"] and self.ctx.deadline < own:
                self.ctx.deadline_hit = True
                self.rec.check(f"serve {self.label}: {what}", "GAP",
                               f"no complete response before the run deadline, not used ({detail})")
                raise Deadline(f"{self.label} {what}")
            raise TimeoutError(detail) if x["late"] else ConnectionError(detail)
        return x["status"], x["body"]

    def wait_ready(self):
        """iso.Server.wait_ready (GET /health until 200, a probe at most 5 s, a pause of 1 s) bounded by the run
        deadline (GAP, Deadline) besides --ready-timeout (FAIL), whichever ends first."""
        limit = time.monotonic() + self.ctx.args.ready_timeout
        while True:
            self.srv.alive()
            now = time.monotonic()
            if now >= min(limit, self.ctx.deadline):
                if self.ctx.deadline <= limit:
                    self.budget(1.0, "readiness")  # the run deadline ended it first: GAP and Deadline
                raise StepFailed(f"server not ready after {self.ctx.args.ready_timeout} s")
            try:
                status, _ = self.get("/health", min(5.0, limit - now), "readiness")
                if status == 200:
                    return
            except OSError:
                pass
            time.sleep(max(0.0, min(1.0, limit - time.monotonic(), self.ctx.deadline - time.monotonic())))

    def scrape(self, path=None):
        """rp.scrape bounded as a whole (get: the run deadline passing first is a GAP and Deadline): (status, parsed
        samples), the body into `path`."""
        try:
            status, raw = self.get("/metrics", 30, "GET /metrics")
        except OSError as e:
            if path:
                path.write_text(f"# {type(e).__name__}: {e}\n")
            return None, []
        if path:
            path.write_bytes(raw)
        return status, rp.parse_metrics(raw.decode(errors="replace"))

    def run(self, items, clients, phase):
        """Every item once (send_items); the records stay in self.batches for close. Stopped at the deadline: GAP and
        Deadline (the phase is not judged); a request past --http-timeout: FAIL and Fatal; another transport error:
        FAIL and StepFailed."""
        slots = [None] * len(items)
        self.batches.append(slots)
        why = send_items(self.ctx, self.url, items, clients, phase, slots)
        records = [r for r in slots if r]
        late = [r for r in records if r.get("late")]
        if why == "deadline":
            self.ctx.deadline_hit = True
            self.rec.check(f"serve {self.label} {phase}", "GAP",
                           f"stopped at the run deadline: {len(records)} of {len(items)} sent (kept; {len(late)} not "
                           "complete by the deadline), not judged")
            raise Deadline(f"{self.label} {phase}")
        if why:
            failed = late or [r for r in records if r["status"] is None]
            first = f"{failed[0]['id']}: {failed[0].get('error')}" if failed else "none recorded"
            self.rec.check(f"serve {self.label} {phase}: transport", "FAIL",
                           iso.scrub(f"{len(records)} of {len(items)} sent, first error {first}; server stopped"))
            if why == "timeout":
                raise Fatal(f"serve {self.label} {phase}: no answer within --http-timeout {self.ctx.args.http_timeout}"
                            " s: suspected hung GPU, no later GPU work runs")
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
        if self.closed:
            return
        self.closed = True
        rp.close_and_write(self.rec, f"serve {self.label}", self.srv.close, self.dir / "requests.jsonl",
                           lambda: [r for batch in self.batches for r in batch if r])


def canonical_body(v):
    """A native answer or error body as an HTTP record holds it (mg.normalized: without usage.latency_ms, keys
    sorted)."""
    if isinstance(v, dict) and isinstance(v.get("usage"), dict):
        v = json.loads(json.dumps(v))
        v["usage"].pop("latency_ms", None)
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def expected_status(v):
    """The /v1/systemone status of a native body (serve.rs error_body, TypeSafe): an answer 200; an error
    (DecideError::to_json) Unsupported 422, Validation (`detail` a list of field errors) 422, Usage (`detail` a message
    or an api_usage_error object) 400. None for anything else (an internal error, an unknown shape): no expected
    status, never a pass."""
    if not isinstance(v, dict):
        return None
    if "answers" in v and "error" not in v and "detail" not in v:
        return 200
    if set(v) == {"error", "detail"} and v["error"] == "unsupported":
        return 422
    if set(v) == {"detail"}:
        return 422 if isinstance(v["detail"], list) else 400 if isinstance(v["detail"], (str, dict)) else None
    return None


def served_inputs(rec, ctx, items):
    """Native inputs with the System One requests the serve step sends: bench.jsonl of the mini reference byte for byte,
    systemone.jsonl one line per case (reference order) whose `request` is the exact HTTP body of that case (the
    bytes of rp.http_items: mg.workload sets `model` to the served model). The original reference and its exports stay
    unchanged. Returns (directory, {item id: raw body}), or None after a FAIL (a case not sent exactly once)."""
    src = reference(ctx, "mini")
    d = ctx.out / "correctness" / "serve-inputs"
    d.mkdir(parents=True)
    sent = [(rid, raw) for rid, _, path, raw in items if rid.startswith("short:") and path == "/v1/systemone"]
    bodies = dict(sent)
    cases = read_jsonl(src / "systemone.jsonl")
    ids = [f"short:{c['id']}" for c in cases]
    if len(bodies) != len(sent) or len(set(ids)) != len(ids) or set(ids) != set(bodies) or not ids:
        rec.check("serve inputs: every System One case sent exactly once", "FAIL",
                  f"{len(sent)} requests ({len(bodies)} ids), {len(ids)} cases ({len(set(ids))} ids), "
                  f"not sent {sorted(set(ids) - set(bodies))[:5]}, not a case {sorted(set(bodies) - set(ids))[:5]}")
        return None
    (d / "bench.jsonl").write_bytes((src / "bench.jsonl").read_bytes())
    with open(d / "systemone.jsonl", "w") as f:
        for c, k in zip(cases, ids):
            f.write(f'{{"id": {json.dumps(c["id"])}, "source": {json.dumps(c.get("source"))}, "request": '
                    f'{bodies[k].decode()}}}\n')
    rows = [{"id": k, "request_sha256": hashlib.sha256(bodies[k]).hexdigest(),
             "model_rewritten": json.loads(bodies[k]) != c["request"]} for c, k in zip(cases, ids)]
    readback = {f"short:{c['id']}": c["request"] for c in read_jsonl(d / "systemone.jsonl")}
    ok = rec.expect("serve inputs: served request bodies written exactly (bench.jsonl a copy of the reference)",
                    readback == {k: json.loads(v) for k, v in bodies.items()} and
                    mg.sha256(d / "bench.jsonl") == mg.sha256(src / "bench.jsonl"),
                    f"{len(ids)} cases, {sum(r['model_rewritten'] for r in rows)} with `model` set to the served model")
    summary = {"bench_sha256": mg.sha256(d / "bench.jsonl"), "systemone_sha256": mg.sha256(d / "systemone.jsonl"),
               "reference_systemone_sha256": mg.sha256(src / "systemone.jsonl"), "cases": rows}
    mg.write_json(d / "inputs.json", summary)
    ctx.results["serve_inputs"] = summary
    return (d, bodies) if ok else None


def served_decide(rec, ctx, k, cid, raw):
    """`basal decide` of one exact served request body (candidate, mini, the flags of the q8_0 tree export: table,
    q8_0, f16, fused, GPU 0; `--batching` only orders the export's bench items, a request runs Engine::decide in
    both): the served path for a case the export did not answer. Bounded and recorded like an export (past its timeout
    FAIL and Fatal, at the deadline GAP and Deadline). Returns its JSON when it is a typed native response (exit 0: an
    answer; exit 2: a Validation, Usage or Unsupported error), else None after a FAIL."""
    d = ctx.out / "correctness" / "serve-decide"
    d.mkdir(parents=True, exist_ok=True)
    slug = f"{k:02d}-{re.sub(r'[^A-Za-z0-9._-]+', '-', cid)}"
    request = d / f"{slug}.request.json"
    request.write_bytes(raw)
    label = f"serve decide {cid}"
    guard(ctx, rec, label)
    argv = gpu_argv(ctx, "candidate", "decide", "mini", "q8_0", ["--request", request])
    stdout = d / f"{slug}.json"
    r = run_bounded(rec, ctx, f"decide-{slug}", argv, {}, d / f"{slug}.log", ctx.args.export_timeout,
                    physical={0}, stdout=stdout)
    if r["stopped"]:
        judge_run(rec, label, r)
        stop_after(label, r)
        return None
    mg.placement_check(rec, f"{label}: memory only on physical GPU 0", r["placement_mib"], {0})
    try:
        v = json.loads(stdout.read_text())
    except (OSError, ValueError) as e:
        rec.check(f"{label}: typed native response", "FAIL", f"exit {r['exit']}, no JSON ({type(e).__name__})")
        return None
    status, sha = expected_status(v), hashlib.sha256(raw).hexdigest()
    same = mg.sha256(request) == sha
    ok = same and ((r["exit"] == 0 and status == 200) or (r["exit"] == 2 and status in (400, 422)))
    rec.expect(f"{label}: typed native response", ok, f"exit {r['exit']}, body for HTTP status {status}, request "
               f"sha256 {sha[:12]} ({'the' if same else 'NOT the'} body sent)")
    return v if ok else None


def served_expected(rec, ctx, served, bodies):
    """item id -> (status, canonical body, source) the q8_0 server must answer, from the q8_0 export of the served
    request bodies (`served`): its served answer (Engine::decide), or its error where the request was refused before
    the model ran. The export answers only requests its prepare_request accepts, and that one refuses as Unsupported
    what serve answers (check_capability: a single option, more than 10 choice options); Validation and Usage come
    from the parse and model check both paths share. So a case refused as Unsupported gets the served path from
    served_decide (one per such case, none named in advance); every other case comes from the export. A line whose
    request is not the exact body sent, or a body without an expected status, is a FAIL. Returns the map (None after
    a FAIL) and the provenance rows."""
    out, rows, problems = {}, [], []
    lines = read_jsonl(served / "systemone.jsonl")
    for k, line in enumerate(lines):
        rid = f"short:{line['id']}"
        raw = bodies.get(rid)
        row = {"id": rid, "request_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None}
        rows.append(row)
        if raw is None or line.get("request") != json.loads(raw):
            problems.append(f"{rid}: the export's request is not the body sent")
            continue
        refused = line.get("error") if "answer" not in line else None
        if isinstance(refused, dict) and refused.get("error") == "unsupported":
            v = served_decide(rec, ctx, k, line["id"], raw)
            row["source"] = "basal decide"
            if v is None:
                problems.append(f"{rid}: no typed basal decide response")
                continue
        else:
            v, row["source"] = line.get("answer", refused), "export"
        status = expected_status(v)
        row.update(expected_status=status, expected_sha256=hashlib.sha256(canonical_body(v).encode()).hexdigest())
        if status is None:
            problems.append(f"{rid}: no expected status ({row['source']}: {canonical_body(v)[:120]})")
            continue
        out[rid] = (status, canonical_body(v), row["source"])
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(bodies):
        problems.append(f"export cases {len(rows)}, served System One requests {len(bodies)}")
    by = Counter(r.get("source") for r in rows)
    rec.expect("serve q8_0 reference: an expected (status, body) for every served System One request", not problems,
               f"{len(out)}/{len(bodies)}: {by['export']} from the export of the served bodies, "
               f"{by['basal decide']} from basal decide" + iso.scrub("".join(f"; {p}" for p in problems[:3])))
    return (None if problems else out), rows


def served_unchanged(rec, original, served):
    """What the export of the served request bodies shares with the q8_0 tree export of the reference (same binary,
    flags, GPU): bench.jsonl, and the System One line of every case whose request is unchanged (its `model` already
    the served model), byte-equal. A case whose request differs (`model` rewritten) is not compared."""
    rec.expect("serve inputs: bench.jsonl of mini-tree-q8-served-dev0 byte-equal to mini-tree-q8-dev0",
               mg.sha256(served / "bench.jsonl") == mg.sha256(original / "bench.jsonl"))

    def lines(path):
        return {json.loads(t)["id"]: t for t in (path / "systemone.jsonl").read_text().splitlines() if t.strip()}

    a, b = lines(original), lines(served)
    same = [k for k in a if k in b and json.loads(a[k]).get("request") == json.loads(b[k]).get("request")]
    bad = [k for k in same if a[k] != b[k]]
    name = "serve inputs: System One lines of the unchanged requests byte-equal to mini-tree-q8-dev0"
    detail = f"{len(same) - len(bad)}/{len(same)} equal, differing {bad[:5]}; {len(b) - len(same)} with `model` " \
             "rewritten not compared"
    if same:
        rec.expect(name, not bad, detail)
    else:
        rec.check(name, "GAP", detail)


def served_parity(rec, ctx, q8_ref, want, provenance, ids):
    """Every served System One request of the q8_0 server equal in (status, canonical body) to the native q8_0
    reference of the same request body (served_expected); the table goes to serve-inputs/q8-served-comparison.json."""
    name = "serve q8-cand-dev0: System One (status, answer: multi, evidence, 400/422) equal to the native q8_0 " \
           "reference of the same requests"
    if want is None:
        rec.check(name, "FAIL", "no complete native reference of the served requests (see the serve inputs checks)")
        return
    table, bad = [], []
    for row in provenance:
        got, exp = q8_ref.get(row["id"]), want.get(row["id"])
        equal = got is not None and exp is not None and got == exp[:2]
        table.append(dict(row, http_status=got[0] if got else None, equal=equal,
                          http_sha256=hashlib.sha256(got[1].encode()).hexdigest() if got and got[1] else None))
        if not equal:
            bad.append(f"{row['id']} (HTTP {got[0] if got else None}, expected {exp[0] if exp else None})")
    missing = sorted(set(ids) ^ {r["id"] for r in provenance})
    by = Counter(r.get("source") for r in provenance)
    summary = {"cases": len(table), "equal": sum(r["equal"] for r in table), "sources": dict(by),
               "http_statuses": dict(Counter(str(r["http_status"]) for r in table)), "not_compared": missing}
    mg.write_json(ctx.out / "correctness" / "serve-inputs" / "q8-served-comparison.json",
                  {**summary, "rows": table})
    ctx.results["q8_served_comparison"] = summary
    rec.expect(name, not bad and not missing and len(table) == len(ids) > 0,
               f"{summary['equal']}/{len(ids)} equal ({by['export']} expected from the export of the served bodies, "
               f"{by['basal decide']} from basal decide), HTTP statuses {summary['http_statuses']}, differing "
               f"{bad[:5]}, not compared {missing[:5]}")


def step_serve(rec, ctx):
    mini = ctx.model["mini"]
    name, table = mini["name"], ctx.tables["mini"]
    ctx.inputs, ctx.long_file = reference(ctx, "mini"), ctx.root / LONG
    items = rp.http_items(ctx, name)
    base = {"dir": mini["dir"], "gemm_table": table, "dtype": "f16", "long_tokens": ctx.args.long_tokens}
    q8_export = export(rec, ctx, "mini-tree-q8-dev0", "mini", "candidate", reference(ctx, "mini"), "tree",
                       quant="q8_0")
    want, provenance = None, []
    derived = served_inputs(rec, ctx, items)
    if derived is not None:
        served = export(rec, ctx, "mini-tree-q8-served-dev0", "mini", "candidate", derived[0], "tree", quant="q8_0")
        if served is not None:
            if q8_export is not None:
                served_unchanged(rec, q8_export, served)
            want, provenance = served_expected(rec, ctx, served, derived[1])

    s = Served(rec, ctx, "none-base-dev0", "base", [dict(base, device=0)])
    try:
        mg.placement_check(rec, "serve none-base-dev0: memory only on physical GPU 0", s.placement, {0})
        recs = s.run(items, 1, "sequential")
        errors = [r["id"] for r in recs if r["status"] is None or r["status"] >= 500]
        ok = rec.expect("serve none-base-dev0: HTTP reference", len(recs) == len(items) and not errors,
                        f"{Counter(str(r['status']) for r in recs)}, transport/5xx {errors[:5]}")
        http_ref = {r["id"]: (r["status"], r["answer"]) for r in recs}
        s.scrape(s.dir / "metrics.txt")
        families = rp.metric_families((s.dir / "metrics.txt").read_text(errors="replace"))
        health, models = s.health_raw, s.models_raw
        s.shutdown_check()
    finally:
        s.close()
    if not ok:
        raise StepFailed("no HTTP reference")
    ctx.results["http_reference_sha256"] = iso.canonical_sha(http_ref)

    def same_surface(srv, label):
        rec.expect(f"serve {label}: /health bytes equal to the base", srv.health_raw == health)
        rec.expect(f"serve {label}: /v1/models bytes equal to the base", srv.models_raw == models)
        status, _ = srv.scrape(srv.dir / "metrics.txt")
        missing = [f for f in families if f not in rp.metric_families((srv.dir / "metrics.txt").read_text(
            errors="replace"))]
        rec.expect(f"serve {label}: every base metric family with the same HELP and TYPE",
                   status == 200 and not missing, f"missing or changed {missing[:3]}")

    s = Served(rec, ctx, "none-cand-dev0", "candidate", [dict(base, device=0)])
    try:
        mg.placement_check(rec, "serve none-cand-dev0: memory only on physical GPU 0", s.placement, {0})
        rp.parity(rec, "serve none-cand-dev0 sequential", s.run(items, 1, "sequential"), http_ref, len(items))
        same_surface(s, "none-cand-dev0")
        s.shutdown_check()
    finally:
        s.close()

    s = Served(rec, ctx, "q8-cand-dev0", "candidate", [dict(base, device=0, quantization="q8_0")])
    try:
        mg.placement_check(rec, "serve q8-cand-dev0: memory only on physical GPU 0", s.placement, {0})
        recs = s.run(items, 1, "sequential")
        errors = [r["id"] for r in recs if r["status"] is None or r["status"] >= 500]
        ok = rec.expect("serve q8-cand-dev0: q8_0 HTTP reference", len(recs) == len(items) and not errors,
                        f"{Counter(str(r['status']) for r in recs)}, transport/5xx {errors[:5]}")
        q8_ref = {r["id"]: (r["status"], r["answer"]) for r in recs}
        served_parity(rec, ctx, q8_ref, want, provenance, [k for k, _, path, _ in items
                                                           if k.startswith("short:") and path == "/v1/systemone"])
        same_surface(s, "q8-cand-dev0")
        s.shutdown_check()
    finally:
        s.close()
    if not ok:
        raise StepFailed("no q8_0 HTTP reference")

    s = Served(rec, ctx, "q8-replicas", "candidate", [dict(base, replica_devices=[0, 1], quantization="q8_0")])
    try:
        mg.placement_check(rec, "serve q8-replicas: memory on physical GPUs 0 and 1", s.placement, {0, 1})
        _, before = s.scrape()
        rp.parity(rec, "serve q8-replicas sequential", s.run(items, 1, "sequential"), q8_ref, len(items))
        for k in range(ctx.args.rounds):
            rp.parity(rec, f"serve q8-replicas {ctx.args.clients} clients round {k}",
                      s.run(items, ctx.args.clients, f"concurrent-{k}"), q8_ref, len(items))
        _, after = s.scrape(s.dir / "metrics.txt")
        routed = rp.delta(rp.replica_stats(after, name, 2), rp.replica_stats(before, name, 2))
        counts = [routed[r]["routed_requests_total"] for r in ("0", "1")]
        rec.expect("serve q8-replicas: both replicas routed requests", all(c and c > 0 for c in counts),
                   f"routed {counts}")
        ctx.results["q8_replicas_routed"] = counts
        s.shutdown_check()
    finally:
        s.close()


# ---------------------------------------------------------------- quality: 44 + 34 against FP32


def endpoint_answer(case, key):
    """(answer, error) of a System One case: `upstream_answer` / `upstream_error` of an upstream reference, or the
    `answer_upstream` object of a runtime export ({"error": ...} when refused); (None, None) when absent."""
    if key == "upstream_answer":
        return case.get("upstream_answer"), case.get("upstream_error")
    v = case.get(key)
    if not isinstance(v, dict):
        return None, None
    return (None, v["error"]) if v.get("error") is not None else (v, None)


def endpoint_decisions(a_dir, b_dir, a_key, b_key):
    """The decision of every question of every System One case of the upstream-compatible endpoint, b against a:
    argmax of `probabilities` with b's values taken by a's keys (first largest, basal_core::decision::argmax), so it
    does not depend on token-identical items. Returns (differences, problems, questions compared); a problem is a
    case, answer, question or distribution missing in one, duplicated or not comparable (never skipped silently)."""
    diffs, problems, n = [], [], 0

    def cases(path, label):
        out = {}
        for c in read_jsonl(path / "systemone.jsonl"):
            if not isinstance(c.get("id"), str) or c["id"] in out:
                problems.append(f"{label}: null or duplicate case id {c.get('id')!r}")
            out[c.get("id")] = c
        return out

    ca, cb = cases(a_dir, "a"), cases(b_dir, "b")
    problems += [f"case {i} only in {side}" for side, x, y in (("a", ca, cb), ("b", cb, ca)) for i in x if i not in y]
    for cid in (i for i in ca if i in cb):
        (ua, ea), (ub, eb) = endpoint_answer(ca[cid], a_key), endpoint_answer(cb[cid], b_key)
        if (ua is None and ea is None) or (ub is None and eb is None):
            problems.append(f"{cid}: no {a_key if ua is None and ea is None else b_key} answer or error")
            continue
        if (ea is None) != (eb is None):
            diffs.append(f"endpoint {cid}: refused in {'a' if ea is not None else 'b'} only")
            continue
        if ea is not None:
            continue  # refused in both: no decision (the error texts are judged by `basal compare`)
        qa, qb = ua.get("answers"), ub.get("answers")
        if not isinstance(qa, dict) or not isinstance(qb, dict) or not qa or set(qa) != set(qb):
            problems.append(f"{cid}: answered questions {sorted(qa or [])} vs {sorted(qb or [])}")
            continue
        for name, x in qa.items():
            pa, pb = (x or {}).get("probabilities"), (qb[name] or {}).get("probabilities")
            if not isinstance(pa, dict) or not isinstance(pb, dict) or not pa or set(pa) != set(pb) or \
                    not all(finite(pa[k]) and finite(pb[k]) for k in pa):
                problems.append(f"{cid}/{name}: probabilities missing, with other keys or not finite")
                continue
            keys = list(pa)
            da, db = argmax([pa[k] for k in keys]), argmax([pb[k] for k in keys])
            n += 1
            if da != db:
                diffs.append(f"endpoint {cid}/{name} ({x.get('type')}): {keys[da]}->{keys[db]}")
    return diffs, problems, n


def decision_differences(cmp, a_dir, b_dir, upstream, b_reference=False):
    """Decision-level differences of b against a: bench flips (calibrated and order average), System One item
    argmax(p_cal) flips on the items with identical tokens and, with an upstream reference `a` (`upstream`), its
    /v1/basal answers against Server.decide (`upstream_endpoint`: errors in the same places, every question in both,
    every `*_same` field true: choice, multi selection, act action/refusal/calibration, evidence spans...), as the
    tracked compatibility reports judge them; between two runtime exports the System One answers (`systemone.cases`)
    instead. Token differences are separate (token_differences): the upstream export format differs on 8 known
    items. The decision of every endpoint question (also of those 8 items, whose noul and score answers have no
    `*_same` field) comes from endpoint_decisions (`b_reference`: b is an upstream reference too, its
    `upstream_answer`); its problems are returned separately: (differences, problems, endpoint questions compared)."""
    out = []
    bench = cmp.get("bench", {})
    out += [f"bench {f['item']}: cal {f['a']}->{f['b']}" for f in bench.get("flips_cal", [])]
    out += [f"bench {f['item']}: avg {f['a']}->{f['b']}" for f in bench.get("flips_avg", [])]
    out += [f"missing {x}" for x in cmp.get("missing", [])]
    problems = []
    a_bench = mg.line_count(a_dir / "bench.jsonl")
    if bench.get("items") != a_bench:
        problems.append(f"bench: {bench.get('items')} of {a_bench} items compared (token mismatch or missing)")
    # items whose prompts and tokens are identical in both exports (the 8 known upstream-format items are not)
    same_tokens = {(c["id"], it.get("item")) for c in cmp.get("systemone", {}).get("cases", [])
                   for it in c.get("items", []) if it.get("tokens_identical") is True}
    b_cases = {c["id"]: c for c in read_jsonl(b_dir / "systemone.jsonl")}
    for a in read_jsonl(a_dir / "systemone.jsonl"):
        b = b_cases.get(a["id"])
        if b is None:
            continue
        b_items = {i.get("name"): i for i in b.get("items") or []}
        for ai in a.get("items") or []:
            bi = b_items.get(ai.get("name"))
            if (a["id"], ai.get("name")) in same_tokens and bi and ai.get("p_cal") and bi.get("p_cal") and \
                    argmax(ai["p_cal"]) != argmax(bi["p_cal"]):
                out.append(f"systemone {a['id']}/{ai['name']}: {argmax(ai['p_cal'])}->{argmax(bi['p_cal'])}")
    a_key = "upstream_answer" if upstream else "answer_upstream"
    b_key = "upstream_answer" if b_reference else "answer_upstream"
    ends, end_problems, compared = endpoint_decisions(a_dir, b_dir, a_key, b_key)
    out += ends
    problems += end_problems
    if upstream:
        for case in cmp.get("upstream_endpoint", []):
            if not case.get("errors_agree"):
                out.append(f"upstream {case['id']}: errors differ")
            for q in case.get("questions", []):
                if q.get("only_in_one"):
                    out.append(f"upstream {case['id']}/{q['question']}: only in one")
                out += [f"upstream {case['id']}/{q['question']}: {k} false"
                        for k, v in (q.get("diff") or {}).items() if k.endswith("_same") and v is False]
    else:
        for case in cmp.get("systemone", {}).get("cases", []):
            if (case.get("error_a") is None) != (case.get("error_b") is None):
                out.append(f"systemone {case['id']}: errors differ")
            for it in case.get("items", []):
                out += [f"systemone {case['id']}/{it['item']}: {k} false"
                        for k, v in (it.get("answer") or {}).items() if k.endswith("_same") and v is False]
    return out, problems, compared


def token_differences(cmp):
    return sorted(x["item"] for x in cmp.get("tokens", {}).get("mismatches", []))


def numeric_summary(cmp):
    b = cmp.get("bench", {})
    keys = ("letter_logit_diff", "order_prob_diff", "avg_prob_diff", "cal_prob_diff", "tv_avg", "tv_cal")
    return {"bench": {k: b.get(k) for k in (*keys, "items", "argmax_agreement_avg", "argmax_agreement_cal")},
            "systemone_numeric": cmp.get("systemone", {}).get("numeric_identical_tokens"),
            "quality": cmp.get("quality")}


def quality_small(rec, ctx, short):
    """44 + 34 against the FP32 reference (criterion unchanged: decisions as in FP32), f16 of the session as the
    control, q8_0 against f16, and the upstream MLX 8-bit reference as context (no threshold)."""
    ref = reference(ctx, short)
    f16 = export(rec, ctx, f"{short}-tree-none-dev0", short, "candidate", ref, "tree")
    q8 = export(rec, ctx, f"{short}-tree-q8-dev0", short, "candidate", ref, "tree", quant="q8_0")
    res = ctx.results.setdefault("quality_small", {}).setdefault(short, {})
    tokens = {}
    for label, a, b in (("fp32-vs-f16", ref, f16), ("fp32-vs-q8", ref, q8), ("f16-vs-q8", f16, q8)):
        if a is None or b is None:
            rec.check(f"quality {short} {label}", "FAIL", "export missing")
            continue
        cmp = compare(rec, ctx, f"{short} {label}", a, b, f"{short}-small-{label}")
        if cmp is None:
            continue
        diffs, problems, compared = decision_differences(cmp, a, b, upstream=label.startswith("fp32"))
        tokens[label] = token_differences(cmp)
        res[label] = {"decision_differences": diffs, "comparison_problems": problems,
                      "endpoint_questions_compared": compared, "token_differences": tokens[label],
                      **numeric_summary(cmp)}
        detail = (f"{compared} endpoint questions, {len(diffs)} decision-level differences"
                  f"{': ' + '; '.join(diffs[:8]) if diffs else ''}")
        rec.expect(f"quality {short} 44+34 {label}: every case, question and distribution compared", not problems,
                   f"{compared} endpoint questions; {'; '.join(problems[:8])}" if problems else
                   f"{compared} endpoint questions")
        if label == "fp32-vs-q8":
            rec.expect(f"quality {short} 44+34: q8_0 decisions equal to FP32 upstream (criterion; else M2)",
                       not diffs and not problems, detail)
        elif label == "fp32-vs-f16":
            rec.check(f"quality {short} 44+34: f16 control decisions equal to FP32 upstream",
                      "PASS" if not diffs else "SIGNAL", detail)
        else:
            rec.check(f"quality {short} 44+34: q8_0 decisions equal to f16 of the session (diagnostic)",
                      "PASS" if not diffs else "SIGNAL", detail)
            rec.expect(f"quality {short} 44+34: f16 and q8_0 prompts and tokens identical", not tokens[label],
                       f"{tokens[label][:5]}")
    if "fp32-vs-f16" in tokens and "fp32-vs-q8" in tokens:
        rec.expect(f"quality {short} 44+34: token differences to the FP32 export the same for f16 and q8_0 "
                   f"(known export-format items)", tokens["fp32-vs-f16"] == tokens["fp32-vs-q8"],
                   f"{len(tokens['fp32-vs-f16'])} items")
    mlx = ctx.root / MLX_Q8.format(short)
    if (mlx / "bench.jsonl").is_file():
        cmp = compare(rec, ctx, f"{short} fp32-vs-mlx-q8 (context)", ref, mlx, f"{short}-small-fp32-vs-mlx-q8")
        if cmp is not None:
            diffs, problems, compared = decision_differences(cmp, ref, mlx, True, b_reference=True)
            res["fp32-vs-mlx-q8-context"] = {"decision_differences": diffs, "comparison_problems": problems,
                                             "endpoint_questions_compared": compared, **numeric_summary(cmp)}
    else:
        rec.check(f"quality {short}: upstream MLX 8-bit context", "GAP", "reference not synced")


# ---------------------------------------------------------------- quality: 900 questions


def load_900(rec, label, out_dir, inputs):
    """index -> record of an export of the 900 inputs; None (FAIL) unless 900 records with unique non-null indices
    0..899, the ids of the inputs and a finite p_cal per option of each."""
    rows = read_jsonl(out_dir / "bench.jsonl")
    idx = [r.get("index") for r in rows]
    by = {r["index"]: r for r in rows if isinstance(r.get("index"), int)}
    ok = len(rows) == 900 and None not in idx and len(by) == 900 and set(by) == set(range(900)) and \
        all(by[r["index"]].get("id") == r["id"] for r in inputs)
    bad_p = [r["id"] for r in inputs if ok and not (
        isinstance(by[r["index"]].get("p_cal"), list) and len(by[r["index"]]["p_cal"]) == len(r["options"]) and
        all(finite(x) for x in by[r["index"]]["p_cal"]))]
    rec.expect(f"{label}: 900 records, unique indices 0..899, input ids, p_cal per option", ok and not bad_p,
               f"{len(rows)} records, {len(by)} unique int indices, p_cal missing or not finite {bad_p[:5]}")
    ok = ok and not bad_p
    return by if ok else None


def analyse_900(rec, ctx, short, runs, inputs, decisions):
    """Decisions of the session's f32 control, f16 and q8_0 against the stored upstream FP32 and basal_rs_f16, per
    question (only ids, distributions, decisions and labels are written)."""
    p = {k: {by[i]["id"]: by[i]["p_cal"] for i in by} for k, by in runs.items()}
    ids = [r["id"] for r in inputs]
    fp32 = {i: decisions[i]["upstream_fp32"] for i in ids}
    stored_f16 = {i: decisions[i]["basal_rs_f16"] for i in ids}
    stored_f32 = {i: decisions[i]["basal_rs_f32"] for i in ids}
    gold = {i: decisions[i].get("gold") for i in ids}

    def flips(x, y):
        return [i for i in ids if argmax(x[i]) != argmax(y[i])]

    def described(diff_ids, x, y):
        return [{"id": i, "fp32_margin": margin(fp32[i]), "a": argmax(x[i]), "b": argmax(y[i]),
                 "tv": tv(x[i], y[i]), "dataset": decisions[i].get("dataset"), "type": decisions[i].get("type")}
                for i in diff_ids]

    def deltas(x, y):
        return {"max_abs_dp": dist([max(abs(a - b) for a, b in zip(x[i], y[i])) for i in ids]),
                "tv": dist([tv(x[i], y[i]) for i in ids])}

    def accuracy(x):
        known = [i for i in ids if isinstance(gold[i], int)]
        return {"correct": sum(argmax(x[i]) == gold[i] for i in known), "with_gold": len(known)}

    res = {"model": short, "questions": len(ids)}
    f32_diffs = flips(p["f32"], fp32)
    control = not f32_diffs
    res["f32_control"] = {"decisions_equal_to_stored_upstream_fp32": f"{900 - len(f32_diffs)}/900",
                          "differences": described(f32_diffs, fp32, p["f32"]),
                          "vs_stored_upstream_fp32": deltas(fp32, p["f32"]),
                          "vs_stored_basal_rs_f32": deltas(stored_f32, p["f32"])}
    rec.check(f"quality {short} 900: f32 control decisions equal to the stored upstream_fp32",
              "PASS" if control else "GAP",
              "900/900" if control else f"{len(f32_diffs)} differ ({', '.join(f32_diffs[:8])}): the comparison "
              "with the stored FP32 is a GAP; nothing is attributed to it (session comparisons stay valid)")
    res["f16_session"] = {"vs_stored_upstream_fp32": described(flips(fp32, p["f16"]), fp32, p["f16"]),
                          "vs_stored_basal_rs_f16": described(flips(stored_f16, p["f16"]), stored_f16, p["f16"]),
                          "deltas_vs_fp32": deltas(fp32, p["f16"])}
    f16_wrong = set(flips(fp32, p["f16"]))
    q8_wrong = set(flips(fp32, p["q8"]))
    new = sorted(q8_wrong - f16_wrong, key=ids.index)
    fixed = sorted(f16_wrong - q8_wrong, key=ids.index)
    shared = sorted(q8_wrong & f16_wrong, key=ids.index)
    res["q8"] = {
        "attributable_to_stored_fp32": control,
        "vs_fp32": {"new": described(new, fp32, p["q8"]), "fixed": described(fixed, fp32, p["q8"]),
                    "shared": [dict(d, same_as_f16=argmax(p["q8"][d["id"]]) == argmax(p["f16"][d["id"]]))
                               for d in described(shared, fp32, p["q8"])]},
        "vs_f16_session": described(flips(p["f16"], p["q8"]), p["f16"], p["q8"]),
        "deltas_vs_fp32": deltas(fp32, p["q8"]), "deltas_vs_f16_session": deltas(p["f16"], p["q8"])}
    groups = {}
    for i in ids:
        for key in (f"dataset:{decisions[i].get('dataset')}", f"type:{decisions[i].get('type')}"):
            g = groups.setdefault(key, {"n": 0, "q8_flips_vs_fp32": 0, "f16_flips_vs_fp32": 0,
                                        "q8_flips_vs_f16": 0})
            g["n"] += 1
            g["q8_flips_vs_fp32"] += i in q8_wrong
            g["f16_flips_vs_fp32"] += i in f16_wrong
            g["q8_flips_vs_f16"] += argmax(p["q8"][i]) != argmax(p["f16"][i])
    res["groups"] = groups
    res["accuracy_vs_gold"] = {"upstream_fp32_stored": accuracy(fp32), "basal_rs_f16_stored": accuracy(stored_f16),
                               "f32_session": accuracy(p["f32"]), "f16_session": accuracy(p["f16"]),
                               "q8_session": accuracy(p["q8"])}
    gpu = ctx.gpu_name
    res["known_close_ties_of_this_model_and_card"] = next(
        (v for (s, card), v in KNOWN_TIES.items() if s == short and card in gpu), [])
    res["distributions"] = [{"id": i, "dataset": decisions[i].get("dataset"), "type": decisions[i].get("type"),
                             "gold": gold[i], "upstream_fp32": fp32[i], "basal_rs_f16_stored": stored_f16[i],
                             "f32_session": p["f32"][i], "f16_session": p["f16"][i], "q8_session": p["q8"][i]}
                            for i in ids]
    mg.write_json(ctx.out / f"quality-900-{short}.json", res)
    acc = res["accuracy_vs_gold"]
    lower = acc["q8_session"]["correct"] < acc["f16_session"]["correct"]
    detail = (f"q8_0 vs FP32: new {len(new)}, fixed {len(fixed)}, shared {len(shared)}"
              f"{'' if control else ' (not attributable: f32 control GAP)'}; vs f16 session "
              f"{len(res['q8']['vs_f16_session'])}; accuracy q8_0 {acc['q8_session']['correct']} / f16 "
              f"{acc['f16_session']['correct']} of {acc['f16_session']['with_gold']}")
    # against the stored FP32 only when the f32 control passed; otherwise against f16 of the session
    concern = bool(new) if control else bool(res["q8"]["vs_f16_session"])
    rec.check(f"quality {short} 900: q8_0 decisions (diagnostic, not a zero-flip gate)",
              "SIGNAL" if (concern or lower) else "PASS",
              detail + ("; new flips or a lower accuracy need an explanation (else M2)" if (concern or lower) else ""))
    summary = {k: v for k, v in res.items() if k != "distributions"}
    ctx.results.setdefault("quality_900", {})[short] = summary


def quality_900(rec, ctx, short):
    inputs_dir = ctx.inputs900[short]
    inputs, decisions = build_900(ctx.root, short)
    runs = {}
    names = {"f32": f"{short}-900-f32", "f16": f"{short}-900-f16-tree", "q8": f"{short}-900-q8-tree"}
    # f32: the flags of the stored basal_rs_f32 run (default single batching and fused kernels, no GEMM table)
    for key, kw in (("f32", {"dtype": "f32", "mode": None, "table": False}), ("f16", {"mode": "tree"}),
                    ("q8", {"mode": "tree", "quant": "q8_0"})):
        name = names[key]
        out = export(rec, ctx, name, short, "candidate", inputs_dir, **kw)
        by = load_900(rec, f"export {name}", out, inputs) if out else None
        if by is None:
            rec.check(f"quality {short} 900: analysis", "FAIL", f"export {name} missing or incomplete")
            return
        runs[key] = by
    for label, a, b in (("f32-vs-f16", "f32", "f16"), ("f32-vs-q8", "f32", "q8"), ("f16-vs-q8", "f16", "q8")):
        cmp = compare(rec, ctx, f"{short} 900 {label}", ctx.exports[names[a]], ctx.exports[names[b]],
                      f"{short}-900-{label}")
        if cmp is None:
            continue
        rec.expect(f"basal compare {short} 900 {label}: nothing missing, 900 compared",
                   not cmp.get("missing") and cmp.get("bench", {}).get("items") == 900,
                   f"missing {len(cmp.get('missing', []))}, compared {cmp.get('bench', {}).get('items')}")
        ctx.results.setdefault("quality_900_compare", {}).setdefault(short, {})[label] = numeric_summary(cmp)
    analyse_900(rec, ctx, short, runs, inputs, decisions)


# ---------------------------------------------------------------- performance: commands and runs

BENCH_METRICS = (("lat2_ms", 1), ("lat2_p95_ms", 1), ("dec_s", -1))
PHASE_METRICS = ((("successful_latency_ms", "p50"), 1), (("successful_latency_ms", "p95"), 1),
                 (("successful_latency_ms", "p99"), 1), (("queue_ms", "p50"), 1), (("queue_ms", "p99"), 1),
                 (("successful_requests_per_second", None), -1), (("successful_questions_per_second", None), -1))


def gpu_argv(ctx, binary, command, short, quant, extra=()):
    m = ctx.model[short]
    argv = [ctx.binary[binary], "--color", "never", command, "--model", m["dir"], "--gemm-table", ctx.tables[short],
            "--device", "0", *extra]
    if quant:
        argv += ["--quantization", quant]
    return [str(a) for a in argv]


def bench_argv(ctx, binary, short, quant):
    return gpu_argv(ctx, binary, "bench", short, quant, ["--reference", reference(ctx, short), "--dtype", "f16"])


def benchmark_argv(ctx, binary, short, quant, out):
    return gpu_argv(ctx, binary, "benchmark", short, quant, ["--scenario", "all", "--concurrency", "1,16", "--out",
                                                             out])


def with_telemetry(ctx, path, fn):
    """fn() with nvidia-smi sampling every 200 ms into `path`; returns (result, telemetry) (closed afterwards)."""
    tel = iso.Telemetry(path)
    try:
        return fn(), tel
    finally:
        tel.close()


def conditions(tel, t0, t1):
    """GPU 0 in [t0, t1] (Unix time of the shared clock): clocks, power, temperature, utilization, coverage."""
    w = tel.window(t0, t1).get("0")
    if not w:
        return {"samples": 0, "missing": True}
    expected = max(1, int((t1 - t0) / iso.Telemetry.INTERVAL_S))
    keep = ("samples", "usable_samples", "max_gap_s", "util_mean", "sm_mean_mhz", "sm_min_mhz", "power_mean_w",
            "power_max_w", "temp_max_c", "mem_peak_mib", "pstates")
    out = {k: w.get(k) for k in keep}
    out["coverage"] = round(w["usable_samples"] / expected, 3)
    out["missing"] = w["usable_samples"] == 0 or w["max_gap_s"] > iso.TELEMETRY_MAX_GAP_S
    return out


def disturbance(rec, label, process, conds):
    """The only outside disturbances recorded as such: a foreign compute process or missing telemetry samples. They
    describe a run; they never remove it or a signal (GAP: weakened evidence, the values stay in the comparison)."""
    seen = (process or {}).get("foreign_compute_processes") or {}
    problems = []
    if seen.get("foreign_pids"):
        problems.append(f"foreign compute processes {sorted(seen['foreign_pids'])}")
    if seen.get("failed_samples") or not seen.get("samples"):
        problems.append(f"compute-process samples {seen.get('samples')} ok, {seen.get('failed_samples')} failed")
    missing = [i for i, c in enumerate(conds) if (c or {}).get("missing")]
    if missing:
        problems.append(f"GPU telemetry missing or gapped in runs {missing}")
    rec.check(f"perf {label}: outside disturbances (foreign compute process, missing samples)",
              "GAP" if problems else "PASS", "; ".join(problems) if problems else "none observed")


class AbWatch:
    """The `basal bench` runs of one ab.py call, from its files: run i starts when ab.py opens its log `v-k.log` (in
    the order a b b a ...) and ends when the next log appears or ab.py exits. Returns "timeout" (stopping ab.py and
    its process group) when a run is past `limit` s, recording which one (ab.py itself has no timeout)."""

    def __init__(self, d, order, limit):
        self.d, self.limit, self.seen, self.late = d, limit, {}, None
        counts = {"a": 0, "b": 0}
        self.logs = []
        for v in order:
            self.logs.append(f"{v}-{counts[v]}.log")
            counts[v] += 1

    def __call__(self):
        started = [name for name in self.logs if (self.d / name).is_file()]
        for name in started:
            self.seen.setdefault(name, time.monotonic())
        if started and time.monotonic() - self.seen[started[-1]] > self.limit:
            self.late = started[-1].removesuffix(".log")
            return "timeout"
        return None


def ab_round(rec, ctx, label, a_argv, b_argv, rounds=4):
    """tools/bench/ab.py (unchanged) with A and B: order a b b a a b b a for 4 rounds, files a-k.json / b-k.json.
    Each `basal bench` is bounded by --bench-timeout (AbWatch), the call by 2 x rounds of it and the run deadline;
    every `basal bench` log is watched for a GEMM search. The caller stores the result, then stop_after()."""
    d = ctx.out / "perf" / label
    d.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(ctx.root / AB), "--a", shlex.join(a_argv), "--b", shlex.join(b_argv), "--rounds",
            str(rounds), "--out", str(d)]
    order = "".join(["ab", "ba"][k % 2] for k in range(rounds))
    watch = AbWatch(d, order, ctx.args.bench_timeout)
    t_start = iso.now()
    r, tel = with_telemetry(ctx, ctx.out / "perf" / f"{label}.telemetry.csv",
                            lambda: run_bounded(rec, ctx, label, argv, {}, ctx.out / "perf" / f"{label}.log",
                                                2 * rounds * ctx.args.bench_timeout, watch=d, progress=watch))
    if r["stopped"] == "timeout" and watch.late:
        rec.check(f"perf {label}: basal bench {watch.late}", "FAIL",
                  f"no exit within --bench-timeout {ctx.args.bench_timeout} s (ab.py stopped)")
    ok = judge_run(rec, f"perf {label} (ab.py)", r)
    runs, prev = {"a": [], "b": []}, t_start
    for v in order:
        k = len(runs[v])
        path = d / f"{v}-{k}.json"
        if not path.is_file():
            break
        end = path.stat().st_mtime - time.time() + iso.now()
        try:
            data = json.loads(path.read_text())
        except ValueError:
            rec.check(f"perf {label}: {path.name}", "FAIL", "not a JSON report")
            break
        runs[v].append({"file": path.name, "data": data, "conditions": conditions(tel, prev, end),
                        "window": "approximate (file times)"})
        prev = end
    complete = ok and len(runs["a"]) == rounds and len(runs["b"]) == rounds
    counted = f"{len(runs['a'])} A, {len(runs['b'])} B"
    if r["stopped"] == "deadline":
        rec.check(f"perf {label}: {rounds} A and {rounds} B runs", "GAP", f"{counted} before the run deadline")
    else:
        rec.expect(f"perf {label}: {rounds} A and {rounds} B runs", complete,
                   counted + ("" if ok else " (ab.py did not complete)"))
    disturbance(rec, label, r, [x["conditions"] for v in "ab" for x in runs[v]])
    return {"label": label, "order": order, "runs": runs, "complete": complete, "rounds": rounds, "process": {
        k: r[k] for k in ("exit", "stopped", "seconds", "cpu_s", "foreign_compute_processes", "t_start_unix",
                          "t_end_unix")}, "late_run": watch.late}


# `basal bench` (bench.rs, the runner passes no --lat-n): the latency passes take min(--lat-n, n - 5) items after 5
# warm-up items; per_item holds exactly those, indices 5.., and the report's lat_n their number.
BENCH_WARMUP_ITEMS, BENCH_LAT_N = 5, 39


def bench_expected(n_items):
    """(lat_n, per_item indices) of a `basal bench` with the default --lat-n on `n_items` reference items (44: 39
    items, indices 5..43)."""
    lat_n = min(BENCH_LAT_N, max(0, n_items - BENCH_WARMUP_ITEMS))
    return lat_n, set(range(BENCH_WARMUP_ITEMS, BENCH_WARMUP_ITEMS + lat_n))


def per_question(a, b, expected):
    """Paired per question of one A/B pair from per_item: median B/A of lat2_ms, questions where B is faster, and
    whether the items are complete: in A and in B unique indices equal to `expected` (bench_expected), a finite
    positive lat2_ms each."""
    ia, ib = a.get("per_item") or [], b.get("per_item") or []
    la = {x.get("index"): x.get("lat2_ms") for x in ia}
    lb = {x.get("index"): x.get("lat2_ms") for x in ib}
    common = sorted(set(la) & set(lb), key=str)
    ratios = [lb[k] / la[k] for k in common if finite(la[k]) and finite(lb[k]) and la[k] > 0 and lb[k] > 0]
    return {"items": len(common), "expected_items": len(expected),
            "median_ratio_b_over_a": statistics.median(ratios) if ratios else None,
            "b_faster_items": sum(r < 1 for r in ratios), "same_items": set(la) == set(lb),
            "expected_indices": set(la) == expected and set(lb) == expected,
            "unique_indices": len(la) == len(ia) and len(lb) == len(ib), "finite_pairs": len(ratios)}


def pair(a, b, direction):
    """A and B of one metric: B-A, B/A and whether B is worse (None without a finite value in both)."""
    if not finite(a) or not finite(b):
        return {"a": a, "b": b, "diff": None, "ratio": None, "b_worse": None}
    return {"a": a, "b": b, "diff": b - a, "ratio": b / a if a else None,
            "b_worse": (b > a) if direction > 0 else (b < a)}


BENCH_REQUIRED = tuple(m for m, _ in BENCH_METRICS) + ("per_question_median_ratio",)
PHASE_REQUIRED = tuple(f"{ph} {m}{'.' + q if q else ''}" for ph in PHASES for (m, q), _ in PHASE_METRICS)


def bench_pairs(rec, label, ab, strict, n_items):
    """Per pair k (a-k, b-k): every bench metric and the paired per-question effect, with the reasons a pair is not
    complete evidence (`problems`): n of the report other than the `n_items` reference items, lat_n or the per_item
    indices other than bench_expected(n_items), duplicates, a lat2_ms not finite and positive. `strict`: A and B must
    agree on accuracy and items (identical numerics)."""
    lat_n, expected = bench_expected(n_items)
    out = []
    for k, (ra, rb) in enumerate(zip(ab["runs"]["a"], ab["runs"]["b"])):
        a, b = ra["data"], rb["data"]
        metrics = {m: pair(a.get(m), b.get(m), d) for m, d in BENCH_METRICS}
        pq = per_question(a, b, expected)
        ratio = pq["median_ratio_b_over_a"]
        metrics["per_question_median_ratio"] = {"ratio": ratio, "b_worse": ratio > 1 if finite(ratio) else None}
        problems = [f"{side}: n {d.get('n')!r}, lat_n {d.get('lat_n')!r}, expected n {n_items}, lat_n {lat_n}"
                    for side, d in (("A", a), ("B", b)) if d.get("n") != n_items or d.get("lat_n") != lat_n]
        if not (expected and pq["expected_indices"] and pq["unique_indices"] and pq["finite_pairs"] == len(expected)):
            problems.append(f"per question: {pq['items']} common items, expected indices {BENCH_WARMUP_ITEMS}.."
                            f"{BENCH_WARMUP_ITEMS + lat_n - 1} in A and B {pq['expected_indices']}, unique indices "
                            f"{pq['unique_indices']}, {pq['finite_pairs']} of {len(expected)} with a finite positive "
                            "lat2_ms")
        if strict:
            same = a.get("acc") == b.get("acc") and pq["same_items"]
            rec.expect(f"perf {label} pair {k}: same accuracy and items in A and B", same,
                       f"acc {a.get('acc')} / {b.get('acc')}, same items {pq['same_items']}")
            if not same:
                problems.append("accuracy or items differ")
        out.append({"pair": k, "a_file": ra["file"], "b_file": rb["file"], "metrics": metrics,
                    "per_question": pq, "load_s": (a.get("load_s"), b.get("load_s")),
                    "gpu_probe_tflops": (a.get("gpu_probe_tflops"), b.get("gpu_probe_tflops")),
                    "acc": (a.get("acc"), b.get("acc")), "conditions": (ra["conditions"], rb["conditions"]),
                    "problems": problems})
    return out


def benchmark_runs(rec, ctx, label, specs, out):
    """`basal benchmark` runs in the given order ([(variant, argv builder)]), each fresh, bounded and watched; every
    run is appended to `out` (owned by the caller, so a stop keeps the runs before it). Past --benchmark-timeout:
    FAIL and Fatal; at the run deadline: GAP and Deadline."""
    d = ctx.out / "perf" / label
    d.mkdir(parents=True, exist_ok=True)
    for k, (variant, build) in enumerate(specs):
        guard(ctx, rec, f"perf {label} run {k}")
        path = d / f"{k}-{variant}.json"
        argv = build(path)
        r, tel = with_telemetry(ctx, d / f"{k}-{variant}.telemetry.csv",
                                lambda argv=argv, k=k, variant=variant: run_bounded(
                                    rec, ctx, f"{label}-{k}-{variant}", argv, {}, d / f"{k}-{variant}.log",
                                    ctx.args.benchmark_timeout))
        ok = judge_run(rec, f"perf {label} run {k} {variant}", r)
        try:
            data = json.loads(path.read_text()) if path.is_file() and path.stat().st_size else None
        except ValueError:
            data = None
            rec.check(f"perf {label} run {k} {variant}: report", "FAIL", "not a JSON report")
        if data is not None:
            rec.expect(f"perf {label} run {k} {variant}: benchmark complete", data.get("complete") is True and
                       data.get("status") == "completed", f"status {data.get('status')}, complete {data.get('complete')}")
        elif ok:
            rec.check(f"perf {label} run {k} {variant}: report", "FAIL", "no report written")
        t0 = r.get("t_start_unix", 0) - time.time() + iso.now()
        run = {"run": k, "variant": variant, "file": path.name, "data": data,
               "conditions": conditions(tel, t0, t0 + r.get("seconds", 0)),
               "process": {x: r.get(x) for x in ("exit", "stopped", "seconds", "cpu_s", "foreign_compute_processes")}}
        out.append(run)
        disturbance(rec, f"{label} run {k} {variant}", run["process"], [run["conditions"]])
        stop_after(f"perf {label} run {k} {variant}", r)
    rec.expect(f"perf {label}: every scheduled run reported", all(r["data"] for r in out) and len(out) == len(specs),
               f"{sum(1 for r in out if r['data'])} of {len(specs)}")
    return out


def phases(data):
    return {f"{p['name']}-c{p['concurrency']}": p for p in (data or {}).get("phases", [])}


def benchmark_pairs(rec, label, runs, strict):
    """Adjacent A/B pairs in the actual order (runs 0-1, 2-3; a pair only when both runs exist): every phase metric,
    defects as FAIL, and the reasons a pair is not complete evidence (`problems`)."""
    out = []
    sources = {json.dumps(r["data"].get("source"), sort_keys=True) for r in runs if r["data"]}
    if sources:
        rec.expect(f"perf {label}: one embedded workload version in every run", len(sources) == 1,
                   f"{len(sources)} sources")
    for k in range(0, len(runs) - 1, 2):
        x, y = runs[k], runs[k + 1]
        a, b = (x, y) if x["variant"] == "A" else (y, x)
        pa, pb = phases(a["data"]), phases(b["data"])
        metrics, problems = {}, []
        for side, ps, run in (("A", pa, a), ("B", pb, b)):
            if run["data"] is None or run["data"].get("complete") is not True:
                problems.append(f"{side} run {run['run']}: no complete report")
            if set(ps) != set(PHASES):
                problems.append(f"{side} phases {sorted(ps)}, expected {list(PHASES)}")
        for ph in sorted(set(pa) | set(pb) | set(PHASES)):
            for (m, q), d in PHASE_METRICS:
                va = pa.get(ph, {}).get(m)
                vb = pb.get(ph, {}).get(m)
                if q:
                    va, vb = (va or {}).get(q), (vb or {}).get(q)
                metrics[f"{ph} {m}{'.' + q if q else ''}"] = pair(va, vb, d)
            if ph not in pa or ph not in pb:
                continue
            defects = {f: (pa[ph].get(f), pb[ph].get(f))
                       for f in ("errors", "complete", "answer_changes_vs_first_success", "planned_requests")}
            good = all(u == v for u, v in defects.values()) and defects["complete"] == (True, True) and \
                (not strict or defects["answer_changes_vs_first_success"] == (0, 0))
            rec.expect(f"perf {label} pair {k // 2} {ph}: errors, completeness and answer changes agree", good,
                       json.dumps(defects))
            if not good:
                problems.append(f"{ph}: {json.dumps(defects)}")
        out.append({"pair": k // 2, "runs": (a["run"], b["run"]), "metrics": metrics, "problems": problems,
                    "load_s": ((a["data"] or {}).get("model", {}).get("load_s"),
                               (b["data"] or {}).get("model", {}).get("load_s")),
                    "memory_pool": ((a["data"] or {}).get("memory_pool"), (b["data"] or {}).get("memory_pool")),
                    "conditions": (a["conditions"], b["conditions"]), "process": (a["process"], b["process"])})
    return out


def evidence_problems(pairs, scheduled, required):
    """Why the pairs of a round cannot carry a conclusion: pairs missing from the schedule, a required metric absent
    or without a finite value in A or B (it counts neither as B worse nor as B not worse), the pairs' own problems."""
    problems = [] if len(pairs) == scheduled else [f"{len(pairs)} of {scheduled} scheduled pairs"]
    for p in pairs:
        bad = [m for m in required if p["metrics"].get(m, {}).get("b_worse") is None]
        if bad:
            problems.append(f"pair {p['pair']}: no finite A and B for {bad[:6]}{' ...' if len(bad) > 6 else ''}")
        problems += [f"pair {p['pair']}: {x}" for x in p.get("problems", [])]
    return problems


def signals(pairs, scheduled):
    """(signals, open): metrics with B worse in every one of the `scheduled` pairs (a signal, not a proof), and
    metrics B worse in some pair and in no pair B not worse, but without a value in every scheduled pair (neither a
    signal nor refuted: incomplete evidence)."""
    names = set().union(*(p["metrics"] for p in pairs)) if pairs else set()
    worse = {m: [p["metrics"].get(m, {}).get("b_worse") for p in pairs] for m in names}
    full = sorted(m for m, w in worse.items() if len(w) == scheduled and all(x is True for x in w))
    partial = sorted(m for m, w in worse.items() if m not in full and True in w and False not in w)
    return full, partial


def spread(runs, metric):
    """min-max of a metric over the runs of each variant (resolution only)."""
    out = {}
    for v, rs in runs.items():
        xs = [r["data"].get(metric) for r in rs if finite(r["data"].get(metric))]
        out[v] = (min(xs), max(xs)) if xs else None
    return out


def profile_run(rec, ctx, label, binary, short, quant, batch):
    """One `basal profile` (bounded by --bench-timeout: past it FAIL and Fatal; at the deadline GAP and Deadline);
    its JSON line or None."""
    d = ctx.out / "perf" / "profile"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{label}-{binary}-{quant or 'f16'}-b{batch}"
    argv = gpu_argv(ctx, binary, "profile", short, quant, ["--reference", reference(ctx, short), "--dtype", "f16",
                                                           "--n", str(ctx.args.profile_n), "--batch", str(batch)])
    r = run_bounded(rec, ctx, name, argv, {}, d / f"{name}.log", ctx.args.bench_timeout, stdout=d / f"{name}.json")
    if not judge_run(rec, f"profile {name}", r):
        stop_after(f"profile {name}", r)
        return None
    try:
        text = (d / f"{name}.json").read_text()
        return json.loads(next(x for x in text.splitlines() if x.startswith("{")))
    except (OSError, StopIteration, ValueError):
        rec.check(f"profile {name}: output", "FAIL", "no JSON line")
        return None


def bench_round(rec, ctx, label, a, b, strict, short, sink):
    """One ab.py round appended to `sink` (pairs and spread included) before a stop is raised."""
    ab = ab_round(rec, ctx, label, a, b)
    entry = {"round": label, "order": ab["order"], "process": ab["process"], "late_run": ab["late_run"],
             "complete": ab["complete"], "pairs": bench_pairs(rec, label, ab, strict, ctx.bench_items[short]),
             "spread": {m: spread(ab["runs"], m) for m, _ in BENCH_METRICS}}
    entry["problems"] = ([] if ab["complete"] else ["ab.py round incomplete"]) + \
        evidence_problems(entry["pairs"], ab["rounds"], BENCH_REQUIRED)
    sink.append(entry)
    stop_after(label, ab["process"])
    return entry


def benchmark_round(rec, ctx, label, specs, strict, sink):
    """Four `basal benchmark` runs appended to `sink` (runs without their raw reports, pairs) also when stopped."""
    runs = []
    entry = {"round": label, "runs": [], "pairs": [], "problems": []}
    sink.append(entry)

    def summarise():
        entry["runs"] = [{k: v for k, v in r.items() if k != "data"} for r in runs]
        entry["pairs"] = benchmark_pairs(rec, label, runs, strict)
        entry["problems"] = ([] if len(runs) == len(specs) and all(r["data"] for r in runs) else
                             [f"{sum(1 for r in runs if r['data'])} of {len(specs)} runs reported"]) + \
            evidence_problems(entry["pairs"], len(specs) // 2, PHASE_REQUIRED)

    try:
        benchmark_runs(rec, ctx, label, specs, runs)
    except (Deadline, Fatal):
        summarise()
        raise
    summarise()
    return entry


def campaign_perf_none(rec, ctx):
    short = ctx.args.perf_model
    res = ctx.results.setdefault("perf_none", {"A": "base (phase 02)", "B": "candidate without --quantization"})
    a, b = bench_argv(ctx, "base", short, None), bench_argv(ctx, "candidate", short, None)
    rounds = res.setdefault("rounds", {"bench": [], "benchmark": []})
    res["conclusion"] = None

    def run_round(tag, do_bench, do_benchmark):
        if do_bench:
            guard(ctx, rec, f"perf-none {tag} bench")
            bench_round(rec, ctx, f"none-bench-{tag}", a, b, True, short, rounds["bench"])
        if do_benchmark:
            specs = [(v, lambda out, v=v: benchmark_argv(ctx, "base" if v == "A" else "candidate", short, None, out))
                     for v in "ABBA"]
            benchmark_round(rec, ctx, f"none-benchmark-{tag}", specs, True, rounds["benchmark"])

    run_round("primary", True, True)
    scheduled = {"bench": 4, "benchmark": 2}
    sig, open_ = {}, {}
    for k in ("bench", "benchmark"):
        sig[k], open_[k] = signals(rounds[k][0]["pairs"], scheduled[k])
    res["primary_signals"], res["primary_open"] = sig, open_
    confirm = {k: bool(sig[k] or open_[k]) for k in sig}
    rec.check("perf-none: primary round signals (diagnostic; the confirmation round decides)", "PASS",
              f"B worse in every pair: {sig}; B worse in every pair with a value, incomplete: {open_}; one "
              "confirmation round and profiles A/B follow" if any(confirm.values()) else
              "no metric with B worse in every pair; no confirmation round")
    if any(confirm.values()):
        res["profiles"] = {}
        run_round("confirmation", confirm["bench"], confirm["benchmark"])
        for v, binary in (("A", "base"), ("B", "candidate")):
            res["profiles"][v] = profile_run(rec, ctx, "none-confirm", binary, short, None, 1)
    problems = [f"{k} {r['round']}: {x}" for k in rounds for r in rounds[k] for x in r["problems"]]
    if any(confirm.values()):
        problems += [f"profile {v}: missing" for v in "AB" if not res["profiles"].get(v)]
    persistent, unresolved = [], []
    for k in ("bench", "benchmark"):
        if not confirm[k]:
            continue
        both = rounds[k][0]["pairs"] + rounds[k][1]["pairs"]
        for m in sig[k] + open_[k]:
            w = [p["metrics"].get(m, {}).get("b_worse") for p in both]
            if False in w:
                continue  # B not worse in a pair: not repeated
            if len(w) == 2 * scheduled[k] and all(x is True for x in w):
                persistent.append(f"{k}: {m}")
            else:
                unresolved.append(f"{k}: {m}")
    res.update(persistent=persistent, unresolved=unresolved, evidence_problems=problems)
    if persistent:
        rec.check("perf-none: persistent signal (B worse in every pair of both rounds)", "SIGNAL",
                  f"{persistent}: analyse pair sizes, per-question consistency, conditions, profiles and the changed "
                  "none path; no cause in the changed code -> user decision M2 (conclusion c); a fix -> revalidation")
    if problems or unresolved:
        res["conclusion"] = None
        rec.check("perf-none: no conclusion (evidence incomplete)", "GAP",
                  f"{len(problems)} problems: {'; '.join(problems[:6])}; not resolved: {unresolved}; every value is "
                  "in summary.json")
    elif persistent:
        res["conclusion"] = "persistent signal: (b) or (c) after the parent's analysis"
    else:
        res["conclusion"] = "a"
        rec.check("perf-none: conclusion (a)", "PASS", "no repeatable slowdown shown at the measured resolution "
                  "(not a claim of no regression); every pair and value is in summary.json")


def campaign_perf_q8(rec, ctx):
    short = ctx.args.perf_model
    res = ctx.results.setdefault("perf_q8", {"A": "candidate f16", "B": "candidate q8_0",
                                             "rule": "memory/time trade-off, descriptive, no speed gate"})
    res.setdefault("bench", [])
    res.setdefault("benchmark", [])
    guard(ctx, rec, "perf-q8 bench")
    bench_round(rec, ctx, "q8-bench", bench_argv(ctx, "candidate", short, None),
                bench_argv(ctx, "candidate", short, "q8_0"), False, short, res["bench"])
    specs = [(v, lambda out, v=v: benchmark_argv(ctx, "candidate", short, None if v == "A" else "q8_0", out))
             for v in "ABBA"]
    benchmark_round(rec, ctx, "q8-benchmark", specs, False, res["benchmark"])
    problems = [f"{r['round']}: {x}" for k in ("bench", "benchmark") for r in res[k] for x in r["problems"]]
    rec.check("perf-q8: every scheduled pair with all metrics (descriptive)", "GAP" if problems else "PASS",
              "; ".join(problems[:6]) if problems else "complete")


def projection_values(m):
    """Values of the four projection matrices of all layers (one forward dequantizes each once), and of qkv + gate|up."""
    c = json.loads((m["dir"] / "config.json").read_text())
    h, nh = c["hidden_size"], c["num_attention_heads"]
    nkv, i = c.get("num_key_value_heads", nh), c["intermediate_size"]
    hd = c.get("head_dim") or h // nh
    qkv, o, gu, down = (nh + 2 * nkv) * hd * h, h * nh * hd, 2 * i * h, h * i
    layers = c["num_hidden_layers"]
    return layers * (qkv + o + gu + down), layers * (qkv + gu)


def campaign_profile(rec, ctx):
    short = ctx.args.perf_model
    total, pure = projection_values(ctx.model[short])
    res = ctx.results.setdefault("profile", {"model": short, "projection_values_per_forward": total})
    for batch in (1, ctx.args.profile_batch):
        guard(ctx, rec, f"profile batch {batch}")
        row = res[f"batch_{batch}"] = {}
        row["f16"] = f16 = profile_run(rec, ctx, "profile", "candidate", short, None, batch)
        row["q8"] = q8 = profile_run(rec, ctx, "profile", "candidate", short, "q8_0", batch)
        if f16 and q8 and q8.get("q8_dequant_values_per_item") and q8.get("q8_dequant_ms_per_item"):
            values, ms = q8["q8_dequant_values_per_item"], q8["q8_dequant_ms_per_item"]
            forwards = values / total
            gemm_ms = sum(f16["sections_ms_per_item"].get(s, 0.0) for s in ("qkv_proj", "gate_up_proj"))
            row["derived"] = {
                "forwards_per_item": forwards,
                "dequant_effective_GBps": values * 3.0625 / (ms / 1e3) / 1e9 if ms else None,
                "f16_gemm_weight_read_effective_GBps": forwards * pure * 2 / (gemm_ms / 1e3) / 1e9 if gemm_ms else None,
                "section_delta_ms_q8_minus_f16": {s: q8["sections_ms_per_item"].get(s, 0.0) -
                                                  f16["sections_ms_per_item"].get(s, 0.0)
                                                  for s in f16["sections_ms_per_item"]},
                "note": "dequantization reads 1.0625 B and writes 2 B per value; the f16 GEMM bandwidth counts only "
                        "the weight bytes of the pure GEMM sections qkv_proj and gate_up_proj; forwards per item from "
                        "the dequantized values of the q8_0 run with the same items and batching"}
        else:
            rec.check(f"profile batch {batch}: derived bandwidth", "GAP", "a profile run or its q8_0 counters missing")
    rec.check("profile: nsys timeline", "GAP", "not run in this campaign (optional per the plan)")


def campaign_memory_mini(rec, ctx):
    res = ctx.results.setdefault("memory_mini", {})
    specs = [(v, lambda out, q=q: benchmark_argv(ctx, "candidate", "mini", q, out)) for v, q in (("f16", None),
                                                                                               ("q8_0", "q8_0"))]
    runs = []
    try:
        benchmark_runs(rec, ctx, "memory-mini", specs, runs)
    except (Deadline, Fatal):  # the runs before the stop stay in the summary
        memory_rows(rec, res, runs)
        raise
    memory_rows(rec, res, runs)


def memory_rows(rec, res, runs):
    for r in runs:
        d = r["data"] or {}
        pool = d.get("memory_pool")
        if not pool or not pool.get("available"):
            rec.check(f"memory-mini {r['variant']}: pool counters", "GAP", f"{(pool or {}).get('gap', 'no report')}")
        res[r["variant"]] = {"memory_pool": pool, "estimated_weight_bytes": d.get("model", {}).get(
            "estimated_weight_bytes"), "load_s": d.get("model", {}).get("load_s"),
            "sampled_resources": d.get("resources"), "phases": {
                k: {x: p.get(x) for x in ("complete", "errors", "successful_requests_per_second")}
                for k, p in phases(d).items()}, "conditions": r["conditions"], "process": r["process"]}


# ---------------------------------------------------------------- revalidation after a code fix


EXPORT_FILES = ("bench.jsonl", "systemone.jsonl", "manifest.json")


def previous_session(prev_dir, out, candidate_sha, models, tables, shorts, root):
    """The pre-fix session for `revalidate`, checked before this run creates --out: a distinct existing raw directory
    (not --out, neither inside the other) whose run.json and summary.json carry one fingerprint with the phase-02
    base, a known candidate SHA-256 other than this run's, the model revisions, GEMM tables and inputs of this run, and
    per model a completed q8_0 tree export on GPU 0 of that session's candidate (command, checks, manifest). Returns
    its provenance; raises ValueError with the reason. Nothing in it is written."""
    prev = prev_dir.resolve()
    cur = out.resolve()
    if not prev.is_dir():
        raise ValueError("--previous-run is not an existing directory")
    if prev == cur or prev in cur.parents or cur in prev.parents:
        raise ValueError("--previous-run must be another session's directory, neither --out nor inside it or around it")
    try:
        run = json.loads((prev / "run.json").read_text())
        summary = json.loads((prev / "summary.json").read_text())
        commands = read_jsonl(prev / "commands.jsonl")
    except (OSError, ValueError) as e:
        raise ValueError(f"--previous-run: run.json, summary.json and commands.jsonl needed ({type(e).__name__})")
    fp = run.get("fingerprint")
    if not isinstance(fp, dict) or summary.get("fingerprint") != fp:
        raise ValueError("--previous-run: no fingerprint, or run.json and summary.json disagree")
    bins = fp.get("binaries") or {}
    base_sha, cand_sha = (bins.get("base") or {}).get("sha256"), (bins.get("candidate") or {}).get("sha256")
    if base_sha != PHASE02_BASE_SHA256:
        raise ValueError(f"--previous-run: base SHA-256 {base_sha}, not the phase-02 binary")
    if not isinstance(cand_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", cand_sha):
        raise ValueError("--previous-run: candidate SHA-256 unknown")
    if cand_sha in (candidate_sha, base_sha):
        raise ValueError("--previous-run: its candidate is this run's candidate or the base (no pre-fix candidate)")
    inputs = fp.get("inputs") or {}
    if (inputs.get("sequential") or {}).get("sha256") != SEQUENTIAL_SHA256:
        raise ValueError("--previous-run: other 900-question data")
    provenance = {"candidate_sha256": cand_sha, "base_sha256": base_sha, "sources": fp.get("sources"),
                  "git": fp.get("git"), "source_commit_note": fp.get("source_commit_note"),
                  "run_json_sha256": mg.sha256(prev / "run.json"), "summary_sha256": mg.sha256(prev / "summary.json"),
                  "previous_complete": summary.get("complete"), "previous_exit_status": summary.get("exit_status"),
                  "models": {}, "exports": {}}
    checks = summary.get("checks") or []
    for short in shorts:
        pm = (fp.get("models") or {}).get(short) or {}
        if pm.get("revision") != models[short]["dir"].name or pm.get("name") != models[short]["name"]:
            raise ValueError(f"--previous-run: model {short} {pm.get('name')} {pm.get('revision')}, this run "
                             f"{models[short]['name']} {models[short]['dir'].name}")
        if ((fp.get("gemm_tables") or {}).get(short) or {}).get("sha256") != mg.sha256(tables[short]):
            raise ValueError(f"--previous-run: another GEMM table for {short}")
        if (inputs.get("references_fp32") or {}).get(short) != tree_sha(root / REFERENCE.format(short)):
            raise ValueError(f"--previous-run: another FP32 reference (inputs) for {short}")
        if (inputs.get("decisions") or {}).get(short) != mg.sha256(root / DECISIONS.format(short)):
            raise ValueError(f"--previous-run: another decisions file for {short}")
        name = f"{short}-tree-q8-dev0"
        d = prev / "exports" / name
        if not all((d / f).is_file() for f in EXPORT_FILES):
            raise ValueError(f"--previous-run: export {name} missing")
        manifest = json.loads((d / "manifest.json").read_text())
        run_m = manifest.get("run") or {}
        if (run_m.get("quantization"), run_m.get("dtype"), run_m.get("prefix_cache"),
                manifest.get("bench_batching")) != ("q8_0", "f16", True, "Tree"):
            raise ValueError(f"--previous-run: export {name} is not a q8_0 f16 tree export with the prefix cache")
        runs_of = [c for c in commands if c.get("step") == f"export-{name}"]
        argv = runs_of[-1].get("argv", []) if runs_of else []
        flags = list(zip(argv, argv[1:]))
        if len(runs_of) != 1 or runs_of[0].get("exit") != 0 or runs_of[0].get("stopped") or runs_of[0].get("env") or \
                ("--quantization", "q8_0") not in flags or ("--batching", "tree") not in flags or \
                "--no-prefix-cache" in argv or "--device" in argv:
            raise ValueError(f"--previous-run: no single completed q8_0 tree command on GPU 0 for export {name}")
        mine = [c for c in checks if c.get("check", "").startswith(f"export {name}:")]
        wanted = (f"export {name}: completed", f"export {name}: every input exported",
                  f"export {name}: manifest quantization q8_0")
        if any(c.get("status") == "FAIL" for c in mine) or \
                not all(any(c.get("check") == w and c.get("status") == "PASS" for c in mine) for w in wanted):
            raise ValueError(f"--previous-run: the checks of export {name} did not pass in that session")
        if ((summary.get("exports") or {}).get(name) or {}).get("exit") != 0:
            raise ValueError(f"--previous-run: summary.json has no completed export {name}")
        provenance["models"][short] = {k: pm.get(k) for k in ("name", "revision", "safetensors_sha256")}
        provenance["exports"][short] = {f: mg.sha256(d / f) for f in EXPORT_FILES}
    return provenance


def campaign_revalidate(rec, ctx):
    """q8_0 tree exports of the final candidate byte-equal to the pre-fix candidate's (--previous-run, checked by
    previous_session); a difference reruns the q8 consistency step and the quality campaign (outputs equal does not
    mean performance equal: the performance campaigns of this session measure the final candidate). Both sessions'
    candidate SHA-256 and export SHA-256 are recorded; without the weights' SHA-256 in both sessions the comparison is
    a GAP."""
    prev, provenance = ctx.args.previous_run, ctx.previous
    res = ctx.results["revalidate"] = {"previous": provenance, "current_candidate_sha256":
                                       ctx.fp["binaries"]["candidate"]["sha256"], "models": {}, "changed": []}
    changed = res["changed"]
    for short in ctx.shorts:
        name = f"{short}-tree-q8-dev0"
        cur = export(rec, ctx, name, short, "candidate", reference(ctx, short), "tree", quant="q8_0")
        old = prev / "exports" / name
        row = res["models"][short] = {"previous_export": provenance["exports"][short]}
        if cur is None:
            rec.check(f"revalidate {short}: q8_0 tree export vs the previous run", "FAIL", "export missing")
            changed.append(short)
            continue
        row["current_export"] = {f: mg.sha256(cur / f) for f in EXPORT_FILES}
        if {f: mg.sha256(old / f) for f in EXPORT_FILES} != provenance["exports"][short]:
            rec.check(f"revalidate {short}: previous export unchanged since the preflight", "FAIL", "files changed")
            changed.append(short)
            continue
        w_prev = provenance["models"][short].get("safetensors_sha256")
        w_cur = ctx.fp["models"][short].get("safetensors_sha256")
        row["weights_sha256"] = {"previous": w_prev, "current": w_cur}
        if w_prev and w_cur and w_prev != w_cur:
            rec.check(f"revalidate {short}: same weights as the previous run", "FAIL", "SHA-256 differs")
            changed.append(short)
            continue
        same = all(row["current_export"][f] == row["previous_export"][f] for f in ("bench.jsonl", "systemone.jsonl"))
        detail = ", ".join(f"{f} {'equal' if row['current_export'][f] == row['previous_export'][f] else 'different'}"
                           for f in ("bench.jsonl", "systemone.jsonl"))
        if not same:
            rec.check(f"revalidate {short}: q8_0 tree export byte-equal to the previous run", "SIGNAL", detail)
            changed.append(short)
        elif not (w_prev and w_cur):
            rec.check(f"revalidate {short}: q8_0 tree export byte-equal to the previous run", "GAP",
                      f"{detail}, but the weights' SHA-256 is not recorded in both sessions (--no-weight-hash)")
        else:
            rec.check(f"revalidate {short}: q8_0 tree export byte-equal to the previous run", "PASS",
                      f"{detail}; previous candidate {provenance['candidate_sha256'][:12]}, same weights")
    if changed:
        rec.check("revalidate: q8_0 results changed or not comparable", "SIGNAL",
                  f"{changed}: q8 consistency, 44+34 and 900 run again")
        if "q8" not in ctx.ran_steps:
            step_q8(rec, ctx)
        if "quality" not in ctx.args.campaigns:
            campaign_quality(rec, ctx)


def campaign_quality(rec, ctx):
    for short in ctx.shorts:
        quality_small(rec, ctx, short)
    for short in ctx.shorts:
        quality_900(rec, ctx, short)


# ---------------------------------------------------------------- fingerprint


def git_state(root):
    """HEAD, SHA-256 of `git diff HEAD --binary` and the changed paths, when `root` is a git checkout."""
    try:
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        if head.returncode != 0:
            return None
        diff = subprocess.run(["git", "-C", root, "diff", "HEAD", "--binary"], capture_output=True, timeout=120)
        status = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True,
                                timeout=60)
        return {"head": head.stdout.strip(), "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
                "changed_paths": len(status.stdout.splitlines())}
    except (OSError, subprocess.SubprocessError):
        return None


def tree_sha(d):
    return {p.name: mg.sha256(p) for p in sorted(Path(d).iterdir()) if p.is_file()} if Path(d).is_dir() else None


def jit_cache(env):
    """Size of the CUDA JIT cache before the first command (a first load may compile PTX); no path recorded."""
    path = Path(env["CUDA_CACHE_PATH"]) if env.get("CUDA_CACHE_PATH") else Path.home() / ".nv" / "ComputeCache"
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else []
    return {"location": "CUDA_CACHE_PATH" if env.get("CUDA_CACHE_PATH") else "default (~/.nv/ComputeCache)",
            "exists": path.is_dir(), "files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def fingerprint(rec, args, ctx, gpus):
    versions = {}
    for label, b in ctx.binary.items():
        try:
            r = subprocess.run([b, "--version"], capture_output=True, text=True, env=ctx.env, timeout=60)
            versions[label] = iso.scrub((r.stdout + r.stderr).strip())
        except (OSError, subprocess.SubprocessError) as e:
            versions[label] = type(e).__name__
    models = {}
    for short, m in ctx.model.items():
        manifest = json.loads((ctx.root / f"reports/decision-sets/manifest-reference-basal-1.5-{short}-fp32.json")
                              .read_text())["model"]
        rec.expect(f"model {short}: revision of the stored references", m["dir"].name == manifest["revision"],
                   f"{m['dir'].name} vs {manifest['revision']}")
        st = m["dir"] / "model.safetensors"
        entry = {"name": m["name"], "revision": m["dir"].name, "safetensors_bytes": st.stat().st_size}
        if not args.no_weight_hash:
            print(f"{time.strftime('%H:%M:%S')} hashing the weights of {short}", flush=True)
            entry["safetensors_sha256"] = mg.sha256(st)
            rec.expect(f"model {short}: weights SHA-256 of the stored references",
                       entry["safetensors_sha256"] == manifest["safetensors_sha256"])
        models[short] = entry
    lib = Path(args.basal_home) / ".local/cuda/12.9.1/lib"
    cpu = None
    if Path("/proc/cpuinfo").is_file():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                    if x.startswith("model name")), None)
    root = ctx.root
    return {
        "runner_sha256": mg.sha256(HERE),
        "helpers_sha256": {"run-replicas-cloud.py": mg.sha256(REPLICAS), "run-isolation-cloud.py": mg.sha256(iso.HERE),
                           "run-multigpu-cloud.py": mg.sha256(iso.HELPER), "ab.py": mg.sha256(root / AB),
                           "build_benchmark.py": mg.sha256(root / BUILD_BENCHMARK)},
        "sources": rp.source_digest(root, ctx.out), "git": git_state(root), "source_commit_note": args.source_commit,
        "binaries": {k: {"sha256": mg.sha256(b), "version_text": versions[k]} for k, b in ctx.binary.items()},
        "binary_roles": {"base": "phase 02 (replicas; source committed as 4894c5e)", "candidate": "W8A16 candidate"},
        "models": models,
        "gemm_tables": {k: {"file": v.name, "sha256": mg.sha256(v)} for k, v in ctx.tables.items()},
        "inputs": {"sequential": {"file": SEQUENTIAL, "sha256": mg.sha256(root / SEQUENTIAL)},
                   "lock_sha256": mg.sha256(root / LOCK),
                   "decisions": {s: mg.sha256(root / DECISIONS.format(s)) for s in ctx.shorts},
                   "references_fp32": {s: tree_sha(reference(ctx, s)) for s in ctx.shorts},
                   "references_mlx_q8": {s: tree_sha(root / MLX_Q8.format(s)) for s in ctx.shorts},
                   "long_states_sha256": mg.sha256(root / LONG), "inputs_900": ctx.inputs900_summary,
                   "config_rs_sha256": mg.sha256(root / "crates/basal-cli/src/config.rs")},
        "gpus": [{"index": g[0], "name": g[1], "driver": g[3], "memory_total_mib": g[4], "power_limit_w": g[5],
                  "clocks_max_sm_mhz": g[6], "clocks_max_mem_mhz": g[7], "persistence_mode": g[8]} for g in gpus],
        "cuda_libraries": sorted(f"{p.name} {p.stat().st_size}" for p in lib.iterdir() if p.is_file())
        if lib.is_dir() else None,
        "cublaslt_tables": args.cublaslt, "python": platform.python_version(), "kernel": platform.release(),
        "cpu": cpu, "cpu_count": os.cpu_count(), "cuda_jit_cache_before_run": jit_cache(ctx.env)}


# ---------------------------------------------------------------- plan (no GPU)


def tracked(root, path):
    try:
        return json.loads((root / path).read_text())
    except (OSError, ValueError):
        return None


def plan_rows(args, root, campaigns, steps):
    """(campaign, what, count, rough seconds each or None, cap seconds each) of a session."""
    shorts = args.models.split()
    items = {s: mg.line_count(root / REFERENCE.format(s) / "bench.jsonl") for s in shorts}
    cases = {s: mg.line_count(root / REFERENCE.format(s) / "systemone.jsonl") for s in shorts}
    longs = mg.line_count(root / LONG)

    def export_s(short, kind, n=None):
        d = tracked(root, TRACKED_EXPORTS.format(short, kind))
        if not d:
            return None
        per = d["bench_s"] / items.get(short, 44)
        return d["run"]["load_s"] + (d["total_s"] if n is None else per * n)

    bench = tracked(root, TRACKED_BENCH.format(args.perf_model))
    bench_s = None
    if bench:  # load, the two latency passes and a few throughput passes of the tracked run (a lower bound)
        bench_s = bench["load_s"] + bench["lat_n"] * (bench["lat1_ms"] + bench["lat2_ms"]) / 1e3 + \
            5 * bench["all_items_s"]
    cap_e, cap_b, cap_m = args.export_timeout, args.bench_timeout, args.benchmark_timeout
    load = "900 sequential + 2 x 900 mixed requests and warm-up"
    rows = []
    if "correctness" in campaigns:
        for s in shorts:
            if "none" in steps:
                rows.append(("correctness", f"{s} none exports (base 2, candidate 5; 44+34)", 7,
                             export_s(s, "f16-tree"), cap_e))
                rows.append(("correctness", f"{s} basal compare of the none exports (no GPU)", 5, 5.0, 900))
            if "q8" in steps:
                rows.append(("correctness", f"{s} q8_0 exports (f16 tracked time as a lower bound)", 5,
                             export_s(s, "f16-tree"), cap_e))
                rows.append(("correctness", f"{s} basal compare of the q8_0 exports (no GPU)", 4, 5.0, 900))
        if "invalid" in steps:
            rows.append(("correctness", "invalid configurations (11) and doctor (2), no load", 13, 2.0, 180))
        if "serve" in steps:
            http = items["mini"] + cases["mini"] + longs + 5
            rows.append(("correctness", f"servers base, candidate, q8_0, q8_0 replicas ({http} requests each; "
                         f"replicas + {args.rounds} x {http} with {args.clients} clients)", 4, None,
                         args.ready_timeout + (1 + args.rounds) * http * 2))
            rows.append(("correctness", "mini q8_0 export of the served System One request bodies (f16 tracked time "
                         "as a lower bound)", 1, export_s("mini", "f16-tree"), cap_e))
            rows.append(("correctness", "conditional worst case: basal decide of each served request that export "
                         "refuses as Unsupported (a model load each)", cases["mini"], None, cap_e))
    if "quality" in campaigns:
        for s in shorts:
            if "correctness" not in campaigns:
                rows.append(("quality", f"{s} f16 and q8_0 exports 44+34", 2, export_s(s, "f16-tree"), cap_e))
            rows.append(("quality", f"{s} 900 f32 control (tracked f32 time per item x 900)", 1,
                         export_s(s, "f32", 900), cap_e))
            rows.append(("quality", f"{s} 900 f16 and q8_0 tree (f16 tracked time; q8_0 lower bound)", 2,
                         export_s(s, "f16-tree", 900), cap_e))
            rows.append(("quality", f"{s} basal compare (7, no GPU)", 7, 5.0, 900))
    if "perf-none" in campaigns:
        rows.append(("perf-none", "ab.py --rounds 4 basal bench (tracked bench as a lower bound)", 8, bench_s, cap_b))
        rows.append(("perf-none", f"basal benchmark A B B A ({load})", 4, None, cap_m))
        rows.append(("perf-none", "conditional worst case: confirmation bench round", 8, bench_s, cap_b))
        rows.append(("perf-none", f"conditional worst case: confirmation benchmark round ({load})", 4, None, cap_m))
        rows.append(("perf-none", "conditional worst case: basal profile A and B", 2, None, cap_b))
    if "perf-q8" in campaigns:
        rows.append(("perf-q8", "ab.py --rounds 4 basal bench f16 vs q8_0", 8, bench_s, cap_b))
        rows.append(("perf-q8", f"basal benchmark A B B A f16 vs q8_0 ({load})", 4, None, cap_m))
    if "profile" in campaigns:
        rows.append(("profile", f"basal profile f16 and q8_0 at batch 1 and {args.profile_batch}, --n "
                     f"{args.profile_n}", 4, None, cap_b))
    if "memory-mini" in campaigns:
        rows.append(("memory-mini", f"basal benchmark mini f16 and q8_0 ({load})", 2, None, cap_m))
    if "revalidate" in campaigns:
        for s in shorts:
            rows.append(("revalidate", f"{s} q8_0 tree export vs the previous run", 1, export_s(s, "f16-tree"),
                         cap_e))
            rows.append(("revalidate", f"{s} conditional worst case: q8 consistency exports", 4,
                         export_s(s, "f16-tree"), cap_e))
            rows.append(("revalidate", f"{s} conditional worst case: their basal compare and the quality compares "
                         "(no GPU)", 4 + 7, 5.0, 900))
            rows.append(("revalidate", f"{s} conditional worst case: f16 export 44+34 (none if the none step ran)", 1,
                         export_s(s, "f16-tree"), cap_e))
            rows.append(("revalidate", f"{s} conditional worst case: 900 f32, f16, q8_0 exports", 3,
                         export_s(s, "f16-tree", 900), cap_e))
    return rows


def show_plan(title, rows):
    print(title)
    known = cap = 0.0
    unknown = []
    for campaign, what, n, each, limit in rows:
        if each is None:
            unknown.append(f"{n} x {what}")
        else:
            known += n * each
        cap += n * limit
        rough = f"~{each:.0f} s each" if each is not None else "no tracked duration"
        print(f"  {campaign:12} {n:3} x {what}: {rough}, cap {limit} s each")
    print(f"  rough known part: ~{known / 60:.0f} min (tracked manifests; q8_0, its quantization at load and a JIT "
          f"of the QUANTIZED module are not in them)")
    print(f"  without a tracked duration: {len(unknown)} kinds of steps; worst case by the per-step caps "
          f"{cap / 3600:.1f} h (the run deadline stops new steps earlier)")


def print_plan(args, root):
    """Counts, the 900-input check and rough time estimates from tracked manifests; no GPU, binary or network."""
    print("900-question inputs (rebuilt in memory from the tracked files, nothing written):")
    for short in args.models.split():
        rows, decisions = build_900(root, short)
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        print(f"  {short}: {len(rows)} records, indices 0..{len(rows) - 1} unique, {len(decisions)} decision ids, "
              f"round trip 900/900, types {dict(Counter(r['type'] for r in rows))}, bench.jsonl sha256 "
              f"{hashlib.sha256(data).hexdigest()}")
    deadline = f"deadline {args.deadline_minutes} min" if args.deadline_minutes else \
        "no --deadline-minutes yet (required for a run)"
    show_plan(f"session: --campaigns {' '.join(args.campaigns)} --steps {' '.join(args.steps)} ({deadline})",
              plan_rows(args, root, args.campaigns, args.steps))
    show_plan("revalidation session, only after a code fix (--campaigns 'correctness revalidate perf-none perf-q8 "
              "profile memory-mini' --steps none --previous-run DIR), worst case:",
              plan_rows(args, root, ("correctness", "revalidate", "perf-none", "perf-q8", "profile", "memory-mini"),
                        ("none",)))
    print("Plus machine setup, prefetch of the weights and their SHA-256. Nothing here is a performance result or an "
          "affordability decision.")


# ---------------------------------------------------------------- main

CAMPAIGNS = ("correctness", "quality", "perf-none", "perf-q8", "profile", "memory-mini", "revalidate")
STEPS = ("none", "q8", "invalid", "serve")


def positive_int(text):
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError(f"{text}: must be a positive integer")
    return v


def positive_finite(text):
    v = float(text)
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"{text}: must be a positive finite number")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base", type=Path, help="phase-02 CUDA binary")
    ap.add_argument("--candidate", type=Path, help="W8A16 CUDA binary")
    ap.add_argument("--expect-base-sha256", help=f"required; must be {PHASE02_BASE_SHA256}")
    ap.add_argument("--expect-candidate-sha256", help="required: SHA-256 the candidate must have")
    ap.add_argument("--out", type=Path, help="new output directory, outside Git (e.g. .cache/cloud/...)")
    ap.add_argument("--campaigns", type=str.split, default=list(CAMPAIGNS[:-1]), help=f"subset of: {' '.join(CAMPAIGNS)}")
    ap.add_argument("--steps", type=str.split, default=list(STEPS), help=f"correctness steps: {' '.join(STEPS)}")
    ap.add_argument("--models", default="mini 4.5B", help="models of correctness and quality (mini and 4.5B)")
    ap.add_argument("--perf-model", default="4.5B", help="model of perf-none, perf-q8 and profile")
    ap.add_argument("--previous-run", type=Path, help="revalidate: raw output directory of the pre-fix candidate")
    ap.add_argument("--source-commit", default=None, help="free text recorded with the source fingerprint")
    ap.add_argument("--prepare-900", type=Path, help="write the 900-question inputs to this new directory and exit")
    ap.add_argument("--clients", type=positive_int, default=16, help="serve q8-replicas: concurrent clients")
    ap.add_argument("--rounds", type=positive_int, default=2, help="serve q8-replicas: concurrent passes")
    ap.add_argument("--long-tokens", type=positive_int, default=3000, help="long lane threshold of the served models")
    ap.add_argument("--profile-n", type=positive_int, default=40, help="basal profile --n")
    ap.add_argument("--profile-batch", type=positive_int, default=16, help="the larger basal profile --batch")
    ap.add_argument("--port", type=positive_int, default=18500)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--export-timeout", type=positive_int, default=1800, help="one basal export (s)")
    ap.add_argument("--bench-timeout", type=positive_int, default=900, help="one basal bench or profile run (s)")
    ap.add_argument("--benchmark-timeout", type=positive_int, default=2400, help="one basal benchmark run (s)")
    ap.add_argument("--ready-timeout", type=positive_int, default=600, help="server readiness (s)")
    ap.add_argument("--http-timeout", type=positive_int, default=300, help="one HTTP request (s)")
    ap.add_argument("--shutdown-timeout", type=positive_int, default=120, help="graceful server shutdown (s)")
    ap.add_argument("--deadline-minutes", type=positive_finite, default=None,
                    help="required for a run: no step, command or request starts after this (fit it to the lease, "
                         "with time left to pull the results and delete the machine)")
    ap.add_argument("--no-weight-hash", action="store_true", help="skip the SHA-256 of the weights (recorded)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan; no GPU, binary or network")
    args = ap.parse_args()

    listed = args.models.split()
    bad = [c for c in args.campaigns if c not in CAMPAIGNS] + [s for s in args.steps if s not in STEPS] + \
        [m for m in listed if m not in SHORTS] + ([] if args.perf_model in SHORTS else [args.perf_model])
    if bad:
        ap.error(f"unknown campaigns, steps or models: {bad}")
    duplicated = [x for xs in (args.campaigns, args.steps, listed) for x, n in Counter(xs).items() if n > 1]
    if duplicated:
        ap.error(f"listed twice: {duplicated}")
    if not args.campaigns:
        ap.error("no campaign")
    if "correctness" in args.campaigns and not args.steps:
        ap.error("correctness without any --steps")
    if {"correctness", "quality", "revalidate"} & set(args.campaigns) and not listed:
        ap.error("correctness, quality and revalidate need --models")
    if args.port > 65535:
        ap.error("--port must be at most 65535")
    if "serve" in args.steps and "correctness" in args.campaigns and "mini" not in listed:
        ap.error("the serve step serves mini: keep mini in --models")
    if ("revalidate" in args.campaigns) != bool(args.previous_run):
        ap.error("revalidate needs --previous-run, and --previous-run is only for revalidate")
    root = Path.cwd()
    missing = [p for p in (SEQUENTIAL, LOCK, BUILD_BENCHMARK, AB, LONG, *(DECISIONS.format(s) for s in SHORTS),
                           *(REFERENCE.format(s) + "/bench.jsonl" for s in SHORTS)) if not (root / p).is_file()]
    if missing:
        ap.error(f"run in the repository root (/work): missing {missing}")
    if args.plan_only:
        try:
            print_plan(args, root)
        except Fatal as e:
            print(f"900 inputs: {e}", file=sys.stderr)
            return 2
        return 0
    if args.prepare_900:
        if args.prepare_900.exists():
            ap.error(f"exists: {args.prepare_900}")
        if root / "reports" in args.prepare_900.resolve().parents:
            ap.error("write the inputs outside reports/ (e.g. .cache/)")
        try:
            print(json.dumps(write_900(root, args.prepare_900, args.models.split()), indent=1))
        except Fatal as e:
            print(f"900 inputs: {e}", file=sys.stderr)
            return 2
        return 0

    if not (args.out and args.base and args.candidate and args.expect_base_sha256 and args.expect_candidate_sha256 and
            args.deadline_minutes):
        ap.error("--out, --base, --candidate, --expect-base-sha256, --expect-candidate-sha256 and --deadline-minutes "
                 "are required")
    if args.expect_base_sha256.lower() != PHASE02_BASE_SHA256:
        ap.error(f"the base of this phase is the phase-02 binary {PHASE02_BASE_SHA256}")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    if root / "reports" in args.out.resolve().parents:
        ap.error("raw output belongs outside reports/ and Git (e.g. .cache/cloud/SESSION)")
    binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    for label, b in binaries.items():
        if not os.access(b, os.X_OK):
            ap.error(f"not an executable: {b}")
        got, want = mg.sha256(b), getattr(args, f"expect_{label}_sha256").lower()
        if got != want:
            ap.error(f"{label} binary SHA-256 is {got}, expected {want}")
    if binaries["base"] == binaries["candidate"] or args.expect_base_sha256 == args.expect_candidate_sha256.lower():
        ap.error("--base and --candidate must be different binaries")

    shorts = args.models.split()
    try:
        known = mg.known_revisions(root)
        models = {s: mg.resolve_model(s, known, args.hf_home) for s in sorted(set(shorts) | {args.perf_model, "mini"})}
        gpus = mg.smi(["--query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,"
                       "clocks.max.mem,persistence_mode", "--format=csv,noheader,nounits"])
        mg.smi([f"--query-gpu={iso.Telemetry.FIELDS}", "--format=csv,noheader,nounits"])
    except (StepFailed, Fatal, OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        print(f"preflight: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if len(gpus) != 2 or len({g[1] for g in gpus}) != 1:
        ap.error(f"needs exactly two identical GPUs, found {[g[1] for g in gpus]}")
    slug = re.sub(r"[^a-z0-9]+", "-", gpus[0][1].lower()).rstrip("-")
    tables = {}
    for s, m in models.items():
        t = root / "crates/basal-cli/gemm-tables" / f"{slug}--{m['slug']}--f16--cublaslt{args.cublaslt}.json"
        if not t.is_file():
            ap.error(f"missing bundled GEMM table {t} (no GEMM search is run)")
        tables[s] = t
    previous = None
    if args.previous_run:
        try:
            previous = previous_session(args.previous_run, args.out, mg.sha256(binaries["candidate"]), models, tables,
                                        shorts, root)
        except (ValueError, KeyError, TypeError, OSError) as e:
            ap.error(f"{e}")

    sys.setswitchinterval(0.0005)
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    rec = Record(out)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args, ctx.out, ctx.root, ctx.shorts = args, out, root, shorts
    ctx.env = mg.with_env(mg.environment(args), {"HF_HUB_OFFLINE": "1"})
    ctx.binary, ctx.model, ctx.tables = binaries, models, tables
    ctx.by_bus = {mg.bus(g[2]): int(g[0]) for g in gpus}
    ctx.gpu_name = gpus[0][1]
    ctx.deadline = time.monotonic() + args.deadline_minutes * 60
    ctx.deadline_hit, ctx.watch_foreign = False, True
    ctx.exports, ctx.export_runs, ctx.q8_diagnostics, ctx.results, ctx.ran_steps = {}, {}, {}, {}, []
    ctx.previous = previous
    ctx.bench_items = {s: mg.line_count(reference(ctx, s) / "bench.jsonl") for s in models}  # bench items per model
    try:
        ctx.inputs900_summary = write_900(root, out / "inputs-900", sorted(set(shorts)))
    except Fatal as e:
        print(f"preflight: 900 inputs: {e}", file=sys.stderr)
        return 2
    ctx.inputs900 = {s: out / "inputs-900" / f"inputs-900-{s}" for s in shorts}

    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w,"
                                  "clocks.max.sm_mhz,clocks.max.mem_mhz,persistence_mode\n" +
                                  "".join(",".join(g) + "\n" for g in gpus))
    fp = ctx.fp = fingerprint(rec, args, ctx, gpus)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("base", "candidate", "out", "hf_home", "basal_home", "plan_only", "previous_run",
                           "prepare_900")}
    params["env"] = {k: ctx.env[k] for k in ("BASAL_NO_UPDATE_CHECK", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE")}
    mg.write_json(out / "run.json", {"fingerprint": fp, "parameters": params,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    work = []
    if "correctness" in args.campaigns:
        work += [(f"correctness/{s}", fn) for s, fn in (("none", step_none), ("q8", step_q8),
                                                        ("invalid", step_invalid), ("serve", step_serve))
                 if s in args.steps]
    work += [(c, fn) for c, fn in (("quality", campaign_quality), ("perf-none", campaign_perf_none),
                                   ("perf-q8", campaign_perf_q8), ("profile", campaign_profile),
                                   ("memory-mini", campaign_memory_mini), ("revalidate", campaign_revalidate))
             if c in args.campaigns]
    started, stopped, complete = [], None, False
    try:
        for label, fn in work:
            print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)
            started.append(label)
            try:
                guard(ctx, rec, label)
                fn(rec, ctx)
                ctx.ran_steps.append(label.split("/")[-1])
            except Deadline:
                break
            except (StepFailed, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
                rec.check(label, "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
        complete = not ctx.deadline_hit and len(started) == len(work)
    except Fatal as e:
        stopped = f"Fatal during {started[-1] if started else 'setup'}: {e}"
    except BaseException as e:  # a signal (SystemExit from terminate), Ctrl-C or an unexpected error: re-raised
        for sig in rp.STOP_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        reason = f" (exit {e.code})" if isinstance(e, SystemExit) else f": {e}" if str(e) else ""
        stopped = f"{type(e).__name__}{reason} during {started[-1] if started else 'setup'}"
        raise
    finally:
        if stopped:
            rec.check("run", "FAIL", iso.scrub(f"stopped: {stopped}; partial records kept, nothing later run"))
        for label, _ in work:
            if label not in started:
                rec.check(label, "GAP", "not run: " + ("the run stopped" if stopped else "the deadline passed"))
        counts = {k: sum(c["status"] == k for c in rec.checks) for k in STATUSES}
        status = 1 if counts["FAIL"] or stopped else 4 if counts["SIGNAL"] else \
            3 if counts["GAP"] or not complete else 0
        mg.write_json(out / "summary.json", {
            "complete": complete, "stopped": iso.scrub(stopped), "exit_status": status, "counts": counts,
            "fingerprint": fp, "parameters": params, "exports": ctx.export_runs,
            "q8_diagnostics": scrubbed(ctx.q8_diagnostics), "results": scrubbed(ctx.results),
            "checks": iso.portable_checks(rec.checks), "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"DONE {counts} exit {status}" + (f" (stopped: {stopped})" if stopped else ""), flush=True)
    return status


def terminate(signum, _frame):
    """SystemExit(128 + signal) once; later SIGTERM, SIGHUP and SIGINT are ignored so the bounded cleanup and
    summary.json are not cut short (SIGKILL still ends the runner)."""
    for sig in rp.STOP_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    for sig in rp.STOP_SIGNALS:
        signal.signal(sig, terminate)
    try:
        sys.exit(main())
    finally:
        for p in list(mg.LIVE):
            mg.stop(p, timeout=30)
