#!/usr/bin/env python3
"""Manual evidence run for optional W4A16 weights (`quantization: q4_k`) on a rented Linux host with two identical
NVIDIA GPUs. Not a test and not a CI gate: it collects raw evidence for a report the operator writes.

  python3 tools/perf/run-w4a16-cloud.py --help
  python3 tools/perf/run-w4a16-cloud.py --plan-only --w8-raw DIR ...      # counts, sources, time forecast (no GPU)
  python3 tools/perf/run-w4a16-cloud.py --prepare-900 .cache/w4a16-inputs  # the 900-question inputs (no GPU)
  cd /work && python3 tools/perf/run-w4a16-cloud.py --base BASE --expect-base-sha256 SHA --base-label TEXT \\
      --candidate CAND --expect-candidate-sha256 SHA --candidate-label TEXT --deadline-minutes M --out out/w4a16-s1

--plan-only and --prepare-900 read only local files: no cloud, SSH, GPU, binary, nvidia-smi, network or secret.

Binaries. --base is the stable W8 binary of source 750c146 (its SHA-256 is the operator's: the W8 candidate build
only when every provenance condition of the plan is shown, otherwise a rebuild of 750c146 with the candidate's
builder); --candidate is the W4A16 build. Both --expect-*-sha256 are required and checked before anything runs;
--base-label / --candidate-label are the declared roles and source labels, recorded as declared (the runner does
not prove a source). --base-provenance / --candidate-provenance (optional JSON of the build: binary_sha256,
source_digest {files, sha256} by the W8 runner's method, embedded_assets {path: sha256 or tree digest}, recipe)
are recorded; a record whose binary_sha256 is missing, malformed or not the binary's is refused before anything
runs, and the candidate's source digest and embedded assets are compared with the synced tree before any GPU work
(different: FAIL and the run stops; not given: a GAP of that binding, never a match).
The helpers are imported unchanged and their SHA-256 checked first: run-w8a16-cloud.py (and through it
run-replicas-cloud.py, run-isolation-cloud.py, run-multigpu-cloud.py) and tools/bench/ab.py. Prerequisites as for
the W8 runner (synced to /work, weights in HF_HOME=/data/hf, CUDA libraries in BASAL_HOME=/data/basal), plus
reports/w4a16-p0-2026-10-12/facts.json (the approved P0 weight diagnostic, SHA-256 pinned below).

Sessions (separate invocations; the parent audits the raw data between them, nothing later runs by default):
  S1  --campaigns 'correctness quality profile memory-mini' (default)
  S2  --campaigns perf-none-family --s1-run DIR     S3  --campaigns perf-q8-family --s1-run DIR
  revalidation (only after an actual code fix): --campaigns revalidate --previous-run DIR
      --expect-previous-candidate-sha256 SHA, then the affected families with --s1-run of that run
A family run reads --s1-run (the raw directory of the S1 or revalidation run of the same base and candidate) before
anything runs: its fingerprint must name the same binaries and synced sources, and it must have no FAIL other than
class quality-criterion. Before any GPU work its recorded runner and helper SHA-256, embedded assets, the family
model's name, revision, config.json and weights, its GEMM table, FP32 reference, decisions and 900 inputs, and the
sequential, lock, long-state, config.rs and P0 facts files are compared with this run's: a difference is a FAIL
that stops the run, a proof missing in either run (e.g. --no-weight-hash) a GAP of that binding. Its counts, FAIL
classes, campaigns and completeness are recorded (an incomplete S1 is a GAP of the family run, not a refusal).

Campaign `correctness`:
  none     candidate exports without --quantization and with --quantization none, mini and 4.5B, tree and single on
           GPU 0, tree on GPU 1, byte-equal to the base export of the same mode on GPU 0, no `basal compare`
           difference, backend description equal except the device ordinal (the W8 runner's step, unchanged)
  q8       base q8_0 tree export on GPU 0; candidate q8_0 tree, single, tree --no-prefix-cache on GPU 0, tree on GPU 1
           and tree under CUDA_VISIBLE_DEVICES=1 --device 0, each byte-equal to the base export, backend description
           and quantization object equal except the device ordinal
  w4       candidate q4_k tree on GPU 0; single, tree --no-prefix-cache, tree on GPU 1, tree under
           CUDA_VISIBLE_DEVICES=1 --device 0 byte-equal to it, description equal except the ordinal
           (q8 and w4: the --no-prefix-cache export pins no template prefixes: prefix_kv = [] required, every other
           field, the quantization object included, equal; the plan's clarification of criteria 2-3)
  invalid  q4_k with bf16 / f32, --kernels candle, the unknown values q4_0 and Q4_K, --quantization q4_k with a
           configuration file, through serve, export, bench, benchmark, decide and profile, for a model absent from
           the cache (HF_HUB_OFFLINE=1): the validation message, no Hub, load or GEMM-search message; no GPU memory of
           the process (a process ending before the first 0.5 s sample is a GAP of that sample, not a PASS). doctor:
           q4_k with bf16 and Q4_K as model failures, q4_k with a known config.json whose projection K is not a
           multiple of 256 (a copy of mini's config.json with intermediate_size + 32 and the checkpoint linked, in
           --out) as a failure, a valid q4_k entry with its estimate, and the same q8_0 estimate text from base and
           candidate. Intel and Metal: GAP (static review only)
  serve    mini servers (YAML, explicit bundled table): base none (HTTP reference), candidate none (every answer,
           /health and /v1/models status 200 and bytes, the HELP/TYPE metric family set of a valid scrape equal in
           both directions to the base's; an invalid reference surface is a GAP, never a pass), base q8_0 (q8_0
           HTTP reference), candidate q8_0
           (equal to the base q8_0 server; its System One answers also equal to the native q8_0 reference of the
           same request bodies), candidate q4_k on GPU 0 (System One equal to the native q4_k reference), candidate
           q4_k replica_devices [0, 1] (equal to the single q4_k server sequentially and with --clients clients for
           --rounds rounds, memory and routed requests on both GPUs, SIGTERM exit 0, no GPU memory left). The native
           reference of a format is its tree export of the exact served bodies (`model` set to mini; SHA-256 and the
           number of rewritten `model` fields recorded); a case that export refuses as Unsupported (its generic
           outcome, no case named in advance) gets `basal decide` of the body with the export's flags. Natural 400/422
           bodies are compared as answers, not as transport failures.
Campaign `quality`, mini and 4.5B (GPU 0): 44 + 34 against the FP32 references with the existing W8 predicate
  (decision_differences: bench flips avg and cal, System One argmax p_cal on token-identical items, every endpoint
  decision by key name, errors in the same places, every typed `*_same` field): FP32 -> q4_k a violation is a FAIL
  of class quality-criterion; FP32 -> f16 of the session (control) a SIGNAL; f16 -> q4_k and q8_0 -> q4_k of the
  session diagnostics (SIGNAL when they differ); FP32 -> q8_0 and FP32 -> MLX 8-bit context only. An incomplete
  comparison is a FAIL of class correctness, never an empty pass. Every changed typed field is kept with the
  answers of FP32, f16, q8_0 and q4_k (evidence spans), and the continuous differences (|dp|, logits) are recorded
  without a threshold. The 900 questions: f32 control (the flags of the stored basal_rs_f32: single, fused, no
  table), candidate f16, q8_0 and q4_k tree and base f16 and q8_0 tree with the table; base and candidate f16 and
  q8_0 byte-equal (else FAIL); f32 control 900/900 equal to the stored upstream_fp32, else every comparison with
  the stored FP32 is a GAP and nothing is attributed to it; q4_k flips to FP32 (new, fixed, shared against the f16
  control), to f16 and q8_0 of the session, FP32 margins, the shift of the fixed FP32 top-two gap and its context
  distribution, |dp| and TV (mean, p50, p90, p99, max), accuracy against gold (stored FP32, f32, f16, q8_0, q4_k;
  per dataset and type). New q4_k flips to FP32 or a lower accuracy than f16 of the session are a SIGNAL (no
  zero-flip gate; an explanation does not close it: M2 after defect fixes).
q4_k loads: the quantization object of every q4_k export manifest, `basal benchmark` and `basal bench` report
  (where the backend description is written) must name a warm-up check of the smallest layer-0 projection with
  chunk_rows 16, and its model and per-projection reconstruction statistics (rows, values, sum_e2, sum_w2, max |e|,
  R and its status, groups total / zero / flattened / zero with non-zero reconstruction with first positions) must
  equal the approved P0 facts exactly (integers, f64 bits; a first position null exactly when its count is 0, else
  the same position) for the same weights (SHA-256 of this run equal to P0's; otherwise a GAP); the object must be
  identical across every observed load. Servers and `basal profile` do not write the description: their loads
  passed the in-code check, the object itself is not observed.
Campaign `profile` (GPU 0, 4.5B): candidate f16, q8_0 and q4_k at --batch 1 and --profile-batch (--n 40); the
  dequantization keys of each format (q8_dequant_* for q8_0, dequant_* for q4_k), effective bytes per second as
  arithmetic (q8_0 3.0625, q4_k 2.5625 B per value; not physical bandwidth or request latency). nsys: GAP.
Campaign `memory-mini` (GPU 0): one `basal benchmark --scenario all --concurrency 1,16` each for mini f16, q8_0
  and q4_k: pool used/reserved at load, resident, runtime peak and after shutdown separately, NVML samples (not a
  peak), 900 sequential and 846 mixed requests at c1 and c16, the same 54 mixed requests skipped (context length)
  in every run with their ids and reasons. load_s of `basal benchmark` excludes the CUDA context creation, that of
  export / bench / profile includes it; the host RSS peak is not measured.
Campaigns `perf-none-family` (S2) and `perf-q8-family` (S3), GPU 0, 4.5B: six fresh `basal benchmark --scenario
  all --concurrency 1,16` runs in the order A B C C B A (none family: A base none, B candidate none, C candidate
  q4_k; q8_0 family: A base q8_0, B candidate q8_0, C candidate q4_k) and `ab.py --rounds 4` of `basal bench
  --reference reports/reference-basal-1.5-4.5B-fp32 --dtype f16` for A/B and for B/C. Regression pairs (fixed
  before the run): benchmark A0/B1 and B4/A5, the four ab.py pairs A/B; the W8 rule: a metric with B worse in every
  pair is a signal, then exactly one confirmation round of only the affected command (ab.py --rounds 4 A/B, or four
  benchmark runs A B B A) and one `basal profile` A and B at batch 1; a signal again in every pair of both rounds
  is a persistent SIGNAL (parent analysis; unexplained: M2); conclusions (a) / persistent / GAP only with complete
  evidence exactly as W8 (every scheduled pair, every metric finite in A and B, per question indices 5..43). The
  trade-off pairs B1/C2 and C3/B4 and the ab.py B/C pairs (B runs belong to two pairs, declared): every metric,
  B-A and B/A, load_s, pools, NVML, without a signal rule or speed gate. Conditions describe, never filter.
Campaign `revalidate` (after a code fix only): --previous-run (a W4 raw directory of the same base, another
  candidate equal to --expect-previous-candidate-sha256, other synced sources, the same models, tables and inputs,
  per model a single completed q4_k tree export on GPU 0 whose checks completed, every input exported and manifest
  quantization q4_k passed; its other checks, e.g. the q4_k load description or P0 equality the fix may address,
  are kept as context) is checked before anything runs. It runs the none and q8 steps, the serve step once (none /
  q8_0 servers, and the q4_k native HTTP and replicas of the final candidate whatever its exports show: export
  bytes do not show serving, routing or shutdown) and the 900 base/candidate f16 and q8_0 byte checks against the
  base, and the q4_k tree exports against the previous run's; a difference (or no weights SHA-256 in both
  sessions: GAP) reruns the w4 step and the quality campaign. Profile and memory-mini of the previous candidate
  are not re-measured (a GAP of the final binary's scope; equal outputs do not show equal time or memory); the
  families affected by the fix run separately with --s1-run of this run. The forecast shows the plan's one
  conditional revalidation session unsplit (revalidate and the affected families on one machine) and R1 / R2 / R3
  as a proposed split that needs a parent plan and forecast revision before any purchase.

Statuses PASS, FAIL (class correctness, quality-criterion or timeout, counted separately in summary.json), GAP
(missing evidence, also everything not run) and SIGNAL (analysis needed). Exit status: 0 all PASS, 1 a FAIL, 4 no
FAIL but a SIGNAL, 3 only a GAP, 2 preflight or usage error; a run stopped by a signal exits 128 + signal, never 0.
Every command of a step, HTTP request, graceful shutdown and direct nvidia-smi / --version probe is bounded by its
own timeout and by the run deadline: min(--deadline-minutes, --lease-end-unix - now) at the start (the operator
computes both from the machine's creation time, its life limit and the reserve to pull the results and delete it;
the runner can only shorten them). Each ends at the absolute end min(its start + own timeout, run deadline); an
exit, response or nvidia-smi result seen only after that end is not timely and is handled like none (kept as
evidence, never a PASS). After the deadline no step, command or request starts; what was cut at it or not run is a GAP
with the reason `deadline`, and the records already answered are kept; the deadline is checked again right before a
server starts (after the idle wait) and also ends a graceful-shutdown wait and the server placement samples (a GAP,
never a timely result; the cleanup after it is bounded separately). A GPU command, an HTTP request (POST, and after
readiness /health, /v1/models and /metrics), readiness, a graceful shutdown or a direct nvidia-smi (also the
placement samples at server start and after its exit) / --version probe past its own timeout is a FAIL of class
timeout that stops the run: a suspected hung GPU, no later GPU work (a single readiness probe past its 5 s is polled
again within --ready-timeout; another transport error is not a timeout; the preflight nvidia-smi, before --out
exists, exits 2 naming the earlier end). Outside this scope, unchanged from the helpers: the git metadata of the
fingerprint keeps the W8 helper's own limits, and the background samplers (telemetry, placement and foreign-process
watches) only observe, with their own limits; the cleanup of a stopped process (SIGTERM, SIGKILL after 60 s) is
bounded separately and never timely evidence, but its final reap can wait on the kernel: the operator's external
machine-life guard stays required. A stopped run writes everything already recorded with complete false. The runner
does not rent, extend or delete machines.

Recorded under --out (must not exist; raw, contains local paths: keep it outside Git, e.g. in .cache/cloud):
checks.txt, commands.jsonl, run.json, summary.json (portable: scrubbed), gpus.csv, source-files.sha256,
inputs-900/, exports/, compare/, correctness/ (invalid, serve, serve-inputs, serve-decide-<format>), perf/,
quality-small-<model>.json (answers of the changed fields), quality-900-<model>.json (ids, distributions,
decisions, labels; no texts). correctness/invalid/known-k-model/model.safetensors is a symbolic link to mini's
weights: pull --out without following symbolic links (the weights are not part of the results).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import signal
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
W8_RUNNER = HERE.with_name("run-w8a16-cloud.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or not Path(path).is_file():
        sys.exit(f"missing {path} (sync tools/perf, tools/bench and tools/decision-sets)")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# the W8 runner (definitions only, its main does not run) and through it one LIVE registry of the helpers
w8 = load_module("basal_w8a16_runner", W8_RUNNER)
rp, iso, mg = w8.rp, w8.iso, w8.mg
StepFailed, Fatal, Deadline = iso.StepFailed, iso.Fatal, w8.Deadline

RUNNER_ID = "run-w4a16-cloud.py"
# the helpers this runner was written against (reused unchanged); another version is refused
HELPERS_SHA256 = {
    "tools/perf/run-w8a16-cloud.py": "92b47c9e341f47fe7a538cbd42a0f0e152ae7199720bda2b5d15f79f83d945ea",
    "tools/perf/run-replicas-cloud.py": "cb22ac7a498522374c4ea0f09cbbb6317a657acf70f05299f182552c332963dc",
    "tools/perf/run-isolation-cloud.py": "c460aee1b21da0e3a8418f9d7c19fc7b4cf5e1d84ebdfe43f98456fb9ca0f582",
    "tools/perf/run-multigpu-cloud.py": "1ddd4910c6a1ff79c297e42a70efc5cae975289147f911700e20e403e2a351d8",
    "tools/bench/ab.py": "17b188dcdf5a6986f4930c63994046111ba7ce80fea89bde820c97d425ac7bec",
}
P0_FACTS = "reports/w4a16-p0-2026-10-12/facts.json"
P0_FACTS_SHA256 = "44005ecc347b52b5f2295ee9b54edbebc7f014e006b2c2fd86e9dcef7c5cbf68"
P0_CHUNK_ROWS = 16
# assets embedded by include_str! and the build recipe: not among the W8 runner's source files
ASSETS = ("crates/basal-cli/benchmark-data", "crates/basal-cli/gemm-tables", "rust-toolchain.toml",
          "tools/release/linux.Dockerfile")
Q8, Q4 = "q8_0", "q4_k"
SHORTS = w8.SHORTS
STATUSES = w8.STATUSES
FAIL_CLASSES = ("correctness", "quality-criterion", "timeout")
ABSENT = "Remek/basal-w4a16-absent"  # never in a cache: validation must fail before any download
SEARCH = w8.SEARCH
PROJECTIONS = ("qkv", "o", "gate_up", "down")
# basal benchmark of mini (embedded workload, plan): requests per phase and the mixed requests over the context limit
MEMORY_MINI_REQUESTS = {"sequential-c1": 900, "mixed-c1": 846, "mixed-c16": 846}
MEMORY_MINI_SKIPPED = 54
DEQUANT_BYTES_PER_VALUE = {Q8: 3.0625, Q4: 2.5625}  # read blocks + write f16, arithmetic only
DEQUANT_KEYS = {Q8: ("q8_dequant_ms_per_item", "q8_dequant_values_per_item", "q8_dequant_scope"),
                Q4: ("dequant_ms_per_item", "dequant_values_per_item", "dequant_scope", "dequant_format")}
# perf families: role -> (binary, quantization); six benchmark runs in this order; the pairs are fixed in advance
FAMILIES = {"none": {"A": ("base", None), "B": ("candidate", None), "C": ("candidate", Q4)},
            "q8": {"A": ("base", Q8), "B": ("candidate", Q8), "C": ("candidate", Q4)}}
FAMILY_ORDER = "ABCCBA"
REGRESSION_PAIRS = (("A0/B1", 0, 1), ("B4/A5", 5, 4))  # (name, run of role A, run of role B)
TRADEOFF_PAIRS = (("B1/C2", 1, 2), ("C3/B4", 4, 3))  # (name, run of role B, run of role C)

CAMPAIGNS = ("correctness", "quality", "profile", "memory-mini", "perf-none-family", "perf-q8-family", "revalidate")
S1_CAMPAIGNS = ("correctness", "quality", "profile", "memory-mini")
FAMILY_CAMPAIGNS = {"perf-none-family": "none", "perf-q8-family": "q8"}
STEPS = ("none", "q8", "w4", "invalid", "serve")


class Record(w8.Record):
    """w8.Record (four statuses) whose FAIL carries a class: an explicit `cls`, else "timeout" for the W8 helpers'
    own-timeout messages (judge_run and export: "no exit within ..."), else "correctness"."""

    def check(self, name, status, detail="", cls=None):
        assert status in STATUSES
        if status == "FAIL":
            cls = cls or ("timeout" if "no exit within" in str(detail) else "correctness")
            assert cls in FAIL_CLASSES
        else:
            cls = None
        tag = f"{status}[{cls}]" if cls else status
        with self.lock:
            self.checks.append({"check": name, "status": status, "class": cls, "detail": detail})
            with open(self.out / "checks.txt", "a") as f:
                f.write(f"{tag} {name}{': ' + detail if detail else ''}\n")
        print(f"{time.strftime('%H:%M:%S')} {tag} {name}{': ' + detail if detail else ''}", flush=True)

    def expect(self, name, ok, detail="", cls=None):
        self.check(name, "PASS" if ok else "FAIL", detail, cls)
        return ok


def judged(rec, name, diffs, problems, detail, on_diff):
    """PASS without differences, `on_diff` ("FAIL" of class quality-criterion or "SIGNAL") with any, GAP when the
    comparison is incomplete and shows none (never an empty pass)."""
    if diffs:
        rec.check(name, on_diff, detail, cls="quality-criterion" if on_diff == "FAIL" else None)
    elif problems:
        rec.check(name, "GAP", f"not judged: the comparison is incomplete ({len(problems)} problems); {detail}")
    else:
        rec.check(name, "PASS", detail)


def read_json(path):
    return json.loads(Path(path).read_text())


def rel(root, path):
    """`path` relative to `root` when inside it, else only its name (no local path in a printed plan)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(root.resolve()))
    except ValueError:
        return p.name


def f64_bits(x):
    return struct.pack("<d", float(x)).hex() if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def reference(ctx, short):
    return ctx.root / w8.REFERENCE.format(short)


def projection_shapes(model_dir):
    """[(projection, N, K)] of a model (basal_gpu::projection_shapes: qkv, o, gate_up, down)."""
    c = json.loads((Path(model_dir) / "config.json").read_text())
    h, nh = c["hidden_size"], c["num_attention_heads"]
    nkv, i = c.get("num_key_value_heads", nh), c["intermediate_size"]
    hd = c.get("head_dim") or h // nh
    return [("qkv", (nh + 2 * nkv) * hd, h), ("o", h, nh * hd), ("gate_up", 2 * i, h), ("down", h, i)]


def expected_warmup(model_dir):
    """(projection, values) of the q4_k warm-up matrix: the smallest layer-0 projection, the first of equals."""
    best = None
    for name, n, k in projection_shapes(model_dir):
        if best is None or n * k < best[1]:
            best = (name, n * k)
    return best


# ---------------------------------------------------------------- binaries, sources and identities


def file_or_tree_sha(path):
    """SHA-256 of a file, or for a directory {files, sha256 over "sha  relative path" lines} (sorted, recursive)."""
    p = Path(path)
    if p.is_file():
        return mg.sha256(p)
    if p.is_dir():
        lines = [f"{mg.sha256(f)}  {f.relative_to(p)}" for f in sorted(p.rglob("*")) if f.is_file()]
        return {"files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}
    return None


def embedded_assets(root):
    return {a: file_or_tree_sha(root / a) for a in ASSETS}


def sources_digest(root):
    """rp.source_digest without writing source-files.sha256 (the same files and digest)."""
    lines = [f"{mg.sha256(p)}  {p.relative_to(root)}" for p in rp.source_files(root)]
    return {"files": len(lines), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest() if lines else None}


def helpers_state(root):
    """{relative path: (actual SHA-256, expected)} of the reused helpers."""
    return {p: (mg.sha256(root / p) if (root / p).is_file() else None, want) for p, want in HELPERS_SHA256.items()}


def read_provenance(path, binary_sha):
    """The optional build record of a binary: (record, its SHA-256); raises ValueError when it is not a JSON object
    or its binary_sha256 is missing, not a SHA-256 or not the binary's (a record not bound to this binary binds
    nothing)."""
    try:
        v = read_json(path)
    except (OSError, ValueError) as e:
        raise ValueError(f"{path.name}: not a readable JSON file ({type(e).__name__})")
    if not isinstance(v, dict):
        raise ValueError(f"{path.name}: not a JSON object")
    got = v.get("binary_sha256")
    if not isinstance(got, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", got):
        raise ValueError(f"{path.name}: no binary_sha256 (a SHA-256 of 64 hexadecimal digits): {got!r}")
    if got.lower() != binary_sha:
        raise ValueError(f"{path.name}: binary_sha256 {got} is not the binary's {binary_sha}")
    return v, mg.sha256(path)


def load_p0(root):
    """(facts or None, provenance): the approved P0 facts, only with the pinned SHA-256."""
    path = root / P0_FACTS
    if not path.is_file():
        return None, {"file": P0_FACTS, "sha256": None, "expected": P0_FACTS_SHA256, "usable": False}
    sha = mg.sha256(path)
    usable = sha == P0_FACTS_SHA256
    return (read_json(path) if usable else None), {"file": P0_FACTS, "sha256": sha, "expected": P0_FACTS_SHA256,
                                                   "usable": usable}


def p0_model(facts, short):
    return next((m for m in (facts or {}).get("models", []) if m.get("label") == short), None)


# ---------------------------------------------------------------- exports and the q4_k load description


def export(rec, ctx, name, short, binary, inputs, mode="tree", quant=None, **kw):
    """w8.export (bounded, cached by name, placement, candidate manifest quantization), plus for the base its
    manifest quantization and for q4_k the load-description checks (q4_load_checks), once per export."""
    first = name not in ctx.exports
    out = w8.export(rec, ctx, name, short, binary, inputs, mode, quant=quant, **kw)
    if not first or out is None:
        return out
    manifest = read_json(out / "manifest.json")
    q = manifest.get("backend", {}).get("quantization")
    if binary == "base":
        want = quant or "none"
        ok = manifest.get("run", {}).get("quantization") == want and \
            (q is None if want == "none" else isinstance(q, dict) and q.get("matrices") == 4 * w8.layers_of(
                ctx.model[short]))
        rec.expect(f"export {name}: manifest quantization {want}", ok,
                   f"run.quantization {manifest.get('run', {}).get('quantization')!r}")
        if isinstance(q, dict):
            ctx.q8_diagnostics[name] = q
    if quant == Q4:
        q4_load_checks(rec, ctx, f"export {name}", short, manifest.get("backend", {}))
    return out


def q4_load_checks(rec, ctx, label, short, backend):
    """The q4_k quantization object of one load (a backend description): its warm-up check of the expected matrix
    with chunk_rows 16, and its statistics equal to P0 (p0_equality). Records its canonical SHA-256 per model."""
    q = (backend or {}).get("quantization")
    if not isinstance(q, dict) or q.get("format") != Q4:
        rec.check(f"{label}: q4_k quantization object", "FAIL", f"{json.dumps(q)[:200] if q else None}")
        return
    m = ctx.model[short]
    layers = w8.layers_of(m)
    name, values = expected_warmup(m["dir"])
    w = q.get("warmup_check") or {}
    re_ = q.get("reconstruction_error") or {}
    ok = (w.get("layer"), w.get("projection"), w.get("values")) == (0, name, values) and \
        "bitwise equal" in str(w.get("result")) and re_.get("chunk_rows") == P0_CHUNK_ROWS and \
        q.get("matrices") == 4 * layers
    rec.expect(f"{label}: warm-up check of layer 0 {name} ({values} values), chunk_rows {P0_CHUNK_ROWS}, "
               f"{4 * layers} matrices", ok,
               f"warmup {json.dumps(w)[:200]}, chunk_rows {re_.get('chunk_rows')}, matrices {q.get('matrices')}")
    status, detail, rows = p0_equality(ctx, short, q)
    rec.check(f"{label}: q4_k load statistics equal to P0 (model and per projection type)", status, detail)
    sha = iso.canonical_sha(q)
    ctx.q4_objects.setdefault(short, {})[label] = sha
    ctx.results.setdefault("q4_p0_equality", {}).setdefault(short, {})[label] = {"status": status, "rows": rows,
                                                                                "object_sha256": sha}


SOURCE_POS_KEYS = {"layer", "projection", "tensor", "tensor_row", "matrix_row", "col"}


def counted(record):
    """(count, first, problem) of a group counter {count, first} (Q4KErrors::describe, the P0 facts): both keys
    required; count a non-negative integer; first explicitly null exactly when count is 0, else a complete position
    (SourcePos::describe: layer, projection, matrix_row, col integers or names; tensor and tensor_row both given or
    both null). A malformed record has a problem and is never equal."""
    def index(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0

    if not isinstance(record, dict) or not {"count", "first"} <= set(record):
        return None, None, "count or first missing"
    n, first = record["count"], record["first"]
    if not index(n):
        return n, first, f"count {n!r} is not a non-negative integer"
    if n == 0:
        return n, first, None if first is None else "count 0 with a non-null first position"
    pos_ok = isinstance(first, dict) and set(first) == SOURCE_POS_KEYS and \
        all(index(first[k]) for k in ("layer", "matrix_row", "col")) and first["projection"] in PROJECTIONS and \
        ((isinstance(first["tensor"], str) and index(first["tensor_row"])) or
         (first["tensor"] is None and first["tensor_row"] is None))
    return n, first, None if pos_ok else f"count {n} without a complete first position"


def p0_equality(ctx, short, q):
    """(status, detail, rows): the model and per-projection statistics of a q4_k object against the P0 facts of the
    same weights. Integers equal, floats as f64 bits (every one required on both sides), group counters by `counted`
    (a null first only with count 0, positions equal exactly). GAP without usable facts or without the same weights
    (SHA-256 of this run equal to P0's); FAIL (a defect) on any difference. The engine publishes the model and
    per-projection aggregates only, so per-matrix P0 records are not compared."""
    pm = p0_model(ctx.p0, short)
    if pm is None:
        return "GAP", f"no usable P0 facts ({ctx.p0_provenance})", []
    ident = pm.get("identity") or {}
    run_m = ctx.fp["models"].get(short) or {}
    w_run, w_p0 = run_m.get("safetensors_sha256"), (ident.get("weights") or {}).get("sha256")
    c_run, c_p0 = run_m.get("config_sha256"), (ident.get("config") or {}).get("sha256")
    if not w_run:
        return "GAP", "the weights' SHA-256 was not computed in this run (--no-weight-hash): not attributable", []
    if (w_run, c_run, run_m.get("revision")) != (w_p0, c_p0, ident.get("revision")):
        return "GAP", "not the weights, config or revision of P0: not comparable", []
    if (ctx.p0.get("reduction") or {}).get("chunk_rows") != P0_CHUNK_ROWS:
        return "GAP", "P0 facts do not record chunk_rows 16", []
    re_ = q.get("reconstruction_error") or {}
    fmt = (pm.get("formats") or {}).get(Q4) or {}
    pairs = [("model", re_.get("model"), fmt.get("aggregate"))] + \
        [(p, (re_.get("per_projection") or {}).get(p), (fmt.get("per_projection_type") or {}).get(p))
         for p in PROJECTIONS]
    rows, diffs = [], []
    if q.get("matrices") != (fmt.get("aggregate") or {}).get("matrices"):
        diffs.append(f"matrices {q.get('matrices')} vs P0 {(fmt.get('aggregate') or {}).get('matrices')}")
    for scope, e, p in pairs:
        s, ratio = (p or {}).get("stats") or {}, (p or {}).get("ratio") or {}
        g, pg = (e or {}).get("groups") or {}, s.get("groups") or {}
        er = (e or {}).get("relative_error_ratio") or {}
        fields = {
            "rows": ((e or {}).get("rows"), s.get("rows")),
            "values": ((e or {}).get("values"), s.get("values")),
            "values (P0 accumulated)": ((e or {}).get("values"), s.get("accumulated")),
            "sum_e2 (f64 bits)": (f64_bits((e or {}).get("sum_e2")), f64_bits(s.get("sum_e2"))),
            "sum_w2 (f64 bits)": (f64_bits((e or {}).get("sum_w2")), f64_bits(s.get("sum_w2"))),
            "max_abs_error (f64 bits)": (f64_bits((e or {}).get("max_abs_error")), f64_bits(s.get("max_abs_error"))),
            "R status": (er.get("status"), ratio.get("status")),
            "R (f64 bits)": (f64_bits(er.get("value")), f64_bits(ratio.get("value"))),
            "groups.total": (g.get("total"), pg.get("total")),
            "groups.zero": (g.get("zero"), pg.get("zero")),
        }
        bad = [k for k, (a, b) in fields.items() if a is None or b is None or a != b]
        for key in ("flattened", "zero_with_nonzero_reconstruction"):
            (na, fa, wa), (nb, fb, wb) = counted(g.get(key)), counted(pg.get(key))
            fields[f"groups.{key}.count"], fields[f"groups.{key}.first"] = (na, nb), (fa, fb)
            if wa or wb or na != nb or fa != fb:
                bad.append(f"groups.{key}" + "".join(f" ({side}: {w})" for side, w in (("engine", wa), ("P0", wb))
                                                     if w))
        # P0 records that the engine refuses at load (it would not have loaded) or skips nothing
        p0_only = {"P0 nonzero_constant groups": (pg.get("nonzero_constant") or {}).get("count"),
                   "P0 skipped rows": s.get("skipped_rows")}
        bad += [k for k, v in p0_only.items() if v != 0]
        rows.append({"scope": scope, "equal": not bad, "differing": bad,
                     "engine": {k: a for k, (a, _) in fields.items()}, "p0": {k: b for k, (_, b) in fields.items()}})
        diffs += [f"{scope}: {k}" for k in bad]
    if diffs:
        return "FAIL", f"{len(diffs)} differences: {'; '.join(diffs[:6])}", rows
    return "PASS", ("model and 4 projection types equal (integers, f64 bits, group counters with their first "
                    "positions); P0 per-matrix records are outside the engine's description"), rows


# ---------------------------------------------------------------- correctness


def same_backend_check(rec, label, ref_dir, out_dir, ordinal, mode):
    """The backend description of `out_dir` equal to `ref_dir`'s (a tree export on GPU 0) in every field, the
    quantization object included, except the device ordinal; a --no-prefix-cache export pins no template prefixes,
    so its `prefix_kv` is checked to be empty (the reference's not) instead of equal."""
    db = read_json(ref_dir / "manifest.json")["backend"]
    dc = read_json(out_dir / "manifest.json")["backend"]
    if mode.endswith("noprefix"):
        rec.expect(f"{label}: prefix_kv empty with --no-prefix-cache",
                   dc.get("prefix_kv") == [],
                   f"reference {db.get('prefix_kv')}, this export {dc.get('prefix_kv')}")
        db, dc = ({k: v for k, v in x.items() if k != "prefix_kv"} for x in (db, dc))
    ok, detail = w8.same_backend(db, dc, ordinal)
    rec.expect(f"{label}: backend description (quantization object included) equal except the device ordinal "
               f"{ordinal}" + (" and prefix_kv" if mode.endswith("noprefix") else ""), ok, detail)


def consistency_variants(short, tag):
    """(name, mode, --device, CUDA_VISIBLE_DEVICES) of the exports compared with the tree export on GPU 0."""
    return ((f"{short}-single-{tag}-dev0", "single", None, None),
            (f"{short}-tree-noprefix-{tag}-dev0", "tree-noprefix", None, None),
            (f"{short}-tree-{tag}-dev1", "tree", 1, None),
            (f"{short}-tree-{tag}-visible1-device0", "tree", 0, "1"))


def step_q8(rec, ctx):
    """q8_0 of the candidate byte-equal to q8_0 of the base, in every mode, card and CUDA_VISIBLE_DEVICES numbering."""
    for short in ctx.shorts:
        ref = reference(ctx, short)
        base = export(rec, ctx, f"{short}-tree-q8-base-dev0", short, "base", ref, "tree", quant=Q8)
        for name, mode, device, visible in ((f"{short}-tree-q8-dev0", "tree", None, None),
                                            *consistency_variants(short, "q8")):
            out = export(rec, ctx, name, short, "candidate", ref, mode, quant=Q8, device=device, visible=visible)
            if base is None or out is None:
                rec.check(f"q8 {name} vs {short}-tree-q8-base-dev0", "FAIL", "export missing")
                continue
            w8.identical(rec, ctx, f"{name} vs {short}-tree-q8-base-dev0", base, out, f"{name}-vs-tree-q8-base-dev0")
            same_backend_check(rec, f"q8 {name} vs the base", base, out, device or 0, mode)


def step_w4(rec, ctx):
    """q4_k results do not depend on the batching mode, prefix cache, card or CUDA_VISIBLE_DEVICES numbering; a
    difference is a defect (no quality waiver)."""
    for short in ctx.shorts:
        inputs = reference(ctx, short)
        ref = export(rec, ctx, f"{short}-tree-q4-dev0", short, "candidate", inputs, "tree", quant=Q4)
        for name, mode, device, visible in consistency_variants(short, "q4"):
            out = export(rec, ctx, name, short, "candidate", inputs, mode, quant=Q4, device=device, visible=visible)
            if ref is None or out is None:
                rec.check(f"q4 {name} vs {short}-tree-q4-dev0", "FAIL", "export missing")
                continue
            w8.identical(rec, ctx, f"{name} vs {short}-tree-q4-dev0", ref, out, f"{name}-vs-tree-q4-dev0")
            same_backend_check(rec, f"q4 {name} vs {short}-tree-q4-dev0", ref, out, device or 0, mode)


INVALID_KINDS = {  # kind -> (CLI flags, YAML fields, quantization, expected message)
    "bf16": (["--dtype", "bf16"], {"dtype": "bf16"}, Q4, "quantization q4_k needs dtype f16"),
    "f32": (["--dtype", "f32"], {"dtype": "f32"}, Q4, "quantization q4_k needs dtype f16"),
    "candle": (["--kernels", "candle"], None, Q4, "quantization q4_k needs kernels fused"),
    "q4_0": ([], {"dtype": "f16"}, "q4_0", 'unknown quantization "q4_0"'),
    "Q4_K": ([], {"dtype": "f16"}, "Q4_K", 'unknown quantization "Q4_K"'),
}


def invalid_cases(d, cfg, ref, request):
    """(label, arguments after the binary, expected message) of every refused q4_k configuration."""
    cmd = {"serve": lambda label: ["serve", "--model", ABSENT],
           "export": lambda label: ["export", "--model", ABSENT, "--inputs", ref, "--out", d / f"{label}-out"],
           "bench": lambda label: ["bench", "--model", ABSENT, "--reference", ref, "--out", d / f"{label}.json"],
           "benchmark": lambda label: ["benchmark", "--model", ABSENT, "--out", d / f"{label}.json"],
           "decide": lambda label: ["decide", "--model", ABSENT, "--request", request],
           "profile": lambda label: ["profile", "--model", ABSENT, "--reference", ref]}
    cases = []
    for kind, (_, fields, quant, needle) in INVALID_KINDS.items():
        if fields is not None:
            cases.append((f"serve-yaml-{kind}", ["serve", "--config", cfg(f"yaml-{kind}", quantization=quant,
                                                                          **fields)], needle))
    cases.append(("serve-cli-with-config", ["serve", "--config", cfg("plain", dtype="f16"), "--quantization", Q4],
                  "--quantization applies to --model"))
    for command, build in cmd.items():
        for kind, (flags, _, quant, needle) in INVALID_KINDS.items():
            label = f"{command}-{kind}"
            cases.append((label, [*build(label), "--quantization", quant, *flags], needle))
    return cases


def step_invalid(rec, ctx):
    """Refused q4_k configurations: the validation message, before any download, load or GEMM search."""
    d = ctx.out / "correctness" / "invalid"
    d.mkdir(parents=True)
    port, cache, ref = ctx.args.port, d / "gemm-cache", reference(ctx, "mini")
    absent = {"repo": ABSENT}
    request = d / "request.json"
    request.write_text(json.dumps({"state": "Klient pisze, że chce oddać telefon.",
                                   "questions": {"q": {"type": "noul", "instructions": "Czy klient prosi o zwrot?"}}}))

    def cfg(label, **kw):
        return w8.serve_yaml(d / f"{label}.yml", port, cache, [dict(absent, **kw)])

    for label, extra, needle in invalid_cases(d, cfg, ref, request):
        w8.guard(ctx, rec, f"invalid {label}")
        mg.port_free_or_fail(port)
        log = d / f"{label}.log"
        r = w8.run_bounded(rec, ctx, f"invalid-{label}", [ctx.binary["candidate"], "--color", "never", *extra],
                           {"HF_HUB_OFFLINE": "1"}, log, 120, physical=set())
        if r["stopped"]:
            w8.judge_run(rec, f"invalid {label}", r)
            w8.stop_after(f"invalid {label}", r)
            continue
        text = log.read_text(errors="replace")
        early = not any(x in text for x in ("fetching the model files", "loaded on GPU", SEARCH, "serving on http"))
        rec.expect(f"invalid {label}: refused before download and load",
                   r["exit"] not in (0, None) and needle in text and early,
                   f"exit {r['exit']} after {r['seconds']:.1f} s, expected message {needle!r}, early {early}")
        if r["placement_mib"]:
            rec.check(f"invalid {label}: no GPU memory of the process", "FAIL", f"{r['placement_mib']}")
        elif r["seconds"] < 1.0:
            rec.check(f"invalid {label}: no GPU memory of the process", "GAP",
                      f"ended after {r['seconds']:.2f} s, before a 0.5 s placement sample (the log shows no load)")
        else:
            rec.check(f"invalid {label}: no GPU memory of the process", "PASS", "none in the 0.5 s samples")
    doctor_cases(rec, ctx, d, port, cache, absent)
    rec.check("invalid q4_k on Metal or Intel", "GAP", "no such device in this run; static review and the "
              "Intel/Metal compile checks only")


def known_k_model(d, mini):
    """A model directory for doctor: mini's config.json with intermediate_size + 32 (the down projection's K no
    longer a multiple of 256) and mini's checkpoint linked (doctor reads only its size); nothing of mini changes."""
    m = d / "known-k-model"
    m.mkdir()
    c = json.loads((mini["dir"] / "config.json").read_text())
    c["intermediate_size"] += 32
    (m / "config.json").write_text(json.dumps(c, indent=1))
    (m / "model.safetensors").symlink_to(mini["dir"] / "model.safetensors")
    return m, c["intermediate_size"]


def doctor_cases(rec, ctx, d, port, cache, absent):
    mini = ctx.model["mini"]
    table = ctx.tables["mini"]
    known_k, k = known_k_model(d, mini)
    valid = {"dir": mini["dir"], "dtype": "f16", "gemm_table": table}
    cases = (  # (label, binary, entries, expected failure text or None, estimate text)
        ("doctor-q4-bf16", "candidate", [dict(absent, dtype="bf16", quantization=Q4)],
         "quantization q4_k needs dtype f16", None),
        ("doctor-q4-unknown", "candidate", [dict(absent, dtype="f16", quantization="Q4_K")],
         'unknown quantization "Q4_K"', None),
        ("doctor-q4-known-k", "candidate", [{"dir": known_k, "dtype": "f16", "quantization": Q4, "gemm_table": table}],
         "quantization q4_k needs every projection K to be a multiple of 256", None),
        ("doctor-q4-mini", "candidate", [dict(valid, quantization=Q4)], None, "q4_k weights"),
        ("doctor-q8-mini-candidate", "candidate", [dict(valid, quantization=Q8)], None, "q8_0 weights"),
        ("doctor-q8-mini-base", "base", [dict(valid, quantization=Q8)], None, "q8_0 weights"))
    estimates = {}
    for label, binary, entries, want_fail, estimate in cases:
        w8.guard(ctx, rec, f"invalid {label}")
        conf = w8.serve_yaml(d / f"{label}.yml", port, cache, entries)
        log = d / f"{label}.json"
        r = w8.run_bounded(rec, ctx, label, [ctx.binary[binary], "--color", "never", "doctor", "--config", conf,
                                             "--json"], {"HF_HUB_OFFLINE": "1"}, log, 180)
        if r["stopped"]:
            w8.judge_run(rec, label, r)
            w8.stop_after(label, r)
            continue
        try:
            report = rp.doctor_report(log.read_text(errors="replace"))
        except ValueError:
            rec.check(label, "FAIL", f"no JSON report (exit {r['exit']})")
            continue
        models = [c for c in report.get("checks", []) if c.get("area") == "models"]
        fails = [c.get("what", "") for c in models if c.get("level") == "fail"]
        if want_fail:
            rec.expect(f"{label}: reported as a model failure ({want_fail!r})", any(want_fail in f for f in fails),
                       iso.scrub(f"{fails}") + (f"; config.json intermediate_size {k}" if "known-k" in label else ""))
        else:
            est = [c.get("what", "") for c in models
                   if estimate in c.get("what", "") and "estimate" in c.get("what", "")]
            rec.expect(f"{label}: valid entry without failure, weight estimate reported", bool(est) and not fails,
                       iso.scrub(f"estimate {est}, failures {fails}"))
            estimates[label] = [re.sub(r"^.*?: ", "", x) for x in est]
    if "doctor-q8-mini-base" in estimates and "doctor-q8-mini-candidate" in estimates:
        rec.expect("doctor: q8_0 estimate of the candidate equal to the base's",
                   estimates["doctor-q8-mini-base"] == estimates["doctor-q8-mini-candidate"],
                   f"base {estimates['doctor-q8-mini-base']}, candidate {estimates['doctor-q8-mini-candidate']}")
    ctx.results["doctor_estimates"] = estimates


# ---------------------------------------------------------------- serve


COMPUTE_APPS = ["--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader,nounits"]


def bounded_wait(own, deadline, fn):
    """fn(limit), limit = the time left before the absolute end min(now + `own`, `deadline`), taken right before the
    call; the time of its return is taken right after it, before anything else. Returns (result, None) for a return
    by the end, else (result or None, why): why = {cause, late_s} for subprocess.TimeoutExpired (late_s None) or a
    normal return seen only after the end (not timely, as w8.run_bounded). cause is the earlier end: "deadline" when
    `deadline` is strictly before the own end, else "timeout" (the convention of w8.run_bounded and w8.Served.get)."""
    own_end = time.monotonic() + own
    end = min(own_end, deadline)
    cause = "deadline" if deadline < own_end else "timeout"
    try:
        result = fn(max(0.0, end - time.monotonic()))
    except subprocess.TimeoutExpired:
        return None, {"cause": cause, "late_s": None}
    seen = time.monotonic()
    return (result, None) if seen <= end else (result, {"cause": cause, "late_s": seen - end})


def not_timely(rec, ctx, what, own, why, observed=""):
    """The GAP or FAIL of an operation that ended past its absolute end (bounded_wait's `why`): the run deadline first
    -> GAP and Deadline; its own limit first -> FAIL of class timeout and Fatal (suspected hung GPU). `observed`
    names what was seen only after the end (kept, never used as a timely result)."""
    late = "" if why["late_s"] is None else f"; {observed or 'its result'} seen only {why['late_s']:.3f} s after the " \
        "end (not timely, not used)"
    if why["cause"] == "deadline":
        ctx.deadline_hit = True
        rec.check(what, "GAP", f"cut at the run deadline (own limit {own} s){late}; not judged")
        raise Deadline(what)
    rec.check(what, "FAIL", f"no exit within its own {own} s{late}", cls="timeout")
    raise Fatal(f"{what}: no exit within {own} s: suspected hung GPU, no later GPU work runs")


def bounded_probe(rec, ctx, what, own, fn):
    """fn(limit) of a short direct foreground subprocess (nvidia-smi, --version) bounded by bounded_wait (own limit
    `own`, the run deadline); nothing starts after the deadline. Past the earlier end, also a normal return seen only
    after it: not_timely (deadline: GAP and Deadline; own limit: FAIL of class timeout and Fatal)."""
    w8.guard(ctx, rec, what)
    result, why = bounded_wait(own, ctx.deadline, fn)
    if why:
        not_timely(rec, ctx, what, own, why)
    return result


def idle_gpus(rec, ctx, what, timeout=30):
    """iso.wait_gpus_idle bounded by the run deadline as well: [] when no compute process holds a GPU, else the rows
    still seen after `timeout` s (a previous server may still be releasing memory)."""
    end = time.monotonic() + timeout
    while True:
        rows = bounded_probe(rec, ctx, what, 30, lambda limit: mg.smi(COMPUTE_APPS, timeout=limit))
        if not rows or time.monotonic() > end:
            return rows
        time.sleep(max(0.0, min(1.0, ctx.deadline - time.monotonic())))


def placement_sample(rec, ctx, what, root_pid):
    """mg.session_placement(root_pid, ctx.by_bus) with its nvidia-smi a bounded_probe (own 30 s, the run deadline)
    instead of the helper's fixed 30 s: physical GPU index -> MiB of the processes in the session of `root_pid`,
    {} when none (observed), None when no row could be read (not observable). The rows are filtered and summed
    exactly as the helper does; a late or timed-out sample raises (Deadline or Fatal) and is never used."""
    rows = bounded_probe(rec, ctx, what, 30, lambda limit: mg.smi(COMPUTE_APPS, timeout=limit))
    seen, out = False, {}
    for row in rows:  # mg.session_placement, unchanged from here
        try:
            pid, b, mib = int(row[0]), mg.bus(row[1]), float(row[2])
        except (ValueError, IndexError):
            continue
        seen = True
        try:
            if os.getsid(pid) != root_pid:
                continue
        except (ProcessLookupError, PermissionError):
            continue
        idx = ctx.by_bus.get(b)
        out[idx] = out.get(idx, 0.0) + mib
    return out if (seen or not rows) else None


class Served(w8.Served):
    """w8.Served whose own-timeout failures are FAILs of class timeout that stop the run (readiness, a request past
    --http-timeout, /health, /v1/models and /metrics after readiness past their 30 s, a graceful shutdown past
    --shutdown-timeout, a placement sample past its 30 s; an exit or sample seen only after its end is not timely):
    a suspected hung GPU. The run deadline is checked again after the idle wait, right before the server starts, and
    also ends the graceful shutdown wait and the placement samples (GAP and Deadline, never a timely result)."""

    def __init__(self, rec, ctx, label, binary, entries, delta=None):  # w8.Served.__init__ with the changes above
        w8.guard(ctx, rec, f"serve {label}")
        self.rec, self.ctx, self.label = rec, ctx, label
        self.dir = ctx.out / "correctness" / "serve" / label
        self.dir.mkdir(parents=True)
        busy = idle_gpus(rec, ctx, f"serve {label}: GPUs idle before the start")
        if busy:
            raise StepFailed(f"compute processes still on the GPUs before {label}: {len(busy)}")
        cfg = w8.serve_yaml(self.dir / "serve.yml", ctx.args.port, self.dir / "gemm-cache", entries)
        w8.guard(ctx, rec, f"serve {label}")  # the idle wait may have used the time left
        self.srv = iso.Server(rec, ctx, self.dir, ctx.binary[binary], cfg, delta or {})
        self.url = self.srv.url
        self.batches, self.closed = [], False
        try:
            try:
                self.wait_ready()
            except StepFailed as e:
                if "not ready after" not in str(e):
                    raise
                rec.check(f"serve {label}: ready", "FAIL", iso.scrub(str(e)), cls="timeout")
                raise Fatal(f"serve {label}: not ready within --ready-timeout: suspected hung GPU, no later GPU work "
                            "runs")
            time.sleep(min(2.0, max(0.0, ctx.deadline - time.monotonic())))
            self.placement = placement_sample(rec, ctx, f"serve {label}: placement sample", self.srv.proc.pid)
            self.health = self.get("/health", 30, "GET /health")
            self.models = self.get("/v1/models", 30, "GET /v1/models")
            self.health_raw, self.models_raw = self.health[1], self.models[1]
            log = (self.dir / "server.log").read_text(errors="replace")
            if SEARCH in log:
                raise StepFailed(f"server {label} announced a GEMM search")
        except BaseException:
            self.close()
            raise

    def get(self, path, limit, what):
        """w8.Served.get; past its own limit after readiness (a readiness probe is polled again within
        --ready-timeout): a FAIL of class timeout and Fatal, the bytes received kept by w8.Served.get. Another
        transport error stays an OSError (not a suspected hang); the run deadline first stays a GAP and Deadline."""
        try:
            return super().get(path, limit, what)
        except TimeoutError as e:
            if what == "readiness":
                raise
            self.rec.check(f"serve {self.label}: {what}", "FAIL",
                           iso.scrub(f"no complete response within its own {limit} s: {e}"), cls="timeout")
            raise Fatal(f"serve {self.label}: {what}: no response within {limit} s: suspected hung GPU, no later GPU "
                        "work runs")

    def run(self, items, clients, phase):
        slots = [None] * len(items)
        self.batches.append(slots)
        why = w8.send_items(self.ctx, self.url, items, clients, phase, slots)
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
                           iso.scrub(f"{len(records)} of {len(items)} sent, first error {first}; server stopped"),
                           cls="timeout" if why == "timeout" else "correctness")
            if why == "timeout":
                raise Fatal(f"serve {self.label} {phase}: no answer within --http-timeout {self.ctx.args.http_timeout}"
                            " s: suspected hung GPU, no later GPU work runs")
            raise StepFailed(f"{phase}: transport error, server {self.label} stopped")
        return records

    def shutdown_check(self):
        """SIGTERM, then the exit awaited until the absolute end min(SIGTERM + --shutdown-timeout, the run deadline)
        (bounded_wait): no exit by it, or an exit seen only after it, is not timely (not_timely: the deadline first a
        GAP and Deadline, the own limit first a FAIL of class timeout and Fatal; the exit status kept in the detail).
        The cleanup after either (close: SIGTERM to the group, SIGKILL after 60 s) is bounded separately and is never
        a graceful shutdown. Then the session's GPU memory after the exit (placement_sample)."""
        what = f"serve {self.label}: graceful shutdown"
        w8.guard(self.ctx, self.rec, what)
        own = self.ctx.args.shutdown_timeout

        def term(limit):
            os.kill(self.srv.proc.pid, signal.SIGTERM)
            return self.srv.proc.wait(limit)

        code, why = bounded_wait(own, self.ctx.deadline, term)
        if why:
            not_timely(self.rec, self.ctx, what, own, why, f"exit {code}")
        log = (self.dir / "server.log").read_text(errors="replace")
        self.rec.expect(what, code == 0 and "shutting down" in log, f"exit {code}")
        name = f"serve {self.label}: no GPU memory left after exit"
        after = placement_sample(self.rec, self.ctx, name, self.srv.proc.pid)
        self.rec.check(name, "FAIL" if after else "PASS" if after == {} and self.placement else "GAP", f"{after}")


def served_decide(rec, ctx, quant, k, cid, raw):
    """`basal decide` of one exact served request body with the flags of the `quant` tree export (candidate, mini,
    table, f16, fused, GPU 0): the served path of a case that export refused as Unsupported. Its JSON when it is a
    typed native response (exit 0: an answer; exit 2: Validation, Usage or Unsupported), else None after a FAIL."""
    d = ctx.out / "correctness" / f"serve-decide-{quant}"
    d.mkdir(parents=True, exist_ok=True)
    slug = f"{k:02d}-{re.sub(r'[^A-Za-z0-9._-]+', '-', cid)}"
    request = d / f"{slug}.request.json"
    request.write_bytes(raw)
    label = f"serve decide {quant} {cid}"
    w8.guard(ctx, rec, label)
    argv = w8.gpu_argv(ctx, "candidate", "decide", "mini", quant, ["--request", request])
    stdout = d / f"{slug}.json"
    r = w8.run_bounded(rec, ctx, f"decide-{quant}-{slug}", argv, {}, d / f"{slug}.log", ctx.args.export_timeout,
                       physical={0}, stdout=stdout)
    if r["stopped"]:
        w8.judge_run(rec, label, r)
        w8.stop_after(label, r)
        return None
    mg.placement_check(rec, f"{label}: memory only on physical GPU 0", r["placement_mib"], {0})
    try:
        v = json.loads(stdout.read_text())
    except (OSError, ValueError) as e:
        rec.check(f"{label}: typed native response", "FAIL", f"exit {r['exit']}, no JSON ({type(e).__name__})")
        return None
    status, sha = w8.expected_status(v), hashlib.sha256(raw).hexdigest()
    same = mg.sha256(request) == sha
    ok = same and ((r["exit"] == 0 and status == 200) or (r["exit"] == 2 and status in (400, 422)))
    rec.expect(f"{label}: typed native response", ok, f"exit {r['exit']}, body for HTTP status {status}, request "
               f"sha256 {sha[:12]} ({'the' if same else 'NOT the'} body sent)")
    return v if ok else None


def served_expected(rec, ctx, quant, served, bodies):
    """item id -> (status, canonical body, source) the `quant` server must answer, from the `quant` export of the
    served request bodies: its answer, or its error where the request was refused before the model ran; a case it
    refused as Unsupported (the export's generic outcome) from served_decide. Returns (map or None, provenance)."""
    out, rows, problems = {}, [], []
    for k, line in enumerate(w8.read_jsonl(served / "systemone.jsonl")):
        rid = f"short:{line['id']}"
        raw = bodies.get(rid)
        row = {"id": rid, "request_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None}
        rows.append(row)
        if raw is None or line.get("request") != json.loads(raw):
            problems.append(f"{rid}: the export's request is not the body sent")
            continue
        refused = line.get("error") if "answer" not in line else None
        if isinstance(refused, dict) and refused.get("error") == "unsupported":
            v = served_decide(rec, ctx, quant, k, line["id"], raw)
            row["source"] = "basal decide"
            if v is None:
                problems.append(f"{rid}: no typed basal decide response")
                continue
        else:
            v, row["source"] = line.get("answer", refused), "export"
        status = w8.expected_status(v)
        row.update(expected_status=status, expected_sha256=hashlib.sha256(w8.canonical_body(v).encode()).hexdigest())
        if status is None:
            problems.append(f"{rid}: no expected status ({row['source']}: {w8.canonical_body(v)[:120]})")
            continue
        out[rid] = (status, w8.canonical_body(v), row["source"])
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(bodies):
        problems.append(f"export cases {len(rows)}, served System One requests {len(bodies)}")
    by = Counter(r.get("source") for r in rows)
    rec.expect(f"serve {quant} reference: an expected (status, body) for every served System One request",
               not problems, f"{len(out)}/{len(bodies)}: {by['export']} from the export of the served bodies, "
               f"{by['basal decide']} from basal decide" + iso.scrub("".join(f"; {p}" for p in problems[:3])))
    return (None if problems else out), rows


def served_unchanged(rec, original, served, names):
    """The export of the served bodies against the tree export of the reference (same binary, flags, GPU):
    bench.jsonl and the System One lines of the cases whose request is unchanged byte-equal."""
    a_name, b_name = names
    rec.expect(f"serve inputs: bench.jsonl of {b_name} byte-equal to {a_name}",
               mg.sha256(served / "bench.jsonl") == mg.sha256(original / "bench.jsonl"))

    def lines(path):
        return {json.loads(t)["id"]: t for t in (path / "systemone.jsonl").read_text().splitlines() if t.strip()}

    a, b = lines(original), lines(served)
    same = [k for k in a if k in b and json.loads(a[k]).get("request") == json.loads(b[k]).get("request")]
    bad = [k for k in same if a[k] != b[k]]
    name = f"serve inputs: System One lines of the unchanged requests of {b_name} byte-equal to {a_name}"
    detail = f"{len(same) - len(bad)}/{len(same)} equal, differing {bad[:5]}; {len(b) - len(same)} with `model` " \
             "rewritten not compared"
    if same:
        rec.expect(name, not bad, detail)
    else:
        rec.check(name, "GAP", detail)


def served_parity(rec, ctx, label, quant, answers, want, provenance, ids):
    """Every served System One request of server `label` equal in (status, canonical body) to the native `quant`
    reference of the same body; the table goes to serve-inputs/<label>-served-comparison.json."""
    name = f"serve {label}: System One (status, answer: multi, evidence, 400/422) equal to the native {quant} " \
           "reference of the same requests"
    if want is None:
        rec.check(name, "FAIL", "no complete native reference of the served requests (see the serve inputs checks)")
        return
    table, bad = [], []
    for row in provenance:
        got, exp = answers.get(row["id"]), want.get(row["id"])
        equal = got is not None and exp is not None and got == exp[:2]
        table.append(dict(row, http_status=got[0] if got else None, equal=equal,
                          http_sha256=hashlib.sha256(got[1].encode()).hexdigest() if got and got[1] else None))
        if not equal:
            bad.append(f"{row['id']} (HTTP {got[0] if got else None}, expected {exp[0] if exp else None})")
    missing = sorted(set(ids) ^ {r["id"] for r in provenance})
    by = Counter(r.get("source") for r in provenance)
    summary = {"cases": len(table), "equal": sum(r["equal"] for r in table), "sources": dict(by),
               "http_statuses": dict(Counter(str(r["http_status"]) for r in table)), "not_compared": missing}
    mg.write_json(ctx.out / "correctness" / "serve-inputs" / f"{label}-served-comparison.json",
                  {**summary, "rows": table})
    ctx.results.setdefault("served_comparison", {})[label] = summary
    rec.expect(name, not bad and not missing and len(table) == len(ids) > 0,
               f"{summary['equal']}/{len(ids)} equal ({by['export']} expected from the export of the served bodies, "
               f"{by['basal decide']} from basal decide), HTTP statuses {summary['http_statuses']}, differing "
               f"{bad[:5]}, not compared {missing[:5]}")


def surface(srv):
    """/health and /v1/models (status, bytes) and the metric families (HELP and TYPE lines) of a running server;
    the families are valid evidence only from a status 200 body with both HELP and TYPE lines."""
    status, _ = srv.scrape(srv.dir / "metrics.txt")
    families = rp.metric_families((srv.dir / "metrics.txt").read_text(errors="replace")) if status == 200 else []
    valid = any(f.startswith("# HELP ") for f in families) and any(f.startswith("# TYPE ") for f in families)
    return {"health": srv.health, "models": srv.models, "metrics_status": status, "metrics_valid": valid,
            "families": families}


def same_surface(rec, label, got, ref, ref_label):
    """/health and /v1/models: status 200 and bytes equal; metric families: both scrapes valid and the HELP and TYPE
    sets equal (both directions; sample values and labels are not compared). An invalid reference surface is a GAP
    (nothing to compare with), never a pass; an invalid surface of this server a FAIL."""
    for key, what in (("health", "/health"), ("models", "/v1/models")):
        name = f"serve {label}: {what} status and bytes equal to {ref_label}"
        if ref[key][0] != 200:
            rec.check(name, "GAP", f"the reference answered status {ref[key][0]}: no valid reference")
        else:
            rec.expect(name, got[key] == ref[key], f"status {got[key][0]}, reference {ref[key][0]}, bytes "
                       f"{'equal' if got[key][1] == ref[key][1] else 'different'}")
    name = f"serve {label}: metric families (HELP and TYPE) equal to {ref_label}"
    if not ref["metrics_valid"]:
        rec.check(name, "GAP", f"no valid reference scrape (status {ref['metrics_status']}, {len(ref['families'])} "
                  "HELP/TYPE lines)")
        return
    missing = sorted(set(ref["families"]) - set(got["families"]))
    extra = sorted(set(got["families"]) - set(ref["families"]))
    rec.expect(name, got["metrics_valid"] and not missing and not extra,
               f"status {got['metrics_status']}, {len(got['families'])} lines (reference {len(ref['families'])}); "
               f"missing {missing[:3]}, extra {extra[:3]}")


def http_reference(rec, s, label, items):
    """Every item once in order on server `s`: (id -> (status, answer)) or StepFailed without a full reference."""
    recs = s.run(items, 1, "sequential")
    errors = [r["id"] for r in recs if r["status"] is None or r["status"] >= 500]
    ok = rec.expect(f"serve {label}: HTTP reference", len(recs) == len(items) and not errors,
                    f"{Counter(str(r['status']) for r in recs)}, transport/5xx {errors[:5]}")
    if not ok:
        raise StepFailed(f"no HTTP reference of {label}")
    return {r["id"]: (r["status"], r["answer"]) for r in recs}


def native_references(rec, ctx, items, quants):
    """quant -> (expected map or None, provenance) from the tree export of the served request bodies."""
    out = {}
    if ctx.served_inputs is None:
        ctx.served_inputs = w8.served_inputs(rec, ctx, items) or False
    derived = ctx.served_inputs or None
    for quant in quants:
        tag = "q8" if quant == Q8 else "q4"
        ref = export(rec, ctx, f"mini-tree-{tag}-dev0", "mini", "candidate", reference(ctx, "mini"), "tree",
                     quant=quant)
        out[quant] = (None, [])
        if derived is None:
            continue
        name = f"mini-tree-{tag}-served-dev0"
        served = export(rec, ctx, name, "mini", "candidate", derived[0], "tree", quant=quant)
        if served is None:
            continue
        if ref is not None:
            served_unchanged(rec, ref, served, (f"mini-tree-{tag}-dev0", name))
        out[quant] = served_expected(rec, ctx, quant, served, derived[1])
    return out


def serve_session(rec, ctx, label, binary, entries, body):
    """A fresh server for `body(s)`, its shutdown checked, its records written once (also when stopped)."""
    s = Served(rec, ctx, label, binary, entries)
    try:
        result = body(s)
        s.shutdown_check()
        return result
    finally:
        s.close()


def step_serve(rec, ctx):
    """The HTTP contracts: none and q8_0 of the candidate equal to the base's servers (q8_0 also to its native
    reference); q4_k equal to its native reference, and its replicas to the single q4_k server."""
    mini = ctx.model["mini"]
    name = mini["name"]
    ctx.inputs, ctx.long_file = reference(ctx, "mini"), ctx.root / w8.LONG
    items = rp.http_items(ctx, name)
    ids = [k for k, _, path, _ in items if k.startswith("short:") and path == "/v1/systemone"]
    entry = {"dir": mini["dir"], "gemm_table": ctx.tables["mini"], "dtype": "f16",
             "long_tokens": ctx.args.long_tokens}
    native = native_references(rec, ctx, items, (Q8, Q4))
    refs = {}

    def reference_server(s, label, key):
        mg.placement_check(rec, f"serve {label}: memory only on physical GPU 0", s.placement, {0})
        refs[key] = http_reference(rec, s, label, items)
        refs[f"{key}-surface"] = surface(s)

    def candidate_server(s, label, key, ref_key, quant):
        mg.placement_check(rec, f"serve {label}: memory only on physical GPU 0", s.placement, {0})
        recs = s.run(items, 1, "sequential")
        answers = {r["id"]: (r["status"], r["answer"]) for r in recs}
        if ref_key:
            rp.parity(rec, f"serve {label} sequential vs {ref_key}", recs, refs[ref_key], len(items))
            same_surface(rec, label, surface(s), refs[f"{ref_key}-surface"], ref_key)
        else:
            errors = [r["id"] for r in recs if r["status"] is None or r["status"] >= 500]
            rec.expect(f"serve {label}: every request answered, no transport error or 5xx",
                       len(recs) == len(items) and not errors, f"{Counter(str(r['status']) for r in recs)}")
            if "none-base-surface" in refs:
                same_surface(rec, label, surface(s), refs["none-base-surface"], "none-base-dev0")
            else:
                rec.check(f"serve {label}: /health, /v1/models and metric families", "GAP",
                          "no none-base-dev0 server to compare with")
        if quant:
            want, provenance = native.get(quant, (None, []))
            served_parity(rec, ctx, label, quant, answers, want, provenance, ids)
        refs[key] = answers

    def part(label, fn):
        try:
            fn()
        except (StepFailed, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
            rec.check(f"serve {label}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))

    def none_part():
        serve_session(rec, ctx, "none-base-dev0", "base", [dict(entry, device=0)],
                      lambda s: reference_server(s, "none-base-dev0", "none-base"))
        serve_session(rec, ctx, "none-cand-dev0", "candidate", [dict(entry, device=0)],
                      lambda s: candidate_server(s, "none-cand-dev0", "none-cand", "none-base", None))

    def q8_part():
        serve_session(rec, ctx, "q8-base-dev0", "base", [dict(entry, device=0, quantization=Q8)],
                      lambda s: reference_server(s, "q8-base-dev0", "q8-base"))
        serve_session(rec, ctx, "q8-cand-dev0", "candidate", [dict(entry, device=0, quantization=Q8)],
                      lambda s: candidate_server(s, "q8-cand-dev0", "q8-cand", "q8-base", Q8))

    def q4_part():
        serve_session(rec, ctx, "q4-cand-dev0", "candidate", [dict(entry, device=0, quantization=Q4)],
                      lambda s: candidate_server(s, "q4-cand-dev0", "q4-cand", None, Q4))
        if "q4-cand" not in refs:
            raise StepFailed("no single q4_k server answers for the replicas")
        serve_session(rec, ctx, "q4-replicas", "candidate", [dict(entry, replica_devices=[0, 1], quantization=Q4)],
                      lambda s: replicas(s))

    def replicas(s):
        mg.placement_check(rec, "serve q4-replicas: memory on physical GPUs 0 and 1", s.placement, {0, 1})
        _, before = s.scrape()
        ref = refs["q4-cand"]
        rp.parity(rec, "serve q4-replicas sequential vs q4-cand-dev0", s.run(items, 1, "sequential"), ref, len(items))
        for k in range(ctx.args.rounds):
            rp.parity(rec, f"serve q4-replicas {ctx.args.clients} clients round {k} vs q4-cand-dev0",
                      s.run(items, ctx.args.clients, f"concurrent-{k}"), ref, len(items))
        _, after = s.scrape(s.dir / "metrics.txt")
        routed = rp.delta(rp.replica_stats(after, name, 2), rp.replica_stats(before, name, 2))
        counts = [routed[r]["routed_requests_total"] for r in ("0", "1")]
        rec.expect("serve q4-replicas: both replicas routed requests", all(c and c > 0 for c in counts),
                   f"routed {counts}")
        ctx.results["q4_replicas_routed"] = counts

    part("none servers", none_part)
    part("q8_0 servers", q8_part)
    part("q4_k servers", q4_part)
    ctx.results["http_reference_sha256"] = {k: iso.canonical_sha(v) for k, v in refs.items()
                                            if not k.endswith("-surface")}


# ---------------------------------------------------------------- quality: 44 + 34 against FP32


def changed_fields(cmp, upstream):
    """Every changed typed field of a comparison: [{case, question, fields, diff}] (upstream: the endpoint answers
    against Server.decide; else the System One answers of two runtime exports)."""
    out = []
    if upstream:
        for case in cmp.get("upstream_endpoint", []):
            for q in case.get("questions", []):
                bad = sorted(k for k, v in (q.get("diff") or {}).items() if k.endswith("_same") and v is False)
                if bad or q.get("only_in_one"):
                    out.append({"case": case.get("id"), "question": q.get("question"), "fields": bad,
                                "only_in_one": q.get("only_in_one"), "diff": q.get("diff")})
    else:
        for case in cmp.get("systemone", {}).get("cases", []):
            for it in case.get("items", []):
                bad = sorted(k for k, v in (it.get("answer") or {}).items() if k.endswith("_same") and v is False)
                if bad:
                    out.append({"case": case.get("id"), "question": it.get("item"), "fields": bad,
                                "diff": it.get("answer")})
    return out


def spans_only(diff):
    """The evidence spans of a diff without their texts (for summary.json)."""
    out = {}
    for k, v in (diff or {}).items():
        if k.startswith("evidence_") and isinstance(v, list):
            out[k] = [x[:2] if isinstance(x, list) else x for x in v]
    return out


def collected(v, key):
    """Every finite number under `key` anywhere in `v`."""
    out = []
    if isinstance(v, dict):
        for k, x in v.items():
            if k == key and w8.finite(x):
                out.append(x)
            else:
                out += collected(x, key)
    elif isinstance(v, list):
        for x in v:
            out += collected(x, key)
    return out


def continuous(cmp, upstream):
    """The comparison's continuous differences (diagnostics, no threshold): W8's numeric summary and the
    distributions of |dp| (max over options) and |confidence change| per compared question."""
    scope = cmp.get("upstream_endpoint", []) if upstream else cmp.get("systemone", {}).get("cases", [])
    return {**w8.numeric_summary(cmp),
            "question_max_abs_prob_diff": w8.dist(collected(scope, "max_abs_prob_diff")),
            "question_abs_confidence_diff": w8.dist([abs(x) for x in collected(scope, "confidence_diff")])}


def question_answer(case, key, question):
    answer, error = w8.endpoint_answer(case, key)
    if error is not None:
        return {"error": error}
    return ((answer or {}).get("answers") or {}).get(question)


SMALL = (  # (label, a, b, role)
    ("fp32-vs-q4", "fp32", "q4", "criterion"), ("fp32-vs-f16", "fp32", "f16", "control"),
    ("f16-vs-q4", "f16", "q4", "diagnostic"), ("q8-vs-q4", "q8", "q4", "diagnostic"),
    ("fp32-vs-q8", "fp32", "q8", "context"))


def quality_small(rec, ctx, short):
    """44 + 34 with the existing W8 predicate: q4_k against FP32 (criterion), the f16 control, q4_k against f16 and
    q8_0 of the session (diagnostics), q8_0 and MLX 8-bit against FP32 (context)."""
    ref = reference(ctx, short)
    dirs = {"fp32": ref, "f16": export(rec, ctx, f"{short}-tree-none-dev0", short, "candidate", ref, "tree"),
            "q8": export(rec, ctx, f"{short}-tree-q8-dev0", short, "candidate", ref, "tree", quant=Q8),
            "q4": export(rec, ctx, f"{short}-tree-q4-dev0", short, "candidate", ref, "tree", quant=Q4)}
    res = ctx.results.setdefault("quality_small", {}).setdefault(short, {})
    tokens, raw = {}, {"model": short, "comparisons": {}, "answers": []}
    touched = set()
    for label, a, b, role in SMALL:
        if dirs[a] is None or dirs[b] is None:
            rec.check(f"quality {short} 44+34 {label}", "FAIL", "export missing")
            continue
        upstream = a == "fp32"
        cmp = w8.compare(rec, ctx, f"{short} {label}", dirs[a], dirs[b], f"{short}-small-{label}")
        if cmp is None:
            continue
        diffs, problems, compared = w8.decision_differences(cmp, dirs[a], dirs[b], upstream=upstream)
        tokens[label] = w8.token_differences(cmp)
        fields = changed_fields(cmp, upstream)
        touched.update((f["case"], f["question"]) for f in fields)
        raw["comparisons"][label] = {"changed_fields": fields}
        res[label] = {"role": role, "decision_differences": diffs, "comparison_problems": problems,
                      "endpoint_questions_compared": compared, "token_differences": tokens[label],
                      "changed_fields": [{k: f[k] for k in ("case", "question", "fields")} | {"spans": spans_only(
                          f["diff"])} for f in fields], "continuous": continuous(cmp, upstream)}
        rec.expect(f"quality {short} 44+34 {label}: every case, question and distribution compared", not problems,
                   f"{compared} endpoint questions" + (f"; {'; '.join(problems[:8])}" if problems else ""))
        detail = (f"{compared} endpoint questions, {len(diffs)} decision-level differences"
                  f"{': ' + '; '.join(diffs[:8]) if diffs else ''}")
        if role == "criterion":
            judged(rec, f"quality {short} 44+34: q4_k decisions equal to FP32 upstream (criterion; else M2)", diffs,
                   problems, detail, "FAIL")
        elif role == "control":
            judged(rec, f"quality {short} 44+34: f16 control decisions equal to FP32 upstream", diffs, problems,
                   detail, "SIGNAL")
        elif role == "diagnostic":
            judged(rec, f"quality {short} 44+34: q4_k decisions equal to {a} of the session (diagnostic)", diffs,
                   problems, detail, "SIGNAL")
            rec.expect(f"quality {short} 44+34: {a} and q4_k prompts and tokens identical", not tokens[label],
                       f"{tokens[label][:5]}")
    if "fp32-vs-f16" in tokens and "fp32-vs-q4" in tokens:
        rec.expect(f"quality {short} 44+34: token differences to the FP32 export the same for f16 and q4_k "
                   "(known export-format items)", tokens["fp32-vs-f16"] == tokens["fp32-vs-q4"],
                   f"{len(tokens['fp32-vs-f16'])} items")
    # the answers of every touched question in every source (evidence spans of FP32, f16, q8_0, q4_k)
    cases = {k: {c["id"]: c for c in w8.read_jsonl(d / "systemone.jsonl")} for k, d in dirs.items() if d is not None}
    for cid, question in sorted(touched, key=str):
        raw["answers"].append({"case": cid, "question": question, "answers": {
            k: question_answer(cases[k].get(cid, {}), "upstream_answer" if k == "fp32" else "answer_upstream",
                               question) for k in cases}})
    mlx = ctx.root / w8.MLX_Q8.format(short)
    if (mlx / "bench.jsonl").is_file():
        cmp = w8.compare(rec, ctx, f"{short} fp32-vs-mlx-q8 (context)", dirs["fp32"], mlx,
                         f"{short}-small-fp32-vs-mlx-q8")
        if cmp is not None:
            diffs, problems, compared = w8.decision_differences(cmp, dirs["fp32"], mlx, True, b_reference=True)
            res["fp32-vs-mlx-q8-context"] = {"decision_differences": diffs, "comparison_problems": problems,
                                             "endpoint_questions_compared": compared, **w8.numeric_summary(cmp)}
    else:
        rec.check(f"quality {short}: upstream MLX 8-bit context", "GAP", "reference not synced")
    mg.write_json(ctx.out / f"quality-small-{short}.json", raw)


# ---------------------------------------------------------------- quality: 900 questions


EXPORTS_900 = (  # (key, binary, export arguments)
    ("f32", "candidate", {"dtype": "f32", "mode": None, "table": False}),  # the flags of the stored basal_rs_f32
    ("f16", "candidate", {"mode": "tree"}), ("q8", "candidate", {"mode": "tree", "quant": Q8}),
    ("q4", "candidate", {"mode": "tree", "quant": Q4}),
    ("base_f16", "base", {"mode": "tree"}), ("base_q8", "base", {"mode": "tree", "quant": Q8}))


def name_900(short, key):
    return {"f32": f"{short}-900-f32", "f16": f"{short}-900-f16-tree", "q8": f"{short}-900-q8-tree",
            "q4": f"{short}-900-q4-tree", "base_f16": f"{short}-900-f16-tree-base",
            "base_q8": f"{short}-900-q8-tree-base"}[key]


def exports_900(rec, ctx, short, keys):
    """The 900 exports of `keys` ({key: records by index or None}) and the base/candidate f16 and q8_0 byte checks
    when both of a pair are among them."""
    inputs, _ = w8.build_900(ctx.root, short)
    runs = {}
    for key, binary, kw in EXPORTS_900:
        if key not in keys:
            continue
        name = name_900(short, key)
        out = export(rec, ctx, name, short, binary, ctx.inputs900[short], **kw)
        runs[key] = w8.load_900(rec, f"export {name}", out, inputs) if out else None
    for key in ("f16", "q8"):
        if key in keys and f"base_{key}" in keys:
            a, b = ctx.exports.get(name_900(short, f"base_{key}")), ctx.exports.get(name_900(short, key))
            if a is None or b is None:
                rec.check(f"900 {short} {key}: candidate byte-equal to the base", "FAIL", "export missing")
            else:
                w8.same_bytes(rec, f"900 {short} {key}: candidate vs base", a, b)
    return runs


def top_two(p):
    """Indices of the largest and second-largest value (first of equals, as argmax)."""
    order = sorted(range(len(p)), key=lambda i: (-p[i], i))
    return order[0], (order[1] if len(order) > 1 else None)


def margin_shift(ids, fp32, f16, x, changes):
    """The shift of the fixed FP32 top-two gap of x against f16 of the session (W8 R2 metric): for each question a,
    b = FP32's two largest indices; shift = (x[a]-x[b]) - (f16[a]-f16[b]). Context: the questions where f16 and x
    both decide as FP32 (none removed from any count). Descriptive only, not a threshold or causal proof."""
    shift, tvs, margins = {}, {}, {}
    for i in ids:
        a, b = top_two(fp32[i])
        tvs[i], margins[i] = w8.tv(f16[i], x[i]), w8.margin(fp32[i])
        if b is not None:
            shift[i] = (x[i][a] - x[i][b]) - (f16[i][a] - f16[i][b])
    context = [i for i in ids if i in shift and w8.argmax(f16[i]) == w8.argmax(fp32[i]) == w8.argmax(x[i])]
    abs_ctx = [abs(shift[i]) for i in context]
    rows = []
    for i in changes:
        a, b = top_two(fp32[i])
        m = margins[i]
        rows.append({
            "id": i, "fp32_winner_index": a, "fp32_runner_up_index": b, "fp32_margin": m,
            "f16_top_two_gap": None if b is None else f16[i][a] - f16[i][b],
            "x_top_two_gap": None if b is None else x[i][a] - x[i][b], "signed_gap_shift_vs_f16": shift.get(i),
            "tv_vs_f16": tvs[i], "f16_decision_equal_fp32": w8.argmax(f16[i]) == a,
            "x_decision_equal_fp32": w8.argmax(x[i]) == a, "x_winner_is_fp32_runner_up": w8.argmax(x[i]) == b,
            "tv_descending_competition_rank": 1 + sum(tvs[j] > tvs[i] for j in ids),
            "fp32_margin_ascending_competition_rank": None if m is None else
            1 + sum(margins[j] is not None and margins[j] < m for j in ids),
            "context_abs_gap_shift_at_least_this_fp32_margin": None if m is None else sum(v >= m for v in abs_ctx)})
    return {"all_questions": len(ids), "context_questions": len(context),
            "context_abs_signed_gap_shift": w8.dist(abs_ctx), "all_signed_gap_shift": w8.dist(list(shift.values())),
            "changes": rows}


def analyse_900(rec, ctx, short, runs, inputs, decisions):
    """q4_k of the session against the stored upstream FP32 (attributable only with a passing f32 control), the f16
    and q8_0 of the session; per question only ids, distributions, decisions and labels are written."""
    p = {k: {by[i]["id"]: by[i]["p_cal"] for i in by} for k, by in runs.items() if by}
    ids = [r["id"] for r in inputs]
    fp32 = {i: decisions[i]["upstream_fp32"] for i in ids}
    stored_f16 = {i: decisions[i]["basal_rs_f16"] for i in ids}
    stored_f32 = {i: decisions[i]["basal_rs_f32"] for i in ids}
    gold = {i: decisions[i].get("gold") for i in ids}

    def flips(x, y):
        return [i for i in ids if w8.argmax(x[i]) != w8.argmax(y[i])]

    def described(diff_ids, x, y):
        return [{"id": i, "fp32_margin": w8.margin(fp32[i]), "a": w8.argmax(x[i]), "b": w8.argmax(y[i]),
                 "tv": w8.tv(x[i], y[i]), "dataset": decisions[i].get("dataset"), "type": decisions[i].get("type")}
                for i in diff_ids]

    def deltas(x, y):
        return {"max_abs_dp": w8.dist([max(abs(a - b) for a, b in zip(x[i], y[i])) for i in ids]),
                "tv": w8.dist([w8.tv(x[i], y[i]) for i in ids])}

    def accuracy(x):
        def acc(subset):
            known = [i for i in subset if isinstance(gold[i], int)]
            return {"correct": sum(w8.argmax(x[i]) == gold[i] for i in known), "with_gold": len(known)}

        groups = {}
        for key in ("dataset", "type"):
            for v in sorted({str(decisions[i].get(key)) for i in ids}):
                groups.setdefault(key, {})[v] = acc([i for i in ids if str(decisions[i].get(key)) == v])
        return {"total": acc(ids), "per_dataset": groups.get("dataset"), "per_type": groups.get("type")}

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
    f16_wrong, q8_wrong, q4_wrong = set(flips(fp32, p["f16"])), set(flips(fp32, p["q8"])), set(flips(fp32, p["q4"]))
    new = sorted(q4_wrong - f16_wrong, key=ids.index)
    fixed = sorted(f16_wrong - q4_wrong, key=ids.index)
    shared = sorted(q4_wrong & f16_wrong, key=ids.index)
    res["f16_session"] = {"vs_stored_upstream_fp32": described(sorted(f16_wrong, key=ids.index), fp32, p["f16"]),
                          "vs_stored_basal_rs_f16": described(flips(stored_f16, p["f16"]), stored_f16, p["f16"]),
                          "deltas_vs_fp32": deltas(fp32, p["f16"])}
    res["q8_session_context"] = {"vs_stored_upstream_fp32": described(sorted(q8_wrong, key=ids.index), fp32, p["q8"]),
                                 "vs_f16_session": described(flips(p["f16"], p["q8"]), p["f16"], p["q8"]),
                                 "deltas_vs_fp32": deltas(fp32, p["q8"])}
    res["q4"] = {
        "attributable_to_stored_fp32": control,
        "vs_fp32": {"new": described(new, fp32, p["q4"]), "fixed": described(fixed, fp32, p["q4"]),
                    "shared": [dict(d, same_as_f16=w8.argmax(p["q4"][d["id"]]) == w8.argmax(p["f16"][d["id"]]))
                               for d in described(shared, fp32, p["q4"])]},
        "vs_f16_session": described(flips(p["f16"], p["q4"]), p["f16"], p["q4"]),
        "vs_q8_session": described(flips(p["q8"], p["q4"]), p["q8"], p["q4"]),
        "deltas_vs_fp32": deltas(fp32, p["q4"]), "deltas_vs_f16_session": deltas(p["f16"], p["q4"]),
        "deltas_vs_q8_session": deltas(p["q8"], p["q4"]),
        "margin_shift_vs_f16_session": margin_shift(
            ids, fp32, p["f16"], p["q4"], sorted(q4_wrong | f16_wrong | set(flips(p["f16"], p["q4"])), key=ids.index))}
    groups = {}
    for i in ids:
        for key in (f"dataset:{decisions[i].get('dataset')}", f"type:{decisions[i].get('type')}"):
            g = groups.setdefault(key, {"n": 0, "f16_flips_vs_fp32": 0, "q8_flips_vs_fp32": 0, "q4_flips_vs_fp32": 0,
                                        "q4_flips_vs_f16": 0, "q4_flips_vs_q8": 0})
            g["n"] += 1
            g["f16_flips_vs_fp32"] += i in f16_wrong
            g["q8_flips_vs_fp32"] += i in q8_wrong
            g["q4_flips_vs_fp32"] += i in q4_wrong
            g["q4_flips_vs_f16"] += w8.argmax(p["q4"][i]) != w8.argmax(p["f16"][i])
            g["q4_flips_vs_q8"] += w8.argmax(p["q4"][i]) != w8.argmax(p["q8"][i])
    res["groups"] = groups
    res["accuracy_vs_gold"] = {"upstream_fp32_stored": accuracy(fp32), "basal_rs_f16_stored": accuracy(stored_f16),
                               "f32_session": accuracy(p["f32"]), "f16_session": accuracy(p["f16"]),
                               "q8_session": accuracy(p["q8"]), "q4_session": accuracy(p["q4"])}
    res["known_close_ties_of_this_model_and_card"] = next(
        (v for (s, card), v in w8.KNOWN_TIES.items() if s == short and card in ctx.gpu_name), [])
    res["distributions"] = [{"id": i, "dataset": decisions[i].get("dataset"), "type": decisions[i].get("type"),
                             "gold": gold[i], "upstream_fp32": fp32[i], "basal_rs_f16_stored": stored_f16[i],
                             "f32_session": p["f32"][i], "f16_session": p["f16"][i], "q8_session": p["q8"][i],
                             "q4_session": p["q4"][i]} for i in ids]
    mg.write_json(ctx.out / f"quality-900-{short}.json", res)
    acc = res["accuracy_vs_gold"]
    lower = acc["q4_session"]["total"]["correct"] < acc["f16_session"]["total"]["correct"]
    detail = (f"q4_k vs FP32: new {len(new)}, fixed {len(fixed)}, shared {len(shared)}"
              f"{'' if control else ' (not attributable: f32 control GAP)'}; vs f16 session "
              f"{len(res['q4']['vs_f16_session'])}, vs q8_0 session {len(res['q4']['vs_q8_session'])}; accuracy "
              f"q4_k {acc['q4_session']['total']['correct']} / f16 {acc['f16_session']['total']['correct']} of "
              f"{acc['f16_session']['total']['with_gold']}")
    concern = bool(new) if control else bool(res["q4"]["vs_f16_session"])
    rec.check(f"quality {short} 900: q4_k decisions (diagnostic, not a zero-flip gate)",
              "SIGNAL" if (concern or lower) else "PASS",
              detail + ("; new flips or a lower accuracy: defect analysis, then M2 (an explanation does not close it)"
                        if (concern or lower) else ""))
    ctx.results.setdefault("quality_900", {})[short] = {k: v for k, v in res.items() if k != "distributions"}


def quality_900(rec, ctx, short):
    inputs, decisions = w8.build_900(ctx.root, short)
    runs = exports_900(rec, ctx, short, [k for k, _, _ in EXPORTS_900])
    missing = [k for k in ("f32", "f16", "q8", "q4") if not runs.get(k)]
    for label, a, b in (("f32-vs-f16", "f32", "f16"), ("f32-vs-q8", "f32", "q8"), ("f32-vs-q4", "f32", "q4"),
                        ("f16-vs-q4", "f16", "q4"), ("q8-vs-q4", "q8", "q4")):
        if a in missing or b in missing:
            continue
        cmp = w8.compare(rec, ctx, f"{short} 900 {label}", ctx.exports[name_900(short, a)],
                         ctx.exports[name_900(short, b)], f"{short}-900-{label}")
        if cmp is None:
            continue
        rec.expect(f"basal compare {short} 900 {label}: nothing missing, 900 compared",
                   not cmp.get("missing") and cmp.get("bench", {}).get("items") == 900,
                   f"missing {len(cmp.get('missing', []))}, compared {cmp.get('bench', {}).get('items')}")
        ctx.results.setdefault("quality_900_compare", {}).setdefault(short, {})[label] = w8.numeric_summary(cmp)
    if missing:
        rec.check(f"quality {short} 900: analysis", "FAIL", f"exports missing or incomplete: {missing}")
        return
    analyse_900(rec, ctx, short, runs, inputs, decisions)


def campaign_quality(rec, ctx):
    for short in ctx.shorts:
        quality_small(rec, ctx, short)
    for short in ctx.shorts:
        quality_900(rec, ctx, short)


# ---------------------------------------------------------------- profile and memory


def campaign_profile(rec, ctx):
    short = ctx.args.perf_model
    total, pure = w8.projection_values(ctx.model[short])
    res = ctx.results.setdefault("profile", {
        "model": short, "projection_values_per_forward": total,
        "scope": "synchronised profile: the dequantization time is part of the projection sections, not added to "
                 "sections_sum_ms; effective bytes per second are arithmetic (not physical bandwidth, not request "
                 "latency)"})
    for batch in (1, ctx.args.profile_batch):
        w8.guard(ctx, rec, f"profile batch {batch}")
        row = res[f"batch_{batch}"] = {}
        for key, quant in (("f16", None), ("q8", Q8), ("q4", Q4)):
            row[key] = w8.profile_run(rec, ctx, "profile", "candidate", short, quant, batch)
        f16 = row["f16"]
        for key, quant in (("q8", Q8), ("q4", Q4)):
            out = row[key]
            if not out:
                rec.check(f"profile batch {batch} {quant}: dequantization keys", "GAP", "no profile output")
                continue
            own, other = DEQUANT_KEYS[quant], DEQUANT_KEYS[Q4 if quant == Q8 else Q8]
            ok = all(k in out for k in own) and not any(k in out for k in other) and \
                (quant == Q8 or out.get("dequant_format") == Q4)
            rec.expect(f"profile batch {batch} {quant}: dequantization keys of the format only", ok,
                       f"keys {sorted(k for k in out if 'dequant' in k)}")
            ms, values = out.get(own[0]), out.get(own[1])
            if not (f16 and w8.finite(ms) and w8.finite(values) and ms > 0):
                rec.check(f"profile batch {batch} {quant}: derived arithmetic", "GAP", "a profile or counter missing")
                continue
            forwards = values / total
            gemm_ms = sum(f16["sections_ms_per_item"].get(s, 0.0) for s in ("qkv_proj", "gate_up_proj"))
            row[f"derived_{key}"] = {
                "forwards_per_item": forwards,
                "dequant_effective_bytes_per_s": values * DEQUANT_BYTES_PER_VALUE[quant] / (ms / 1e3),
                "bytes_per_value": DEQUANT_BYTES_PER_VALUE[quant],
                "f16_gemm_weight_read_effective_bytes_per_s":
                    forwards * pure * 2 / (gemm_ms / 1e3) if gemm_ms else None,
                "section_delta_ms_vs_f16": {s: out["sections_ms_per_item"].get(s, 0.0) - f16["sections_ms_per_item"]
                                            .get(s, 0.0) for s in f16["sections_ms_per_item"]},
                "note": "arithmetic: blocks read plus 2 B of f16 written per value over the synchronised "
                        "dequantization time; not a physical bandwidth or request latency"}
    if ctx.args.profile_batch == 1:
        rec.check("profile: --profile-batch 1 repeats batch 1", "GAP", "the plan asks for batch 1 and 16")
    rec.check("profile: nsys timeline", "GAP", "not planned")


def campaign_memory_mini(rec, ctx):
    res = ctx.results.setdefault("memory_mini", {
        "load_s_scope": "basal benchmark: load without the CUDA context creation (the pool probe creates it first)",
        "host_rss_peak": "not measured", "nvml": "device samples every 200 ms, not a peak"})
    specs = [(v, lambda out, q=q: w8.benchmark_argv(ctx, "candidate", "mini", q, out))
             for v, q in (("f16", None), (Q8, Q8), (Q4, Q4))]
    runs = []
    try:
        w8.benchmark_runs(rec, ctx, "memory-mini", specs, runs)
    finally:
        w8.memory_rows(rec, res, runs)
        memory_workloads(rec, res, runs)
        for r in runs:
            if r["variant"] == Q4 and r["data"]:
                q4_load_checks(rec, ctx, "memory-mini q4_k benchmark", "mini", r["data"].get("backend"))


def memory_workloads(rec, res, runs):
    """Every run: 900 sequential and 846 mixed requests at c1 and c16, the same 54 mixed requests skipped over the
    context limit (ids and reasons kept), the same in every run."""
    skipped = {}
    for r in runs:
        d = r["data"]
        if not d:
            continue
        w = d.get("workloads") or {}
        planned = {k: p.get("planned_requests") for k, p in w8.phases(d).items()}
        skips = (w.get("mixed") or {}).get("skipped") or []
        reasons = all(isinstance(s.get("detail"), list) and s["detail"] and
                      all(x.get("type") == "context_length_exceeded" for x in s["detail"]) for s in skips)
        ids = sorted(str(s.get("id")) for s in skips)
        skipped[r["variant"]] = ids
        res.setdefault(r["variant"], {})["skipped_mixed"] = [{"id": s.get("id"), "class": s.get("class"),
                                                             "reasons": sorted({x.get("type") for x in s["detail"]})}
                                                            for s in skips if isinstance(s.get("detail"), list)]
        rec.expect(f"memory-mini {r['variant']}: 900 sequential, 846 mixed at c1 and c16, {MEMORY_MINI_SKIPPED} mixed "
                   "requests skipped over the context limit", planned == MEMORY_MINI_REQUESTS and
                   (w.get("sequential") or {}).get("skipped_requests") == 0 and
                   (w.get("mixed") or {}).get("skipped_requests") == MEMORY_MINI_SKIPPED == len(set(ids)) and reasons,
                   f"planned {planned}, skipped {len(ids)} (context_length_exceeded only: {reasons})")
    if len(skipped) > 1:
        rec.expect("memory-mini: the same skipped mixed requests in every run",
                   len({json.dumps(v) for v in skipped.values()}) == 1, f"{ {k: len(v) for k, v in skipped.items()} }")


# ---------------------------------------------------------------- performance families


def benchmark_pair(rec, label, name, a, b, strict):
    """One predeclared pair of `basal benchmark` runs (role a, role b; w8.benchmark_pairs for a fixed pair): every
    phase metric, defects as FAIL, the reasons the pair is not complete evidence."""
    pa, pb = w8.phases(a["data"]), w8.phases(b["data"])
    metrics, problems = {}, []
    for side, ps, run in (("a", pa, a), ("b", pb, b)):
        if run["data"] is None or run["data"].get("complete") is not True:
            problems.append(f"{side} run {run['run']}: no complete report")
        if set(ps) != set(w8.PHASES):
            problems.append(f"{side} phases {sorted(ps)}, expected {list(w8.PHASES)}")
    for ph in sorted(set(pa) | set(pb) | set(w8.PHASES)):
        for (m, q), d in w8.PHASE_METRICS:
            va, vb = pa.get(ph, {}).get(m), pb.get(ph, {}).get(m)
            if q:
                va, vb = (va or {}).get(q), (vb or {}).get(q)
            metrics[f"{ph} {m}{'.' + q if q else ''}"] = w8.pair(va, vb, d)
        if ph not in pa or ph not in pb:
            continue
        defects = {f: (pa[ph].get(f), pb[ph].get(f))
                   for f in ("errors", "complete", "answer_changes_vs_first_success", "planned_requests")}
        good = all(u == v for u, v in defects.values()) and defects["complete"] == (True, True) and \
            (not strict or defects["answer_changes_vs_first_success"] == (0, 0))
        rec.expect(f"perf {label} pair {name} {ph}: errors, completeness and answer changes agree", good,
                   json.dumps(defects))
        if not good:
            problems.append(f"{ph}: {json.dumps(defects)}")
    return {"pair": name, "runs": (a["run"], b["run"]), "roles": (a["variant"], b["variant"]), "metrics": metrics,
            "problems": problems,
            "load_s": tuple((r["data"] or {}).get("model", {}).get("load_s") for r in (a, b)),
            "memory_pool": ((a["data"] or {}).get("memory_pool"), (b["data"] or {}).get("memory_pool")),
            "conditions": (a["conditions"], b["conditions"]), "process": (a["process"], b["process"])}


def fixed_pairs(rec, label, runs, spec, strict):
    """The predeclared pairs `spec` [(name, a index, b index)] of the runs present (no re-pairing)."""
    by = {r["run"]: r for r in runs}
    out = []
    for name, i, j in spec:
        if i in by and j in by:
            out.append(benchmark_pair(rec, label, name, by[i], by[j], strict))
    return out


def family_argv(ctx, fam, role, kind, out=None):
    binary, quant = FAMILIES[fam][role]
    short = ctx.args.perf_model
    if kind == "bench":
        return w8.bench_argv(ctx, binary, short, quant)
    return w8.benchmark_argv(ctx, binary, short, quant, out)


def q4_bench_checks(rec, ctx, label, d, files):
    for f in files:
        path = d / f
        try:
            data = read_json(path) if path.is_file() else None
        except ValueError:
            data = None
        if data is not None:
            q4_load_checks(rec, ctx, f"perf {label} {f}", ctx.args.perf_model, data.get("backend"))


def campaign_family(rec, ctx, fam):
    """S2 (none) or S3 (q8_0): ab.py A/B and B/C, six benchmark runs A B C C B A; the W8 signal rule on the
    regression pairs, the trade-off pairs descriptive."""
    short = ctx.args.perf_model
    roles = {r: {"binary": b, "quantization": q or "none"} for r, (b, q) in FAMILIES[fam].items()}
    res = ctx.results.setdefault(f"perf_{fam}_family", {
        "model": short, "roles": roles, "benchmark_order": " ".join(FAMILY_ORDER),
        "regression_pairs": {"benchmark": [n for n, _, _ in REGRESSION_PAIRS], "bench": "the 4 ab.py pairs A/B"},
        "tradeoff_pairs": {"benchmark": [n for n, _, _ in TRADEOFF_PAIRS], "bench": "the 4 ab.py pairs B/C"},
        "shared_runs": "benchmark runs B1 and B4 belong to a regression and a trade-off pair (declared in advance)",
        "rule": "regression: a metric with B worse in every pair is a signal -> one confirmation round of the "
                "affected command and profiles A/B at batch 1; trade-off: descriptive, no speed gate"})
    rounds = res.setdefault("rounds", {"bench_ab": [], "bench_bc": [], "benchmark": [], "confirmation_bench": [],
                                       "confirmation_benchmark": []})
    res["conclusion"] = None
    bench = {r: family_argv(ctx, fam, r, "bench") for r in "ABC"}
    w8.guard(ctx, rec, f"perf {fam} family bench A/B")
    w8.bench_round(rec, ctx, f"{fam}-family-bench-ab", bench["A"], bench["B"], True, short, rounds["bench_ab"])
    w8.guard(ctx, rec, f"perf {fam} family bench B/C")
    try:
        w8.bench_round(rec, ctx, f"{fam}-family-bench-bc", bench["B"], bench["C"], False, short, rounds["bench_bc"])
    finally:
        q4_bench_checks(rec, ctx, f"{fam}-family-bench-bc", ctx.out / "perf" / f"{fam}-family-bench-bc",
                        [f"b-{k}.json" for k in range(4)])
    label = f"{fam}-family-benchmark"
    specs = [(f"{role}", lambda out, role=role: family_argv(ctx, fam, role, "benchmark", out)) for role in FAMILY_ORDER]
    runs = []
    entry = {"round": label, "runs": [], "regression_pairs": [], "tradeoff_pairs": [], "problems": {}}
    rounds["benchmark"].append(entry)
    try:
        w8.benchmark_runs(rec, ctx, label, specs, runs)
    finally:
        sources = {json.dumps(r["data"].get("source"), sort_keys=True) for r in runs if r["data"]}
        if sources:
            rec.expect(f"perf {label}: one embedded workload version in every run", len(sources) == 1,
                       f"{len(sources)} sources")
        entry["runs"] = [{k: v for k, v in r.items() if k != "data"} for r in runs]
        entry["regression_pairs"] = fixed_pairs(rec, label, runs, REGRESSION_PAIRS, True)
        entry["tradeoff_pairs"] = fixed_pairs(rec, label, runs, TRADEOFF_PAIRS, False)
        reported = sum(1 for r in runs if r["data"])
        missing = [] if reported == len(specs) else [f"{reported} of {len(specs)} runs reported"]
        entry["problems"] = {
            "regression": missing + w8.evidence_problems(entry["regression_pairs"], 2, w8.PHASE_REQUIRED),
            "tradeoff": missing + w8.evidence_problems(entry["tradeoff_pairs"], 2, w8.PHASE_REQUIRED)}
        for r in runs:
            if r["variant"] == "C" and r["data"]:
                q4_load_checks(rec, ctx, f"perf {label} run {r['run']} C", short, r["data"].get("backend"))
    family_conclusion(rec, ctx, fam, res, rounds, bench)


def family_conclusion(rec, ctx, fam, res, rounds, bench):
    short = ctx.args.perf_model
    primary = {"bench": rounds["bench_ab"][0]["pairs"], "benchmark": rounds["benchmark"][0]["regression_pairs"]}
    scheduled = {"bench": 4, "benchmark": 2}
    sig, open_ = {}, {}
    for k in primary:
        sig[k], open_[k] = w8.signals(primary[k], scheduled[k])
    res["primary_signals"], res["primary_open"] = sig, open_
    confirm = {k: bool(sig[k] or open_[k]) for k in sig}
    rec.check(f"perf {fam} family: primary regression signals (diagnostic; the confirmation round decides)", "PASS",
              f"B worse in every pair: {sig}; B worse in every pair with a value, incomplete: {open_}; one "
              "confirmation round of the affected command and profiles A/B follow" if any(confirm.values()) else
              "no metric with B worse in every pair; no confirmation round")
    if any(confirm.values()):
        res["profiles"] = {}
        if confirm["bench"]:
            w8.guard(ctx, rec, f"perf {fam} family confirmation bench")
            w8.bench_round(rec, ctx, f"{fam}-family-bench-ab-confirmation", bench["A"], bench["B"], True, short,
                           rounds["confirmation_bench"])
        if confirm["benchmark"]:
            specs = [(v, lambda out, v=v: family_argv(ctx, fam, v, "benchmark", out)) for v in "ABBA"]
            w8.benchmark_round(rec, ctx, f"{fam}-family-benchmark-confirmation", specs, True,
                               rounds["confirmation_benchmark"])
        for v in "AB":
            binary, quant = FAMILIES[fam][v]
            res["profiles"][v] = w8.profile_run(rec, ctx, f"{fam}-confirm", binary, short, quant, 1)
    problems = [f"bench {r['round']}: {x}" for r in rounds["bench_ab"] + rounds["confirmation_bench"]
                for x in r["problems"]]
    problems += [f"benchmark {r['round']}: {x}" for r in rounds["benchmark"] for x in r["problems"]["regression"]]
    problems += [f"benchmark {r['round']}: {x}" for r in rounds["confirmation_benchmark"] for x in r["problems"]]
    if any(confirm.values()):
        problems += [f"profile {v}: missing" for v in "AB" if not res["profiles"].get(v)]
    second = {"bench": (rounds["confirmation_bench"] or [{}])[0].get("pairs", []),
              "benchmark": (rounds["confirmation_benchmark"] or [{}])[0].get("pairs", [])}
    persistent, unresolved = [], []
    for k in primary:
        if not confirm[k]:
            continue
        both = primary[k] + second[k]
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
        rec.check(f"perf {fam} family: persistent regression signal (B worse in every pair of both rounds)", "SIGNAL",
                  f"{persistent}: analyse pairs, per-question consistency, conditions, profiles and the changed path; "
                  "no cause in the changed code -> user decision M2 (conclusion c); a fix -> revalidation")
    if problems or unresolved:
        rec.check(f"perf {fam} family: no regression conclusion (evidence incomplete)", "GAP",
                  f"{len(problems)} problems: {'; '.join(problems[:6])}; not resolved: {unresolved}; every value is "
                  "in summary.json")
    elif persistent:
        res["conclusion"] = "persistent signal: (b) or (c) after the parent's analysis"
    else:
        res["conclusion"] = "a"
        rec.check(f"perf {fam} family: regression conclusion (a)", "PASS", "no repeatable slowdown shown at the "
                  "measured resolution (not a claim of no regression); every pair and value is in summary.json")
    tradeoff = [f"bench {r['round']}: {x}" for r in rounds["bench_bc"] for x in r["problems"]]
    tradeoff += [f"benchmark {r['round']}: {x}" for r in rounds["benchmark"] for x in r["problems"]["tradeoff"]]
    rec.check(f"perf {fam} family: every trade-off pair with all metrics (descriptive, no speed gate)",
              "GAP" if tradeoff else "PASS", "; ".join(tradeoff[:6]) if tradeoff else "complete")


# ---------------------------------------------------------------- revalidation after a code fix


def previous_session(args, out, base_sha, cand_sha, cur_sources, models, tables, shorts, root):
    """The pre-fix W4 session, checked before --out is created: a distinct raw directory of this runner whose run.json
    and summary.json carry one fingerprint with this run's base, the candidate --expect-previous-candidate-sha256
    (other than this run's), other synced sources (a code fix), the model revisions, GEMM tables and inputs of this
    run, and per model a completed q4_k tree export on GPU 0 (command, checks, manifest). Returns its provenance;
    raises ValueError."""
    prev, cur = args.previous_run.resolve(), out.resolve()
    if not prev.is_dir():
        raise ValueError("--previous-run is not an existing directory")
    if prev == cur or prev in cur.parents or cur in prev.parents:
        raise ValueError("--previous-run must be another session's directory, neither --out nor inside it or around it")
    try:
        run, summary = read_json(prev / "run.json"), read_json(prev / "summary.json")
        commands = w8.read_jsonl(prev / "commands.jsonl")
    except (OSError, ValueError) as e:
        raise ValueError(f"--previous-run: run.json, summary.json and commands.jsonl needed ({type(e).__name__})")
    fp = run.get("fingerprint")
    if not isinstance(fp, dict) or summary.get("fingerprint") != fp or fp.get("runner") != RUNNER_ID:
        raise ValueError("--previous-run: not a raw directory of this runner, or run.json and summary.json disagree")
    bins = fp.get("binaries") or {}
    p_base, p_cand = (bins.get("base") or {}).get("sha256"), (bins.get("candidate") or {}).get("sha256")
    if p_base != base_sha:
        raise ValueError(f"--previous-run: base SHA-256 {p_base}, this run's base {base_sha}")
    if p_cand != args.expect_previous_candidate_sha256.lower():
        raise ValueError(f"--previous-run: candidate SHA-256 {p_cand}, not --expect-previous-candidate-sha256")
    if p_cand in (cand_sha, base_sha):
        raise ValueError("--previous-run: its candidate is this run's candidate or the base (no pre-fix candidate)")
    p_src = (fp.get("sources") or {}).get("sha256")
    if not p_src or p_src == cur_sources["sha256"]:
        raise ValueError("--previous-run: the same (or unrecorded) synced sources: revalidation follows a code fix "
                         "only")
    inputs = fp.get("inputs") or {}
    if (inputs.get("sequential") or {}).get("sha256") != w8.SEQUENTIAL_SHA256:
        raise ValueError("--previous-run: other 900-question data")
    provenance = {"candidate_sha256": p_cand, "base_sha256": p_base, "sources": fp.get("sources"), "git": fp.get("git"),
                  "declared_labels": {k: (bins.get(k) or {}).get("declared_label") for k in ("base", "candidate")},
                  "runner_sha256": fp.get("runner_sha256"), "run_json_sha256": mg.sha256(prev / "run.json"),
                  "summary_sha256": mg.sha256(prev / "summary.json"), "previous_complete": summary.get("complete"),
                  "previous_exit_status": summary.get("exit_status"), "models": {}, "exports": {},
                  "export_other_checks": {}}
    checks = summary.get("checks") or []
    for short in shorts:
        pm = (fp.get("models") or {}).get(short) or {}
        if pm.get("revision") != models[short]["dir"].name or pm.get("name") != models[short]["name"]:
            raise ValueError(f"--previous-run: model {short} {pm.get('name')} {pm.get('revision')}")
        if ((fp.get("gemm_tables") or {}).get(short) or {}).get("sha256") != mg.sha256(tables[short]):
            raise ValueError(f"--previous-run: another GEMM table for {short}")
        if (inputs.get("references_fp32") or {}).get(short) != w8.tree_sha(root / w8.REFERENCE.format(short)):
            raise ValueError(f"--previous-run: another FP32 reference for {short}")
        if (inputs.get("decisions") or {}).get(short) != mg.sha256(root / w8.DECISIONS.format(short)):
            raise ValueError(f"--previous-run: another decisions file for {short}")
        name = f"{short}-tree-q4-dev0"
        d = prev / "exports" / name
        if not all((d / f).is_file() for f in w8.EXPORT_FILES):
            raise ValueError(f"--previous-run: export {name} missing")
        manifest = read_json(d / "manifest.json")
        run_m = manifest.get("run") or {}
        if (run_m.get("quantization"), run_m.get("dtype"), run_m.get("prefix_cache"),
                manifest.get("bench_batching")) != (Q4, "f16", True, "Tree"):
            raise ValueError(f"--previous-run: export {name} is not a q4_k f16 tree export with the prefix cache")
        runs_of = [c for c in commands if c.get("step") == f"export-{name}"]
        argv = runs_of[-1].get("argv", []) if runs_of else []
        flags = list(zip(argv, argv[1:]))
        if len(runs_of) != 1 or runs_of[0].get("exit") != 0 or runs_of[0].get("stopped") or runs_of[0].get("env") or \
                ("--quantization", Q4) not in flags or ("--batching", "tree") not in flags or \
                "--no-prefix-cache" in argv or "--device" in argv:
            raise ValueError(f"--previous-run: no single completed q4_k tree command on GPU 0 for export {name}")
        # the export itself must have passed; its other checks (the q4_k load description, P0 equality, placement)
        # may be the very defect the fix addresses: kept as context, not a refusal
        mine = [c for c in checks if c.get("check", "").startswith(f"export {name}:")]
        wanted = (f"export {name}: completed", f"export {name}: every input exported",
                  f"export {name}: manifest quantization {Q4}")
        if not all(any(c.get("check") == w and c.get("status") == "PASS" for c in mine) and
                   not any(c.get("check") == w and c.get("status") == "FAIL" for c in mine) for w in wanted):
            raise ValueError(f"--previous-run: the export checks of {name} (completed, every input exported, manifest "
                             "quantization q4_k) did not pass in that session")
        if ((summary.get("exports") or {}).get(name) or {}).get("exit") != 0:
            raise ValueError(f"--previous-run: summary.json has no completed export {name}")
        provenance["models"][short] = {k: pm.get(k) for k in ("name", "revision", "safetensors_sha256")}
        provenance["exports"][short] = {f: mg.sha256(d / f) for f in w8.EXPORT_FILES}
        provenance["export_other_checks"][short] = [
            {k: c.get(k) for k in ("check", "status", "class", "detail")} for c in mine
            if c.get("check") not in wanted and c.get("status") != "PASS"]
    return provenance


def campaign_revalidate(rec, ctx):
    """After a code fix: the none and q8_0 contracts against the base again (exports, servers, 900 bytes), the q4_k
    native HTTP and replicas of the final candidate (once, whatever its exports show: export bytes do not show
    serving, routing or shutdown), and the q4_k tree exports against the pre-fix candidate's; a difference reruns
    the w4 step and the quality campaign."""
    provenance = ctx.previous
    res = ctx.results["revalidate"] = {
        "previous": provenance, "current_candidate_sha256": ctx.fp["binaries"]["candidate"]["sha256"],
        "current_sources": ctx.fp["sources"], "models": {}, "changed": []}
    for step, fn in (("none", w8.step_none), ("q8", step_q8), ("serve", step_serve)):
        if step not in ctx.ran_steps:  # once per run, also when correctness started these steps before
            ctx.ran_steps.append(step)
            fn(rec, ctx)
    for short in ctx.shorts:
        exports_900(rec, ctx, short, ("f16", "q8", "base_f16", "base_q8"))
    changed = res["changed"]
    for short in ctx.shorts:
        name = f"{short}-tree-q4-dev0"
        cur = export(rec, ctx, name, short, "candidate", reference(ctx, short), "tree", quant=Q4)
        old = ctx.args.previous_run / "exports" / name
        other = provenance["export_other_checks"][short]
        row = res["models"][short] = {"previous_export": provenance["exports"][short],
                                      "previous_export_other_checks": other}
        label = f"revalidate {short}: q4_k tree export byte-equal to the previous candidate's"
        context = f"; context: {len(other)} other checks of the previous {name} not PASS " \
                  f"({dict(Counter(c['status'] for c in other))}, listed in summary.json)" if other else ""
        if cur is None:
            rec.check(label, "FAIL", "export missing")
            changed.append(short)
            continue
        row["current_export"] = {f: mg.sha256(cur / f) for f in w8.EXPORT_FILES}
        if {f: mg.sha256(old / f) for f in w8.EXPORT_FILES} != provenance["exports"][short]:
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
            rec.check(label, "SIGNAL", detail + context)
            changed.append(short)
        elif not (w_prev and w_cur):
            rec.check(label, "GAP", f"{detail}, but the weights' SHA-256 is not in both sessions: not shown "
                      f"equal{context}")
            changed.append(short)
        else:
            rec.check(label, "PASS",
                      f"{detail}; previous candidate {provenance['candidate_sha256'][:12]}, same weights{context}")
    if changed:
        rec.check("revalidate: q4_k results changed or not shown equal", "SIGNAL",
                  f"{changed}: the w4 consistency step, 44+34 and 900 run again")
        if "w4" not in ctx.ran_steps:
            ctx.ran_steps.append("w4")
            step_w4(rec, ctx)
        if "quality" not in ctx.args.campaigns:
            campaign_quality(rec, ctx)
    for c in ("profile", "memory-mini"):
        if c not in ctx.args.campaigns:
            rec.check(f"revalidate: {c} of the final candidate", "GAP", "not re-measured: the earlier measurement "
                      "keeps its binary's label; equal outputs do not show equal time or memory")


# ---------------------------------------------------------------- S1 evidence of a family run


def identity_proof(fp, short):
    """What a family run must share with its S1 run, from a fingerprint (None: not recorded there): the runner and
    helpers, synced sources and embedded assets, the family's model (name, revision, config, weights), its GEMM
    table, FP32 reference, decisions and 900-question inputs, the sequential, lock and long-state files, config.rs
    and the P0 facts."""
    m = (fp.get("models") or {}).get(short) or {}
    inputs = fp.get("inputs") or {}
    i900 = ((inputs.get("inputs_900") or {}).get("models") or {}).get(short) or {}
    return {"runner SHA-256": fp.get("runner_sha256"), "helpers SHA-256": fp.get("helpers_sha256"),
            "synced sources": fp.get("sources"), "embedded assets": fp.get("embedded_assets"),
            f"model {short} name": m.get("name"), f"model {short} revision": m.get("revision"),
            f"model {short} config.json": m.get("config_sha256"), f"model {short} weights": m.get("safetensors_sha256"),
            f"GEMM table {short}": ((fp.get("gemm_tables") or {}).get(short) or {}).get("sha256"),
            f"FP32 reference {short}": (inputs.get("references_fp32") or {}).get(short),
            f"decisions {short}": (inputs.get("decisions") or {}).get(short),
            f"900 inputs {short}": i900.get("bench_sha256"),
            "sequential workload": (inputs.get("sequential") or {}).get("sha256"),
            "lock file": inputs.get("lock_sha256"), "long states": inputs.get("long_states_sha256"),
            "config.rs": inputs.get("config_rs_sha256"), "P0 facts": (inputs.get("p0_facts") or {}).get("sha256")}


def s1_evidence(args, out, base_sha, cand_sha, cur_sources):
    """The S1 (or revalidation) raw directory a family run builds on, read before anything runs: this runner's
    raw directory with the same base and candidate SHA-256 and synced sources; no FAIL of a class other than
    quality-criterion (S2/S3 do not run after one). Returns what is recorded, with the identity proof of the family's
    model (identity_proof) that binding_checks compares with this run's before any GPU work; raises ValueError."""
    prev, cur = args.s1_run.resolve(), out.resolve()
    if not prev.is_dir() or prev == cur or prev in cur.parents or cur in prev.parents:
        raise ValueError("--s1-run must be an existing directory of another run, neither --out nor around it")
    try:
        run, summary = read_json(prev / "run.json"), read_json(prev / "summary.json")
    except (OSError, ValueError) as e:
        raise ValueError(f"--s1-run: run.json and summary.json needed ({type(e).__name__})")
    fp = run.get("fingerprint")
    if not isinstance(fp, dict) or summary.get("fingerprint") != fp or fp.get("runner") != RUNNER_ID:
        raise ValueError("--s1-run: not a raw directory of this runner, or run.json and summary.json disagree")
    bins = fp.get("binaries") or {}
    got = ((bins.get("base") or {}).get("sha256"), (bins.get("candidate") or {}).get("sha256"))
    if got != (base_sha, cand_sha):
        raise ValueError(f"--s1-run: binaries {got}, this run ({base_sha}, {cand_sha}): not the same candidate")
    if (fp.get("sources") or {}).get("sha256") != cur_sources["sha256"]:
        raise ValueError("--s1-run: other synced sources than this run's")
    classes = summary.get("fail_classes")
    if not isinstance(classes, dict):
        raise ValueError("--s1-run: summary.json has no FAIL classes")
    blocking = {k: v for k, v in classes.items() if k != "quality-criterion" and v}
    if blocking:
        raise ValueError(f"--s1-run: FAILs of classes {blocking}: S2/S3 run only without a FAIL other than "
                         "quality-criterion")
    params = summary.get("parameters") or {}
    return {"dir": prev.name, "run_json_sha256": mg.sha256(prev / "run.json"),
            "summary_sha256": mg.sha256(prev / "summary.json"), "campaigns": params.get("campaigns"),
            "steps": params.get("steps"), "complete": summary.get("complete"),
            "exit_status": summary.get("exit_status"),
            "stopped": summary.get("stopped"), "counts": summary.get("counts"), "fail_classes": classes,
            "models": {k: {x: v.get(x) for x in ("name", "revision", "config_sha256", "safetensors_sha256")}
                       for k, v in (fp.get("models") or {}).items()},
            "identity": identity_proof(fp, args.perf_model)}


# ---------------------------------------------------------------- fingerprint


def fingerprint(rec, args, ctx, gpus):
    """The run's identities. A --version probe cut at the run deadline or past its own limit (bounded_probe) is
    recorded and kept in ctx.setup_stop: main stops before any GPU work (Fatal) or at the first step (deadline)."""
    versions = {}
    for label, b in ctx.binary.items():
        try:
            r = bounded_probe(rec, ctx, f"{label} binary: --version", 60, lambda limit, b=b: subprocess.run(
                [b, "--version"], capture_output=True, text=True, env=ctx.env, timeout=limit))
            versions[label] = iso.scrub((r.stdout + r.stderr).strip())
        except (Deadline, Fatal) as e:
            versions[label] = f"not recorded: {type(e).__name__}"
            ctx.setup_stop = ctx.setup_stop or e
        except (OSError, subprocess.SubprocessError) as e:
            versions[label] = type(e).__name__
    root = ctx.root
    models = {}
    for short, m in ctx.model.items():
        manifest = read_json(root / f"reports/decision-sets/manifest-reference-basal-1.5-{short}-fp32.json")["model"]
        rec.expect(f"model {short}: revision of the stored references", m["dir"].name == manifest["revision"],
                   f"{m['dir'].name} vs {manifest['revision']}")
        st = m["dir"] / "model.safetensors"
        entry = {"name": m["name"], "revision": m["dir"].name, "safetensors_bytes": st.stat().st_size,
                 "config_sha256": mg.sha256(m["dir"] / "config.json")}
        if not args.no_weight_hash:
            print(f"{time.strftime('%H:%M:%S')} hashing the weights of {short}", flush=True)
            entry["safetensors_sha256"] = mg.sha256(st)
            rec.expect(f"model {short}: weights SHA-256 of the stored references",
                       entry["safetensors_sha256"] == manifest["safetensors_sha256"])
        models[short] = entry
    sources = rp.source_digest(root, ctx.out)
    assets = embedded_assets(root)
    binaries = {}
    for k, b in ctx.binary.items():
        prov = ctx.provenance.get(k)
        binaries[k] = {"sha256": mg.sha256(b), "version_text": versions[k],
                       "declared_label": iso.scrub(getattr(args, f"{k}_label")),
                       "provenance": None if prov is None else {"sha256": prov[1], "record": iso.scrub(json.dumps(
                           prov[0], sort_keys=True))}}
    lib = Path(args.basal_home) / ".local/cuda/12.9.1/lib"
    cpu = None
    if Path("/proc/cpuinfo").is_file():
        cpu = next((x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                    if x.startswith("model name")), None)
    return {
        "runner": RUNNER_ID, "runner_sha256": mg.sha256(HERE),
        "helpers_sha256": {p: a for p, (a, _) in helpers_state(root).items()} |
        {"tools/decision-sets/build_benchmark.py": mg.sha256(root / w8.BUILD_BENCHMARK)},
        "sources": sources, "embedded_assets": assets, "git": w8.git_state(root),
        "source_commit_note": args.source_commit, "binaries": binaries,
        "binary_roles": {"base": "stable W8 base (declared_label: source and provenance as declared by the operator)",
                         "candidate": "W4A16 candidate (declared_label)"},
        "models": models,
        "gemm_tables": {k: {"file": v.name, "sha256": mg.sha256(v)} for k, v in ctx.tables.items()},
        "inputs": {"sequential": {"file": w8.SEQUENTIAL, "sha256": mg.sha256(root / w8.SEQUENTIAL)},
                   "lock_sha256": mg.sha256(root / w8.LOCK),
                   "decisions": {s: mg.sha256(root / w8.DECISIONS.format(s)) for s in ctx.shorts},
                   "references_fp32": {s: w8.tree_sha(reference(ctx, s)) for s in ctx.shorts},
                   "references_mlx_q8": {s: w8.tree_sha(root / w8.MLX_Q8.format(s)) for s in ctx.shorts},
                   "long_states_sha256": mg.sha256(root / w8.LONG), "inputs_900": ctx.inputs900_summary,
                   "config_rs_sha256": mg.sha256(root / "crates/basal-cli/src/config.rs"),
                   "p0_facts": ctx.p0_provenance},
        "gpus": [{"index": g[0], "name": g[1], "driver": g[3], "memory_total_mib": g[4], "power_limit_w": g[5],
                  "clocks_max_sm_mhz": g[6], "clocks_max_mem_mhz": g[7], "persistence_mode": g[8]} for g in gpus],
        "cuda_libraries": sorted(f"{p.name} {p.stat().st_size}" for p in lib.iterdir() if p.is_file())
        if lib.is_dir() else None,
        "cublaslt_tables": args.cublaslt, "python": platform.python_version(), "kernel": platform.release(),
        "cpu": cpu, "cpu_count": os.cpu_count(), "cuda_jit_cache_before_run": w8.jit_cache(ctx.env),
        "deadline": ctx.deadline_basis}


def binding_checks(rec, ctx):
    """The declared build records against the synced tree (candidate), the P0 facts and a family run's S1 evidence;
    returns the problems that forbid GPU work (an invalid binding of the candidate, a known S1 identity difference)."""
    fp = ctx.fp
    blocking = []
    prov = (ctx.provenance.get("candidate") or (None,))[0]
    for key, mine in (("source_digest", fp["sources"]), ("embedded_assets", fp["embedded_assets"])):
        name = f"candidate build record: {key} equal to the synced tree"
        if not prov or key not in prov:
            rec.check(name, "GAP", "not given (--candidate-provenance): the binary is not bound to the synced tree")
            continue
        want = prov[key]
        ok = (want.get("sha256") == mine.get("sha256") and want.get("files") == mine.get("files")) \
            if key == "source_digest" and isinstance(want, dict) else want == mine
        rec.expect(name, ok, f"declared {json.dumps(want)[:200]}, synced {json.dumps(mine)[:200]}")
        if not ok:
            blocking.append(name)
    base = (ctx.provenance.get("base") or (None,))[0]
    rec.check("base binary: declared source label", "GAP",
              f"{ctx.args.base_label!r} as declared (build record {'given' if base else 'not given'}); the runner "
              "does not verify the base's source (parent provenance)")
    rec.check("P0 facts: the approved file", "PASS" if ctx.p0_provenance["usable"] else "GAP",
              f"{P0_FACTS} SHA-256 {ctx.p0_provenance['sha256']}, expected {P0_FACTS_SHA256}")
    if ctx.s1:
        s1 = ctx.s1
        # identity: a known difference forbids GPU work; a proof missing in either run is a GAP of that binding
        theirs, mine = s1["identity"], identity_proof(fp, ctx.args.perf_model)
        equal = []
        for key, b in mine.items():
            a = theirs.get(key)
            if a is None or b is None:
                rec.check(f"S1 evidence: {key}", "GAP", f"not recorded in {'the S1 run' if a is None else 'this run'}"
                          f"{' (--no-weight-hash)' if key.endswith('weights') else ''}: not shown equal")
            elif a != b:
                rec.check(f"S1 evidence: {key} equal to this run's", "FAIL",
                          iso.scrub(f"S1 {json.dumps(a)[:160]}, this run {json.dumps(b)[:160]}"))
                blocking.append(f"S1 {key}")
            else:
                equal.append(key)
        if equal:
            rec.check("S1 evidence: identities equal to this run's", "PASS", f"{len(equal)} of {len(mine)}: "
                      f"{', '.join(equal)}")
        done = set(s1.get("campaigns") or [])
        complete = s1.get("complete") is True and (set(S1_CAMPAIGNS) <= done or "revalidate" in done)
        rec.check("S1 evidence: complete run of the S1 (or revalidation) campaigns", "PASS" if complete else "GAP",
                  f"campaigns {sorted(done)}, complete {s1.get('complete')}, exit {s1.get('exit_status')}, counts "
                  f"{s1.get('counts')}, FAIL classes {s1.get('fail_classes')} (the parent audits the raw data)")
    return blocking


# ---------------------------------------------------------------- plan (no GPU, no binary)


W8_COMMANDS = "reports/w8a16-2026-10-11/commands.json"
W8_NONE_FACTS = "reports/multigpu-w8-none-2026-10-11/facts.json"
W8_Q8_ORIGINAL = "reports/multigpu-w8-original-2026-10-11/perf-q8-original.json"
LIFE_MINUTES = 120  # creation to deletion of a machine (plan)


class PlanSources:
    """The tracked W8 reports (a), private raw W8 run directories (b, --w8-raw) and the P0 facts (allowance); every
    value comes with its file, SHA-256 and field. Reads files only."""

    def __init__(self, root, raw_dirs):
        self.root, self.tracked, self.private, self.servers, self.sessions = root, [], [], {}, []
        for path in (W8_COMMANDS, W8_Q8_ORIGINAL):
            p = root / path
            if p.is_file():
                steps = {}
                for c in read_json(p).get("commands", []):
                    if w8.finite(c.get("seconds")):
                        steps.setdefault(c["step"], c["seconds"])
                self.tracked.append((path, mg.sha256(p), steps))
        p = root / W8_NONE_FACTS
        self.none_facts = (W8_NONE_FACTS, mg.sha256(p), read_json(p)) if p.is_file() else None
        for d in raw_dirs:
            found = [d / "commands.jsonl"] if (d / "commands.jsonl").is_file() else sorted(d.glob("*/commands.jsonl"))
            for f in found:
                steps = {}
                for line in f.read_text().splitlines():
                    if line.strip():
                        c = json.loads(line)
                        if w8.finite(c.get("seconds")):
                            steps.setdefault(c["step"], []).append(c["seconds"])
                self.private.append((rel(root, f), mg.sha256(f), steps))
                served = 0.0
                for t in sorted(f.parent.glob("correctness/serve/*/telemetry.csv")):
                    times = [float(x.split(",", 1)[0]) for x in t.read_text().splitlines()[1:] if x.split(",", 1)[0]]
                    if len(times) > 1:
                        served += times[-1] - times[0]
                        self.servers.setdefault(t.parent.name, []).append(
                            (times[-1] - times[0], {"kind": "b", "file": rel(root, t), "sha256": mg.sha256(t),
                                                    "field": "last t - first t (server lifetime)", "private": True}))
                self.session_wall(f.parent, sum(sum(x) for x in steps.values()), served)
        facts, self.p0_provenance = load_p0(root)
        self.p0 = {s: ((p0_model(facts, s) or {}).get("formats", {}).get(Q4, {}).get("aggregate", {})
                       .get("host_pass_seconds")) for s in SHORTS} if facts else {}
        self.p0_hash = {s: ((p0_model(facts, s) or {}).get("seconds") or {}).get("hash") for s in SHORTS} \
            if facts else {}

    def session_wall(self, d, commands_s, servers_s):
        """Runner wall time of a W8 session (run.json started to summary.json finished, after its weight hashing)
        against its recorded commands and servers: the orchestration not in the per-step rows."""
        def when(path, key):
            return datetime.datetime.strptime(read_json(path)[key], "%Y-%m-%dT%H:%M:%S%z")

        try:
            wall = (when(d / "summary.json", "finished") - when(d / "run.json", "started")).total_seconds()
        except (OSError, ValueError, KeyError):
            return
        self.sessions.append({"files": [rel(self.root, d / "run.json"), rel(self.root, d / "summary.json")],
                              "sha256": [mg.sha256(d / "run.json"), mg.sha256(d / "summary.json")],
                              "wall_min": wall / 60, "commands_min": commands_s / 60,
                              "servers_min": servers_s / 60})

    def cmd(self, step):
        """(seconds, [source]) of a W8 command: a tracked report first, else the largest of the private records."""
        for path, sha, steps in self.tracked:
            if step in steps:
                return steps[step], [{"kind": "a", "file": path, "sha256": sha,
                                      "field": f"commands[step={step}].seconds"}]
        hits = [(max(steps[step]), path, sha) for path, sha, steps in self.private if step in steps]
        if hits:
            s, path, sha = max(hits)
            return s, [{"kind": "b", "file": path, "sha256": sha, "field": f"step {step}: seconds", "private": True}]
        return None, []

    def none_benchmark(self, round_, variant):
        if not self.none_facts:
            return None, []
        path, sha, d = self.none_facts
        xs = [r["process"]["seconds"] for r in d["benchmark_rounds"][round_]["runs"] if r.get("variant") == variant]
        if not xs:
            return None, []
        return max(xs), [{"kind": "a", "file": path, "sha256": sha,
                          "field": f"benchmark_rounds.{round_}.runs[variant={variant}].process.seconds (max)"}]

    def server(self, label):
        hits = self.servers.get(label) or []
        if not hits:
            return None, []
        s, src = max(hits, key=lambda x: x[0])
        return s, [src]


def resolve(src, basis):
    """(seconds, sources, note) of a basis: ("cmd", step), ("cmds", [steps]: the largest), ("none-benchmark", round,
    variant), ("server", label), ("scale", factor, basis), ("q8-bench-process",) (assumption c:
    one q8_0 `basal bench` process = (q8-bench - none-bench-primary / 2) / 4), ("assume", seconds, why)."""
    kind = basis[0]
    if kind == "cmd":
        s, srcs = src.cmd(basis[1])
        return s, srcs, None
    if kind == "cmds":
        best = (None, [], None)
        for step in basis[1]:
            s, srcs = src.cmd(step)
            if s is not None and (best[0] is None or s > best[0]):
                best = (s, srcs, None)
        return best
    if kind == "none-benchmark":
        s, srcs = src.none_benchmark(basis[1], basis[2])
        return s, srcs, None
    if kind == "server":
        s, srcs = src.server(basis[1])
        return s, srcs, None
    if kind == "scale":
        s, srcs, note = resolve(src, basis[2])
        return (None if s is None else basis[1] * s), srcs, note
    if kind == "q8-bench-process":
        q8, a, _ = resolve(src, ("cmd", "q8-bench"))
        f16, b, _ = resolve(src, ("cmd", "none-bench-primary"))
        if q8 is None or f16 is None:
            return None, a + b, "assumption needs both W8 ab.py records"
        why = "assumption: one q8_0 basal bench process = (W8 q8-bench [4 f16 + 4 q8_0] - W8 none-bench-primary " \
              "[8 f16] / 2) / 4"
        return (q8 - f16 / 2) / 4, a + b + [{"kind": "c", "field": why}], "derived assumption"
    if kind == "assume":
        return basis[1], [{"kind": "c", "field": f"assumption: {basis[2]}"}], "assumption"
    raise ValueError(basis)


def plan_rows(args, root, src):
    """Every step of every session: dicts with session, campaign, what, count, basis, q4 loads, model of the
    allowance, cap (s), conditional, kind, HTTP requests."""
    items = {s: mg.line_count(root / w8.REFERENCE.format(s) / "bench.jsonl") for s in SHORTS}
    cases = {s: mg.line_count(root / w8.REFERENCE.format(s) / "systemone.jsonl") for s in SHORTS}
    http = items["mini"] + cases["mini"] + mg.line_count(root / w8.LONG) + 5
    cap_e, cap_b, cap_m = args.export_timeout, args.bench_timeout, args.benchmark_timeout
    cap_s = args.ready_timeout + args.shutdown_timeout
    rows = []

    loads = {"export": 1, "decide": 1, "benchmark": 1, "profile": 1, "server": 1, "ab": 8}

    def row(session, campaign, what, count, basis, kind, q4=0, model=None, cap=None, conditional=False, requests=0):
        # a q4_k replica server loads the model once per replica (q4 = 2)
        rows.append({"session": session, "campaign": campaign, "what": what, "count": count, "basis": basis,
                     "kind": kind, "model_loads_each": max(loads.get(kind, 0), q4 if kind == "server" else 0),
                     "q4_loads_each": q4, "model": model, "cap_s": cap, "conditional": conditional,
                     "http_requests_each": requests})

    def none_rows(session, campaign):
        for s in SHORTS:
            for mode in ("tree", "single"):
                row(session, campaign, f"{s} base none {mode} export", 1, ("cmd", f"export-{s}-{mode}-base-dev0"),
                    "export", cap=cap_e)
                names = [f"{s}-{mode}-none-dev0", f"{s}-{mode}-none-explicit-dev0"] + \
                    ([f"{s}-tree-none-dev1"] if mode == "tree" else [])
                for n in names:
                    row(session, campaign, f"{n} export", 1, ("cmd", f"export-{n}"), "export", cap=cap_e)
                    row(session, campaign, f"{n} vs base: basal compare (no GPU)", 1, ("cmd", f"compare-{n}-vs-base"),
                        "compare", cap=900)

    def q8_rows(session, campaign):
        for s in SHORTS:
            row(session, campaign, f"{s} base q8_0 tree export (proxy: the W8 q8_0 tree export of the W8 candidate)",
                1, ("cmd", f"export-{s}-tree-q8-dev0"), "export", cap=cap_e)
            for n in (f"{s}-tree-q8-dev0", f"{s}-single-q8-dev0", f"{s}-tree-noprefix-q8-dev0", f"{s}-tree-q8-dev1",
                      f"{s}-tree-q8-visible1-device0"):
                row(session, campaign, f"{n} export", 1, ("cmd", f"export-{n}"), "export", cap=cap_e)
                row(session, campaign, f"{n} vs base: basal compare (proxy: a W8 q8_0 compare)", 1,
                    ("cmd", f"compare-{s}-single-q8-dev0-vs-tree-q8-dev0"), "compare", cap=900)

    def w4_rows(session, campaign, conditional=False, with_tree=True):
        for s in SHORTS:
            for n, w in (((f"{s}-tree-q4-dev0", f"{s}-tree-q8-dev0"),) if with_tree else ()) + (
                         (f"{s}-single-q4-dev0", f"{s}-single-q8-dev0"),
                         (f"{s}-tree-noprefix-q4-dev0", f"{s}-tree-noprefix-q8-dev0"),
                         (f"{s}-tree-q4-dev1", f"{s}-tree-q8-dev1"),
                         (f"{s}-tree-q4-visible1-device0", f"{s}-tree-q8-visible1-device0")):
                row(session, campaign, f"{n} export (W8 q8_0 process + P0 allowance)", 1, ("cmd", f"export-{w}"),
                    "export", q4=1, model=s, cap=cap_e, conditional=conditional)
                if n != f"{s}-tree-q4-dev0":
                    row(session, campaign, f"{n} vs tree: basal compare (proxy)", 1,
                        ("cmd", f"compare-{s}-single-q8-dev0-vs-tree-q8-dev0"), "compare", cap=900,
                        conditional=conditional)

    def setup_rows(session, shorts, note):
        row(session, "setup", f"SHA-256 of the {' and '.join(shorts)} weights ({note})", 1,
            ("assume", sum(src.p0_hash.get(s) or 0.0 for s in shorts) or None, "the local P0 hash seconds of "
             f"{' and '.join(shorts)} (tracked P0 facts models[].seconds.hash); not a cloud measurement"), "setup")

    def serve_rows(session, campaign):
        row(session, campaign, "mini q8_0 export of the served request bodies", 1,
            ("cmd", "export-mini-tree-q8-served-dev0"), "export", cap=cap_e)
        row(session, campaign, "basal decide of a served q8_0 body the export refused as Unsupported (W8 count 2)", 2,
            ("cmds", ["decide-09-so-score-one-level", "decide-10-so-eleven-options"]), "decide", cap=cap_e)
        row(session, campaign, "mini q4_k export of the served request bodies (q8_0 + P0)", 1,
            ("cmd", "export-mini-tree-q8-served-dev0"), "export", q4=1, model="mini", cap=cap_e)
        row(session, campaign, "basal decide of a served q4_k body the export refused as Unsupported (q8_0 + P0)",
            2, ("cmds", ["decide-09-so-score-one-level", "decide-10-so-eleven-options"]), "decide", q4=1,
            model="mini", cap=cap_e)
        servers = [("none-base-dev0", "none-base-dev0", 0, 1), ("none-cand-dev0", "none-cand-dev0", 0, 1),
                   ("q8-base-dev0 (proxy: W8 q8-cand-dev0)", "q8-cand-dev0", 0, 1),
                   ("q8-cand-dev0", "q8-cand-dev0", 0, 1),
                   ("q4-cand-dev0 (W8 q8-cand-dev0 + P0)", "q8-cand-dev0", 1, 1),
                   (f"q4-replicas [0, 1] (W8 q8-replicas + 2 P0; sequential + {args.rounds} x {args.clients} "
                    "clients)", "q8-replicas", 2, 1 + args.rounds)]
        for what, w8_label, q4, passes in servers:
            row(session, campaign, f"server {what}", 1, ("server", w8_label), "server", q4=q4, model="mini",
                cap=cap_s + passes * http * 2, requests=passes * http)

    def baseline_900_rows(session, campaign):
        for s in SHORTS:
            row(session, campaign, f"{s} 900 f16 tree, candidate and base", 2, ("cmd", f"export-{s}-900-f16-tree"),
                "export", cap=cap_e)
            row(session, campaign, f"{s} 900 q8_0 tree, candidate and base", 2, ("cmd", f"export-{s}-900-q8-tree"),
                "export", cap=cap_e)

    def quality_rows(session, campaign, conditional=False):
        for s in SHORTS:
            row(session, campaign, f"{s} 44+34 basal compare (5 + MLX context; proxy)", 6,
                ("cmd", f"compare-{s}-small-fp32-vs-q8"), "compare", cap=900, conditional=conditional)
            row(session, campaign, f"{s} 900 f32 control export", 1, ("cmd", f"export-{s}-900-f32"), "export",
                cap=cap_e, conditional=conditional)
            row(session, campaign, f"{s} 900 q4_k tree export (W8 q8_0 900 + P0)", 1,
                ("cmd", f"export-{s}-900-q8-tree"), "export", q4=1, model=s, cap=cap_e, conditional=conditional)
            row(session, campaign, f"{s} 900 basal compare (5; proxy)", 5, ("cmd", f"compare-{s}-900-f32-vs-q8"),
                "compare", cap=900, conditional=conditional)

    def family_rows(session, fam):
        c = f"perf-{fam}-family"
        setup_rows(session, [args.perf_model], "every invocation hashes its models")
        q8pp = ("q8-bench-process",)
        if fam == "none":
            ab_ab = ("cmd", "none-bench-primary")
            ab_bc = ("cmd", "q8-bench")
            bench = {"A": ("none-benchmark", "primary", "A"), "B": ("none-benchmark", "primary", "B"),
                     "C": ("cmds", ["q8-benchmark-1-B", "q8-benchmark-2-B"])}
            conf_bench, conf_benchmark = ab_ab, [("none-benchmark", "confirmation", v) for v in "ABBA"]
            profiles = [("cmd", "none-confirm-base-f16-b1"), ("cmd", "none-confirm-candidate-f16-b1")]
            bc_note = "W8 q8-bench (4 f16 + 4 q8_0) + 4 P0"
        else:
            ab_ab = ("scale", 8, q8pp)
            ab_bc = ("scale", 8, q8pp)
            q8b = ("cmds", ["q8-benchmark-1-B", "q8-benchmark-2-B"])
            bench = {"A": q8b, "B": q8b, "C": q8b}
            conf_bench, conf_benchmark = ab_ab, [q8b] * 4
            profiles = [("cmd", "profile-candidate-q8_0-b1")] * 2
            bc_note = "8 q8_0 processes (assumption) + 4 P0"
        row(session, c, "ab.py --rounds 4 A/B (8 basal bench processes)", 1, ab_ab, "ab", cap=8 * cap_b)
        row(session, c, f"ab.py --rounds 4 B/C ({bc_note})", 1, ab_bc, "ab", q4=4, model=args.perf_model,
            cap=8 * cap_b)
        for k, role in enumerate(FAMILY_ORDER):
            row(session, c, f"basal benchmark run {k} {role}" + (" (W8 q8_0 benchmark + P0)" if role == "C" else ""),
                1, bench[role], "benchmark", q4=int(role == "C"), model=args.perf_model, cap=cap_m)
        row(session, c, "conditional: confirmation ab.py A/B", 1, conf_bench, "ab", cap=8 * cap_b, conditional=True)
        for k, b in enumerate(conf_benchmark):
            row(session, c, f"conditional: confirmation basal benchmark run {k} {'ABBA'[k]}", 1, b, "benchmark",
                cap=cap_m, conditional=True)
        for v, b in zip("AB", profiles):
            row(session, c, f"conditional: basal profile {v} batch 1", 1, b, "profile", cap=cap_b, conditional=True)

    # S1
    setup_rows("S1", list(SHORTS), "P0 local hash time as the basis; cloud disk unknown")
    none_rows("S1", "correctness/none")
    q8_rows("S1", "correctness/q8")
    w4_rows("S1", "correctness/w4")
    d = Path("/nonexistent")
    n_invalid = len(invalid_cases(d, lambda label, **kw: d, d, d))
    row("S1", "correctness/invalid", "refused q4_k configurations (no load)", n_invalid,
        ("cmds", ["invalid-serve-yaml-bf16", "invalid-export-unknown", "invalid-profile-candle"]), "invalid", cap=120)
    row("S1", "correctness/invalid", "doctor (q4_k bf16, Q4_K, known K, valid q4_k, q8_0 base and candidate)", 6,
        ("cmds", ["doctor-bf16", "doctor-q8-mini"]), "doctor", cap=180)
    serve_rows("S1", "correctness/serve")
    quality_rows("S1", "quality")
    baseline_900_rows("S1", "quality")
    for b in (1, args.profile_batch):
        for q, step in (("f16", "f16"), (Q8, "q8_0"), (Q4, "q8_0")):
            row("S1", "profile", f"basal profile {q} batch {b}" + (" (q8_0 + P0)" if q == Q4 else ""), 1,
                ("cmd", f"profile-candidate-{step}-b{16 if b > 1 else 1}"), "profile", q4=int(q == Q4),
                model=args.perf_model, cap=cap_b)
    for v, step in (("f16", "memory-mini-0-f16"), (Q8, "memory-mini-1-q8_0"), (Q4, "memory-mini-1-q8_0")):
        row("S1", "memory-mini", f"basal benchmark mini {v}" + (" (q8_0 + P0)" if v == Q4 else ""), 1, ("cmd", step),
            "benchmark", q4=int(v == Q4), model="mini", cap=cap_m)
    # S2, S3
    family_rows("S2", "none")
    family_rows("S3", "q8")
    # the one conditional revalidation session of the plan: R1 (the revalidate invocation) and, when the fix
    # affects them, the families as separate invocations R2 and R3 (plan_report: one session, and a proposed split)
    setup_rows("R1", list(SHORTS), "as S1")
    none_rows("R1", "revalidate/none")
    q8_rows("R1", "revalidate/q8")
    serve_rows("R1", "revalidate/serve")  # q4_k native HTTP and replicas of the final candidate: always, once
    baseline_900_rows("R1", "revalidate/900")
    for s in SHORTS:
        row("R1", "revalidate", f"{s} q4_k tree export vs the previous candidate (q8_0 + P0)", 1,
            ("cmd", f"export-{s}-tree-q8-dev0"), "export", q4=1, model=s, cap=cap_e)
    w4_rows("R1", "revalidate/conditional w4", conditional=True, with_tree=False)  # tree: the export above
    quality_rows("R1", "revalidate/conditional quality", conditional=True)
    family_rows("R2", "none")
    family_rows("R3", "q8")
    return rows, {"http_requests_per_pass": http, "bench_items": items, "systemone_cases": cases}


SESSIONS = {"S1": "S1: correctness, quality, profile, memory-mini", "S2": "S2: perf-none-family",
            "S3": "S3: perf-q8-family",
            "R1": "R1: revalidate invocation (conditional): correctness none/q8, servers incl. q4_k native HTTP and "
                  "replicas, 900 bytes, q4_k vs previous, conditional w4 and quality",
            "R2": "R2: revalidated perf-none-family invocation (conditional, if affected)",
            "R3": "R3: revalidated perf-q8-family invocation (conditional, if affected)"}
SPLIT_NOTE = "PROPOSED split, not authorised by the Ready plan, which plans one revalidation session; a parent " \
             "plan and forecast revision is needed before any purchase"
# the Ready plan's one conditional revalidation session: the revalidate invocation and the affected families, one
# machine (one bootstrap, one collect/delete reserve), unsplit
REVALIDATION_SESSION = {"R1+R2": "revalidation session (plan, unsplit): revalidate + perf-none-family",
                        "R1+R3": "revalidation session (plan, unsplit): revalidate + perf-q8-family",
                        "R1+R2+R3": "revalidation session (plan, unsplit): revalidate + both families"}


def session_summary(args, rs, title):
    """Totals of the rows `rs` run on one machine: runner seconds nominal and with every conditional step, counts,
    and the machine minutes with the assumed bootstrap and collect/delete reserve against LIFE_MINUTES."""
    nominal = [r for r in rs if not r["conditional"]]

    def total(xs):
        return sum(r["count"] * r["seconds_each"] for r in xs if r["seconds_each"] is not None)

    kinds = Counter()
    for r in rs:
        kinds[r["kind"]] += r["count"]
    out = {"title": title, "nominal_s": total(nominal), "with_all_conditional_s": total(rs),
           "p0_allowance_s": sum(r["count"] * r["p0_allowance_each"] for r in rs if r["seconds_each"] is not None),
           "steps_without_source": [f"{r['count']} x {r['what']}" for r in rs
                                    if r["seconds_each"] is None and r["count"]],
           "caps_worst_case_s": sum(r["count"] * (r["cap_s"] or 0) for r in rs),
           "counts_with_conditional": dict(kinds),
           "model_loads": sum(r["count"] * r["model_loads_each"] for r in rs),
           "q4_loads": sum(r["count"] * r["q4_loads_each"] for r in rs),
           "http_requests": sum(r["count"] * r["http_requests_each"] for r in rs)}
    for key, part in (("nominal_s", nominal), ("with_all_conditional_s", rs)):
        m = out[key] / 60 + args.bootstrap_minutes + args.collect_delete_minutes
        out[key.replace("_s", "_machine_minutes")] = m
        # without a source for every step the fit is not determined (never a "fits" from missing data)
        known = all(r["seconds_each"] is not None for r in part if r["count"])
        out[key.replace("_s", "_fits_life")] = (m <= LIFE_MINUTES) if known else None
    return out


def plan_report(args, root):
    """The forecast of every session from its sources; minutes and seconds only, nothing measured here."""
    src = PlanSources(root, args.w8_raw)
    rows, facts = plan_rows(args, root, src)
    out = {"scope": "planning estimate from W8 records (a tracked, b private) and explicit assumptions (c); the P0 "
                    "q4_k host pass per q4_k load is an uncalibrated allowance, not a measured engine overhead or a "
                    "bound; GPU kernels, dequantization, cloud CPU and the warm-up D2H check are unknown; nothing "
                    "here is a measurement of this campaign",
           "p0_allowance_seconds_per_load": src.p0, "p0_facts": src.p0_provenance, "facts": facts,
           "life_minutes": LIFE_MINUTES, "bootstrap_minutes_assumed": args.bootstrap_minutes,
           "collect_delete_minutes_assumed": args.collect_delete_minutes, "w8_sessions": src.sessions,
           "sessions": {}, "revalidation_session": {}, "proposed_split_note": SPLIT_NOTE, "rows": []}
    for r in rows:
        s, srcs, note = resolve(src, r["basis"])
        allowance = r["q4_loads_each"] * (src.p0.get(r["model"]) or 0.0) if r["q4_loads_each"] else 0.0
        if r["q4_loads_each"] and not src.p0.get(r["model"]):
            note = (note or "") + "; P0 allowance unavailable"
        each = None if s is None else s + allowance
        out["rows"].append({k: v for k, v in r.items() if k != "basis"} | {
            "basis_seconds_each": s, "p0_allowance_each": allowance, "seconds_each": each, "sources": srcs,
            "note": note})
    for sid, title in SESSIONS.items():
        out["sessions"][sid] = session_summary(args, [r for r in out["rows"] if r["session"] == sid], title)
        out["sessions"][sid]["proposed_split_part"] = sid.startswith("R")
    for key, title in REVALIDATION_SESSION.items():
        ids = key.split("+")
        out["revalidation_session"][key] = session_summary(args, [r for r in out["rows"] if r["session"] in ids],
                                                           title)
    return out


def print_plan(args, root):
    print("900-question inputs (rebuilt in memory from the tracked files, nothing written):")
    for short in args.models.split():
        rows, decisions = w8.build_900(root, short)
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        print(f"  {short}: {len(rows)} records, indices 0..{len(rows) - 1} unique, {len(decisions)} decision ids, "
              f"round trip public_request 900/900, types {dict(Counter(r['type'] for r in rows))}, datasets "
              f"{len({r['dataset'] for r in rows})}, bench.jsonl sha256 {hashlib.sha256(data).hexdigest()}")
    plan = plan_report(args, root)
    chosen = sorted({"S1" if c in S1_CAMPAIGNS else "R1" if c == "revalidate" else
                     "S2/R2" if c == "perf-none-family" else "S3/R3" for c in args.campaigns})
    print(f"\nthis invocation's --campaigns {' '.join(args.campaigns)}: session {', '.join(chosen)}; every session "
          "of the plan follows (S1 -> parent audit of the raw data -> S2 -> S3; after an actual code fix only, the one "
          "conditional revalidation session: the revalidate invocation R1 and the affected families R2 / R3)")
    print(f"\nsources: (a) tracked W8 report, (b) private raw W8 record (--w8-raw: {len(args.w8_raw)} given), "
          "(c) explicit assumption; P0 allowance per q4_k load (tracked P0 facts "
          f"{P0_FACTS}, sha256 {plan['p0_facts']['sha256']}, usable {plan['p0_facts']['usable']}): "
          f"{ {k: v for k, v in plan['p0_allowance_seconds_per_load'].items()} } s")
    print(plan["scope"])
    for w in plan["w8_sessions"]:
        print(f"  W8 session (b, private: {', '.join(w['files'])}, sha256 {', '.join(x[:12] for x in w['sha256'])}): "
              f"runner wall {w['wall_min']:.1f} min, recorded commands {w['commands_min']:.1f} min + servers "
              f"{w['servers_min']:.1f} min: orchestration between them is small and not added below")
    for sid, title in SESSIONS.items():
        print(f"\n{title}")
        for r in (x for x in plan["rows"] if x["session"] == sid and x["count"]):
            src = "; ".join(f"({s['kind']}) {s.get('file', '')} {s.get('sha256', '')[:12] if s.get('sha256') else ''} "
                            f"{s['field']}".replace("  ", " ") for s in r["sources"]) or "NO SOURCE"
            each = "no source" if r["seconds_each"] is None else \
                f"{r['seconds_each']:.1f} s each" + (f" (incl. P0 {r['p0_allowance_each']:.1f} s)"
                                                    if r["p0_allowance_each"] else "")
            cond = " [conditional]" if r["conditional"] else ""
            note = f" ({r['note']})" if r["note"] else ""
            print(f"  {r['campaign']:32} {r['count']:3} x {r['what']}{cond}: {each}{note}; cap {r['cap_s']} s; {src}")
        print_totals(args, plan["sessions"][sid], SPLIT_NOTE if sid.startswith("R") else None)
    print("\nthe one conditional revalidation session of the Ready plan, unsplit (rows R1, R2, R3 above; one machine, "
          "one bootstrap and one collect/delete reserve; every invocation hashes its models):")
    for key, s in plan["revalidation_session"].items():
        print(f"\n{s['title']} ({key})")
        print_totals(args, s, None)
    print(f"\nrunner deadline of a session: --deadline-minutes = {LIFE_MINUTES} - minutes since the machine's creation "
          f"- the collect/delete reserve (or --lease-end-unix); the parent guard enforces life and budget.")
    print("Not included: orchestration between commands (telemetry, idle waits; small in the W8 sessions above), the "
          "weights prefetch. Nothing here is a measurement, a bound or an affordability decision.")
    if args.plan_json:
        mg.write_json(args.plan_json, plan)
        print(f"plan written: {args.plan_json.name}")


def print_totals(args, s, split):
    """The totals of a session_summary; `split`: the note of a proposed split part (its fit is not a plan fit)."""
    print(f"  = runner nominal {s['nominal_s'] / 60:.1f} min, with every conditional step "
          f"{s['with_all_conditional_s'] / 60:.1f} min (of which P0 allowance {s['p0_allowance_s'] / 60:.1f} min); "
          f"steps without a source: {s['steps_without_source'] or 'none'}; per-step caps (not a forecast) "
          f"{s['caps_worst_case_s'] / 3600:.1f} h")
    print(f"    counts with conditional steps {s['counts_with_conditional']}, model loads {s['model_loads']} (q4_k "
          f"{s['q4_loads']}), HTTP requests {s['http_requests']}")
    for key, label in (("nominal", "nominal"), ("with_all_conditional", "with every conditional step")):
        m, fits = s[f"{key}_machine_minutes"], s[f"{key}_fits_life"]
        if fits is None:
            verdict = "NOT DETERMINED: steps without a source (give the W8 records or an explicit assumption)"
        elif fits:
            verdict = "fits"
        else:
            verdict = "DOES NOT FIT: a split and a revised forecast are needed before a purchase (parent plan " \
                      "revision; the runner does not cut steps, pairs or metrics)"
        if split:
            verdict = f"{verdict} as a separate machine session ({split})"
        print(f"    machine life {label}: {m:.1f} min = runner (known steps) + bootstrap {args.bootstrap_minutes} + "
              f"collect/delete {args.collect_delete_minutes} (assumptions c) of {LIFE_MINUTES}: {verdict}")


# ---------------------------------------------------------------- main


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


def non_negative(text):
    v = float(text)
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError(f"{text}: must be a non-negative finite number")
    return v


def sha256_arg(text):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise argparse.ArgumentTypeError(f"{text}: not a SHA-256")
    return text.lower()


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    head, _, rest = __doc__.partition("\n\n")
    ap = argparse.ArgumentParser(description=head, epilog=rest, formatter_class=HelpFormatter)
    ap.add_argument("--base", type=Path, help="stable W8 CUDA binary (source 750c146)")
    ap.add_argument("--candidate", type=Path, help="W4A16 CUDA binary")
    ap.add_argument("--expect-base-sha256", type=sha256_arg, help="required: SHA-256 the base must have")
    ap.add_argument("--expect-candidate-sha256", type=sha256_arg, help="required: SHA-256 the candidate must have")
    ap.add_argument("--base-label", help="required: declared role and source of the base (recorded as declared)")
    ap.add_argument("--candidate-label", help="required: declared role and source of the candidate")
    ap.add_argument("--base-provenance", type=Path, help="optional JSON build record of the base")
    ap.add_argument("--candidate-provenance", type=Path, help="optional JSON build record of the candidate")
    ap.add_argument("--out", type=Path, help="new output directory, outside Git (e.g. .cache/cloud/...)")
    ap.add_argument("--campaigns", type=str.split, default=list(S1_CAMPAIGNS), help=f"of: {' '.join(CAMPAIGNS)}")
    ap.add_argument("--steps", type=str.split, default=list(STEPS), help=f"correctness steps: {' '.join(STEPS)}")
    ap.add_argument("--models", default="mini 4.5B", help="models of correctness, quality and revalidate")
    ap.add_argument("--perf-model", default="4.5B", help="model of the families and profile")
    ap.add_argument("--s1-run", type=Path, help="family runs: raw directory of the S1 or revalidation run")
    ap.add_argument("--previous-run", type=Path, help="revalidate: raw directory of the pre-fix candidate's run")
    ap.add_argument("--expect-previous-candidate-sha256", type=sha256_arg, help="revalidate: its candidate's SHA-256")
    ap.add_argument("--source-commit", default=None, help="free text recorded with the source fingerprint")
    ap.add_argument("--prepare-900", type=Path, help="write the 900-question inputs to this new directory and exit")
    ap.add_argument("--clients", type=positive_int, default=16, help="serve q4-replicas: concurrent clients")
    ap.add_argument("--rounds", type=positive_int, default=2, help="serve q4-replicas: concurrent passes")
    ap.add_argument("--long-tokens", type=positive_int, default=3000, help="long lane threshold of the served models")
    ap.add_argument("--profile-n", type=positive_int, default=40, help="basal profile --n")
    ap.add_argument("--profile-batch", type=positive_int, default=16, help="the larger basal profile --batch")
    ap.add_argument("--port", type=positive_int, default=18500)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--export-timeout", type=positive_int, default=1800, help="one basal export or decide (s)")
    ap.add_argument("--bench-timeout", type=positive_int, default=900, help="one basal bench or profile run (s)")
    ap.add_argument("--benchmark-timeout", type=positive_int, default=2400, help="one basal benchmark run (s)")
    ap.add_argument("--ready-timeout", type=positive_int, default=600, help="server readiness (s)")
    ap.add_argument("--http-timeout", type=positive_int, default=300, help="one HTTP request (s)")
    ap.add_argument("--shutdown-timeout", type=positive_int, default=120, help="graceful server shutdown (s)")
    ap.add_argument("--deadline-minutes", type=positive_finite, default=None,
                    help="required for a run: no step, command or request starts after it (machine life limit minus "
                         "the time since its creation minus the collect/delete reserve)")
    ap.add_argument("--lease-end-unix", type=positive_finite, default=None,
                    help="optional absolute end (Unix time) of the runner's share of the machine's life; the earlier "
                         "of it and --deadline-minutes applies")
    ap.add_argument("--no-weight-hash", action="store_true",
                    help="skip the SHA-256 of the weights (recorded; P0 equality and revalidation become GAPs)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and forecast; no GPU, binary or network")
    ap.add_argument("--w8-raw", type=Path, action="append", default=[],
                    help="plan: a private raw W8 run directory (commands.jsonl; server telemetry), repeatable")
    ap.add_argument("--plan-json", type=Path, help="plan: also write the forecast as JSON to this new file")
    ap.add_argument("--bootstrap-minutes", type=non_negative, default=10.0,
                    help="plan: assumed machine creation to runner start (W8: about 9-10 min)")
    ap.add_argument("--collect-delete-minutes", type=non_negative, default=10.0,
                    help="plan: assumed reserve to pull the results with SHA-256 and delete the machine")
    return ap, ap.parse_args()


def validate(ap, args, root):
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
    families = [c for c in args.campaigns if c in FAMILY_CAMPAIGNS]
    if families and len(args.campaigns) != 1:
        ap.error("a perf family is a session of its own (S2 or S3): one family, no other campaign")
    if "correctness" in args.campaigns and not args.steps:
        ap.error("correctness without any --steps")
    if {"correctness", "quality", "revalidate"} & set(args.campaigns) and not listed:
        ap.error("correctness, quality and revalidate need --models")
    if ("serve" in args.steps and "correctness" in args.campaigns or "revalidate" in args.campaigns) and \
            "mini" not in listed:
        ap.error("the serve step and revalidate serve mini: keep mini in --models")
    if args.port > 65535:
        ap.error("--port must be at most 65535")
    missing = [p for p in (w8.SEQUENTIAL, w8.LOCK, w8.BUILD_BENCHMARK, w8.AB, w8.LONG,
                           *(w8.DECISIONS.format(s) for s in SHORTS),
                           *(w8.REFERENCE.format(s) + "/bench.jsonl" for s in SHORTS)) if not (root / p).is_file()]
    if missing:
        ap.error(f"run in the repository root (/work): missing {missing}")
    changed = {p: a for p, (a, want) in helpers_state(root).items() if a != want}
    if changed:
        ap.error(f"helpers other than the ones this runner was written against: {sorted(changed)}")


def main():
    ap, args = parse_args()
    root = Path.cwd()
    validate(ap, args, root)
    for p in (args.plan_json, args.prepare_900):
        if p is not None and (p.exists() or root / "reports" in p.resolve().parents):
            ap.error(f"{p}: must be new and outside reports/ (e.g. .cache/)")
    if args.plan_only:
        try:
            print_plan(args, root)
        except Fatal as e:
            print(f"900 inputs: {e}", file=sys.stderr)
            return 2
        return 0
    if args.prepare_900:
        try:
            print(json.dumps(w8.write_900(root, args.prepare_900, args.models.split()), indent=1))
        except Fatal as e:
            print(f"900 inputs: {e}", file=sys.stderr)
            return 2
        return 0

    t0, w0 = time.monotonic(), time.time()  # the run's deadline counts from here
    if bool(set(args.campaigns) & set(FAMILY_CAMPAIGNS)) != bool(args.s1_run):
        ap.error("a perf family needs --s1-run, and --s1-run is only for a perf family")
    if ("revalidate" in args.campaigns) != bool(args.previous_run) or \
            bool(args.previous_run) != bool(args.expect_previous_candidate_sha256):
        ap.error("revalidate needs --previous-run and --expect-previous-candidate-sha256, and they are only for it")
    if not (args.out and args.base and args.candidate and args.expect_base_sha256 and args.expect_candidate_sha256 and
            args.deadline_minutes and args.base_label and args.candidate_label):
        ap.error("--out, --base, --candidate, --expect-base-sha256, --expect-candidate-sha256, --base-label, "
                 "--candidate-label and --deadline-minutes are required")
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    if root / "reports" in args.out.resolve().parents:
        ap.error("raw output belongs outside reports/ and Git (e.g. .cache/cloud/SESSION)")
    binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    shas = {}
    for label, b in binaries.items():
        if not os.access(b, os.X_OK):
            ap.error(f"not an executable: {b}")
        shas[label], want = mg.sha256(b), getattr(args, f"expect_{label}_sha256")
        if shas[label] != want:
            ap.error(f"{label} binary SHA-256 is {shas[label]}, expected {want}")
    if binaries["base"] == binaries["candidate"] or shas["base"] == shas["candidate"]:
        ap.error("--base and --candidate must be different binaries")
    provenance = {}
    for label in binaries:
        path = getattr(args, f"{label}_provenance")
        if path is not None:
            try:
                provenance[label] = read_provenance(path, shas[label])
            except ValueError as e:
                ap.error(f"--{label}-provenance: {e}")
    lease_s = None if args.lease_end_unix is None else args.lease_end_unix - w0
    if lease_s is not None and lease_s <= 0:
        ap.error("--lease-end-unix has passed")
    requested = args.deadline_minutes
    seconds = min(requested * 60, lease_s) if lease_s is not None else requested * 60
    cur_sources = sources_digest(root)

    shorts = args.models.split()
    needed = set()
    if {"correctness", "quality", "revalidate"} & set(args.campaigns):
        needed |= set(shorts)
    if "memory-mini" in args.campaigns or ("serve" in args.steps and "correctness" in args.campaigns):
        needed.add("mini")
    if {"profile", *FAMILY_CAMPAIGNS} & set(args.campaigns):
        needed.add(args.perf_model)
    def smi(query):  # bounded_wait (own 30 s, the run deadline); nothing starts after it; late: exit 2 with the cause
        if time.monotonic() >= t0 + seconds:
            raise TimeoutError("the run deadline passed before nvidia-smi started")
        rows, why = bounded_wait(30, t0 + seconds, lambda limit: mg.smi(query, timeout=limit))
        if why:
            raise TimeoutError("nvidia-smi " + ("cut at the run deadline" if why["cause"] == "deadline" else
                                                "past its own 30 s") +
                               ("" if why["late_s"] is None else
                                f" (its result seen only {why['late_s']:.3f} s after the end, not used)"))
        return rows

    try:
        known = mg.known_revisions(root)
        models = {s: mg.resolve_model(s, known, args.hf_home) for s in sorted(needed)}
        gpus = smi(["--query-gpu=index,name,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,"
                    "clocks.max.mem,persistence_mode", "--format=csv,noheader,nounits"])
        smi([f"--query-gpu={iso.Telemetry.FIELDS}", "--format=csv,noheader,nounits"])
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
    previous = s1 = None
    try:
        if args.previous_run:
            previous = previous_session(args, args.out, shas["base"], shas["candidate"], cur_sources, models, tables,
                                        shorts, root)
        if args.s1_run:
            s1 = s1_evidence(args, args.out, shas["base"], shas["candidate"], cur_sources)
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
    ctx.binary, ctx.model, ctx.tables, ctx.provenance = binaries, models, tables, provenance
    ctx.by_bus = {mg.bus(g[2]): int(g[0]) for g in gpus}
    ctx.gpu_name = gpus[0][1]
    ctx.deadline = t0 + seconds
    args.deadline_minutes = seconds / 60  # the effective value, for the messages of the W8 helpers
    ctx.deadline_basis = {"requested_minutes": requested, "lease_end_seconds_left": None if lease_s is None else
                          round(lease_s, 3), "effective_minutes": round(seconds / 60, 3)}
    ctx.deadline_hit, ctx.watch_foreign = False, True
    ctx.exports, ctx.export_runs, ctx.q8_diagnostics, ctx.results, ctx.ran_steps = {}, {}, {}, {}, []
    ctx.q4_objects, ctx.served_inputs, ctx.setup_stop = {}, None, None
    ctx.previous, ctx.s1 = previous, s1
    ctx.p0, ctx.p0_provenance = load_p0(root)
    ctx.bench_items = {s: mg.line_count(reference(ctx, s) / "bench.jsonl") for s in SHORTS}
    try:
        ctx.inputs900_summary = w8.write_900(root, out / "inputs-900", sorted(set(shorts)))
    except Fatal as e:
        print(f"preflight: 900 inputs: {e}", file=sys.stderr)
        return 2
    ctx.inputs900 = {s: out / "inputs-900" / f"inputs-900-{s}" for s in shorts}
    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w,"
                                  "clocks.max.sm_mhz,clocks.max.mem_mhz,persistence_mode\n" +
                                  "".join(",".join(g) + "\n" for g in gpus))
    fp = ctx.fp = fingerprint(rec, args, ctx, gpus)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("base", "candidate", "out", "hf_home", "basal_home", "plan_only", "previous_run", "s1_run",
                           "prepare_900", "w8_raw", "plan_json", "base_provenance", "candidate_provenance")}
    params = w8.scrubbed(params)
    params["env"] = {k: ctx.env[k] for k in ("BASAL_NO_UPDATE_CHECK", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE")}
    mg.write_json(out / "run.json", {"fingerprint": fp, "parameters": params, "previous": previous, "s1_evidence": s1,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})

    work = []
    if "correctness" in args.campaigns:
        work += [(f"correctness/{s}", fn) for s, fn in (("none", w8.step_none), ("q8", step_q8), ("w4", step_w4),
                                                        ("invalid", step_invalid), ("serve", step_serve))
                 if s in args.steps]
    work += [(c, fn) for c, fn in (("quality", campaign_quality), ("profile", campaign_profile),
                                   ("memory-mini", campaign_memory_mini), ("revalidate", campaign_revalidate))
             if c in args.campaigns]
    work += [(c, lambda rec, ctx, fam=fam: campaign_family(rec, ctx, fam)) for c, fam in FAMILY_CAMPAIGNS.items()
             if c in args.campaigns]
    started, stopped, complete = [], None, False
    try:
        if isinstance(ctx.setup_stop, Fatal):
            raise ctx.setup_stop
        blocking = binding_checks(rec, ctx)
        if blocking:
            raise Fatal(f"invalid binding before any GPU work: {blocking}")
        for label, fn in work:
            print(f"{time.strftime('%H:%M:%S')} {label}", flush=True)
            started.append(label)
            try:
                w8.guard(ctx, rec, label)
                ctx.ran_steps.append(label.split("/")[-1])  # started: a step runs at most once per run
                fn(rec, ctx)
            except Deadline:
                break
            except (StepFailed, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
                rec.check(label, "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
        for short, objects in ctx.q4_objects.items():
            rec.expect(f"q4_k {short}: one quantization object in every observed load", len(set(objects.values())) == 1,
                       f"{len(objects)} loads, {len(set(objects.values()))} distinct objects")
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
            hang = "suspected hung GPU" in stopped
            rec.check("run", "FAIL", iso.scrub(f"stopped: {stopped}; partial records kept, nothing later run"),
                      cls="timeout" if hang else "correctness")
        for label, _ in work:
            if label not in started:
                rec.check(label, "GAP", "not run: " + ("the run stopped" if stopped else "the deadline passed"))
        counts = {k: sum(c["status"] == k for c in rec.checks) for k in STATUSES}
        classes = {k: sum(c["status"] == "FAIL" and c.get("class") == k for c in rec.checks) for k in FAIL_CLASSES}
        status = 1 if counts["FAIL"] or stopped else 4 if counts["SIGNAL"] else \
            3 if counts["GAP"] or not complete else 0
        portable = iso.portable_checks(rec.checks)
        for p, c in zip(portable, rec.checks):
            p["class"] = c.get("class")
        mg.write_json(out / "summary.json", {
            "complete": complete, "stopped": iso.scrub(stopped), "exit_status": status, "counts": counts,
            "fail_classes": classes, "fingerprint": fp, "parameters": params, "exports": ctx.export_runs,
            "quantization_objects": w8.scrubbed(ctx.q8_diagnostics), "q4_objects": ctx.q4_objects,
            "results": w8.scrubbed(ctx.results), "checks": portable, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"DONE {counts} FAIL classes {classes} exit {status}" + (f" (stopped: {stopped})" if stopped else ""),
              flush=True)
    return status


if __name__ == "__main__":
    for sig in rp.STOP_SIGNALS:
        signal.signal(sig, w8.terminate)
    try:
        sys.exit(main())
    finally:
        for p in list(mg.LIVE):
            mg.stop(p, timeout=30)
