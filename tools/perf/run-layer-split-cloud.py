#!/usr/bin/env python3
"""Manual evidence run for one model split by layer ranges across two CUDA GPUs (`layer_split`, `--layer-split`) on
a rented Linux host with two identical NVIDIA GPUs. Not a test and not a CI gate: it collects raw evidence for a
report the operator writes (phase 05 of the multi-GPU plan, sessions L1, L2 and L3).

  python3 tools/perf/run-layer-split-cloud.py --help
  python3 tools/perf/run-layer-split-cloud.py --plan-only [--forecast F] [--prepared-900 DIR]  # counts, sources, time
  python3 tools/perf/run-layer-split-cloud.py --prepare-900 DIR                    # the 900-question inputs (no GPU)
  cd /work && python3 tools/perf/run-layer-split-cloud.py --campaign l1 --base BASE --expect-base-sha256 SHA \\
      --base-label TEXT --candidate CAND --expect-candidate-sha256 SHA --candidate-label TEXT --deadline-minutes M \\
      [--request-unix T] [--lease-end-unix T] --out out/layer-split-l1
  ... --campaign l2 --l1-run out/layer-split-l1 --out out/layer-split-l2      (l3 likewise)

--help, --plan-only and --prepare-900 read only local files: no cloud, SSH, GPU, CUDA library, binary, nvidia-smi,
network or secret. --plan-only checks an optional build record (--*-provenance) only as JSON, never the binary.

Binaries: --base is d42a400 rebuilt by the paired builder, --candidate d42a400 plus the phase-05 change, both of one
recipe. --expect-*-sha256 are checked before anything runs; --*-label are recorded as declared; an optional build
record (--*-provenance: binary_sha256, source_digest, embedded_assets, recipe) must name the binary's SHA-256 or the
run is refused; the candidate's source digest and embedded assets are compared with the synced tree before any GPU
work (different: FAIL and nothing runs; not given: a GAP of that binding). The synced sources must contain
crates/basal-gpu/src/split.rs (the candidate tree); the digest covers it (rp.source_files, every crates/*/src/**/*.rs).
Imported unchanged and SHA-256-checked before they are loaded: tools/perf/run-w4a16-cloud.py and through it the W8,
replicas, isolation and placement runners, and tools/bench/ab.py; tools/perf/cuda-reserve.py (the capacity helper,
run as its own process) must have the SHA-256 pinned here (RESERVE_HELPER_SHA256). Prerequisites as for the W4
runner: the tree synced to /work, weights in HF_HOME=/data/hf, CUDA libraries in BASAL_HOME=/data/basal, the bundled
GEMM tables of the card.

Sessions (one campaign per invocation; the parent audits the raw data between them):
  l1  units C1-C6 (--units narrows it only for a reviewed revalidation of whole units). Hard cap 85 min from the
      machine request (C5 alone: 35); --deadline-minutes (counted from the runner start) at most cap - 10 min.
  l2  default-path regression and split `none` throughput (110 min cap, --deadline-minutes <= 100).
  l3  quantized modes: single-GPU pairs and split descriptive (80 min cap, --deadline-minutes <= 70).
l2 and l3 need --l1-run (one or more raw l1 directories): the same runner id, base and candidate SHA-256, synced
sources and embedded assets, no FAIL of any class, not stopped, and together every unit C1-C6 completed; otherwise
the run is refused before any GPU work. A capacity GAP of C5 does not refuse l2/l3: it stays a GAP (REQ-006 open).
Identities recorded in both runs (models, weights, tables, inputs) must be equal (a difference stops the run before
GPU work); a runner or helper SHA-256 other than the l1 run's is a GAP of that binding (the parent's change-class
matrix decides what it invalidates), never a silent relabel.

Every comparison is fixed here before any rental; nothing is added, dropped or re-paired after a run.
C1  default path unchanged, mini and 4.5B, 44 + 34: base none tree and single on GPU 0, q8_0 and q4_k tree on GPU 0;
    candidate none tree and single on GPU 0 and tree on GPU 1, q8_0 and q4_k tree on GPU 0 and GPU 1 (22 exports);
    900 questions tree none, q8_0, q4_k, base and candidate (12). Each candidate export: bench.jsonl and
    systemone.jsonl byte-equal to the base of its mode and flags (GPU 1 against the base on GPU 0), `basal compare`
    without a difference, the manifest predicate (single-GPU variant), every q4_k load's quantization object
    against the P0 facts (W4 rule).
C2  split against the base on one GPU, same flags: 4.5B none 0:30,1:30 tree, single, budget, single with
    --state-cache-mb 512, tree --no-prefix-cache; 0:1,1:59 tree; 0:59,1:1 tree; 1:30,0:30 tree;
    CUDA_VISIBLE_DEVICES=1,0 with 0:30,1:30 tree; 4.5B q8_0 and q4_k 0:30,1:30 tree, single, tree --no-prefix-cache
    and 1:30,0:30 tree; mini 0:16,1:16 none, 1:16,0:16 none, 0:16,1:16 q8_0 and q4_k (tree); 4.5B 900 0:30,1:30 tree
    none, q8_0, q4_k (24 split exports; base adds the flags C1 has not: 7). Bytes, compare and the manifest predicate
    (split variant); memory of the process on both physical GPUs. Only when a split export differs: f32 900 exports
    of 4.5B and mini (the flags of the stored basal_rs_f32) and FP32 comparisons of each differing export as a
    diagnostic with the W8/W4 decision predicate; the FAIL stays.
C3  43 refused configurations through serve (YAML and CLI), export, benchmark, decide, profile and doctor: exit
    other than 0, the expected message, no download, GEMM table search, load or serving message, no GPU memory of
    the process (a process shorter than the 0.5 s sample is a GAP of that sample); mixed GPU types, Metal and Intel:
    GAP (static review only).
C4  HTTP, 4.5B, YAML with the explicit table, long_tokens 3000, state_cache_mb 512, metrics on: base none GPU 0
    (reference); candidate none GPU 0 (answers, /health, /v1/models bytes, metric families equal to the base);
    candidate none 0:30,1:30 (sequential, 16 clients x 2 rounds with the long requests first, the device label
    "0,1", SIGTERM with requests in flight: exit 0, every in-flight answer equal to the reference, no GPU memory
    left); base q8_0 and candidate q8_0 split, base q4_k and candidate q4_k split (sequential and 16 clients x 1);
    base mini GPU 1 (reference) and one candidate process with 4.5B 0:30,1:30 and mini device 1 under parallel
    traffic of both. Long requests: long-4000-q1 and long-4000-q5. Answers compared as status and canonical body
    without usage.latency_ms only; natural 400/422 are answers.
C5  capacity proof (4.5B none, 44 + 34 tree), the plan's method: cuda-reserve.py monitor, calibration exports
    (candidate single and 0:30,1:30), A = floor((U_single + max_g U_split_g) / 2) MiB and the window
    max_g U_split_g + 1 GiB <= A <= U_single - 1 GiB (else GAP, nothing more), reservation leaving A free on both
    GPUs (+-64 MiB, else GAP), candidate and base single attempts under it (real OOM expected, context checked,
    memory back to A +-64 MiB within 5 s), candidate split export (bitwise equal to the unlimited base and split
    exports) and server (answers equal to the C4 reference) under it, then GPU 1 at A1 = floor(U_split_1 / 2)
    (>= 1 GiB, else GAP): the split export must fail at stage 1 on device 1 with OOM. Before the next step under a
    reservation, the helper's samples after the previous process ended must show the reservation intact (the same
    5 s rule; otherwise a GAP of the control). Every outcome is classified before its check is recorded (a request
    OOM of the split server is capacity, an answer difference correctness); the final C5 record is PASS only when
    every step was judged without a FAIL or GAP, also on an early or exceptional end, and none is among the records
    the steps rest on: the own records of every export C5 runs or compares with (the cached C1/C2 references
    included), the C4 base reference server's and its session's completion, the calibration byte checks, the
    placement checks and the split server's shutdown (a FAIL keeps its class; a FAIL of a d42a400 reference, or its
    session not completed, is a GAP of C5; other units' records are not folded). Statuses fixed below.
C6  basal profile 4.5B 0:30,1:30 none, q8_0, q4_k at --batch 1 and 16 (--n 40): the stage_transfer section and
    stage_transfers present and finite (descriptive; no speed claim).
L2  4.5B none, explicit table: six fresh `basal benchmark --scenario all --concurrency 1,16` in the order A B C C B
    A (A base GPU 0, B candidate GPU 0, C candidate 0:30,1:30) and `ab.py --rounds 4` of `basal bench --reference
    reports/reference-basal-1.5-4.5B-fp32 --dtype f16` for A/B and B/C. Regression pairs: benchmark A0/B1, B4/A5
    and the four ab.py pairs A/B; the 21 benchmark fields (3 phases x 7) and the 4 bench fields (lat2_ms,
    lat2_p95_ms, dec_s, per-question median ratio over items 5..43). A field with B worse in every scheduled pair,
    or worse in some and nowhere not worse but incomplete, starts exactly one confirmation round of only that
    command (benchmark A B B A or ab.py rounds 4) and `basal profile` A and B at batch 1; worse in every pair of both
    rounds is a persistent SIGNAL. No threshold. Trade-off pairs B1/C2, C3/B4 and ab.py B/C: descriptive.
L3  4.5B: ab.py rounds 4 q8_0 A/B and q4_k A/B (base vs candidate on GPU 0) with the same rule on the 4 bench
    fields (confirmation ab.py rounds 4 and profiles A, B at batch 1); descriptive ab.py rounds 4 candidate single
    vs 0:30,1:30 for q8_0 and q4_k; one benchmark of the split candidate in q8_0 and in q4_k. The new candidate's
    single-GPU q8_0/q4_k HTTP benchmark is not measured (a GAP by the plan; S3 described only d42a400).

Manifest predicate (candidate against the base of the same model, mode and flags): the key set of every level
(top, run, backend, nested objects) exactly the pinned d42a400 schema plus the declared split keys; bench_s,
total_s and run.load_s present, finite and > 0 (measured, never compared); tool, inputs, model, model_dir,
bench_batching, host_math, calibration, request_batching, run.dtype/quantization/readout/kernels/prefix_cache and
every backend field but device deep-equal (canonical JSON, number types included); backend.device the base text
with the process ordinal (single) or the pinned split text; split only: run.layer_split [{device, layers}] as
given, backend.layer_split computed from the layout, config.json and evidence_head.pt, backend.stage_transfer the
pinned text. Any other difference is a FAIL of class correctness. Benchmark, profile and bench reports are
descriptive: their split-only fields must be present and finite with a split and absent without.

Statuses PASS, FAIL (class correctness, capacity or timeout), GAP (missing evidence, also anything not run) and
SIGNAL (analysis needed). Exit status: 0 all PASS, 1 a FAIL, 4 no FAIL but a SIGNAL, 3 only GAPs, 2 preflight or
usage error (also: an engine-owned text still pending in ENGINE_CONTRACT, or a bound text or report field missing from
the synced candidate source, ENGINE_FRAGMENTS); a run stopped by a signal exits 128 + signal, never 0.

Deadlines: the run's absolute end is the earliest of start + --deadline-minutes, --lease-end-unix and, with
--request-unix, the machine request + the campaign's hard cap - the 600 s collection reserve; a deadline at or
before the start is refused. Every command, HTTP request, graceful shutdown, readiness wait, helper action and
nvidia-smi / --version probe has its own limit and ends at min(its own end, run end); nothing starts after the run
end; what was cut or not run is a GAP (reason deadline). A GPU command, request, shutdown, helper action or probe
past its own limit is a FAIL of class timeout and stops all later GPU work (suspected hung GPU). Children are
started in their own sessions and registered (LIVE); an interruption, error or the end stops only those exact
process groups (SIGTERM, SIGKILL after 60 s). The reservation helper is registered by its process id (private
helper-processes.jsonl), excluded from the foreign-compute observation and from the idle-GPU wait; any other
compute process a command's watch observes (every 2 s) is a GAP of that command's evidence, its values and
comparisons kept (L2/L3 runs: the W8 disturbance rule). HTTP servers are observed for outside processes only by the
idle wait before their start; their request intervals are unobserved (not claimed free of other processes). The
runner never rents, extends or deletes machines.

Recorded under --out (must not exist; raw, contains local paths: keep outside Git, e.g. .cache/cloud): checks.txt,
commands.jsonl, run.json, summary.json (scrubbed), gpus.csv, source-files.sha256, helper-processes.jsonl,
inputs-900/, exports/, compare/, invalid/, serve/, capacity/, perf/.
"""
import argparse
import copy
import datetime
import decimal
import hashlib
import importlib.util
import json
import math
import os
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
RUNNER_ID = "run-layer-split-cloud.py"
W4_RUNNER = HERE.with_name("run-w4a16-cloud.py")
RESERVE_HELPER = HERE.with_name("cuda-reserve.py")
# the helpers this runner was written against, relative to the repository root (checked before loading)
HELPERS_SHA256 = {
    "tools/perf/run-w4a16-cloud.py": "97e9df05d8b07e7a72c0e4011db9555f45ea88888a5a71b3e6d27837d8ffc917",
    "tools/perf/run-w8a16-cloud.py": "92b47c9e341f47fe7a538cbd42a0f0e152ae7199720bda2b5d15f79f83d945ea",
    "tools/perf/run-replicas-cloud.py": "cb22ac7a498522374c4ea0f09cbbb6317a657acf70f05299f182552c332963dc",
    "tools/perf/run-isolation-cloud.py": "c460aee1b21da0e3a8418f9d7c19fc7b4cf5e1d84ebdfe43f98456fb9ca0f582",
    "tools/perf/run-multigpu-cloud.py": "1ddd4910c6a1ff79c297e42a70efc5cae975289147f911700e20e403e2a351d8",
    "tools/bench/ab.py": "17b188dcdf5a6986f4930c63994046111ba7ce80fea89bde820c97d425ac7bec",
}
RESERVE_HELPER_PATH = "tools/perf/cuda-reserve.py"  # run as its own process; this exact version only
RESERVE_HELPER_SHA256 = "ebea01edcd9ff59320fd31c9c22fb8863141e3f19c49088230eb9d9ac04cf090"
SPLIT_SOURCE = "crates/basal-gpu/src/split.rs"

# --------------------------------------------------------------------------- the shared engine/runner contract

# backend.stage_transfer of every split export: the routine literal pinned by the coordinator with the engine
STAGE_TRANSFER = ("Sequential device-to-device API copy after source drain and destination initialization; "
                  "destination and source PTDS completion are confirmed. Boundary time includes allocation, "
                  "initialization, synchronization, source release and context binding; physical copy route is not "
                  "established.")
SINGLE_DEVICE = re.compile(r"^(NVIDIA GPU through candle Device::new_cuda\()(\d+)(\))$")


def split_device_text(stages):
    a, b = (d for d, _ in stages)
    return f"NVIDIA GPUs through candle Device::new_cuda({a}) and Device::new_cuda({b}), consecutive layer ranges"


# The d42a400 manifest schema (crates/basal-cli/src/export.rs manifest, main.rs Export `run`, basal-gpu describe):
# the key set of every level. The split adds exactly SPLIT_RUN_KEYS / SPLIT_BACKEND_KEYS.
MANIFEST_KEYS = {"tool", "inputs", "model", "model_dir", "backend", "bench_batching", "host_math", "calibration",
                 "request_batching", "bench_s", "total_s", "run"}
RUN_KEYS = {"dtype", "quantization", "readout", "kernels", "prefix_cache", "load_s"}
BACKEND_KEYS = {"backend", "device", "tensor_library", "weights", "activations", "attention", "kernels", "readout",
                "padding", "attention_kernel", "gemm", "prefix_kv", "state_cache_bytes", "last_layer"}
SPLIT_RUN_KEYS = {"layer_split"}
SPLIT_BACKEND_KEYS = {"layer_split", "stage_transfer"}
MEASURED = (("bench_s",), ("total_s",), ("run", "load_s"))

# Engine-owned texts the Ready plan names without spelling them out, bound to the phase-05 candidate source as read
# for this runner (the coordinator has not frozen them yet; ENGINE_FRAGMENTS refuses a run whose synced source no
# longer carries them). A value None is pending: a run whose units use it is refused (exit 2).
# key -> (units or sessions using it, value, what it is).
ENGINE_CONTRACT = {
    "c3.too_few_stages": (("C3",), "a layer split needs at least two stages, got 1", "LayerSplitError::TooFewStages"),
    "c3.zero_layers": (("C3",), "layer split: stage 0 on device 0 has no layers", "LayerSplitError::ZeroLayers"),
    "c3.duplicate_device": (("C3",), "layer split: device 0 is given twice", "LayerSplitError::DuplicateDevice"),
    "c3.dtype_bf16": (("C3",), "layer split: needs dtype f16, not bf16", "LayerSplitError::Unsupported (dtype)"),
    "c3.dtype_f32": (("C3",), "layer split: needs dtype f16, not f32", "LayerSplitError::Unsupported (dtype)"),
    "c3.kernels": (("C3",), "layer split: needs kernels fused, not candle", "LayerSplitError::Unsupported (kernels)"),
    "c3.doctor_estimate": (("C3",), r"layer split 0:30,1:30, weights device 0 ~\d+\.\d GB, device 1 ~\d+\.\d GB "
                                    r"\(estimate\)", "doctor: the per-device weight estimate of a valid split (regex)"),
    "c5.model_context": (("C5",), "loading {name}", "the model's context in a load error chain ({name}: basal.json)"),
}
# Static fragments of the synced candidate source that carry the bound texts and report fields: (path, fragment).
ENGINE_FRAGMENTS = (
    ("crates/basal-gpu/src/split.rs", "a layer split needs at least two stages, got {stages}"),
    ("crates/basal-gpu/src/split.rs", "layer split: stage {stage} on device {device} has no layers"),
    ("crates/basal-gpu/src/split.rs", "layer split: device {device} is given twice"),
    ("crates/basal-gpu/src/split.rs", "layer split covers {covered} of {layers} layers"),
    ("crates/basal-gpu/src/split.rs", "needs dtype f16, not {}"),
    ("crates/basal-gpu/src/split.rs", "needs kernels fused, not candle"),
    ("crates/basal-gpu/src/split.rs", '"layer split: {reason}"'),
    ("crates/basal-cli/src/config.rs", "give either `device` or `layer_split`"),
    ("crates/basal-cli/src/config.rs", "replicas of a layer-split model are not supported"),
    ("crates/basal-gpu/src/lib.rs", "CUDA device {device} does not exist"),
    ("crates/basal-cli/src/doctor.rs", "layer split {split}, weights {} (estimate)"),
    ("crates/basal-cli/src/doctor.rs", "device {d} ~{:.1} GB"),
    ("crates/basal-cli/src/main.rs", 'format!("loading {}", manifest.name)'),
    ("crates/basal-cli/src/main.rs", 'format!("loading {} with layer split {split}", manifest.name)'),
    ("crates/basal-gpu/src/lib.rs", "layer {l} of stage {s} on {kind} device {device}"),
    ("crates/basal-gpu/src/lib.rs", "stage {s} (layers {}–{})"),
    ("crates/basal-gpu/src/lib.rs", STAGE_TRANSFER),
    ("crates/basal-gpu/src/lib.rs", "NVIDIA GPUs through candle {list}, consecutive layer ranges"),
    ("crates/basal-gpu/src/lib.rs", '"first_layer": st.layers.start'),
    ("crates/basal-gpu/src/lib.rs", '"last_layer": st.layers.end.saturating_sub(1), "holds": holds'),
    ("crates/basal-gpu/src/lib.rs", 'holds.extend(["norm", "lm_head"])'),
    ("crates/basal-gpu/src/lib.rs", 'holds.push("evidence_head")'),
    ("crates/basal-gpu/src/lib.rs", '"from_stage": i, "to_stage": i + 1, "from_device": b.from_device, "to_device"'),
    ("crates/basal-gpu/src/lib.rs", '"copies": b.copies.load(Ordering::Relaxed), "bytes": b.bytes.load('),
    ("crates/basal-gpu/src/lib.rs", '"transfer_ms": ms(&b.transfer_ns), "gate_wait_ms": ms(&b.gate_wait_ns)'),
    ("crates/basal-cli/src/main.rs", 'run["layer_split"] = serde_json::to_value(split.stages())'),
    ("crates/basal-cli/src/main.rs", 'out["stage_transfers"] = t.report()'),
    ("crates/basal-cli/src/main.rs", 'r["gpu_probe_tflops_per_device"]'),
    ("crates/basal-cli/src/benchmark.rs", 'report["layer_split"] = json!({"stages": split.stages(), "devices": []})'),
    ("crates/basal-cli/src/benchmark.rs", 'report["layer_split"]["transfers"] = t.report()'),
    ("crates/basal-cli/src/benchmark.rs", 'report["settings"]["layer_split"] = json!(split.stages())'),
    ("crates/basal-cli/src/benchmark.rs", 'report["layer_split"]["devices"][i]["memory_pool"]'),
    ("crates/basal-cli/src/benchmark/resources.rs", 'window["per_device"]'),
    ("crates/basal-cli/src/benchmark/resources.rs", 'summary["per_device"]'),
)
TRANSFER_KEYS = {"from_stage", "to_stage", "from_device", "to_device", "copies", "bytes", "transfer_ms", "gate_wait_ms",
                 "scope"}
# Fixed texts: given by the plan, by existing source (gpu_identity) or by clap / serde for a value or field error.
C3_TEXT = {
    "coverage-59": "covers 59 of 60 layers", "coverage-61": "covers 61 of 60 layers",
    "device-and-split": "give either `device` or `layer_split`",
    "replicas-and-split": "replicas of a layer-split model are not supported",
    "absent-device-2": "CUDA device 2 does not exist", "absent-device-1": "CUDA device 1 does not exist",
    "cli-conflict": "cannot be used with", "cli-parse": "invalid value",
    "yaml-unknown-field": "unknown field `layer`", "yaml-missing-layers": "missing field `layers`",
    "too-few-stages": "c3.too_few_stages", "zero-layers": "c3.zero_layers", "duplicate-device": "c3.duplicate_device",
    "dtype-bf16": "c3.dtype_bf16", "dtype-f32": "c3.dtype_f32", "kernels": "c3.kernels",
}


def contract_value(key):
    return ENGINE_CONTRACT[key][1]


def needle(kind):
    text = C3_TEXT[kind]
    return contract_value(text) if text in ENGINE_CONTRACT else text


def pending_contract(scopes):
    return [k for k, (used, v, _) in ENGINE_CONTRACT.items() if set(used) & set(scopes) and v is None]


def run_scopes(campaign, units):
    return list(units) if campaign == "l1" else [campaign.upper()]


def missing_fragments(root):
    """ENGINE_FRAGMENTS absent from the synced candidate source: the bound texts or fields drifted (a run is
    refused until they are bound again in a reviewed runner revision)."""
    texts, out = {}, []
    for path, fragment in ENGINE_FRAGMENTS:
        if path not in texts:
            f = root / path
            texts[path] = f.read_text(errors="replace") if f.is_file() else ""
        if fragment not in texts[path]:
            out.append(f"{path}: {fragment[:60]}")
    return out


# --------------------------------------------------------------------------- campaigns, layouts, fixed inventories

CAMPAIGNS = ("l1", "l2", "l3")
UNITS = ("C1", "C2", "C3", "C4", "C5", "C6")
CAP_MINUTES = {"l1": 85, "l2": 110, "l3": 80, "c5-only": 35}  # hard machine life from the request (plan)
RESERVE_S = 600  # final collection/verification/delete reserve (operations contract)
STATUSES = ("PASS", "FAIL", "GAP", "SIGNAL")
FAIL_CLASSES = ("correctness", "capacity", "timeout")
SHORTS = ("mini", "4.5B")
LAYERS = {"mini": 32, "4.5B": 60}  # num_hidden_layers, checked against config.json before any GPU work
Q8, Q4 = "q8_0", "q4_k"
QTAG = {None: "none", Q8: "q8", Q4: "q4"}
ABSENT = "Remek/basal-layer-split-absent"  # never in a cache: validation must fail before any download
L45, L45_REV, L45_FIRST, L45_LAST = ((0, 30), (1, 30)), ((1, 30), (0, 30)), ((0, 1), (1, 59)), ((0, 59), (1, 1))
LMINI, LMINI_REV = ((0, 16), (1, 16)), ((1, 16), (0, 16))
LONG_IDS = ("long:long-4000-q1", "long:long-4000-q5")  # the plan's long states of the HTTP campaign
HTTP_LONG_TOKENS, HTTP_STATE_CACHE_MB = 3000, 512
MIB, GIB_MIB = 1 << 20, 1024
C5_TOLERANCE_MIB = 64
C5_RETURN_S = 5.0
MONITOR_MAX_GAP_S = 0.5  # the helper samples every 50 ms; a longer hole in a judged window is a GAP of the sample
CALIBRATION_IDLE_S = 2.0
REGRESSION_PAIRS = (("A0/B1", 0, 1), ("B4/A5", 5, 4))  # (name, run of role A, run of role B), fixed in advance
TRADEOFF_PAIRS = (("B1/C2", 1, 2), ("C3/B4", 4, 3))  # (name, run of role B, run of role C), descriptive
FAMILY_ORDER = "ABCCBA"


def layout_text(stages):
    return ",".join(f"{d}:{n}" for d, n in stages)


def layout_slug(stages):
    return "split-" + "-".join(f"{d}.{n}" for d, n in stages)


def physical_of(device, visible):
    """The physical (nvidia-smi, PCI order) index of process ordinal `device` under CUDA_VISIBLE_DEVICES `visible`."""
    return int(visible.split(",")[device]) if visible else device


def xspec(unit, short, binary, quant=None, mode="tree", device=0, layout=None, visible=None, state_cache=0,
          inputs="ref"):
    """One export of the fixed inventory; its name identifies it (and caches it within a run)."""
    place = layout_slug(layout) if layout else f"d{device}"
    btag = "cand" if binary == "candidate" else binary
    extra = f"-sc{state_cache}" if state_cache else ""
    vis = f"-cvd{visible.replace(',', '')}" if visible else ""
    kind = "900" if inputs == "900" else mode
    name = f"{short}-{kind}{extra}-{QTAG[quant]}-{btag}-{place}{vis}"
    return {"name": name, "unit": unit, "short": short, "binary": binary, "quant": quant, "mode": mode,
            "device": None if layout else device, "layout": layout, "visible": visible, "state_cache": state_cache,
            "inputs": inputs}


def c1_pairs():
    """[(candidate spec, base spec)] of C1: the default path against d42a400 (44 + 34 and 900)."""
    out = []
    for s in SHORTS:
        for quant, mode, device in ((None, "tree", 0), (None, "single", 0), (None, "tree", 1), (Q8, "tree", 0),
                                    (Q8, "tree", 1), (Q4, "tree", 0), (Q4, "tree", 1)):
            out.append((xspec("C1", s, "candidate", quant, mode, device), xspec("C1", s, "base", quant, mode, 0)))
    for s in SHORTS:
        for quant in (None, Q8, Q4):
            out.append((xspec("C1", s, "candidate", quant, inputs="900"),
                        xspec("C1", s, "base", quant, inputs="900")))
    return out


def c2_pairs():
    """[(split candidate spec, base spec on GPU 0 of the same mode and flags)] of C2."""
    out = []

    def add(short, quant, mode, layout, state_cache=0, visible=None, inputs="ref"):
        out.append((xspec("C2", short, "candidate", quant, mode, layout=layout, visible=visible,
                          state_cache=state_cache, inputs=inputs),
                    xspec("C2", short, "base", quant, mode, 0, state_cache=state_cache, inputs=inputs)))

    for mode in ("tree", "single", "budget"):
        add("4.5B", None, mode, L45)
    add("4.5B", None, "single", L45, state_cache=512)
    add("4.5B", None, "tree-noprefix", L45)
    for layout in (L45_FIRST, L45_LAST, L45_REV):
        add("4.5B", None, "tree", layout)
    add("4.5B", None, "tree", L45, visible="1,0")
    for quant in (Q8, Q4):
        for mode in ("tree", "single", "tree-noprefix"):
            add("4.5B", quant, mode, L45)
        add("4.5B", quant, "tree", L45_REV)
    add("mini", None, "tree", LMINI)
    add("mini", None, "tree", LMINI_REV)
    add("mini", Q8, "tree", LMINI)
    add("mini", Q4, "tree", LMINI)
    for quant in (None, Q8, Q4):
        add("4.5B", quant, "tree", L45, inputs="900")
    return out


def c3_cases(d, cfg, ref, request, model45):
    """(label, arguments after the binary, CUDA_VISIBLE_DEVICES or None, expected kind) of every refused
    configuration; `cfg(label, **entry)` writes a YAML whose one model entry has `entry`, `model45` is the local 4.5B
    directory (its config.json is read by the coverage check), `ref` the 44 + 34 inputs, `request` a decide body."""
    split = [{"device": 0, "layers": 30}, {"device": 1, "layers": 30}]

    def ls(text):
        return [{"device": int(x.split(":")[0]), "layers": int(x.split(":")[1])} for x in text.split(",")]

    absent, real = {"repo": ABSENT}, {"dir": model45}
    yaml = (("one-stage", absent, {"layer_split": ls("0:60")}, None, "too-few-stages"),
            ("zero", absent, {"layer_split": ls("0:0,1:60")}, None, "zero-layers"),
            ("duplicate", absent, {"layer_split": ls("0:30,0:30")}, None, "duplicate-device"),
            ("coverage-59", real, {"layer_split": ls("0:30,1:29")}, None, "coverage-59"),
            ("coverage-61", real, {"layer_split": ls("0:30,1:31")}, None, "coverage-61"),
            ("absent-device-2", absent, {"layer_split": ls("0:30,2:30")}, None, "absent-device-2"),
            ("device-0", absent, {"layer_split": split, "device": 0}, None, "device-and-split"),
            ("device-1", absent, {"layer_split": split, "device": 1}, None, "device-and-split"),
            ("replicas", absent, {"layer_split": split, "replica_devices": [0, 1]}, None, "replicas-and-split"),
            ("bf16", absent, {"layer_split": split, "dtype": "bf16"}, None, "dtype-bf16"),
            ("f32", absent, {"layer_split": split, "dtype": "f32"}, None, "dtype-f32"),
            ("unknown-field", absent, {"layer_split": [{"device": 0, "layer": 30}, {"device": 1, "layers": 30}]},
             None, "yaml-unknown-field"),
            ("missing-layers", absent, {"layer_split": [{"device": 0}, {"device": 1, "layers": 30}]}, None,
             "yaml-missing-layers"),
            ("visible0", absent, {"layer_split": split}, "0", "absent-device-1"))
    cases = [(f"serve-yaml-{label}", ["serve", "--config", cfg(f"yaml-{label}", **dict(base, **fields))], vis, kind)
             for label, base, fields, vis, kind in yaml]
    cases.append(("serve-cli-with-config", ["serve", "--config", cfg("plain", **absent), "--layer-split", "0:30,1:30"],
                  None, "cli-conflict"))
    cases.append(("serve-cli-with-device", ["serve", "--model", ABSENT, "--device", "0", "--layer-split", "0:30,1:30"],
                  None, "cli-conflict"))
    cases.append(("serve-cli-with-replicas", ["serve", "--model", ABSENT, "--replica-devices", "0,1", "--layer-split",
                                              "0:30,1:30"], None, "cli-conflict"))
    for label, text in (("semicolon", "0:30;1:30"), ("missing", "0:30,1:"), ("letter", "a:30,1:30"),
                        ("negative", "0:-1,1:61"), ("empty", "")):
        cases.append((f"serve-cli-parse-{label}", ["serve", "--model", ABSENT, "--layer-split", text], None,
                      "cli-parse"))
    cmd = {"export": lambda label, m: ["export", "--model", m, "--inputs", ref, "--out", d / f"{label}-out"],
           "benchmark": lambda label, m: ["benchmark", "--model", m, "--out", d / f"{label}.json"],
           "decide": lambda label, m: ["decide", "--model", m, "--request", request],
           "profile": lambda label, m: ["profile", "--model", m, "--reference", ref]}
    other = (("export", "zero", ABSENT, ["--layer-split", "0:0,1:60"], None, "zero-layers"),
             ("export", "coverage-59", model45, ["--layer-split", "0:30,1:29"], None, "coverage-59"),
             ("export", "candle", ABSENT, ["--layer-split", "0:30,1:30", "--kernels", "candle"], None, "kernels"),
             ("export", "bf16", ABSENT, ["--layer-split", "0:30,1:30", "--dtype", "bf16"], None, "dtype-bf16"),
             ("export", "absent-device-2", ABSENT, ["--layer-split", "0:30,2:30"], None, "absent-device-2"),
             ("export", "with-device", ABSENT, ["--layer-split", "0:30,1:30", "--device", "1"], None, "cli-conflict"),
             ("benchmark", "duplicate", ABSENT, ["--layer-split", "0:30,0:30"], None, "duplicate-device"),
             ("benchmark", "coverage-61", model45, ["--layer-split", "0:30,1:31"], None, "coverage-61"),
             ("benchmark", "f32", ABSENT, ["--layer-split", "0:30,1:30", "--dtype", "f32"], None, "dtype-f32"),
             ("benchmark", "visible0", ABSENT, ["--layer-split", "0:30,1:30"], "0", "absent-device-1"),
             ("decide", "one-stage", ABSENT, ["--layer-split", "0:60"], None, "too-few-stages"),
             ("decide", "parse-letter", ABSENT, ["--layer-split", "a:30,1:30"], None, "cli-parse"),
             ("decide", "candle", ABSENT, ["--layer-split", "0:30,1:30", "--kernels", "candle"], None, "kernels"),
             ("profile", "coverage-59", model45, ["--layer-split", "0:30,1:29"], None, "coverage-59"),
             ("profile", "with-device", ABSENT, ["--layer-split", "0:30,1:30", "--device", "0"], None,
              "cli-conflict"),
             ("profile", "bf16", ABSENT, ["--layer-split", "0:30,1:30", "--dtype", "bf16"], None, "dtype-bf16"))
    for command, label, model, flags, vis, kind in other:
        name = f"{command}-{label}"
        cases.append((name, [*cmd[command](name, model), *flags], vis, kind))
    return cases


def doctor_cases(model45, table):
    """(label, YAML entries, expected failure kind or None for the valid entry)."""
    split = [{"device": 0, "layers": 30}, {"device": 1, "layers": 30}]
    real = {"dir": model45, "dtype": "f16", "gemm_table": table}
    return (("doctor-device", [{"repo": ABSENT, "layer_split": split, "device": 0}], "device-and-split"),
            ("doctor-replicas", [{"repo": ABSENT, "layer_split": split, "replica_devices": [0, 1]}],
             "replicas-and-split"),
            ("doctor-duplicate", [{"repo": ABSENT, "layer_split": [{"device": 0, "layers": 30},
                                                                   {"device": 0, "layers": 30}]}],
             "duplicate-device"),
            ("doctor-coverage-59", [dict(real, layer_split=[{"device": 0, "layers": 30},
                                                            {"device": 1, "layers": 29}])], "coverage-59"),
            ("doctor-valid", [dict(real, layer_split=split)], None))


def c4_servers():
    """(label, binary, model entries [(short, placement)], quantization, reference key, passes) of C4, in order.
    placement: ("device", n) or ("split", layout); passes: (clients, rounds) of the concurrent passes after the
    sequential one."""
    return (("none-base-d0", "base", [("4.5B", ("device", 0))], None, None, ()),
            ("none-cand-d0", "candidate", [("4.5B", ("device", 0))], None, "none-base-d0", ()),
            ("none-cand-split", "candidate", [("4.5B", ("split", L45))], None, "none-base-d0", ((16, 2),)),
            ("q8-base-d0", "base", [("4.5B", ("device", 0))], Q8, None, ()),
            ("q8-cand-split", "candidate", [("4.5B", ("split", L45))], Q8, "q8-base-d0", ((16, 1),)),
            ("q4-base-d0", "base", [("4.5B", ("device", 0))], Q4, None, ()),
            ("q4-cand-split", "candidate", [("4.5B", ("split", L45))], Q4, "q4-base-d0", ((16, 1),)),
            ("mini-base-d1", "base", [("mini", ("device", 1))], None, None, ()),
            ("mixed-cand", "candidate", [("4.5B", ("split", L45)), ("mini", ("device", 1))], None, "mixed", ()))


C6_PROFILES = tuple((quant, batch) for quant in (None, Q8, Q4) for batch in (1, 16))


# --------------------------------------------------------------------------- helper import (lazy: --help needs none)

w4 = w8 = rp = iso = mg = None
StepFailed = Fatal = Deadline = None
Served = None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def helpers_state(root):
    """{relative path: (actual SHA-256 or None, expected)} of the imported helpers."""
    return {p: (file_sha256(root / p) if (root / p).is_file() else None, want) for p, want in HELPERS_SHA256.items()}


def import_helpers(root):
    """Checks every imported helper's SHA-256, then loads the W4 runner (definitions only; it loads W8, replicas,
    isolation and placement, one LIVE registry). Exits 2 on another version."""
    global w4, w8, rp, iso, mg, StepFailed, Fatal, Deadline, Served
    changed = {p: a for p, (a, want) in helpers_state(root).items() if a != want}
    if changed:
        print(f"helpers other than the ones this runner was written against (sync tools/perf, tools/bench): "
              f"{sorted(changed)}", file=sys.stderr)
        sys.exit(2)
    if root.resolve() / "tools/perf" != HERE.parent:
        print("run in the repository root that holds this runner (/work)", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("basal_w4a16_runner", W4_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    w4 = module
    w8, rp, iso, mg = w4.w8, w4.rp, w4.iso, w4.mg
    StepFailed, Fatal, Deadline = w4.StepFailed, w4.Fatal, w4.Deadline
    Served = make_served(w4.Served)


class Record:
    """The checks (four statuses; a FAIL carries a class: explicit, else "timeout" for the helpers' own-timeout
    messages, else "correctness") and commands of a run; the files are appended at once."""

    def __init__(self, out):
        self.out, self.checks, self.lock = out, [], threading.Lock()

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
                f.write(f"{tag} {name}{': ' + str(detail) if detail else ''}\n")
        print(f"{time.strftime('%H:%M:%S')} {tag} {name}{': ' + str(detail) if detail else ''}", flush=True)

    def expect(self, name, ok, detail="", cls=None):
        self.check(name, "PASS" if ok else "FAIL", detail, cls)
        return ok

    def command(self, entry):
        with self.lock, open(self.out / "commands.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- small helpers


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def finite_positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def finite_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def diff_paths(a, b, path="", out=None, limit=40):
    """Where two JSON values differ (keys on one side only, other types, other values; canonical JSON equality, so
    1 and 1.0 differ); at most `limit` paths."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append(f"{p}: only in {'expected' if k in a else 'candidate'}")
            else:
                diff_paths(a[k], b[k], p, out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, limit)
    elif canonical(a) != canonical(b):
        out.append(f"{path}: {canonical(a)[:80]} != {canonical(b)[:80]}")
    return out


def get_path(v, path):
    for k in path:
        if not isinstance(v, dict) or k not in v:
            return None, False
        v = v[k]
    return v, True


def scrub(text):
    return iso.scrub(text) if iso else text


# --------------------------------------------------------------------------- manifest predicate


def expected_layer_split(stages, layers, evidence):
    """backend.layer_split of a split from the layout, num_hidden_layers and the evidence head: ranges by layer
    counts, last_layer inclusive, holds embed on stage 0 and norm, lm_head (and evidence_head) on the last stage."""
    out, first = [], 0
    for s, (device, n) in enumerate(stages):
        holds = []
        if s == 0:
            holds.append("embed")
        if s == len(stages) - 1:
            holds += ["norm", "lm_head"] + (["evidence_head"] if evidence else [])
        out.append({"stage": s, "device": device, "first_layer": first, "last_layer": first + n - 1, "holds": holds})
        first += n
    if first != layers:
        raise ValueError(f"layout {layout_text(stages)} covers {first} of {layers} layers")
    return out


def base_schema_problems(base, quant):
    """The base manifest against the pinned d42a400 key sets (it is the reference: any deviation means the
    predicate cannot hold)."""
    problems = []
    if set(base) != MANIFEST_KEYS:
        problems.append(f"top keys {sorted(set(base) ^ MANIFEST_KEYS)}")
    if not isinstance(base.get("run"), dict) or set(base["run"]) != RUN_KEYS:
        problems.append(f"run keys {sorted(set(base.get('run') or {}) ^ RUN_KEYS)}")
    want = BACKEND_KEYS | ({"quantization"} if quant else set())
    if not isinstance(base.get("backend"), dict) or set(base["backend"]) != want:
        problems.append(f"backend keys {sorted(set(base.get('backend') or {}) ^ want)}")
    mb = SINGLE_DEVICE.match(str((base.get("backend") or {}).get("device")))
    if not mb or mb.group(2) != "0":
        problems.append(f"base device {(base.get('backend') or {}).get('device')!r} is not GPU 0's text")
    return problems


def manifest_predicate(base, cand, spec, ctx):
    """(problems, detail) of the candidate manifest against the base of the same model, mode and flags: the exact
    expected manifest is the base's with the three measured values taken from the candidate (after checking them)
    and the declared transformations applied; then canonical equality of the whole object."""
    problems = base_schema_problems(base, spec["quant"])
    for path in MEASURED:
        for side, m in (("base", base), ("candidate", cand)):
            v, present = get_path(m, path)
            if not present or not finite_positive(v):
                problems.append(f"{side} {'.'.join(path)} {v!r}: not a present finite positive number")
    if problems:
        return problems, "; ".join(problems[:6])
    expected = copy.deepcopy(base)
    for path in MEASURED:
        target = expected
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = get_path(cand, path)[0]
    if spec["layout"]:
        stages = spec["layout"]
        m = ctx.model[spec["short"]]
        expected["backend"]["device"] = split_device_text(stages)
        expected["run"]["layer_split"] = [{"device": d, "layers": n} for d, n in stages]
        expected["backend"]["layer_split"] = expected_layer_split(stages, m["layers"], m["evidence"])
        expected["backend"]["stage_transfer"] = STAGE_TRANSFER
    else:
        mb = SINGLE_DEVICE.match(base["backend"]["device"])
        expected["backend"]["device"] = f"{mb.group(1)}{spec['device']}{mb.group(3)}"
    want_run = RUN_KEYS | (SPLIT_RUN_KEYS if spec["layout"] else set())

    def keys(v):
        return set(v) if isinstance(v, dict) else set()

    for level, want, got in (("top", MANIFEST_KEYS, keys(cand)), ("run", want_run, keys(cand.get("run"))),
                             ("backend", set(expected["backend"]), keys(cand.get("backend")))):
        if got != want:
            problems.append(f"{level} keys: missing {sorted(want - got)}, extra {sorted(got - want)}")
    if canonical(expected) != canonical(cand):
        problems += diff_paths(expected, cand)
    variant = "split: device text, run.layer_split, backend.layer_split, stage_transfer" if spec["layout"] else \
        f"single: device ordinal {spec['device']}"
    detail = "; ".join(problems[:8]) if problems else (
        f"keys exact, measured bench_s/total_s/run.load_s finite and positive, every other field as expected "
        f"({variant})")
    return problems, detail


# --------------------------------------------------------------------------- exports


MODES_EXTRA = {"tree": ["--batching", "tree"], "single": ["--batching", "single"], "budget": ["--batching", "budget"],
               "tree-noprefix": ["--batching", "tree", "--no-prefix-cache"]}


def export_argv(ctx, spec, out):
    m = ctx.model[spec["short"]]
    argv = [ctx.binary[spec["binary"]], "--color", "never", "export", "--model", m["dir"]]
    if spec.get("dtype"):
        argv += ["--dtype", spec["dtype"]]
    if spec.get("table", True):
        argv += ["--gemm-table", ctx.tables[spec["short"]]]
    argv += MODES_EXTRA[spec["mode"]] if spec["mode"] else []
    if spec["quant"]:
        argv += ["--quantization", spec["quant"]]
    if spec["layout"]:
        argv += ["--layer-split", layout_text(spec["layout"])]
    elif spec["device"] is not None:
        argv += ["--device", str(spec["device"])]
    if spec["state_cache"]:
        argv += ["--state-cache-mb", str(spec["state_cache"])]
    inputs = ctx.inputs900[spec["short"]] if spec["inputs"] == "900" else w4.reference(ctx, spec["short"])
    return [str(a) for a in argv + ["--inputs", inputs, "--out", out]], inputs


def expected_physical(spec):
    if spec["layout"]:
        return {physical_of(d, spec["visible"]) for d, _ in spec["layout"]}
    return {physical_of(spec["device"] or 0, spec["visible"])}


def foreign_gap(rec, ctx, what, r):
    """The qualification of one direct command's evidence: non-helper compute processes the command's ForeignWatch
    observed (w8.run_bounded; the registered reservation helper excluded) make a GAP of that evidence, recorded once
    here. The values, comparisons and FAILs of the command stay as they are (never removed, re-paired or excused);
    the raw process ids stay in commands.jsonl. Nothing is claimed for what was not sampled (the watch samples every
    2 s)."""
    foreign = own_foreign(ctx, r)
    if foreign:
        rec.check(f"{what}: outside compute processes", "GAP",
                  f"{len(foreign)} foreign compute processes observed meanwhile (commands.jsonl); evidence kept, "
                  "qualified")
    return foreign


def export(rec, ctx, spec):
    """One `basal export` of the fixed inventory, cached by name within the run (export_once). The span of the
    records it made while it ran, (first, end) indices of rec.checks, is kept as ctx.export_checks[name], also when it
    raised: the export's own qualification for a later user of the cached directory (C5)."""
    name = spec["name"]
    if name in ctx.exports:
        return ctx.exports[name]
    first = len(rec.checks)
    try:
        return export_once(rec, ctx, spec)
    finally:
        ctx.export_checks[name] = (first, len(rec.checks))


def export_once(rec, ctx, spec):
    """The export itself: bounded by --export-timeout (past it FAIL timeout and Fatal), every input exported, memory
    only on the expected physical GPUs, the manifest's quantization, and for q4_k the W4 load checks (P0). Returns
    the directory, or None when it did not complete with every input; a placement, manifest, q4_k or
    foreign-process record does not change the return (its FAIL or GAP stays in rec.checks)."""
    name = spec["name"]
    w8.guard(ctx, rec, f"export {name}")
    out = ctx.out / "exports" / name
    out.parent.mkdir(exist_ok=True)
    argv, inputs = export_argv(ctx, spec, out)
    delta = {"CUDA_VISIBLE_DEVICES": spec["visible"]} if spec["visible"] else {}
    physical = expected_physical(spec)
    r = w8.run_bounded(rec, ctx, f"export-{name}", argv, delta, out.parent / f"{name}.log", ctx.args.export_timeout,
                       physical=physical)
    ctx.exports[name] = None
    ctx.export_runs[name] = {k: r[k] for k in ("exit", "stopped", "seconds", "cpu_s", "t_start_unix", "t_end_unix")}
    ctx.export_foreign[name] = r.get("foreign_compute_processes")
    foreign_gap(rec, ctx, f"export {name}", r)
    if r["stopped"] == "timeout":
        rec.check(f"export {name}: completed", "FAIL", f"no exit within --export-timeout {ctx.args.export_timeout} s",
                  cls="timeout")
        raise Fatal(f"export {name} exceeded --export-timeout: suspected hung GPU, nothing later runs")
    if not w8.judge_run(rec, f"export {name}", r):
        if r["stopped"] == "deadline":
            raise Deadline(name)
        return None
    want = (mg.line_count(Path(inputs) / "bench.jsonl"), mg.line_count(Path(inputs) / "systemone.jsonl"))
    got = (mg.line_count(out / "bench.jsonl"), mg.line_count(out / "systemone.jsonl"))
    if not rec.expect(f"export {name}: every input exported", got == want, f"{got} lines, inputs {want}"):
        return None
    mg.placement_check(rec, f"export {name}: memory only on physical GPUs {sorted(physical)}", r["placement_mib"],
                       physical)
    manifest = read_json(out / "manifest.json")
    ctx.export_runs[name]["load_s"] = (manifest.get("run") or {}).get("load_s")
    q = (manifest.get("backend") or {}).get("quantization")
    want_q = spec["quant"] or "none"
    layers = ctx.model[spec["short"]]["layers"]
    ok = (manifest.get("run") or {}).get("quantization") == want_q and \
        (q is None if want_q == "none" else isinstance(q, dict) and q.get("matrices") == 4 * layers)
    rec.expect(f"export {name}: manifest quantization {want_q}", ok,
               f"run.quantization {(manifest.get('run') or {}).get('quantization')!r}, backend.quantization "
               f"{json.dumps(q)[:200] if q else None}")
    if spec["quant"] == Q4:
        w4.q4_load_checks(rec, ctx, f"export {name}", spec["short"], manifest.get("backend"))
    ctx.exports[name] = out
    return out


def judge_pair(rec, ctx, cand_spec, base_spec):
    """Candidate against the base: bytes, `basal compare` without a difference, the manifest predicate. Returns
    whether the bytes were equal (None when an export is missing)."""
    base = export(rec, ctx, base_spec)
    cand = export(rec, ctx, cand_spec)
    label = f"{cand_spec['name']} vs {base_spec['name']}"
    if base is None or cand is None:
        rec.check(f"{cand_spec['unit']} {label}", "FAIL", "export missing")
        return None
    same = w8.same_bytes(rec, label, base, cand)
    cmp = w8.compare(rec, ctx, label, base, cand, f"{cand_spec['name']}-vs-{base_spec['name']}")
    if cmp is not None:
        problems = mg.compare_problems(cmp)
        rec.expect(f"basal compare {label}: no difference", not problems, "; ".join(problems[:5]))
    problems, detail = manifest_predicate(read_json(base / "manifest.json"), read_json(cand / "manifest.json"),
                                          cand_spec, ctx)
    rec.expect(f"manifest {label}: comparison contract ({'split' if cand_spec['layout'] else 'single GPU'})",
               not problems, scrub(detail))
    ctx.results.setdefault("pairs", {})[label] = {"bytes_equal": same, "manifest_problems": scrub(problems[:8])}
    if cand_spec["visible"]:
        ctx.results.setdefault("visible_mappings", {})[cand_spec["name"]] = {
            "CUDA_VISIBLE_DEVICES": cand_spec["visible"],
            "process_to_physical": {str(d): physical_of(d, cand_spec["visible"]) for d, _ in cand_spec["layout"]}}
    return same


# --------------------------------------------------------------------------- C1, C2


def unit_c1(rec, ctx):
    for cand, base in c1_pairs():
        judge_pair(rec, ctx, cand, base)


def unit_c2(rec, ctx):
    differing = []
    for cand, base in c2_pairs():
        if judge_pair(rec, ctx, cand, base) is False:
            differing.append((cand, base))
    ctx.results["c2_differing"] = [c["name"] for c, _ in differing]
    if differing:
        fp32_diagnostic(rec, ctx, differing)
    else:
        rec.check("C2 conditional FP32 diagnostic", "PASS", "not needed: no split export differs from its base")


F32_900 = {"dtype": "f32", "mode": None, "table": False}  # the flags of the stored basal_rs_f32 (W4 EXPORTS_900)


def fp32_diagnostic(rec, ctx, differing):
    """Only after a differing split export (its FAIL stays): f32 900 controls of 4.5B and mini, and the FP32
    decision comparison (W8/W4 predicate) of every differing export and its base."""
    res = ctx.results.setdefault("c2_fp32_diagnostic", {})
    controls = {}
    for short in SHORTS:
        spec = dict(xspec("C2", short, "candidate", None, None, inputs="900"), **F32_900)
        spec.update(name=f"{short}-900-f32-control-cand", device=None)
        controls[short] = export(rec, ctx, spec)
    seen = set()
    for cand, base in differing:
        short = cand["short"]
        for spec in (base, cand):
            if spec["name"] in seen:
                continue
            seen.add(spec["name"])
            d = ctx.exports.get(spec["name"])
            if spec["inputs"] == "900":
                ctl = controls.get(short)
                if d is None or ctl is None:
                    rec.check(f"C2 diagnostic {spec['name']} vs f32 control", "GAP", "export missing")
                    continue
                inputs, decisions = w8.build_900(ctx.root, short)
                runs = {k: w8.load_900(rec, f"C2 diagnostic export {n}", x, inputs)
                        for k, n, x in (("x", spec["name"], d), ("f32", f"{short} f32 control", ctl))}
                if None in runs.values():
                    continue
                ids = [r["id"] for r in inputs]
                p = {k: {by[i]["id"]: by[i]["p_cal"] for i in by} for k, by in runs.items()}
                fp32 = {i: decisions[i]["upstream_fp32"] for i in ids}
                control_ok = all(w8.argmax(p["f32"][i]) == w8.argmax(fp32[i]) for i in ids)
                flips = [i for i in ids if w8.argmax(p["x"][i]) != w8.argmax(fp32[i])]
                res[spec["name"]] = {"vs_stored_upstream_fp32_flips": flips,
                                     "f32_control_equal_to_stored_fp32": control_ok}
                rec.check(f"C2 diagnostic {spec['name']}: 900 decisions vs FP32 (control "
                          f"{'valid' if control_ok else 'not equal to the stored FP32: not attributable'})",
                          "SIGNAL" if flips else "PASS", f"{len(flips)} decisions differ from the stored upstream FP32")
                w8.compare(rec, ctx, f"{spec['name']} vs f32 control", ctl, d, f"{spec['name']}-vs-f32-control")
            else:
                ref = w4.reference(ctx, short)
                if d is None:
                    rec.check(f"C2 diagnostic {spec['name']} vs FP32", "GAP", "export missing")
                    continue
                cmp = w8.compare(rec, ctx, f"{spec['name']} vs FP32", ref, d, f"{spec['name']}-vs-fp32")
                if cmp is None:
                    continue
                diffs, problems, compared = w8.decision_differences(cmp, ref, d, upstream=True)
                res[spec["name"]] = {"decision_differences": diffs, "comparison_problems": problems,
                                     "endpoint_questions_compared": compared}
                rec.check(f"C2 diagnostic {spec['name']}: 44+34 decisions vs FP32 (W8/W4 predicate)",
                          "SIGNAL" if diffs else "GAP" if problems else "PASS",
                          f"{compared} endpoint questions, {len(diffs)} decision-level differences, "
                          f"{len(problems)} comparison problems")


# --------------------------------------------------------------------------- C3


def serve_yaml(path, port, cache, entries, max_inflight=4096):
    """Serve configuration: strings, lists and objects as JSON (valid YAML flow scalars); keys only when given (the
    base binary knows no `layer_split`)."""
    lines = [f"addr: 127.0.0.1:{port}", "metrics: true", f"max_inflight: {max_inflight}",
             f"gemm_cache: {json.dumps(str(cache))}", "models:"]
    for e in entries:
        first = "path" if "dir" in e else "repo"
        lines.append(f"  - {first}: {json.dumps(str(e.get('dir', e.get('repo'))))}")
        for k in ("gemm_table", "dtype", "quantization", "device", "replica_devices", "layer_split", "long_tokens",
                  "state_cache_mb"):
            if k in e:
                v = str(e[k]) if isinstance(e[k], Path) else e[k]
                lines.append(f"    {k}: {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def unit_c3(rec, ctx):
    """Refused configurations: the expected message, before any download, GEMM table search or load."""
    d = ctx.out / "invalid"
    d.mkdir(parents=True)
    port, cache, ref = ctx.args.port, d / "gemm-cache", w4.reference(ctx, "mini")
    request = d / "request.json"
    request.write_text(json.dumps({"state": "Klient pisze, że chce oddać telefon.",
                                   "questions": {"q": {"type": "noul", "instructions": "Czy klient prosi o zwrot?"}}}))
    model45 = ctx.model["4.5B"]["dir"]

    def cfg(label, **entry):
        return serve_yaml(d / f"{label}.yml", port, cache, [entry])

    for label, extra, visible, kind in c3_cases(d, cfg, ref, request, model45):
        w8.guard(ctx, rec, f"invalid {label}")
        mg.port_free_or_fail(port)
        log = d / f"{label}.log"
        delta = {"HF_HUB_OFFLINE": "1", **({"CUDA_VISIBLE_DEVICES": visible} if visible else {})}
        r = w8.run_bounded(rec, ctx, f"invalid-{label}", [str(ctx.binary["candidate"]), "--color", "never",
                                                          *map(str, extra)],
                           delta, log, 120, physical=set())
        foreign_gap(rec, ctx, f"invalid {label}", r)
        if r["stopped"]:
            w8.judge_run(rec, f"invalid {label}", r)
            w8.stop_after(f"invalid {label}", r)
            continue
        text = log.read_text(errors="replace")
        want = needle(kind)
        early = not any(x in text for x in ("fetching the model files", "loaded on GPU", w8.SEARCH, "serving on http"))
        rec.expect(f"invalid {label}: refused before download, table search and load",
                   r["exit"] not in (0, None) and want in text and early,
                   f"exit {r['exit']} after {r['seconds']:.1f} s, expected message {want!r}, early {early}")
        if r["placement_mib"]:
            rec.check(f"invalid {label}: no GPU memory of the process", "FAIL", f"{r['placement_mib']}")
        elif r["seconds"] < 1.0:
            rec.check(f"invalid {label}: no GPU memory of the process", "GAP",
                      f"ended after {r['seconds']:.2f} s, before a 0.5 s placement sample (the log shows no load)")
        else:
            rec.check(f"invalid {label}: no GPU memory of the process", "PASS", "none in the 0.5 s samples")
    for label, entries, kind in doctor_cases(model45, ctx.tables["4.5B"]):
        w8.guard(ctx, rec, f"invalid {label}")
        conf = serve_yaml(d / f"{label}.yml", port, cache, entries)
        log = d / f"{label}.json"
        r = w8.run_bounded(rec, ctx, label, [str(ctx.binary["candidate"]), "--color", "never", "doctor", "--config",
                                             str(conf), "--json"], {"HF_HUB_OFFLINE": "1"}, log, 180)
        foreign_gap(rec, ctx, label, r)
        if r["stopped"]:
            w8.judge_run(rec, label, r)
            w8.stop_after(label, r)
            continue
        try:
            report = rp.doctor_report(log.read_text(errors="replace"))
        except ValueError:
            rec.check(label, "FAIL", f"no JSON report (exit {r['exit']})")
            continue
        checks = [c for c in report.get("checks", []) if c.get("area") in ("config", "models")]
        fails = [f"{c.get('area')}: {c.get('what', '')}" for c in checks if c.get("level") == "fail"]
        if kind:  # a configuration the parser refuses is a `config` failure, a refused entry a `models` failure
            want = needle(kind)
            rec.expect(f"{label}: reported as a failure ({want!r})", any(want in f for f in fails),
                       scrub(f"exit {r['exit']}, {fails}"))
        else:
            want = re.compile(contract_value("c3.doctor_estimate"))
            est = [c.get("what", "") for c in checks if c.get("area") == "models" and want.search(c.get("what", ""))]
            rec.expect(f"{label}: valid entry without a config or model failure, per-device estimate reported",
                       bool(est) and not fails, scrub(f"estimate {est}, failures {fails}"))
    for what in ("mixed GPU types in a layer split", "a layer split on Metal", "a layer split on Intel"):
        rec.check(f"invalid {what}", "GAP", "not observable on this pair of identical CUDA cards; static review and "
                  "the CI compile checks only")


# --------------------------------------------------------------------------- HTTP (C4, C5 step 6)


def make_served(base):
    class SplitServed(base):
        """w4.Served (bounded readiness, requests, scrapes, graceful shutdown, placement samples; own-timeout FAILs of
        class timeout stop the run) with this runner's configuration keys and the idle-GPU wait excluding the
        registered reservation helper."""

        def __init__(self, rec, ctx, label, binary, entries, delta=None):
            w8.guard(ctx, rec, f"serve {label}")
            self.rec, self.ctx, self.label = rec, ctx, label
            self.dir = ctx.out / "serve" / label
            self.dir.mkdir(parents=True)
            busy = idle_except_helpers(rec, ctx, f"serve {label}: GPUs idle before the start")
            if busy:
                raise StepFailed(f"compute processes still on the GPUs before {label}: {len(busy)}")
            cfg = serve_yaml(self.dir / "serve.yml", ctx.args.port, self.dir / "gemm-cache", entries)
            w8.guard(ctx, rec, f"serve {label}")
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
                    raise Fatal(f"serve {label}: not ready within --ready-timeout: suspected hung GPU, no later GPU "
                                "work runs")
                time.sleep(min(2.0, max(0.0, ctx.deadline - time.monotonic())))
                self.placement = w4.placement_sample(rec, ctx, f"serve {label}: placement sample", self.srv.proc.pid)
                self.health = self.get("/health", 30, "GET /health")
                self.models = self.get("/v1/models", 30, "GET /v1/models")
                self.health_raw, self.models_raw = self.health[1], self.models[1]
                log = (self.dir / "server.log").read_text(errors="replace")
                if w8.SEARCH in log:
                    raise StepFailed(f"server {label} announced a GEMM search")
            except BaseException:
                self.close()
                raise

    return SplitServed


def foreign_rows(ctx, rows):
    """nvidia-smi compute-app rows other than the registered reservation helper's."""
    return [r for r in (rows or []) if not (r and r[0].strip().isdigit() and int(r[0]) in ctx.helper_pids)]


def idle_except_helpers(rec, ctx, what, timeout=30):
    """w4.idle_gpus that ignores the registered helper processes: [] when no other compute process holds a GPU."""
    end = time.monotonic() + timeout
    while True:
        rows = foreign_rows(ctx, w4.bounded_probe(rec, ctx, what, 30,
                                                  lambda limit: mg.smi(w4.COMPUTE_APPS, timeout=limit)))
        if not rows or time.monotonic() > end:
            return rows
        time.sleep(max(0.0, min(1.0, ctx.deadline - time.monotonic())))


def model_entry(ctx, short, placement, quant, long_tokens=HTTP_LONG_TOKENS, state_cache=HTTP_STATE_CACHE_MB):
    m = ctx.model[short]
    e = {"dir": m["dir"], "gemm_table": ctx.tables[short], "dtype": "f16", "long_tokens": long_tokens,
         "state_cache_mb": state_cache}
    if quant:
        e["quantization"] = quant
    if placement[0] == "device":
        e["device"] = placement[1]
    else:
        e["layer_split"] = [{"device": d, "layers": n} for d, n in placement[1]]
    return e


def http_items(ctx, short, long=True):
    """rp.http_items of `short` (its 44 + 34 inputs, the five malformed/default requests) with only the plan's long
    states (LONG_IDS), or none."""
    ctx.inputs, ctx.long_file = w4.reference(ctx, short), ctx.root / w8.LONG
    items = rp.http_items(ctx, ctx.model[short]["name"])
    return [it for it in items if not it[0].startswith("long:") or (long and it[0] in LONG_IDS)]


def long_first(items):
    """The concurrent order: long requests first, so the short ones run while the long lane computes."""
    return [it for it in items if it[1] == "long"] + [it for it in items if it[1] != "long"]


def reference_server(rec, ctx, label, binary, entries, items, key):
    """A base server's sequential answers and surface, kept as ctx.http_refs[key]; the span of the records its
    session made (placement, reference, graceful shutdown) as ctx.http_ref_checks[key], also when it raised, and
    ctx.http_ref_complete[key]: True only when the whole session (its shutdown check and close included) returned
    normally. The answers are kept as soon as they are complete, so a session that raised after them (its error is
    recorded by the caller, outside the span) leaves cached answers with complete False."""
    if key in ctx.http_refs:
        return ctx.http_refs[key]

    def body(s):
        mg.placement_check(rec, f"serve {label}: memory only on the expected physical GPU", s.placement,
                           {e.get("device", 0) for e in entries})
        answers = w4.http_reference(rec, s, label, items)
        ctx.http_refs[key] = {"answers": answers, "surface": w4.surface(s)}

    first, complete = len(rec.checks), False
    try:
        serve_session(rec, ctx, label, binary, entries, body)
        complete = True
    finally:
        ctx.http_ref_checks[key] = (first, len(rec.checks))
        ctx.http_ref_complete[key] = complete
    return ctx.http_refs[key]


def serve_session(rec, ctx, label, binary, entries, body, shutdown=None):
    s = Served(rec, ctx, label, binary, entries)
    try:
        result = body(s)
        (shutdown or (lambda srv: srv.shutdown_check()))(s)
        return result
    finally:
        s.close()


def label_check(rec, s, name, devices_text):
    status, samples = s.scrape(s.dir / "metrics-label.txt")
    got = rp.total(samples, "basal_replica_info", model=name, replica=0, device=devices_text)
    rec.expect(f"serve {s.label}: basal_replica_info device label {devices_text!r}", status == 200 and got == 1,
               f"status {status}, value {got}")


JOIN_GRACE_S = 5.0  # past a request's own end: its watchdog (w8.bounded_request) has cut the socket by then


def shutdown_inflight(rec, ctx, s, items, reference, name):
    """SIGTERM while the `items` (long requests) are in flight: their answers equal the reference, exit 0 and the
    shutdown logged within --shutdown-timeout (bounded like w4.Served.shutdown_check), no GPU memory left.

    Each request is sent once (no retry, none added), bounded as a whole by its own end min(its start +
    --http-timeout, run end) and kept with that end's cause and its raw record (s.batches, also partial). Its thread
    is joined by that end plus JOIN_GRACE_S on every exit, never with a new budget; an interruption or error cuts the
    requests still in flight first. Before any answer is judged: a request not complete by its own --http-timeout
    is a FAIL of class timeout and Fatal (no later GPU work), one cut by the run end a GAP and Deadline. Natural
    transport errors and answer differences stay FAILs of class correctness."""
    w8.guard(ctx, rec, f"serve {s.label}: shutdown with requests in flight")
    results, active, lock = [], set(), threading.Lock()
    s.batches.append(results)
    bounds = {}  # id -> (end, cause), fixed before the request starts

    def send(it, t, end, cause):
        r = w8.send_bounded(s.url, it[2], it[3], end, active)
        r.update(phase="shutdown-inflight", id=it[0], cls=it[1], timeout_s=round(end - t, 3))
        if r.get("late"):
            r["late"] = cause
        with lock:
            results.append(r)

    threads = []
    for it in items:
        t = time.monotonic()
        own_end = t + ctx.args.http_timeout
        end = min(own_end, ctx.deadline)
        bounds[it[0]] = (end, "deadline" if ctx.deadline < own_end else "timeout")
        threads.append(threading.Thread(target=send, args=(it, t, *bounds[it[0]]), daemon=True))

    def join(limit):
        for th in threads:
            if th.ident is not None:  # started
                th.join(max(0.0, limit - time.monotonic()))

    join_end = max((end for end, _ in bounds.values()), default=time.monotonic()) + JOIN_GRACE_S
    what = f"serve {s.label}: graceful shutdown with requests in flight"
    own = ctx.args.shutdown_timeout
    try:
        for th in threads:
            th.start()
        time.sleep(min(ctx.args.inflight_delay, max(0.0, ctx.deadline - time.monotonic())))
        status, held = s.scrape(s.dir / "metrics-at-sigterm.txt")
        inflight = rp.total(held, "basal_model_inflight_requests", model=name)

        def term(limit):
            os.kill(s.srv.proc.pid, signal.SIGTERM)
            return s.srv.proc.wait(limit)

        code, why = w4.bounded_wait(own, ctx.deadline, term)
        if why:
            w4.not_timely(rec, ctx, what, own, why, f"exit {code}")
        join(join_end)
    except BaseException:
        for cut in list(active):
            cut("interrupted")
        join(min(join_end, time.monotonic() + JOIN_GRACE_S))
        raise
    with lock:
        judged = list(results)
    answered = {r["id"] for r in judged}
    unfinished = [it[0] for it in items if it[0] not in answered]  # thread still running past its end + grace
    rec.check(f"serve {s.label}: requests in flight at SIGTERM", "PASS" if (inflight or 0) > 0 else "GAP",
              f"basal_model_inflight_requests {inflight} (scrape status {status})")
    log = (s.dir / "server.log").read_text(errors="replace")
    rec.expect(what, code == 0 and "shutting down" in log, f"exit {code}")
    late = {c: [r["id"] for r in judged if r.get("late") == c] + [i for i in unfinished if bounds[i][1] == c]
            for c in ("timeout", "deadline")}
    if late["timeout"]:
        rec.check(f"serve {s.label}: in-flight requests complete within --http-timeout", "FAIL",
                  f"not complete by their own {ctx.args.http_timeout} s: {late['timeout']} (records kept)",
                  cls="timeout")
        raise Fatal(f"serve {s.label}: an in-flight request had no answer within --http-timeout "
                    f"{ctx.args.http_timeout} s: suspected hung GPU, no later GPU work runs")
    if late["deadline"]:
        ctx.deadline_hit = True
        rec.check(f"serve {s.label}: in-flight requests", "GAP",
                  f"cut at the run deadline: {late['deadline']} (records kept), not judged")
        raise Deadline(f"{s.label} shutdown with requests in flight")
    bad = [r["id"] for r in judged if (r["status"], r.get("answer")) != reference.get(r["id"])]
    rec.expect(f"serve {s.label}: every in-flight request answered with the reference answer",
               len(judged) == len(items) and not bad,
               f"{len(judged)}/{len(items)}, differing {bad}, statuses "
               f"{dict(Counter(str(r['status']) for r in judged))}")
    after = w4.placement_sample(rec, ctx, f"serve {s.label}: no GPU memory left after exit", s.srv.proc.pid)
    rec.check(f"serve {s.label}: no GPU memory left after exit",
              "FAIL" if after else "PASS" if after == {} and s.placement else "GAP", f"{after}")


def unit_c4(rec, ctx):
    items = {s: http_items(ctx, s) for s in SHORTS}
    name45, name_mini = ctx.model["4.5B"]["name"], ctx.model["mini"]["name"]
    results = ctx.results.setdefault("c4", {})
    results["outside_processes"] = ("observed only by the idle wait before each server start (the registered helper "
                                    "excluded); the request intervals are not sampled: unobserved, not claimed free "
                                    "of other compute processes")

    def candidate(s, label, ref_key, passes, split, quant):
        ref = ctx.http_refs.get(ref_key)
        if ref is None:
            raise StepFailed(f"no reference {ref_key} for {label}")
        physical = {0, 1} if split else {0}
        mg.placement_check(rec, f"serve {label}: memory on physical GPUs {sorted(physical)}", s.placement, physical)
        its = items["4.5B"]
        rp.parity(rec, f"serve {label} sequential vs {ref_key}", s.run(its, 1, "sequential"), ref["answers"],
                  len(its))
        w4.same_surface(rec, label, w4.surface(s), ref["surface"], ref_key)
        for clients, rounds in passes:
            for k in range(rounds):
                rp.parity(rec, f"serve {label} {clients} clients round {k} (long requests first) vs {ref_key}",
                          s.run(long_first(its), clients, f"concurrent-{k}"), ref["answers"], len(its))
        if split:
            label_check(rec, s, name45, ",".join(str(d) for d, _ in L45))
        results[label] = {"reference": ref_key, "passes": [list(p) for p in passes]}

    def mixed(s):
        """4.5B split and mini on GPU 1 in one process: parallel traffic of both, answers equal to the references.
        Excluded by design (the process serves two models): the unknown-model error (it lists the served models)
        and the /v1/basal request without a model (the default model); mini's malformed requests (no model)."""
        refs = {"4.5B": ctx.http_refs.get("none-base-d0"), "mini": ctx.http_refs.get("mini-base-d1")}
        if None in refs.values():
            raise StepFailed("mixed-cand needs both references (none-base-d0, mini-base-d1)")
        mg.placement_check(rec, "serve mixed-cand: memory on physical GPUs [0, 1]", s.placement, {0, 1})
        skip = {"err-systemone-unknown-model", "basal-default-model"}
        per, reference = {}, {}
        for short in SHORTS:
            for rid, cls, path, raw in items[short]:
                if rid in skip or (short == "mini" and rid.startswith("err-")):
                    continue
                per.setdefault(short, []).append((f"{short}/{rid}", cls, path, raw))
                reference[f"{short}/{rid}"] = refs[short]["answers"][rid]
        a, b = per["4.5B"], per["mini"]
        mixed_items = [x for k in range(max(len(a), len(b))) for x in (a[k:k + 1] + b[k:k + 1])]  # interleaved
        recs = s.run(long_first(mixed_items), 16, "parallel-both-models")
        rp.parity(rec, "serve mixed-cand 16 clients, both models in parallel", recs, reference, len(mixed_items))
        label_check(rec, s, name45, "0,1")
        status, samples = s.scrape(s.dir / "metrics-mini.txt")
        got = rp.total(samples, "basal_replica_info", model=name_mini, replica=0, device="1")
        rec.expect("serve mixed-cand: mini basal_replica_info device label '1'", status == 200 and got == 1,
                   f"value {got}")
        results["mixed-cand"] = {"requests": len(mixed_items), "excluded": sorted(skip) + ["mini err-*"]}

    for label, binary, models, quant, ref_key, passes in c4_servers():
        entries = [model_entry(ctx, short, place, quant) for short, place in models]
        try:
            if ref_key is None:
                short = models[0][0]
                reference_server(rec, ctx, label, binary, entries, items[short], label)
            elif label == "mixed-cand":
                serve_session(rec, ctx, label, binary, entries, lambda s: mixed(s))
            elif label == "none-cand-split":
                ref = ctx.http_refs.get(ref_key)
                if ref is None:
                    raise StepFailed(f"no reference {ref_key} for {label}")
                long_items = [it for it in items["4.5B"] if it[0] in LONG_IDS]
                serve_session(rec, ctx, label, binary, entries,
                              lambda s: candidate(s, label, ref_key, passes, True, quant),
                              shutdown=lambda s: shutdown_inflight(rec, ctx, s, long_items, ref["answers"], name45))
            else:
                serve_session(rec, ctx, label, binary, entries,
                              lambda s: candidate(s, label, ref_key, passes, models[0][1][0] == "split", quant))
        except (StepFailed, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
            rec.check(f"serve {label}", "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))


# --------------------------------------------------------------------------- C5: the reservation helper


class HelperNotReady(Exception):
    """The reservation helper exited, or left an unreadable ready file, before it was ready and within its limits:
    the capacity evidence of that step is incomplete (a GAP), not a product failure."""


class Reservation:
    """One cuda-reserve.py process (monitor or reserve) in its own session, registered by its process id (excluded
    from the foreign-compute observation), bounded: readiness within --helper-ready-timeout, its stop within
    --helper-stop-timeout (past either: FAIL timeout and Fatal), its own --max-seconds a little past the run end."""

    def __init__(self, rec, ctx, label, mode, leave=None):
        w8.guard(ctx, rec, f"capacity {label}: start the helper")
        self.rec, self.ctx, self.label = rec, ctx, label
        self.dir = ctx.out / "capacity" / label
        self.dir.mkdir(parents=True)
        self.series, self.ready_path, self.result_path = (self.dir / f for f in ("series.csv", "ready.json",
                                                                                   "result.json"))
        left = max(1.0, ctx.deadline - time.monotonic()) + ctx.args.helper_stop_timeout + 60
        argv = [sys.executable, "-I", str(ctx.root / RESERVE_HELPER_PATH), mode, "--series", str(self.series),
                "--ready", str(self.ready_path), "--result", str(self.result_path), "--interval-ms", "50",
                "--tolerance-mib", str(C5_TOLERANCE_MIB), "--max-seconds", f"{left:.0f}", "--parent-pid",
                str(os.getpid())]
        for dev, mib in (leave or {}).items():
            argv += ["--leave", f"{dev}={mib}"]
        self.argv, self.ready, self.result, self.exit = argv, None, None, None
        self.proc = mg.spawn(argv, ctx.env, self.dir / "helper.log")
        ctx.helper_pids.add(self.proc.pid)
        entry = {"label": label, "mode": mode, "pid": self.proc.pid, "argv": argv,
                 "t_start_unix": round(time.time(), 3)}
        with open(ctx.out / "helper-processes.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        rec.command({"step": f"helper-{label}", "argv": argv, "env": {}, "registered_helper": True})

    def wait_ready(self):
        """The helper's ready record, observed by the absolute end min(start + --helper-ready-timeout, run end)
        (w4.bounded_wait's rule: the run end is the cause only when strictly earlier). Each observation (the ready
        file, the helper's exit) is judged on the monotonic clock before it is accepted, and the ready record again
        after it was read: seen after the end it is not timely (w4.not_timely: own limit FAIL timeout and Fatal, run
        end GAP and Deadline), kept in late-observation.json and never used. An exit before readiness within the
        limits, or an unreadable ready file, raises HelperNotReady (incomplete evidence of the step)."""
        what = f"capacity {self.label}: helper ready"
        own = self.ctx.args.helper_ready_timeout
        own_end = time.monotonic() + own
        end = min(own_end, self.ctx.deadline)
        cause = "deadline" if self.ctx.deadline < own_end else "timeout"

        def not_timely(seen, observed, raw):
            (self.dir / "late-observation.json").write_text(json.dumps(
                {"observed": observed, "seconds_after_end": round(seen - end, 3), "raw": raw}, indent=1) + "\n")
            w4.not_timely(self.rec, self.ctx, what, own, {"cause": cause, "late_s": seen - end}, observed)

        while True:
            ready = self.ready_path.is_file()
            code = None if ready else self.proc.poll()
            seen = time.monotonic()
            if ready:
                try:
                    record, problem = read_json(self.ready_path), None
                except (OSError, ValueError) as e:
                    record, problem = None, f"{type(e).__name__}: {e}"
                read = time.monotonic()
                if max(seen, read) > end:
                    not_timely(max(seen, read), "the ready file", record)
                if record is None:
                    raise HelperNotReady(f"{what}: unreadable ready file ({problem})")
                self.ready = record
                return record
            if code is not None:
                if seen > end:
                    not_timely(seen, f"the helper's exit {code}", {"exit": code})
                raise HelperNotReady(f"{what}: the helper exited {code} before it was ready (see "
                                     f"{self.dir.name}/helper.log)")
            if seen >= end:
                w4.not_timely(self.rec, self.ctx, what, own, {"cause": cause, "late_s": None})
            time.sleep(max(0.0, min(0.05, end - time.monotonic())))

    def samples(self):
        """device -> [(t_unix, free bytes)] from the series so far (a partial last line ignored)."""
        out = {}
        try:
            lines = self.series.read_text().splitlines()[1:]
        except OSError:
            return out
        for line in lines:
            parts = line.split(",")
            if len(parts) != 4:
                continue
            try:
                out.setdefault(int(parts[1]), []).append((float(parts[0]), int(parts[2])))
            except ValueError:
                continue
        return out

    def alive(self):
        return self.proc.poll() is None

    def stop(self):
        """SIGTERM, the exit awaited within --helper-stop-timeout: the release of the held memory is a cleanup bounded
        by its own limit only, so it may finish after the run end (the measurement deadline never cuts it). Returns
        the exit status; past its own limit FAIL timeout and Fatal, after the cleanup of its process group."""
        if self.proc.poll() is not None:
            self.exit = self.proc.returncode
        else:
            what = f"capacity {self.label}: helper stop"
            own = self.ctx.args.helper_stop_timeout

            def term(limit):
                os.kill(self.proc.pid, signal.SIGTERM)
                return self.proc.wait(limit)

            code, why = w4.bounded_wait(own, math.inf, term)  # a release: bounded by its own limit, also after the end
            if why:
                self.close()
                w4.not_timely(self.rec, self.ctx, what, own, why, f"exit {code}")
            self.exit = code
        mg.LIVE.discard(self.proc)
        self.ctx.helper_pids.discard(self.proc.pid)
        try:
            self.result = read_json(self.result_path)
        except (OSError, ValueError):
            self.result = None
        return self.exit

    def close(self):
        """Cleanup on any path: the helper's own process group only (SIGTERM, SIGKILL after 60 s)."""
        if self.proc.poll() is None:
            mg.stop(self.proc)
        mg.LIVE.discard(self.proc)
        if self.proc.returncode is not None:
            self.ctx.helper_pids.discard(self.proc.pid)


def window(samples, t0, t1):
    """The samples in [t0, t1] and the longest hole (edges included)."""
    xs = [(t, f) for t, f in samples if t0 <= t <= t1]
    edges = [t0, *[t for t, _ in xs], t1]
    return xs, max(b - a for a, b in zip(edges, edges[1:]))


def usage_mib(samples, t0, t1):
    """(MiB used, problem): median free over CALIBRATION_IDLE_S before t0 minus the minimum free in [t0, t1]."""
    idle, gap_i = window(samples, t0 - CALIBRATION_IDLE_S, t0)
    run, gap_r = window(samples, t0, t1)
    if len(idle) < 10 or not run or max(gap_i, gap_r) > MONITOR_MAX_GAP_S:
        return None, f"monitor samples: idle {len(idle)}, during {len(run)}, longest hole {max(gap_i, gap_r):.3f} s"
    return (statistics.median(f for _, f in idle) - min(f for _, f in run)) / MIB, None


def returned(res, level_mib, t_end):
    """(status, detail): the free memory of one device back to `level_mib` +- tolerance within C5_RETURN_S after
    t_end and staying there to the end of that window; GAP without full sample coverage."""
    xs, gap = window(res, t_end, t_end + C5_RETURN_S)
    if not xs or gap > MONITOR_MAX_GAP_S:
        return "GAP", f"{len(xs)} samples, longest hole {gap:.3f} s"
    tol = C5_TOLERANCE_MIB * MIB
    ok_from = next((k for k in range(len(xs)) if all(abs(f - level_mib * MIB) <= tol for _, f in xs[k:])), None)
    last = xs[-1][1] / MIB
    if ok_from is None:
        return "FAIL", f"last free {last:.0f} MiB, level {level_mib} MiB +-{C5_TOLERANCE_MIB}"
    return "PASS", f"within {xs[ok_from][0] - t_end:.2f} s; last free {last:.0f} MiB, level {level_mib} MiB"


def own_foreign(ctx, r):
    """Foreign compute processes seen during a command, without the registered helper."""
    seen = (r.get("foreign_compute_processes") or {}).get("foreign_pids") or {}
    return {p: n for p, n in seen.items() if not (p.isdigit() and int(p) in ctx.helper_pids)}


OOM = re.compile(r"CUDA_ERROR_OUT_OF_MEMORY|out of memory", re.IGNORECASE)
LAYER_CONTEXT = re.compile(r"layer \d+ of stage \d+ on CUDA device \d+")
STAGE_CONTEXT = re.compile(r"\bstage \d+\b")
DEVICE_CONTEXT = re.compile(r"\bCUDA device \d+\b")


def failure_context(text, model_name):
    """What a failing load's log shows: OOM, panic, model/stage/device/layer context and where it failed."""
    layer = LAYER_CONTEXT.search(text)
    return {"oom": bool(OOM.search(text)), "panicked": "panicked" in text,
            "model": contract_value("c5.model_context") is not None and
            contract_value("c5.model_context").format(name=model_name) in text,
            "stage": bool(STAGE_CONTEXT.search(text)), "device": bool(DEVICE_CONTEXT.search(text)),
            "layer": layer.group(0) if layer else None,
            "where": "load (" + layer.group(0) + ")" if layer else ("load" if "stage" in text else "inference or "
                                                                    "unattributed"),
            "stage1_device1": bool(re.search(r"\bstage 1\b", text)) and bool(re.search(r"\bCUDA device 1\b", text))}


def failing_export(rec, ctx, label, spec, res):
    """An export under a reservation (expected to fail, except step 5): (run result, log text), or (None, None) when
    it was stopped for a GEMM-search announcement (judge_run recorded that FAIL; the caller records the C5 outcome).
    At the run end GAP and Deadline, past --export-timeout FAIL timeout and Fatal (judge_run, stop_after)."""
    w8.guard(ctx, rec, f"capacity {label}")
    d = ctx.out / "capacity"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{label}-out"
    argv, _ = export_argv(ctx, spec, out)
    r = w8.run_bounded(rec, ctx, f"capacity-{label}", argv, {}, d / f"{label}.log", ctx.args.export_timeout,
                       physical=expected_physical(spec))
    res[label] = {k: r.get(k) for k in ("exit", "stopped", "seconds", "t_start_unix", "t_end_unix", "placement_mib")}
    foreign = own_foreign(ctx, r)
    if foreign:
        res[label]["foreign"] = len(foreign)  # their process ids stay in commands.jsonl (raw, private)
    if r["stopped"]:
        w8.judge_run(rec, f"capacity {label}", r)
        w8.stop_after(f"capacity {label}", r)
        return None, None
    return r, (d / f"{label}.log").read_text(errors="replace")


# Every step judged (PASS or FAIL) before C5 can pass: steps 1-7 of the plan, the base control included.
C5_STEPS = ("calibration", "window", "reserve-A", "single-cand", "single-base", "split-export", "split-server",
            "reserve-A-A1", "tight-split")


class C5:
    """The capacity proof of the plan (steps 1-7) with its fixed statuses: PASS, FAIL correctness (split output
    under the limit differs), FAIL capacity (candidate behaviour), GAP (incomplete evidence; REQ-006 stays open).
    Every step outcome is classified before its check is recorded; the final "C5 capacity proof" record is computed
    once from all of them, also when the method ends early or exceptionally (run end: GAP and Deadline; a command
    past its own limit: FAIL timeout and Fatal; the helper not ready: GAP). PASS only when every step of C5_STEPS was
    judged without a FAIL or GAP, and without a FAIL or GAP among the records the steps rest on (adopt): the own
    records of each export C5 runs or compares with (the unlimited references cached from C1/C2 included:
    completion, inputs, placement, manifest quantization, foreign processes), the C4 base reference server's
    records and its session's completion, the calibration byte checks, the placement checks of steps 5 and 6 and the
    split server's graceful shutdown. Records of other units that C5 does not use are not folded."""

    def __init__(self, rec, ctx):
        self.rec, self.ctx = rec, ctx
        self.res = ctx.results.setdefault("c5", {"steps": {}})
        self.steps = self.res["steps"]
        self.outcome = {"fail_correctness": [], "fail_capacity": [], "fail_timeout": [], "gap": [],
                        "supporting_gap": []}
        self.completed, self.finished, self.adopted = set(), False, set()
        # wall time the last process under the current reservation ended (its post-exit samples not yet judged)
        self.last_end = None
        self.name = ctx.model["4.5B"]["name"]
        self.single = xspec("C5", "4.5B", "candidate", None, "tree", 0)
        self.split = xspec("C5", "4.5B", "candidate", None, "tree", layout=L45)
        self.base_single = xspec("C5", "4.5B", "base", None, "tree", 0)
        self.unlimited = {"base": xspec("C1", "4.5B", "base", None, "tree", 0),
                          "candidate": xspec("C1", "4.5B", "candidate", None, "tree", 0),
                          "split": xspec("C2", "4.5B", "candidate", None, "tree", layout=L45)}

    def gap(self, step, why, check=True):
        """A GAP of `step`; check=False when the observation's own check was already recorded (never twice)."""
        self.outcome["gap"].append(f"{step}: {why}")
        if check:
            self.rec.check(f"C5 {step}", "GAP", why)

    def fail(self, cls, step, why):
        self.outcome[f"fail_{cls}"].append(f"{step}: {why}")

    # A span (export, reference_server, judged) holds exactly the records of its call because every rec.check of
    # the units runs on the main thread; the watcher and request threads only collect. A thread that records
    # itself would make the spans inexact.
    def adopt(self, step, span, base=False):
        """Folds the FAIL and GAP records of `span` ((first, end) indices of rec.checks, taken where the evidence `step`
        rests on was recorded) into the final C5 record, each record once and by reference (checks[i]: the order of
        summary.json checks); no check is added, removed or relabelled. A FAIL keeps its recorded class; a GAP, and
        with base=True a FAIL of a d42a400 reference (not candidate behaviour: incomplete evidence, as the base
        control of step 4), are a GAP of C5 (supporting_gap). Never changes which step runs next."""
        for i in range(*span):
            c = self.rec.checks[i]
            if i in self.adopted or c["status"] not in ("FAIL", "GAP"):
                continue
            self.adopted.add(i)
            tag = f"{c['status']}[{c['class']}]" if c["class"] else c["status"]
            entry = f"{step}: checks[{i}] {tag} {c['check']}"
            if c["status"] == "FAIL" and not base:
                self.outcome[f"fail_{c['class']}"].append(entry)
            else:
                self.outcome["supporting_gap"].append(entry)

    def judged(self, step, fn):
        """fn(), a supporting check of `step`, with the records it made adopted; returns fn's result. The records are
        adopted also when fn raises, and its exception propagates unchanged (run: the run end, a command past its own
        limit and errors are outcomes of their own), so a FAIL recorded before it keeps its precedence."""
        first = len(self.rec.checks)
        try:
            return fn()
        finally:
            self.adopt(step, (first, len(self.rec.checks)))

    def reference(self, key, step):
        """The unlimited export `key` (self.unlimited) that `step` compares with, cached from C1/C2 or exported at the
        start of the method, with that export's own records adopted (the base one as a d42a400 reference); None when
        it is missing (the caller keeps the comparison not made as a GAP of C5)."""
        name = self.unlimited[key]["name"]
        span = self.ctx.export_checks.get(name)
        if span is not None:
            self.adopt(step, span, base=key == "base")
        return self.ctx.exports.get(name)

    def finish(self):
        if self.finished:
            return
        self.finished = True
        o = self.outcome
        o["not_judged"] = [s for s in C5_STEPS if s not in self.completed]
        fails = [c for c in ("correctness", "capacity", "timeout") if o[f"fail_{c}"]]
        status = "FAIL" if fails else "GAP" if o["gap"] or o["supporting_gap"] or o["not_judged"] else "PASS"
        self.res["status"], self.res["outcome"] = status, o
        self.rec.check("C5 capacity proof", status, scrub(f"{o}; " + (
            "REQ-006 stays open (a GAP is kept and never renamed)" if status != "PASS" else
            "a single-card capacity error and split success under the same per-card limit")),
            cls=fails[0] if fails else None)

    def run(self):
        """The method with its final record on every exit: the run end (GAP, Deadline re-raised), a command past
        its own limit (FAIL timeout, Fatal re-raised), the helper not ready (GAP, the run goes on), an interruption
        or an error (GAP of C5, re-raised to the unit's own handling)."""
        try:
            self.method()
        except Deadline:
            self.gap("run end", "cut at the run deadline (its GAP is recorded where it was cut)", check=False)
            raise
        except Fatal as e:
            self.fail("timeout", "own limit", scrub(str(e)))
            raise
        except HelperNotReady as e:
            self.gap("helper", scrub(str(e)))
        except BaseException as e:
            self.gap("stopped", scrub(f"{type(e).__name__}: {e}"), check=False)
            raise
        finally:
            self.finish()

    def method(self):
        rec, ctx = self.rec, self.ctx
        # references first: nothing unlimited runs while a reservation holds memory
        for spec in self.unlimited.values():
            export(rec, ctx, spec)
        items = http_items(ctx, "4.5B")
        reference_server(rec, ctx, "none-base-d0", "base", [model_entry(ctx, "4.5B", ("device", 0), None)], items,
                         "none-base-d0")
        busy = idle_except_helpers(rec, ctx, "C5: GPUs idle before the calibration")
        if busy:
            self.gap("start", f"foreign compute processes on the GPUs: {len(busy)}")
            return
        usage, complete = self.calibrate()
        u_single = usage.get(("single", 0))
        u_split = [usage.get(("split", d)) for d in (0, 1)]
        if u_single is None or None in u_split or not complete:
            self.gap("window", "calibration incomplete: steps 2-7 do not start")
            return
        self.completed.add("calibration")
        top = max(u_split)
        a = math.floor((u_single + top) / 2)
        self.steps["window"] = {"U_single_mib": u_single, "U_split_mib": u_split, "A_mib": a,
                                "lower_mib": top + GIB_MIB, "upper_mib": u_single - GIB_MIB}
        if not (top + GIB_MIB <= a <= u_single - GIB_MIB):
            self.gap("capacity window", f"A {a} MiB outside [{top + GIB_MIB:.0f}, {u_single - GIB_MIB:.0f}]: steps "
                     "3-7 do not start; a method change needs a reviewed plan and runner revision")
            return
        rec.check("C5 capacity window", "PASS", f"A {a} MiB in [{top + GIB_MIB:.0f}, {u_single - GIB_MIB:.0f}]")
        self.completed.add("window")
        if not self.limited(a, items):
            return
        a1 = math.floor(u_split[1] / 2)
        self.steps["A1_mib"] = a1
        if a1 < GIB_MIB:
            self.gap("GPU 1 tight reservation", f"A1 {a1} MiB < 1 GiB")
            return
        self.tight(a, a1)

    def calibrate(self):
        """Step 1: the monitor without a reservation and the two calibration exports; (U per GPU in MiB, complete).
        Not complete (steps 2-7 do not start): an export not completed, foreign processes, monitor samples missing,
        the monitor's exit. Each calibration export's own records, its unlimited reference's (reference) and the
        byte check are adopted; they do not change which step runs next."""
        rec, ctx = self.rec, self.ctx
        usage, complete = {}, True
        mon = Reservation(rec, ctx, "monitor", "monitor")
        try:
            mon.wait_ready()
            for kind, spec, ref, devices in (("single", self.single, "candidate", (0,)),
                                             ("split", self.split, "split", (0, 1))):
                time.sleep(min(CALIBRATION_IDLE_S + 1.0, max(0.0, ctx.deadline - time.monotonic())))
                spec = dict(spec, name=f"c5-calibration-{kind}")
                out = export(rec, ctx, spec)
                self.adopt(f"calibration {kind}", ctx.export_checks[spec["name"]])
                run = ctx.export_runs.get(spec["name"]) or {}
                self.steps[f"calibration-{kind}"] = {k: run.get(k) for k in ("exit", "seconds")}
                if out is None:
                    self.gap(f"calibration {kind}", "the calibration export did not complete")
                    complete = False
                    continue
                name = self.unlimited[ref]["name"]
                unlimited = self.reference(ref, f"calibration {kind}")
                if unlimited is not None:
                    self.judged(f"calibration {kind}",
                                lambda: w8.same_bytes(rec, f"C5 calibration {kind} vs {name}", unlimited, out))
                else:  # its own records (adopted by reference) say why; noted in the C5 outcome, no new check
                    self.outcome["supporting_gap"].append(f"calibration {kind}: {name} missing, not compared")
                foreign = own_foreign(ctx, {"foreign_compute_processes": ctx.export_foreign.get(spec["name"])})
                if foreign:  # export() recorded the GAP of this observation (foreign_gap), adopted above
                    complete = False
                time.sleep(0.5)  # monitor samples past the end
                per = mon.samples()
                for dev in devices:
                    u, why = usage_mib(per.get(dev, []), run.get("t_start_unix") or 0, run.get("t_end_unix") or 0)
                    if why:
                        self.gap(f"calibration {kind} GPU {dev}", why)
                        complete = False
                    usage[(kind, dev)] = u
            self.steps["usage_mib"] = {f"{k} GPU {d}": v for (k, d), v in usage.items()}
            code = mon.stop()
            self.steps["monitor_exit"] = code
            if code != 0:
                self.gap("monitor", f"helper exit {code}")
                complete = False
        finally:
            mon.close()
        return usage, complete

    def ready(self, rsv, label, levels):
        """The reservation helper ready with every level within the tolerance (else a GAP)."""
        ready = rsv.wait_ready()
        self.last_end = None
        devs = ready.get("devices") or {}
        self.steps[label] = {d: {k: v for k, v in x.items() if k != "name"} for d, x in devs.items()}
        bad = [d for d in levels if not (devs.get(str(d)) or {}).get("within_tolerance")]
        if bad:
            self.gap(label, f"reservation not within +-{C5_TOLERANCE_MIB} MiB on GPUs {bad}")
            return False
        self.rec.check(f"C5 {label}", "PASS", ", ".join(
            f"GPU {d}: free {devs[str(d)]['free_after_bytes'] / MIB:.0f} MiB, target {lvl} MiB"
            for d, lvl in levels.items()))
        self.completed.add(label)
        return True

    def release(self, rsv, label):
        """Stop the helper (bounded) and record its exit; its process group is cleaned up on every path."""
        try:
            code = rsv.stop()
        finally:
            rsv.close()
        self.steps[f"{label}-exit"] = code
        self.rec.check(f"C5 {label} released: helper exit 0", "PASS" if code == 0 else "GAP", f"exit {code}")
        if code != 0:
            self.outcome["gap"].append(f"{label} release: helper exit {code}")

    def settle(self, t_unix, what):
        """Wait until the wall time `t_unix` (helper samples after a process end), never past the run end: the run
        end reached first is a GAP and Deadline (w8.guard), never a judged window."""
        left = t_unix - time.time()
        if left > 0:
            time.sleep(min(left, max(0.0, self.ctx.deadline - time.monotonic())))
        if time.time() < t_unix:
            w8.guard(self.ctx, self.rec, f"C5 {what}: helper samples after the process end")

    def held(self, rsv, levels, step):
        """The reservation still held before `step`: the helper alive and every level's free memory within the
        tolerance. After a process under this reservation ended (self.last_end), judged on the helper samples after
        that end (the 5 s rule of `returned`: back within C5_RETURN_S and staying, waited for up to C5_RETURN_S +
        0.5 s); before any process, on the latest sample. Otherwise a GAP of `step`: the control is not intact
        (incomplete evidence; the candidate's own return is judged by wait_return, steps 4 and 7)."""
        if self.last_end is not None:
            self.settle(self.last_end + C5_RETURN_S + 0.5, f"before {step}")
        per = rsv.samples()
        bad = []
        for d, lvl in levels.items():
            if self.last_end is not None:
                status, detail = returned(per.get(d, []), lvl, self.last_end)
                if status != "PASS":
                    bad.append(f"GPU {d}: {detail}")
            elif not per.get(d) or abs(per[d][-1][1] - lvl * MIB) > C5_TOLERANCE_MIB * MIB:
                bad.append(f"GPU {d}: latest free {per[d][-1][1] / MIB:.0f} MiB" if per.get(d) else
                           f"GPU {d}: no sample")
        alive = rsv.alive()
        if not alive or bad:
            self.gap(step, f"reservation not intact (helper alive {alive}; outside +-{C5_TOLERANCE_MIB} MiB of the "
                     f"level: {bad})")
            return False
        self.last_end = None
        return True

    def wait_return(self, rsv, levels, r, what):
        """Free memory of every level's GPU back to its level +- tolerance within 5 s of the process end (helper
        samples, waited for, never past the run end): PASS, FAIL capacity (not back) or GAP (samples missing)."""
        self.settle(r["t_end_unix"] + C5_RETURN_S + 0.5, what)
        per = rsv.samples()
        for dev, lvl in levels.items():
            status, detail = returned(per.get(dev, []), lvl, r["t_end_unix"])
            self.rec.check(f"C5 {what}: GPU {dev} free memory back to {lvl} MiB within {C5_RETURN_S:.0f} s", status,
                           detail, cls="capacity" if status == "FAIL" else None)
            if status == "FAIL":
                self.fail("capacity", what, f"GPU {dev} memory not back ({detail})")
            elif status == "GAP":
                self.outcome["gap"].append(f"{what} GPU {dev} return: {detail}")
        self.last_end = None  # judged here: the next held() reads the latest sample

    def search_stopped(self, label):
        """failing_export stopped `label` for a GEMM-search announcement; judge_run recorded its FAIL (class
        correctness). The base control is then not valid (GAP of C5); a candidate step fails (correctness)."""
        if label == "single-base":
            self.gap(label, "base control not valid: the command announced a GEMM search (its FAIL is recorded)",
                     check=False)
        else:
            self.fail("correctness", label, "the command announced a GEMM search although an explicit table was given")

    def limited(self, a, items):
        """Steps 3-6 under A on both GPUs; False when a GAP or FAIL stops the method."""
        rec, ctx = self.rec, self.ctx
        levels = {0: a, 1: a}
        rsv = None
        try:
            rsv = Reservation(rec, ctx, "reserve-A", "reserve", levels)
            if not self.ready(rsv, "reserve-A", levels):
                return False
            cand = None
            for label, spec in (("single-cand", dict(self.single, name="c5-single-cand")),
                                ("single-base", dict(self.base_single, name="c5-single-base"))):
                if not self.held(rsv, levels, label):
                    return False
                r, text = failing_export(rec, ctx, label, spec, self.steps)
                if r is None:
                    self.search_stopped(label)
                    return False
                self.last_end = r["t_end_unix"]
                c = failure_context(text, self.name)
                self.steps[label]["log"] = c
                if self.steps[label].get("foreign"):
                    self.gap(label, f"{self.steps[label]['foreign']} foreign compute processes (commands.jsonl)")
                if label == "single-cand":
                    cand = r
                    if r["exit"] == 0:
                        self.gap(label, "the candidate on one GPU completed under A: the control did not limit")
                        continue
                    missing = [k for k in ("oom", "model", "stage", "device") if not c[k]]
                    bad = c["panicked"] or bool(missing)
                    rec.check("C5 candidate single GPU under A: an OOM error with model, stage and device context, "
                              "no panic", "FAIL" if bad else "PASS",
                              f"exit {r['exit']}, missing {missing}, panicked {c['panicked']}, where {c['where']}",
                              cls="capacity" if bad else None)
                    if bad:
                        self.fail("capacity", label, f"missing {missing}, panicked {c['panicked']}")
                    self.wait_return(rsv, {0: a}, r, "candidate single")
                    self.completed.add(label)
                else:
                    valid = r["exit"] != 0 and c["oom"]
                    rec.check("C5 base single GPU under A: control valid (exit not 0 and OOM)",
                              "PASS" if valid else "GAP",
                              f"exit {r['exit']}, OOM {c['oom']}, panicked {c['panicked']} (a base panic is an "
                              f"observation of d42a400, not a candidate FAIL), where {c['where']}")
                    if valid:
                        self.completed.add(label)
                    else:
                        self.outcome["gap"].append(f"{label}: base control not valid")
                    if r["exit"] == 0 and cand is not None and cand["exit"] != 0:
                        rec.check("C5 analysis point: the base passes under A and the candidate single does not",
                                  "SIGNAL", "possible memory difference of the default path; the phase stays open "
                                  "until it is analysed")
            if not self.held(rsv, levels, "split-export"):
                return False
            if not self.split_export():
                return False
            if not self.held(rsv, levels, "split-server"):
                return False
            if not self.split_server(items):
                return False
            return self.held(rsv, levels, "after the split server")
        finally:
            if rsv is not None:
                self.release(rsv, "reserve-A")

    def split_export(self):
        """Step 5: the split export under A completes, bitwise equal to the unlimited base and split exports. Returns
        whether the method goes on (False after a GEMM-search stop)."""
        rec, ctx = self.rec, self.ctx
        spec = dict(self.split, name="c5-split-export-under-A")
        r, text = failing_export(rec, ctx, "split-export", spec, self.steps)
        if r is None:
            self.search_stopped("split-export")
            return False
        self.last_end = r["t_end_unix"]
        self.completed.add("split-export")
        if self.steps["split-export"].get("foreign"):
            self.gap("split-export", f"{self.steps['split-export']['foreign']} foreign compute processes "
                     "(commands.jsonl)")
        if r["exit"] != 0:
            c = failure_context(text, self.name)
            self.steps["split-export"]["log"] = c
            cls = "capacity" if c["oom"] else "correctness"
            rec.check("C5 split export under A: completed", "FAIL",
                      f"exit {r['exit']}, OOM {c['oom']} (the window held)", cls=cls)
            self.fail(cls, "split-export", f"exit {r['exit']}")
            return True
        out = ctx.out / "capacity" / "split-export-out"
        want = (mg.line_count(w4.reference(ctx, "4.5B") / "bench.jsonl"),
                mg.line_count(w4.reference(ctx, "4.5B") / "systemone.jsonl"))
        got = (mg.line_count(out / "bench.jsonl"), mg.line_count(out / "systemone.jsonl"))
        if not rec.expect("C5 split export under A: every input exported", got == want, f"{got} lines, inputs {want}"):
            self.fail("correctness", "split-export", "inputs missing")
            return True
        self.judged("split-export", lambda: mg.placement_check(
            rec, "C5 split export under A: memory on physical GPUs [0, 1]", r["placement_mib"], {0, 1}))
        for key in ("base", "split"):
            ref = self.reference(key, "split-export")
            if ref is None:
                self.gap(f"split-export vs {key}", "the unlimited export is missing")
                continue
            if not w8.same_bytes(rec, f"C5 split export under A vs unlimited {self.unlimited[key]['name']}", ref, out):
                self.fail("correctness", "split-export", f"differs from {self.unlimited[key]['name']}")
        base = ctx.exports.get(self.unlimited["base"]["name"])
        if base is not None:
            problems, detail = manifest_predicate(read_json(base / "manifest.json"), read_json(out / "manifest.json"),
                                                  spec, ctx)
            if not rec.expect("C5 split export under A: manifest comparison contract (split)", not problems,
                              scrub(detail)):
                self.fail("correctness", "split-export", "manifest")
        return True

    def split_server(self, items):
        """Step 6: the split server under A (C4's configuration with long_tokens 0, short bodies only) answers as the
        C4 base reference, the same requests sent once (w8.send_items, no retry). Each outcome is classified before
        its check is recorded: an OOM at the start or in a request (its error answer, its transport failure with the
        OOM in the server log) with the window held is a FAIL of class capacity; a genuine answer difference or
        another transport error or 5xx a FAIL of class correctness; a request past --http-timeout a FAIL of class
        timeout and Fatal; the run end a GAP and Deadline; GPUs still busy or the port taken before the start a GAP.
        The records of the C4 base reference server (a d42a400 reference), this server's placement check and its
        graceful shutdown (exit 0 and logged, no GPU memory left) are adopted; a base reference session that did not
        complete (ctx.http_ref_complete) is a GAP of C5, its cached answers still used. Returns whether the method
        goes on."""
        rec, ctx = self.rec, self.ctx
        span = ctx.http_ref_checks.get("none-base-d0")
        if span is not None:
            self.adopt("split-server", span, base=True)
        ref = ctx.http_refs.get("none-base-d0")
        if ref is None:
            self.gap("split-server", "no C4 base reference")
            return False
        if not ctx.http_ref_complete.get("none-base-d0", False):  # missing: completion not shown either
            self.outcome["supporting_gap"].append(
                "split-server: the base reference session none-base-d0 did not complete (it raised after its answers "
                "were kept; that error is recorded where it was caught); its cached answers are still compared")
        short = [it for it in items if it[1] != "long"]
        entries = [model_entry(ctx, "4.5B", ("split", L45), None, long_tokens=0)]
        label = "c5-split-under-A"
        log = ctx.out / "serve" / label / "server.log"

        def log_oom():
            return bool(OOM.search(log.read_text(errors="replace"))) if log.is_file() else False

        def body(s):
            self.judged("split-server", lambda: mg.placement_check(
                rec, "C5 split server under A: memory on physical GPUs [0, 1]", s.placement, {0, 1}))
            slots = [None] * len(short)
            s.batches.append(slots)  # written with the server's records (raw, also partial)
            why = w8.send_items(ctx, s.url, short, 1, "sequential", slots)
            records = [r for r in slots if r]
            if why == "deadline":
                ctx.deadline_hit = True
                rec.check(f"serve {label} sequential", "GAP", f"stopped at the run deadline: {len(records)} of "
                          f"{len(short)} sent (kept), not judged")
                raise Deadline(f"{label} sequential")
            if why == "timeout":
                late = [r for r in records if r.get("late")]
                first = f"{late[0]['id']}: {late[0].get('error')}" if late else "none recorded"
                rec.check(f"serve {label} sequential: transport", "FAIL", iso.scrub(
                    f"{len(records)} of {len(short)} sent, no answer within --http-timeout, first {first}"),
                    cls="timeout")
                raise Fatal(f"serve {label} sequential: no answer within --http-timeout {ctx.args.http_timeout} s: "
                            "suspected hung GPU, no later GPU work runs")
            self.judge_answers(records, ref["answers"], len(short), log_oom())

        try:
            serve_session(rec, ctx, label, "candidate", entries, body,
                          shutdown=lambda s: self.judged("split-server", s.shutdown_check))
            self.steps["split-server"] = {"requests": len(short)}
        except StepFailed as e:
            text = str(e)
            if text.startswith("compute processes still on the GPUs") or text.endswith("already in use"):
                self.gap("split-server", iso.scrub(f"{text} (the server did not start)"))
                return False
            cls = "capacity" if log_oom() else "correctness"
            rec.check("C5 split server under A", "FAIL", iso.scrub(f"{e}; OOM in the log {cls == 'capacity'}"), cls=cls)
            self.fail(cls, "split-server", iso.scrub(text))
            self.completed.add("split-server")
        finally:
            self.last_end = time.time()  # the server has exited (shutdown awaited, else its group stopped)
        return True

    def judge_answers(self, records, reference, expected, oom_logged):
        """Step 6 answers against the reference (rp.parity's comparison), classified before they are recorded: a
        differing record whose error answer, received bytes or error names an OOM (or a transport failure or 5xx
        while the server log shows one) is the capacity outcome; every other difference, transport error or 5xx,
        and requests not answered without an OOM, the correctness outcome."""
        name = "C5 split server under A sequential vs none-base-d0"

        def oom(r):
            text = " ".join(str(r.get(k) or "") for k in ("answer", "received", "error"))
            return bool(OOM.search(text)) or (oom_logged and (r["status"] is None or r["status"] >= 500))

        differing = [r for r in records
                     if r["status"] is None or (r["status"], r.get("answer")) != reference.get(r["id"])]
        errors = [r for r in records if r["status"] is None or r["status"] >= 500]
        oom_ids = sorted({r["id"] for r in differing + errors if oom(r)})
        other = [r["id"] for r in differing if r["id"] not in oom_ids]
        other_errors = [f"{r['id']}: {r.get('status') or r.get('error')}" for r in errors if r["id"] not in oom_ids]
        unanswered = expected - len(records)
        correct = not other and not other_errors and (not unanswered or bool(oom_ids))
        rec = self.rec
        rec.expect(f"{name}: answers equal to the reference, no other transport error or 5xx", correct,
                   iso.scrub(f"{len(records)}/{expected} answered, {len(other)} differing {other[:5]}, other errors "
                             f"{other_errors[:5]}, {len(oom_ids)} OOM errors judged as capacity"), cls="correctness")
        rec.expect(f"{name}: no OOM in a request under A (the window held)", not oom_ids,
                   f"{len(oom_ids)} requests with an OOM error {oom_ids[:5]}; OOM in the server log {oom_logged}",
                   cls="capacity")
        if not correct:
            self.fail("correctness", "split-server", f"{len(other)} differing, {len(other_errors)} other errors, "
                      f"{unanswered} not answered")
        if oom_ids:
            self.fail("capacity", "split-server", f"{len(oom_ids)} requests ended with OOM")
        self.completed.add("split-server")

    def tight(self, a, a1):
        """Step 7: GPU 0 at A, GPU 1 at A1; the split export fails at stage 1 on device 1 with OOM, no panic, and
        both GPUs return to their levels; the reservation is released with helper exit 0."""
        rec, ctx = self.rec, self.ctx
        levels = {0: a, 1: a1}
        rsv = None
        try:
            rsv = Reservation(rec, ctx, "reserve-A-A1", "reserve", levels)
            if not self.ready(rsv, "reserve-A-A1", levels) or not self.held(rsv, levels, "tight-split"):
                return
            r, text = failing_export(rec, ctx, "tight-split", dict(self.split, name="c5-tight-split"), self.steps)
            if r is None:
                self.search_stopped("tight-split")
                return
            self.last_end = r["t_end_unix"]
            if self.steps["tight-split"].get("foreign"):
                self.gap("tight-split", f"{self.steps['tight-split']['foreign']} foreign compute processes "
                         "(commands.jsonl)")
            c = failure_context(text, self.name)
            self.steps["tight-split"]["log"] = c
            if r["exit"] == 0:
                self.gap("tight-split", "the split export completed with GPU 1 at A1: the control did not limit")
                return
            bad = c["panicked"] or not (c["oom"] and c["stage1_device1"])
            rec.check("C5 split with GPU 1 at A1: OOM at stage 1 on device 1, no panic", "FAIL" if bad else "PASS",
                      f"exit {r['exit']}, OOM {c['oom']}, stage 1 / device 1 {c['stage1_device1']}, panicked "
                      f"{c['panicked']}", cls="capacity" if bad else None)
            if bad:
                self.fail("capacity", "tight-split", "context")
            self.wait_return(rsv, levels, r, "split with GPU 1 at A1")
            self.completed.add("tight-split")
        finally:
            if rsv is not None:
                self.release(rsv, "reserve-A-A1")


def unit_c5(rec, ctx):
    C5(rec, ctx).run()


# --------------------------------------------------------------------------- reports: split-only fields


def split_field_problems(report, kind, split, stages=None):
    """The split-only fields of a `basal benchmark`, `basal profile` or `basal bench` report (candidate schema, see
    ENGINE_FRAGMENTS): with a split present, in stage order and finite; without a split absent. Returns the
    problems. benchmark: settings.layer_split as given; layer_split {stages, devices [{device, gpu, resources,
    memory_pool}], transfers}; resources.per_device and every phase's resources.per_device. profile: stage_transfers
    and the stage_transfer section. bench: gpu_probe_tflops_per_device [{device, before, after}]."""
    problems = []
    phases = report.get("phases") if isinstance(report.get("phases"), list) else []
    paths = {"benchmark": [("settings", "layer_split"), ("layer_split",), ("resources", "per_device")] +
             [("phases", k, "resources", "per_device") for k in range(len(phases))],
             "profile": [("stage_transfers",), ("sections_ms_per_item", "stage_transfer")],
             "bench": [("gpu_probe_tflops_per_device",)]}[kind]
    devices = [d for d, _ in stages] if stages else []

    def get(path):
        v = report
        for k in path:
            if isinstance(k, int):
                if not isinstance(v, list) or k >= len(v):
                    return None, False
            elif not isinstance(v, dict) or k not in v:
                return None, False
            v = v[k]
        return v, True

    def per_device(v, where, required):
        if not isinstance(v, list) or [x.get("device") if isinstance(x, dict) else None for x in v] != devices:
            return [f"{where}: devices {canonical(v)[:80]}, expected {devices}"]
        bad = [f"{where}[{i}] lacks {sorted(required - set(x))}" for i, x in enumerate(v) if not required <= set(x)]
        bad += [f"{where}[{i}] not finite" for i, x in enumerate(v) if not numbers_finite(x)]
        return bad

    for path in paths:
        name = ".".join(str(k) for k in path)
        v, present = get(path)
        if not split:
            if present:
                problems.append(f"{name} present without a split")
            continue
        if not present:
            problems.append(f"{name} missing")
        elif path == ("settings", "layer_split"):
            if v != [{"device": d, "layers": n} for d, n in stages]:
                problems.append(f"settings.layer_split {canonical(v)[:80]}")
        elif path == ("layer_split",):
            if not isinstance(v, dict) or set(v) != {"stages", "devices", "transfers"}:
                problems.append(f"layer_split keys {sorted(v) if isinstance(v, dict) else v!r}")
            else:
                if v["stages"] != [{"device": d, "layers": n} for d, n in stages]:
                    problems.append(f"layer_split.stages {canonical(v['stages'])[:80]}")
                problems += per_device(v["devices"], "layer_split.devices", {"device", "gpu", "resources",
                                                                              "memory_pool"})
                problems += transfer_problems(v["transfers"], stages, "layer_split.transfers")
        elif path == ("stage_transfers",):
            problems += transfer_problems(v, stages, "stage_transfers")
        elif path == ("sections_ms_per_item", "stage_transfer"):
            if not (finite_number(v) and v >= 0):
                problems.append(f"sections_ms_per_item.stage_transfer {v!r}")
        elif path == ("gpu_probe_tflops_per_device",):
            problems += per_device(v, name, {"device", "before", "after"})
        else:  # resources.per_device of the run and of every phase
            problems += per_device(v, name, {"device"})
    return problems


def numbers_finite(v):
    """Every number in `v` finite (None is an absent sample, strings are labels)."""
    if isinstance(v, dict):
        return all(numbers_finite(x) for x in v.values())
    if isinstance(v, list):
        return all(numbers_finite(x) for x in v)
    return not isinstance(v, (int, float)) or isinstance(v, bool) or math.isfinite(v)


def transfer_problems(transfers, stages, where):
    """One entry per boundary in stage order with exactly TRANSFER_KEYS: from/to stage and device of the layout,
    copies and bytes positive integers (the boundary was used), transfer_ms and gate_wait_ms finite and non-negative,
    a scope text."""
    if not isinstance(transfers, list) or len(transfers) != len(stages) - 1:
        return [f"{where}: {len(stages) - 1} boundaries expected, got {canonical(transfers)[:80]}"]
    out = []
    for k, t in enumerate(transfers):
        if not isinstance(t, dict) or set(t) != TRANSFER_KEYS:
            out.append(f"{where}[{k}]: keys {sorted(t) if isinstance(t, dict) else t!r}")
            continue
        want = (k, k + 1, stages[k][0], stages[k + 1][0])
        if (t["from_stage"], t["to_stage"], t["from_device"], t["to_device"]) != want:
            out.append(f"{where}[{k}]: stages/devices {canonical(t)[:80]}, expected {want}")
        counts = [x for x in ("copies", "bytes") if not (isinstance(t[x], int) and not isinstance(t[x], bool) and
                                                         t[x] > 0)]
        times = [x for x in ("transfer_ms", "gate_wait_ms") if not (finite_number(t[x]) and t[x] >= 0)]
        if counts or times or not (isinstance(t["scope"], str) and t["scope"]):
            out.append(f"{where}[{k}]: counts {counts}, times {times} or the scope invalid")
    return out


def check_report(rec, ctx, label, data, kind, split):
    if data is None:
        return
    problems = split_field_problems(data, kind, split, L45 if split else None)
    rec.expect(f"{label}: split-only fields {'present and finite' if split else 'absent'}", not problems,
               "; ".join(problems[:6]))
    if split:
        ctx.results.setdefault("transfers", {})[label] = {
            "transfers": (data.get("layer_split") or {}).get("transfers") if kind == "benchmark" else
            data.get("stage_transfers"), "per_device_probe": data.get("gpu_probe_tflops_per_device")}


# --------------------------------------------------------------------------- C6 and the performance argv


def gpu_argv(ctx, binary, command, short, quant, place, extra=()):
    """w8.gpu_argv with the placement: ("device", n) -> --device n, ("split", layout) -> --layer-split."""
    m = ctx.model[short]
    argv = [ctx.binary[binary], "--color", "never", command, "--model", m["dir"], "--gemm-table", ctx.tables[short]]
    argv += ["--device", str(place[1])] if place[0] == "device" else ["--layer-split", layout_text(place[1])]
    argv += list(extra)
    if quant:
        argv += ["--quantization", quant]
    return [str(a) for a in argv]


def bench_argv(ctx, binary, quant, place):
    return gpu_argv(ctx, binary, "bench", "4.5B", quant, place, ["--reference", w4.reference(ctx, "4.5B"),
                                                                 "--dtype", "f16"])


def benchmark_argv(ctx, binary, quant, place, out):
    return gpu_argv(ctx, binary, "benchmark", "4.5B", quant, place, ["--scenario", "all", "--concurrency", "1,16",
                                                                     "--out", out])


def profile_split(rec, ctx, label, quant, batch):
    """One `basal profile` of the split candidate (bounded by --bench-timeout like w8.profile_run); its JSON line."""
    d = ctx.out / "perf" / "profile"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{label}-cand-split-{QTAG[quant]}-b{batch}"
    argv = gpu_argv(ctx, "candidate", "profile", "4.5B", quant, ("split", L45),
                    ["--reference", w4.reference(ctx, "4.5B"), "--dtype", "f16", "--n", str(ctx.args.profile_n),
                     "--batch", str(batch)])
    r = w8.run_bounded(rec, ctx, name, argv, {}, d / f"{name}.log", ctx.args.bench_timeout, stdout=d / f"{name}.json",
                       physical={0, 1})
    foreign_gap(rec, ctx, f"profile {name}", r)
    if not w8.judge_run(rec, f"profile {name}", r):
        w8.stop_after(f"profile {name}", r)
        return None
    mg.placement_check(rec, f"profile {name}: memory on physical GPUs [0, 1]", r["placement_mib"], {0, 1})
    try:
        text = (d / f"{name}.json").read_text()
        return json.loads(next(x for x in text.splitlines() if x.startswith("{")))
    except (OSError, StopIteration, ValueError):
        rec.check(f"profile {name}: output", "FAIL", "no JSON line")
        return None


def unit_c6(rec, ctx):
    res = ctx.results.setdefault("c6", {"scope": "synchronised profile: times are not request latency or speed"})
    for quant, batch in C6_PROFILES:
        w8.guard(ctx, rec, f"C6 profile {QTAG[quant]} batch {batch}")
        out = profile_split(rec, ctx, "c6", quant, batch)
        res[f"{QTAG[quant]}-b{batch}"] = None if out is None else {
            k: out.get(k) for k in ("sections_ms_per_item", "stage_transfers")}
        check_report(rec, ctx, f"C6 profile {QTAG[quant]} batch {batch}", out, "profile", True)


# --------------------------------------------------------------------------- L2, L3: the regression rule


def bench_reports(ctx, label):
    d = ctx.out / "perf" / label
    out = {}
    for f in sorted(d.glob("[ab]-*.json")):
        try:
            out[f.name] = read_json(f)
        except ValueError:
            out[f.name] = None
    return out


def ab_pair(rec, ctx, label, a_argv, b_argv, split_b, sink, quant):
    """One ab.py rounds-4 round (w8.bench_round: strict, A and B the same accuracy and items) with the split-only
    fields of every bench report and the q4_k load checks."""
    try:
        return w8.bench_round(rec, ctx, label, a_argv, b_argv, True, "4.5B", sink)
    finally:
        for f, data in bench_reports(ctx, label).items():
            check_report(rec, ctx, f"perf {label} {f}", data, "bench", split_b and f.startswith("b-"))
            if quant == Q4 and data:
                w4.q4_load_checks(rec, ctx, f"perf {label} {f}", "4.5B", data.get("backend"))


def regression_rule(rec, ctx, label, res, primary, scheduled, problems, confirm, profiles):
    """The W8/W4 rule, one definition for L2 and L3. primary: {command: pairs}; scheduled: {command: pairs per
    round}; problems: the primary rounds' evidence problems; confirm: {command: fn() -> (pairs, problems)} (exactly
    one confirmation round of only that command); profiles: fn() -> {role: profile or None} (A and B at batch 1)."""
    sig, open_ = {}, {}
    for k in primary:
        sig[k], open_[k] = w8.signals(primary[k], scheduled[k])
    res["primary_signals"], res["primary_open"] = sig, open_
    run = {k: bool(sig[k] or open_[k]) for k in sig}
    rec.check(f"perf {label}: primary regression signals (diagnostic; the confirmation round decides)", "PASS",
              f"B worse in every pair: {sig}; worse where a value exists, incomplete: {open_}; "
              + ("one confirmation round of the affected command and profiles A/B follow" if any(run.values()) else
                 "no confirmation round"))
    second, problems = {k: [] for k in primary}, list(problems)
    if any(run.values()):
        for k in primary:
            if run[k]:
                second[k], extra = confirm[k]()
                problems += extra
        res["profiles"] = profiles()
        problems += [f"profile {v}: missing" for v in "AB" if not res["profiles"].get(v)]
    persistent, unresolved = [], []
    for k in primary:
        if not run[k]:
            continue
        both = primary[k] + second[k]
        for m in sig[k] + open_[k]:
            w = [p["metrics"].get(m, {}).get("b_worse") for p in both]
            if False in w:
                continue
            if len(w) == 2 * scheduled[k] and all(x is True for x in w):
                persistent.append(f"{k}: {m}")
            else:
                unresolved.append(f"{k}: {m}")
    res.update(persistent=persistent, unresolved=unresolved, evidence_problems=problems)
    if persistent:
        rec.check(f"perf {label}: persistent regression signal (B worse in every pair of both rounds)", "SIGNAL",
                  f"{persistent}: analysis (pairs, per-question, conditions, profiles, the changed path); a real "
                  "cause -> fix and revalidation by the change-class matrix; none -> the user's decision on this "
                  "result")
    if problems or unresolved:
        rec.check(f"perf {label}: no regression conclusion (evidence incomplete)", "GAP",
                  f"{len(problems)} problems: {'; '.join(problems[:6])}; not resolved: {unresolved}")
        res["conclusion"] = None
    elif persistent:
        res["conclusion"] = "persistent signal"
    else:
        res["conclusion"] = "a"
        rec.check(f"perf {label}: regression conclusion (a)", "PASS", "no repeatable slowdown shown at the measured "
                  "resolution (not a claim of no regression); every pair and value is in summary.json")


def profiles_ab(rec, ctx, label, quant):
    out = {}
    for v, binary in (("A", "base"), ("B", "candidate")):
        p = w8.profile_run(rec, ctx, label, binary, "4.5B", quant, 1)
        check_report(rec, ctx, f"profile {label} {v}", p, "profile", False)
        out[v] = p
    return out


def campaign_l2(rec, ctx):
    res = ctx.results.setdefault("l2", {
        "roles": {"A": "base none GPU 0", "B": "candidate none GPU 0", "C": "candidate none 0:30,1:30"},
        "benchmark_order": " ".join(FAMILY_ORDER), "regression_pairs": [n for n, _, _ in REGRESSION_PAIRS],
        "tradeoff_pairs": [n for n, _, _ in TRADEOFF_PAIRS], "rule": "W8/W4 rule, no threshold; trade-off descriptive"})
    roles = {"A": ("base", ("device", 0)), "B": ("candidate", ("device", 0)), "C": ("candidate", ("split", L45))}
    bench = {r: bench_argv(ctx, b, None, p) for r, (b, p) in roles.items()}
    rounds = res.setdefault("rounds", {"bench_ab": [], "bench_bc": [], "benchmark": [], "confirmation_bench": [],
                                       "confirmation_benchmark": []})
    w8.guard(ctx, rec, "L2 bench A/B")
    ab_pair(rec, ctx, "l2-bench-ab", bench["A"], bench["B"], False, rounds["bench_ab"], None)
    w8.guard(ctx, rec, "L2 bench B/C")
    ab_pair(rec, ctx, "l2-bench-bc", bench["B"], bench["C"], True, rounds["bench_bc"], None)
    label = "l2-benchmark"
    specs = [(role, lambda out, role=role: benchmark_argv(ctx, roles[role][0], None, roles[role][1], out))
             for role in FAMILY_ORDER]
    runs = []
    entry = {"round": label, "runs": [], "regression_pairs": [], "tradeoff_pairs": [], "problems": {}}
    rounds["benchmark"].append(entry)
    try:
        w8.benchmark_runs(rec, ctx, label, specs, runs)
    finally:
        summarise_family(rec, ctx, label, runs, specs, entry)
    primary = {"bench": rounds["bench_ab"][0]["pairs"], "benchmark": entry["regression_pairs"]}
    problems = [f"bench {x}" for x in rounds["bench_ab"][0]["problems"]] + \
        [f"benchmark {x}" for x in entry["problems"]["regression"]]

    def confirm_bench():
        e = ab_pair(rec, ctx, "l2-bench-ab-confirmation", bench["A"], bench["B"], False, rounds["confirmation_bench"],
                    None)
        return e["pairs"], [f"bench confirmation {x}" for x in e["problems"]]

    def confirm_benchmark():
        cspecs = [(v, lambda out, v=v: benchmark_argv(ctx, roles[v][0], None, roles[v][1], out)) for v in "ABBA"]
        e = w8.benchmark_round(rec, ctx, "l2-benchmark-confirmation", cspecs, True, rounds["confirmation_benchmark"])
        for r in e["runs"]:
            check_report(rec, ctx, f"perf l2-benchmark-confirmation run {r['run']} {r['variant']}",
                         read_report(ctx, "l2-benchmark-confirmation", r), "benchmark", False)
        return e["pairs"], [f"benchmark confirmation {x}" for x in e["problems"]]

    regression_rule(rec, ctx, "L2 none", res, primary, {"bench": 4, "benchmark": 2}, problems,
                    {"bench": confirm_bench, "benchmark": confirm_benchmark},
                    lambda: profiles_ab(rec, ctx, "l2-confirm", None))
    tradeoff = [f"bench {x}" for x in rounds["bench_bc"][0]["problems"]] + \
        [f"benchmark {x}" for x in entry["problems"]["tradeoff"]]
    rec.check("perf L2: every trade-off pair (B/C, split) with all 25 fields (descriptive, no rule)",
              "GAP" if tradeoff else "PASS", "; ".join(tradeoff[:6]) if tradeoff else "complete")


def read_report(ctx, label, run):
    path = ctx.out / "perf" / label / run["file"]
    try:
        return read_json(path) if path.is_file() and path.stat().st_size else None
    except ValueError:
        return None


def summarise_family(rec, ctx, label, runs, specs, entry):
    sources = {json.dumps(r["data"].get("source"), sort_keys=True) for r in runs if r["data"]}
    if sources:
        rec.expect(f"perf {label}: one embedded workload version in every run", len(sources) == 1,
                   f"{len(sources)} sources")
    entry["runs"] = [{k: v for k, v in r.items() if k != "data"} for r in runs]
    entry["regression_pairs"] = w4.fixed_pairs(rec, label, runs, REGRESSION_PAIRS, True)
    entry["tradeoff_pairs"] = w4.fixed_pairs(rec, label, runs, TRADEOFF_PAIRS, True)
    reported = sum(1 for r in runs if r["data"])
    missing = [] if reported == len(specs) else [f"{reported} of {len(specs)} runs reported"]
    entry["problems"] = {
        "regression": missing + w8.evidence_problems(entry["regression_pairs"], 2, w8.PHASE_REQUIRED),
        "tradeoff": missing + w8.evidence_problems(entry["tradeoff_pairs"], 2, w8.PHASE_REQUIRED)}
    for r in runs:
        check_report(rec, ctx, f"perf {label} run {r['run']} {r['variant']}", r["data"], "benchmark",
                     r["variant"] == "C")
        if r["data"]:
            ctx.results.setdefault("memory", {})[f"{label} run {r['run']} {r['variant']}"] = {
                "memory_pool": r["data"].get("memory_pool"), "load_s": (r["data"].get("model") or {}).get("load_s"),
                "layer_split": r["data"].get("layer_split")}


def campaign_l3(rec, ctx):
    res = ctx.results.setdefault("l3", {"rule": "W8/W4 rule on the 4 bench fields; split descriptive"})
    for quant in (Q8, Q4):
        tag = QTAG[quant]
        part = res.setdefault(f"{tag}_pair", {"A": f"base {quant} GPU 0", "B": f"candidate {quant} GPU 0"})
        rounds = part.setdefault("rounds", {"bench_ab": [], "confirmation_bench": []})
        a, b = bench_argv(ctx, "base", quant, ("device", 0)), bench_argv(ctx, "candidate", quant, ("device", 0))
        w8.guard(ctx, rec, f"L3 {tag} bench A/B")
        ab_pair(rec, ctx, f"l3-{tag}-bench-ab", a, b, False, rounds["bench_ab"], quant)

        def confirm(quant=quant, tag=tag, a=a, b=b, rounds=rounds):
            e = ab_pair(rec, ctx, f"l3-{tag}-bench-ab-confirmation", a, b, False, rounds["confirmation_bench"], quant)
            return e["pairs"], [f"bench confirmation {x}" for x in e["problems"]]

        regression_rule(rec, ctx, f"L3 {quant}", part, {"bench": rounds["bench_ab"][0]["pairs"]}, {"bench": 4},
                        [f"bench {x}" for x in rounds["bench_ab"][0]["problems"]], {"bench": confirm},
                        lambda quant=quant, tag=tag: profiles_ab(rec, ctx, f"l3-{tag}-confirm", quant))
    for quant in (Q8, Q4):
        tag = QTAG[quant]
        sink = res.setdefault(f"{tag}_split_bench", [])
        w8.guard(ctx, rec, f"L3 {tag} bench single/split")
        e = ab_pair(rec, ctx, f"l3-{tag}-bench-single-split", bench_argv(ctx, "candidate", quant, ("device", 0)),
                    bench_argv(ctx, "candidate", quant, ("split", L45)), True, sink, quant)
        rec.check(f"perf L3 {quant}: single vs split pairs with all 4 fields (descriptive, no rule)",
                  "GAP" if e["problems"] else "PASS", "; ".join(e["problems"][:6]) if e["problems"] else "complete")
    label = "l3-split-benchmark"
    specs = [(f"C-{QTAG[q]}", lambda out, q=q: benchmark_argv(ctx, "candidate", q, ("split", L45), out))
             for q in (Q8, Q4)]
    runs = []
    try:
        w8.benchmark_runs(rec, ctx, label, specs, runs)
    finally:
        res["split_benchmark"] = [{k: v for k, v in r.items() if k != "data"} for r in runs]
        for r in runs:
            check_report(rec, ctx, f"perf {label} run {r['run']} {r['variant']}", r["data"], "benchmark", True)
            if r["data"]:
                ctx.results.setdefault("memory", {})[f"{label} run {r['run']} {r['variant']}"] = {
                    "memory_pool": r["data"].get("memory_pool"), "load_s": (r["data"].get("model") or {}).get("load_s"),
                    "layer_split": r["data"].get("layer_split")}
                res.setdefault("split_benchmark_phases", {})[r["variant"]] = {
                    ph: {(f"{m}.{q}" if q else m): ((p.get(m) or {}).get(q) if q else p.get(m))
                         for (m, q), _ in w8.PHASE_METRICS} for ph, p in w8.phases(r["data"]).items()}
                if r["variant"] == "C-q4":
                    w4.q4_load_checks(rec, ctx, f"perf {label} run {r['run']} q4_k", "4.5B", r["data"].get("backend"))
    rec.check("perf L3: the new candidate's single-GPU q8_0/q4_k HTTP benchmark (c1/c16)", "GAP",
              "not measured by the plan: S3 describes only d42a400 and is not evidence for this candidate; for these "
              "modes the ab.py pairs are the only evidence")


# --------------------------------------------------------------------------- fingerprint and bindings


def model_identity(m):
    return {"name": m["name"], "revision": m["dir"].name, "layers": m["layers"], "evidence_head": m["evidence"]}


def fingerprint(rec, args, ctx, gpus):
    versions = {}
    for label, b in ctx.binary.items():
        try:
            r = w4.bounded_probe(rec, ctx, f"{label} binary: --version", 60, lambda limit, b=b: subprocess.run(
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
        entry = dict(model_identity(m), safetensors_bytes=st.stat().st_size,
                     config_sha256=mg.sha256(m["dir"] / "config.json"))
        if not args.no_weight_hash:
            print(f"{time.strftime('%H:%M:%S')} hashing the weights of {short}", flush=True)
            entry["safetensors_sha256"] = mg.sha256(st)
            rec.expect(f"model {short}: weights SHA-256 of the stored references",
                       entry["safetensors_sha256"] == manifest["safetensors_sha256"])
        models[short] = entry
    lib = Path(args.basal_home) / ".local/cuda/12.9.1/lib"
    binaries = {}
    for k, b in ctx.binary.items():
        prov = ctx.provenance.get(k)
        binaries[k] = {"sha256": mg.sha256(b), "version_text": versions[k],
                       "declared_label": iso.scrub(getattr(args, f"{k}_label")),
                       "provenance": None if prov is None else {"sha256": prov[1], "record": iso.scrub(json.dumps(
                           prov[0], sort_keys=True))}}
    return {
        "runner": RUNNER_ID, "runner_sha256": mg.sha256(HERE), "campaign": args.campaign, "units": ctx.units,
        "helpers_sha256": {p: a for p, (a, _) in helpers_state(root).items()} |
        {"tools/decision-sets/build_benchmark.py": mg.sha256(root / w8.BUILD_BENCHMARK)},
        "reservation_helper": {"path": RESERVE_HELPER_PATH, "sha256": mg.sha256(root / RESERVE_HELPER_PATH),
                               "processes": "registered in helper-processes.jsonl when started (private)"},
        "sources": rp.source_digest(root, ctx.out), "split_source_sha256": mg.sha256(root / SPLIT_SOURCE),
        "embedded_assets": w4.embedded_assets(root), "git": w8.git_state(root),
        "source_commit_note": args.source_commit, "binaries": binaries,
        "binary_roles": {"base": "d42a400 rebuilt (declared_label)",
                         "candidate": "phase-05 candidate (declared_label)"},
        "models": models,
        "gemm_tables": {k: {"file": v.name, "sha256": mg.sha256(v)} for k, v in ctx.tables.items()},
        "inputs": {"sequential": {"file": w8.SEQUENTIAL, "sha256": mg.sha256(root / w8.SEQUENTIAL)},
                   "lock_sha256": mg.sha256(root / w8.LOCK),
                   "decisions": {s: mg.sha256(root / w8.DECISIONS.format(s)) for s in SHORTS},
                   "references_fp32": {s: w8.tree_sha(w4.reference(ctx, s)) for s in SHORTS},
                   "long_states_sha256": mg.sha256(root / w8.LONG), "long_ids": list(LONG_IDS),
                   "inputs_900": ctx.inputs900_summary,
                   "config_rs_sha256": mg.sha256(root / "crates/basal-cli/src/config.rs"),
                   "p0_facts": ctx.p0_provenance},
        "engine_contract": {k: v for k, (_, v, _) in ENGINE_CONTRACT.items()}, "stage_transfer": STAGE_TRANSFER,
        "engine_fragments": {"count": len(ENGINE_FRAGMENTS), "missing": missing_fragments(root),
                             "sources": {p: mg.sha256(root / p) for p in sorted({p for p, _ in ENGINE_FRAGMENTS})}},
        "gpus": [{"index": g[0], "name": g[1], "driver": g[3], "memory_total_mib": g[4], "power_limit_w": g[5],
                  "clocks_max_sm_mhz": g[6], "clocks_max_mem_mhz": g[7], "persistence_mode": g[8]} for g in gpus],
        "cuda_libraries": sorted(f"{p.name} {p.stat().st_size}" for p in lib.iterdir() if p.is_file())
        if lib.is_dir() else None,
        "cublaslt_tables": args.cublaslt, "python": sys.version.split()[0], "kernel": os.uname().release,
        "cpu_count": os.cpu_count(), "cuda_jit_cache_before_run": w8.jit_cache(ctx.env), "deadline": ctx.deadline_basis}


def l1_identity(fp):
    """What an l2/l3 run must share with its l1 runs (None: not recorded there)."""
    inputs = fp.get("inputs") or {}
    out = {"embedded assets": fp.get("embedded_assets"), "split source": fp.get("split_source_sha256"),
           "sequential workload": (inputs.get("sequential") or {}).get("sha256"),
           "lock file": inputs.get("lock_sha256"),
           "long states": inputs.get("long_states_sha256"), "config.rs": inputs.get("config_rs_sha256"),
           "P0 facts": (inputs.get("p0_facts") or {}).get("sha256"), "stage_transfer": fp.get("stage_transfer")}
    m = (fp.get("models") or {}).get("4.5B") or {}
    for k in ("name", "revision", "config_sha256", "safetensors_sha256", "layers", "evidence_head"):
        out[f"model 4.5B {k}"] = m.get(k)
    out["GEMM table 4.5B"] = ((fp.get("gemm_tables") or {}).get("4.5B") or {}).get("sha256")
    out["FP32 reference 4.5B"] = (inputs.get("references_fp32") or {}).get("4.5B")
    return out


def l1_evidence(args, out, shas, cur_sources):
    """The l1 raw directories an l2/l3 run builds on, read before anything runs; raises ValueError (refusal)."""
    found, covered = [], set()
    cur = out.resolve()
    for d in args.l1_run:
        prev = d.resolve()
        if not prev.is_dir() or prev == cur or prev in cur.parents or cur in prev.parents:
            raise ValueError(f"--l1-run {d.name}: not an existing directory of another run")
        try:
            run, summary = read_json(prev / "run.json"), read_json(prev / "summary.json")
        except (OSError, ValueError) as e:
            raise ValueError(f"--l1-run {d.name}: run.json and summary.json needed ({type(e).__name__})")
        fp = run.get("fingerprint")
        if not isinstance(fp, dict) or summary.get("fingerprint") != fp or fp.get("runner") != RUNNER_ID or \
                fp.get("campaign") != "l1":
            raise ValueError(f"--l1-run {d.name}: not an l1 raw directory of this runner, or run.json and "
                             "summary.json disagree")
        bins = fp.get("binaries") or {}
        got = ((bins.get("base") or {}).get("sha256"), (bins.get("candidate") or {}).get("sha256"))
        if got != (shas["base"], shas["candidate"]):
            raise ValueError(f"--l1-run {d.name}: binaries {got}, this run ({shas['base']}, {shas['candidate']})")
        if (fp.get("sources") or {}).get("sha256") != cur_sources["sha256"]:
            raise ValueError(f"--l1-run {d.name}: other synced sources than this run's")
        counts, classes = summary.get("counts") or {}, summary.get("fail_classes") or {}
        if counts.get("FAIL") != 0 or any(classes.get(k) for k in FAIL_CLASSES) or summary.get("stopped"):
            raise ValueError(f"--l1-run {d.name}: FAIL {counts.get('FAIL')}, classes {classes}, stopped "
                             f"{summary.get('stopped')!r}: l2/l3 run only after an l1 without a FAIL")
        units = summary.get("units") or {}
        done = {u for u, x in units.items() if isinstance(x, dict) and x.get("completed") is True}
        covered |= done
        found.append({"dir": prev.name, "run_json_sha256": mg.sha256(prev / "run.json"),
                      "summary_sha256": mg.sha256(prev / "summary.json"), "units_completed": sorted(done),
                      "counts": counts, "exit_status": summary.get("exit_status"),
                      "c5_status": ((summary.get("results") or {}).get("c5") or {}).get("status"),
                      "runner_sha256": fp.get("runner_sha256"), "helpers_sha256": fp.get("helpers_sha256"),
                      "reservation_helper_sha256": (fp.get("reservation_helper") or {}).get("sha256"),
                      "identity": l1_identity(fp)})
    if set(UNITS) - covered:
        raise ValueError(f"--l1-run: units {sorted(set(UNITS) - covered)} not completed in the given l1 runs")
    return found


def binding_checks(rec, ctx):
    """Build records against the synced tree, P0 facts, and (l2/l3) the l1 identities; returns the problems that
    forbid GPU work."""
    fp, blocking = ctx.fp, []
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
              "does not verify the base's source (the parent's build audit)")
    rec.check("P0 facts: the approved file", "PASS" if ctx.p0_provenance["usable"] else "GAP",
              f"{w4.P0_FACTS} SHA-256 {ctx.p0_provenance['sha256']}, expected {w4.P0_FACTS_SHA256}")
    for ev in ctx.l1 or []:
        mine = l1_identity(fp)
        for key, b in mine.items():
            a = ev["identity"].get(key)
            if a is None or b is None:
                rec.check(f"l1 evidence {ev['dir']}: {key}", "GAP", "not recorded in "
                          f"{'the l1 run' if a is None else 'this run'}: not shown equal")
            elif a != b:
                rec.check(f"l1 evidence {ev['dir']}: {key} equal to this run's", "FAIL",
                          iso.scrub(f"l1 {json.dumps(a)[:160]}, this run {json.dumps(b)[:160]}"))
                blocking.append(f"l1 {key}")
        for key, a, b in (("runner SHA-256", ev["runner_sha256"], fp["runner_sha256"]),
                          ("helpers SHA-256", ev["helpers_sha256"], fp["helpers_sha256"]),
                          ("reservation helper SHA-256", ev["reservation_helper_sha256"],
                           fp["reservation_helper"]["sha256"])):
            rec.check(f"l1 evidence {ev['dir']}: {key} equal to this run's", "PASS" if a == b else "GAP",
                      "equal" if a == b else "different: what it invalidates is the parent's change-class decision")
        rec.check(f"l1 evidence {ev['dir']}: units {ev['units_completed']}, C5 {ev['c5_status']}",
                  "PASS" if ev["c5_status"] == "PASS" or "C5" not in ev["units_completed"] else "GAP",
                  f"exit {ev['exit_status']}, counts {ev['counts']}" +
                  ("" if ev["c5_status"] == "PASS" else "; a C5 GAP keeps REQ-006 open"))
    return blocking


# --------------------------------------------------------------------------- plan (no GPU, no binary, no network)


D1 = decimal.Decimal("0.1")
ASSUMPTIONS = {  # the plan's explicit planning assumptions (seconds unless named); none is a measurement
    "A1": (decimal.Decimal("1.15"), "split GPU command time = single x 1.15 (uncalibrated; also covers the boundary "
                                    "protocol; not a measured bound)"),
    "A2": (decimal.Decimal("120"), "one 4.5B server case, worst"),
    "A3": (decimal.Decimal("30"), "one export failing at load under the reservation"),
    "A4": (decimal.Decimal("15"), "one reservation-helper action"),
    "A5": (decimal.Decimal("60"), "runner preflight, fingerprints, weight/input hashes"),
    "A6": (decimal.Decimal("60"), "all refused-configuration and doctor commands together"),
    "BOOT": (decimal.Decimal("14"), "minutes from the machine request to the measurement start"),
    "RESERVE": (decimal.Decimal("10"), "minutes to collect, verify SHA-256 and delete"),
}


def r1(x):
    return decimal.Decimal(str(x)).quantize(D1, rounding=decimal.ROUND_HALF_UP)


class Records:
    """Primary command durations of the W4 sessions on this card (commands.jsonl), each bound to its relative path
    and SHA-256 (the paths come from the forecast's `sources`; a SHA-256 other than the recorded one is refused)."""

    def __init__(self, root, forecast):
        self.files, self.steps = {}, {}
        for key in ("S1", "S2", "S3"):
            src = (forecast.get("sources") or {}).get(key) or {}
            path = root / str(src.get("path", ""))
            if not src.get("path") or not path.is_file():
                raise ValueError(f"forecast source {key}: no file")
            sha = mg.sha256(path)
            if sha != src.get("sha256"):
                raise ValueError(f"forecast source {key}: SHA-256 {sha}, recorded {src.get('sha256')}")
            self.files[key] = {"file": w4.rel(root, path), "sha256": sha}
            steps = {}
            for line in path.read_text().splitlines():
                if line.strip():
                    c = json.loads(line)
                    if w8.finite(c.get("seconds")) and c.get("exit") == 0 and not c.get("stopped"):
                        steps[c["step"]] = c["seconds"]
            self.steps[key] = steps

    def value(self, basis):
        """(seconds as Decimal, [sources]) of a basis:
        ("max", session, [steps] or prefix filter) the largest recorded step, rounded to 0.1 s (HALF_UP);
        ("scale", "A1", basis) x the assumption; ("half", basis) half at 1 and half at A1 (one ab.py of single and
        split); ("sum", [bases]); ("mul", n, basis); ("q4-ab",) one q4_k A/B ab.py: 2 x (S2 bench-bc - S2 bench-ab/2);
        ("assume", key); ("wait",) the runner's C5 post-exit wait bound."""
        kind = basis[0]
        if kind == "max":
            sess, sel = basis[1], basis[2]
            steps = self.steps[sess]
            names = [s for s in steps if (s in sel if isinstance(sel, (list, tuple)) else sel(s))]
            if not names:
                raise ValueError(f"no recorded step for {basis}")
            best = max(names, key=lambda s: steps[s])
            return r1(steps[best]), [dict(self.files[sess], field=f"max over {len(names)} steps: {best} "
                                                                  f"{steps[best]} s")]
        if kind == "scale":
            v, src = self.value(basis[2])
            return v * ASSUMPTIONS[basis[1]][0], src + [{"assumption": basis[1]}]
        if kind == "half":
            v, src = self.value(basis[1])
            return v / 2 + v / 2 * ASSUMPTIONS["A1"][0], src + [{"assumption": "A1 (split half)"}]
        if kind == "sum":
            total, src = decimal.Decimal(0), []
            for b in basis[1]:
                v, s = self.value(b)
                total, src = total + v, src + s
            return total, src
        if kind == "mul":
            v, src = self.value(basis[2])
            return v * basis[1], src
        if kind == "q4-ab":
            bc, s1 = self.value(("max", "S2", ["none-family-bench-bc"]))
            ab, s2 = self.value(("max", "S2", ["none-family-bench-ab"]))
            return 2 * (bc - ab / 2), s1 + s2 + [{"derived": "ab.py q4_k A/B = 2 x (4 f16 + 4 q4_k bench processes "
                                                             "- 4 f16 processes): 8 q4_k processes"}]
        if kind == "assume":
            return ASSUMPTIONS[basis[1]][0], [{"assumption": basis[1]}]
        if kind == "wait":
            return r1(C5_RETURN_S + 0.5), [{"derived": f"the runner's wait bound after a process end, C5_RETURN_S + "
                                                       f"0.5 = {C5_RETURN_S + 0.5} s (an upper bound of the added "
                                                       "sleep, no GPU command; not a measurement)"}]
        raise ValueError(basis)


def s1_export(short, quant, mode_filter):
    """A selector of S1 44 + 34 exports of `short` and `quant` (none: no q8/q4 tag) whose mode passes the filter."""
    def sel(step):
        if not step.startswith(f"export-{short}-") or "-900-" in step or "served" in step:
            return False
        q = "q8" if "-q8-" in step else "q4" if "-q4-" in step else "none"
        return q == QTAG[quant] and mode_filter(step)
    return sel


def inventory():
    """The fixed command inventory of every session: exports, compares, servers and their HTTP requests, invalid and
    doctor commands, helper actions, profiles, benchmark and ab.py runs (unconditional and conditional)."""
    d = Path("/nonexistent")
    c1, c2 = c1_pairs(), c2_pairs()
    c1_exports = {s["name"]: s for pair in c1 for s in pair}
    c2_exports = {s["name"]: s for pair in c2 for s in pair if s["name"] not in c1_exports}
    n_c3 = len(c3_cases(d, lambda label, **kw: d, d, d, d)) + len(doctor_cases(d, d))
    per_pass = 44 + 34 + len(LONG_IDS) + 5
    servers = []
    for label, binary, models, quant, ref, passes in c4_servers():
        if label == "mixed-cand":
            requests = (per_pass - 2) + (per_pass - 5)  # 4.5B without 2, mini without 5 (mixed())
        else:
            requests = per_pass * (1 + sum(r for _, r in passes)) + (len(LONG_IDS) if label == "none-cand-split" else 0)
        servers.append({"label": label, "binary": binary, "loads": len(models), "requests": requests})
    diag_base = {b["name"] for c, b in c2}
    inv = {
        "C1": {"exports_base": sum(1 for s in c1_exports.values() if s["binary"] == "base"),
               "exports_cand": sum(1 for s in c1_exports.values() if s["binary"] == "candidate"),
               "compares": len(c1),
               "exports_44_34": sum(1 for s in c1_exports.values() if s["inputs"] == "ref"),
               "exports_900": sum(1 for s in c1_exports.values() if s["inputs"] == "900")},
        "C2": {"exports_cand_split": len(c2), "exports_base_extra": len(c2_exports) - len(c2), "compares": len(c2),
               "conditional_fp32": {"f32_exports": 2, "compares_worst": len(c2) + len(diag_base)}},
        "C3": {"commands": n_c3},
        "C4": {"servers": servers, "requests_per_pass": per_pass},
        "C5": {"helper_actions": 6, "calibration_exports": 2, "failing_loads": 3, "split_export": 1, "server": 1,
               "server_requests": per_pass - len(LONG_IDS)},
        "C6": {"profiles": len(C6_PROFILES)},
        "L2": {"benchmark_runs": 6, "ab_calls": 2, "bench_processes": 16,
               "conditional": {"benchmark_runs": 4, "ab_calls": 1, "bench_processes": 8, "profiles": 2}},
        "L3": {"ab_calls": 4, "bench_processes": 32, "benchmark_runs": 2,
               "conditional": {"ab_calls": 2, "bench_processes": 16, "profiles": 4}},
    }
    c5 = inv["C5"]
    inv["totals"] = {  # model loads (a server: one per model entry; ab.py: one per bench process) and HTTP requests
        "L1": {"model_loads": inv["C1"]["exports_base"] + inv["C1"]["exports_cand"] + inv["C2"]["exports_cand_split"] +
               inv["C2"]["exports_base_extra"] + sum(x["loads"] for x in servers) + c5["calibration_exports"] +
               c5["failing_loads"] + c5["split_export"] + c5["server"] + inv["C6"]["profiles"],
               "model_loads_conditional": inv["C2"]["conditional_fp32"]["f32_exports"],
               "servers": len(servers) + c5["server"],
               "http_requests": sum(x["requests"] for x in servers) + c5["server_requests"],
               "commands_without_load": inv["C1"]["compares"] + inv["C2"]["compares"] + inv["C3"]["commands"],
               "helper_actions": c5["helper_actions"]},
        "L2": {"model_loads": inv["L2"]["benchmark_runs"] + inv["L2"]["bench_processes"],
               "model_loads_conditional": sum(inv["L2"]["conditional"][k] for k in
                                              ("benchmark_runs", "bench_processes", "profiles"))},
        "L3": {"model_loads": inv["L3"]["benchmark_runs"] + inv["L3"]["bench_processes"],
               "model_loads_conditional": inv["L3"]["conditional"]["bench_processes"] +
               inv["L3"]["conditional"]["profiles"]}}
    return inv


def plan_rows(inv):
    """The rows of the forecast, in the forecast's order: (session, label, count, commands, basis, conditional)."""
    c1, c2 = c1_pairs(), c2_pairs()
    names = {s["name"]: s for pair in c1 for s in pair}

    def n(pred):
        return sum(1 for s in names.values() if pred(s))

    def tree(step):
        return "-tree-" in step

    def any_mode(step):
        return True

    mini_none = s1_export("mini", None, lambda s: s != "export-mini-tree-base-dev0")
    rows = [
        ("L1", "runner preflight, fingerprints, weight/input hashes", 1, 1, ("assume", "A5"), False),
        ("L1", "C1 mini base none tree GPU 0 (cold first)", 1, 1, ("max", "S1", ["export-mini-tree-base-dev0"]),
         False),
        ("L1", "C1 mini none: base single, candidate tree/single GPU 0, tree GPU 1",
         n(lambda s: s["short"] == "mini" and s["quant"] is None and s["inputs"] == "ref") - 1, None,
         ("max", "S1", mini_none), False),
        ("L1", "C1 mini base q8_0 tree (first q8_0 load)", 1, 1, ("max", "S1", ["export-mini-tree-q8-base-dev0"]),
         False),
        ("L1", "C1 mini candidate q8_0 tree GPU 0/1", n(lambda s: s["short"] == "mini" and s["quant"] == Q8 and
                                                         s["binary"] == "candidate" and s["inputs"] == "ref"),
         None,
         ("max", "S1", s1_export("mini", Q8, lambda s: "base" not in s)), False),
        ("L1", "C1 mini q4_k: base tree, candidate tree GPU 0/1",
         n(lambda s: s["short"] == "mini" and s["quant"] == Q4 and s["inputs"] == "ref"), None,
         ("max", "S1", s1_export("mini", Q4, any_mode)), False),
        ("L1", "C1 4.5B none: base tree/single, candidate tree/single GPU 0, tree GPU 1",
         n(lambda s: s["short"] == "4.5B" and s["quant"] is None and s["inputs"] == "ref"), None,
         ("max", "S1", s1_export("4.5B", None, any_mode)), False),
        ("L1", "C1 4.5B q8_0: base tree, candidate tree GPU 0/1",
         n(lambda s: s["short"] == "4.5B" and s["quant"] == Q8 and s["inputs"] == "ref"), None,
         ("max", "S1", s1_export("4.5B", Q8, tree)), False),
        ("L1", "C1 4.5B q4_k: base tree, candidate tree GPU 0/1",
         n(lambda s: s["short"] == "4.5B" and s["quant"] == Q4 and s["inputs"] == "ref"), None,
         ("max", "S1", s1_export("4.5B", Q4, tree)), False),
        ("L1", "C1 basal compare 44 + 34 (no GPU)", sum(1 for c, _ in c1 if c["inputs"] == "ref"), None,
         ("max", "S1", lambda s: s.startswith("compare-") and "-900-" not in s), False)]
    for short in ("mini", "4.5B"):
        for quant, steps in ((None, ["export-{}-900-f16-tree", "export-{}-900-f16-tree-base"]),
                             (Q8, ["export-{}-900-q8-tree", "export-{}-900-q8-tree-base"]),
                             (Q4, ["export-{}-900-q4-tree"])):
            rows.append(("L1", f"C1 {short} 900 {QTAG[quant]} tree, base and candidate", 2, None,
                         ("max", "S1", [x.format(short) for x in steps]), False))
    cmp900 = ("max", "S1", lambda s: s.startswith("compare-") and "-900-" in s)
    none45 = ("max", "S1", s1_export("4.5B", None, any_mode))
    q845 = ("max", "S1", s1_export("4.5B", Q8, any_mode))
    q445 = ("max", "S1", s1_export("4.5B", Q4, any_mode))
    split_n = Counter((c["short"], c["quant"], c["inputs"]) for c, _ in c2)
    extra_n = Counter((b["short"], b["quant"], b["inputs"]) for _, b in c2 if b["name"] not in names)
    rows += [
        ("L1", "C1 basal compare 900 (no GPU)", sum(1 for c, _ in c1 if c["inputs"] == "900"), None, cmp900, False),
        ("L1", "C2 4.5B none split (9 layouts and flags)", split_n[("4.5B", None, "ref")], None,
         ("scale", "A1", none45), False),
        ("L1", "C2 4.5B base none extra flags", extra_n[("4.5B", None, "ref")], None, none45, False),
        ("L1", "C2 4.5B q8_0 split", split_n[("4.5B", Q8, "ref")], None, ("scale", "A1", q845), False),
        ("L1", "C2 4.5B base q8_0 extra flags", extra_n[("4.5B", Q8, "ref")], None, q845, False),
        ("L1", "C2 4.5B q4_k split", split_n[("4.5B", Q4, "ref")], None, ("scale", "A1", q445), False),
        ("L1", "C2 4.5B base q4_k extra flags", extra_n[("4.5B", Q4, "ref")], None, q445, False),
        ("L1", "C2 mini split none, reversed, q8_0, q4_k (q4_k time for all)",
         sum(split_n[("mini", q, "ref")] for q in (None, Q8, Q4)), None,
         ("scale", "A1", ("max", "S1", s1_export("mini", Q4, any_mode))), False),
        ("L1", "C2 4.5B 900 split none/q8_0/q4_k (q4_k time for all)",
         sum(split_n[("4.5B", q, "900")] for q in (None, Q8, Q4)), None,
         ("scale", "A1", ("max", "S1", ["export-4.5B-900-q4-tree"])), False),
        ("L1", "C2 basal compare 44 + 34 and 900 (no GPU)", len(c2), None, cmp900, False),
        ("L1", "C2 CONDITIONAL FP32 diagnostic: 4.5B and mini 900 f32 exports + worst-case compares "
               f"({inv['C2']['conditional_fp32']['compares_worst']})", 1,
         2 + inv["C2"]["conditional_fp32"]["compares_worst"],
         ("sum", [("max", "S1", ["export-4.5B-900-f32"]), ("max", "S1", ["export-mini-900-f32"]),
                  ("mul", inv["C2"]["conditional_fp32"]["compares_worst"], cmp900)]), True),
        ("L1", "C3 refused configurations and doctor (no load)", 1, inv["C3"]["commands"], ("assume", "A6"), False),
        ("L1", "C4 HTTP servers", len(inv["C4"]["servers"]), None, ("assume", "A2"), False),
        ("L1", "C5 helper actions (monitor; reserve A, A; reserve A, A1: start and stop)", 6, None, ("assume", "A4"),
         False),
        ("L1", "C5 calibration exports under the monitor (single + split)", 1, 2,
         ("sum", [none45, ("scale", "A1", none45)]), False),
        ("L1", "C5 candidate and base single under A (expected OOM)", 2, None, ("assume", "A3"), False),
        ("L1", "C5 split export under A", 1, None, ("scale", "A1", none45), False),
        ("L1", "C5 split server under A (short bodies)", 1, None, ("assume", "A2"), False),
        ("L1", "C5 split with GPU 1 at A1 (expected stage-1 OOM)", 1, None, ("assume", "A3"), False),
        ("L1", "C6 split profiles none/q8_0/q4_k at batch 1 and 16", 1, len(C6_PROFILES),
         ("scale", "A1", ("sum", [("max", "S1", [f"profile-candidate-{q}-b{b}"]) for q in ("f16", "q8_0", "q4_k")
                                  for b in (1, 16)])), False),
        ("L1", "C5 post-exit helper samples before the next step (after base single, split export, split server)",
         3, 0, ("wait",), False)]
    ab_none = ("max", "S2", ["none-family-bench-ab"])
    bm_ab = ("max", "S2", lambda s: re.fullmatch(r"none-family-benchmark-(confirmation-)?\d-[AB]", s) is not None)
    rows += [
        ("L2", "runner preflight", 1, 1, ("assume", "A5"), False),
        ("L2", "benchmark A (base) x2, B (candidate single) x2", 4, None, bm_ab, False),
        ("L2", "benchmark C (candidate split) x2", 2, None, ("scale", "A1", bm_ab), False),
        ("L2", "ab.py rounds 4 A/B", 1, 8, ab_none, False),
        ("L2", "ab.py rounds 4 B/C (half single, half split)", 1, 8, ("half", ab_none), False),
        ("L2", "CONDITIONAL confirmation benchmark A B B A", 4, None, bm_ab, True),
        ("L2", "CONDITIONAL confirmation ab.py rounds 4 A/B", 1, 8, ab_none, True),
        ("L2", "CONDITIONAL profile A and B at batch 1", 2, None,
         ("max", "S2", ["none-confirm-base-f16-b1", "none-confirm-candidate-f16-b1"]), True)]
    ab_q8 = ("max", "S3", ["q8-family-bench-ab"])
    rows += [
        ("L3", "runner preflight", 1, 1, ("assume", "A5"), False),
        ("L3", "ab.py rounds 4 q8_0 A/B", 1, 8, ab_q8, False),
        ("L3", "ab.py rounds 4 q4_k A/B", 1, 8, ("q4-ab",), False),
        ("L3", "ab.py rounds 4 q8_0 candidate single vs split", 1, 8, ("half", ab_q8), False),
        ("L3", "ab.py rounds 4 q4_k candidate single vs split", 1, 8, ("half", ("q4-ab",)), False),
        ("L3", "benchmark candidate split q8_0", 1, None,
         ("scale", "A1", ("max", "S3", lambda s: re.fullmatch(r"q8-family-benchmark-(confirmation-)?\d-[AB]", s)
                          is not None)), False),
        ("L3", "benchmark candidate split q4_k (S3 q4_k run time)", 1, None,
         ("scale", "A1", ("max", "S3", lambda s: re.fullmatch(r"q8-family-benchmark-\d-C", s) is not None)), False),
        ("L3", "CONDITIONAL confirmation ab.py rounds 4 q8_0 A/B", 1, 8, ab_q8, True),
        ("L3", "CONDITIONAL confirmation ab.py rounds 4 q4_k A/B", 1, 8, ("q4-ab",), True),
        ("L3", "CONDITIONAL profiles q8_0 A, B and q4_k A, B at batch 1", 1, 4,
         ("sum", [("mul", 2, ("max", "S3", ["q8-confirm-base-q8_0-b1", "q8-confirm-candidate-q8_0-b1"])),
                  ("mul", 2, ("max", "S1", ["profile-candidate-q4_k-b1"]))]), True)]
    return rows


def plan_report(args, root):
    """Counts of every session from the fixed inventory, and with --forecast the time forecast from the recorded W4
    command durations (each with its file and SHA-256) and the plan's explicit assumptions."""
    inv = inventory()
    report = {"scope": "planning estimate: counts from this runner's fixed inventory; durations from recorded W4 "
                       "commands of other binaries on the same card and explicit assumptions; nothing here is a "
                       "measurement of this campaign, a bound or a purchase decision", "inventory": inv,
              "assumptions": {k: [str(v), why] for k, (v, why) in ASSUMPTIONS.items()},
              "caps_minutes_from_request": CAP_MINUTES, "reserve_s": RESERVE_S,
              "engine_contract_pending": {c: pending_contract(run_scopes(c, UNITS)) for c in CAMPAIGNS},
              "engine_contract": {k: {"value": v, "used_by": list(used), "what": what}
                                  for k, (used, v, what) in ENGINE_CONTRACT.items()},
              "engine_fragments": {"count": len(ENGINE_FRAGMENTS), "missing": missing_fragments(root),
                                   "sources": {p: mg.sha256(root / p) if (root / p).is_file() else None
                                               for p in sorted({p for p, _ in ENGINE_FRAGMENTS})}}}
    if not args.forecast:
        report["time"] = "NOT DETERMINED: give --forecast (the private forecast whose sources name the W4 records)"
        return report
    forecast = read_json(args.forecast)
    recs = Records(root, forecast)
    report["forecast"] = {"file": w4.rel(root, args.forecast), "sha256": mg.sha256(args.forecast),
                          "revision": forecast.get("revision")}
    report["records"] = recs.files
    report["observed_boot_minutes"] = forecast.get("observed_boot_minutes_request_to_measurement_start")
    rows = []
    for session, label, count, commands, basis, conditional in plan_rows(inv):
        each, sources = recs.value(basis)
        rows.append({"session": session, "label": label, "count": count,
                     "commands": count if commands is None else commands, "seconds_each": str(r1(each)),
                     "seconds": str(r1(each * count)), "conditional": conditional, "sources": sources})
    report["rows"] = rows
    report["sessions"] = {}
    for session in ("L1", "L2", "L3"):
        mine = [r for r in rows if r["session"] == session]
        theirs = (forecast.get("steps") or {}).get(session) or []
        uncond = sum(decimal.Decimal(r["seconds"]) for r in mine if not r["conditional"]) / 60
        cond = sum(decimal.Decimal(r["seconds"]) for r in mine if r["conditional"]) / 60
        fixed = ASSUMPTIONS["BOOT"][0] + ASSUMPTIONS["RESERVE"][0]
        cap = CAP_MINUTES[session.lower()]
        stated = (forecast.get("recomputed_from_rows") or {}).get(session) or {}
        report["sessions"][session] = {
            "unconditional_min": f"{uncond:.4f}", "conditional_min": f"{cond:.4f}",
            "worst_from_request_min": f"{uncond + cond + fixed:.4f}", "cap_min": cap,
            "runner_deadline_max_min": cap - int(ASSUMPTIONS["RESERVE"][0]),
            "fits_cap": uncond + cond + fixed <= cap,
            "forecast_rows": len(theirs), "rows": len(mine),
            "forecast_recomputed": {k: stated.get(k) for k in ("unconditional_min", "conditional_min",
                                                              "worst_from_request_min", "cap_min")},
            "row_differences": [
                {"row": k, "label": m["label"], "forecast_step": t.get("step"), "count": [m["count"], t.get("count")],
                 "seconds": [m["seconds"], t.get("seconds")]}
                for k, (m, t) in enumerate(zip(mine, theirs))
                if m["count"] != t.get("count") or decimal.Decimal(m["seconds"]) != r1(t.get("seconds", 0))]}
    report["initial_caps_total_min"] = sum(CAP_MINUTES[s] for s in CAMPAIGNS)
    report["note"] = ("L1 worst case against its 85-minute cap; the initial fixed caps L1 + L2 + L3 total 275 min. A "
                      "further fix cycle or revalidation is not promised by these numbers: each needs its own fresh "
                      "concrete forecast of whole units before any purchase (the plan's change-class matrix).")
    return report


def print_plan(args, root):
    print("900-question inputs (rebuilt in memory from the tracked files, nothing written):")
    built = {}
    for short in SHORTS:
        rows, decisions = w8.build_900(root, short)
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
        built[short] = hashlib.sha256(data).hexdigest()
        print(f"  {short}: {len(rows)} records, {len(decisions)} decision ids, bench.jsonl sha256 {built[short]}")
    if args.prepared_900:
        for short in SHORTS:
            p = args.prepared_900 / f"inputs-900-{short}" / "bench.jsonl"
            got = mg.sha256(p) if p.is_file() else None
            print(f"  prepared {w4.rel(root, p)}: sha256 {got} -> "
                  f"{'equal to the rebuild' if got == built[short] else 'DIFFERENT or missing'}")
    plan = plan_report(args, root)
    inv = plan["inventory"]
    print(f"\n{plan['scope']}")
    print(f"\nfixed inventory: {json.dumps(inv, indent=1)}")
    print("\nengine-owned texts bound to the candidate source (not frozen by the coordinator):")
    for k, x in plan["engine_contract"].items():
        print(f"  {k} ({', '.join(x['used_by'])}): {x['value']!r}")
    print(f"  pending (a run whose units use one is refused, exit 2): {plan['engine_contract_pending']}")
    frag = plan["engine_fragments"]
    print(f"  {frag['count']} source fragments, missing in this tree: {frag['missing'] or 'none'}; sources "
          f"{json.dumps(frag['sources'], indent=1)}")
    print("\nexplicit assumptions (not measurements):")
    for k, (v, why) in plan["assumptions"].items():
        print(f"  {k} = {v}: {why}")
    if "rows" not in plan:
        print(f"\ntime: {plan['time']}")
    else:
        print(f"\nforecast {plan['forecast']['file']} sha256 {plan['forecast']['sha256']} "
              f"({plan['forecast']['revision']})")
        for k, v in plan["records"].items():
            print(f"  primary record {k}: {v['file']} sha256 {v['sha256']}")
        print(f"  observed request-to-measurement minutes (as recorded in the forecast): "
              f"{plan['observed_boot_minutes']}")
        for session in ("L1", "L2", "L3"):
            s = plan["sessions"][session]
            print(f"\n{session}:")
            for r in (x for x in plan["rows"] if x["session"] == session):
                src = "; ".join(x.get("field") or (f"assumption {x['assumption']}" if "assumption" in x else
                                                   f"derived: {x.get('derived')}") for x in r["sources"])
                print(f"  {r['count']:3} x {r['seconds_each']:>7} s = {r['seconds']:>8} s"
                      f"{' [conditional]' if r['conditional'] else ''} {r['label']} (commands {r['commands']}); {src}")
            print(f"  = unconditional {s['unconditional_min']} min, conditional {s['conditional_min']} min, worst from "
                  f"request {s['worst_from_request_min']} min (+ boot {ASSUMPTIONS['BOOT'][0]} + reserve "
                  f"{ASSUMPTIONS['RESERVE'][0]}) of the {s['cap_min']}-minute cap: "
                  f"{'fits' if s['fits_cap'] else 'DOES NOT FIT'}; run end: the earliest of --deadline-minutes "
                  f"from the runner start (at most {s['runner_deadline_max_min']} min), --lease-end-unix and, with "
                  f"--request-unix, the request + {s['cap_min']} min - 600 s ({s['runner_deadline_max_min']} min "
                  "from the request); only the request and lease bindings tie it to the machine's life (operations "
                  "pass both)")
            print(f"    forecast (recomputed from its rows): {s['forecast_recomputed']}; rows {s['rows']} here, "
                  f"{s['forecast_rows']} there")
            for x in s["row_differences"]:
                print(f"    differs from the forecast row {x['row']} ({x['forecast_step']}): count {x['count'][0]} "
                      f"here / {x['count'][1]} there, seconds {x['seconds'][0]} / {x['seconds'][1]}")
        print(f"\ninitial fixed caps L1 + L2 + L3: {plan['initial_caps_total_min']} min; {plan['note']}")
    if args.plan_json:
        mg.write_json(args.plan_json, plan)
        print(f"plan written: {args.plan_json.name}")


# --------------------------------------------------------------------------- main


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


def sha256_arg(text):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise argparse.ArgumentTypeError(f"{text}: not a SHA-256")
    return text.lower()


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    head, _, rest = __doc__.partition("\n\n")
    ap = argparse.ArgumentParser(description=head, epilog=rest, formatter_class=HelpFormatter)
    ap.add_argument("--campaign", choices=CAMPAIGNS, help="the session of this invocation")
    ap.add_argument("--units", type=str.split, default=list(UNITS),
                    help="l1: units to run (whole units; a subset only for a reviewed revalidation)")
    ap.add_argument("--base", type=Path, help="d42a400 CUDA binary (paired build)")
    ap.add_argument("--candidate", type=Path, help="phase-05 candidate CUDA binary (paired build)")
    ap.add_argument("--expect-base-sha256", type=sha256_arg, help="required: SHA-256 the base must have")
    ap.add_argument("--expect-candidate-sha256", type=sha256_arg, help="required: SHA-256 the candidate must have")
    ap.add_argument("--base-label", help="required: declared role and source of the base (recorded as declared)")
    ap.add_argument("--candidate-label", help="required: declared role and source of the candidate")
    ap.add_argument("--base-provenance", type=Path, help="optional JSON build record of the base")
    ap.add_argument("--candidate-provenance", type=Path, help="optional JSON build record of the candidate")
    ap.add_argument("--l1-run", type=Path, action="append", default=[], help="l2/l3: an l1 raw directory, repeatable")
    ap.add_argument("--out", type=Path, help="new output directory, outside Git (e.g. .cache/cloud/...)")
    ap.add_argument("--source-commit", default=None, help="free text recorded with the source fingerprint")
    ap.add_argument("--prepare-900", type=Path, help="write the 900-question inputs to this new directory and exit")
    ap.add_argument("--prepared-900", type=Path, help="plan: a --prepare-900 directory to check against the rebuild")
    ap.add_argument("--profile-n", type=positive_int, default=40, help="basal profile --n")
    ap.add_argument("--inflight-delay", type=positive_finite, default=1.0,
                    help="C4: seconds between sending the long requests and SIGTERM")
    ap.add_argument("--port", type=positive_int, default=18500)
    ap.add_argument("--cublaslt", default="120901", help="cuBLASLt version of the bundled GEMM table names")
    ap.add_argument("--hf-home", default="/data/hf")
    ap.add_argument("--basal-home", default="/data/basal")
    ap.add_argument("--export-timeout", type=positive_int, default=1800, help="one basal export (s)")
    ap.add_argument("--bench-timeout", type=positive_int, default=900, help="one basal bench or profile (s)")
    ap.add_argument("--benchmark-timeout", type=positive_int, default=2400, help="one basal benchmark (s)")
    ap.add_argument("--ready-timeout", type=positive_int, default=600, help="server readiness (s)")
    ap.add_argument("--http-timeout", type=positive_int, default=300, help="one HTTP request (s)")
    ap.add_argument("--shutdown-timeout", type=positive_int, default=120, help="graceful server shutdown (s)")
    ap.add_argument("--helper-ready-timeout", type=positive_int, default=120, help="reservation helper ready (s)")
    ap.add_argument("--helper-stop-timeout", type=positive_int, default=30, help="reservation helper stop (s)")
    ap.add_argument("--deadline-minutes", type=positive_finite, default=None,
                    help="required for a run: minutes from the runner start (at most the campaign cap - 10)")
    ap.add_argument("--request-unix", type=positive_finite, default=None,
                    help="optional Unix time of the machine request: the run ends by request + cap - 600 s")
    ap.add_argument("--lease-end-unix", type=positive_finite, default=None,
                    help="optional absolute end (Unix time) of the runner's share of the machine's life")
    ap.add_argument("--no-weight-hash", action="store_true",
                    help="skip the SHA-256 of the weights (recorded; P0 equality and l1 identity become GAPs)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and forecast; no GPU, binary or network")
    ap.add_argument("--forecast", type=Path, help="plan: the private forecast JSON (its sources name the W4 records)")
    ap.add_argument("--plan-json", type=Path, help="plan: also write the plan as JSON to this new file")
    return ap, ap.parse_args()


def validate(ap, args, root):
    if args.port > 65535:
        ap.error("--port must be at most 65535")
    for p in (args.plan_json, args.prepare_900):
        if p is not None and (p.exists() or root / "reports" in p.resolve().parents):
            ap.error(f"{p}: must be new and outside reports/ (e.g. .cache/)")
    if args.plan_only or args.prepare_900:
        for label in ("base", "candidate"):
            path = getattr(args, f"{label}_provenance")
            if path is None:
                continue
            try:  # JSON only: never the binary
                v = read_json(path)
                got = v.get("binary_sha256") if isinstance(v, dict) else None
                if not isinstance(got, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", got):
                    raise ValueError(f"no binary_sha256: {got!r}")
                want = getattr(args, f"expect_{label}_sha256")
                if want and got.lower() != want:
                    raise ValueError(f"binary_sha256 {got} is not --expect-{label}-sha256 {want}")
            except (OSError, ValueError) as e:
                ap.error(f"--{label}-provenance: {e}")
        return
    if not args.campaign:
        ap.error("--campaign is required for a run")
    bad = [u for u in args.units if u not in UNITS]
    if bad or not args.units or len(set(args.units)) != len(args.units):
        ap.error(f"--units: {args.units} (of {' '.join(UNITS)}, each once)")
    if args.campaign != "l1" and args.units != list(UNITS):
        ap.error("--units is for l1")
    if (args.campaign == "l1") == bool(args.l1_run):
        ap.error("l2 and l3 need --l1-run, and --l1-run is only for them")
    pending = pending_contract(run_scopes(args.campaign, args.units))
    if pending:
        ap.error(f"engine-owned texts or fields not yet pinned for {args.campaign} (integration gate, the coordinator "
                 f"freezes the shared contract): {pending}")
    need = [k for k in ("out", "base", "candidate", "expect_base_sha256", "expect_candidate_sha256",
                        "base_label", "candidate_label", "deadline_minutes")
            if not getattr(args, k)]
    if need:
        ap.error(f"required for a run: {['--' + k.replace('_', '-') for k in need]}")
    cap = CAP_MINUTES["c5-only" if args.campaign == "l1" and args.units == ["C5"] else args.campaign]
    if args.deadline_minutes > cap - RESERVE_S / 60:
        ap.error(f"--deadline-minutes {args.deadline_minutes} exceeds the {args.campaign} cap {cap} min minus the "
                 f"{RESERVE_S // 60}-minute reserve")
    if not (root / SPLIT_SOURCE).is_file():
        ap.error(f"{SPLIT_SOURCE} missing: the synced tree is not the phase-05 candidate")
    drift = missing_fragments(root)
    if drift:
        ap.error(f"the synced candidate source lacks texts or fields this runner is bound to (ENGINE_FRAGMENTS; bind "
                 f"them again in a reviewed runner revision): {drift}")
    helper = root / RESERVE_HELPER_PATH
    if not helper.is_file() or mg.sha256(helper) != RESERVE_HELPER_SHA256:
        ap.error(f"{RESERVE_HELPER_PATH}: SHA-256 {mg.sha256(helper) if helper.is_file() else None}, expected "
                 f"{RESERVE_HELPER_SHA256}")
    missing = [p for p in (w8.SEQUENTIAL, w8.LOCK, w8.BUILD_BENCHMARK, w8.AB, w8.LONG, w4.P0_FACTS,
                           *(w8.DECISIONS.format(s) for s in SHORTS),
                           *(w8.REFERENCE.format(s) + "/bench.jsonl" for s in SHORTS)) if not (root / p).is_file()]
    if missing:
        ap.error(f"run in the repository root (/work): missing {missing}")


def run_campaign(ap, args, root):
    t0, w0 = time.monotonic(), time.time()  # the run's deadline counts from here
    if args.out.exists():
        ap.error(f"output already exists: {args.out}")
    if root / "reports" in args.out.resolve().parents:
        ap.error("raw output belongs outside reports/ and Git (e.g. .cache/cloud/SESSION)")
    binaries = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
    shas = {}
    for label, b in binaries.items():
        if not os.access(b, os.X_OK):
            ap.error(f"not an executable: {b}")
        want = args.expect_base_sha256 if label == "base" else args.expect_candidate_sha256
        shas[label] = mg.sha256(b)
        if shas[label] != want:
            ap.error(f"{label} binary SHA-256 is {shas[label]}, expected {want}")
    if binaries["base"] == binaries["candidate"] or shas["base"] == shas["candidate"]:
        ap.error("--base and --candidate must be different binaries")
    provenance = {}
    for label, arg in (("base", args.base_provenance), ("candidate", args.candidate_provenance)):
        if arg is not None:
            try:
                provenance[label] = w4.read_provenance(arg, shas[label])
            except ValueError as e:
                ap.error(f"--{label}-provenance: {e}")
    ends = {"deadline_minutes": t0 + args.deadline_minutes * 60}
    cap = CAP_MINUTES["c5-only" if args.campaign == "l1" and args.units == ["C5"] else args.campaign]
    if args.lease_end_unix is not None:
        ends["lease_end_unix"] = t0 + (args.lease_end_unix - w0)
    if args.request_unix is not None:
        ends["request_cap"] = t0 + (args.request_unix + cap * 60 - RESERVE_S - w0)
    deadline = min(ends.values())
    if deadline <= t0:
        ap.error(f"the run's end has passed: {sorted(k for k, v in ends.items() if v <= t0)}")
    cur_sources = w4.sources_digest(root)
    needed = set(SHORTS) if args.campaign == "l1" else {"4.5B"}
    try:
        known = mg.known_revisions(root)
        models = {s: mg.resolve_model(s, known, args.hf_home) for s in sorted(needed)}
        for s, m in models.items():
            m["layers"] = w8.layers_of(m)
            m["evidence"] = (m["dir"] / "evidence_head.pt").is_file()
            if m["layers"] != LAYERS[s]:
                raise ValueError(f"{s}: config.json has {m['layers']} layers, the fixed layouts need {LAYERS[s]}")

        def smi(query):
            if time.monotonic() >= deadline:
                raise TimeoutError("the run deadline passed before nvidia-smi started")
            rows, why = w4.bounded_wait(30, deadline, lambda limit: mg.smi(query, timeout=limit))
            if why:
                raise TimeoutError("nvidia-smi " + ("cut at the run deadline" if why["cause"] == "deadline" else
                                                    "past its own 30 s"))
            return rows

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
            ap.error(f"missing bundled GEMM table {t.name} (no GEMM search is run)")
        tables[s] = t
    l1 = None
    if args.l1_run:
        try:
            l1 = l1_evidence(args, args.out, shas, cur_sources)
        except (ValueError, KeyError, TypeError, OSError) as e:
            ap.error(f"{e}")

    sys.setswitchinterval(0.0005)
    args.out.mkdir(parents=True)
    out = args.out.resolve()
    rec = Record(out)

    class Ctx:
        pass

    ctx = Ctx()
    ctx.args, ctx.out, ctx.root, ctx.shorts = args, out, root, sorted(needed)
    ctx.env = mg.with_env(mg.environment(args), {"HF_HUB_OFFLINE": "1"})
    ctx.binary, ctx.model, ctx.tables, ctx.provenance = binaries, models, tables, provenance
    ctx.by_bus = {mg.bus(g[2]): int(g[0]) for g in gpus}
    ctx.gpu_name = gpus[0][1]
    ctx.deadline = deadline
    args.deadline_minutes = (deadline - t0) / 60  # the effective value, for the messages of the helpers
    ctx.deadline_basis = {"campaign_cap_minutes": cap, "reserve_s": RESERVE_S,
                          "ends_seconds_from_start": {k: round(v - t0, 3) for k, v in ends.items()},
                          "effective_minutes": round((deadline - t0) / 60, 3)}
    ctx.deadline_hit, ctx.watch_foreign, ctx.setup_stop = False, True, None
    ctx.exports, ctx.export_runs, ctx.export_foreign, ctx.q8_diagnostics, ctx.results = {}, {}, {}, {}, {}
    ctx.q4_objects, ctx.http_refs, ctx.helper_pids, ctx.l1 = {}, {}, set(), l1
    ctx.export_checks, ctx.http_ref_checks = {}, {}  # name -> (first, end) of its records in rec.checks
    ctx.http_ref_complete = {}  # key -> the reference session returned normally (reference_server)
    ctx.units = args.units if args.campaign == "l1" else []
    ctx.p0, ctx.p0_provenance = w4.load_p0(root)
    ctx.bench_items = {s: mg.line_count(w4.reference(ctx, s) / "bench.jsonl") for s in SHORTS}
    try:
        ctx.inputs900_summary = w8.write_900(root, out / "inputs-900", list(SHORTS)) if args.campaign == "l1" else None
    except Fatal as e:
        print(f"preflight: 900 inputs: {e}", file=sys.stderr)
        return 2
    ctx.inputs900 = {s: out / "inputs-900" / f"inputs-900-{s}" for s in SHORTS}
    (out / "gpus.csv").write_text("index,name,pci.bus_id,driver_version,memory.total_mib,power.limit_w,"
                                  "clocks.max.sm_mhz,clocks.max.mem_mhz,persistence_mode\n" +
                                  "".join(",".join(g) + "\n" for g in gpus))
    fp = ctx.fp = fingerprint(rec, args, ctx, gpus)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
              if k not in ("base", "candidate", "out", "hf_home", "basal_home", "plan_only", "l1_run", "prepare_900",
                           "prepared_900", "plan_json", "forecast", "base_provenance", "candidate_provenance")}
    params = w8.scrubbed(params)
    params["env"] = {k: ctx.env[k] for k in ("BASAL_NO_UPDATE_CHECK", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE")}
    mg.write_json(out / "run.json", {"fingerprint": fp, "parameters": params, "l1_evidence": l1,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if args.campaign == "l1":
        fns = {"C1": unit_c1, "C2": unit_c2, "C3": unit_c3, "C4": unit_c4, "C5": unit_c5, "C6": unit_c6}
        work = [(u, fns[u]) for u in UNITS if u in args.units]
    else:
        work = [(args.campaign.upper(), campaign_l2 if args.campaign == "l2" else campaign_l3)]
    units = {label: {"started": False, "completed": False} for label, _ in work}
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
            units[label].update(started=True, first_check=len(rec.checks))
            try:
                w8.guard(ctx, rec, label)
                fn(rec, ctx)
                units[label]["completed"] = not ctx.deadline_hit
            except Deadline:
                break
            except (StepFailed, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
                rec.check(label, "FAIL", iso.scrub(f"{type(e).__name__}: {e}"))
                units[label]["completed"] = True  # ran to its end with a FAIL (a FAIL refuses l2/l3 anyway)
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
        bounds = [units[label].get("first_check") for label, _ in work] + [len(rec.checks)]
        for k, (label, _) in enumerate(work):  # the checks recorded while each unit ran
            start = bounds[k]
            end = next((b for b in bounds[k + 1:] if b is not None), len(rec.checks))
            units[label]["checks"] = dict(Counter(c["status"] for c in rec.checks[start:end])) if start is not None \
                else {}
        counts = {k: sum(c["status"] == k for c in rec.checks) for k in STATUSES}
        classes = {k: sum(c["status"] == "FAIL" and c.get("class") == k for c in rec.checks) for k in FAIL_CLASSES}
        status = 1 if counts["FAIL"] or stopped else 4 if counts["SIGNAL"] else \
            3 if counts["GAP"] or not complete else 0
        portable = iso.portable_checks(rec.checks)
        for p, c in zip(portable, rec.checks):
            p["class"] = c.get("class")
        mg.write_json(out / "summary.json", {
            "complete": complete, "stopped": iso.scrub(stopped), "exit_status": status, "counts": counts,
            "fail_classes": classes, "units": units, "fingerprint": fp, "parameters": params,
            "exports": ctx.export_runs, "q4_objects": ctx.q4_objects, "results": w8.scrubbed(ctx.results),
            "helper_processes": (out / "helper-processes.jsonl").read_text().count("\n")
            if (out / "helper-processes.jsonl").is_file() else 0,
            "checks": portable, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"DONE {counts} FAIL classes {classes} exit {status}" + (f" (stopped: {stopped})" if stopped else ""),
              flush=True)
    return status


def main():
    ap, args = parse_args()
    root = Path.cwd()
    import_helpers(root)
    for sig in rp.STOP_SIGNALS:  # SystemExit(128 + signal) once; later signals ignored so the cleanup completes
        signal.signal(sig, w8.terminate)
    validate(ap, args, root)
    if args.plan_only:
        try:
            print_plan(args, root)
        except (Fatal, ValueError, OSError, KeyError) as e:
            print(f"plan: {type(e).__name__}: {e}", file=sys.stderr)
            return 2
        return 0
    if args.prepare_900:
        try:
            print(json.dumps(w8.write_900(root, args.prepare_900, list(SHORTS)), indent=1))
        except Fatal as e:
            print(f"900 inputs: {e}", file=sys.stderr)
            return 2
        return 0
    return run_campaign(ap, args, root)


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        if mg is not None:
            for p in list(mg.LIVE):
                mg.stop(p, timeout=30)
