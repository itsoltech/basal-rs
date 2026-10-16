"""Compare opt-in CUDA attention candidates on an existing GPU; never rents or promotes a default.

Inputs (--measurement-artifacts): manifest.json, requests.jsonl from the latency ladder,
short.jsonl, mixed-mini.jsonl, mixed.jsonl, and <model>-fp32-reference/ from the five-GPU
campaign, gemm-tables/ with the frozen original tables, and provenance.json (files -> SHA256).
Optional gemm-retuned uses retuned-gemm-tables/ and --retuned-attention. Freeze a new input
pack and provenance after tuning; never overwrite a completed pack. Keep provenance alongside inputs. Outputs must be new and remain outside Git.
Screen is a single exploratory pass; confirm accepts two variants (ABBAABBA) or three (ABCCBAABCCBA,
ABBAABBA for each pair), discarding r1 per variant in downstream analysis. An accurate-first HTTP schedule
requires --baseline-control, which seeds all input answers from the frozen parent outside timing aggregates.
Quality exports every requested
variant in single/tree/budget/no-prefix/cache modes and preserves FP32 comparisons.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import statistics
import struct
import subprocess
import threading
import time
import urllib.request

VARIANTS = {
    "baseline": None, "default": None, "accurate": None, "auto": None, "balanced": None,
    "tc64": "tc64", "tc-sw128": "tc-sw128", "tc64-sw128": "tc64-sw128",
    "gemm-retuned": None, "tc-pv1": "tc-pv1", "tc-qk1": "tc-qk1", "tc-f16": "tc-f16",
    "upstream-fast": None, "batch-budget": "tc", "tc-static": "tc-static",
    "tc-qk1-static": "tc-qk1-static", "tc-f16-static": "tc-f16-static",
}
NUMERICAL_VARIANTS = {"default", "auto", "balanced", "tc-pv1", "tc-qk1", "tc-f16", "tc-qk1-static", "tc-f16-static"}
PROFILE_VARIANTS = {"accurate", "auto", "balanced"}


def bits(value):
    if isinstance(value, float):
        return ("f64", struct.pack("!d", value).hex())
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bits(v) for v in value]
    return value


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement-artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--models-root", type=Path, required=True)
    ap.add_argument("--phase", choices=["screen", "confirm", "quality"], required=True)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=["baseline", "default", "tc64", "tc-sw128", "tc64-sw128"])
    ap.add_argument("--reference-python", type=Path, help="Python interpreter of a pinned upstream environment")
    ap.add_argument("--reference-startup-timeout", type=int, default=1200,
                    help="upstream preparation timeout in seconds (60..3600); outside HTTP timing")
    ap.add_argument("--upstream", type=Path, help="unmodified pinned upstream checkout for upstream-fast")
    ap.add_argument("--models", nargs="+", choices=["mini", "4.5B", "max"], default=["mini", "4.5B", "max"])
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32],
                    help="ascending closed-loop client counts; up to512; use a guarded ascending ramp")
    ap.add_argument("--mixed-clients", nargs="+", type=int, help="explicit mixed client counts; ascending subset of --concurrency")
    ap.add_argument("--cooling-timeout", type=int, default=600, help="maximum seconds per temperature wait")
    ap.add_argument("--no-clock-lock", action="store_true", help="leave SM clocks unchanged on a shared host")
    ap.add_argument("--start-temperature", type=int, help="cool before each ladder shape, load phase and export; waits excluded from HTTP timings")
    ap.add_argument("--skip-ladder", action="store_true", help="HTTP throughput-only run; record no single-request ladder coverage")
    ap.add_argument("--baseline-control", action="store_true",
                    help="collect all HTTP input IDs from the frozen baseline before an accurate-first schedule; excluded from timing aggregates")
    ap.add_argument("--short-samples", type=int, help="requests per short-client phase; positive multiple of the short corpus size (default screen88/confirm264)")
    ap.add_argument("--max-batch-tokens", type=int, help="explicit native server batch token budget; default8192; does not change export batching")
    ap.add_argument("--max-large-inflight", type=int, help="explicit large/expanded request limit; requires a binary exposing the serve option; default leaves the binary's limit unchanged")
    ap.add_argument("--retuned-attention", choices=["default", "tc64", "tc-sw128", "tc64-sw128", "tc-pv1", "tc-qk1", "tc-f16"], default="default",
                    help="attention for gemm-retuned, using retuned-gemm-tables/<model>-f16.json")
    ap.add_argument("--deadline", type=float, required=True, help="absolute UNIX deadline, before VM expiry")
    a = ap.parse_args()
    if not a.concurrency or a.concurrency != sorted(set(a.concurrency)) or a.concurrency[0] != 1 or a.concurrency[-1] > 512:
        ap.error("concurrency must be ascending distinct values from1 through512, starting at1")
    if a.mixed_clients is not None and (not a.mixed_clients or a.mixed_clients != sorted(set(a.mixed_clients)) or not set(a.mixed_clients) <= set(a.concurrency)):
        ap.error("mixed-clients must be an ascending subset of concurrency")
    if not 1 <= a.cooling_timeout <= 1800:
        ap.error("cooling-timeout must be from1 through1800 seconds")
    if len(set(a.variants)) != len(a.variants):
        ap.error("variants must be distinct")
    if a.start_temperature is not None and not 20 <= a.start_temperature <= 65:
        ap.error("start-temperature must be from20 through65 Celsius")
    if a.max_batch_tokens is not None and a.max_batch_tokens <= 0:
        ap.error("max-batch-tokens must be positive")
    if a.max_large_inflight is not None and a.max_large_inflight <= 0:
        ap.error("max-large-inflight must be positive")
    if not 60 <= a.reference_startup_timeout <= 3600:
        ap.error("reference-startup-timeout must be from60 through3600 seconds")
    numerical_variants = {v for v in a.variants if (a.retuned_attention if v == "gemm-retuned" else v) in NUMERICAL_VARIANTS}
    if "upstream-fast" in a.variants:
        if a.phase == "quality" or a.reference_python is None or a.upstream is None:
            ap.error("upstream-fast requires HTTP phase, reference-python and upstream")
        if not a.reference_python.is_file() or not a.upstream.is_dir():
            ap.error("upstream environment missing")
    if a.phase == "confirm" and len(a.variants) not in [2, 3]:
        ap.error("confirm requires two or three variants")
    if a.phase == "quality" and a.variants[0] != "baseline":
        ap.error("quality requires baseline first")
    if a.phase != "quality" and a.variants[0] != "baseline" and not (a.variants[0] == "accurate" and a.baseline_control):
        ap.error("HTTP requires baseline first, or accurate first with --baseline-control")
    if a.baseline_control and (a.phase == "quality" or a.variants[0] != "accurate"):
        ap.error("baseline-control requires an accurate-first HTTP schedule")
    for p in [a.binary, a.baseline]:
        if not p.is_file():
            ap.error(f"binary missing: {p}")
    a.binary = a.binary.resolve()
    a.baseline = a.baseline.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.measurement_artifacts / "manifest.json").read_text())
    expected_baseline = manifest["binary_sha256"]
    if hashlib.sha256(a.baseline.read_bytes()).hexdigest() != expected_baseline:
        raise ValueError("baseline binary SHA does not match the frozen latency campaign")
    reference_engine = None
    if "upstream-fast" in a.variants:
        upstream_commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
        if upstream_commit != manifest["upstream_commit"] or subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip():
            raise ValueError("upstream checkout differs from frozen reference")
        reference_engine = {"commit": upstream_commit, "mode": "fast", "dtype": "BF16", "scheduler": "auto", "max_batch": 64, "batch_window_ms": 3, "admission": "upstream default unbounded queue; bounded by closed-loop clients"}
    provenance = json.loads((a.measurement_artifacts / "provenance.json").read_text())
    for relative, sha in provenance["files"].items():
        path = a.measurement_artifacts / relative
        if not path.resolve().is_relative_to(a.measurement_artifacts.resolve()):
            raise ValueError("input provenance path escapes artifact root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("input hash differs: " + relative)
    ladder = [json.loads(x) for x in (a.measurement_artifacts / "requests.jsonl").read_text().splitlines()]
    if a.skip_ladder:
        if a.phase == "quality":
            ap.error("skip-ladder applies only to HTTP measurements")
        ladder = []
    corpora = {name: [json.loads(x) for x in (a.measurement_artifacts / (name + ".jsonl")).read_text().splitlines()]
               for name in ["short", "mixed-mini", "mixed"]}
    for name, rows in corpora.items():
        if not rows or len({x["id"] for x in rows}) != len(rows):
            raise ValueError("empty corpus or duplicate IDs: " + name)
    short_samples = a.short_samples if a.short_samples is not None else 88 if a.phase == "screen" else 264
    if a.phase != "quality" and (short_samples <= 0 or short_samples % len(corpora["short"])):
        ap.error("short-samples must be a positive multiple of the short corpus size")
    progress = []
    environment = os.environ | {"BASAL_NO_UPDATE_CHECK": "1"}
    # A variant is selected per process. Inherited experimental knobs must not contaminate the control.
    for key in ["BASAL_ATT", "BASAL_WGP_LOAD", "BASAL_NORM", "BASAL_HOST_TIMING", "BASAL_GEMM", "BASAL_PROFILE", "BASAL_CONFIG"]:
        environment.pop(key, None)
    if a.phase == "confirm":
        indices = ([0, 1, 1, 0, 0, 1, 1, 0] if len(a.variants) == 2
                   else [0, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0])
    else:
        indices = list(range(len(a.variants)))
    method = {"phase": a.phase, "variants": a.variants, "ladder_skipped": a.skip_ladder, "models": a.models,
              "numerical_variants": [v for v in a.variants if v in numerical_variants],
              "reference_engine": reference_engine,
              "startup_timeout_s": {"native": 120, "upstream-fast": a.reference_startup_timeout},
              "baseline_comparison_only_variants": ["upstream-fast"] if reference_engine else [],
              "invariance_exempt_variants": ["upstream-fast"] if reference_engine else [],
              "parity_contract": "structural variants equal baseline; numerical variants invariant within their own variant and require separate FP32 quality review; upstream differences recorded separately without a batch-invariance promise",
              "schedule": [a.variants[i] for i in indices],
              "expected_ladder": {"basal-2.0-" + suffix: [r["id"] for r in ladder if r["model"] == "basal-2.0-" + suffix] for suffix in a.models},
              "corpora": {name: [x["id"] for x in rows] for name, rows in corpora.items()}, "screen_is_exploratory": a.phase == "screen",
              "excluded_repetitions": [1] if a.phase == "confirm" else [],
              "timing": "before JSON serialization through full HTTP body read; parse afterwards",
              "ladder_samples_per_shape": 2 if a.phase == "screen" else 10,
              "short_samples_per_phase": short_samples,
              "mixed_samples_per_phase": "one pass over all IDs", "mixed_clients": a.mixed_clients if a.mixed_clients is not None else [a.concurrency[-1]] if a.phase == "screen" else [c for c in [8, 32] if c in a.concurrency],
              "short_clients": a.concurrency, "clock_lock_disabled": a.no_clock_lock,
              "baseline_control": a.baseline_control,
              "batch_budget_variant": "batch-budget requires a candidate wiring the server budget to Engine.forward_max_tokens; baseline retains the old engine default",
              "native_settings_scope": "max_inflight, max_large_inflight and max_batch_tokens apply only to the native server",
              "max_inflight": 1024,
              "max_batch_tokens": a.max_batch_tokens if a.max_batch_tokens is not None else 8192,
              "http_reference_sha256": hashlib.sha256((a.measurement_artifacts / "http-reference-answers.json").read_bytes()).hexdigest() if (a.measurement_artifacts / "http-reference-answers.json").exists() else None,
              "max_large_inflight": a.max_large_inflight if a.max_large_inflight is not None else 4,
              "large_limit_explicit": a.max_large_inflight is not None,
              "start_temperature_c": a.start_temperature, "cooling_timeout_s": a.cooling_timeout,
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cooling_scope": "before variants, ladder warmups, short warmup/client phases, mixed warmup/client phases and exports",
              "retuned_attention": a.retuned_attention, "state_cache_mb": 0, "orders": 1, "input_manifest": manifest,
              "binary_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest()
                                for k, p in [("candidate", a.binary), ("baseline", a.baseline)]}}
    (a.out / "method.json").write_text(json.dumps(method, indent=2) + "\n")

    def record(**item):
        progress.append(item)
        (a.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(item), flush=True)

    def remaining(reserve=0):
        left = a.deadline - time.time() - reserve
        if left <= 0:
            raise TimeoutError("campaign deadline reserve")
        return left

    def cooldown(tag):
        if a.start_temperature is None:
            return
        begin = time.monotonic()
        while True:
            remaining(90)
            if time.monotonic()-begin >= a.cooling_timeout:
                raise TimeoutError("cooling deadline: " + tag)
            value = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5)
            temperature = float(value.strip())
            if not math.isfinite(temperature):
                raise ValueError("invalid GPU temperature")
            if temperature <= a.start_temperature:
                record(tag=tag, stage="cooled", temperature_c=temperature, waited_s=time.monotonic()-begin)
                return
            time.sleep(2)

    def table_flags(model, variant):
        # All variants use the same frozen batch-invariant table on the approved GPU.
        directory = "retuned-gemm-tables" if variant == "gemm-retuned" else "gemm-tables"
        table = a.measurement_artifacts / directory / (model + "-f16.json")
        if not table.is_file():
            raise ValueError("required frozen GEMM table missing: " + str(table))
        return ["--gemm-table", str(table)]

    def variant_env(variant):
        env = environment.copy()
        attention = a.retuned_attention if variant == "gemm-retuned" else variant
        if variant == "upstream-fast":
            env["PYTHONPATH"] = str(a.upstream)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(a.out.parent / "upstream-compile-cache")
        if VARIANTS[attention] is not None:
            env["BASAL_ATT"] = VARIANTS[attention]
        return env

    def binary(variant):
        return str(a.baseline if variant == "baseline" else a.binary)

    def command(cmd, variant, log, limit=600):
        if variant in PROFILE_VARIANTS and cmd[0] == "export":
            cmd = [*cmd, "--inference-mode", variant]
        with log.open("x") as output:
            return subprocess.run([binary(variant), "--color", "never", *map(str, cmd)],
                                  env=variant_env(variant), stdout=output, stderr=subprocess.STDOUT,
                                  timeout=min(limit, remaining(30)), check=True)

    @contextlib.contextmanager
    def server(variant, model, tag):
        # Refuse to accidentally time a leftover server answering health.
        try:
            probe = socket.create_connection(("127.0.0.1", 18765), timeout=.5)
        except ConnectionRefusedError:
            pass
        else:
            probe.close()
            raise RuntimeError("benchmark port already occupied")
        cmd = [binary(variant), "--color", "never", "serve", "--model", str(a.models_root / model),
               "--orders", "1", "--state-cache-mb", "0", "--addr", "127.0.0.1:18765", *table_flags(model, variant)]
        if variant == "upstream-fast":
            cmd = [str(a.reference_python), "-m", "basal.server", "--model", str(a.models_root / model), "--mode", "fast", "--orders", "1", "--state-cache-gb", "0", "--context-overflow", "refuse", "--host", "127.0.0.1", "--port", "18765"]
        else:
            if variant in PROFILE_VARIANTS:
                cmd += ["--inference-mode", variant]
            if a.max_large_inflight is not None:
                cmd += ["--max-large-inflight", str(a.max_large_inflight)]
            if a.max_batch_tokens is not None:
                cmd += ["--max-batch-tokens", str(a.max_batch_tokens)]
        with (a.out / (tag + "-server.log")).open("x") as output:
            process = subprocess.Popen(cmd, env=variant_env(variant), stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            begin = time.monotonic()
            try:
                until = time.time() + min(a.reference_startup_timeout if variant == "upstream-fast" else 120, remaining(45))
                while process.poll() is None and time.time() < until:
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:18765/health", timeout=.5) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError("server startup failed")
                record(tag=tag, stage="ready", startup_s=time.monotonic()-begin)
                yield
                if process.poll() is not None:
                    raise RuntimeError("server exited during measurement")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                record(tag=tag, stage="stopped", exit_code=process.returncode)

    # Structural variants share the baseline reference. Numerical candidates have their own
    # reference across phases/repetitions and need a separate FP32 quality decision.
    references = {}
    reference_path = a.measurement_artifacts / "http-reference-answers.json"
    if reference_path.exists():
        if "http-reference-answers.json" not in provenance["files"]:
            raise ValueError("HTTP reference must be part of frozen input provenance")
        for row in json.loads(reference_path.read_text()):
            key = (row["model"], row["id"])
            if row.get("variant") != "baseline-compatible" or key in references or not isinstance(row.get("answers"), dict) or not row["answers"] or not finite(row["answers"]):
                raise ValueError("invalid or duplicate frozen HTTP reference")
            references[key] = row["answers"]
    lock = threading.Lock()

    def phase(rows, model, tag, variant, label, n, clients, output):
        latencies = []
        stop = threading.Event()
        counts = {}
        started = time.perf_counter()
        def worker(client):
            conn = http.client.HTTPConnection("127.0.0.1", 18765)
            try:
                for index in range(client, n, clients):
                    if stop.is_set():
                        return
                    row = rows[index % len(rows)]
                    conn.timeout = min(180, remaining(25))
                    if conn.sock is not None:
                        conn.sock.settimeout(conn.timeout)
                    item = {"tag": tag, "phase": label, "id": row["id"], "index": index, "clients": clients,
                            "class": row.get("class", "ladder"), "valid": False}
                    begin = time.perf_counter_ns()
                    try:
                        body = json.dumps(dict(row["request"], model=model), ensure_ascii=False).encode()
                        conn.request("POST", "/v1/basal", body, {"Content-Type": "application/json"})
                        response = conn.getresponse()
                        raw = response.read()
                        item.update(elapsed_ms=(time.perf_counter_ns()-begin)/1e6, http_status=response.status)
                        value = json.loads(raw)
                        item["response"] = value
                        answers = value.get("answers")
                        usage = value.get("usage", {})
                        expected = row.get("prompt_tokens")
                        valid = (response.status == 200 and isinstance(answers, dict) and bool(answers)
                                 and set(answers) == set(row["request"]["questions"])
                                 and all(isinstance(v, dict) and isinstance(v.get("type"), str) for v in answers.values())
                                 and finite(value)
                                 and usage.get("questions") == len(answers) and not usage.get("truncated")
                                 and (expected is None or usage.get("input_tokens") == expected))
                        if not valid:
                            raise ValueError("invalid response or token count")
                        with lock:
                            baseline_key = (model, row["id"])
                            key = (model, variant, row["id"]) if variant in numerical_variants or variant == "upstream-fast" else baseline_key
                            if baseline_key in references:
                                item["baseline_bitwise_equal"] = bits(answers) == bits(references[baseline_key])
                            item["answer_bitwise_invariant"] = key not in references or bits(answers) == bits(references[key])
                            if variant != "upstream-fast" and not item["answer_bitwise_invariant"]:
                                raise ValueError("answers differ bitwise across phase/variant")
                            references.setdefault(key, answers)
                        item["valid"] = True
                    except Exception as error:
                        stop.set()
                        item.update(elapsed_ms=(time.perf_counter_ns()-begin)/1e6, error=type(error).__name__,
                                    detail=str(error))
                        raise
                    finally:
                        with lock:
                            output.write(json.dumps(item, ensure_ascii=False) + "\n")
                            output.flush()
                            if item["valid"]:
                                latencies.append(item["elapsed_ms"])
                                counts[item["class"]] = counts.get(item["class"], 0) + 1
            finally:
                conn.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
            futures = [pool.submit(worker, client) for client in range(clients)]
            for future in futures:
                future.result()
        elapsed = time.perf_counter()-started
        # All clients have drained; no next phase overlaps GPU work.
        ordered = sorted(latencies)
        classes = {}
        # Per-class latency remains in observations; keep the summary deliberately small.
        for cls, count in counts.items():
            classes[cls] = count
        record(tag=tag, stage="phase_finished", phase=label, clients=clients, samples=len(latencies),
               elapsed_s=elapsed, requests_per_s=len(latencies)/elapsed, p50_ms=statistics.median(latencies),
               p95_ms=ordered[min(len(ordered)-1, int(.95*len(ordered)))], classes=classes,
               ids=[row["id"] for row in rows])

    query = "name,driver_version,compute_cap,memory.total,power.limit,clocks.max.sm,clocks.max.memory"
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader"], text=True).strip()
    (a.out / "hardware.json").write_text(json.dumps({"gpu_query": query, "gpu": gpu}, indent=2) + "\n")
    clock = subprocess.check_output(["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    locked = None
    if not a.no_clock_lock and shutil.which("sudo"):
        locked = subprocess.run(["sudo", "-n", "nvidia-smi", "-lgc", clock + "," + clock], capture_output=True)
    (a.out / "clock-lock.json").write_text(json.dumps({"requested_sm_mhz": clock if locked else None,
        "exit_code": locked.returncode if locked else None, "disabled": a.no_clock_lock}) + "\n")
    telemetry_log = (a.out / "telemetry.csv").open("x")
    telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "200"], stdout=telemetry_log)
    try:
        for suffix in a.models:
            model = "basal-2.0-" + suffix
            if model not in manifest["models"]:
                raise ValueError("model missing from input manifest")
            repetitions = {v: 0 for v in a.variants}
            if a.baseline_control:
                # Seed independent parent answers for every later input, including models
                # absent from a previously collected HTTP reference. The measured schedule
                # still contains only the requested variants and repetitions.
                control_rows = {}
                for row in [*corpora["short"], *corpora["mixed-mini" if suffix == "mini" else "mixed"],
                            *[x for x in ladder if x["model"] == model]]:
                    if row["id"] in control_rows and row["request"] != control_rows[row["id"]]["request"]:
                        raise ValueError("one input ID has different requests in the control corpus")
                    control_rows.setdefault(row["id"], row)
                tag = model + "-baseline-control"
                cooldown(tag)
                with server("baseline", model, tag), (a.out / (tag + "-observations.jsonl")).open("x") as output:
                    phase(list(control_rows.values()), model, tag, "baseline", "baseline-control",
                          len(control_rows), a.concurrency[-1], output)
            for variant_index in indices:
                variant = a.variants[variant_index]
                repetitions[variant] += 1
                tag = f"{model}-{variant}-r{repetitions[variant]}"
                remaining(90)
                cooldown(tag)
                if a.phase == "quality":
                    inputs = a.measurement_artifacts / (model + "-fp32-reference")
                    base = a.out / f"{model}-baseline-r1-tree"
                    bit_control = a.out / f"{model}-{variant}-r1-tree" if variant in numerical_variants else base
                    for mode in ["tree", "single", "budget", "no-prefix", "cache"]:
                        cooldown(tag + "-" + mode)
                        dest = a.out / (tag + "-" + mode)
                        flags = ["--batching", mode if mode in ["tree", "single", "budget"] else "tree"]
                        if mode == "no-prefix":
                            flags += ["--no-prefix-cache"]
                        if mode == "cache":
                            flags += ["--state-cache-mb", "512"]
                        command(["export", "--model", a.models_root/model, "--orders", "1", "--upstream",
                                 "--inputs", inputs, "--out", dest, *flags, *table_flags(model, variant)], variant, Path(str(dest)+".log"))
                        command(["compare", "--a", base, "--b", dest, "--out", str(dest)+"-compare.json"],
                                variant, Path(str(dest)+"-compare.log"))
                        command(["compare", "--a", inputs, "--b", dest, "--out", str(dest)+"-fp32.json"],
                                variant, Path(str(dest)+"-fp32.log"))
                        for name in ["bench.jsonl", "systemone.jsonl"]:
                            control = [json.loads(x) for x in (bit_control/name).read_text().splitlines()]
                            result = [json.loads(x) for x in (dest/name).read_text().splitlines()]
                            reference = [json.loads(x) for x in (inputs/name).read_text().splitlines()]
                            expected_errors = {x["id"] for x in reference if x.get("upstream_error") or x.get("error")}
                            expected_ids = [x["id"] for x in reference]
                            for rows in [control, result]:
                                if [x["id"] for x in rows] != expected_ids or not finite(rows):
                                    raise ValueError("export coverage or nonfinite values: " + tag + " " + mode)
                                if {x["id"] for x in rows if x.get("error")} != expected_errors:
                                    raise ValueError("export error IDs differ from frozen reference: " + tag + " " + mode)
                            if bits(control) != bits(result):
                                raise ValueError("export differs bitwise from its required tree control: " + tag + " " + mode)
                            record(tag=tag, stage="expected_export_rejections", mode=mode, corpus=name, ids=sorted(expected_errors))
                        baseline_equal = all(bits([json.loads(x) for x in (base/name).read_text().splitlines()]) ==
                                             bits([json.loads(x) for x in (dest/name).read_text().splitlines()])
                                             for name in ["bench.jsonl", "systemone.jsonl"])
                        record(tag=tag, stage="export_bitwise_equal", mode=mode,
                               control="own-tree" if variant in numerical_variants else "baseline-tree",
                               baseline_bitwise_equal=baseline_equal)
                    continue
                with server(variant, model, tag), (a.out/(tag+"-observations.jsonl")).open("x") as output:
                    for row in [r for r in ladder if r["model"] == model]:
                        cooldown(tag + "-" + row["id"])
                        phase([row], model, tag, variant, "ladder-warmup", 2, 1, output)
                        phase([row], model, tag, variant, "ladder", 2 if a.phase == "screen" else 10, 1, output)
                    short = corpora["short"]
                    cooldown(tag + "-short-warmup")
                    phase(short, model, tag, variant, "short-warmup", len(short), 1, output)
                    for clients in a.concurrency:
                        cooldown(tag + "-short-" + str(clients))
                        phase(short, model, tag, variant, "short", short_samples, clients, output)
                    mixed = corpora["mixed-mini" if suffix == "mini" else "mixed"]
                    cooldown(tag + "-mixed-warmup")
                    phase(mixed, model, tag, variant, "mixed-warmup", min(16, len(mixed)), 1, output)
                    for clients in method["mixed_clients"]:
                        cooldown(tag + "-mixed-" + str(clients))
                        phase(mixed, model, tag, variant, "mixed", len(mixed), clients, output)
                (a.out / "reference-answers.json").write_text(json.dumps(
                    [{"model": key[0], "id": key[-1], "variant": key[1] if len(key) == 3 else "baseline-compatible", "answers": answer}
                     for key, answer in references.items()],
                    ensure_ascii=False, indent=2) + "\n")
    except Exception as error:
        record(stage="failed", error=type(error).__name__, detail=str(error))
        raise
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait()
        telemetry_log.close()
        if locked is not None and locked.returncode == 0:
            subprocess.run(["sudo", "-n", "nvidia-smi", "-rgc"], capture_output=True)
    (a.out / "complete.json").write_text(json.dumps({"execution_finished": True, "phase": a.phase}) + "\n")


if __name__ == "__main__":
    main()
