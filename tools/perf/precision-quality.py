"""Collect single-client answers against upstream FP32 and served upstream fast BF16.

This is a quality measurement, not a latency benchmark. Inputs and all output directories
must be frozen/new. A host guard must own the GPU, resource limits and process deadline.
"""
import argparse
import contextlib
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time
import urllib.request

STRUCTURAL_PARENTS = {"tc-static": "baseline", "tc-qk1-static": "tc-qk1", "tc-f16-static": "tc-f16",
                      "inference-balanced": "tc-qk1-static", "inference-accurate": "baseline"}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def bits(value):
    if isinstance(value, float):
        return ("f64", struct.pack("!d", value).hex())
    if isinstance(value, dict):
        return {k: bits(v) for k, v in value.items()}
    if isinstance(value, list):
        return [bits(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement-artifacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=["mini", "4.5B", "max"], required=True)
    p.add_argument("--models-root", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--reference-python", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--deadline", type=float, required=True)
    p.add_argument("--reference-variant", choices=["upstream-fp32", "native-fp32"], default="upstream-fp32",
                   help="native FP32 is supplementary evidence, distinct from the upstream FP32 reference")
    p.add_argument("--variants", nargs="+", default=["upstream-fp32", "upstream-fast", "baseline", "tc-static", "tc-qk1", "tc-f16"],
                   choices=["upstream-fp32", "upstream-fast", "baseline", "tc-static", "tc-qk1", "tc-f16", "tc-pv1",
                            "tc-qk1-static", "tc-f16-static", "native-fp32", "inference-balanced", "inference-accurate", "inference-auto"])
    a = p.parse_args()
    if not a.variants or len(a.variants) != len(set(a.variants)) or a.variants[0] != a.reference_variant:
        p.error("distinct variants must start with the selected FP32 reference")
    for child, parent in STRUCTURAL_PARENTS.items():
        if child in a.variants and (parent not in a.variants or a.variants.index(parent) > a.variants.index(child)):
            p.error(child + " requires its parent before it: " + parent)
    root = a.measurement_artifacts.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    for rel, digest in json.loads((root / "provenance.json").read_text())["files"].items():
        path = root / rel
        if not path.resolve().is_relative_to(root) or sha256(path) != digest:
            raise ValueError("frozen input hash/path differs: " + rel)
    commit = subprocess.check_output(["git", "-C", str(a.upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["upstream_commit"] or subprocess.check_output(["git", "-C", str(a.upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("upstream must match its immutable pinned checkout")
    model = "basal-2.0-" + a.model
    cases = [json.loads(x) for x in (root / (model + "-requests.jsonl")).read_text().splitlines()]
    if not cases or len({x["id"] for x in cases}) != len(cases):
        raise ValueError("empty/duplicate corpus")
    a.out.mkdir(parents=True, exist_ok=False)
    method = {"purpose": "quality-only, timings diagnostic", "model": model, "variants": a.variants,
              "cases": len(cases), "upstream_commit": commit, "native_sha256": sha256(a.binary),
              "runner_sha256": sha256(Path(__file__)), "corpus_sha256": sha256(root / (model + "-requests.jsonl")),
              "orders": 1, "state_cache": "disabled", "reference_variant": a.reference_variant,
              "fp32_reference": ("unmodified upstream eager float32; TF32 follows the upstream/PyTorch policy" if a.reference_variant == "upstream-fp32"
                                 else "native FP32, SIMT attention; Candle reduced-precision F32 GEMM defaults false; supplementary reference"),
              "bf16_reference": "unmodified upstream fast, compile/CUDA graphs; product default attention" if "upstream-fast" in a.variants else "not collected in this run",
              "structural_parents": STRUCTURAL_PARENTS, "numerical_variants": ["tc-qk1", "tc-f16", "tc-pv1", "tc-qk1-static", "tc-f16-static", "inference-balanced", "inference-auto"],
              "model_files_sha256": {n: sha256(a.models_root / model / n) for n in ["config.json", "basal.json", "CALIBRATION.json", "model.safetensors"]}}
    (a.out / "method.json").write_text(json.dumps(method, indent=2) + "\n")
    events = []
    def event(**data):
        events.append({"time": time.time(), **data})
        (a.out / "progress.json").write_text(json.dumps(events, indent=2) + "\n")
        print(json.dumps(data), flush=True)

    def remaining(reserve=30):
        left = a.deadline - time.time() - reserve
        if left <= 0:
            raise TimeoutError("quality campaign deadline reserve")
        return left

    environment = os.environ.copy()
    for key in ["BASAL_ATT", "BASAL_WGP_LOAD", "BASAL_NORM", "BASAL_HOST_TIMING", "BASAL_GEMM", "BASAL_PROFILE", "BASAL_CONFIG"]:
        environment.pop(key, None)
    environment.update(BASAL_NO_UPDATE_CHECK="1", PYTHONPATH=str(a.upstream),
                       TORCHINDUCTOR_CACHE_DIR=str(a.out.parent / "upstream-compile-cache"))

    @contextlib.contextmanager
    def server(variant):
        try:
            probe = socket.create_connection(("127.0.0.1", 18765), timeout=.5)
        except ConnectionRefusedError:
            pass
        else:
            probe.close()
            raise RuntimeError("quality port is already occupied")
        env = environment.copy()
        if variant.startswith("upstream-"):
            cmd = [str(a.reference_python), "-m", "basal.server", "--model", str(a.models_root / model),
                   "--mode", "eager" if variant == "upstream-fp32" else "fast", "--dtype",
                   "float32" if variant == "upstream-fp32" else "bfloat16", "--orders", "1", "--state-cache-gb", "0",
                   "--context-overflow", "refuse", "--host", "127.0.0.1", "--port", "18765"]
            timeout = 1800
        else:
            if not variant.startswith("inference-"):
                env["BASAL_ATT"] = "tc" if variant in ("baseline", "native-fp32") else variant
            cmd = [str(a.binary), "--color", "never", "serve", "--model", str(a.models_root / model),
                   "--orders", "1", "--state-cache-mb", "0", "--addr", "127.0.0.1:18765", "--max-large-inflight", "128"]
            if variant.startswith("inference-"):
                cmd += ["--inference-mode", variant.removeprefix("inference-")]
            if variant == "native-fp32":
                cmd += ["--dtype", "f32"]
            else:
                cmd += ["--gemm-table", str(root / "gemm-tables" / (model + "-f16.json"))]
            timeout = 180
        with (a.out / (variant + "-server.log")).open("x") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            begin = time.monotonic()
            try:
                until = time.monotonic() + min(timeout, remaining(60))
                while process.poll() is None and time.monotonic() < until:
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:18765/health", timeout=.5) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(.5)
                else:
                    raise RuntimeError("server startup failed: " + variant)
                event(stage="ready", variant=variant, startup_s=time.monotonic()-begin)
                yield
                if process.poll() is not None:
                    raise RuntimeError("server exited during quality measurement")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                event(stage="stopped", variant=variant, exit_code=process.returncode)

    gold = {}
    parents = {parent: {} for parent in STRUCTURAL_PARENTS.values()}
    try:
        with (a.out / "observations.jsonl").open("x") as output:
            for variant in a.variants:
                with server(variant):
                    conn = http.client.HTTPConnection("127.0.0.1", 18765)
                    try:
                        for i, case in enumerate(cases):
                            conn.timeout = min(300, remaining())
                            if conn.sock is not None:
                                conn.sock.settimeout(conn.timeout)
                            row = {"variant": variant, "id": case["id"], "cohort": case["cohort"], "valid": False}
                            try:
                                body = json.dumps(dict(case["request"], model=model), ensure_ascii=False).encode()
                                conn.request("POST", "/v1/basal", body, {"Content-Type": "application/json"})
                                response = conn.getresponse()
                                raw = response.read()
                                value = json.loads(raw)
                                row.update(http_status=response.status, response=value)
                                usage, answers = value.get("usage", {}), value.get("answers")
                                if response.status != 200 or not isinstance(answers, dict) or set(answers) != set(case["request"]["questions"]) or not finite(value) or usage.get("truncated"):
                                    raise ValueError("invalid response, nonfinite value or truncated input")
                                if variant == a.reference_variant:
                                    gold[case["id"]] = value
                                elif usage.get("input_tokens") != gold[case["id"]]["usage"].get("input_tokens"):
                                    raise ValueError("input-token count differs from FP32")
                                if variant in parents:
                                    parents[variant][case["id"]] = answers
                                if variant in STRUCTURAL_PARENTS:
                                    parent = STRUCTURAL_PARENTS[variant]
                                    row["parent_bitwise_equal"] = bits(parents[parent].get(case["id"])) == bits(answers)
                                    if not row["parent_bitwise_equal"]:
                                        raise ValueError("structural candidate differs from parent: " + parent)
                                row["valid"] = True
                            except Exception as error:
                                row.update(error=type(error).__name__, detail=str(error))
                                raise
                            finally:
                                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                                output.flush()
                            if (i+1) % 25 == 0:
                                event(stage="answers", variant=variant, completed=i+1, total=len(cases))
                    finally:
                        conn.close()
        (a.out / "complete.json").write_text(json.dumps({"execution_finished": True, "cases": len(cases), "variants": a.variants}) + "\n")
    except BaseException as error:
        event(stage="failed", error=type(error).__name__, detail=str(error))
        raise


if __name__ == "__main__":
    main()
