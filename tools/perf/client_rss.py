"""Manual RSS measurement of the HTTP CLI process, without loading a model.

Build release first, then run:
  python3 tools/perf/client_rss.py --binary target/release/basal --out reports/NEW-DIRECTORY

The loopback HTTP fixture and stdin/stdout drivers live in this Python process;
only the child CLI's RSS is measured. os.wait4 supplies the lifetime peak;
ps samples show whether RSS grows after drained batches. No inference or model
quality is measured. This is a manual measurement, not a CI/test framework.
"""

import argparse
import hashlib
import http.server
import json
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
from pathlib import Path


MIB = 1024 * 1024


class Server(http.server.ThreadingHTTPServer):
    # TCPServer defaults to five pending connections, below the CLI's measured concurrency.
    request_queue_size = 256
    daemon_threads = True


class Fixture(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200, b'{"models":[{"name":"rss-fixture"}]}')

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        if size > 70 * MIB:
            self.close_connection = True
            self.reply(413, b"{}")
            return
        body = self.rfile.read(size)
        if self.server.slow_first:
            request = json.loads(body)
            if request["state"].get("id") == 0:
                time.sleep(2)
        if self.server.delay:
            time.sleep(self.server.delay)
        self.reply(self.server.status, self.server.response)


class Drain:
    """Discard stdout immediately; track ordered records without retaining them."""

    def __init__(self, pipe, value):
        self.pipe, self.value = pipe, value
        self.condition = threading.Condition()
        self.count = self.errors = self.bytes = 0
        self.failure = None
        self.error_examples = []
        self.first_at = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            for line in self.pipe:
                if self.value:
                    if json.loads(line) != 0.75:
                        raise ValueError("unexpected scalar answer")
                else:
                    record = json.loads(line)
                    if record["index"] != self.count:
                        raise ValueError("unordered output index")
                    self.errors += not record["ok"]
                    if not record["ok"] and len(self.error_examples) < 3:
                        self.error_examples.append(record["error"])
                with self.condition:
                    self.count += 1
                    self.bytes += len(line)
                    if self.first_at is None:
                        self.first_at = time.monotonic()
                    self.condition.notify_all()
        except Exception as error:
            self.failure = str(error)
        finally:
            with self.condition:
                self.condition.notify_all()

    def until(self, count):
        deadline = time.monotonic() + 120
        with self.condition:
            while self.count < count and not self.failure:
                if not self.thread.is_alive():
                    raise RuntimeError(
                        f"stdout closed at {self.count}, expected {count}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"waiting for output {count}, got {self.count}")
                self.condition.wait(min(remaining, 0.5))
        if self.failure:
            raise RuntimeError(self.failure)


def line_bytes(kind, size, dense_rows=0):
    if kind == "invalid":
        return b'{"broken":\n'
    if kind == "jsonl":
        if dense_rows:
            rows = [{"n": i, "ok": True, "tag": "a"} for i in range(dense_rows)]
            return (
                json.dumps({"id": 1, "rows": rows}, separators=(",", ":")).encode()
                + b"\n"
            )
        return (
            json.dumps({"id": 1, "text": "x" * size}, separators=(",", ":")).encode()
            + b"\n"
        )
    if kind == "text":
        return b"x" * size
    return b"x" * size + b"\n"


def measure(
    binary,
    out,
    spec,
    repeat,
    scan_leaks=False,
    initial_idle=0.5,
    sample_rss=True,
    heap=False,
):
    name = f"{spec['name']}-{repeat}"
    response = {
        "model": "rss-fixture",
        "answers": {"result": {"type": "noul", "noul": 0.75}},
    }
    if spec.get("response_bytes"):
        response["padding"] = "r" * spec["response_bytes"]
    server = Server(("127.0.0.1", 0), Fixture)
    server.response = json.dumps(response, separators=(",", ":")).encode()
    server.status = spec.get("status", 200)
    server.slow_first = spec.get("slow_first", False)
    server.delay = spec.get("delay", 0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    kind = spec["kind"]
    value = spec.get("value", kind == "text")
    cmd = [
        str(binary),
        "client",
        "--ask",
        "Is this urgent?",
        "--input",
        "jsonl" if kind == "invalid" else kind,
        "--jobs",
        str(spec["jobs"]),
        "--url",
        f"http://127.0.0.1:{server.server_port}",
        "--model",
        "rss-fixture",
        "--max-input-bytes",
        str(spec.get("limit", 2 * MIB)),
        "--timeout",
        "15",
    ]
    if value:
        cmd += ["--value"]
    if spec.get("errors"):
        cmd += ["--keep-going"]
    # Never inherit authentication, proxy or Rust backtrace settings into a fixture run.
    env = {
        key: val
        for key, val in os.environ.items()
        if not key.upper().endswith("_PROXY")
        and not key.startswith("BASAL_")
        and key not in ("RUST_BACKTRACE", "RUST_LIB_BACKTRACE")
    }
    samples, phases, wait_result = [], [], {}
    leak_scan = None
    heap_snapshot = None
    stop = threading.Event()
    start = time.monotonic()
    with (out / f"{name}.stderr").open("wb") as stderr:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=env
        )
        drain = Drain(proc.stdout, value)

        def reap():
            _, status, usage = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            wait_result.update(
                exit_code=proc.returncode,
                peak_rss_bytes=usage.ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                cpu_user_s=usage.ru_utime,
                cpu_system_s=usage.ru_stime,
            )
            stop.set()

        def sample():
            if not sample_rss:
                return
            while not stop.is_set():
                result = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    samples.append(
                        {
                            "at_s": time.monotonic() - start,
                            "rss_bytes": int(result.stdout.strip()) * 1024,
                        }
                    )
                stop.wait(0.1)

        waiter = threading.Thread(target=reap, daemon=True)
        sampler = threading.Thread(target=sample, daemon=True)
        waiter.start()
        sampler.start()
        try:
            # Keep stdin open before and after drained phases: stable RSS is measured in the same process.
            time.sleep(initial_idle)
            count = 0
            for phase in spec["phases"]:
                phase_start = time.monotonic()
                line = line_bytes(kind, phase["bytes"], spec.get("dense_rows", 0))
                first = line
                if server.slow_first and count == 0:
                    first = line.replace(b'"id":1', b'"id":0', 1)
                # Bounded producer memory and actual pipe backpressure, including oversize records.
                for i in range(phase["records"]):
                    proc.stdin.write(first if i == 0 else line)
                proc.stdin.flush()
                count += phase["records"]
                if kind == "text":
                    proc.stdin.close()  # Whole text is framed by EOF.
                drain.until(count)
                drained_at = time.monotonic()
                if kind != "text":
                    time.sleep(0.8)
                idle = [
                    s["rss_bytes"]
                    for s in samples
                    if s["at_s"] >= drained_at - start + 0.2
                ]
                elapsed = drained_at - phase_start
                phases.append(
                    {
                        **phase,
                        "input_record_bytes": len(line),
                        "records_total": count,
                        "elapsed_s": elapsed,
                        "records_per_s": phase["records"] / elapsed,
                        "wall_ms_per_record": elapsed * 1000 / phase["records"],
                        "drained_at_s": drained_at - start,
                        "idle_rss_bytes": statistics.median(idle) if idle else None,
                    }
                )
            if heap and not proc.stdin.closed:
                snapshot = subprocess.run(
                    ["vmmap", "-summary", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                # Keep allocator totals, not process paths, PIDs or addresses from the native header.
                lines = snapshot.stdout.splitlines()
                relevant = [
                    line for line in lines if line.startswith("Physical footprint")
                ]
                for i, line in enumerate(lines):
                    if line.startswith("MALLOC ZONE"):
                        relevant += [
                            re.sub(r"_0x[0-9a-fA-F]+", "", item) for item in lines[i:]
                        ]
                        break
                (out / f"{name}.heap.txt").write_text(
                    "\n".join(relevant).rstrip() + "\n"
                )
                heap_snapshot = {
                    "exit_code": snapshot.returncode,
                    "output_file": f"{name}.heap.txt",
                }
            if scan_leaks and not proc.stdin.closed:
                scan = subprocess.run(
                    ["leaks", "--quiet", "--noContent", "--nostacks", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                (out / f"{name}.leaks.txt").write_text(
                    (scan.stdout + scan.stderr).rstrip() + "\n"
                )
                leak_scan = {
                    "exit_code": scan.returncode,
                    "output_file": f"{name}.leaks.txt",
                }
            if not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            if not stop.wait(20):
                raise TimeoutError("CLI did not exit after EOF")
            waiter.join()
            drain.thread.join(5)
            if drain.thread.is_alive() or drain.failure:
                raise RuntimeError(drain.failure or "stdout drain did not finish")
            expected = 1 if spec.get("errors") else 0
            if wait_result["exit_code"] != expected or drain.count != count:
                raise RuntimeError(
                    f"unexpected exit/output: {wait_result}, records={drain.count}/{count}"
                )
            if drain.errors != spec.get("errors", 0):
                raise RuntimeError(f"unexpected error count {drain.errors}")
        except BaseException as error:
            stop.wait(1)
            raise RuntimeError(
                f"case={name}; child={wait_result}; stdout_records={drain.count}; "
                f"stdout_failure={drain.failure}; errors={drain.error_examples}; "
                f"stderr={name}.stderr"
            ) from error
        finally:
            if not stop.is_set():
                # Popen.kill() polls/reaps first, which would race with the sole wait4 owner above.
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                waiter.join(5)
            sampler.join(2)
            proc.stdout.close()
            if not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            server.shutdown()
            server.server_close()
    result = {
        "name": spec["name"],
        "repeat": repeat,
        "jobs": spec["jobs"],
        "kind": kind,
        "response_bytes": len(server.response),
        "status": server.status,
        "slow_first": server.slow_first,
        "server_delay_s": server.delay,
        "elapsed_s": time.monotonic() - start,
        "output_records": drain.count,
        "output_errors": drain.errors,
        "output_bytes": drain.bytes,
        "time_to_first_output_s": drain.first_at - start if drain.first_at else None,
        "phases": phases,
        "rss_samples": samples,
        "leak_scan": leak_scan,
        "heap_snapshot": heap_snapshot,
        **wait_result,
    }
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "name": name,
                "peak_mib": round(result["peak_rss_bytes"] / MIB, 2),
                "idle_mib": [
                    round(p["idle_rss_bytes"] / MIB, 2) if p["idle_rss_bytes"] else None
                    for p in phases
                ],
                "records": drain.count,
                "seconds": round(result["elapsed_s"], 2),
            }
        ),
        flush=True,
    )
    return result


def cases():
    def case(name, kind, size, records, jobs=1, **extra):
        return dict(
            name=name,
            kind=kind,
            jobs=jobs,
            phases=[dict(bytes=size, records=records)],
            **extra,
        )

    result = [
        case("text-64B-value", "text", 64, 1),
        case("text-1MiB-value", "text", MIB, 1),
    ]
    for jobs in (1, 8, 32):
        result += [
            case(f"lines-256B-j{jobs}", "lines", 256, 5000, jobs),
            case(f"jsonl-1MiB-j{jobs}", "jsonl", MIB, 64, jobs),
        ]
    result += [
        case("jsonl-1KiB-j8", "jsonl", 1024, 5000, 8),
        case("response-1MiB-j8", "lines", 256, 64, 8, response_bytes=MIB),
        case("slow-first-1MiB-j32", "jsonl", MIB, 64, 32, slow_first=True),
        case("paced-1MiB-j32", "jsonl", MIB, 64, 32, delay=0.05),
        case("invalid-json-20k", "invalid", 0, 20000, 8, errors=20000),
        case("http-503-5k", "lines", 256, 5000, 8, status=503, errors=5000),
        dict(
            name="oversize-then-small",
            kind="lines",
            jobs=8,
            errors=8,
            phases=[dict(bytes=3 * MIB, records=8), dict(bytes=256, records=5000)],
        ),
        dict(
            name="soak-120k-j8",
            kind="lines",
            jobs=8,
            phases=[dict(bytes=256, records=20000) for _ in range(6)],
        ),
        dict(
            name="large-small-cycles-j8",
            kind="jsonl",
            jobs=8,
            phases=[dict(bytes=1024, records=1000)]
            + [
                phase
                for _ in range(3)
                for phase in (
                    dict(bytes=MIB, records=64),
                    dict(bytes=1024, records=5000),
                )
            ],
        ),
    ]
    result += [
        case("jsonl-dense-20k-objects-j8", "jsonl", 0, 32, 8, dense_rows=20000),
        case("response-12MiB-j8", "lines", 256, 32, 8, response_bytes=12 * MIB),
        case("response-12MiB-j1", "lines", 256, 32, response_bytes=12 * MIB),
        case("jsonl-1MiB-value-j8", "jsonl", MIB, 64, 8, value=True),
        dict(
            name="large-small-20-cycles-j8",
            kind="jsonl",
            jobs=8,
            phases=[dict(bytes=1024, records=1000)]
            + [
                phase
                for _ in range(20)
                for phase in (
                    dict(bytes=MIB, records=64),
                    dict(bytes=1024, records=1000),
                )
            ],
        ),
    ]
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/basal"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--case", action="append", help="run only named cases (repeatable)"
    )
    parser.add_argument(
        "--leaks",
        action="store_true",
        help="macOS: scan live CLI after final drained phase; run separately",
    )
    parser.add_argument(
        "--heap",
        action="store_true",
        help="macOS: collect final vmmap allocator totals; run separately",
    )
    parser.add_argument("--initial-idle", type=float, default=0.5)
    parser.add_argument(
        "--no-samples", action="store_true", help="diagnostic: lifetime peak only"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.initial_idle < 0:
        parser.error("--initial-idle must be nonnegative")
    if (args.leaks or args.heap) and sys.platform != "darwin":
        parser.error("--leaks/--heap require macOS")
    selected = [c for c in cases() if not args.case or c["name"] in args.case]
    if not selected or (args.case and set(args.case) - {c["name"] for c in selected}):
        parser.error("unknown case")
    binary = args.binary.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "tool": "tools/perf/client_rss.py",
        "os": platform.system(),
        "os_version": platform.mac_ver()[0]
        if sys.platform == "darwin"
        else platform.release(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "repeats": args.repeats,
        "cases": [c["name"] for c in selected],
        "leaks_scan": args.leaks,
        "heap_snapshot": args.heap,
        "samples_enabled": not args.no_samples,
        "initial_idle_s": args.initial_idle,
        "peak_method": "os.wait4 ru_maxrss (macOS bytes, Linux KiB converted to bytes)",
        "samples_method": "ps RSS KiB converted to bytes, nominal interval 100 ms",
        "limits": "Synthetic HTTP fixture; no inference, GPU, local Engine, TLS, or quality measurement. "
        "Throughput is fixture/driver limited; wall_ms_per_record is amortized batch time, "
        "not request latency. No allocator/live-allocation instrumentation.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    for repeat in range(1, args.repeats + 1):
        for spec in selected:
            results.append(
                measure(
                    binary,
                    args.out,
                    spec,
                    repeat,
                    args.leaks,
                    args.initial_idle,
                    not args.no_samples,
                    args.heap,
                )
            )
    summary = [
        {k: v for k, v in result.items() if k != "rss_samples"} for result in results
    ]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
