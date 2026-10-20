#!/usr/bin/env python3
"""Capture an existing batch and service health for a manual operator qualification.

Read-only API client, no inference generator, synthetic results, assertions or CI gate.
Captured files are private raw data. Review/redact them before preparing a public archive.
"""

import argparse
import json
import math
import os
from pathlib import Path
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


TERMINAL = {"completed", "failed", "expired", "cancelled"}


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer token to a redirect target.
        return None


class Capture:
    def __init__(self, args, token, output):
        self.args = args
        self.token = token
        self.output = output
        self.client = build_opener(NoRedirects())
        self.started = time.monotonic()
        self.events = (output / "observations.jsonl").open("x", encoding="utf-8")

    def observation(self, endpoint, status, body=None, error=None):
        self.events.write(json.dumps({
            "elapsed_ns": time.monotonic_ns() - self.started_ns,
            "endpoint": endpoint,
            "http_status": status,
            "body": body,
            "error": error,
        }, ensure_ascii=False) + "\n")
        self.events.flush()

    def response(self, path):
        request = Request(self.args.base_url.rstrip("/") + path,
                          headers={"Authorization": "Bearer " + self.token})
        return self.client.open(request, timeout=self.args.timeout)

    def read_json(self, path, label):
        try:
            with self.response(path) as response:
                raw = response.read(self.args.max_json_bytes + 1)
                if len(raw) > self.args.max_json_bytes:
                    self.observation(label, response.status, error="json_capture_limit")
                    return None
                body = json.loads(raw)
                self.observation(label, response.status, body=body)
                return body
        except HTTPError as error:
            self.observation(label, error.code, error="http_error")
            error.close()
        except (URLError, TimeoutError, OSError, ValueError):
            # Exception text can contain URLs or filesystem details; retain only the category.
            self.observation(label, None, error="transport_or_json_error")
        return None

    def download(self, file_id, name):
        label = name + "_download"
        partial = self.output / (name + ".jsonl.part")
        size = 0
        try:
            with self.response("/v1/files/" + quote(file_id, safe="") + "/content") as response:
                with partial.open("xb") as dest:
                    while True:
                        chunk = response.read(min(65536, self.args.max_file_bytes - size + 1))
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > self.args.max_file_bytes:
                            self.observation(label, response.status, error="file_capture_limit")
                            return
                        dest.write(chunk)
                partial.rename(self.output / (name + ".jsonl"))
                self.observation(label, response.status, body={"captured_bytes": size})
        except HTTPError as error:
            self.observation(label, error.code, error="http_error")
            error.close()
        except (URLError, TimeoutError, OSError):
            self.observation(label, None, error="transport_or_file_error")

    def run(self):
        self.started_ns = time.monotonic_ns()
        deadline = self.started + self.args.duration
        try:
            while True:
                batch = self.read_json("/v1/batches/" + quote(self.args.batch_id, safe=""), "batch")
                self.read_json("/health/batches", "health")
                if isinstance(batch, dict) and batch.get("status") in TERMINAL:
                    for field, name in (("output_file_id", "output"), ("error_file_id", "error")):
                        file_id = batch.get(field)
                        if isinstance(file_id, str) and file_id:
                            self.download(file_id, name)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, self.args.interval))
        finally:
            self.events.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, required=True, help="plaintext bearer token, mode 0600")
    parser.add_argument("--batch-id", required=True, help="already submitted batch; this tool never creates work")
    parser.add_argument("--output", type=Path, required=True, help="new private directory outside Git")
    parser.add_argument("--duration", type=float, default=60, help="sampling duration in seconds; requests can add time")
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=10, help="socket timeout for each operation")
    parser.add_argument("--max-json-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-file-bytes", type=int, default=200 * 1024 * 1024)
    args = parser.parse_args()
    location = urlsplit(args.base_url)
    if (location.scheme not in {"http", "https"} or not location.hostname
            or location.username is not None or location.password is not None
            or location.path not in {"", "/"} or location.query or location.fragment):
        parser.error("base URL must be an HTTP(S) origin without credentials")
    if location.scheme == "http" and location.hostname not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("a remote origin requires HTTPS with certificate verification")
    if any(not math.isfinite(value) or value <= 0 for value in (args.duration, args.interval, args.timeout)):
        parser.error("durations must be finite and positive")
    if args.max_json_bytes <= 0 or args.max_file_bytes <= 0:
        parser.error("capture limits must be positive")
    if not args.batch_id or len(args.batch_id) > 256:
        parser.error("batch id must contain between 1 and 256 characters")
    os.umask(0o077)
    try:
        meta = args.token_file.stat()
        if not stat.S_ISREG(meta.st_mode) or meta.st_mode & 0o077:
            parser.error("token file must be a private regular file (0600)")
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token or len(token) > 8192 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            parser.error("token file must contain one nonempty ASCII bearer token")
        args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
        Capture(args, token, args.output).run()
    except OSError:
        parser.exit(1, "capture failed: private file or output directory unavailable\n")
    except KeyboardInterrupt:
        parser.exit(130, "capture interrupted; retain partial private observations\n")
    print("Capture finished. Observations require operator review; no qualification verdict was assigned.")


if __name__ == "__main__":
    main()
