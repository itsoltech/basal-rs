#!/usr/bin/env python3
"""Verify a downloaded SDK artifact, rejecting unexpected files and checksum paths."""

import argparse
import hashlib
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    expected = set()
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (packages/[A-Za-z0-9_.-]+)", line)
        if not match or match[2] in expected:
            raise SystemExit("Invalid checksum manifest")
        digest, filename = match.groups()
        path = directory / filename
        if path.is_symlink() or not path.is_file() or path.parent.is_symlink():
            raise SystemExit("Artifact must contain regular files")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Artifact checksum mismatch: {filename}")
        expected.add(filename)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if not expected or actual != expected | {"SHA256SUMS"}:
        raise SystemExit("Artifact contains missing or unexpected files")
    print(f"Verified {len(expected)} distribution hashes")


if __name__ == "__main__":
    main()
