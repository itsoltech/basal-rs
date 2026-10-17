#!/usr/bin/env python3
"""Collect only the expected SDK distributions and their SHA-256 hashes."""

import argparse
import hashlib
from pathlib import Path
import shutil

from sdk_versions import SDKS, read_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", choices=SDKS)
    sdk = parser.parse_args().sdk
    root = Path(__file__).resolve().parents[2]
    version = read_version(root, sdk)
    expected = {
        "python": [(root / "sdk/python/dist", f"basal_client-{version}-py3-none-any.whl"),
                   (root / "sdk/python/dist", f"basal_client-{version}.tar.gz")],
        "typescript": [(root / "sdk/typescript/dist", f"itsoltech-basal-client-{version}.tgz")],
        "csharp": [(root / "sdk/csharp/dist", f"ITSOL.Basal.Client.{version}.nupkg")],
        "rust": [(root / "target/package", f"basal-client-{version}.crate")],
    }[sdk]
    artifact = root / "sdk" / sdk / "artifact"
    if artifact.is_symlink():
        raise SystemExit("Artifact directory must not be a symlink")
    if artifact.exists():
        shutil.rmtree(artifact)
    (artifact / "packages").mkdir(parents=True)
    checksums = []
    for directory, filename in expected:
        source = directory / filename
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"Missing regular distribution: {filename}")
        data = source.read_bytes()
        (artifact / "packages" / filename).write_bytes(data)
        checksums.append(f"{hashlib.sha256(data).hexdigest()}  packages/{filename}")
    (artifact / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"Collected {sdk} distributions with SHA-256")


if __name__ == "__main__":
    main()
