#!/usr/bin/env python3
"""Build wheel from sdist using the single, hash-locked Flit backend. No resolver or build frontend."""

import os
from importlib.metadata import version
from pathlib import Path
import tarfile
import tempfile

from flit_core import buildapi


def main() -> None:
    if version("flit_core") != "4.1.0":
        raise SystemExit("Install the hash-locked tools/sdk/requirements-build.txt first")
    package = Path(__file__).resolve().parents[2] / "sdk/python"
    output = package / "dist"
    output.mkdir(exist_ok=True)
    os.chdir(package)
    sdist = buildapi.build_sdist(str(output))
    with tempfile.TemporaryDirectory(prefix="basal-python-build-") as temporary:
        with tarfile.open(output / sdist) as archive:
            archive.extractall(temporary, filter="data")
        directories = list(Path(temporary).iterdir())
        if len(directories) != 1 or not directories[0].is_dir():
            raise SystemExit("Unexpected source distribution structure")
        os.chdir(directories[0])
        wheel = buildapi.build_wheel(str(output))
    print(sdist)
    print(wheel)


if __name__ == "__main__":
    main()
