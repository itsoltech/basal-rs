#!/usr/bin/env python3
"""Check independent SDK manifest/lock versions and a selected SDK release tag."""

import argparse
import json
from pathlib import Path
import re

from sdk_versions import SDKS, STABLE_VERSION, parse_tag, read_version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", choices=SDKS, help="Check only this package; default: check each independently")
    parser.add_argument("--tag", help="Require sdk-<sdk>-vX.Y.Z to match the selected manifest")
    parser.add_argument("--version-only", action="store_true", help="Print only the selected package version")
    parser.add_argument("--github-output", type=Path, help="Append selected sdk/version to a GitHub job output file")
    args = parser.parse_args()
    sdk = args.sdk
    tag_version = None
    if args.tag is not None:
        tag_sdk, tag_version = parse_tag(args.tag)
        if sdk is not None and sdk != tag_sdk:
            raise SystemExit(f"Tag selects {tag_sdk}, but --sdk selects {sdk}")
        sdk = tag_sdk
    if sdk is None and (args.version_only or args.github_output):
        parser.error("--version-only/--github-output require --sdk or --tag")
    root = Path(__file__).resolve().parents[2]
    versions = {name: read_version(root, name) for name in ((sdk,) if sdk else SDKS)}
    if tag_version is not None and versions[sdk] != tag_version:
        raise SystemExit(f"Expected sdk-{sdk}-v{versions[sdk]}, got {args.tag!r}")
    if sdk is None:
        # The maintained API description has its own version, independent of packages.
        contract = json.loads((root / "contracts/basal/openapi.json").read_text())["info"]["version"]
        if not isinstance(contract, str) or not re.fullmatch(STABLE_VERSION, contract):
            raise SystemExit("Contract description version must be X.Y.Z")
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(f"sdk={sdk}\nversion={versions[sdk]}\n")
    print(versions[sdk] if args.version_only else ", ".join(f"{name}: {version}" for name, version in versions.items()))


if __name__ == "__main__":
    main()
