#!/usr/bin/env python3
"""Audit the locked Python consumer/builder and npm builder graphs through OSV, without audit dependencies."""

import argparse
import json
from pathlib import Path
import re
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", choices=["python", "typescript"], help="Audit only the selected SDK; default: both")
    sdk = parser.parse_args().sdk
    root = Path(__file__).resolve().parents[2]
    packages: set[tuple[str, str, str]] = set()
    for filename in (("requirements-build.txt", "requirements-runtime.txt") if sdk != "typescript" else ()):
        for line in (root / "tools/sdk" / filename).read_text().splitlines():
            match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if match:
                packages.add(("PyPI", re.sub(r"[-_.]+", "-", match[1]).lower(), match[2]))
    lock = json.loads((root / "sdk/typescript/package-lock.json").read_text()) if sdk != "python" else {"packages": {}}
    for path, info in lock["packages"].items():
        if path.startswith("node_modules/"):
            packages.add(("npm", info.get("name", path.rsplit("node_modules/", 1)[1]), info["version"]))
    ordered = sorted(packages)
    body = {"queries": [{"package": {"ecosystem": ecosystem, "name": name}, "version": version}
                        for ecosystem, name, version in ordered]}
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        results = json.load(response)["results"]
    if len(results) != len(ordered) or any(result.get("next_page_token") for result in results):
        raise SystemExit("OSV returned an incomplete audit")
    findings = []
    for package, result in zip(ordered, results):
        findings.extend(f"{package[0]} {package[1]} {package[2]}: {item['id']}"
                        for item in result.get("vulns", []))
    if findings:
        raise SystemExit("Known dependency advisories:\n" + "\n".join(findings))
    print(f"OSV: no known advisories for {len(ordered)} locked {sdk or 'Python/npm'} packages")


if __name__ == "__main__":
    main()
