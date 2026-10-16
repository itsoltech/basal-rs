"""Read independent SDK versions from their manifests using the standard library."""

import json
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET

SDKS = ("python", "typescript", "csharp", "rust")
STABLE_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def read_version(root: Path, sdk: str) -> str:
    if sdk == "python":
        version = tomllib.loads((root / "sdk/python/pyproject.toml").read_text())["project"]["version"]
    elif sdk == "typescript":
        version = json.loads((root / "sdk/typescript/package.json").read_text())["version"]
        lock = json.loads((root / "sdk/typescript/package-lock.json").read_text())
        if lock.get("version") != version or lock["packages"][""]["version"] != version:
            raise SystemExit("TypeScript manifest and npm lock versions differ")
    elif sdk == "csharp":
        version = ET.parse(root / "sdk/csharp/ITSOL.Basal.Client/ITSOL.Basal.Client.csproj").findtext("PropertyGroup/Version")
    elif sdk == "rust":
        version = tomllib.loads((root / "sdk/rust/Cargo.toml").read_text())["package"]["version"]
        lock = tomllib.loads((root / "Cargo.lock").read_text())
        entries = [item for item in lock["package"] if item["name"] == "basal-client" and "source" not in item]
        if len(entries) != 1 or entries[0]["version"] != version:
            raise SystemExit("Rust SDK manifest and workspace lock versions differ")
    else:
        raise SystemExit(f"Unknown SDK: {sdk}")
    if not isinstance(version, str) or not re.fullmatch(STABLE_VERSION, version):
        raise SystemExit(f"{sdk} version must be a stable X.Y.Z version")
    return version


def parse_tag(tag: str) -> tuple[str, str]:
    match = re.fullmatch(rf"sdk-({'|'.join(SDKS)})-v({STABLE_VERSION})", tag)
    if not match:
        raise SystemExit("Expected sdk-<python|typescript|csharp|rust>-vX.Y.Z")
    return match[1], match[2]
