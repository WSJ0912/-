from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compact_license(metadata: importlib.metadata.PackageMetadata) -> str:
    expression = str(metadata.get("License-Expression") or "").strip()
    if expression:
        return expression
    license_value = str(metadata.get("License") or "").strip()
    if license_value and len(license_value) <= 120 and "\n" not in license_value:
        return license_value
    classifiers = metadata.get_all("Classifier") or []
    approved = [
        value.rsplit(" :: ", 1)[-1]
        for value in classifiers
        if value.startswith("License :: OSI Approved :: ")
    ]
    return ", ".join(approved) or "UNKNOWN - review package metadata"


def python_dependencies() -> list[tuple[str, str, str]]:
    dependencies = []
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        if not name or name == "medical-imaging-platform":
            continue
        dependencies.append((name, distribution.version, compact_license(distribution.metadata)))
    return sorted(set(dependencies), key=lambda item: item[0].lower())


def npm_name(lock_path: str) -> str | None:
    match = re.search(r"(?:^|/)node_modules/(.+)$", lock_path.replace("\\", "/"))
    if not match:
        return None
    parts = match.group(1).split("/node_modules/")[-1].split("/")
    return "/".join(parts[:2]) if parts[0].startswith("@") else parts[0]


def node_dependencies() -> list[tuple[str, str, str]]:
    lock = json.loads((ROOT / "apps" / "desktop" / "package-lock.json").read_text(encoding="utf-8"))
    dependencies = []
    for lock_path, value in lock.get("packages", {}).items():
        name = npm_name(lock_path)
        if not name or not value.get("version"):
            continue
        license_value = value.get("license")
        if not license_value:
            package_file = ROOT / "apps" / "desktop" / lock_path / "package.json"
            if package_file.is_file():
                package_metadata = json.loads(package_file.read_text(encoding="utf-8"))
                license_value = package_metadata.get("license") or package_metadata.get("licenses")
        license_value = license_value or "UNKNOWN - review package metadata"
        if isinstance(license_value, list):
            license_value = ", ".join(
                str(item.get("type", "UNKNOWN")) if isinstance(item, dict) else str(item)
                for item in license_value
            )
        dependencies.append((name, str(value["version"]), str(license_value)))
    return sorted(set(dependencies), key=lambda item: (item[0].lower(), item[1]))


def table(items: list[tuple[str, str, str]]) -> str:
    rows = ["| Package | Version | Declared license |", "| --- | --- | --- |"]
    for name, version, license_value in items:
        rows.append(f"| `{name}` | `{version}` | {license_value.replace('|', '/')} |")
    return "\n".join(rows)


def main() -> None:
    output = ROOT / "third_party" / "DEPENDENCY_LICENSES.md"
    content = f"""# Dependency license inventory

Generated from `apps/desktop/package-lock.json` and Python distribution metadata.
This is an inventory for release review, not legal advice. A release owner must
resolve every `UNKNOWN` entry and retain license/notice files required by each
dependency before public binary distribution.

## Node/Electron dependency tree

{table(node_dependencies())}

## Python build environment

{table(python_dependencies())}
"""
    output.write_text(content, encoding="utf-8", newline="\n")
    print(output)


if __name__ == "__main__":
    main()
