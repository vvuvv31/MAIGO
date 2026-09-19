#!/usr/bin/env python3
"""Verify bundled source snapshots and locally installed accepted packages."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-packages", action="store_true",
                        help="Fail instead of reporting missing large binaries")
    parser.add_argument("--package-root", type=Path, default=REPO,
                        help="Repository/data root containing the large packages")
    args = parser.parse_args()
    manifest = json.loads((HERE / "package_manifest.json").read_text())
    source_manifest = HERE / "source_manifest.sha256"
    failures = []
    checked_sources = 0
    for line in source_manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = HERE / relative.strip()
        if not path.is_file() or digest(path) != expected:
            failures.append(f"source mismatch: {path}")
        checked_sources += 1
    missing = []
    checked_packages = 0
    for entry in manifest["packages"]:
        path = args.package_root / entry["path"]
        if not path.is_file():
            missing.append(str(path))
            continue
        if digest(path) != entry["sha256"]:
            failures.append(f"package mismatch: {path}")
        checked_packages += 1
    if args.require_packages:
        failures.extend(f"missing package: {path}" for path in missing)
    print(json.dumps({
        "source_files_checked": checked_sources,
        "installed_packages_checked": checked_packages,
        "missing_optional_large_packages": missing,
        "failures": failures,
        "ok": not failures,
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
