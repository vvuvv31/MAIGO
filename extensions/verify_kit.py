#!/usr/bin/env python3
"""Check package presence and sizes; physics loaders validate schemas and coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-packages", action="store_true",
                        help="Fail instead of reporting missing large binaries")
    parser.add_argument("--package-root", type=Path, default=REPO,
                        help="Repository/data root containing the large packages")
    args = parser.parse_args()
    manifest = json.loads((HERE / "package_manifest.json").read_text())
    failures = []
    missing = []
    checked_packages = 0
    for entry in manifest["packages"]:
        path = args.package_root / entry["path"]
        if not path.is_file():
            missing.append(str(path))
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"package size mismatch: {path}")
        checked_packages += 1
    if args.require_packages:
        failures.extend(f"missing package: {path}" for path in missing)
    print(json.dumps({
        "installed_packages_checked": checked_packages,
        "missing_optional_large_packages": missing,
        "failures": failures,
        "ok": not failures,
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
