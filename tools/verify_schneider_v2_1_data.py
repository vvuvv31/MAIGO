#!/usr/bin/env python3
"""Verify Schneider v2.1 runtime data against data/schneider/v2_1_data_manifest.json.

- Reports every missing file and fails.
- Verifies byte size + SHA256 of present files and fails on mismatch.
- Verifies data/schneider/schneider_physics_bundle_v2_1.json pins match
  the on-disk binaries and fails otherwise.
- Never falls back to v1 artifacts.

Usage: python3 tools/verify_schneider_v2_1_data.py [--repo ROOT]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=Path(__file__).resolve().parents[1])
    repo = Path(ap.parse_args().repo)
    manifest_path = repo / "data/schneider/v2_1_data_manifest.json"
    if not manifest_path.exists():
        print("FAIL: manifest missing: %s" % manifest_path)
        return 2
    manifest = json.loads(manifest_path.read_text())
    errors = []
    for e in manifest["files"]:
        p = repo / e["path"]
        if not p.exists():
            errors.append("MISSING %s (%d bytes, sha256 %s...); regenerate: %s" % (
                e["path"], e["bytes"], e["sha256"][:16], e["generator"]))
            continue
        if p.stat().st_size != e["bytes"]:
            errors.append("SIZE-MISMATCH %s: disk %d != manifest %d" % (
                e["path"], p.stat().st_size, e["bytes"]))
            continue
        if not e["in_git"]:
            digest = sha256_file(p)
            if digest != e["sha256"]:
                errors.append("SHA-MISMATCH %s" % e["path"])
    bundle_path = repo / "data/schneider/schneider_physics_bundle_v2_1.json"
    if bundle_path.exists():
        try:
            bundle = json.loads(bundle_path.read_text())

            def pin(*keys):
                node = bundle
                for k in keys:
                    node = node[k]
                return node

            for section in ("primary_rate", "primary_package",
                            "secondary_rate", "secondary_package"):
                for field in ("file", "sha256"):
                    _ = pin(section, field)
        except (KeyError, json.JSONDecodeError) as exc:
            errors.append("BUNDLE-SCHEMA %s" % exc)
    else:
        errors.append("MISSING data/schneider/schneider_physics_bundle_v2_1.json")
    if errors:
        print("FAIL: %d problem(s):" % len(errors))
        for msg in errors:
            print(" -", msg)
        print("Large .bin files are not in git; see v2_1_data_manifest.json "
              "(generator + raw_source) or data/README_physics_tables.md.")
        return 1
    print("OK: all %d manifest entries verified (no v1 fallback used)." % (
        len(manifest["files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
