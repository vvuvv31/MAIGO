#!/usr/bin/env python3
"""Check Schneider package layout, file links, projectile registry and target order.

Payload loaders additionally enforce physical domains and sampling support.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

BINARY_SECTIONS = ("primary_rate", "primary_package",
                   "secondary_rate", "secondary_package")


def read_magic_version(p: Path):
    with open(p, "rb") as f:
        header = f.read(12)
    if len(header) < 12:
        return None, None
    try:
        magic = header[:8].decode("ascii")
    except UnicodeDecodeError:
        return None, None
    (version,) = struct.unpack("<I", header[8:12])
    return magic, version


def channel_target_sequence(channels_path: Path):
    doc = json.loads(channels_path.read_text())
    seq, seen = [], set()
    for ch in doc["channels"]:
        key = (ch.get("projectile_z"), ch.get("projectile_a"),
               ch.get("target_element_z"))
        if key not in seen:
            seen.add(key)
            seq.append(ch.get("target_element_z"))
    return seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=Path(__file__).resolve().parents[3])
    repo = Path(ap.parse_args().repo)
    errors = []

    manifest_path = repo / "data/schneider/v2_1_data_manifest.json"
    if not manifest_path.exists():
        print("FAIL: manifest missing: %s" % manifest_path)
        return 2
    manifest = json.loads(manifest_path.read_text())
    by_path = {e["path"]: e for e in manifest["files"]}

    for e in manifest["files"]:
        p = repo / e["path"]
        if not p.exists():
            errors.append("MISSING %s (%d bytes); regenerate: %s" % (
                e["path"], e["bytes"], e["generator"]))
            continue
        if p.stat().st_size != e["bytes"]:
            errors.append("SIZE-MISMATCH %s: disk %d != manifest %d" % (
                e["path"], p.stat().st_size, e["bytes"]))
            continue
        if "magic" in e:
            magic, version = read_magic_version(p)
            if (magic, version) != (e["magic"], e["version"]):
                errors.append(
                    "MAGIC-MISMATCH %s: disk %r v%s != manifest %r v%s "
                    "(wrong file, e.g. v1 artifact at a v2 path?)" % (
                        e["path"], magic, version, e["magic"], e["version"]))

    bundle_path = repo / "data/schneider/schneider_physics_bundle_v2_1.json"
    if not bundle_path.exists():
        errors.append("MISSING data/schneider/schneider_physics_bundle_v2_1.json")
        print("FAIL: %d problem(s):" % len(errors))
        for msg in errors:
            print(" -", msg)
        return 1
    bundle = json.loads(bundle_path.read_text())
    for section in BINARY_SECTIONS + ("stopping_table",):
        try:
            node = bundle[section]
            bfile = node["file"]
        except KeyError as exc:
            errors.append("BUNDLE-SCHEMA %s.%s" % (section, exc))
            continue
        ment = by_path.get(bfile)
        if ment is None:
            errors.append("BUNDLE-PIN %s -> %s not in manifest" % (section, bfile))
            continue
        if "channels_file" in node:
            cpath = repo / node["channels_file"]
            if not cpath.exists():
                errors.append("BUNDLE-CHANNELS missing %s" % node["channels_file"])
    for section in ("primary_package", "secondary_package"):
        node = bundle.get(section, {})
        if "channels_file" not in node:
            continue
        cpath = repo / node["channels_file"]
        if not cpath.exists():
            continue
        # Channels files store targets in sorted order per projectile group
        # while the bundle lists rate-table target-axis order: compare sets
        # for cross-file consistency, and the exact file order against the
        # manifest-recorded ordered signature (catches swaps/reorders).
        seq = channel_target_sequence(cpath)
        if sorted(set(seq)) != sorted(bundle.get("target_order", [])):
            errors.append("TARGET-ORDER %s channel targets != bundle target set" % section)
        ment = by_path.get(str(cpath.relative_to(repo)))
        if ment is not None and "channels_target_order" in ment:
            if seq != list(ment["channels_target_order"]):
                errors.append("TARGET-ORDER %s file order != manifest signature" % section)
    if manifest.get("projectile_registry") != bundle.get("projectile_registry"):
        errors.append("REGISTRY bundle != manifest")
    if manifest.get("target_order") != bundle.get("target_order"):
        errors.append("TARGET-ORDER bundle != manifest")

    for e in manifest["files"]:
        if not e["path"].endswith(".metadata.json"):
            continue
        p = repo / e["path"]
        if not p.exists():
            continue
        try:
            meta = json.loads(p.read_text())
        except json.JSONDecodeError:
            errors.append("METADATA-JSON %s unparseable" % e["path"])
            continue
        for field in ("data_filename",):
            if field not in meta:
                errors.append("METADATA-SCHEMA %s missing %s" % (e["path"], field))
        if "data_filename" in meta and "file_size_bytes" not in meta:
            errors.append("METADATA-SCHEMA %s missing file_size_bytes" % e["path"])
        if "data_filename" in meta:
            data_path = Path(repo) / "data" / "schneider" / meta["data_filename"]
            if not data_path.exists():
                errors.append("METADATA-LINK %s -> missing %s" % (e["path"], meta["data_filename"]))
            if "file_size_bytes" in meta and data_path.exists() and \
                    data_path.stat().st_size != meta["file_size_bytes"]:
                errors.append("METADATA-LINK %s size != actual %s" % (e["path"], meta["data_filename"]))

    if errors:
        print("FAIL: %d problem(s):" % len(errors))
        for msg in errors:
            print(" -", msg)
        print("Large .bin files are not in git; see v2_1_data_manifest.json "
              "(generator + raw_source) or data/README_physics_tables.md.")
        return 1
    print("OK: all %d manifest entries + bundle links/registry/order checked." % (
        len(manifest["files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
