#!/usr/bin/env python3
"""Generate data/schneider/v2_1_data_manifest.json.

Records every v2.1 runtime file: relative path, byte size, SHA256,
generator, raw sources, plus binary magic/version and bundle registry /
target-order signatures for tamper-evident verification by
tools/verify_schneider_v2_1_data.py.
"""
import hashlib
import json
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BINARIES = [
    ("data/schneider/schneider_inelastic_rates_v2_1.bin",
     "python3 tools/compile_primary_rates_v2_1.py --accept-residual-need",
     ["/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2",
      "/mnt/sda/wuwei/secondary-rates-v2-1/raw/c12_dump.json"]),
    ("data/schneider/secondary_inelastic_rates_v2_1.bin",
     "python3 tools/compile_rates_v2_1.py --accept-residual-need",
     ["/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2",
      "/mnt/sda/wuwei/secondary-rates-v2-1/raw"]),
    ("data/schneider/cinel03_c12_targets_v2_1.bin",
     "python3 tools/compile_primary_package_v2_1.py",
     ["/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2",
      "/mnt/sda/wuwei/cinel03-campaigns/production"]),
    ("data/schneider/cinel03_secondary_targets_v2_1_14p.bin",
     "python3 tools/compile_v2_package.py --tag v2_1",
     ["/mnt/sda/wuwei/cinel03-campaigns/schneider-secondary-v2",
      "/mnt/sda/wuwei/step20_secondary_campaigns/raw"]),
]

SMALL_COMMITTED = [
    "data/schneider/schneider_physics_bundle_v2_1.json",
    "data/schneider/schneider_stopping_v1.bin",
    "data/schneider/c12_schneider_inelastic_mass_xs_v2_1.csv",
    "data/schneider/cinel03_c12_targets_v2_1.channels.json",
    "data/schneider/cinel03_c12_targets_v2_1.metadata.json",
    "data/schneider/cinel03_secondary_targets_v2_1_14p.channels.json",
    "data/schneider/cinel03_secondary_targets_v2_1_14p.metadata.json",
    "data/schneider/schneider_inelastic_rates_v2_1.metadata.json",
    "data/schneider/secondary_inelastic_rates_v2_1.metadata.json",
    "data/HUtoMaterialSchneider.txt",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def magic_version(p: Path):
    with open(p, "rb") as f:
        header = f.read(12)
    magic = header[:8].decode("ascii")
    (version,) = struct.unpack("<I", header[8:12])
    return magic, version


def main() -> None:
    entries = []
    for rel, generator, raw in BINARIES:
        p = REPO / rel
        magic, version = magic_version(p)
        entries.append({"path": rel, "bytes": p.stat().st_size,
                        "sha256": sha256_file(p), "magic": magic,
                        "version": version, "generator": generator,
                        "raw_source": raw, "in_git": False})
    for rel in SMALL_COMMITTED:
        p = REPO / rel
        entry = {"path": rel, "bytes": p.stat().st_size,
                 "sha256": sha256_file(p), "generator": "committed",
                 "raw_source": [], "in_git": True}
        if rel.endswith(".bin"):
            magic, version = magic_version(p)
            entry["magic"] = magic
            entry["version"] = version
        if rel.endswith(".channels.json"):
            doc = json.loads(p.read_text())
            entry["channels_target_order"] = [
                ch.get("target_element_z") for ch in doc["channels"]]
        entries.append(entry)
    bundle = json.loads(
        (REPO / "data/schneider/schneider_physics_bundle_v2_1.json").read_text())
    registry_sig = hashlib.sha256(
        json.dumps(bundle["projectile_registry"], sort_keys=True).encode()).hexdigest()
    targets_sig = hashlib.sha256(
        json.dumps(bundle["target_order"]).encode()).hexdigest()
    out = REPO / "data/schneider/v2_1_data_manifest.json"
    out.write_text(json.dumps(
        {"schema_version": 2,
         "note": "Large .bin files are NOT in git. Regenerate with the listed "
                 "generator or copy from a pinned artifact copy. "
                 "tools/verify_schneider_v2_1_data.py enforces this manifest, "
                 "including bundle pins, magic/version, and registry/target signatures.",
         "bundle_registry_sha256": registry_sig,
         "bundle_target_order_sha256": targets_sig,
         "files": entries}, indent=1) + "\n")
    print("wrote", out, len(entries), "entries")


if __name__ == "__main__":
    main()
