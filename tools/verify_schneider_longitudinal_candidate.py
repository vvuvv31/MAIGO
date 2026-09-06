#!/usr/bin/env python3
"""Verify the frozen, unvalidated longitudinal candidate; never promote it."""
import argparse
import hashlib
import json
from pathlib import Path

DATA_SHA = "f42140bc6a99ca90ef9a7fc267c22be9a6d8002192644dbe45b7a40c4e10820b"
META_SHA = "32a37a8235d75954ea020685cd659be2ed6c7d766fad5725eaabdae91035339a"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify(path, raw=False):
    meta_path = path.with_suffix(".metadata.json")
    contract_path = path.with_suffix(".candidate.json")
    if path.stat().st_size != 128 or digest(path) != DATA_SHA or digest(meta_path) != META_SHA:
        raise ValueError("Candidate CSV/metadata SHA or size mismatch")
    contract = json.loads(contract_path.read_text())
    expected = dict(schema_version=1, status="unvalidated_diagnostic",
                    kernel="homogeneous_only_exact_voxel_segments_v1",
                    data_sha256=DATA_SHA, metadata_sha256=META_SHA,
                    scale=1, energy_min_MeV_u=150, energy_max_MeV_u=225,
                    reference_density_g_cm3=0.01131606474518776)
    for key, value in expected.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError("Candidate contract mismatch: " + key)
    if raw:
        for item in json.loads(meta_path.read_text())["inputs"]:
            if digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Raw input SHA mismatch: " + item["path"])
    return dict(status="unvalidated_diagnostic", data_sha256=DATA_SHA,
                metadata_sha256=META_SHA, manifest_sha256=digest(contract_path),
                raw_checked=raw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] /
        "data/schneider/schneider_section0_c12_delta_longitudinal_v1.csv")
    parser.add_argument("--verify-raw", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.data, args.verify_raw), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, "FAIL: " + str(exc) + "\n")
