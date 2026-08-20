#!/usr/bin/env python3
"""Cross-language ELPKG v1 test runner used by CTest (no pytest dependency)."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "startup/package_tools/compile_elastic_package.py"
INTERACTION_FIELDS = [
    "interaction_id", "projectile_Z", "projectile_A", "incident_energy_MeV_per_u",
    "outgoing_projectile_energy_MeV_per_u", "incident_direction_x",
    "incident_direction_y", "incident_direction_z", "outgoing_direction_x",
    "outgoing_direction_y", "outgoing_direction_z", "local_deposit_MeV",
    "product_count", "product_offset_zero_based", "continuation_disposition",
]
PRODUCT_FIELDS = [
    "interaction_id", "product_index", "pdg_id", "atomic_number_Z", "mass_number_A",
    "charge_e", "kinetic_energy_MeV", "direction_x", "direction_y", "direction_z",
    "generation", "transport_disposition",
]


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compile_fixture(directory: Path, *, z: int, a: int, bins: int = 1,
                    fill: str = "none") -> Path:
    interaction = {
        "interaction_id": 1, "projectile_Z": z, "projectile_A": a,
        "incident_energy_MeV_per_u": 5.0,
        "outgoing_projectile_energy_MeV_per_u": 3.0,
        "incident_direction_x": 0.0, "incident_direction_y": 0.0,
        "incident_direction_z": 1.0, "outgoing_direction_x": 0.0,
        "outgoing_direction_y": 0.0, "outgoing_direction_z": 1.0,
        "local_deposit_MeV": 0.0, "product_count": 2,
        "product_offset_zero_based": 0, "continuation_disposition": "continue",
    }
    # For an ion, event energies are per nucleon while products are MeV.
    products = [
        {"interaction_id": 1, "product_index": 1, "pdg_id": 2212,
         "atomic_number_Z": 1, "mass_number_A": 1, "charge_e": 1.0,
         "kinetic_energy_MeV": 1.5 * a, "direction_x": 1.0,
         "direction_y": 0.0, "direction_z": 0.0, "generation": 1,
         "transport_disposition": "queue"},
        {"interaction_id": 1, "product_index": 2, "pdg_id": 22,
         "atomic_number_Z": 0, "mass_number_A": 0, "charge_e": 0.0,
         "kinetic_energy_MeV": 0.5 * a, "direction_x": 0.0,
         "direction_y": 1.0, "direction_z": 0.0, "generation": 1,
         "transport_disposition": "local_deposit"},
    ]
    source = directory / f"source_{z}_{a}.json"
    source.write_text(json.dumps({
        "projectile": {"Z": z, "A": a}, "material": "G4_WATER",
        "physics_model": "G4HadronElasticPhysicsHP",
    }) + "\n", encoding="utf-8")
    interactions = directory / f"interactions_{z}_{a}.csv.gz"
    product_table = directory / f"products_{z}_{a}.csv.gz"
    write_gzip_csv(interactions, INTERACTION_FIELDS, [interaction])
    write_gzip_csv(product_table, PRODUCT_FIELDS, products)
    output = directory / f"elastic_{z}_{a}.bin"
    sidecar = directory / f"elastic_{z}_{a}.compiled.json"
    subprocess.run([
        sys.executable, str(COMPILER), "--metadata", str(source),
        "--interactions", str(interactions), "--products", str(product_table),
        "--output", str(output), "--output-metadata", str(sidecar),
        "--energy-bin-count", str(bins), "--energy-bin-width-mevu", "10",
        "--fill-empty", fill,
    ], check=True, text=True)
    return output


def expect_loader(loader: Path, package: Path, success: bool, *identity: int) -> None:
    result = subprocess.run([str(loader), str(package), *map(str, identity)],
                            text=True, capture_output=True)
    if (result.returncode == 0) != success:
        raise RuntimeError(f"Unexpected loader result for {package.name}: {result.stderr}")


def mutate(source: Path, destination: Path, offset: int, payload: bytes) -> None:
    data = bytearray(source.read_bytes())
    data[offset:offset + len(payload)] = payload
    destination.write_bytes(data)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_elastic_package_tests.py LOADER")
    loader = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="carbon-elastic-test-") as name:
        directory = Path(name)
        proton = compile_fixture(directory, z=1, a=1)
        ion = compile_fixture(directory, z=6, a=12)
        alias = compile_fixture(directory, z=1, a=1, bins=3, fill="nearest")
        expect_loader(loader, proton, True, 1, 1)
        expect_loader(loader, ion, True, 6, 12)
        expect_loader(loader, alias, True, 1, 1)

        # Header magic, first-bin offset, event direction, event energy, and
        # first-event product range are all independently rejected.
        mutations = {
            "bad_magic": (0, b"BADMAGIC"),
            "bad_offset": (60 + 8, struct.pack("<I", 99)),
            "bad_direction": (60 + 16 + 16, struct.pack("<f", 2.0)),
            "bad_energy": (60 + 16 + 4, struct.pack("<f", 99.0)),
            "bad_product_range": (60 + 16 + 28, struct.pack("<I", 99)),
        }
        for label, (offset, payload) in mutations.items():
            mutated = directory / f"{label}.bin"
            mutate(proton, mutated, offset, payload)
            expect_loader(loader, mutated, False)

        # Change an aliased event payload without changing its natural source.
        data = bytearray(alias.read_bytes())
        first_event = 60 + 3 * 16
        second_event = first_event + 44
        data[second_event + 8:second_event + 12] = struct.pack("<f", 4.0)
        mutated_alias = directory / "bad_alias_payload.bin"
        mutated_alias.write_bytes(data)
        expect_loader(loader, mutated_alias, False)
    print("ELPKG cross-language and mutation tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
