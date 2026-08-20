"""Synthetic ELPKG v1 contract tests; no TOPAS or GPU runtime is required."""

from __future__ import annotations

import csv
import gzip
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "startup/package_tools/compile_elastic_package.py"


INTERACTION_FIELDS = [
    "interaction_id", "projectile_Z", "projectile_A", "incident_energy_MeV_per_u",
    "outgoing_projectile_energy_MeV_per_u", "incident_direction_x", "incident_direction_y",
    "incident_direction_z", "outgoing_direction_x", "outgoing_direction_y",
    "outgoing_direction_z", "local_deposit_MeV", "product_count",
    "product_offset_zero_based", "continuation_disposition",
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


def run_compile(tmp_path: Path, interactions: list[dict[str, object]], products: list[dict[str, object]], *, z: int = 1, a: int = 1, bins: int = 1, width: float = 10.0, fill: str = "none") -> subprocess.CompletedProcess[str]:
    interaction_path = tmp_path / "interactions.csv.gz"
    product_path = tmp_path / "products.csv.gz"
    metadata_path = tmp_path / "source.json"
    output_path = tmp_path / "elastic.bin"
    sidecar_path = tmp_path / "elastic.compiled.json"
    write_gzip_csv(interaction_path, INTERACTION_FIELDS, interactions)
    write_gzip_csv(product_path, PRODUCT_FIELDS, products)
    metadata_path.write_text(json.dumps({
        "projectile": {"Z": z, "A": a}, "material": "G4_WATER",
        "physics_model": "G4HadronElasticPhysicsHP",
        "provenance": {"process": "hadElastic"},
    }) + "\n", encoding="utf-8")
    return subprocess.run([
        sys.executable, str(COMPILER), "--metadata", str(metadata_path),
        "--interactions", str(interaction_path), "--products", str(product_path),
        "--output", str(output_path), "--output-metadata", str(sidecar_path),
        "--energy-bin-count", str(bins), "--energy-bin-width-mevu", str(width),
        "--fill-empty", fill, "--energy-tolerance-mev", "0.01",
    ], text=True, capture_output=True, check=False)


def proton_event(outgoing: float = 3.0, *, z: int = 1, a: int = 1, energy: float = 5.0) -> tuple[dict[str, object], list[dict[str, object]]]:
    interaction = {
        "interaction_id": 1, "projectile_Z": z, "projectile_A": a,
        "incident_energy_MeV_per_u": energy,
        "outgoing_projectile_energy_MeV_per_u": outgoing,
        "incident_direction_x": 0.0, "incident_direction_y": 0.0, "incident_direction_z": 1.0,
        "outgoing_direction_x": 0.0, "outgoing_direction_y": 0.0, "outgoing_direction_z": 1.0,
        "local_deposit_MeV": 0.0, "product_count": 2,
        "product_offset_zero_based": 0, "continuation_disposition": "continue",
    }
    products = [
        {"interaction_id": 1, "product_index": 1, "pdg_id": 2212,
         "atomic_number_Z": 1, "mass_number_A": 1, "charge_e": 1.0,
         "kinetic_energy_MeV": 1.5, "direction_x": 1.0, "direction_y": 0.0,
         "direction_z": 0.0, "generation": 1, "transport_disposition": "queue"},
        {"interaction_id": 1, "product_index": 2, "pdg_id": 22,
         "atomic_number_Z": 0, "mass_number_A": 0, "charge_e": 0.0,
         "kinetic_energy_MeV": 0.5, "direction_x": 0.0, "direction_y": 1.0,
         "direction_z": 0.0, "generation": 1, "transport_disposition": "local_deposit"},
    ]
    return interaction, products


def test_valid_proton_and_neutral_recoil(tmp_path: Path) -> None:
    interaction, products = proton_event()
    result = run_compile(tmp_path, [interaction], products)
    assert result.returncode == 0, result.stderr
    sidecar = json.loads((tmp_path / "elastic.compiled.json").read_text())
    assert sidecar["kind"] == "elastic"
    assert sidecar["projectile"] == {"Z": 1, "A": 1}
    assert sidecar["records"] == {"bins": 1, "events": 1, "products": 2}
    assert sidecar["record_sizes_bytes"]["event"] == 44
    assert sidecar["record_sizes_bytes"]["product"] == 36
    assert struct.unpack("<8s", (tmp_path / "elastic.bin").read_bytes()[:8])[0] == b"ELPKG01\0"


def test_valid_generic_ion(tmp_path: Path) -> None:
    interaction, products = proton_event(outgoing=4.0, z=6, a=12, energy=5.0)
    products[0]["kinetic_energy_MeV"] = 10.0
    products[1]["kinetic_energy_MeV"] = 2.0
    result = run_compile(tmp_path, [interaction], products, z=6, a=12)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "elastic.compiled.json").read_text())["projectile"] == {"Z": 6, "A": 12}


def test_nearest_fill_is_explicitly_recorded(tmp_path: Path) -> None:
    interaction, products = proton_event()
    result = run_compile(tmp_path, [interaction], products, bins=3, fill="nearest")
    assert result.returncode == 0, result.stderr
    sidecar = json.loads((tmp_path / "elastic.compiled.json").read_text())
    assert sidecar["bin_semantics"]["alias_is_explicit"] is True
    assert len(sidecar["bin_semantics"]["nearest_fill_aliases"]) == 2


@pytest.mark.parametrize("mutation", ["energy", "direction", "projectile", "misbin", "neutral_charge", "kill"])
def test_invalid_elastic_data_is_rejected(tmp_path: Path, mutation: str) -> None:
    interaction, products = proton_event()
    if mutation == "energy":
        interaction["outgoing_projectile_energy_MeV_per_u"] = 7.0
    elif mutation == "direction":
        products[0]["direction_x"] = 2.0
    elif mutation == "projectile":
        interaction["projectile_Z"] = 2
    elif mutation == "misbin":
        interaction["incident_energy_MeV_per_u"] = 30.0
    elif mutation == "neutral_charge":
        products[1]["charge_e"] = 1.0
    elif mutation == "kill":
        interaction["continuation_disposition"] = "kill"
    result = run_compile(tmp_path, [interaction], products, bins=2)
    assert result.returncode != 0
    assert result.stderr or result.stdout
