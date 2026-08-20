#!/usr/bin/env python3
"""Exercise elastic YAML validation and the plan-only preload boundary."""

from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "startup/package_tools/compile_elastic_package.py"


def write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(directory: Path) -> tuple[Path, Path]:
    metadata = directory / "elastic_metadata.json"
    metadata.write_text(json.dumps({
        "projectile": {"Z": 1, "A": 1},
        "material": "G4_WATER",
        "physics_model": "G4HadronElasticPhysicsHP",
    }) + "\n", encoding="utf-8")
    interaction_fields = [
        "interaction_id", "projectile_Z", "projectile_A",
        "incident_energy_MeV_per_u", "outgoing_projectile_energy_MeV_per_u",
        "incident_direction_x", "incident_direction_y", "incident_direction_z",
        "outgoing_direction_x", "outgoing_direction_y", "outgoing_direction_z",
        "local_deposit_MeV", "product_count", "product_offset_zero_based",
        "continuation_disposition",
    ]
    interaction = {
        "interaction_id": 1, "projectile_Z": 1, "projectile_A": 1,
        "incident_energy_MeV_per_u": 5.0,
        "outgoing_projectile_energy_MeV_per_u": 3.0,
        "incident_direction_x": 0.0, "incident_direction_y": 0.0,
        "incident_direction_z": 1.0, "outgoing_direction_x": 0.0,
        "outgoing_direction_y": 0.0, "outgoing_direction_z": 1.0,
        "local_deposit_MeV": 0.0, "product_count": 2,
        "product_offset_zero_based": 0, "continuation_disposition": "continue",
    }
    product_fields = [
        "interaction_id", "product_index", "pdg_id", "atomic_number_Z",
        "mass_number_A", "charge_e", "kinetic_energy_MeV", "direction_x",
        "direction_y", "direction_z", "generation", "transport_disposition",
    ]
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
    interactions = directory / "interactions.csv.gz"
    product_table = directory / "products.csv.gz"
    write_gzip_csv(interactions, interaction_fields, [interaction])
    write_gzip_csv(product_table, product_fields, products)
    package = directory / "elastic.bin"
    sidecar = directory / "elastic.compiled.json"
    subprocess.run([
        sys.executable, str(COMPILER), "--metadata", str(metadata),
        "--interactions", str(interactions), "--products", str(product_table),
        "--output", str(package), "--output-metadata", str(sidecar),
        "--energy-bin-count", "1", "--energy-bin-width-mevu", "10",
    ], check=True, cwd=ROOT)

    xs = directory / "elastic.csv"
    xs.write_text(
        "energy_MeV_per_u,water_macroscopic_cross_section_per_mm\n"
        "0.1,0.01\n10.0,0.01\n", encoding="utf-8")
    return package, xs


def config_text(package: Path, xs: Path, *, enabled: bool = True) -> str:
    return f"""number_of_histories: 1
initial_energy_MeVu: 5.0
primary_atomic_number: 1
primary_mass_number: 1
primary_rest_mass_MeV: 938.27208816
phantom_length_mm: 10.0
depth_bin_width_mm: 1.0
maximum_step_mm: 0.5
maximum_relative_energy_loss: 0.01
energy_cutoff_MeV: 0.1
device: serial
primary_stopping_power_file: {ROOT / 'data/stopping_power_water_geant4_11_3_2.csv'}
primary_inelastic_cross_section_file: {ROOT / 'data/c12_inelastic_cross_sections_water_geant4_11_3_2.csv'}
enable_primary_elastic_interactions: {str(enabled).lower()}
primary_elastic_cross_section_file: {xs}
primary_elastic_package_file: {package}
primary_elastic_package_physics_model: G4HadronElasticPhysicsHP
package_identity_validation: strict
output_file:
"""


def run(binary: Path, config: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(binary), "--config", str(config), *extra],
                          cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_elastic_config_preload.py CARBON_MC")
    binary = Path(sys.argv[1])
    with tempfile.TemporaryDirectory(prefix="carbon-elastic-config-") as name:
        directory = Path(name)
        package, xs = make_fixture(directory)
        config = directory / "elastic.yaml"
        config.write_text(config_text(package, xs), encoding="utf-8")

        planned = run(binary, config, "--plan-only")
        assert planned.returncode == 0, planned.stderr
        assert "Elastic packages: bins=1; events=1; products=2" in planned.stdout
        assert "Elastic cross section: 2 bins" in planned.stdout

        normal = run(binary, config)
        assert normal.returncode != 0
        assert "require the legacy SYCL backend" in normal.stderr

        sidecar = package.with_suffix(".compiled.json")
        original = json.loads(sidecar.read_text(encoding="utf-8"))
        for key, value in (("kind", "reaction"),
                           ("physics_model", "wrong-model")):
            mutated = dict(original)
            mutated[key] = value
            sidecar.write_text(json.dumps(mutated), encoding="utf-8")
            rejected = run(binary, config, "--plan-only")
            assert rejected.returncode != 0, key
            sidecar.write_text(json.dumps(original), encoding="utf-8")

        mutated = dict(original)
        mutated["projectile"] = {"Z": 6, "A": 12}
        sidecar.write_text(json.dumps(mutated), encoding="utf-8")
        assert run(binary, config, "--plan-only").returncode != 0
        sidecar.write_text(json.dumps(original), encoding="utf-8")

        mutated = dict(original)
        mutated["energy_range_MeV_per_u"] = {"minimum": 0.0, "maximum": 2.0}
        sidecar.write_text(json.dumps(mutated), encoding="utf-8")
        assert run(binary, config, "--plan-only").returncode != 0
        sidecar.write_text(json.dumps(original), encoding="utf-8")

        mutated = dict(original)
        mutated["output"] = dict(original["output"])
        mutated["output"]["sha256"] = "0" * 64
        sidecar.write_text(json.dumps(mutated), encoding="utf-8")
        assert run(binary, config, "--plan-only").returncode != 0
        sidecar.write_text(json.dumps(original), encoding="utf-8")

        xs.write_text(
            "energy_MeV_per_u,water_macroscopic_cross_section_per_mm\n"
            "0.1,0.01\n4.0,0.01\n", encoding="utf-8")
        assert run(binary, config, "--plan-only").returncode != 0
        xs.write_text(
            "energy_MeV_per_u,water_macroscopic_cross_section_per_mm\n"
            "0.1,0.01\n10.0,0.01\n", encoding="utf-8")

        disabled = directory / "disabled.yaml"
        disabled.write_text(config_text(package, xs, enabled=False), encoding="utf-8")
        assert run(binary, disabled, "--plan-only").returncode != 0
        disabled.write_text(config_text(package, xs, enabled=False).replace(
            f"primary_elastic_cross_section_file: {xs}\n", ""
        ).replace(f"primary_elastic_package_file: {package}\n", "").replace(
            "primary_elastic_package_physics_model: G4HadronElasticPhysicsHP\n", ""
        ), encoding="utf-8")
        baseline = run(binary, disabled, "--plan-only")
        assert baseline.returncode == 0, baseline.stderr
    print("Elastic config/preload tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
