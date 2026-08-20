#!/usr/bin/env python3
"""Focused tests for the deterministic adaptive elastic campaign planner."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "startup/elastic_package/plan_adaptive_elastic_campaign.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_inputs(directory: Path, counts: list[int], *, zero_energy: float | None = None,
                projectile: tuple[int, int] = (1, 1), maximum: float = 4.01) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "direct.csv"
    metadata_path = directory / "direct.metadata.json"
    coverage_path = directory / "coverage.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "energy_MeV_per_u", "water_macroscopic_cross_section_per_mm",
        ], lineterminator="\n")
        writer.writeheader()
        count = int(round(maximum / 0.01)) + 1
        for index in range(count):
            energy = index * 0.01
            value = 0.0 if energy < 0.9 else 0.002
            if zero_energy is not None and abs(energy - zero_energy) < 1.0e-12:
                value = 0.0
            writer.writerow({
                "energy_MeV_per_u": f"{energy:g}",
                "water_macroscopic_cross_section_per_mm": f"{value:g}",
            })
    metadata = {
        "schema_version": 1,
        "kind": "direct_elastic_cross_section_query",
        "source": "direct G4HadronicProcessStore query",
        "projectile": {"Z": projectile[0], "A": projectile[1]},
        "material": "G4_WATER",
        "process": "hadElastic",
        "physics_model": "G4HadronElasticPhysicsHP",
        "versions": {"TOPAS": "4.2.p3", "Geant4": "11.3.2"},
        "grid": {"minimum_MeV_per_u": 0.0, "maximum_MeV_per_u": maximum,
                 "step_MeV_per_u": 0.01, "points": count, "endpoint_inclusive": True},
        "units": {"energy": "MeV/u", "water_macroscopic_cross_section": "1/mm"},
        "output": {"path": csv_path.name, "sha256": digest(csv_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    coverage_path.write_text(json.dumps({
        "bins": [{"index": index, "event_count": value} for index, value in enumerate(counts)],
    }, indent=2) + "\n", encoding="utf-8")
    return coverage_path, csv_path, metadata_path


def run_planner(tmp_path: Path, coverage: Path, direct: Path, metadata: Path,
                *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(PLANNER), "--coverage", str(coverage),
        "--direct-xs-csv", str(direct), "--direct-xs-metadata", str(metadata),
        "--output", str(tmp_path / "manifest.json"), *extra,
    ], text=True, capture_output=True, check=False)


def test_deterministic_plan_and_formula(tmp_path: Path) -> None:
    coverage, direct, metadata = make_inputs(tmp_path / "inputs", [500, 1000, 10])
    first = run_planner(tmp_path / "first", coverage, direct, metadata,
                        "--energy-max-mevu", "2", "--first-batch-size", "1")
    assert first.returncode == 0, first.stderr
    first_bytes = (tmp_path / "first" / "manifest.json").read_bytes()
    second = run_planner(tmp_path / "second", coverage, direct, metadata,
                         "--energy-max-mevu", "2", "--first-batch-size", "1")
    assert second.returncode == 0, second.stderr
    assert first_bytes == (tmp_path / "second" / "manifest.json").read_bytes()
    document = json.loads(first_bytes)
    assert document["summary"]["bins"] == 3
    assert document["summary"]["planned_bins"] == 2
    assert document["summary"]["skipped_bins"] == 1
    assert document["summary"]["planned_histories"] == 2473
    assert document["summary"]["first_batch_bins"] == [2]
    assert document["summary"]["first_batch_entries"][0]["bin"] == 2
    assert document["summary"]["first_batch_entries"][0]["label"] == "proton_2p5mevu_g4_water"
    entry = document["entries"][0]
    assert entry["status"] == "planned"
    assert entry["representative_energy_MeV_per_u"] == 0.9
    assert entry["event_probability"] > 0.0
    assert entry["histories"] == 830
    assert document["provenance"]["continuous_em_loss_enabled"] is False
    assert document["provenance"]["multiple_scattering_enabled"] is False
    assert document["provenance"]["scored_processes"] == ["hadElastic"]


def test_zero_xs_is_rejected_for_unsatisfied_bin(tmp_path: Path) -> None:
    coverage, direct, metadata = make_inputs(
        tmp_path / "inputs", [1000, 10, 10], zero_energy=1.5)
    result = run_planner(tmp_path, coverage, direct, metadata,
                         "--energy-max-mevu", "2")
    assert result.returncode != 0
    assert "zero or invalid direct XS" in result.stderr


def test_history_cap_is_rejected(tmp_path: Path) -> None:
    coverage, direct, metadata = make_inputs(tmp_path / "inputs", [0, 0, 0])
    result = run_planner(tmp_path, coverage, direct, metadata,
                         "--energy-max-mevu", "2", "--max-histories", "100")
    assert result.returncode != 0
    assert "above max-histories" in result.stderr


def test_satisfied_bins_are_skipped(tmp_path: Path) -> None:
    coverage, direct, metadata = make_inputs(tmp_path / "inputs", [1000, 1000, 1000])
    result = run_planner(tmp_path, coverage, direct, metadata,
                         "--energy-max-mevu", "2")
    assert result.returncode == 0, result.stderr
    document = json.loads((tmp_path / "manifest.json").read_text())
    assert all(entry["status"] == "skipped_satisfied" for entry in document["entries"])
    assert all(entry["histories"] == 0 for entry in document["entries"])


def test_generic_projectile_and_nonzero_energy_grid(tmp_path: Path) -> None:
    coverage, direct, metadata = make_inputs(
        tmp_path / "inputs", [0, 0], projectile=(2, 4), maximum=4.01)
    result = run_planner(
        tmp_path, coverage, direct, metadata,
        "--projectile-z", "2", "--projectile-a", "4",
        "--projectile-name", "GenericIon",
        "--energy-min-mevu", "1", "--energy-max-mevu", "2",
    )
    assert result.returncode == 0, result.stderr
    document = json.loads((tmp_path / "manifest.json").read_text())
    assert [entry["representative_energy_MeV_per_u"] for entry in document["entries"]] == [1.5, 2.5]
    assert document["identity"]["projectile"] == {"name": "GenericIon", "Z": 2, "A": 4}
    assert all("proton" not in entry["label"] for entry in document["entries"])
    assert document["entries"][0]["label"] == "ion_z2a4_1p5mevu_g4_water"


def test_real_coverage_summary(tmp_path: Path) -> None:
    coverage = ROOT / "out/proton_water_qgsp_bic_hp/elastic_low_bin_10k/proton_water_hadelastic_10k_x5_plus_low.coverage.json"
    direct = ROOT / "data/proton_water_direct_elastic_xs.csv"
    metadata = ROOT / "data/proton_water_direct_elastic_xs.metadata.json"
    result = run_planner(tmp_path, coverage, direct, metadata, "--first-batch-size", "16")
    assert result.returncode == 0, result.stderr
    document = json.loads((tmp_path / "manifest.json").read_text())
    assert document["summary"]["bins"] == 251
    assert document["summary"]["skipped_bins"] == 1
    assert document["summary"]["planned_bins"] == 250
    assert document["summary"]["first_batch_bins"] == [
        172, 201, 202, 207, 208, 212, 213, 214,
        221, 223, 228, 231, 232, 237, 249, 250,
    ]
    assert [entry["bin"] for entry in document["summary"]["first_batch_entries"]] == document["summary"]["first_batch_bins"]
    assert document["summary"]["first_batch_entries"][-1]["label"] == "proton_250p5mevu_g4_water"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        test_deterministic_plan_and_formula(temporary / "deterministic")
        test_zero_xs_is_rejected_for_unsatisfied_bin(temporary / "zero")
        test_history_cap_is_rejected(temporary / "cap")
        test_satisfied_bins_are_skipped(temporary / "skip")
        test_generic_projectile_and_nonzero_energy_grid(temporary / "generic")
        test_real_coverage_summary(temporary / "real")
    print("adaptive elastic campaign tests passed")
