"""Focused tests for the generic hadronic-elastic extraction campaign."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "startup/elastic_package/run_elastic_campaign.py"
AGGREGATOR = ROOT / "startup/elastic_package/aggregate_elastic_campaign.py"
TEMPLATE = ROOT / "startup/elastic_package/elastic_only.txt.in"

INTERACTION_FIELDS = [
    "interaction_id", "run_id", "thread_id", "event_id", "projectile_Z", "projectile_A",
    "incident_energy_MeV_per_u", "outgoing_projectile_energy_MeV_per_u",
    "macroscopic_elastic_per_mm",
    "incident_direction_x", "incident_direction_y", "incident_direction_z",
    "outgoing_direction_x", "outgoing_direction_y", "outgoing_direction_z",
    "local_deposit_MeV", "product_count", "product_offset_zero_based",
    "continuation_disposition", "process_name", "process_type", "process_subtype",
]
PRODUCT_FIELDS = [
    "interaction_id", "product_index", "pdg_id", "atomic_number_Z", "mass_number_A",
    "charge_e", "kinetic_energy_MeV", "direction_x", "direction_y", "direction_z",
    "generation", "transport_disposition",
]
DIRECT_QUERY_FIELDS = [
    "energy_MeV_per_u", "water_macroscopic_cross_section_per_mm",
]
DIRECT_QUERY_POINTS = 25101


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_direct_query(directory: Path, *, value: float = 0.0042,
                      projectile: tuple[int, int] = (1, 1),
                      material: str = "G4_WATER",
                      physics_model: str = "G4HadronElasticPhysicsHP",
                      maximum: float = 251.0,
                      step: float = 0.01) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    points = int(round(maximum / step)) + 1
    csv_path = directory / "direct_elastic_xs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=DIRECT_QUERY_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        for index in range(points):
            writer.writerow({
                "energy_MeV_per_u": f"{index * step:.12g}",
                "water_macroscopic_cross_section_per_mm": f"{value + index * 1.0e-8:.12g}",
            })
    metadata_path = directory / "direct_elastic_xs.metadata.json"
    metadata_path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "direct_elastic_cross_section_query",
        "source": "direct G4HadronicProcessStore query",
        "projectile": {"Z": projectile[0], "A": projectile[1]},
        "material": material,
        "process": "hadElastic",
        "physics_model": physics_model,
        "grid": {"minimum_MeV_per_u": 0.0, "maximum_MeV_per_u": maximum,
                 "step_MeV_per_u": step, "points": points,
                 "endpoint_inclusive": True},
        "units": {"energy": "MeV/u",
                  "water_macroscopic_cross_section": "1/mm"},
        "output": {"path": str(csv_path), "sha256": digest(csv_path)},
    }, indent=2) + "\n", encoding="utf-8")
    return csv_path, metadata_path


def refresh_direct_metadata(csv_path: Path, metadata_path: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output"]["sha256"] = digest(csv_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_aggregate(tmp_path: Path, arguments: list[str],
                  direct: tuple[Path, Path] | None = None) -> subprocess.CompletedProcess[str]:
    if direct is None:
        direct = make_direct_query(tmp_path / "direct_query")
    return subprocess.run([
        sys.executable, str(AGGREGATOR), *arguments,
        "--direct-xs-csv", str(direct[0]),
        "--direct-xs-metadata", str(direct[1]),
    ], text=True, capture_output=True, check=False)


def make_source(directory: Path, energy: float, *, old_id: int = 1,
                process_name: str = "hadElastic", elastic_xs: object = 0.0125,
                extraction_mode: str = "transport-degrading",
                continuous_em_loss_enabled: bool = True) -> Path:
    directory.mkdir(parents=True)
    interactions = directory / "prepared_interactions.csv.gz"
    products = directory / "prepared_products.csv.gz"
    write_csv(interactions, INTERACTION_FIELDS, [{
        "interaction_id": old_id, "run_id": 0, "thread_id": 0, "event_id": 1,
        "projectile_Z": 1, "projectile_A": 1,
        "incident_energy_MeV_per_u": energy,
        "outgoing_projectile_energy_MeV_per_u": energy,
        "macroscopic_elastic_per_mm": elastic_xs,
        "incident_direction_x": 0, "incident_direction_y": 0, "incident_direction_z": 1,
        "outgoing_direction_x": 0, "outgoing_direction_y": 0, "outgoing_direction_z": 1,
        "local_deposit_MeV": 0, "product_count": 0, "product_offset_zero_based": 0,
        "continuation_disposition": "continue", "process_name": process_name,
        "process_type": 4, "process_subtype": 111,
    }])
    write_csv(products, PRODUCT_FIELDS, [])
    metadata = directory / "prepared.metadata.json"
    metadata.write_text(json.dumps({
        "projectile": {"Z": 1, "A": 1}, "material": "G4_WATER",
        "physics_model": "G4HadronElasticPhysicsHP",
        "provenance": {"scorer": "CarbonElasticNtuple", "processes": ["hadElastic"],
                       "extraction_mode": extraction_mode,
                       "continuous_em_loss_enabled": continuous_em_loss_enabled,
                       "sampling_purpose": "test-package-sampling"},
        "files": {
            "interactions": {"path": str(interactions), "sha256": digest(interactions)},
            "products": {"path": str(products), "sha256": digest(products)},
        },
    }) + "\n", encoding="utf-8")
    return metadata


def make_default_grid_source(directory: Path) -> Path:
    directory.mkdir(parents=True)
    interactions = directory / "prepared_interactions.csv.gz"
    products = directory / "prepared_products.csv.gz"
    rows = []
    for interaction_id in range(1, 252):
        energy = interaction_id - 1 + 0.25
        rows.append({
            "interaction_id": interaction_id, "run_id": 0, "thread_id": 0,
            "event_id": interaction_id, "projectile_Z": 1, "projectile_A": 1,
            "incident_energy_MeV_per_u": energy,
            "outgoing_projectile_energy_MeV_per_u": energy,
            "macroscopic_elastic_per_mm": 0.01,
            "incident_direction_x": 0, "incident_direction_y": 0, "incident_direction_z": 1,
            "outgoing_direction_x": 0, "outgoing_direction_y": 0, "outgoing_direction_z": 1,
            "local_deposit_MeV": 0, "product_count": 0,
            "product_offset_zero_based": 0, "continuation_disposition": "continue",
            "process_name": "hadElastic", "process_type": 4, "process_subtype": 111,
        })
    write_csv(interactions, INTERACTION_FIELDS, rows)
    write_csv(products, PRODUCT_FIELDS, [])
    metadata = directory / "prepared.metadata.json"
    metadata.write_text(json.dumps({
        "projectile": {"Z": 1, "A": 1}, "material": "G4_WATER",
        "physics_model": "G4HadronElasticPhysicsHP",
        "provenance": {"scorer": "CarbonElasticNtuple", "processes": ["hadElastic"],
                       "extraction_mode": "transport-degrading",
                       "continuous_em_loss_enabled": True,
                       "sampling_purpose": "test-package-sampling"},
        "files": {
            "interactions": {"path": str(interactions), "sha256": digest(interactions)},
            "products": {"path": str(products), "sha256": digest(products)},
        },
    }) + "\n", encoding="utf-8")
    return metadata


def test_template_has_only_strong_hadronic_elastic_scorer() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    quantity_lines = [line for line in text.splitlines() if "/Quantity" in line]
    assert quantity_lines == ['s:Sc/ElasticEvents/Quantity = "CarbonElasticNtuple"']
    assert "g4em-standard" not in text  # supplied explicitly by the runner
    assert "@PHYSICS_MODULES@" in text
    assert "Coulomb" in text and "multiple scattering" in text


def test_runner_defaults_to_side_effect_free_proton_dry_run(tmp_path: Path) -> None:
    root = tmp_path / "remote"
    result = subprocess.run([
        sys.executable, str(RUNNER), "--remote-root", str(root),
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[dry-run] energy=") == 5
    assert "energy=70 MeV/u total=70 MeV" in result.stdout
    assert "energy=250 MeV/u total=250 MeV" in result.stdout
    assert not root.exists()


def test_runner_converts_mevu_and_prints_generic_ion_expression(tmp_path: Path) -> None:
    result = subprocess.run([
        sys.executable, str(RUNNER), "--remote-root", str(tmp_path / "remote"),
        "--projectile-z", "2", "--projectile-a", "4",
        "--energies-mevu", "100",
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "projectile=GenericIon(2,4) Z=2 A=4" in result.stdout
    assert "energy=100 MeV/u total=400 MeV" in result.stdout
    assert "ion_z2a4_100mevu" in result.stdout


def test_fixed_energy_mode_is_explicit_and_uses_no_em_module(tmp_path: Path) -> None:
    result = subprocess.run([
        sys.executable, str(RUNNER), "--remote-root", str(tmp_path / "remote"),
        "--energies-mevu", "0.5", "--histories", "100",
        "--extraction-mode", "fixed-energy-package-sampling",
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "g4h-phy_QGSP_BIC_HP" in result.stdout
    assert "g4h-elastic_HP" in result.stdout
    assert "g4em-standard" not in result.stdout
    assert "--continuous-em-loss-enabled false" in result.stdout
    assert "--sampling-purpose elastic-package-fixed-energy-sampling-not-dose-reference" in result.stdout


def test_aggregate_preserves_mixed_extraction_provenance(tmp_path: Path) -> None:
    degrading = make_source(tmp_path / "degrading", 0.25)
    fixed = make_source(
        tmp_path / "fixed", 1.25, extraction_mode="fixed-energy-package-sampling",
        continuous_em_loss_enabled=False,
    )
    prefix = tmp_path / "mixed"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(degrading),
        "--input-metadata", str(fixed), "--output-prefix", str(prefix),
        "--energy-min-mevu", "0", "--energy-max-mevu", "1",
    ])
    assert result.returncode == 0, result.stderr
    metadata = json.loads(Path(f"{prefix}.metadata.json").read_text())
    provenance = metadata["provenance"]
    assert provenance["mixed_extraction_modes"] is True
    assert provenance["contains_non_dose_reference_sampling"] is True
    assert provenance["extraction_modes"] == [
        "fixed-energy-package-sampling", "transport-degrading",
    ]
    assert [source["continuous_em_loss_enabled"] for source in provenance["sources"]] == [True, False]


def test_aggregate_reindexes_runs_and_compiles_without_nearest_fill(tmp_path: Path) -> None:
    first = make_source(tmp_path / "run1", 1.25, old_id=1)
    second = make_source(tmp_path / "run2", 2.25, old_id=1)
    prefix = tmp_path / "combined" / "proton_elastic"
    direct = make_direct_query(tmp_path / "direct_query")
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(first),
        "--input-metadata", str(second), "--output-prefix", str(prefix),
        "--energy-min-mevu", "1", "--energy-max-mevu", "2", "--compile",
    ], direct)
    assert result.returncode == 0, result.stderr
    with gzip.open(Path(f"{prefix}.interactions.csv.gz"), "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["interaction_id"] for row in rows] == ["1", "2"]
    assert [row["product_offset_zero_based"] for row in rows] == ["0", "0"]
    report = json.loads(Path(f"{prefix}.coverage.json").read_text())
    assert report["energy_grid"]["interval"] == "[minimum, maximum)"
    assert report["event_count_summary"] == {
        "minimum": 1, "p05_nearest_rank": 1, "median_nearest_rank": 1,
    }
    assert report["xs_empty_bins"] == []
    assert report["bins"][0]["elastic_xs_per_mm"] == {
        "median": 0.0125, "minimum": 0.0125, "maximum": 0.0125,
        "absolute_spread": 0.0, "relative_spread_to_median": 0.0,
    }
    assert Path(f"{prefix}.elastic_xs.csv").read_bytes() == direct[0].read_bytes()
    xs_sidecar = json.loads(Path(f"{prefix}.elastic_xs.metadata.json").read_text())
    assert xs_sidecar["source"] == "direct query"
    assert xs_sidecar["aggregation"].startswith("none;")
    assert xs_sidecar["event_xs_diagnostics"].endswith("not used for runtime XS")
    assert xs_sidecar["covered_points"] == DIRECT_QUERY_POINTS
    assert xs_sidecar["complete"] is True
    assert xs_sidecar["output"]["sha256"] == digest(Path(f"{prefix}.elastic_xs.csv"))
    sidecar = json.loads(Path(f"{prefix}.compiled.json").read_text())
    assert sidecar["bin_semantics"]["empty_bin_policy"] == "none"
    assert sidecar["bin_semantics"]["nearest_fill_aliases"] == []


def test_aggregate_default_grid_starts_at_zero(tmp_path: Path) -> None:
    source = make_default_grid_source(tmp_path / "source")
    prefix = tmp_path / "default_grid"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(source),
        "--output-prefix", str(prefix), "--compile",
    ])
    assert result.returncode == 0, result.stderr
    report = json.loads(Path(f"{prefix}.coverage.json").read_text())
    assert report["energy_grid"]["minimum_MeV_per_u"] == 0.0
    assert report["energy_grid"]["maximum_exclusive_MeV_per_u"] == 251.0
    assert len(report["bins"]) == 251
    sidecar = json.loads(Path(f"{prefix}.compiled.json").read_text())
    assert sidecar["energy_range_MeV_per_u"] == {"minimum": 0.0, "maximum": 251.0}
    assert sidecar["records"]["bins"] == 251


def test_aggregate_reports_gap_and_refuses_compile(tmp_path: Path) -> None:
    source = make_source(tmp_path / "run", 1.25)
    prefix = tmp_path / "gap"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(source),
        "--output-prefix", str(prefix), "--energy-min-mevu", "1",
        "--energy-max-mevu", "2", "--compile",
    ])
    assert result.returncode != 0
    report = json.loads(Path(f"{prefix}.coverage.json").read_text())
    assert report["empty_bins"] == [1]
    assert report["xs_empty_bins"] == [1]
    assert report["eligible_for_compile"] is False
    assert not Path(f"{prefix}.bin").exists()


def test_aggregate_enforces_configured_minimum_events(tmp_path: Path) -> None:
    source = make_source(tmp_path / "run", 1.25)
    prefix = tmp_path / "low"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(source),
        "--output-prefix", str(prefix), "--energy-min-mevu", "1",
        "--energy-max-mevu", "1", "--min-events-per-bin", "2", "--compile",
    ])
    assert result.returncode != 0
    report = json.loads(Path(f"{prefix}.coverage.json").read_text())
    assert report["empty_bins"] == []
    assert report["bins_below_minimum"] == [0]
    assert not Path(f"{prefix}.bin").exists()


def test_aggregate_does_not_trust_process_metadata(tmp_path: Path) -> None:
    source = make_source(tmp_path / "run", 1.25, process_name="CoulombScat")
    prefix = tmp_path / "wrong_process"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(source),
        "--output-prefix", str(prefix), "--energy-min-mevu", "1",
        "--energy-max-mevu", "1",
    ])
    assert result.returncode != 0
    assert "is not hadElastic" in result.stderr


def test_aggregate_rejects_invalid_elastic_xs(tmp_path: Path) -> None:
    for label, value in (("negative", -0.1), ("nonfinite", "nan"), ("missing", "")):
        source = make_source(tmp_path / label, 1.25, elastic_xs=value)
        result = run_aggregate(tmp_path / label, [
            "--input-metadata", str(source),
            "--output-prefix", str(tmp_path / f"out_{label}"),
            "--energy-min-mevu", "1", "--energy-max-mevu", "1",
        ])
        assert result.returncode != 0
        assert "elastic" in result.stderr.lower()


def test_aggregate_requires_explicit_direct_query_inputs(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", 1.25)
    result = subprocess.run([
        sys.executable, str(AGGREGATOR), "--input-metadata", str(source),
        "--output-prefix", str(tmp_path / "missing_direct"),
        "--energy-min-mevu", "1", "--energy-max-mevu", "1",
    ], text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "direct-xs-csv" in result.stderr
    assert "direct-xs-metadata" in result.stderr


def test_aggregate_rejects_invalid_direct_query_provenance(tmp_path: Path) -> None:
    cases = [
        ("hash", "SHA-256 mismatch", "csv_hash"),
        ("identity", "direct-query identity", "identity"),
        ("model", "direct-query identity", "model"),
        ("grid", "cover required maximum", "grid"),
        ("nonuniform", "not uniform", "nonuniform"),
        ("coverage", "exactly 25101 points", "coverage"),
    ]
    for label, message, mutation in cases:
        case_dir = tmp_path / label
        source = make_source(case_dir / "source", 1.25)
        direct = make_direct_query(case_dir / "direct")
        if mutation == "csv_hash":
            text = direct[0].read_text(encoding="utf-8")
            direct[0].write_text(text.replace("0.0042", "0.0043", 1), encoding="utf-8")
        elif mutation == "identity":
            metadata = json.loads(direct[1].read_text(encoding="utf-8"))
            metadata["projectile"]["Z"] = 2
            direct[1].write_text(json.dumps(metadata), encoding="utf-8")
        elif mutation == "model":
            metadata = json.loads(direct[1].read_text(encoding="utf-8"))
            metadata["physics_model"] = "wrong-model"
            direct[1].write_text(json.dumps(metadata), encoding="utf-8")
        elif mutation == "grid":
            metadata = json.loads(direct[1].read_text(encoding="utf-8"))
            metadata["grid"]["maximum_MeV_per_u"] = 250.0
            direct[1].write_text(json.dumps(metadata), encoding="utf-8")
        elif mutation == "nonuniform":
            lines = direct[0].read_text(encoding="utf-8").splitlines()
            lines[101] = lines[101].replace("1,", "1.01,", 1)
            direct[0].write_text("\n".join(lines) + "\n", encoding="utf-8")
            refresh_direct_metadata(*direct)
        elif mutation == "coverage":
            lines = direct[0].read_text(encoding="utf-8").splitlines()
            direct[0].write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            refresh_direct_metadata(*direct)
        result = run_aggregate(case_dir, [
            "--input-metadata", str(source), "--output-prefix", str(case_dir / "out"),
            "--energy-min-mevu", "1", "--energy-max-mevu", "1",
        ], direct)
        assert result.returncode != 0, label
        assert message.lower() in result.stderr.lower(), (label, result.stderr)


def test_aggregate_accepts_an_ion_query_grid_beyond_proton_range(tmp_path: Path) -> None:
    source = make_default_grid_source(tmp_path / "source")
    direct = make_direct_query(tmp_path / "direct", maximum=401.0)
    prefix = tmp_path / "extended"
    result = run_aggregate(tmp_path, [
        "--input-metadata", str(source), "--output-prefix", str(prefix),
    ], direct)
    assert result.returncode == 0, result.stderr
    assert Path(f"{prefix}.elastic_xs.csv").read_bytes() == direct[0].read_bytes()
    sidecar = json.loads(Path(f"{prefix}.elastic_xs.metadata.json").read_text())
    assert sidecar["grid"]["maximum_MeV_per_u"] == 401.0
    assert sidecar["covered_points"] == 40101


def test_event_xs_diagnostics_do_not_change_runtime_query_table(tmp_path: Path) -> None:
    direct = make_direct_query(tmp_path / "direct", value=0.0031)
    first_source = make_source(tmp_path / "source1", 1.25, elastic_xs=0.0125)
    second_source = make_source(tmp_path / "source2", 1.25, elastic_xs=0.9875)
    common = ["--energy-min-mevu", "1", "--energy-max-mevu", "1"]
    first = run_aggregate(tmp_path / "run1", [
        "--input-metadata", str(first_source), "--output-prefix", str(tmp_path / "one"),
        *common,
    ], direct)
    second = run_aggregate(tmp_path / "run2", [
        "--input-metadata", str(second_source), "--output-prefix", str(tmp_path / "two"),
        *common,
    ], direct)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    one_xs = Path(f"{tmp_path / 'one'}.elastic_xs.csv")
    two_xs = Path(f"{tmp_path / 'two'}.elastic_xs.csv")
    assert one_xs.read_bytes() == direct[0].read_bytes()
    assert two_xs.read_bytes() == direct[0].read_bytes()
    one_report = json.loads(Path(f"{tmp_path / 'one'}.coverage.json").read_text())
    two_report = json.loads(Path(f"{tmp_path / 'two'}.coverage.json").read_text())
    assert one_report["bins"][0]["elastic_xs_per_mm"]["median"] == 0.0125
    assert two_report["bins"][0]["elastic_xs_per_mm"]["median"] == 0.9875
    assert one_report["xs_aggregation"].endswith("not used for runtime XS")
    metadata = json.loads(Path(f"{tmp_path / 'one'}.elastic_xs.metadata.json").read_text())
    assert metadata["source"] == "direct query"
    assert metadata["input"]["sha256"] == digest(direct[0])


if __name__ == "__main__":
    import tempfile

    test_template_has_only_strong_hadronic_elastic_scorer()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        test_runner_defaults_to_side_effect_free_proton_dry_run(temporary / "dry")
        test_runner_converts_mevu_and_prints_generic_ion_expression(temporary / "ion")
        test_fixed_energy_mode_is_explicit_and_uses_no_em_module(temporary / "fixed")
        test_aggregate_preserves_mixed_extraction_provenance(temporary / "mixed")
        test_aggregate_reindexes_runs_and_compiles_without_nearest_fill(temporary / "merge")
        test_aggregate_default_grid_starts_at_zero(temporary / "default")
        test_aggregate_reports_gap_and_refuses_compile(temporary / "gap")
        test_aggregate_enforces_configured_minimum_events(temporary / "low")
        test_aggregate_does_not_trust_process_metadata(temporary / "process")
        test_aggregate_rejects_invalid_elastic_xs(temporary / "xs")
        test_aggregate_requires_explicit_direct_query_inputs(temporary / "required")
        test_aggregate_rejects_invalid_direct_query_provenance(temporary / "provenance")
        test_aggregate_accepts_an_ion_query_grid_beyond_proton_range(temporary / "extended")
        test_event_xs_diagnostics_do_not_change_runtime_query_table(temporary / "decoupled")
    print("elastic campaign tests passed")
