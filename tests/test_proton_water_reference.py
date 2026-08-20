#!/usr/bin/env python3
"""Synthetic contract tests for the TOPAS proton-water reference converter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "startup/proton_water/convert_topas_reference.py"
TEMPLATE = ROOT / "startup/proton_water/proton_water_cascade.txt.in"


def write_grid(path: Path, quantity: str, unit: str, values: list[float], width_cm: float = 0.5) -> None:
    lines = [
        "# TOPAS Version: 4.2.p3",
        f"# Results for scorer: {quantity}",
        "# X in 1 bin  of 10 cm",
        "# Y in 1 bin  of 10 cm",
        f"# Z in {len(values)} bins of {width_cm:g} cm",
        f"# {quantity} ( {unit} ) : Sum",
    ]
    lines.extend(f"0, 0, {index}, {value:g}" for index, value in enumerate(values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_reaction_fixture(path: Path, histories: int) -> None:
    path.with_suffix(".header").write_text(
        f"Number of Original Histories: {histories}\nNumber of Scored Entries: 2\n",
        encoding="ascii",
    )
    rows = [
        # kind run thread event sequence interaction_track track parent pdg name Z A q E Einc xs projectile Z/A x y z dx dy dz weight process type subtype model
        "interaction 0 0 2 0 1 1 0 2212 proton 1 1 1 90 100 0.0 0.1 1 1 0 0 -9 0 0 1 1 protonInelastic 4 121 -1",
        "interaction 0 0 3 0 1 1 0 2212 proton 1 1 1 80 100 0.0 0.1 1 1 0 0 -4 0 0 1 1 protonInelastic 4 121 -1",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


class ProtonReferenceContractTest(unittest.TestCase):
    def make_input(self, root: Path) -> tuple[Path, Path]:
        stem = "proton_100MeV_water_reference"
        base = root / stem
        write_grid(base.with_name(base.name + "_total_energy_deposit.csv"), "EnergyDeposit", "MeV", [10, 8, 4, 1])
        write_grid(base.with_name(base.name + "_primary_proton_energy_deposit.csv"), "EnergyDeposit", "MeV", [9, 7, 3, 1])
        write_grid(base.with_name(base.name + "_total_dose.csv"), "DoseToMedium", "Gy", [1, 0.8, 0.4, 0.1])
        write_grid(base.with_name(base.name + "_primary_proton_dose.csv"), "DoseToMedium", "Gy", [0.9, 0.7, 0.3, 0.1])
        write_grid(
            base.with_name(base.name + "_primary_let.csv"),
            "myHadronLET",
            "MeV/mm/(g/cm3)",
            [5, 6, 7, 8],
        )
        write_grid(
            base.with_name(base.name + "_primary_let_denominator.csv"),
            "myHadronLET_Denominator",
            "MeV",
            [1, 1, 1, 1],
        )
        write_grid(
            base.with_name(base.name + "_all_hadron_let.csv"),
            "myHadronLET",
            "MeV/mm/(g/cm3)",
            [5.5, 6.5, 7.5, 8.5],
        )
        write_grid(
            base.with_name(base.name + "_all_hadron_let_denominator.csv"),
            "myHadronLET_Denominator",
            "MeV",
            [1.1, 1.1, 1.1, 1.1],
        )
        # TOPAS writes world-Z rows in the opposite order from physical
        # entrance depth.  These raw rows therefore become [10, 8, 4, 1]
        # after conversion, a valid non-increasing survival profile.
        write_grid(
            base.with_name(base.name + "_primary_crossing_count.csv"),
            "PrimaryCrossingCount",
            "",
            [1, 4, 8, 10],
        )
        write_grid(base.with_name(base.name + "_primary_fluence.csv"), "Fluence", "/mm2", [0.001, 0.0008, 0.0004, 0.0001])
        write_reaction_fixture(base.with_name(base.name + "_cascade.phsp"), histories=10)
        return base, root / "converted"

    def run_converter(self, input_dir: Path, output_dir: Path, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(CONVERTER),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--stem",
            "proton_100MeV_water_reference",
            "--histories",
            "10",
            "--phantom-half-length-mm",
            "10",
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if expect_success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_converter_emits_compare_schema_and_reaction_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "raw"
            input_dir.mkdir()
            base, output_dir = self.make_input(input_dir)
            self.run_converter(input_dir, output_dir)

            expected = {
                "total_energy_deposit.csv": ("depth_mm", "energy_deposition_MeV"),
                "primary_proton_energy_deposit.csv": ("depth_mm", "energy_deposition_MeV"),
                "primary_let.csv": ("depth_mm", "numerator_MeV"),
                "all_hadron_let.csv": ("depth_mm", "numerator_MeV"),
                "primary_let_denominator.csv": ("depth_mm", "denominator_MeV"),
                "all_hadron_let_denominator.csv": ("depth_mm", "denominator_MeV"),
                "primary_survival.csv": ("depth_mm", "count"),
                "primary_fluence_diagnostic.csv": ("depth_mm", "count"),
                "first_reactions.csv": ("depth_mm", "count"),
                "reaction_rate.csv": ("depth_mm", "count"),
            }
            for filename, header in expected.items():
                path = output_dir / filename
                self.assertTrue(path.is_file(), filename)
                self.assertTrue(path.read_text(encoding="ascii").startswith("# scorer:"))
                with path.open(encoding="ascii", newline="") as stream:
                    rows = list(csv.reader(line for line in stream if not line.startswith("#")))
                self.assertEqual(tuple(rows[0]), header, filename)
                self.assertEqual(len(rows), 5, filename)

            with (output_dir / "primary_survival.csv").open(encoding="ascii", newline="") as stream:
                survival = list(csv.DictReader(line for line in stream if not line.startswith("#")))
            self.assertEqual([float(row["count"]) for row in survival], [10, 8, 4, 1])
            diagnostic = list(
                csv.DictReader(
                    line
                    for line in (output_dir / "primary_fluence_diagnostic.csv").read_text(encoding="ascii").splitlines()
                    if not line.startswith("#")
                )
            )
            self.assertEqual([float(row["count"]) for row in diagnostic], [1, 4, 8, 10])
            with (output_dir / "first_reactions.csv").open(encoding="ascii", newline="") as stream:
                first = list(csv.DictReader(line for line in stream if not line.startswith("#")))
            with (output_dir / "reaction_rate.csv").open(encoding="ascii", newline="") as stream:
                rate = list(csv.DictReader(line for line in stream if not line.startswith("#")))
            self.assertEqual([float(row["count"]) for row in first], [0, 0, 1, 1])
            self.assertEqual([float(row["count"]) for row in rate], [0, 0, 1, 1])
            self.assertTrue((output_dir / "manifest.json").is_file())
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="ascii"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["scorers"]["primary_survival"]["scorer"], "PrimaryCrossingCount")
            self.assertIn("at most once per depth bin", manifest["scorers"]["primary_survival"]["semantics"])
            self.assertIn("not formal survival", manifest["scorers"]["primary_fluence_diagnostic"]["semantics"])
            self.assertEqual(manifest["scorers"]["primary_let"]["scorer"], "myHadronLET")
            self.assertEqual(manifest["scorers"]["primary_let_denominator"]["scorer"], "myHadronLET_Denominator")
            self.assertIn("charged mesons and muons excluded", manifest["scorers"]["all_hadron_let"]["coverage"])
            self.assertTrue(base.with_name(base.name + "_cascade.header").is_file())

    def test_missing_required_scorer_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "raw"
            input_dir.mkdir()
            self.make_input(input_dir)
            (input_dir / "proton_100MeV_water_reference_primary_let.csv").unlink()
            result = self.run_converter(input_dir, Path(directory) / "converted", expect_success=False)
            self.assertIn("required TOPAS scorer is missing", result.stderr)

    def test_primary_crossing_count_rejects_non_integer_and_increasing_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "raw"
            input_dir.mkdir()
            base, output_dir = self.make_input(input_dir)
            crossing = base.with_name(base.name + "_primary_crossing_count.csv")
            write_grid(crossing, "PrimaryCrossingCount", "", [1, 4.5, 8, 10])
            result = self.run_converter(input_dir, output_dir, expect_success=False)
            self.assertIn("is not an integer", result.stderr)

            write_grid(crossing, "PrimaryCrossingCount", "", [1, 4, 9, 8])
            result = self.run_converter(input_dir, output_dir, expect_success=False)
            self.assertIn("survival must be non-increasing", result.stderr)

    def test_template_is_proton_binary_cascade_reference(self) -> None:
        text = TEMPLATE.read_text(encoding="ascii")
        self.assertIn('BeamParticle = "proton"', text)
        self.assertIn('g4h-phy_QGSP_BIC_HP', text)
        self.assertNotIn('"g4ion-inclxx"', text)
        self.assertIn('s:Sc/PrimaryProtonLET/Quantity = "myHadronLET"', text)
        self.assertIn('s:Sc/PrimaryProtonLETDenominator/Quantity = "myHadronLET_Denominator"', text)
        self.assertIn('s:Sc/AllHadronLETDenominator/Quantity = "myHadronLET_Denominator"', text)
        self.assertIn('s:Sc/PrimaryCrossing/Quantity = "PrimaryCrossingCount"', text)
        self.assertIn("i:Sc/PrimaryCrossing/ProjectileZ = 1", text)
        self.assertIn("i:Sc/PrimaryCrossing/ProjectileA = 1", text)
        self.assertNotIn('Quantity = "ProtonLET"', text)
        for suffix in (
            "_total_energy_deposit",
            "_primary_proton_energy_deposit",
            "_primary_let",
            "_all_hadron_let",
            "_primary_crossing_count",
            "_primary_fluence",
            "_cascade",
        ):
            self.assertIn(f'@OUTPUT_STEM@{suffix}', text)


if __name__ == "__main__":
    unittest.main()
