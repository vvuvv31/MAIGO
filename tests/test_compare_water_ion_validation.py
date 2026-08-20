#!/usr/bin/env python3
"""Synthetic contract tests for scripts/compare_water_ion_validation.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "compare_water_ion_validation.py"
COVERAGE = "charged ions and nuclei with atomic number Z>=1; charged mesons and muons excluded"


def write_profile(path: Path, column: str, values: list[float], depths: list[float] | None = None) -> None:
    if depths is None:
        depths = [float(index) for index in range(len(values))]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["depth_mm", column])
        writer.writeheader()
        for depth, value in zip(depths, values):
            writer.writerow({"depth_mm": depth, column: value})


def set_csv_value(path: Path, column: str, depth: float, value: float) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if float(row["depth_mm"]) == depth:
            row[column] = str(value)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class CompareWaterIonValidationTest(unittest.TestCase):
    def make_case(self, root: Path, gpu_dose_scale: float = 1.0) -> dict[str, Path]:
        dose = [1.0, 2.0, 8.0, 10.0, 8.0, 3.0, 0.0]
        gpu_dose = [value * gpu_dose_scale for value in dose]
        primary_let = [1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 0.0]
        all_let = [1.1, 1.2, 1.3, 1.5, 1.7, 2.1, 0.0]
        primary_den = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        all_den = [1.1, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0]
        write_profile(root / "topas_dose.csv", "energy_deposition_MeV", dose)
        write_profile(root / "gpu_dose.csv", "energy_deposition_MeV", gpu_dose)
        write_profile(root / "topas_primary_let.csv", "letd_MeV_per_mm_per_g_cm3", [a * b for a, b in zip(primary_let, primary_den)])
        write_profile(root / "topas_primary_let_denominator.csv", "denominator_MeV", primary_den)
        write_profile(root / "topas_all_charged_ion_let.csv", "letd_MeV_per_mm_per_g_cm3", [a * b for a, b in zip(all_let, all_den)])
        write_profile(root / "topas_all_charged_ion_let_denominator.csv", "denominator_MeV", all_den)
        with (root / "gpu_let.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "depth_mm",
                    "primary_numerator",
                    "primary_denominator_MeV",
                    "all_hadron_numerator",
                    "all_hadron_denominator_MeV",
                ],
            )
            writer.writeheader()
            for depth, primary, all_value, primary_weight, all_weight in zip(
                range(len(primary_let)), primary_let, all_let, primary_den, all_den
            ):
                writer.writerow(
                    {
                        "depth_mm": depth,
                        "primary_numerator": primary * 1.01 * primary_weight,
                        "primary_denominator_MeV": primary_weight,
                        "all_hadron_numerator": all_value * 1.02 * all_weight,
                        "all_hadron_denominator_MeV": all_weight,
                    }
                )
        survival = [1000.0, 995.0, 990.0, 984.0, 978.0, 970.0, 965.0]
        first_reactions = [0.0, 5.0, 4.0, 5.0, 4.0, 7.0, 0.0]
        reaction_rate = [0.0, 5.0, 5.0, 6.0, 6.0, 8.0, 5.0]
        gpu_primary_reactions = first_reactions
        for stem, values in (
            ("topas_survival", survival),
            ("gpu_survival", survival),
            ("topas_first_reactions", first_reactions),
            ("topas_reaction_rate", reaction_rate),
            ("gpu_primary_reactions", gpu_primary_reactions),
        ):
            write_profile(root / f"{stem}.csv", "count", values)
        manifest = {
            "schema_version": 2,
            "depth_bins": len(dose),
            "depth_bin_width_mm": 1.0,
            "let_definition": {
                "weighting": "dose",
                "all_hadron_coverage": COVERAGE,
            },
            "scorers": {
                "primary_let": {
                    "coverage": "primary proton (Z=1,A=1,generation=Primary)",
                    "denominator": "topas_primary_let_denominator.csv",
                },
                "all_hadron_let": {
                    "coverage": COVERAGE,
                    "denominator": "topas_all_charged_ion_let_denominator.csv",
                },
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "topas_dose": root / "topas_dose.csv",
            "gpu_dose": root / "gpu_dose.csv",
            "topas_primary_let": root / "topas_primary_let.csv",
            "topas_primary_den": root / "topas_primary_let_denominator.csv",
            "topas_all_let": root / "topas_all_charged_ion_let.csv",
            "topas_all_den": root / "topas_all_charged_ion_let_denominator.csv",
            "gpu_let": root / "gpu_let.csv",
            "topas_survival": root / "topas_survival.csv",
            "gpu_survival": root / "gpu_survival.csv",
            "topas_first_reactions": root / "topas_first_reactions.csv",
            "topas_reaction_rate": root / "topas_reaction_rate.csv",
            "gpu_primary_reactions": root / "gpu_primary_reactions.csv",
            "manifest": manifest_path,
            "output": root / "report.json",
        }

    def run_case(self, root: Path, paths: dict[str, Path], require_pass: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SCRIPT),
            "--topas-histories",
            "1000",
            "--gpu-histories",
            "1000",
            "--topas-dose",
            str(paths["topas_dose"]),
            "--gpu-dose",
            str(paths["gpu_dose"]),
            "--topas-primary-let-numerator",
            str(paths["topas_primary_let"]),
            "--topas-primary-let-denominator",
            str(paths["topas_primary_den"]),
            "--topas-all-charged-ion-let-numerator",
            str(paths["topas_all_let"]),
            "--topas-all-charged-ion-let-denominator",
            str(paths["topas_all_den"]),
            "--gpu-let",
            str(paths["gpu_let"]),
            "--topas-manifest",
            str(paths["manifest"]),
            "--topas-survival",
            str(paths["topas_survival"]),
            "--gpu-survival",
            str(paths["gpu_survival"]),
            "--topas-first-reactions",
            str(paths["topas_first_reactions"]),
            "--topas-reaction-rate",
            str(paths["topas_reaction_rate"]),
            "--gpu-primary-reactions",
            str(paths["gpu_primary_reactions"]),
            "--output",
            str(paths["output"]),
        ]
        if require_pass:
            command.append("--require-pass")
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_raw_moments_pass_and_legacy_source_mapping_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            result = self.run_case(root, paths)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertIn("all_charged_ion_let_mean_absolute_percent", report["metrics"])
            self.assertNotIn("all_hadron_let_mean_absolute_percent", report["metrics"])
            self.assertEqual(report["sources"]["gpu_let"]["all_charged_ion_numerator_column"], "all_hadron_numerator")
            self.assertEqual(report["sources"]["topas_reference"]["source_scorer_keys"]["all_charged_ion_let"], "all_hadron_let")

    def test_first_reaction_metric_excludes_continuations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            result = self.run_case(root, paths)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(paths["output"].read_text(encoding="utf-8"))["metrics"]
            self.assertEqual(metrics["topas_first_reactions"], 25.0)
            self.assertEqual(metrics["topas_all_primary_interaction_records"], 35.0)
            self.assertEqual(metrics["continuation_excess"], 10.0)
            self.assertEqual(metrics["gpu_primary_reactions"], 25.0)
            self.assertEqual(metrics["first_reaction_relative_difference"], 0.0)
            self.assertNotIn("reaction_relative_percent", metrics)

    def test_missing_first_reactions_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            paths["topas_first_reactions"].unlink()
            result = self.run_case(root, paths, require_pass=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("topas_first_reactions", result.stderr)

    def test_both_zero_denominators_are_skipped_without_division(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            result = self.run_case(root, paths)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(paths["output"].read_text(encoding="utf-8"))
            stats = report["let_denominator_policy"]
            self.assertEqual(stats["primary"]["skipped_both_zero_denominator_bins"], 1)
            self.assertEqual(stats["all_charged_ion"]["skipped_both_zero_denominator_bins"], 1)

    def test_unilateral_zero_denominator_fails_without_division(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            set_csv_value(paths["topas_primary_den"], "denominator_MeV", 5.0, 1.0)
            set_csv_value(paths["gpu_let"], "primary_denominator_MeV", 5.0, 0.0)
            result = self.run_case(root, paths)
            self.assertEqual(result.returncode, 1)
            report = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertGreater(report["let_denominator_policy"]["primary"]["zero_denominator_mismatch_bins"], 0)

    def test_grid_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            write_profile(paths["topas_all_den"], "denominator_MeV", [1.1] * 7, depths=[0, 1, 2, 3, 4, 5, 7])
            result = self.run_case(root, paths, require_pass=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("depth grid mismatch", result.stderr)

    def test_coverage_metadata_mismatch_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root)
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["scorers"]["all_hadron_let"]["coverage"] = "primary proton only"
            paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            result = self.run_case(root, paths, require_pass=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("coverage metadata mismatch", result.stderr)

    def test_large_dose_bias_fails_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_case(root, gpu_dose_scale=1.10)
            result = self.run_case(root, paths)
            self.assertEqual(result.returncode, 1)
            report = json.loads(paths["output"].read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["integral"])


if __name__ == "__main__":
    unittest.main()
