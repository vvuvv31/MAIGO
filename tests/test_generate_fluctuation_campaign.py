"""Contract tests for deterministic thin-slab campaign rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "startup/proton_water/generate_fluctuation_campaign.py"


class GenerateFluctuationCampaignTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_generate(
        self,
        *,
        z: int = 1,
        a: int = 1,
        energies: str = "200,100",
        densities: str | None = "0.02,0.01",
        extra_arguments: list[str] | None = None,
        output_name: str = "campaign",
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        output = self.root / output_name
        arguments = [
                sys.executable, str(SCRIPT), "--output-dir", str(output),
                "--projectile-z", str(z), "--projectile-a", str(a),
                "--energies-mevu", energies,
                "--histories", "1000", "--threads", "4",
                "--seed-base", "2026082200", "--production-cut-mm", "0.05",
                "--topas-version", "4.2.p3", "--geant4-version", "11.3.2",
        ]
        if densities is not None:
            arguments.extend(["--areal-densities-g-per-cm2", densities])
        if extra_arguments is not None:
            arguments.extend(extra_arguments)
        result = subprocess.run(
            arguments,
            text=True, capture_output=True, check=False,
        )
        return result, output

    def test_generates_sorted_complete_proton_campaign(self) -> None:
        result, output = self.run_generate()
        self.assertEqual(result.returncode, 0, result.stderr)
        campaign = json.loads(
            (output / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(campaign["grid"]["energies_MeV_per_u"], [100.0, 200.0])
        self.assertEqual(campaign["grid"]["density_mode"], "explicit_common")
        self.assertEqual(
            campaign["grid"]["areal_densities_g_per_cm2"], [0.01, 0.02]
        )
        self.assertEqual(campaign["grid"]["point_count"], 4)
        self.assertEqual(
            [point["seed"] for point in campaign["points"]],
            [2026082200, 2026082201, 2026082202, 2026082203],
        )
        first_manifest_path = output / campaign["points"][0]["run_manifest"]["path"]
        first = json.loads(first_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(first["projectile"]["name"], "proton")
        self.assertAlmostEqual(first["slab_full_thickness_mm"], 0.1)
        self.assertAlmostEqual(first["slab_half_thickness_mm"], 0.05)
        config = output / first["files"]["config"]["path"]
        text = config.read_text(encoding="utf-8")
        self.assertNotIn("@", text)
        self.assertIn('sv:Ph/Default/Modules = 1 "g4em-standard_opt4"', text)
        self.assertNotIn("g4h-elastic", text)
        self.assertNotIn("g4ion", text)
        self.assertIn('s:So/IonBeam/BeamParticle = "proton"', text)

    def test_generic_ion_total_energy_uses_mass_number(self) -> None:
        result, output = self.run_generate(z=2, a=4)
        self.assertEqual(result.returncode, 0, result.stderr)
        campaign = json.loads(
            (output / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        first_path = output / campaign["points"][0]["run_manifest"]["path"]
        first = json.loads(first_path.read_text(encoding="utf-8"))
        self.assertEqual(first["projectile"]["name"], "GenericIon(2,4)")
        self.assertEqual(first["total_energy_MeV"], 400.0)
        config = output / first["files"]["config"]["path"]
        self.assertIn(
            's:So/IonBeam/BeamParticle = "GenericIon(2,4)"',
            config.read_text(encoding="utf-8"),
        )

    def test_step_control_generates_ragged_density_grid(self) -> None:
        table = self.root / "stopping.csv"
        table.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n"
            "10,10\n"
            "20,1\n",
            encoding="utf-8",
        )
        result, output = self.run_generate(
            energies="20,10,15",
            densities=None,
            extra_arguments=[
                "--stopping-power-file", str(table),
                "--maximum-step-mm", "0.025",
                "--maximum-relative-energy-loss", "0.005",
                "--density-fractions", "0.5,1",
            ],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        campaign = json.loads(
            (output / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        grid = campaign["grid"]
        self.assertEqual(grid["density_mode"], "step_control")
        density_rows = grid["areal_densities_by_energy_g_per_cm2"]
        self.assertEqual(
            [row["energy_MeV_per_u"] for row in density_rows],
            [10.0, 15.0, 20.0],
        )
        for actual, expected in zip(
            [row["values"] for row in density_rows],
            [
                [0.00025, 0.0005],
                [0.0006818181818181818, 0.0013636363636363635],
                [0.00125, 0.0025],
            ],
        ):
            for actual_value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(actual_value, expected_value)
        self.assertEqual(grid["point_count"], 6)
        self.assertEqual(
            [point["seed"] for point in campaign["points"]],
            list(range(2026082200, 2026082206)),
        )
        self.assertEqual(
            [(point["energy_MeV_per_u"], point["areal_density_g_per_cm2"])
             for point in campaign["points"]],
            [
                (10.0, 0.00025), (10.0, 0.0005),
                (15.0, 0.0006818181818181818),
                (15.0, 0.0013636363636363635),
                (20.0, 0.00125), (20.0, 0.0025),
            ],
        )
        provenance = grid["step_control"]
        self.assertEqual(provenance["stopping_power_file"]["path"], str(table))
        self.assertEqual(
            provenance["stopping_power_file"]["sha256"],
            hashlib.sha256(table.read_bytes()).hexdigest(),
        )
        self.assertIn("A * energy_MeV_per_u", provenance["formula"])

    def test_step_control_uses_mass_number_in_energy_limit(self) -> None:
        table = self.root / "helium_stopping.csv"
        table.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n10,100\n20,100\n",
            encoding="utf-8",
        )
        result, output = self.run_generate(
            z=2, a=4, energies="10", densities=None,
            extra_arguments=[
                "--stopping-power-file", str(table),
                "--density-fractions", "0.5,1",
            ],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        campaign = json.loads(
            (output / "campaign_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            campaign["grid"]["areal_densities_by_energy_g_per_cm2"],
            [{"energy_MeV_per_u": 10.0, "values": [0.0001, 0.0002]}],
        )

    def test_rejects_invalid_step_control_inputs(self) -> None:
        valid = self.root / "valid.csv"
        valid.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n10,10\n20,1\n",
            encoding="utf-8",
        )
        duplicate = self.root / "duplicate.csv"
        duplicate.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n10,10\n10,9\n",
            encoding="utf-8",
        )
        nonfinite = self.root / "nonfinite.csv"
        nonfinite.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n10,10\n20,nan\n",
            encoding="utf-8",
        )
        cases = [
            (["--stopping-power-file", str(valid)], "outside stopping-power table"),
            (["--stopping-power-file", str(duplicate)], "duplicate stopping-power"),
            (["--stopping-power-file", str(nonfinite)], "nonfinite stopping-power"),
            (["--stopping-power-file", str(valid), "--density-fractions", "1.1,1"],
             "density fractions must be no greater than one"),
        ]
        for index, (arguments, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                result, _ = self.run_generate(
                    energies="100", densities=None,
                    extra_arguments=arguments,
                    output_name=f"invalid_step_{index}",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rejects_both_density_modes(self) -> None:
        table = self.root / "stopping.csv"
        table.write_text(
            "energy_MeVu,stopping_power_MeV_per_mm\n100,10\n200,5\n",
            encoding="utf-8",
        )
        result, _ = self.run_generate(
            extra_arguments=["--stopping-power-file", str(table)]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)

    def test_rejects_duplicate_grid_or_invalid_identity(self) -> None:
        for arguments, expected in (
            ({"energies": "100,100"}, "grid values must be unique"),
            ({"z": 3, "a": 2}, "projectile A must be at least Z"),
        ):
            with self.subTest(arguments=arguments):
                result, _ = self.run_generate(output_name="invalid", **arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_refuses_nonempty_output_directory(self) -> None:
        output = self.root / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("user data\n", encoding="utf-8")
        result, _ = self.run_generate(output_name="occupied")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "user data\n")


if __name__ == "__main__":
    unittest.main()
