"""Synthetic contract tests for the fluctuation runtime CSV compiler."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "startup/package_tools/compile_energy_loss_fluctuation_grid.py"


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_point(
    path: Path,
    energy: float,
    density: float,
    *,
    z: int = 1,
    a: int = 1,
    probabilities: list[float] | None = None,
    quantiles: list[float] | None = None,
) -> None:
    probabilities = probabilities or [0.0, 0.5, 1.0]
    quantiles = quantiles or [0.5, 1.0, 1.5]
    point = {
        "schema": "maigo-energy-loss-fluctuation-point-v1",
        "projectile": {"Z": z, "A": a},
        "material": "G4_WATER",
        "energy_MeV_per_u": energy,
        "areal_density_g_per_cm2": density,
        "histories": 100,
        "sample_energy_loss_MeV": {"mean": 0.2, "minimum": 0.1, "maximum": 0.3},
        "inverse_cdf": {
            "probabilities": probabilities,
            "loss_over_mean_quantiles": quantiles,
        },
        "sources": {
            "header": {"path": "source.header", "sha256": digest_text("header")},
            "phsp": {"path": "source.phsp", "sha256": digest_text("phsp")},
        },
    }
    path.write_text(json.dumps(point) + "\n", encoding="utf-8")


class CompileEnergyLossFluctuationGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def points(self) -> list[Path]:
        paths: list[Path] = []
        for energy in (100.0, 200.0):
            for density in (0.01, 0.02):
                path = self.root / f"point_{energy:g}_{density:g}.json"
                write_point(path, energy, density)
                paths.append(path)
        return paths

    def run_compile(
        self, points: list[Path]
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        output = self.root / "runtime.csv"
        metadata = self.root / "runtime.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--points",
                *(str(path) for path in points),
                "--output", str(output), "--output-metadata", str(metadata),
            ],
            text=True, capture_output=True, check=False,
        )
        return result, output, metadata

    def test_compiles_complete_grid_in_runtime_order(self) -> None:
        points = list(reversed(self.points()))
        result, output, metadata_path = self.run_compile(points)
        self.assertEqual(result.returncode, 0, result.stderr)
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(
            rows[0],
            [
                "projectile_Z", "projectile_A", "material",
                "energy_MeV_per_u", "areal_density_g_per_cm2",
                "q_0", "q_0.5", "q_1",
            ],
        )
        self.assertEqual(
            [(float(row[3]), float(row[4])) for row in rows[1:]],
            [(100.0, 0.01), (100.0, 0.02), (200.0, 0.01), (200.0, 0.02)],
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["grid"]["point_count"], 4)
        self.assertEqual(metadata["histories"]["total"], 400)
        self.assertFalse(metadata["normalization"]["empirical_energy_or_projectile_scale"])
        self.assertEqual(len(metadata["output"]["sha256"]), 64)

    def test_rejects_energy_with_only_one_density(self) -> None:
        result, output, _ = self.run_compile(self.points()[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at least two density points", result.stderr)
        self.assertFalse(output.exists())

    def test_compiles_ragged_density_grid(self) -> None:
        points: list[Path] = []
        for energy, densities in (
            (100.0, (0.001, 0.0025)),
            (200.0, (0.002, 0.0025)),
        ):
            for density in densities:
                path = self.root / f"ragged_{energy:g}_{density:g}.json"
                write_point(path, energy, density)
                points.append(path)
        result, output, metadata_path = self.run_compile(points)
        self.assertEqual(result.returncode, 0, result.stderr)
        with output.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(
            [(float(row[3]), float(row[4])) for row in rows[1:]],
            [(100.0, 0.001), (100.0, 0.0025),
             (200.0, 0.002), (200.0, 0.0025)],
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["grid"]["areal_densities_g_per_cm2"],
            [0.001, 0.002, 0.0025],
        )
        self.assertEqual(
            metadata["grid"]["areal_densities_by_energy_g_per_cm2"],
            [
                {"energy_MeV_per_u": 100.0, "values": [0.001, 0.0025]},
                {"energy_MeV_per_u": 200.0, "values": [0.002, 0.0025]},
            ],
        )

    def test_rejects_duplicate_point(self) -> None:
        points = self.points()
        duplicate = self.root / "duplicate.json"
        write_point(duplicate, 100.0, 0.01)
        result, _, _ = self.run_compile(points + [duplicate])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("duplicate fluctuation grid point", result.stderr)

    def test_rejects_identity_or_probability_mismatch(self) -> None:
        for mutation in ("identity", "probability"):
            with self.subTest(mutation=mutation):
                points = self.points()
                if mutation == "identity":
                    write_point(points[0], 100.0, 0.01, z=2, a=4)
                    expected = "does not match"
                else:
                    write_point(
                        points[0], 100.0, 0.01,
                        probabilities=[0.0, 0.25, 1.0],
                        quantiles=[0.0, 1.0, 4.0 / 3.0],
                    )
                    expected = "probability grid does not match"
                result, _, _ = self.run_compile(points)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_rejects_nonunit_or_nonmonotone_quantiles(self) -> None:
        for quantiles, expected in (
            ([0.5, 1.0, 1.6], "not 1"),
            ([0.5, 1.2, 1.0], "nonnegative and nondecreasing"),
        ):
            with self.subTest(quantiles=quantiles):
                points = self.points()
                write_point(points[0], 100.0, 0.01, quantiles=quantiles)
                result, _, _ = self.run_compile(points)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
