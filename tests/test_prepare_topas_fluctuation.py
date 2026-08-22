"""Synthetic tests for one fluctuation-package grid point."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "startup/package_tools/prepare_topas_fluctuation.py"
sys.path.insert(0, str(ROOT / "startup/package_tools"))

from prepare_topas_fluctuation import default_probabilities  # noqa: E402
from prepare_topas_fluctuation import empirical_quantile  # noqa: E402
from prepare_topas_fluctuation import exact_type7_quantile_integral  # noqa: E402
from prepare_topas_fluctuation import trapezoid_integral  # noqa: E402


COLUMNS = (
    "Run ID", "Event ID", "Thread ID", "Primary Track ID",
    "Atomic Number Z", "Mass Number A", "Material Name",
    "Entry Kinetic Energy (MeV)", "Exit Kinetic Energy (MeV)",
    "Primary Kinetic Energy Loss (MeV)", "Primary Local Deposit (MeV)",
    "Primary Path Length (mm)", "Primary Step Count", "Material Consistent",
    "Completed", "Completion Status",
)


def write_fixture(root: Path, rows: list[str], histories: int | None = None) -> Path:
    stem = root / "point"
    original = len(rows) if histories is None else histories
    header = [
        f"Number of Original Histories: {original}",
        f"Number of Scored Entries: {len(rows)}",
        "",
        "Columns of data are as follows:",
        *(f"{index}: {name}" for index, name in enumerate(COLUMNS, 1)),
    ]
    Path(f"{stem}.header").write_text("\n".join(header) + "\n", encoding="utf-8")
    Path(f"{stem}.phsp").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stem


def valid_rows() -> list[str]:
    return [
        f"0 {event} 0 1 1 1 G4_WATER 100 {100 - loss:g} {loss:g} "
        f"{0.8 * loss:g} 0.5 1 1 1 exited"
        for event, loss in enumerate((1.0, 2.0, 3.0, 4.0))
    ]


class PrepareTopasFluctuationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_prepare(
        self, rows: list[str], *, histories: int | None = None
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        stem = write_fixture(self.root, rows, histories)
        output = self.root / "point.json"
        result = subprocess.run(
            [
                sys.executable, str(SCRIPT), "--input", str(stem),
                "--output", str(output), "--projectile-z", "1",
                "--projectile-a", "1", "--material", "G4_WATER",
                "--energy-mev-per-u", "100",
                "--areal-density-g-per-cm2", "0.05",
                "--expected-histories", "4",
            ],
            text=True, capture_output=True, check=False,
        )
        return result, output

    def test_prepares_unit_mean_monotone_inverse_cdf(self) -> None:
        result, output = self.run_prepare(valid_rows())
        self.assertEqual(result.returncode, 0, result.stderr)
        point = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(point["projectile"], {"Z": 1, "A": 1})
        self.assertEqual(point["histories"], 4)
        self.assertAlmostEqual(point["sample_energy_loss_MeV"]["mean"], 2.5)
        probabilities = point["inverse_cdf"]["probabilities"]
        quantiles = point["inverse_cdf"]["loss_over_mean_quantiles"]
        self.assertEqual((probabilities[0], probabilities[-1]), (0.0, 1.0))
        self.assertEqual(len(probabilities), 335)
        self.assertIn(0.99999, probabilities)
        self.assertAlmostEqual(
            probabilities[-2] - probabilities[-3], 0.00001, places=15
        )
        self.assertTrue(all(a <= b for a, b in zip(quantiles, quantiles[1:])))
        integral = sum(
            0.5 * (a + b) * (pb - pa)
            for pa, pb, a, b in zip(
                probabilities, probabilities[1:], quantiles, quantiles[1:]
            )
        )
        self.assertAlmostEqual(integral, 1.0, places=12)
        self.assertEqual(len(point["sources"]["header"]["sha256"]), 64)

    def test_default_grid_preserves_long_tail_mean_before_normalization(self) -> None:
        sample_count = 100_000
        samples = [
            -math.log1p(-(index + 0.5) / sample_count)
            for index in range(sample_count)
        ]
        sample_mean = math.fsum(samples) / sample_count
        normalized = [value / sample_mean for value in samples]
        probabilities = default_probabilities()
        quantiles = [
            empirical_quantile(normalized, probability)
            for probability in probabilities
        ]
        exact_mean = exact_type7_quantile_integral(normalized)
        self.assertLess(
            abs(trapezoid_integral(probabilities, quantiles) / exact_mean - 1.0),
            0.001,
        )

    def test_rejects_missing_history(self) -> None:
        result, output = self.run_prepare(valid_rows()[:3], histories=4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wrote 3 rows for 4 histories", result.stderr)
        self.assertFalse(output.exists())

    def test_rejects_wrong_identity(self) -> None:
        rows = valid_rows()
        rows[0] = rows[0].replace("1 1 G4_WATER", "2 4 G4_WATER")
        result, _ = self.run_prepare(rows)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected projectile Z1A1", result.stderr)

    def test_rejects_non_exiting_primary(self) -> None:
        rows = valid_rows()
        rows[0] = rows[0].replace("1 1 exited", "1 1 stopped")
        result, _ = self.run_prepare(rows)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not exit", result.stderr)

    def test_rejects_energy_closure_failure(self) -> None:
        rows = valid_rows()
        fields = rows[0].split()
        fields[9] = "1.5"
        rows[0] = " ".join(fields)
        result, _ = self.run_prepare(rows)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match recorded loss", result.stderr)


if __name__ == "__main__":
    unittest.main()
