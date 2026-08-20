#!/usr/bin/env python3
"""Contract tests for TOPAS primary table conversion."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "startup" / "package_tools" / "deduplicate_primary_tables.py"


class DeduplicatePrimaryTablesTest(unittest.TestCase):
    def run_converter(
        self, kind: str, rows: list[tuple[float, ...]]
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        raw = root / "table.phsp"
        raw.write_text(
            "".join(" ".join(f"{value:.12g}" for value in row) + "\n" for row in rows),
            encoding="utf-8",
        )
        header = root / "table.header"
        header.write_text("synthetic TOPAS table header\n", encoding="utf-8")
        output = root / "runtime.csv"
        diagnostic = root / "diagnostic.csv"
        metadata = root / "metadata.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--kind",
                kind,
                "--input",
                str(raw),
                "--header",
                str(header),
                "--output",
                str(output),
                "--diagnostic-output",
                str(diagnostic),
                "--metadata",
                str(metadata),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        return result, output, diagnostic, metadata, temporary

    def test_stopping_output_uses_electronic_stopping_power(self) -> None:
        row_low = (1.0, 10.0, 3.5, 3.7, 0.1)
        row_high = (2.0, 20.0, 2.5, 2.7, 0.2)
        result, output, diagnostic, metadata, temporary = self.run_converter(
            "stopping", [row_high] * 56 + [row_low] * 56
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        with output.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(
                list(csv.reader(stream)),
                [
                    ["energy_MeVu", "stopping_power_MeV_per_mm"],
                    ["1.0", "3.5"],
                    ["2.0", "2.5"],
                ],
            )
        with diagnostic.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.reader(stream))[0]), 5)
        record = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(record["worker_copies_per_energy"], 56)
        self.assertEqual(record["runtime_source_column_indices"], [0, 2])

    def test_cross_section_output_uses_water_macroscopic_value(self) -> None:
        row_low = (1.0, 1.0, 10.0, 20.0, 0.01, 0.02, 0.03, 33.0)
        row_high = (2.0, 2.0, 11.0, 21.0, 0.04, 0.05, 0.09, 11.0)
        result, output, _diagnostic, metadata, temporary = self.run_converter(
            "cross-section", [row_high] * 56 + [row_low] * 56
        )
        self.addCleanup(temporary.cleanup)
        self.assertEqual(result.returncode, 0, result.stderr)
        with output.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(
                list(csv.reader(stream)),
                [
                    ["energy_MeV_per_u", "water_macroscopic_cross_section_per_mm"],
                    ["1.0", "0.03"],
                    ["2.0", "0.09"],
                ],
            )
        record = json.loads(metadata.read_text(encoding="utf-8"))
        self.assertEqual(record["runtime_source_column_indices"], [0, 6])

    def test_worker_mismatch_is_fatal(self) -> None:
        reference = (1.0, 10.0, 3.5, 3.7, 0.1)
        mismatch = (1.0, 10.0, 3.6, 3.7, 0.1)
        result, output, _diagnostic, _metadata, temporary = self.run_converter(
            "stopping", [reference] * 55 + [mismatch]
        )
        self.addCleanup(temporary.cleanup)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Worker mismatch", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
