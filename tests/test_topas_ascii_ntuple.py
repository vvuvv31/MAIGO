"""Synthetic contract tests for the fluctuation TOPAS ntuple reader."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "startup/package_tools"))

from topas_ascii_ntuple import (  # noqa: E402
    ENERGY_LOSS_COLUMNS,
    read_energy_loss_fluctuation_ntuple,
)


def write_fixture(
    tmp_path: Path,
    rows: list[str],
    *,
    columns: tuple[str, ...] = ENERGY_LOSS_COLUMNS,
    scored_entries: int | None = None,
) -> Path:
    stem = tmp_path / "fluctuation.sample"
    entries = len(rows) if scored_entries is None else scored_entries
    header_lines = [
        "Number of Original Histories: 2",
        f"Number of Scored Entries: {entries}",
        "",
        "Columns of data are as follows:",
        *(f"{index:2d}: {name}" for index, name in enumerate(columns, start=1)),
        "",
    ]
    Path(f"{stem}.header").write_text("\n".join(header_lines), encoding="utf-8")
    Path(f"{stem}.phsp").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stem


VALID_ROWS = [
    "0 10 3 1 1 1 G4_WATER 100 99.75 0.25 0.20 0.5 2 1 1 exited",
    "0 11 3 1 1 1 G4_WATER 100 99.50 0.50 0.41 0.5 3 true true exited",
]


class TopasAsciiNtupleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reads_stem_header_and_typed_rows(self) -> None:
        stem = write_fixture(self.root, VALID_ROWS)
        header, rows = read_energy_loss_fluctuation_ntuple(stem)
        self.assertEqual(header.original_histories, 2)
        self.assertEqual(header.scored_entries, 2)
        self.assertEqual(rows[0].atomic_number, 1)
        self.assertAlmostEqual(rows[0].kinetic_energy_loss_mev, 0.25)
        self.assertTrue(rows[0].material_consistent)
        self.assertEqual(rows[1].step_count, 3)
        self.assertEqual(rows[1].completion_status, "exited")

        from_header = read_energy_loss_fluctuation_ntuple(Path(f"{stem}.header"))
        from_phsp = read_energy_loss_fluctuation_ntuple(Path(f"{stem}.phsp"))
        self.assertEqual(from_header, (header, rows))
        self.assertEqual(from_phsp, (header, rows))

    def test_rejects_column_schema_drift(self) -> None:
        columns = list(ENERGY_LOSS_COLUMNS)
        columns[9] = "Energy Loss"
        stem = write_fixture(self.root, VALID_ROWS, columns=tuple(columns))
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            read_energy_loss_fluctuation_ntuple(stem)

    def test_rejects_header_row_count_mismatch(self) -> None:
        stem = write_fixture(self.root, VALID_ROWS, scored_entries=3)
        with self.assertRaisesRegex(ValueError, "declares 3.*parsed 2"):
            read_energy_loss_fluctuation_ntuple(stem)

    def test_rejects_malformed_rows(self) -> None:
        cases = [
        ("0 10 3 1 1", "expected 16 fields"),
        (
            "0 10 3 1 1 1 G4_WATER 100 99.75 nan 0.20 0.5 2 1 1 exited",
            "non-finite",
        ),
        (
            "0 10 3 1 1 1 G4_WATER 100 99.75 0.25 0.20 0.5 2 maybe 1 exited",
            "expected boolean",
        ),
        (
            "0 10 3 1 1 1 G4_WATER 100 99.75 0.25 0.20 inf 2 1 1 exited",
            "non-finite",
        ),
        ]
        for row, message in cases:
            with self.subTest(message=message):
                stem = write_fixture(self.root, [row])
                with self.assertRaisesRegex(ValueError, message):
                    read_energy_loss_fluctuation_ntuple(stem)


if __name__ == "__main__":
    unittest.main()
