#!/usr/bin/env python3
"""Focused tests for detailed TOPAS particle-dose partitioning."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_topas_species import (
    DETAIL_COLUMNS,
    SPECIES,
    nonnegative_residual,
    write_combined_csv,
)


class TopasSpeciesPartitionTests(unittest.TestCase):
    def test_nonnegative_residual_closes_parent(self) -> None:
        parent = np.asarray([1.0, 2.0, 3.0])
        first = np.asarray([0.2, 0.7, 1.1])
        second = np.asarray([0.3, 0.5, 0.4])

        residual, minimum = nonnegative_residual(parent, [first, second], "test")

        np.testing.assert_allclose(first + second + residual, parent)
        self.assertAlmostEqual(minimum, 0.5)

    def test_overlapping_children_are_rejected(self) -> None:
        parent = np.asarray([1.0, 1.0])
        child = np.asarray([0.5, 1.1])

        with self.assertRaisesRegex(SystemExit, "children exceed their parent"):
            nonnegative_residual(parent, [child], "overlap")

    def test_combined_csv_preserves_legacy_and_detail_columns(self) -> None:
        depth = np.asarray([0.25, 0.75])
        total = np.asarray([1.0, 2.0])
        components = {name: np.zeros(2) for name in (*SPECIES, "other")}
        components["other"] = total.copy()
        details = {name: np.zeros(2) for name in DETAIL_COLUMNS}
        details["unclassified"] = total.copy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "combined.csv"
            write_combined_csv(path, depth, total, components, details)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 2)
        self.assertIn("other_MeV_per_primary", rows[0])
        self.assertIn("electron_positron_MeV_per_primary", rows[0])
        self.assertIn("helium3_MeV_per_primary", rows[0])
        self.assertEqual(float(rows[1]["unclassified_MeV_per_primary"]), 2.0)


if __name__ == "__main__":
    unittest.main()
