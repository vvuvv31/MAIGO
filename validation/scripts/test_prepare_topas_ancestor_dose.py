#!/usr/bin/env python3
"""Focused tests for sparse ancestor-attributed 3D dose processing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_topas_ancestor_dose import read_sparse_topas, sparse_to_dense


class TopasAncestorDoseTests(unittest.TestCase):
    def test_sparse_rows_reconstruct_xyz_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dose.csv"
            path.write_text(
                "# TOPAS Version: test\n# Parameter File: case.txt\n1, 0, 2, 3.5\n0, 1, 1, 2.0\n",
                encoding="utf-8",
            )
            coordinates, values, metadata = read_sparse_topas(path, (2, 2, 3))
            dense = sparse_to_dense(coordinates, values, (2, 2, 3))

        self.assertEqual(metadata["topas_version"], "test")
        self.assertEqual(dense[1, 0, 2], 3.5)
        self.assertEqual(dense[0, 1, 1], 2.0)
        self.assertEqual(np.count_nonzero(dense), 2)

    def test_header_only_sparse_file_is_valid_zero_dose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("# TOPAS Version: test\n# no nonzero rows\n", encoding="utf-8")
            coordinates, values, _ = read_sparse_topas(path, (2, 2, 3))

        self.assertEqual(coordinates.shape, (0, 3))
        self.assertEqual(values.shape, (0,))

    def test_duplicate_voxels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text("0, 0, 0, 1\n0, 0, 0, 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate voxel"):
                read_sparse_topas(path, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
