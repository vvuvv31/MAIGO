#!/usr/bin/env python3
"""Focused tests for the TOPAS minibeam dense-dose converter."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("prepare_minibeam_topas_dose.py")
SPEC = importlib.util.spec_from_file_location("prepare_minibeam_topas_dose", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MinibeamDoseConverterTest(unittest.TestCase):
    def test_axis_mapping_and_mhd_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "dose.binheader"
            header.write_text(
                "\n".join(
                    (
                        "# X in 3 bins of 0.01 cm",
                        "# Y in 4 bins of 0.02 cm",
                        "# Z in 1 bin  of 0.1 cm",
                        "# DoseToMedium ( Gy ) : Sum",
                        "",
                    )
                ),
                encoding="ascii",
            )
            # Native TOPAS logical order is (Z, Y, X), X fastest.
            native = np.empty((1, 4, 3), dtype="<f8")
            for iz in range(1):
                for iy in range(4):
                    for ix in range(3):
                        native[iz, iy, ix] = 100.0 * iz + 10.0 * iy + ix
            binary = root / "dose.bin"
            native.tofile(binary)

            output_mhd = root / "mapped.mhd"
            metrics_json = root / "metrics.json"
            metrics = MODULE.convert(
                binary, header, output_mhd, metrics_json, expected_pitch_mm=None
            )

            mapped = np.fromfile(root / "mapped.raw", dtype="<f4").reshape(4, 1, 3)
            np.testing.assert_array_equal(mapped, native.transpose(1, 0, 2))
            self.assertEqual(metrics["output_shape_gpu_zyx"], [4, 1, 3])
            self.assertEqual(metrics["spacing_gpu_xyz_mm"], [0.1, 1.0, 0.2])
            np.testing.assert_allclose(
                metrics["offset_first_voxel_center_gpu_xyz_mm"],
                [-0.1, 0.0, 0.1],
            )
            text = output_mhd.read_text(encoding="ascii")
            self.assertIn("DimSize = 3 1 4", text)
            self.assertIn("ElementSpacing = 0.1 1 0.2", text)

    def test_rejects_binary_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = root / "dose.binheader"
            header.write_text(
                "# X in 2 bins of 0.1 cm\n"
                "# Y in 2 bins of 0.1 cm\n"
                "# Z in 2 bins of 0.1 cm\n",
                encoding="ascii",
            )
            binary = root / "dose.bin"
            binary.write_bytes(b"not-a-dose")
            geometry = MODULE.parse_binheader(header)
            with self.assertRaisesRegex(ValueError, "does not match"):
                MODULE.infer_dtype(binary, geometry)


if __name__ == "__main__":
    unittest.main()
