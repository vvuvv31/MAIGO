#!/usr/bin/env python3
"""End-to-end tests for TPS-90 CT axis reorientation."""

from __future__ import annotations

import array
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HEADER = struct.Struct("<IIIIIffffff")
MAGIC = 0x47544343
SCRIPT = Path(__file__).with_name("reorient_ct_grid_tps_90.py")


class ReorientCtGridTest(unittest.TestCase):
    def _write_grid(self, path: Path) -> tuple[list[float], bytes, bytes]:
        nx, ny, nz = 3, 2, 2
        density = [float(i) for i in range(nx * ny * nz)]
        material = bytes(range(nx * ny * nz))
        tail = struct.pack("<I", 2) + struct.pack("<ffff", 0.9, 1.1, 70.0, 80.0)
        header = HEADER.pack(
            MAGIC, 3, nx, ny, nz, -1.5, -1.0, -2.0, 1.0, 1.0, 2.0
        )
        values = array.array("f", density)
        path.write_bytes(header + values.tobytes() + material + tail)
        return density, material, tail

    def _run(self, direction: str) -> tuple[tuple, list[float], bytes, bytes, dict]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.bin"
            output = root / "output.bin"
            metadata = root / "output.json"
            density, material, tail = self._write_grid(source)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--metadata",
                    str(metadata),
                    "--beam-patient-x-direction",
                    direction,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            payload = output.read_bytes()
            header = HEADER.unpack_from(payload)
            count = header[2] * header[3] * header[4]
            values = array.array("f")
            values.frombytes(payload[HEADER.size : HEADER.size + 4 * count])
            materials = payload[HEADER.size + 4 * count : HEADER.size + 5 * count]
            output_tail = payload[HEADER.size + 5 * count :]
            meta = json.loads(metadata.read_text(encoding="utf-8"))
        return header, list(values), materials, output_tail, meta

    def test_positive_and_negative_patient_x_order(self) -> None:
        positive = self._run("positive")
        negative = self._run("negative")

        self.assertEqual(positive[0][2:5], (2, 2, 3))
        self.assertEqual(negative[0][2:5], (2, 2, 3))
        plane = 2 * 2
        self.assertEqual(
            negative[1],
            positive[1][2 * plane : 3 * plane]
            + positive[1][plane : 2 * plane]
            + positive[1][0:plane],
        )
        self.assertEqual(
            negative[2],
            positive[2][2 * plane : 3 * plane]
            + positive[2][plane : 2 * plane]
            + positive[2][0:plane],
        )
        self.assertEqual(positive[3], negative[3])
        self.assertEqual(negative[4]["beam_patient_x_direction"], "negative")
        self.assertEqual(negative[4]["axis_mapping"], "gpu(x,y,z)=patient(y,z,-x)")


if __name__ == "__main__":
    unittest.main()
