#!/usr/bin/env python3
"""Rebin TOPAS Patient DoseToMedium CSV (417×505×35) → compare grid (104×126×35).

Block-averages with factors (4,4,1) matching match_gpu_to_physical_dose defaults.
"""

from __future__ import annotations

import argparse
import array
import csv
from pathlib import Path


def load_csv(paths: list[Path], nx: int, ny: int, nz: int) -> array.array:
    data = array.array("f", [0.0] * (nx * ny * nz))
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = [p.strip() for p in s.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    ix, iy, iz = (
                        int(float(parts[0])),
                        int(float(parts[1])),
                        int(float(parts[2])),
                    )
                    val = float(parts[3])
                except ValueError:
                    continue
                if val == 0.0 or not (
                    0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz
                ):
                    continue
                data[iz * nx * ny + iy * nx + ix] += val
    return data


def rebin(
    src: array.array,
    src_shape: tuple[int, int, int],
    dst_shape: tuple[int, int, int],
) -> list[float]:
    snx, sny, snz = src_shape
    dnx, dny, dnz = dst_shape
    bx = max(1, snx // dnx)
    by = max(1, sny // dny)
    bz = max(1, snz // dnz)
    acc = [0.0] * (dnx * dny * dnz)
    counts = [0] * (dnx * dny * dnz)
    for iz in range(snz):
        oz = iz // bz
        if oz >= dnz:
            continue
        for iy in range(sny):
            oy = iy // by
            if oy >= dny:
                continue
            for ix in range(snx):
                ox = ix // bx
                if ox >= dnx:
                    continue
                lin = oz * dnx * dny + oy * dnx + ox
                v = src[iz * snx * sny + iy * snx + ix]
                acc[lin] += v
                counts[lin] += 1
    for i, c in enumerate(counts):
        if c:
            acc[i] /= float(c)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more TOPAS CSVs; multiple sequential-run doses are summed",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--src-shape", nargs=3, type=int, default=(417, 505, 35))
    ap.add_argument("--dst-shape", nargs=3, type=int, default=(104, 126, 35))
    args = ap.parse_args()
    src = load_csv(args.input, *args.src_shape)
    if max(src) <= 0:
        raise SystemExit(f"Empty source dose: {args.input}")
    dst = rebin(src, tuple(args.src_shape), tuple(args.dst_shape))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dnx, dny, dnz = args.dst_shape
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# Rebinned TOPAS dose ({len(args.input)} input file(s)) "
            f"{args.src_shape} → {args.dst_shape} block-average\n"
        )
        handle.write("# ix, iy, iz, dose\n")
        for iz in range(dnz):
            for iy in range(dny):
                for ix in range(dnx):
                    v = dst[iz * dnx * dny + iy * dnx + ix]
                    if v == 0.0:
                        continue
                    handle.write(f"{ix}, {iy}, {iz}, {v:.8g}\n")
    print(f"wrote {args.output} nonzero={sum(1 for v in dst if v)} max={max(dst):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
