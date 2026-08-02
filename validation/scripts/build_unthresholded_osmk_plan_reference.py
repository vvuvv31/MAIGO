#!/usr/bin/env python3
"""Build an unthresholded weighted TOPAS plan dose from per-spot OSMK scorers.

This is intentionally separate from the sparse Dij builder.  A per-voxel,
per-spot sparsification threshold can discard billions of individually small
values that become material after optimized spot weights are summed.  Here
each raw OSMK_Dtotal scorer is multiplied by its resultGUI.w value before it
is accumulated, so no sparse-Dij threshold is involved.
"""

from __future__ import annotations

import argparse
import array
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np

from match_gpu_to_physical_dose import read_mhd, write_mhd


ID_RE = re.compile(
    r"^iv:Tf/Scatterer1/L0/Values\s*=\s*(\d+)\s+([^\r\n]+)", re.MULTILINE
)


def parse_global_spot_ids(path: Path) -> np.ndarray:
    match = ID_RE.search(path.read_text(encoding="utf-8-sig"))
    if match is None:
        raise ValueError(f"{path}: missing global spot-ID vector L0")
    declared = int(match.group(1))
    values = np.fromstring(match.group(2), sep=" ", dtype=np.float64)
    if values.size != declared:
        raise ValueError(
            f"{path}: declared {declared} spot IDs but parsed {values.size}"
        )
    ids = values.astype(np.int64)
    if np.any(values != ids) or np.any(ids <= 0):
        raise ValueError(f"{path}: spot IDs must be positive integers")
    return ids


def resolve_scorer(scorer_dir: Path, ion: str, chunk: int, run: int) -> Path:
    exact = scorer_dir / f"OSMK_Dtotal_{ion}_{chunk:02d}_Run_{run:04d}.bin"
    if exact.is_file():
        return exact
    matches = sorted(
        scorer_dir.glob(f"OSMK_Dtotal_*_{chunk:02d}_Run_{run:04d}.bin")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one Dtotal scorer for chunk={chunk:02d}, run={run:04d}; "
            f"found {len(matches)} in {scorer_dir}"
        )
    return matches[0]


def load_reference_grid(path: Path) -> tuple[tuple[int, int, int], tuple[float, ...], tuple[float, ...], np.ndarray]:
    metadata, values = read_mhd(path)
    shape_xyz = tuple(int(value) for value in metadata["DimSize"].split())
    spacing = tuple(float(value) for value in metadata["ElementSpacing"].split())
    offset = tuple(float(value) for value in metadata["Offset"].split())
    nx, ny, nz = shape_xyz
    volume = np.asarray(values, dtype=np.float64).reshape(nz, ny, nx)
    return shape_xyz, spacing, offset, volume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorer-dir",
        type=Path,
        default=Path("ct/RT07575/minibeam/scripts_c"),
    )
    parser.add_argument(
        "--spots-dir",
        type=Path,
        default=Path("ct/RT07575/minibeam/scripts_c"),
    )
    parser.add_argument(
        "--weights-mat",
        type=Path,
        default=Path("ct/RT07575/minibeam/code_v2/RBE_dose_result_c.mat"),
    )
    parser.add_argument(
        "--thresholded-reference",
        type=Path,
        default=Path(
            "out/ct/RT07575/minibeam_plan/reference_resultGUI_physicalDose.mhd"
        ),
        help="MHD made from thresholded Dij times resultGUI.w; supplies the grid",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "out/ct/RT07575/minibeam_plan/reference_unthresholded_osmk.mhd"
        ),
    )
    parser.add_argument("--ion", default="c")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    shape_xyz, spacing, offset, thresholded = load_reference_grid(
        args.thresholded_reference
    )
    nx, ny, nz = shape_xyz
    expected_values = nx * ny * nz
    expected_bytes = expected_values * np.dtype("<f8").itemsize

    with h5py.File(args.weights_mat, "r") as handle:
        weights = np.asarray(handle["resultGUI/w"], dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("resultGUI.w must be finite and nonnegative")

    work: list[tuple[int, int, int]] = []
    for chunk in range(1, 100):
        spots_path = args.spots_dir / f"spots_{args.ion}_{chunk:02d}.txt"
        if not spots_path.is_file():
            break
        for run, global_id in enumerate(parse_global_spot_ids(spots_path)):
            work.append((chunk, run, int(global_id) - 1))
    if not work:
        raise FileNotFoundError(f"no numbered spots_{args.ion}_XX.txt in {args.spots_dir}")
    global_ids = np.asarray([item[2] for item in work], dtype=np.int64)
    if work and (
        len(work) != weights.size
        or not np.array_equal(np.sort(global_ids), np.arange(weights.size))
    ):
        raise ValueError(
            f"spot files contain {len(work)} IDs but resultGUI.w has {weights.size}"
        )

    output = np.zeros((nz, ny, nx), dtype=np.float64)
    used = 0
    skipped_zero = 0
    start = time.perf_counter()
    for position, (chunk, run, column) in enumerate(work, start=1):
        weight = float(weights[column])
        if weight == 0.0:
            skipped_zero += 1
            continue
        scorer = resolve_scorer(args.scorer_dir, args.ion, chunk, run)
        if scorer.stat().st_size != expected_bytes:
            raise ValueError(
                f"{scorer}: {scorer.stat().st_size} bytes, expected {expected_bytes}"
            )
        values = np.fromfile(scorer, dtype="<f8", count=expected_values)
        # TOPAS writes X fastest.  C-order (Z,Y,X) therefore maps directly to
        # the patient MHD axes used by the existing thresholded reference.
        values *= weight
        output += values.reshape(nz, ny, nx)
        used += 1
        if args.progress_every > 0 and position % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"processed {position}/{len(work)} spots "
                f"({used} positive weight), {elapsed:.1f} s",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(
        args.output,
        array.array("f", output.astype(np.float32).ravel()),
        shape_xyz,
        spacing,
        offset,
        "Gy",
        extra_lines=[
            "Comment = unthresholded raw TOPAS OSMK_Dtotal times resultGUI.w",
            f"PositiveWeightSpots = {used}",
            f"ZeroWeightSpotsSkipped = {skipped_zero}",
        ],
    )

    difference = output - thresholded
    report = {
        "scorer_dir": str(args.scorer_dir),
        "spots_dir": str(args.spots_dir),
        "weights_mat": str(args.weights_mat),
        "thresholded_reference": str(args.thresholded_reference),
        "output": str(args.output),
        "spot_count": len(work),
        "positive_weight_spots": used,
        "zero_weight_spots_skipped": skipped_zero,
        "thresholded_integral_Gy_voxel": float(thresholded.sum()),
        "unthresholded_integral_Gy_voxel": float(output.sum()),
        "restored_integral_Gy_voxel": float(difference.sum()),
        "restored_integral_percent_of_thresholded": (
            float(100.0 * difference.sum() / thresholded.sum())
            if thresholded.sum() != 0.0
            else None
        ),
        "maximum_restored_voxel_Gy": float(difference.max()),
        "minimum_restored_voxel_Gy": float(difference.min()),
        "elapsed_seconds": time.perf_counter() - start,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
