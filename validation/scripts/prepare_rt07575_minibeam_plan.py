#!/usr/bin/env python3
"""Extract the RT07575 minibeam C-12 plan reference and spot weights."""

from __future__ import annotations

import argparse
import array
import json
import re
import struct
from pathlib import Path

import h5py
import numpy as np

from match_gpu_to_physical_dose import write_mhd


COUNT_RE = re.compile(r"^iv:Tf/Scatterer1/L4/Values\s*=\s*(\d+)\s+", re.MULTILINE)
CCTG_HEADER = struct.Struct("<IIIIIffffff")
CCTG_MAGIC = 0x47544343


def spot_count(path: Path) -> int:
    match = COUNT_RE.search(path.read_text(encoding="utf-8-sig"))
    if match is None:
        raise ValueError(f"{path}: missing L4 history-count vector")
    return int(match.group(1))


def validate_centered_ct_low_edge(path: Path) -> dict[str, object]:
    """Reject the stale RT07575 grid that stored voxel centers as low edges."""
    payload = path.read_bytes()
    if len(payload) < CCTG_HEADER.size:
        raise ValueError(f"truncated CCTG header: {path}")
    magic, version, nx, ny, nz, ox, oy, oz, sx, sy, sz = CCTG_HEADER.unpack_from(
        payload
    )
    if magic != CCTG_MAGIC or version not in {1, 2, 3}:
        raise ValueError(f"unsupported CCTG magic/version: {path}")
    expected = (-0.5 * nx * sx, -0.5 * ny * sy, -0.5 * sz)
    actual = (ox, oy, oz)
    errors = tuple(actual[i] - expected[i] for i in range(3))
    if any(abs(value) > 1.0e-5 for value in errors):
        raise ValueError(
            f"{path}: stale/non-centered CCTG low edge {actual}; expected "
            f"{expected}. RT07575's old grid stored first-voxel centers as edges, "
            "causing a half-voxel minibeam phase error. Regenerate it with "
            "validation/scripts/prepare_ct_grid.py (or pass "
            "--skip-ct-origin-check only for an intentionally non-centered grid)."
        )
    return {
        "path": str(path),
        "shape_xyz": [nx, ny, nz],
        "spacing_xyz_mm": [sx, sy, sz],
        "low_edge_xyz_mm": [ox, oy, oz],
        "expected_centered_low_edge_xyz_mm": list(expected),
        "low_edge_error_xyz_mm": list(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mat",
        type=Path,
        default=Path("ct/RT07575/minibeam/code_v2/RBE_dose_result_c.mat"),
    )
    parser.add_argument(
        "--spots-dir",
        type=Path,
        default=Path("ct/RT07575/minibeam/scripts_c"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/ct/RT07575/minibeam_plan"),
    )
    parser.add_argument("--dij-histories-per-spot", type=int, default=100_000)
    parser.add_argument("--history-divisor", type=float, default=100.0)
    parser.add_argument(
        "--patient-ct-grid",
        type=Path,
        default=Path("ct/grid/patient_ct_rt07575_edge_corrected.bin"),
        help="Centered patient CCTG checked for the voxel-center/low-edge convention",
    )
    parser.add_argument(
        "--skip-ct-origin-check",
        action="store_true",
        help="Allow an intentionally non-centered patient CCTG",
    )
    args = parser.parse_args()

    ct_origin_check = None
    if not args.skip_ct_origin_check:
        ct_origin_check = validate_centered_ct_low_edge(args.patient_ct_grid)

    spot_paths = [args.spots_dir / f"spots_c_0{i}.txt" for i in range(1, 5)]
    counts = [spot_count(path) for path in spot_paths]

    with h5py.File(args.mat, "r") as handle:
        dose_h5 = np.asarray(handle["resultGUI/physicalDose"], dtype=np.float64)
        weights = np.asarray(handle["resultGUI/w"], dtype=np.float64).reshape(-1)
    if dose_h5.shape != (35, 417, 505):
        raise ValueError(f"unexpected HDF5 physicalDose shape: {dose_h5.shape}")
    if sum(counts) != weights.size:
        raise ValueError(f"spot count {sum(counts)} != weight count {weights.size}")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "weights_resultGUI_w.csv"
    np.savetxt(weights_path, weights, fmt="%.9g")

    # MATLAB logical order is (X,Y,Z); h5py presents the reversed axes.  MHD
    # raw storage is written in numpy (Z,Y,X) order.
    dose_zyx = np.transpose(dose_h5, (0, 2, 1)).astype(np.float32)
    reference_path = args.output_dir / "reference_resultGUI_physicalDose.mhd"
    write_mhd(
        reference_path,
        array.array("f", dose_zyx.ravel()),
        (417, 505, 35),
        (0.5, 0.5, 2.0),
        (-103.5, -44.3, -814.19),
        "Gy",
        extra_lines=[
            "Comment = resultGUI.physicalDose from RBE_dose_result_c.mat",
            "ReferenceKind = TOPAS sparse-Dij physical dose times optimized w",
        ],
    )

    nominal_histories = float(weights.sum()) * args.dij_histories_per_spot
    requested_histories = round(nominal_histories / args.history_divisor)
    split = np.cumsum([0, *counts])
    group_weights = [float(weights[split[i] : split[i + 1]].sum()) for i in range(4)]
    for i in range(4):
        np.savetxt(
            args.output_dir / f"weights_c_0{i + 1}.csv",
            weights[split[i] : split[i + 1]],
            fmt="%.9g",
        )
    np.savetxt(args.output_dir / "weights_angle01.csv", weights[: split[2]], fmt="%.9g")
    np.savetxt(args.output_dir / "weights_angle02.csv", weights[split[2] :], fmt="%.9g")
    exact_group_histories = np.asarray(group_weights) * (
        args.dij_histories_per_spot / args.history_divisor
    )
    group_histories = np.floor(exact_group_histories).astype(np.int64)
    remaining = requested_histories - int(group_histories.sum())
    order = np.argsort(-(exact_group_histories - group_histories), kind="stable")
    group_histories[order[:remaining]] += 1
    report = {
        "mat": str(args.mat),
        "spot_files": [str(path) for path in spot_paths],
        "spot_counts": counts,
        "weight_count": int(weights.size),
        "positive_weight_count": int(np.count_nonzero(weights > 0.0)),
        "weight_sum": float(weights.sum()),
        "weight_sum_by_file": group_weights,
        "dij_histories_per_spot": args.dij_histories_per_spot,
        "nominal_optimized_histories": nominal_histories,
        "history_divisor": args.history_divisor,
        "requested_gpu_histories": requested_histories,
        "requested_gpu_histories_by_file": group_histories.tolist(),
        "requested_gpu_histories_by_angle": [
            int(group_histories[:2].sum()),
            int(group_histories[2:].sum()),
        ],
        "gpu_to_reference_dose_scale": args.history_divisor,
        "physical_dose_shape_xyz": [417, 505, 35],
        "physical_dose_max_Gy": float(dose_zyx.max()),
        "physical_dose_sum_Gy": float(dose_zyx.sum(dtype=np.float64)),
        "ct_origin_check": ct_origin_check,
    }
    report_path = args.output_dir / "plan_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {weights_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
