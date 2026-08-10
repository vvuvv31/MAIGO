#!/usr/bin/env python3
"""Cross-validate CT single-spot transport against independent TOPAS Dij columns.

The split is fixed by energy: 180/220 MeV/u are calibration energies and
200/240 MeV/u are held out.  Four spots spanning the transverse-radius
distribution are selected per energy.  No optimized full-plan dose or plan
weight enters the fit; the calibration set may determine only one common
absolute GPU-to-Dij response factor.

MATLAB v7.3 input requires h5py.  In this repository it is available in the
optional ``.plot-deps`` environment, e.g. ``PYTHONPATH=.plot-deps``.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from fit_ct_energy_layer_response import map_gpu_numpy
from match_gpu_to_physical_dose import gamma_3d, read_mhd
from prepare_ct_prelim_topas import load_combined


def select_spots(energies: np.ndarray, x: np.ndarray, y: np.ndarray,
                 selected_energies: tuple[float, ...]) -> list[int]:
    chosen: list[int] = []
    quantiles = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
    for energy in selected_energies:
        indices = np.flatnonzero(np.isclose(energies, energy, rtol=0.0, atol=1.0e-6))
        if indices.size < len(quantiles):
            raise SystemExit(f"energy {energy:g} has only {indices.size} spots")
        radius = np.hypot(x[indices] - np.median(x[indices]),
                          y[indices] - np.median(y[indices]))
        order = indices[np.argsort(radius, kind="stable")]
        positions = [round(q * (order.size - 1)) for q in quantiles]
        chosen.extend(int(order[position]) for position in positions)
    return chosen


def sparse_column(jc: np.ndarray, ir, data, column: int,
                  voxel_count: int) -> np.ndarray:
    begin = int(jc[column])
    end = int(jc[column + 1])
    result = np.zeros(voxel_count, dtype=np.float64)
    rows = np.asarray(ir[begin:end], dtype=np.int64)
    result[rows] = np.asarray(data[begin:end], dtype=np.float64)
    # MATLAB Dij rows are [Y, X, Z]; MHD is [X, Y, Z] with X fastest.
    return result.reshape(35, 104, 126).transpose(0, 2, 1).reshape(-1)


def fit_terms(gpu: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    mask = reference >= 0.10 * float(reference.max())
    gg = float(np.dot(gpu[mask], gpu[mask]))
    gr = float(np.dot(gpu[mask], reference[mask]))
    rr = float(np.dot(reference[mask], reference[mask]))
    return gg, gr, rr


def idd(values: np.ndarray) -> np.ndarray:
    return values.reshape(35, 126, 104).sum(axis=(0, 1))


def evaluate(reference: np.ndarray, gpu: np.ndarray, scale: float) -> dict[str, float]:
    scaled = scale * gpu
    mask = reference >= 0.10 * float(reference.max())
    nrmse = float(np.sqrt(np.mean((scaled[mask] - reference[mask]) ** 2)) /
                  reference.max())
    rid = idd(reference)
    gid = idd(scaled)
    corr = float(np.dot(rid, gid) /
                 math.sqrt(float(np.dot(rid, rid) * np.dot(gid, gid))))
    gamma = gamma_3d(
        array.array("f", reference.astype(np.float32)), scaled.tolist(),
        (104, 126, 35), (2.0, 2.0, 2.0), 2.0, 2.0, 10.0,
        50000, 20260723, False, 0.5,
    )
    gamma_local = gamma_3d(
        array.array("f", reference.astype(np.float32)), scaled.tolist(),
        (104, 126, 35), (2.0, 2.0, 2.0), 2.0, 2.0, 10.0,
        50000, 20260723, True, 0.5,
    )
    return {
        "global_gamma_2pct_2mm_thr10_percent": gamma["pass_percent"],
        "local_gamma_2pct_2mm_thr10_percent": gamma_local["pass_percent"],
        "nrmse_high_dose": nrmse,
        "idd_correlation": corr,
        "idd_peak_shift_mm": 2.0 * (int(np.argmax(gid)) - int(np.argmax(rid))),
        "integral_relative_difference": float(scaled.sum() / reference.sum() - 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dij", type=Path, default=Path("ct/dij_physical_sparse_c.mat"))
    parser.add_argument("--spots", nargs="+", type=Path,
                        default=[Path("ct/topas/spots_c_01.txt"),
                                 Path("ct/topas/spots_c_02.txt")])
    parser.add_argument("--config", type=Path,
                        default=Path("config/beam_ct_physics_baseline_1M.yaml"))
    parser.add_argument("--carbon-binary", type=Path,
                        default=Path("build/oneapi-release/carbon_mc"))
    parser.add_argument("--histories", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/physics_calibration/spot_cross_validation"))
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError as error:
        raise SystemExit(
            "h5py is required; run with PYTHONPATH=.plot-deps or install h5py"
        ) from error

    channels = load_combined(args.spots)
    energies = np.asarray(channels[2], dtype=np.float64) / 12.0
    trans_x = np.asarray(channels[5], dtype=np.float64)
    trans_y = np.asarray(channels[6], dtype=np.float64)
    train_energies = (180.0, 220.0)
    validation_energies = (200.0, 240.0)
    chosen = select_spots(
        energies, trans_x, trans_y, train_energies + validation_energies
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    dose_dir = args.output_dir / "dose"
    log_dir = args.output_dir / "logs"
    weights_dir.mkdir(exist_ok=True)
    dose_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    manifest: list[dict[str, object]] = []
    for index in chosen:
        split = "calibration" if float(energies[index]) in train_energies else "validation"
        tag = f"spot_{index + 1:04d}_e{energies[index]:g}"
        weights_path = weights_dir / f"{tag}.csv"
        weights = np.zeros(energies.size, dtype=np.float64)
        weights[index] = 1.0
        weights_path.write_text("".join(f"{value:.1f}\n" for value in weights),
                                encoding="ascii")
        dose_path = dose_dir / f"{tag}.mhd"
        log_path = log_dir / f"{tag}.log"
        entry: dict[str, object] = {
            "spot_index_zero_based": index,
            "spot_id": int(round(float(channels[0][index]))),
            "energy_MeVu": float(energies[index]),
            "trans_x_mm": float(trans_x[index]),
            "trans_y_mm": float(trans_y[index]),
            "radial_position_mm": float(math.hypot(trans_x[index], trans_y[index])),
            "split": split,
            "weights_file": str(weights_path),
            "dose_mhd": str(dose_path),
        }
        manifest.append(entry)
        if not args.skip_run:
            command = [
                str(args.carbon_binary), "--config", str(args.config),
                "--device", args.device, "--histories", str(args.histories),
                "--spot-weights", str(weights_path),
                "--voxel-dose-mhd", str(dose_path),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    with h5py.File(args.dij, "r") as dij_file:
        dimensions = tuple(
            int(value) for value in np.asarray(dij_file["dij/dimensions"]).reshape(-1)
        )
        if dimensions != (126, 104, 35):
            raise SystemExit(f"unexpected Dij dimensions {dimensions}")
        jc = np.asarray(dij_file["dij/physicalDose/jc"]).reshape(-1)
        ir = dij_file["dij/physicalDose/ir"]
        data = dij_file["dij/physicalDose/data"]
        voxel_count = int(np.prod(dimensions))
        cached: list[tuple[dict[str, object], np.ndarray, np.ndarray, float, float, float]] = []
        train_gg = train_gr = 0.0
        for entry in manifest:
            index = int(entry["spot_index_zero_based"])
            reference = sparse_column(jc, ir, data, index, voxel_count)
            metadata, values = read_mhd(Path(str(entry["dose_mhd"])))
            gpu_shape = tuple(int(value) for value in metadata["DimSize"].split())
            gpu = map_gpu_numpy(values, gpu_shape).astype(np.float64).reshape(-1)
            gg, gr, rr = fit_terms(gpu, reference)
            best_scale = gr / gg
            entry["best_fit_scale_gpu_to_dij"] = best_scale
            entry["best_fit"] = evaluate(reference, gpu, best_scale)
            cached.append((entry, reference, gpu, gg, gr, rr))
            if entry["split"] == "calibration":
                train_gg += gg
                train_gr += gr

    calibration_scale = train_gr / train_gg
    for entry, reference, gpu, _gg, _gr, _rr in cached:
        entry["fixed_calibration_scale"] = evaluate(reference, gpu, calibration_scale)

    def aggregate(split: str, field: str) -> dict[str, float]:
        rows = [entry for entry in manifest if entry["split"] == split]
        keys = list(rows[0][field].keys())
        return {
            key: float(np.mean([float(entry[field][key]) for entry in rows]))
            for key in keys
        }

    report = {
        "method": "single common absolute response calibrated without full-plan dose",
        "dij_histories_per_spot": 50000,
        "gpu_histories_per_spot": args.histories,
        "calibration_energies_MeVu": list(train_energies),
        "held_out_validation_energies_MeVu": list(validation_energies),
        "calibration_scale_gpu_to_dij": calibration_scale,
        "calibration_mean_fixed_scale": aggregate("calibration", "fixed_calibration_scale"),
        "validation_mean_fixed_scale": aggregate("validation", "fixed_calibration_scale"),
        "spots": manifest,
    }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
