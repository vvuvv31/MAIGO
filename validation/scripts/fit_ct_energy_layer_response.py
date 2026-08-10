#!/usr/bin/env python3
"""Diagnose energy-layer response using independently simulated weighted layers."""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path

import numpy as np

from match_gpu_to_physical_dose import gamma_3d, read_mhd, write_mhd


def map_gpu_numpy(values: array.array, shape: tuple[int, int, int]) -> np.ndarray:
    gx, gy, gz = shape
    if (gx, gy, gz) != (505, 35, 417):
        raise ValueError(f"unexpected GPU shape {shape}")
    gpu = np.asarray(values, dtype=np.float32).reshape(gz, gy, gx)
    patient = gpu.transpose(1, 2, 0)[:, :, ::-1]  # z, y, patient-x (xneg)
    cropped = patient[:, :504, :416]
    return cropped.reshape(35, 126, 4, 104, 4).mean(axis=(2, 4))


def gamma_rates(ref: np.ndarray, evaluated: np.ndarray) -> tuple[float, float]:
    ref_array = array.array("f", ref.reshape(-1).astype(np.float32))
    eval_list = evaluated.reshape(-1).astype(np.float32).tolist()
    common = ((104, 126, 35), (2.0, 2.0, 2.0), 3.0, 3.0, 10.0, 50000, 0)
    global_result = gamma_3d(ref_array, eval_list, *common, interpolation_step_mm=0.5)
    local_result = gamma_3d(
        ref_array, eval_list, *common, local_dose=True, interpolation_step_mm=0.5
    )
    return global_result["pass_percent"], local_result["pass_percent"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("ct/code/physical_dose.mhd"))
    parser.add_argument(
        "--layers-dir", type=Path, default=Path("out/ct/tuning/energy_layers_200000")
    )
    parser.add_argument(
        "--weights-summary",
        type=Path,
        default=Path("out/ct/tuning/energy_layer_weights/summary.json"),
    )
    parser.add_argument("--histories-per-layer", type=float, default=200000.0)
    parser.add_argument("--dij-histories-per-spot", type=float, default=50000.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/ct/tuning/energy_layer_fit")
    )
    args = parser.parse_args()

    ref_meta, ref_values = read_mhd(args.reference)
    ref = np.asarray(ref_values, dtype=np.float64).reshape(35, 126, 104)
    summaries = json.loads(args.weights_summary.read_text(encoding="utf-8"))
    summaries = [row for row in summaries if row["energy_MeVu"] >= 170.0]
    columns = []
    energies = []
    theory_coefficients = []
    for row in summaries:
        energy = float(row["energy_MeVu"])
        path = args.layers_dir / f"{energy:g}MeVu/dose.mhd"
        meta, values = read_mhd(path)
        shape = tuple(int(v) for v in meta["DimSize"].split())
        mapped = map_gpu_numpy(values, shape).astype(np.float64)
        theory = (
            args.dij_histories_per_spot
            * float(row["weight_sum"])
            / args.histories_per_layer
        )
        columns.append((mapped * theory).reshape(-1))
        energies.append(energy)
        theory_coefficients.append(theory)

    basis = np.column_stack(columns)
    target = ref.reshape(-1)
    selected = target >= 0.10 * float(target.max())
    train = selected & ((np.arange(target.size) % 2) == 0)
    test = selected & ~train

    baseline = basis.sum(axis=1)
    scale = float(np.dot(baseline[selected], target[selected]) / np.dot(baseline[selected], baseline[selected]))
    baseline *= scale
    results = []
    best = None
    norms = np.sqrt(np.sum(basis[train] ** 2, axis=0))
    norms = np.maximum(norms, 1.0e-30)
    for ridge in (0.0, 0.01, 0.1, 1.0, 10.0):
        matrix = basis[train]
        rhs = target[train]
        if ridge > 0.0:
            matrix = np.vstack((matrix, np.sqrt(ridge) * np.diag(norms)))
            rhs = np.concatenate((rhs, np.sqrt(ridge) * norms))
        factors = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        factors = np.maximum(factors, 0.0)
        evaluated = basis @ factors
        test_nrmse = float(
            np.sqrt(np.mean((evaluated[test] - target[test]) ** 2)) / target.max()
        )
        entry = {
            "ridge": ridge,
            "test_nrmse": test_nrmse,
            "factors": [float(value) for value in factors],
        }
        results.append(entry)
        if best is None or test_nrmse < best[0]:
            best = (test_nrmse, evaluated, entry)

    assert best is not None
    baseline_global, baseline_local = gamma_rates(ref, baseline.reshape(ref.shape))
    for entry in results:
        evaluated = basis @ np.asarray(entry["factors"], dtype=np.float64)
        global_rate, local_rate = gamma_rates(ref, evaluated.reshape(ref.shape))
        entry["gamma_global_3pct_3mm_thr10"] = global_rate
        entry["gamma_local_3pct_3mm_thr10"] = local_rate
    fitted = best[1].reshape(ref.shape)
    selected_result = next(row for row in results if row["ridge"] == best[2]["ridge"])
    fitted_global = selected_result["gamma_global_3pct_3mm_thr10"]
    fitted_local = selected_result["gamma_local_3pct_3mm_thr10"]
    report = {
        "energies_MeVu": energies,
        "theory_coefficients_applied_to_layer_runs": theory_coefficients,
        "baseline_global_scale": scale,
        "baseline_gamma_global_3pct_3mm_thr10": baseline_global,
        "baseline_gamma_local_3pct_3mm_thr10": baseline_local,
        "ridge_candidates": results,
        "selected_ridge": best[2]["ridge"],
        "selected_layer_response_factors": best[2]["factors"],
        "fitted_test_nrmse": best[0],
        "fitted_gamma_global_3pct_3mm_thr10": fitted_global,
        "fitted_gamma_local_3pct_3mm_thr10": fitted_local,
        "warning": "Layer fit is diagnostic and uses physical_dose; it is not independent validation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_mhd(
        args.output_dir / "energy_layer_fit.mhd",
        array.array("f", fitted.reshape(-1).astype(np.float32)),
        (104, 126, 35),
        tuple(float(v) for v in ref_meta["ElementSpacing"].split()),
        tuple(float(v) for v in ref_meta["Offset"].split()),
        "Gy (diagnostic layer fit)",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
