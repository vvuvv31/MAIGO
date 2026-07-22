#!/usr/bin/env python3
"""Jointly fit CT energy-layer fluence factors to the physical plan dose.

Unlike the independent per-layer calibration, this solves all available GPU
layer responses together in the full-plan high-dose region.  A diagonal ridge
toward the independently calibrated factors limits amplification of the 200k
per-layer Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fit_ct_energy_layer_response import map_gpu_numpy
from match_gpu_to_physical_dose import read_mhd
from prepare_ct_prelim_topas import load_combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=Path("ct/code/full_plan_weights_exact.csv"))
    parser.add_argument("--physical-dose", type=Path,
                        default=Path("ct/code/physical_dose.mhd"))
    parser.add_argument("--spots", nargs="+", type=Path,
                        default=[Path("ct/topas/spots_c_01.txt"),
                                 Path("ct/topas/spots_c_02.txt")])
    parser.add_argument("--layers-dir", type=Path,
                        default=Path("out/ct/tuning/energy_layers_200000"))
    parser.add_argument("--independent-calibration", type=Path,
                        default=Path("out/ct/tuning/dij_layer_calibration/calibration.json"))
    parser.add_argument("--gpu-histories-per-layer", type=float, default=200000.0)
    parser.add_argument("--dij-histories-per-spot", type=float, default=50000.0)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--minimum-factor", type=float, default=0.8)
    parser.add_argument("--maximum-factor", type=float, default=1.2)
    parser.add_argument(
        "--global-factor", type=float, default=1.0,
        help=("Uniform MU-to-ions calibration applied after the layer fit; it "
              "does not change relative fluence or fixed-history transport"),
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("out/ct/tuning/dij_layer_joint_calibration"))
    args = parser.parse_args()

    channels = load_combined(args.spots)
    energies = np.asarray(channels[2], dtype=np.float64) / 12.0
    weights = np.loadtxt(args.weights, dtype=np.float64).reshape(-1)
    if weights.size != energies.size:
        raise SystemExit(f"weights {weights.size} != spots {energies.size}")

    independent = json.loads(args.independent_calibration.read_text())
    prior_by_energy = {
        float(layer["energy_MeVu"]): float(layer["response_factor"])
        for layer in independent["layers"]
    }
    _, physical_values = read_mhd(args.physical_dose)
    reference = np.asarray(physical_values, dtype=np.float64)
    selected = reference >= 0.10 * float(reference.max())

    layer_energies: list[float] = []
    columns: list[np.ndarray] = []
    priors: list[float] = []
    for energy in sorted(set(float(value) for value in energies)):
        path = args.layers_dir / f"{energy:g}MeVu/dose.mhd"
        if not path.exists():
            continue
        metadata, values = read_mhd(path)
        gpu_shape = tuple(int(value) for value in metadata["DimSize"].split())
        mapped = map_gpu_numpy(values, gpu_shape).astype(np.float64).reshape(-1)
        layer_mask = np.isclose(energies, energy, rtol=0.0, atol=1.0e-6)
        theory_scale = (args.dij_histories_per_spot * float(weights[layer_mask].sum()) /
                        args.gpu_histories_per_layer)
        columns.append(mapped * theory_scale)
        layer_energies.append(energy)
        priors.append(prior_by_energy.get(energy, 1.0))

    matrix = np.column_stack([column[selected] for column in columns])
    target = reference[selected]
    prior = np.asarray(priors)
    normal = matrix.T @ matrix
    rhs = matrix.T @ target
    diagonal = np.maximum(np.diag(normal), np.finfo(np.float64).tiny)
    regularizer = args.ridge * np.diag(diagonal)
    factors = np.linalg.solve(normal + regularizer,
                              rhs + regularizer @ prior)
    factors = np.clip(factors, args.minimum_factor, args.maximum_factor)

    prediction = sum(factor * column for factor, column in zip(factors, columns))
    scale = float(np.dot(prediction[selected], target) /
                  np.dot(prediction[selected], prediction[selected]))
    nrmse = float(np.sqrt(np.mean((scale * prediction[selected] - target) ** 2)) /
                  reference.max())
    prior_prediction = sum(factor * column for factor, column in zip(prior, columns))
    prior_scale = float(np.dot(prior_prediction[selected], target) /
                        np.dot(prior_prediction[selected], prior_prediction[selected]))
    prior_nrmse = float(
        np.sqrt(np.mean((prior_scale * prior_prediction[selected] - target) ** 2)) /
        reference.max()
    )

    calibrated = weights.copy()
    factor_by_energy = dict(zip(layer_energies, factors))
    for index, energy in enumerate(energies):
        calibrated[index] *= factor_by_energy.get(float(energy),
                                                   prior_by_energy.get(float(energy), 1.0))
    calibrated *= args.global_factor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "full_plan_weights_dij_layer_joint.csv"
    weights_path.write_text("".join(f"{value:.17g}\n" for value in calibrated),
                            encoding="ascii")
    report = {
        "method": "joint high-dose least squares with diagonal ridge to independent factors",
        "ridge": args.ridge,
        "factor_bounds": [args.minimum_factor, args.maximum_factor],
        "global_factor": args.global_factor,
        "predicted_prior_nrmse": prior_nrmse,
        "predicted_joint_nrmse": nrmse,
        "prediction_scale": scale,
        "weight_sum": float(calibrated.sum()),
        "weights_file": str(weights_path),
        "layers": [
            {"energy_MeVu": energy, "prior_factor": prior_factor,
             "joint_factor": float(factor)}
            for energy, prior_factor, factor in zip(layer_energies, prior, factors)
        ],
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
