#!/usr/bin/env python3
"""Calibrate GPU energy-layer fluence against the original physical Dij.

For every energy layer this script reconstructs ``Dij[:, layer] @ x[layer]``
and compares it with an independently transported GPU layer dose.  The only
fitted degree of freedom is one scalar response per energy.  The resulting
weight file preserves all relative spot weights within each layer.

MATLAB v7.3 files require the optional ``h5py`` package.
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
    parser.add_argument("--dij", type=Path, default=Path("ct/dij_physical_sparse_c.mat"))
    parser.add_argument(
        "--optimizer-result",
        type=Path,
        default=Path("ct/code/physical_dose_result.mat"),
    )
    parser.add_argument(
        "--spots",
        nargs="+",
        type=Path,
        default=[Path("ct/topas/spots_c_01.txt"), Path("ct/topas/spots_c_02.txt")],
    )
    parser.add_argument(
        "--layers-dir",
        type=Path,
        default=Path("out/ct/tuning/energy_layers_200000"),
    )
    parser.add_argument("--gpu-histories-per-layer", type=float, default=200000.0)
    parser.add_argument("--dij-histories-per-spot", type=float, default=50000.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("out/ct/tuning/dij_layer_calibration")
    )
    args = parser.parse_args()

    try:
        import h5py
    except ImportError as error:
        raise SystemExit(
            "h5py is required to read MATLAB v7.3 files; install h5py in the "
            "active Python environment"
        ) from error

    channels = load_combined(args.spots)
    energies = np.asarray(channels[2], dtype=np.float64) / 12.0
    layers = sorted(set(float(value) for value in energies))

    with h5py.File(args.optimizer_result, "r") as optimizer:
        weights = np.asarray(optimizer["x"], dtype=np.float64).reshape(-1)
    if weights.size != energies.size:
        raise SystemExit(f"weights {weights.size} != spots {energies.size}")

    available_layers = [
        energy
        for energy in layers
        if (args.layers_dir / f"{energy:g}MeVu/dose.mhd").exists()
    ]
    if not available_layers:
        raise SystemExit(f"no layer dose files found below {args.layers_dir}")

    with h5py.File(args.dij, "r") as dij_file:
        dimensions = tuple(
            int(value) for value in np.asarray(dij_file["dij/dimensions"]).reshape(-1)
        )
        if dimensions != (126, 104, 35):
            raise SystemExit(f"unexpected Dij dimensions {dimensions}")
        column_ptr = np.asarray(dij_file["dij/physicalDose/jc"]).reshape(-1)
        row_index = dij_file["dij/physicalDose/ir"]
        data = dij_file["dij/physicalDose/data"]
        if column_ptr.size != weights.size + 1:
            raise SystemExit(
                f"Dij columns {column_ptr.size - 1} != weights {weights.size}"
            )

        references = {
            energy: np.zeros(int(np.prod(dimensions)), dtype=np.float64)
            for energy in available_layers
        }
        for column, (energy, weight) in enumerate(zip(energies, weights)):
            energy = float(energy)
            if weight == 0.0 or energy not in references:
                continue
            begin = int(column_ptr[column])
            end = int(column_ptr[column + 1])
            rows = np.asarray(row_index[begin:end], dtype=np.int64)
            references[energy][rows] += np.asarray(data[begin:end]) * weight

    calibrated = weights.copy()
    report: list[dict[str, float | int | str]] = []
    for energy in layers:
        layer_mask = np.isclose(energies, energy, rtol=0.0, atol=1.0e-6)
        layer_path = args.layers_dir / f"{energy:g}MeVu/dose.mhd"
        if energy not in references:
            factor = 1.0
            status = "no GPU layer dose; unchanged"
        else:
            metadata, values = read_mhd(layer_path)
            gpu_shape = tuple(int(value) for value in metadata["DimSize"].split())
            gpu = map_gpu_numpy(values, gpu_shape).astype(np.float64).reshape(-1)

            # MATLAB sparse rows are [Y, X, Z].  Convert to MHD [X, Y, Z]
            # with X fastest before comparing with the mapped GPU array.
            reference = (
                references[energy]
                .reshape(35, 104, 126)
                .transpose(0, 2, 1)
                .reshape(-1)
            )
            threshold = 0.10 * float(reference.max())
            selected = reference >= threshold
            denominator = float(np.dot(gpu[selected], gpu[selected]))
            if denominator <= 0.0:
                raise SystemExit(f"empty GPU high-dose region for {energy:g} MeV/u")
            fitted_scale = float(np.dot(gpu[selected], reference[selected]) / denominator)
            theory_scale = (
                args.dij_histories_per_spot
                * float(weights[layer_mask].sum())
                / args.gpu_histories_per_layer
            )
            factor = fitted_scale / theory_scale
            status = "calibrated"
        calibrated[layer_mask] *= factor
        report.append(
            {
                "energy_MeVu": energy,
                "spots": int(np.count_nonzero(layer_mask & (weights > 0.0))),
                "weight_sum_before": float(weights[layer_mask].sum()),
                "response_factor": factor,
                "weight_sum_after": float(calibrated[layer_mask].sum()),
                "status": status,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = args.output_dir / "full_plan_weights_dij_layer_calibrated.csv"
    weights_path.write_text(
        "".join(f"{value:.17g}\n" for value in calibrated), encoding="ascii"
    )
    summary = {
        "method": "one scalar GPU/Dij response calibration per energy layer",
        "dij_histories_per_spot": args.dij_histories_per_spot,
        "gpu_histories_per_layer": args.gpu_histories_per_layer,
        "spot_count": int(weights.size),
        "weight_sum_before": float(weights.sum()),
        "weight_sum_after": float(calibrated.sum()),
        "weights_file": str(weights_path),
        "layers": report,
    }
    summary_path = args.output_dir / "calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
