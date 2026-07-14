#!/usr/bin/env python3
"""Compare TOPAS and GPU species-resolved depth-dose CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SPECIES_COLUMNS = (
    ("total", "total_MeV_per_primary"),
    ("primary_c12", "primary_c12_MeV_per_primary"),
    ("secondary_carbon", "secondary_carbon_MeV_per_primary"),
    ("boron", "boron_MeV_per_primary"),
    ("beryllium", "beryllium_MeV_per_primary"),
    ("lithium", "lithium_MeV_per_primary"),
    ("helium", "helium_MeV_per_primary"),
    ("proton", "proton_MeV_per_primary"),
    ("other", "other_MeV_per_primary"),
)


def load(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = data.dtype.names or ()
    required = {"depth_mm", *(column for _, column in SPECIES_COLUMNS)}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    if len(data) < 3 or np.any(np.diff(data["depth_mm"]) <= 0.0):
        raise ValueError(f"{path} depth grid must be strictly increasing")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    evaluation = load(args.evaluation)
    if len(reference) != len(evaluation) or not np.allclose(
        reference["depth_mm"], evaluation["depth_mm"], rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("Reference and evaluation depth grids do not match")

    metrics: dict[str, object] = {
        "reference": args.reference.as_posix(),
        "evaluation": args.evaluation.as_posix(),
        "species": {},
    }
    for label, column in SPECIES_COLUMNS:
        reference_curve = np.asarray(reference[column])
        evaluation_curve = np.asarray(evaluation[column])
        reference_integral = float(np.trapz(reference_curve, reference["depth_mm"]))
        evaluation_integral = float(np.trapz(evaluation_curve, evaluation["depth_mm"]))
        metrics["species"][label] = {
            "topas_integral_MeV_mm_per_primary": reference_integral,
            "gpu_integral_MeV_mm_per_primary": evaluation_integral,
            "integral_relative_percent":
                (evaluation_integral / reference_integral - 1.0) * 100.0,
            "nrmse_normalized_to_topas_maximum": float(
                np.sqrt(np.mean((evaluation_curve - reference_curve) ** 2))
                / np.max(reference_curve)
            ),
        }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
