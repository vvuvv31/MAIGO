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


def curve_metrics(
    reference_curve: np.ndarray,
    evaluation_curve: np.ndarray,
    depth_mm: np.ndarray,
) -> dict[str, float]:
    reference_integral = float(np.trapz(reference_curve, depth_mm))
    evaluation_integral = float(np.trapz(evaluation_curve, depth_mm))
    reference_maximum = float(np.max(reference_curve))
    return {
        "topas_integral_MeV_mm_per_primary": reference_integral,
        "gpu_integral_MeV_mm_per_primary": evaluation_integral,
        "integral_relative_percent":
            (evaluation_integral / reference_integral - 1.0) * 100.0,
        "nrmse_normalized_to_topas_maximum": float(
            np.sqrt(np.mean((evaluation_curve - reference_curve) ** 2))
            / reference_maximum
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--tail-start-mm", type=float, default=90.0)
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
        metrics["species"][label] = curve_metrics(
            reference_curve, evaluation_curve, reference["depth_mm"]
        )

    reference_names = set(reference.dtype.names or ())
    if "electron_positron_MeV_per_primary" in reference_names:
        electron_curve = np.asarray(reference["electron_positron_MeV_per_primary"])
        topas_other_curve = np.asarray(reference["other_MeV_per_primary"])
        gpu_other_curve = np.asarray(evaluation["other_MeV_per_primary"])
        topas_non_electron_other = np.maximum(topas_other_curve - electron_curve, 0.0)
        depth_mm = np.asarray(reference["depth_mm"])
        tail_mask = depth_mm >= args.tail_start_mm
        if np.count_nonzero(tail_mask) < 2:
            raise ValueError("Tail interval must contain at least two depth bins")

        other_integral = float(np.trapz(topas_other_curve, depth_mm))
        electron_integral = float(np.trapz(electron_curve, depth_mm))
        attribution_adjusted: dict[str, object] = {
            "semantics": (
                "TOPAS other excluding direct electron/positron tracks is compared "
                "with GPU other. The condensed-history GPU assigns electronic "
                "stopping locally to the transported ion instead of a separate "
                "electron category."
            ),
            "topas_electron_positron_fraction_of_other_percent":
                electron_integral / other_integral * 100.0,
            "other_excluding_electron_positron": curve_metrics(
                topas_non_electron_other, gpu_other_curve, depth_mm
            ),
            "tail_start_mm": args.tail_start_mm,
            "tail_other_excluding_electron_positron": curve_metrics(
                topas_non_electron_other[tail_mask],
                gpu_other_curve[tail_mask],
                depth_mm[tail_mask],
            ),
        }
        if {
            "gamma_MeV_per_primary",
            "neutron_MeV_per_primary",
        }.issubset(reference_names):
            neutral_direct = (
                np.asarray(reference["gamma_MeV_per_primary"])
                + np.asarray(reference["neutron_MeV_per_primary"])
            )
            attribution_adjusted["topas_gamma_neutron_direct_integral_MeV_mm_per_primary"] = (
                float(np.trapz(neutral_direct, depth_mm))
            )
        metrics["attribution_adjusted"] = attribution_adjusted

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
