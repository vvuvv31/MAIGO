#!/usr/bin/env python3
"""Compare a TOPAS secondary-ion Copper foil case with the current GPU model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ion_pdg(atomic_number: int, mass_number: int) -> int:
    return 1_000_000_000 + atomic_number * 10_000 + mass_number * 10


def load_table(
    path: Path,
    atomic_number: int,
    mass_number: int,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(
                line for line in stream if not line.startswith("#")
            )
            if int(row["atomic_number"]) == atomic_number
            and int(row["mass_number"]) == mass_number
        ]
    if not rows:
        raise ValueError(
            f"{path}: no Z/A={atomic_number}/{mass_number} rows"
        )
    energy = np.asarray(
        [float(row["energy_MeVu"]) for row in rows], dtype=np.float64
    )
    values = np.asarray(
        [float(row[value_column]) for row in rows], dtype=np.float64
    )
    if np.any(np.diff(energy) <= 0.0):
        raise ValueError(f"{path}: energy grid is not strictly increasing")
    return energy, values


def current_gpu_no_inelastic_prediction(
    atomic_number: int,
    mass_number: int,
    initial_energy_mevu: float,
    thickness_mm: float,
    stopping_power_path: Path,
    cross_section_path: Path,
    step_mm: float = 0.01,
) -> dict[str, float | bool]:
    sp_energy, stopping_power = load_table(
        stopping_power_path,
        atomic_number,
        mass_number,
        "stopping_power_MeV_per_mm",
    )
    xs_energy, macroscopic_xs = load_table(
        cross_section_path,
        atomic_number,
        mass_number,
        "macroscopic_inelastic_cross_section_per_mm",
    )
    energy_mev = mass_number * initial_energy_mevu
    optical_depth = 0.0
    distance_mm = 0.0
    while distance_mm < thickness_mm and energy_mev > 0.0:
        path_step = min(step_mm, thickness_mm - distance_mm)
        energy_mevu = energy_mev / mass_number
        mean_loss = float(
            np.interp(
                energy_mevu,
                sp_energy,
                stopping_power,
                left=stopping_power[0],
                right=stopping_power[-1],
            )
        ) * path_step
        mid_energy_mevu = max(
            0.0, (energy_mev - 0.5 * mean_loss) / mass_number
        )
        optical_depth += float(
            np.interp(
                mid_energy_mevu,
                xs_energy,
                macroscopic_xs,
                left=macroscopic_xs[0],
                right=macroscopic_xs[-1],
            )
        ) * path_step
        energy_mev -= mean_loss
        distance_mm += path_step
    stopped = energy_mev <= 0.0
    return {
        "stopped_electromagnetically": stopped,
        "exit_energy_MeV": max(0.0, energy_mev),
        "no_inelastic_survival_probability": (
            0.0 if stopped else math.exp(-optical_depth)
        ),
        "integrated_inelastic_optical_depth": optical_depth,
        "integration_step_mm": step_mm,
    }


def statistics(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0}
    standard_deviation = float(np.std(values))
    return {
        "count": int(values.size),
        "sum_MeV": float(np.sum(values)),
        "mean_MeV": float(np.mean(values)),
        "std_MeV": standard_deviation,
        "standard_error_MeV": standard_deviation / math.sqrt(values.size),
        "minimum_MeV": float(np.min(values)),
        "maximum_MeV": float(np.max(values)),
    }


def reaction_event_ids(path: Path) -> tuple[set[int], int]:
    events: set[int] = set()
    reaction_count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 23:
                raise ValueError(
                    f"{path}:{line_number}: expected 23 columns, "
                    f"found {len(fields)}"
                )
            if fields[0] == "reaction":
                events.add(int(fields[2]))
                reaction_count += 1
    return events, reaction_count


def charged_category(atomic_number: np.ndarray, mass_number: np.ndarray) -> np.ndarray:
    result = np.full(atomic_number.shape, "other_charged", dtype="U20")
    result[(atomic_number == 1) & (mass_number == 1)] = "proton"
    result[atomic_number == 2] = "helium"
    result[atomic_number == 3] = "lithium"
    result[atomic_number == 4] = "beryllium"
    result[atomic_number == 5] = "boron"
    result[atomic_number == 6] = "carbon"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-phsp", type=Path, required=True)
    parser.add_argument("--reaction-phsp", type=Path, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--atomic-number", type=int, required=True)
    parser.add_argument("--mass-number", type=int, required=True)
    parser.add_argument("--initial-energy-mevu", type=float, required=True)
    parser.add_argument("--thickness-mm", type=float, required=True)
    parser.add_argument(
        "--stopping-power",
        type=Path,
        default=Path(
            "data/ion_stopping_power_copper_geant4_11_3_2.csv"
        ),
    )
    parser.add_argument(
        "--cross-section",
        type=Path,
        default=Path(
            "data/ion_cross_sections_copper_geant4_11_3_2.csv"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.histories <= 0
        or args.atomic_number <= 0
        or args.mass_number <= 0
        or args.initial_energy_mevu <= 0.0
        or args.thickness_mm <= 0.0
    ):
        raise ValueError("histories, ion identity, energy, and thickness must be positive")

    phase = np.loadtxt(args.exit_phsp, dtype=np.float64, ndmin=2)
    if phase.shape[1] != 14:
        raise ValueError(
            f"{args.exit_phsp}: expected 14 columns, found {phase.shape[1]}"
        )
    pdg = phase[:, 7].astype(np.int64)
    event_id = phase[:, 11].astype(np.int64)
    track_id = phase[:, 12].astype(np.int64)
    parent_id = phase[:, 13].astype(np.int64)
    atomic_number = np.where(pdg >= 1_000_000_000, (pdg // 10_000) % 1000, 0)
    mass_number = np.where(pdg >= 1_000_000_000, (pdg // 10) % 1000, 0)
    primary = (
        (pdg == ion_pdg(args.atomic_number, args.mass_number))
        & (track_id == 1)
        & (parent_id == 0)
    )
    reacted_events, reaction_count = reaction_event_ids(args.reaction_phsp)
    event_reacted = np.fromiter(
        (int(value) in reacted_events for value in event_id),
        dtype=bool,
        count=event_id.size,
    )
    unreacted_primary = primary & ~event_reacted
    reacted_primary = primary & event_reacted
    charged = atomic_number > 0
    reacted_charged_secondary = charged & event_reacted & ~primary
    categories = charged_category(atomic_number, mass_number)

    prediction = current_gpu_no_inelastic_prediction(
        args.atomic_number,
        args.mass_number,
        args.initial_energy_mevu,
        args.thickness_mm,
        args.stopping_power,
        args.cross_section,
    )
    expected_unreacted = (
        args.histories
        * float(prediction["no_inelastic_survival_probability"])
    )
    category_results: dict[str, dict[str, float | int]] = {}
    for category in (
        "carbon",
        "boron",
        "beryllium",
        "lithium",
        "helium",
        "proton",
        "other_charged",
    ):
        mask = reacted_charged_secondary & (categories == category)
        category_results[category] = statistics(phase[mask, 5])

    result = {
        "case": {
            "histories": args.histories,
            "atomic_number": args.atomic_number,
            "mass_number": args.mass_number,
            "initial_energy_MeVu": args.initial_energy_mevu,
            "thickness_mm": args.thickness_mm,
        },
        "current_gpu_model": {
            **prediction,
            "expected_no_inelastic_exit_count": expected_unreacted,
        },
        "topas": {
            "events_with_primary_inelastic_reaction": len(reacted_events),
            "primary_inelastic_reaction_count": reaction_count,
            "unreacted_primary_exit": statistics(
                phase[unreacted_primary, 5]
            ),
            "reacted_primary_continuation_exit": statistics(
                phase[reacted_primary, 5]
            ),
            "all_primary_exit": statistics(phase[primary, 5]),
            "charged_tertiary_exit_from_reacted_events": statistics(
                phase[reacted_charged_secondary, 5]
            ),
            "charged_tertiary_by_category": category_results,
        },
    }
    topas_unreacted_count = int(
        result["topas"]["unreacted_primary_exit"]["count"]  # type: ignore[index]
    )
    result["comparison"] = {
        "topas_unreacted_over_gpu_expectation": (
            topas_unreacted_count / expected_unreacted
            if expected_unreacted > 0.0 else None
        ),
        "topas_unreacted_mean_minus_gpu_exit_energy_MeV": (
            float(
                result["topas"]["unreacted_primary_exit"].get(  # type: ignore[index]
                    "mean_MeV", math.nan
                )
            )
            - float(prediction["exit_energy_MeV"])
        ),
        "missing_charged_tertiary_energy_per_incident_history_MeV": (
            float(
                result["topas"][
                    "charged_tertiary_exit_from_reacted_events"
                ].get("sum_MeV", 0.0)  # type: ignore[index]
            )
            / args.histories
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    energy_sets = (
        ("Unreacted primary", phase[unreacted_primary, 5], "tab:blue"),
        ("Reacted primary continuation", phase[reacted_primary, 5], "tab:red"),
        (
            "Charged tertiary",
            phase[reacted_charged_secondary, 5],
            "tab:green",
        ),
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    maximum = max(
        args.mass_number * args.initial_energy_mevu,
        *(float(np.max(values)) for _, values, _ in energy_sets if values.size),
    )
    bins = np.linspace(0.0, maximum, 81)
    for label, values, color in energy_sets:
        if values.size:
            axis.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.2,
                label=f"{label} (n={values.size})",
                color=color,
            )
    axis.axvline(
        float(prediction["exit_energy_MeV"]),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Current GPU no-inelastic exit energy",
    )
    axis.set_xlabel("Kinetic energy at Copper exit (MeV)")
    axis.set_ylabel("Particles / bin")
    axis.set_title(
        f"Z/A={args.atomic_number}/{args.mass_number}, "
        f"{args.initial_energy_mevu:g} MeV/u, "
        f"{args.thickness_mm:g} mm Copper"
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(args.output_plot, dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    print(args.output_json)
    print(args.output_plot)


if __name__ == "__main__":
    main()
