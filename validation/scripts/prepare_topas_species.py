#!/usr/bin/env python3
"""Combine species-resolved TOPAS scorers into one energy-accounted IDD table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPECIES = (
    "primary_c12",
    "secondary_carbon",
    "boron",
    "beryllium",
    "lithium",
    "helium",
    "proton",
)

DETAIL_SCORERS = (
    "electron_positron",
    "gamma",
    "neutron",
    "deuteron",
    "triton",
    "alpha",
    "helium3",
)

OTHER_BREAKDOWN = (
    "electron_positron",
    "gamma",
    "neutron",
    "deuteron",
    "triton",
    "unclassified",
)

HELIUM_BREAKDOWN = (
    "alpha",
    "helium3",
    "helium_other",
)

DETAIL_COLUMNS = (*OTHER_BREAKDOWN, *HELIUM_BREAKDOWN)

DETAILED_CLOSURE_COMPONENTS = (
    "primary_c12",
    "secondary_carbon",
    "boron",
    "beryllium",
    "lithium",
    *HELIUM_BREAKDOWN,
    "proton",
    "deuteron",
    "triton",
    "electron_positron",
    "gamma",
    "neutron",
    "unclassified",
)

COLORS = {
    "primary_c12": "#1f77b4",
    "secondary_carbon": "#17becf",
    "boron": "#2ca02c",
    "beryllium": "#bcbd22",
    "lithium": "#ff7f0e",
    "helium": "#d62728",
    "proton": "#9467bd",
    "other": "#7f7f7f",
    "electron_positron": "#8c564b",
    "gamma": "#e377c2",
    "neutron": "#7f7f7f",
    "deuteron": "#aec7e8",
    "triton": "#98df8a",
    "alpha": "#ff9896",
    "helium3": "#c5b0d5",
    "helium_other": "#c49c94",
    "unclassified": "#c7c7c7",
}

DETAIL_LABELS = {
    "electron_positron": "Electron / positron",
    "gamma": "Gamma",
    "neutron": "Neutron",
    "deuteron": "Deuteron",
    "triton": "Triton",
    "alpha": "Alpha",
    "helium3": "He-3",
    "helium_other": "Other helium",
    "unclassified": "Unclassified residual",
}


def read_topas_values(path: Path, z_column: int, value_column: int) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    rows: list[list[float]] = []
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if stripped.startswith("# TOPAS Version:"):
                metadata["topas_version"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("# Parameter File:"):
                metadata["parameter_file"] = stripped.split(":", 1)[1].strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(value.strip()) for value in stripped.split(",")])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric scorer rows found in {path}")
    data = np.asarray(rows, dtype=float)
    required_column = max(z_column, value_column)
    if data.ndim != 2 or data.shape[1] <= required_column:
        raise ValueError(
            f"Requested column {required_column}, but {path} has numeric shape {data.shape}"
        )

    z_indices = data[:, z_column].astype(int)
    values = data[:, value_column]
    order = np.argsort(z_indices)
    return z_indices[order], values[order], metadata


def output_path(directory: Path, case_name: str, category: str) -> Path:
    return directory / f"species_{case_name}_{category}_energy_deposit.csv"


def write_combined_csv(
    path: Path,
    depth_mm: np.ndarray,
    total: np.ndarray,
    components: dict[str, np.ndarray],
    details: dict[str, np.ndarray],
) -> None:
    maximum = float(np.max(total))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "depth_mm",
                "total_MeV_per_primary",
                *(f"{name}_MeV_per_primary" for name in (*SPECIES, "other")),
                *(f"{name}_MeV_per_primary" for name in DETAIL_COLUMNS),
                "relative_total",
            ]
        )
        for index, depth in enumerate(depth_mm):
            writer.writerow(
                [
                    f"{depth:.12g}",
                    f"{total[index]:.12g}",
                    *(f"{components[name][index]:.12g}" for name in (*SPECIES, "other")),
                    *(f"{details[name][index]:.12g}" for name in DETAIL_COLUMNS),
                    f"{total[index] / maximum if maximum > 0.0 else 0.0:.12g}",
                ]
            )


def write_plot(
    path: Path,
    depth_mm: np.ndarray,
    total: np.ndarray,
    components: dict[str, np.ndarray],
    details: dict[str, np.ndarray],
    tail_start_mm: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = (*SPECIES, "other")
    labels = (
        "Primary C-12",
        "Secondary carbon",
        "Boron",
        "Beryllium",
        "Lithium",
        "Helium",
        "Proton",
        "Other",
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(depth_mm, total, color="black", linewidth=1.4, label="Total")
    axes[0].stackplot(
        depth_mm,
        *(components[name] for name in names),
        labels=labels,
        colors=[COLORS[name] for name in names],
        alpha=0.8,
    )
    axes[0].set_ylabel("Energy deposit (MeV / primary / 0.5 mm)")
    axes[0].set_title("TOPAS species-resolved carbon-ion IDD")
    axes[0].axvline(tail_start_mm, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=3, fontsize=8)

    positive_total = total[total > 0.0]
    floor = max(float(np.max(total)) * 1.0e-7, float(np.min(positive_total)) if positive_total.size else 1.0e-12)
    detail_plot_names = (
        "electron_positron",
        "gamma",
        "neutron",
        "deuteron",
        "triton",
        "alpha",
        "helium3",
        "unclassified",
    )
    for name in detail_plot_names:
        axes[1].semilogy(
            depth_mm,
            np.maximum(details[name], floor),
            color=COLORS[name],
            linewidth=1.0,
            label=DETAIL_LABELS[name],
        )
    axes[1].axvline(tail_start_mm, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[1].set_xlabel("Depth in water (mm)")
    axes[1].set_ylabel("Energy deposit (MeV / primary / 0.5 mm)")
    axes[1].set_title("TOPAS other-dose and helium-isotope detail")
    axes[1].set_ylim(bottom=floor)
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(ncol=4, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nonnegative_residual(
    parent: np.ndarray,
    children: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, float]:
    raw = parent - np.sum(children, axis=0)
    tolerance = max(float(np.max(parent)) * 2.0e-6, 1.0e-12)
    minimum = float(np.min(raw))
    if minimum < -tolerance:
        index = int(np.argmin(raw))
        raise SystemExit(
            f"{label} children exceed their parent at bin {index} by "
            f"{-raw[index]:.6g} MeV/primary"
        )
    return np.maximum(raw, 0.0), minimum


def parse_log(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"path": path.as_posix(), "sha256": sha256(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    geant4_match = re.search(r"Geant4 version Name:\s+(\S+)", text)
    elapsed_match = re.search(
        r"^\s*Total:\s+User=[^\n]*?Real=([0-9.]+)s", text, flags=re.MULTILINE
    )
    if geant4_match:
        result["geant4_version"] = geant4_match.group(1)
    if elapsed_match:
        result["elapsed_real_s"] = float(elapsed_match.group(1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("validation/topas/output"))
    parser.add_argument("--case", choices=("smoke", "development", "reference"), required=True)
    parser.add_argument("--histories", type=float, required=True)
    parser.add_argument("--bin-width-mm", type=float, default=0.5)
    parser.add_argument("--z-column", type=int, default=2)
    parser.add_argument("--value-column", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--tail-start-mm", type=float, default=90.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    if args.histories <= 0:
        raise SystemExit("--histories must be positive")

    total_path = output_path(args.input_dir, args.case, "total")
    z_indices, total_sum, header_metadata = read_topas_values(
        total_path, args.z_column, args.value_column
    )
    total = total_sum / args.histories
    components: dict[str, np.ndarray] = {}
    input_files: dict[str, dict[str, str]] = {
        "total": {"path": total_path.as_posix(), "sha256": sha256(total_path)}
    }

    for name in SPECIES:
        path = output_path(args.input_dir, args.case, name)
        species_z, species_sum, species_metadata = read_topas_values(
            path, args.z_column, args.value_column
        )
        if not np.array_equal(species_z, z_indices):
            raise SystemExit(f"Depth-bin mismatch between {total_path} and {path}")
        if species_metadata.get("topas_version") != header_metadata.get("topas_version"):
            raise SystemExit(f"TOPAS version mismatch between {total_path} and {path}")
        components[name] = species_sum / args.histories
        input_files[name] = {"path": path.as_posix(), "sha256": sha256(path)}

    details: dict[str, np.ndarray] = {}
    for name in DETAIL_SCORERS:
        path = output_path(args.input_dir, args.case, name)
        detail_z, detail_sum, detail_metadata = read_topas_values(
            path, args.z_column, args.value_column
        )
        if not np.array_equal(detail_z, z_indices):
            raise SystemExit(f"Depth-bin mismatch between {total_path} and {path}")
        if detail_metadata.get("topas_version") != header_metadata.get("topas_version"):
            raise SystemExit(f"TOPAS version mismatch between {total_path} and {path}")
        details[name] = detail_sum / args.histories
        input_files[name] = {"path": path.as_posix(), "sha256": sha256(path)}

    classified = np.sum([components[name] for name in SPECIES], axis=0)
    raw_other = total - classified
    roundoff_tolerance = max(float(np.max(total)) * 2.0e-6, 1.0e-12)
    minimum_other = float(np.min(raw_other))
    if minimum_other < -roundoff_tolerance:
        index = int(np.argmin(raw_other))
        raise SystemExit(
            "Species scorers exceed total energy deposit at "
            f"z-bin {z_indices[index]} by {-raw_other[index]:.6g} MeV/primary"
        )
    components["other"] = np.maximum(raw_other, 0.0)

    details["helium_other"], minimum_raw_helium_other = nonnegative_residual(
        components["helium"],
        [details["alpha"], details["helium3"]],
        "Helium partition",
    )
    details["unclassified"], minimum_raw_unclassified = nonnegative_residual(
        components["other"],
        [details[name] for name in OTHER_BREAKDOWN if name != "unclassified"],
        "Other partition",
    )

    reconstructed = np.sum([components[name] for name in (*SPECIES, "other")], axis=0)
    closure = reconstructed - total
    detailed_arrays = {
        **components,
        **details,
    }
    detailed_reconstructed = np.sum(
        [detailed_arrays[name] for name in DETAILED_CLOSURE_COMPONENTS], axis=0
    )
    detailed_closure = detailed_reconstructed - total
    detailed_closure_max_abs = float(np.max(np.abs(detailed_closure)))
    detailed_closure_tolerance = max(float(np.max(total)) * 4.0e-6, 1.0e-12)
    if detailed_closure_max_abs > detailed_closure_tolerance:
        raise SystemExit(
            "Detailed particle partition does not close: maximum error "
            f"{detailed_closure_max_abs:.6g} exceeds tolerance "
            f"{detailed_closure_tolerance:.6g} MeV/primary/bin"
        )
    depth_mm = (z_indices + 0.5) * args.bin_width_mm
    write_combined_csv(args.output_csv, depth_mm, total, components, details)
    write_plot(args.plot, depth_mm, total, components, details, args.tail_start_mm)

    total_deposit = float(np.sum(total))
    component_deposits = {
        name: float(np.sum(components[name])) for name in (*SPECIES, "other")
    }
    fractions = {
        name: value / total_deposit if total_deposit > 0.0 else 0.0
        for name, value in component_deposits.items()
    }
    tail_mask = depth_mm >= args.tail_start_mm
    tail_total = float(np.sum(total[tail_mask]))
    tail_component_deposits = {
        name: float(np.sum(components[name][tail_mask]))
        for name in (*SPECIES, "other")
    }
    tail_fractions = {
        name: value / tail_total if tail_total > 0.0 else 0.0
        for name, value in tail_component_deposits.items()
    }
    detail_deposits = {
        name: float(np.sum(details[name])) for name in DETAIL_COLUMNS
    }
    other_deposit = component_deposits["other"]
    other_breakdown_fractions = {
        name: detail_deposits[name] / other_deposit if other_deposit > 0.0 else 0.0
        for name in OTHER_BREAKDOWN
    }
    helium_deposit = component_deposits["helium"]
    helium_breakdown_fractions = {
        name: detail_deposits[name] / helium_deposit if helium_deposit > 0.0 else 0.0
        for name in HELIUM_BREAKDOWN
    }
    tail_detail_deposits = {
        name: float(np.sum(details[name][tail_mask])) for name in DETAIL_COLUMNS
    }
    tail_other_deposit = tail_component_deposits["other"]
    tail_other_breakdown_fractions = {
        name: tail_detail_deposits[name] / tail_other_deposit
        if tail_other_deposit > 0.0
        else 0.0
        for name in OTHER_BREAKDOWN
    }
    peak_index = int(np.argmax(total))
    log_path = args.log or args.input_dir / f"species-{args.case}_topas.log"
    if not log_path.exists():
        raise SystemExit(f"TOPAS log not found: {log_path}")
    metadata = {
        "case": args.case,
        "histories": int(args.histories),
        "seed": args.seed,
        "topas_version": header_metadata.get("topas_version", "unknown"),
        "parameter_file": header_metadata.get("parameter_file", "unknown"),
        "bin_width_mm": args.bin_width_mm,
        "bin_count": int(len(z_indices)),
        "scoring_semantics": "direct energy deposition by mutually exclusive particle-track categories",
        "other_semantics": "legacy total minus broad classified categories; partitioned within scorer roundoff into electron_positron, gamma, neutron, deuteron, triton, and unclassified",
        "helium_semantics": "legacy Z=2 secondary category; partitioned within scorer roundoff into alpha, helium3, and helium_other",
        "neutral_scorer_semantics": "gamma and neutron columns contain direct track energy deposition only; energy transferred to charged descendants is scored on those descendant tracks",
        "detailed_closure_components": list(DETAILED_CLOSURE_COMPONENTS),
        "total_deposited_MeV_per_primary": total_deposit,
        "component_deposited_MeV_per_primary": component_deposits,
        "component_fraction_of_deposited_energy": fractions,
        "classified_fraction_before_other": sum(fractions[name] for name in SPECIES),
        "peak_depth_mm": float(depth_mm[peak_index]),
        "peak_MeV_per_primary_per_bin": float(total[peak_index]),
        "tail_start_mm": args.tail_start_mm,
        "tail_deposited_MeV_per_primary": tail_total,
        "tail_component_deposited_MeV_per_primary": tail_component_deposits,
        "tail_component_fraction": tail_fractions,
        "detail_deposited_MeV_per_primary": detail_deposits,
        "other_breakdown_fraction": other_breakdown_fractions,
        "helium_breakdown_fraction": helium_breakdown_fractions,
        "tail_detail_deposited_MeV_per_primary": tail_detail_deposits,
        "tail_other_breakdown_fraction": tail_other_breakdown_fractions,
        "minimum_raw_other_MeV_per_primary_per_bin": minimum_other,
        "minimum_raw_unclassified_MeV_per_primary_per_bin": minimum_raw_unclassified,
        "minimum_raw_helium_other_MeV_per_primary_per_bin": minimum_raw_helium_other,
        "closure_max_abs_MeV_per_primary_per_bin": float(np.max(np.abs(closure))),
        "detailed_closure_max_abs_MeV_per_primary_per_bin": detailed_closure_max_abs,
        "detailed_closure_tolerance_MeV_per_primary_per_bin": detailed_closure_tolerance,
        "input_files": input_files,
        "topas_log": parse_log(log_path),
        "output_csv": args.output_csv.as_posix(),
        "plot": args.plot.as_posix(),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.metadata}")
    print(f"Wrote {args.plot}")
    print(f"Total deposited energy: {total_deposit:.6f} MeV/primary")
    print(f"Classified before other: {metadata['classified_fraction_before_other']:.3%}")
    print(
        "Other breakdown: "
        + ", ".join(
            f"{name} {other_breakdown_fractions[name]:.2%}" for name in OTHER_BREAKDOWN
        )
    )
    print(
        f"Tail from {args.tail_start_mm:g} mm: {tail_total:.6f} MeV/primary "
        f"(He {tail_fractions['helium']:.2%}, proton {tail_fractions['proton']:.2%})"
    )
    print(f"Maximum closure error: {metadata['closure_max_abs_MeV_per_primary_per_bin']:.3e} MeV/primary/bin")
    print(
        "Maximum detailed closure error: "
        f"{metadata['detailed_closure_max_abs_MeV_per_primary_per_bin']:.3e} "
        "MeV/primary/bin"
    )


if __name__ == "__main__":
    main()
