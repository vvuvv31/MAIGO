#!/usr/bin/env python3
"""Normalize remote multi-energy TOPAS total-IDD CSVs and write metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_topas_energy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for raw in stream:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                rows.append([float(v.strip()) for v in stripped.split(",")])
            except ValueError:
                continue
    if not rows:
        raise SystemExit(f"No numeric rows in {path}")
    data = np.asarray(rows, dtype=float)
    # x,y,z,sum,std
    z = data[:, 2].astype(int)
    values = data[:, 3]
    order = np.argsort(z)
    return z[order], values[order]


def write_normalized(
    z_indices: np.ndarray,
    values_per_primary: np.ndarray,
    output: Path,
    bin_width_mm: float,
) -> None:
    maximum = float(np.max(values_per_primary)) if values_per_primary.size else 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ["depth_mm", "energy_deposition_MeV_per_primary", "relative_dose"]
        )
        for z_index, value in zip(z_indices, values_per_primary, strict=True):
            writer.writerow(
                [
                    f"{(int(z_index) + 0.5) * bin_width_mm:.12g}",
                    f"{float(value):.12g}",
                    f"{(float(value) / maximum) if maximum > 0.0 else 0.0:.12g}",
                ]
            )


def curve_metrics(depth: np.ndarray, dose: np.ndarray) -> dict[str, float]:
    normalized = dose / np.max(dose)
    peak = int(np.argmax(dose))

    def distal(level: float) -> float:
        for index in range(peak + 1, len(depth)):
            if normalized[index] <= level < normalized[index - 1]:
                x0, x1 = depth[index - 1], depth[index]
                y0, y1 = normalized[index - 1], normalized[index]
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        return float("nan")

    def proximal(level: float) -> float:
        for index in range(peak, 0, -1):
            if normalized[index - 1] <= level < normalized[index]:
                x0, x1 = depth[index - 1], depth[index]
                y0, y1 = normalized[index - 1], normalized[index]
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        return float("nan")

    r80 = distal(0.8)
    r50 = distal(0.5)
    p50 = proximal(0.5)
    return {
        "integral_MeV_per_primary": float(np.sum(dose)),
        "peak_depth_mm": float(depth[peak]),
        "peak_value_MeV_per_primary": float(np.max(dose)),
        "R80_mm": r80,
        "FWHM_mm": r50 - p50,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-mevu", type=float, required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--parameter-file", type=str, required=True)
    parser.add_argument("--bin-width-mm", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=56)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    z_indices, sums = load_topas_energy_csv(args.raw_csv)
    if len(z_indices) != 800:
        raise SystemExit(f"Expected 800 depth bins, found {len(z_indices)} in {args.raw_csv}")
    values = sums / float(args.histories)
    write_normalized(z_indices, values, args.output_csv, args.bin_width_mm)

    depth = (z_indices.astype(float) + 0.5) * args.bin_width_mm
    metrics = curve_metrics(depth, values)
    metadata = {
        "status": "multi-energy TOPAS total IDD for GPU comparison",
        "particle": "GenericIon(6,12)",
        "energy_MeVu": args.energy_mevu,
        "total_kinetic_energy_MeV": args.energy_mevu * 12.0,
        "material": "TOPAS Water_75eV",
        "phantom_mm": [300.0, 300.0, 400.0],
        "depth_bin_width_mm": args.bin_width_mm,
        "histories": args.histories,
        "seed": args.seed,
        "threads": args.threads,
        "parameter_file": args.parameter_file,
        "raw_output": {
            "path": args.raw_csv.as_posix(),
            "sha256": sha256(args.raw_csv),
        },
        "normalized_output": {
            "path": args.output_csv.as_posix(),
            "sha256": sha256(args.output_csv),
        },
        "metrics": metrics,
        "global_scale_applied": False,
        "physics_modules": [
            "g4em-standard_opt4",
            "g4h-phy_QGSP_BIC_HP",
            "g4decay",
            "g4ion-binarycascade",
            "g4h-elastic_HP",
            "g4stopping",
        ],
    }
    if args.log is not None and args.log.exists():
        metadata["log"] = {"path": args.log.as_posix(), "sha256": sha256(args.log)}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"E={args.energy_mevu:g} MeV/u  R80={metrics['R80_mm']:.3f} mm  "
        f"integral={metrics['integral_MeV_per_primary']:.2f} MeV/primary  "
        f"-> {args.output_csv}"
    )


if __name__ == "__main__":
    main()
