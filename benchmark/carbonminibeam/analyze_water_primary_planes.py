#!/usr/bin/env python3
"""Compare TOPAS primary-C12 plane crossings with GPU primary fluence.

The slit array width dominates an ordinary transverse RMS.  This diagnostic
therefore also folds x modulo the slit pitch and reports the periodic Fourier
modulation that directly controls peak/valley preservation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C12_PDG = 1000060120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-mhd", type=Path, required=True)
    parser.add_argument("--topas-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depths-mm", type=float, nargs="+", default=[40, 80, 100, 115, 124])
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    parser.add_argument("--central-half-width-mm", type=float, default=18.0)
    return parser.parse_args()


def parse_mhd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    dims = [int(value) for value in values["DimSize"].split()]
    spacing = [float(value) for value in values["ElementSpacing"].split()]
    offset = [float(value) for value in values["Offset"].split()]
    raw = path.parent / values["ElementDataFile"]
    data = np.fromfile(raw, dtype="<f4")
    expected = int(np.prod(dims))
    if data.size != expected or dims[1] != 1:
        raise ValueError(f"Unexpected GPU fluence dimensions {dims}: {data.size} values")
    dose = data.reshape((dims[2], dims[1], dims[0]))[:, 0, :]
    x_mm = offset[0] + np.arange(dims[0]) * spacing[0]
    depth_mm = offset[2] + np.arange(dims[2]) * spacing[2]
    return dose.astype(np.float64), x_mm, depth_mm


def modulation(x_mm: np.ndarray, weights: np.ndarray, pitch_mm: float) -> dict[str, float]:
    total = float(weights.sum())
    if total <= 0.0:
        return {"harmonic_1": 0.0, "harmonic_2": 0.0, "folded_rms_mm": 0.0}
    phase = 2.0 * np.pi * x_mm / pitch_mm
    folded = (x_mm + 0.5 * pitch_mm) % pitch_mm - 0.5 * pitch_mm
    return {
        "harmonic_1": float(np.abs(np.sum(weights * np.exp(1j * phase))) / total),
        "harmonic_2": float(np.abs(np.sum(weights * np.exp(2j * phase))) / total),
        "folded_rms_mm": float(np.sqrt(np.sum(weights * folded * folded) / total)),
    }


def main() -> None:
    args = parse_args()
    gpu, x_mm, gpu_depth_mm = parse_mhd(args.gpu_mhd)
    central = np.abs(x_mm) <= args.central_half_width_mm
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "pitch_mm": args.pitch_mm,
        "central_half_width_mm": args.central_half_width_mm,
        "depths": [],
    }
    fig, axes = plt.subplots(len(args.depths_mm), 1, figsize=(9, 2.5 * len(args.depths_mm)))
    axes = np.atleast_1d(axes)
    for axis, requested_depth in zip(axes, args.depths_mm):
        label = f"{int(round(requested_depth)):03d}"
        phase_path = args.topas_dir / "output" / f"primary_{label}.phsp"
        rows = np.atleast_2d(np.loadtxt(phase_path, dtype=np.float64))
        rows = rows[(rows[:, 7] == C12_PDG) & (rows[:, 13] == 0)]
        topas_x_mm = rows[:, 0] * 10.0
        topas_central = np.abs(topas_x_mm) <= args.central_half_width_mm
        counts, _ = np.histogram(
            topas_x_mm[topas_central],
            bins=np.append(x_mm - 0.05, x_mm[-1] + 0.05),
        )
        depth_index = int(np.argmin(np.abs(gpu_depth_mm - requested_depth)))
        gpu_profile = gpu[depth_index].copy()
        topas_profile = counts.astype(np.float64)
        gpu_norm = gpu_profile[central] / gpu_profile[central].sum()
        topas_norm = topas_profile[central] / topas_profile[central].sum()
        gpu_mod = modulation(x_mm[central], gpu_profile[central], args.pitch_mm)
        topas_mod = modulation(x_mm[central], topas_profile[central], args.pitch_mm)
        direction_x = rows[topas_central, 3]
        record = {
            "requested_depth_mm": requested_depth,
            "gpu_depth_mm": float(gpu_depth_mm[depth_index]),
            "topas_parent_c12": int(rows.shape[0]),
            "topas_mean_energy_MeV": float(rows[:, 5].mean()),
            "topas_direction_x_rms_mrad": float(np.sqrt(np.mean(direction_x * direction_x)) * 1000.0),
            "topas_abs_direction_x_q99_mrad": float(np.quantile(np.abs(direction_x), 0.99) * 1000.0),
            "profile_pearson": float(np.corrcoef(gpu_norm, topas_norm)[0, 1]),
            "profile_l1": float(np.abs(gpu_norm - topas_norm).sum()),
            "gpu": gpu_mod,
            "topas": topas_mod,
            "gpu_over_topas_harmonic_1": gpu_mod["harmonic_1"] / topas_mod["harmonic_1"],
            "gpu_over_topas_harmonic_2": gpu_mod["harmonic_2"] / topas_mod["harmonic_2"],
        }
        metrics["depths"].append(record)
        axis.plot(x_mm[central], topas_norm, label="TOPAS", lw=1.2)
        axis.plot(x_mm[central], gpu_norm, label="GPU", lw=1.0)
        axis.set_title(f"depth {requested_depth:g} mm")
        axis.set_ylabel("normalized primary fluence")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("x [mm]")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "primary_fluence_profiles.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "primary_plane_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
