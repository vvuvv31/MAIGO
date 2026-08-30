#!/usr/bin/env python3
"""Compare MAIGO/FRED and TOPAS 1M carbon dose using only 3D scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENERGIES = (100, 200, 300, 400)
NX, NY, NZ = 400, 400, 800
DX_MM, DY_MM, DZ_MM = 0.2, 0.2, 0.5


def distal_crossing(z: np.ndarray, y: np.ndarray, fraction: float) -> float:
    peak = int(np.argmax(y))
    target = fraction * float(y[peak])
    for i in range(peak, len(y) - 1):
        if y[i] >= target and y[i + 1] < target:
            if y[i + 1] == y[i]:
                return float(z[i])
            return float(z[i] + (target - y[i]) * (z[i + 1] - z[i]) /
                         (y[i + 1] - y[i]))
    return float("nan")


def lateral_rms(plane: np.ndarray) -> float:
    axis = (np.arange(NX, dtype=np.float64) - NX / 2 + 0.5) * DX_MM
    total = float(np.sum(plane, dtype=np.float64))
    if total <= 0:
        return float("nan")
    x2 = float(np.sum(plane * axis[None, :] ** 2, dtype=np.float64)) / total
    y2 = float(np.sum(plane * axis[:, None] ** 2, dtype=np.float64)) / total
    return float(np.sqrt(max(0.0, 0.5 * (x2 + y2))))


def analyze_energy(root: Path, topas_root: Path, energy: int) -> tuple[dict, np.ndarray, np.ndarray]:
    gpu_path = root / f"out/gpu_fred_paper_1M_e{energy}/voxel_dose.raw"
    topas_path = topas_root / f"e{energy}/topas_emittance_inelastic_e{energy}.bin"
    expected_gpu = NX * NY * NZ * np.dtype("<f4").itemsize
    expected_topas = NX * NY * NZ * np.dtype("<f8").itemsize
    if gpu_path.stat().st_size != expected_gpu:
        raise ValueError(f"wrong GPU size: {gpu_path}")
    if topas_path.stat().st_size != expected_topas:
        raise ValueError(f"wrong TOPAS size: {topas_path}")

    gpu = np.memmap(gpu_path, dtype="<f4", mode="r", shape=(NZ, NY, NX))
    topas = np.memmap(topas_path, dtype="<f8", mode="r", shape=(NZ, NY, NX))
    gpu_idd = np.sum(gpu, axis=(1, 2), dtype=np.float64)
    topas_idd = np.sum(topas, axis=(1, 2), dtype=np.float64)
    z = (np.arange(NZ, dtype=np.float64) + 0.5) * DZ_MM

    gp, tp = int(np.argmax(gpu_idd)), int(np.argmax(topas_idd))
    mask = topas_idd >= 0.01 * topas_idd[tp]
    normalized_rmse = float(np.sqrt(np.mean((gpu_idd[mask] - topas_idd[mask]) ** 2)) /
                            topas_idd[tp])
    integral_gpu = float(np.sum(gpu_idd) * DZ_MM)
    integral_topas = float(np.sum(topas_idd) * DZ_MM)

    sample_depths = [5.0, 0.5 * z[tp], max(0.25, z[tp] - 5.0), min(399.75, z[tp] + 5.0)]
    lateral = []
    for depth in sample_depths:
        iz = int(np.clip(round(depth / DZ_MM - 0.5), 0, NZ - 1))
        gt, tt = lateral_rms(gpu[iz]), lateral_rms(topas[iz])
        lateral.append({
            "depth_mm": float(z[iz]),
            "gpu_rms_mm": gt,
            "topas_rms_mm": tt,
            "difference_percent": 100.0 * (gt - tt) / tt,
        })

    result = {
        "energy_MeVu": energy,
        "gpu_file": str(gpu_path),
        "topas_file": str(topas_path),
        "gpu_peak_depth_mm": float(z[gp]),
        "topas_peak_depth_mm": float(z[tp]),
        "peak_depth_difference_mm": float(z[gp] - z[tp]),
        "gpu_peak_Gy": float(gpu_idd[gp]),
        "topas_peak_Gy": float(topas_idd[tp]),
        "peak_height_difference_percent": 100.0 * (gpu_idd[gp] - topas_idd[tp]) / topas_idd[tp],
        "gpu_distal_R80_mm": distal_crossing(z, gpu_idd, 0.8),
        "topas_distal_R80_mm": distal_crossing(z, topas_idd, 0.8),
        "gpu_distal_R50_mm": distal_crossing(z, gpu_idd, 0.5),
        "topas_distal_R50_mm": distal_crossing(z, topas_idd, 0.5),
        "integral_gpu_Gy_mm": integral_gpu,
        "integral_topas_Gy_mm": integral_topas,
        "integral_difference_percent": 100.0 * (integral_gpu - integral_topas) / integral_topas,
        "idd_normalized_rmse_percent": 100.0 * normalized_rmse,
        "lateral_rms": lateral,
    }
    result["distal_R80_difference_mm"] = result["gpu_distal_R80_mm"] - result["topas_distal_R80_mm"]
    result["distal_R50_difference_mm"] = result["gpu_distal_R50_mm"] - result["topas_distal_R50_mm"]
    return result, topas_idd, gpu_idd


def markdown(results: list[dict]) -> str:
    lines = [
        "# FRED-model GPU vs TOPAS 1M 3D dose comparison",
        "",
        "All IDDs are X/Y sums of the 400x400x800 3D DoseToMedium scorers; no 1D scorer or independent GPU depth tally is used.",
        "",
        "| E (MeV/u) | Peak shift (mm) | Peak diff (%) | R80 diff (mm) | R50 diff (mm) | Integral diff (%) | IDD NRMSE (%) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['energy_MeVu']} | {r['peak_depth_difference_mm']:.3f} | "
            f"{r['peak_height_difference_percent']:.3f} | {r['distal_R80_difference_mm']:.3f} | "
            f"{r['distal_R50_difference_mm']:.3f} | {r['integral_difference_percent']:.3f} | "
            f"{r['idd_normalized_rmse_percent']:.3f} |"
        )
    lines += ["", "## Lateral dose RMS", ""]
    for r in results:
        lines += [f"### {r['energy_MeVu']} MeV/u", "", "| Depth (mm) | GPU RMS (mm) | TOPAS RMS (mm) | Diff (%) |", "|---:|---:|---:|---:|"]
        for p in r["lateral_rms"]:
            lines.append(f"| {p['depth_mm']:.2f} | {p['gpu_rms_mm']:.4f} | {p['topas_rms_mm']:.4f} | {p['difference_percent']:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--topas-root", type=Path, default=Path("/mnt/sda/wuwei/carbon_emittance_inelastic_1M"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/fred_topas_1M_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results, curves = [], []
    for energy in ENERGIES:
        result, topas_idd, gpu_idd = analyze_energy(args.root, args.topas_root, energy)
        results.append(result)
        curves.append((energy, topas_idd, gpu_idd))

    report = {"scorer": "3D DoseToMedium, X/Y summed", "histories": 1_000_000, "results": results}
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "comparison.md").write_text(markdown(results))

    z = (np.arange(NZ) + 0.5) * DZ_MM
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (energy, topas_idd, gpu_idd) in zip(axes.flat, curves):
        ax.plot(z, topas_idd, label="TOPAS", lw=1.6)
        ax.plot(z, gpu_idd, label="GPU FRED model", lw=1.4)
        ax.set(title=f"{energy} MeV/u", xlabel="Depth (mm)", ylabel="X/Y-summed dose (Gy)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(args.output_dir / "idd_comparison.png", dpi=180)
    print(markdown(results))


if __name__ == "__main__":
    main()
