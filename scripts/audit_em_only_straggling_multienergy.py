#!/usr/bin/env python3
"""Audit scale-free packaged energy straggling against TOPAS EM-only dose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ENERGIES = (70, 100, 150, 200, 250)
DEFAULT_ROOT = Path("out/proton_water_qgsp_bic_hp/em_only_multienergy_1M")
TOPAS_250 = Path(
    "out/proton_water_qgsp_bic_hp/full_physics_compare_1M/250MeV/"
    "dose_em_only_pair/topas_em_only/result/"
    "proton_250MeV_water_em_only_1M_total_dose.csv"
)
GPU_250 = Path(
    "out/proton_water_qgsp_bic_hp/full_physics_compare_1M/250MeV/"
    "dose_em_only_pair/gpu_em_only_packaged_fluctuation"
)


def read_topas(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"{path}: expected four TOPAS columns")
        values.append(float(fields[3]))
    if len(values) != 1400:
        raise ValueError(f"{path}: expected 1400 bins, found {len(values)}")
    # TOPAS Z bin zero is at the downstream face for this component placement.
    dose = np.asarray(values[::-1], dtype=np.float64)
    depth = (np.arange(dose.size, dtype=np.float64) + 0.5) * 0.5
    return depth, dose


def read_gpu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    depth = np.asarray([float(row["depth_mm"]) for row in rows], dtype=np.float64)
    dose = np.asarray([float(row["dose_Gy"]) for row in rows], dtype=np.float64)
    if depth.size != 1400:
        raise ValueError(f"{path}: expected 1400 bins, found {depth.size}")
    return depth, dose


def integral(depth: np.ndarray, values: np.ndarray) -> float:
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, depth))


def crossing(
    depth: np.ndarray,
    dose: np.ndarray,
    threshold: float,
    peak_index: int,
    direction: int,
) -> float:
    index = peak_index - 1 if direction < 0 else peak_index
    stop = -1 if direction < 0 else dose.size - 1
    for current in range(index, stop, direction):
        following = current + direction
        y0, y1 = dose[current], dose[following]
        if (y0 - threshold) * (y1 - threshold) <= 0.0 and y0 != y1:
            return float(
                depth[current]
                + (threshold - y0)
                * (depth[following] - depth[current])
                / (y1 - y0)
            )
    raise ValueError("profile crossing not found")


def fwhm(depth: np.ndarray, dose: np.ndarray) -> float:
    peak_index = int(np.argmax(dose))
    half = 0.5 * float(dose[peak_index])
    proximal = crossing(depth, dose, half, peak_index, -1)
    distal = crossing(depth, dose, half, peak_index, 1)
    return distal - proximal


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, Any] = {}
    for key, prefix in (
        ("backend", "Backend:"),
        ("histories", "Histories:"),
        ("energy_balance_error", "Energy balance error:"),
    ):
        lines = [line for line in text.splitlines() if line.startswith(prefix)]
        if not lines:
            raise ValueError(f"{path}: missing {prefix}")
        value = lines[-1].split(prefix, 1)[1].strip().split()[0]
        if key == "backend":
            result[key] = lines[-1].split(prefix, 1)[1].strip()
        else:
            result[key] = int(value) if key == "histories" else float(value)
    exit_lines = [line for line in text.splitlines() if line.startswith("EXIT_CODE=")]
    result["exit_code"] = int(exit_lines[-1].split("=", 1)[1]) if exit_lines else None
    result["complete_output_marker"] = "Dose scorer output (Gy):" in text
    return result


def audit_energy(root: Path, energy: int) -> dict[str, Any]:
    if energy == 250:
        topas_path = TOPAS_250
        gpu_dir = GPU_250
        log_path = gpu_dir / "run.log"
    else:
        topas_path = (
            root / "topas" / f"{energy}MeV"
            / f"proton_{energy}MeV_water_em_only_1M_total_dose.csv"
        )
        gpu_dir = root / "gpu" / f"{energy}MeV"
        log_path = gpu_dir / "run_1M.log"
    gpu_path = gpu_dir / "dose_Gy.csv"
    config_path = gpu_dir / "config.yaml"

    topas_depth, topas = read_topas(topas_path)
    gpu_depth, gpu = read_gpu(gpu_path)
    if not np.allclose(topas_depth, gpu_depth, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{energy} MeV: depth grids differ")
    config = config_path.read_text(encoding="utf-8", errors="replace")
    required = (
        "number_of_histories: 1000000",
        "energy_straggling_model: packaged_fluctuation",
        "straggling_scale: 1.0",
        "enable_primary_attenuation: false",
        "enable_primary_elastic_interactions: false",
        "enable_secondary_generation: false",
        "enable_fragment_cascade: false",
        "enable_neutral_transport: false",
    )
    missing = [line for line in required if line not in config]
    if missing:
        raise ValueError(f"{config_path}: missing required settings: {missing}")
    run = parse_log(log_path)
    bad_exit = run["exit_code"] is not None and run["exit_code"] != 0
    if (
        run["histories"] != 1_000_000
        or "+packaged-fluctuation" not in run["backend"]
        or bad_exit
        or not run["complete_output_marker"]
    ):
        raise ValueError(f"{log_path}: incomplete run: {run}")

    topas_peak_index = int(np.argmax(topas))
    gpu_peak_index = int(np.argmax(gpu))
    topas_peak = float(topas[topas_peak_index])
    gpu_peak = float(gpu[gpu_peak_index])
    mask = topas > 0.01 * topas_peak
    signed = 100.0 * (gpu[mask] / topas[mask] - 1.0)
    topas_integral = integral(topas_depth, topas)
    gpu_integral = integral(gpu_depth, gpu)
    topas_fwhm = fwhm(topas_depth, topas)
    gpu_fwhm = fwhm(gpu_depth, gpu)
    return {
        "energy_MeV": energy,
        "histories": run["histories"],
        "straggling_model": "packaged_fluctuation",
        "straggling_scale": 1.0,
        "topas_peak_Gy": topas_peak,
        "gpu_peak_Gy": gpu_peak,
        "peak_height_error_percent": 100.0 * (gpu_peak / topas_peak - 1.0),
        "topas_peak_depth_mm": float(topas_depth[topas_peak_index]),
        "gpu_peak_depth_mm": float(gpu_depth[gpu_peak_index]),
        "peak_depth_error_mm": float(gpu_depth[gpu_peak_index] - topas_depth[topas_peak_index]),
        "topas_fwhm_mm": topas_fwhm,
        "gpu_fwhm_mm": gpu_fwhm,
        "fwhm_error_mm": gpu_fwhm - topas_fwhm,
        "topas_integral_Gy_mm": topas_integral,
        "gpu_integral_Gy_mm": gpu_integral,
        "integral_error_percent": 100.0 * (gpu_integral / topas_integral - 1.0),
        "roi_bins": int(np.count_nonzero(mask)),
        "roi_mean_signed_error_percent": float(np.mean(signed)),
        "roi_mae_percent": float(np.mean(np.abs(signed))),
        "roi_p95_absolute_error_percent": float(np.percentile(np.abs(signed), 95.0)),
        "roi_within_1_percent": float(np.mean(np.abs(signed) <= 1.0) * 100.0),
        "energy_balance_error": run["energy_balance_error"],
        "backend": run["backend"],
        "run_completion": {
            "exit_code": run["exit_code"],
            "complete_output_marker": run["complete_output_marker"],
        },
        "inputs": {
            "topas_dose": str(topas_path),
            "gpu_dose": str(gpu_path),
            "gpu_config": str(config_path),
            "gpu_log": str(log_path),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# EM-only Scale-free Straggling Audit",
        "",
        "Absolute TOPAS DoseToMedium and GPU dose are compared without normalization.",
        "All GPU runs use packaged_fluctuation with straggling_scale = 1.0; all nuclear processes are disabled.",
        "",
        "| Energy | Peak height | Peak depth | FWHM error | Integral | ROI MAE | ROI p95 | Within 1% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["energies"]:
        lines.append(
            f"| {item['energy_MeV']} MeV | {item['peak_height_error_percent']:+.3f}% | "
            f"{item['peak_depth_error_mm']:+.3f} mm | {item['fwhm_error_mm']:+.3f} mm | "
            f"{item['integral_error_percent']:+.4f}% | {item['roi_mae_percent']:.3f}% | "
            f"{item['roi_p95_absolute_error_percent']:.3f}% | "
            f"{item['roi_within_1_percent']:.2f}% |"
        )
    lines += [
        "",
        "ROI is defined by TOPAS dose greater than 1% of its peak. Peak height is a single 0.5 mm bin and is therefore more sensitive to independent Monte Carlo noise than FWHM or integral metrics.",
        "",
    ]
    return "\n".join(lines)


def plot_report(report: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(report["energies"]), 2, figsize=(12.0, 14.0),
        gridspec_kw={"width_ratios": (1.45, 1.0)}, constrained_layout=True,
    )
    topas_color = "#202124"
    gpu_color = "#d55e00"
    error_color = "#0072b2"
    for row, item in enumerate(report["energies"]):
        topas_depth, topas = read_topas(Path(item["inputs"]["topas_dose"]))
        gpu_depth, gpu = read_gpu(Path(item["inputs"]["gpu_dose"]))
        peak = float(np.max(topas))
        roi = topas > 0.01 * peak
        error = np.full_like(topas, np.nan)
        error[roi] = 100.0 * (gpu[roi] / topas[roi] - 1.0)
        visible = np.flatnonzero(topas > 0.005 * peak)
        x_max = min(700.0, float(topas_depth[visible[-1]]) + 5.0)

        dose_ax, error_ax = axes[row]
        dose_ax.plot(
            topas_depth, topas * 1.0e6, color=topas_color,
            linewidth=1.8, label="TOPAS",
        )
        dose_ax.plot(
            gpu_depth, gpu * 1.0e6, color=gpu_color,
            linewidth=1.35, label="GPU",
        )
        dose_ax.set_xlim(0.0, x_max)
        dose_ax.set_ylim(bottom=0.0)
        dose_ax.set_ylabel("Absolute dose [uGy]")
        dose_ax.set_title(f"{item['energy_MeV']} MeV proton")
        dose_ax.grid(alpha=0.2, linewidth=0.6)
        dose_ax.text(
            0.015, 0.95,
            f"peak {item['peak_height_error_percent']:+.3f}%  |  "
            f"FWHM {item['fwhm_error_mm']:+.3f} mm",
            transform=dose_ax.transAxes, va="top", fontsize=8.5,
        )
        if row == 0:
            dose_ax.legend(frameon=False, loc="upper right")

        error_ax.axhspan(-1.0, 1.0, color="#009e73", alpha=0.10)
        error_ax.axhline(0.0, color=topas_color, linewidth=0.8)
        error_ax.plot(topas_depth, error, color=error_color, linewidth=1.0)
        error_ax.set_xlim(0.0, x_max)
        error_ax.set_ylim(-6.0, 6.0)
        error_ax.set_yticks((-6, -3, 0, 3, 6))
        error_ax.set_ylabel("GPU / TOPAS - 1 [%]")
        error_ax.set_title(
            f"ROI MAE {item['roi_mae_percent']:.3f}%  |  "
            f"p95 {item['roi_p95_absolute_error_percent']:.3f}%"
        )
        error_ax.grid(alpha=0.2, linewidth=0.6)
        if row == len(report["energies"]) - 1:
            dose_ax.set_xlabel("Depth in water [mm]")
            error_ax.set_xlabel("Depth in water [mm]")

    figure.suptitle(
        "Scale-free packaged energy straggling: absolute IDD\n"
        "1M histories per code and energy; no normalization; straggling scale = 1",
        fontsize=13,
    )
    png = output_root / "scale_free_straggling_absolute_dose_multienergy.png"
    pdf = output_root / "scale_free_straggling_absolute_dose_multienergy.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "scope": "proton water EM-only absolute-dose multienergy validation",
        "normalization": "none",
        "histories_per_code_per_energy": 1_000_000,
        "energies": [audit_energy(args.root, energy) for energy in ENERGIES],
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "scale_free_straggling_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.root / "scale_free_straggling_audit.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(markdown(report))
    if args.plot:
        png, pdf = plot_report(report, args.root)
        print(f"wrote {png}")
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
