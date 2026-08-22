#!/usr/bin/env python3
"""Audit full-physics carbon IDD with no empirical straggling scale."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


os.environ.setdefault("MPLCONFIGDIR", "/tmp/maigo-matplotlib")

ENERGIES = (100, 200, 300, 400)
SEEDS = tuple(f"seed_202607{day}" for day in range(15, 20))
DEFAULT_ROOT = Path("out/carbon_scale_free_idd_1M")
TOPAS_ROOT = Path(
    "benchmark/phantom/"
    "formal_results_topas_v4.2.3_g4.11.3.2_100k_20260811/A1"
)
TOPAS_FILENAMES = {
    100: "e100_development_dose.csv",
    200: "development_dose.csv",
    300: "e300_development_dose.csv",
    400: "e400_development_dose.csv",
}
PRIMARY_PACKAGE = Path(
    "data/packages/topas_water_inclxx_1M_stitch7_primary_3d.bin"
)
CASCADE_PACKAGE = Path(
    "data/packages/topas_400MeVu_water_inclxx_1M_cascade_3d.bin"
)


def read_topas(path: Path) -> tuple[np.ndarray, np.ndarray]:
    text = path.read_text(encoding="utf-8", errors="replace")
    grid = re.search(
        r"^#\s*Z\s+in\s+(\d+)\s+bins?\s+of\s+([0-9.eE+-]+)\s*cm\s*$",
        text,
        flags=re.MULTILINE,
    )
    if grid is None:
        raise ValueError(f"{path}: missing TOPAS Z grid")
    bins, width_mm = int(grid.group(1)), float(grid.group(2)) * 10.0
    indices: list[int] = []
    values: list[float] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            indices.append(int(float(fields[2])))
            values.append(float(fields[3]))
    if len(values) != bins:
        raise ValueError(f"{path}: expected {bins} bins, found {len(values)}")
    depth = (np.asarray(indices, dtype=np.float64) + 0.5) * width_mm
    return depth, np.asarray(values, dtype=np.float64)


def read_topas_ensemble(energy: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    curves: list[np.ndarray] = []
    reference_depth: np.ndarray | None = None
    for seed in SEEDS:
        path = TOPAS_ROOT / seed / "output" / TOPAS_FILENAMES[energy]
        depth, dose = read_topas(path)
        if reference_depth is None:
            reference_depth = depth
        elif not np.allclose(depth, reference_depth, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: TOPAS seed grids differ")
        curves.append(dose)
    assert reference_depth is not None
    samples_1m = np.stack(curves) * 10.0
    return reference_depth, samples_1m.mean(axis=0), samples_1m


def read_gpu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([float(row["depth_mm"]) for row in rows]),
        np.asarray([float(row["dose_Gy"]) for row in rows]),
    )


def integrate(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[mask]
    # Every authoritative scorer uses the same constant 0.5 mm bin width.
    return float(selected.sum() * 0.5)


def interpolate_crossing(
    depth: np.ndarray,
    dose: np.ndarray,
    threshold: float,
    start: int,
    direction: int,
) -> float:
    current = start
    while 0 <= current + direction < dose.size:
        following = current + direction
        y0, y1 = dose[current], dose[following]
        if (y0 - threshold) * (y1 - threshold) <= 0.0 and y0 != y1:
            fraction = (threshold - y0) / (y1 - y0)
            return float(depth[current] + fraction * (depth[following] - depth[current]))
        current = following
    raise ValueError("profile crossing not found")


def shape_metrics(depth: np.ndarray, dose: np.ndarray) -> dict[str, float]:
    peak_index = int(np.argmax(dose))
    peak = float(dose[peak_index])
    proximal_50 = interpolate_crossing(depth, dose, 0.5 * peak, peak_index, -1)
    distal_50 = interpolate_crossing(depth, dose, 0.5 * peak, peak_index, 1)
    distal_80 = interpolate_crossing(depth, dose, 0.8 * peak, peak_index, 1)
    return {
        "peak_Gy": peak,
        "peak_depth_mm": float(depth[peak_index]),
        "fwhm_mm": distal_50 - proximal_50,
        "distal_r80_mm": distal_80,
    }


def validate_config(path: Path, energy: int) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "number_of_histories": 1_000_000,
        "initial_energy_MeVu": float(energy),
        "primary_atomic_number": 6,
        "primary_mass_number": 12,
        "energy_straggling_model": "gaussian_clamped",
        "straggling_scale": 1.0,
        "enable_step_stable_straggling": False,
        "enable_secondary_energy_straggling": False,
        "primary_reaction_package_file": str(PRIMARY_PACKAGE),
        "cascade_package_file": str(CASCADE_PACKAGE),
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    forbidden = (
        "straggling_scale_energies_MeVu",
        "straggling_scale_values",
    )
    present_forbidden = [key for key in forbidden if key in config]
    if mismatches or present_forbidden:
        raise ValueError(
            f"{path}: invalid scale-free config: mismatches={mismatches}, "
            f"forbidden={present_forbidden}"
        )
    return config


def validate_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    histories = re.findall(r"^Histories:\s+(\d+)", text, flags=re.MULTILINE)
    backends = re.findall(r"^Backend:\s+(.+)$", text, flags=re.MULTILINE)
    if not histories or not backends:
        raise ValueError(f"{path}: missing run summary")
    overflow_names = ("Secondary queue overflow", "Cascade queue overflow")
    overflow = {}
    for name in overflow_names:
        values = re.findall(rf"^{re.escape(name)}:\s+(\d+)", text, flags=re.MULTILINE)
        if not values:
            raise ValueError(f"{path}: missing {name}")
        overflow[name] = int(values[-1])
    valid = (
        int(histories[-1]) == 1_000_000
        and "+straggling" in backends[-1]
        and all(value == 0 for value in overflow.values())
        and "Dose scorer output (Gy):" in text
    )
    if not valid:
        raise ValueError(f"{path}: incomplete or invalid run")
    return {
        "histories": int(histories[-1]),
        "backend": backends[-1],
        "queue_overflow": overflow,
    }


def relative_error(numerator: float, denominator: float) -> float:
    return 100.0 * (numerator / denominator - 1.0)


def audit_energy(root: Path, energy: int) -> dict[str, Any]:
    run_dir = root / f"{energy}MeVu"
    config_path = run_dir / "config.yaml"
    log_path = run_dir / "run.log"
    gpu_path = run_dir / "dose_Gy.csv"
    validate_config(config_path, energy)
    log = validate_log(log_path)
    depth, topas, topas_samples = read_topas_ensemble(energy)
    gpu_depth, gpu = read_gpu(gpu_path)
    if not np.allclose(depth, gpu_depth, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{energy} MeV/u: GPU and TOPAS grids differ")

    topas_shape = shape_metrics(depth, topas)
    gpu_shape = shape_metrics(depth, gpu)
    topas_peak_index = int(np.argmax(topas))
    peak_depth = float(depth[topas_peak_index])
    peak_window = np.abs(depth - peak_depth) <= 2.0
    pre_bragg = depth < peak_depth - 5.0
    distal_tail = depth > peak_depth + 5.0
    roi = topas > 0.01 * topas_shape["peak_Gy"]
    signed = 100.0 * (gpu[roi] / topas[roi] - 1.0)

    regions: dict[str, dict[str, float]] = {}
    for name, mask in (
        ("pre_bragg", pre_bragg),
        ("peak_pm_2mm", peak_window),
        ("distal_tail", distal_tail),
        ("total", np.ones_like(depth, dtype=bool)),
    ):
        topas_integral = integrate(topas, mask)
        gpu_integral = integrate(gpu, mask)
        regions[name] = {
            "topas_integral_Gy_mm": topas_integral,
            "gpu_integral_Gy_mm": gpu_integral,
            "error_percent": relative_error(gpu_integral, topas_integral),
        }

    seed_totals = np.asarray([integrate(sample) for sample in topas_samples])
    return {
        "energy_MeVu": energy,
        "topas_peak_Gy": topas_shape["peak_Gy"],
        "gpu_peak_Gy": gpu_shape["peak_Gy"],
        "peak_height_error_percent": relative_error(
            gpu_shape["peak_Gy"], topas_shape["peak_Gy"]
        ),
        "topas_peak_depth_mm": topas_shape["peak_depth_mm"],
        "gpu_peak_depth_mm": gpu_shape["peak_depth_mm"],
        "peak_depth_error_mm": (
            gpu_shape["peak_depth_mm"] - topas_shape["peak_depth_mm"]
        ),
        "fwhm_error_mm": gpu_shape["fwhm_mm"] - topas_shape["fwhm_mm"],
        "distal_r80_error_mm": (
            gpu_shape["distal_r80_mm"] - topas_shape["distal_r80_mm"]
        ),
        "regions": regions,
        "roi_definition": "TOPAS ensemble mean > 1% of its peak",
        "roi_bins": int(np.count_nonzero(roi)),
        "roi_mean_signed_error_percent": float(np.mean(signed)),
        "roi_mae_percent": float(np.mean(np.abs(signed))),
        "roi_p95_absolute_error_percent": float(np.percentile(np.abs(signed), 95.0)),
        "roi_within_1_percent": float(np.mean(np.abs(signed) <= 1.0) * 100.0),
        "topas_seed_total_integral_relative_sd_percent": float(
            100.0 * np.std(seed_totals, ddof=1) / np.mean(seed_totals)
        ),
        "run": log,
        "inputs": {
            "gpu_config": str(config_path),
            "gpu_log": str(log_path),
            "gpu_dose": str(gpu_path),
            "topas_seeds": [
                str(TOPAS_ROOT / seed / "output" / TOPAS_FILENAMES[energy])
                for seed in SEEDS
            ],
        },
    }


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Carbon Scale-free Full-physics IDD Audit",
        "",
        "GPU uses 1M histories and TOPAS uses the mean of five independent 100k runs multiplied by 10. This is only a history-count conversion; no profile normalization is applied.",
        "All GPU runs use gaussian_clamped straggling with scalar scale = 1.0, no energy-wise scale table, and the required primary and cascade packages.",
        "",
        "| Energy | Peak height | Peak depth | R80 | FWHM | Peak +/-2 mm | Pre-Bragg | Tail | Total | ROI MAE / p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["energies"]:
        regions = item["regions"]
        lines.append(
            f"| {item['energy_MeVu']} MeV/u | "
            f"{item['peak_height_error_percent']:+.3f}% | "
            f"{item['peak_depth_error_mm']:+.3f} mm | "
            f"{item['distal_r80_error_mm']:+.3f} mm | "
            f"{item['fwhm_error_mm']:+.3f} mm | "
            f"{regions['peak_pm_2mm']['error_percent']:+.3f}% | "
            f"{regions['pre_bragg']['error_percent']:+.3f}% | "
            f"{regions['distal_tail']['error_percent']:+.3f}% | "
            f"{regions['total']['error_percent']:+.3f}% | "
            f"{item['roi_mae_percent']:.3f}% / "
            f"{item['roi_p95_absolute_error_percent']:.3f}% |"
        )
    lines += [
        "",
        "Peak height is a single 0.5 mm bin. Peak +/-2 mm integral, R80, and FWHM are more robust measures of primary Bragg-peak agreement. Pre-Bragg and distal-tail integrals include nuclear-fragment contributions and therefore do not isolate energy straggling.",
        "",
    ]
    return "\n".join(lines)


def plot_report(report: dict[str, Any], root: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(ENERGIES),
        2,
        figsize=(13.0, 14.5),
        gridspec_kw={"width_ratios": (1.5, 1.0)},
        constrained_layout=True,
    )
    for row, item in enumerate(report["energies"]):
        energy = item["energy_MeVu"]
        depth, topas, _ = read_topas_ensemble(energy)
        gpu_depth, gpu = read_gpu(Path(item["inputs"]["gpu_dose"]))
        roi = topas > 0.01 * np.max(topas)
        error = np.full_like(topas, np.nan)
        error[roi] = 100.0 * (gpu[roi] / topas[roi] - 1.0)
        visible = np.flatnonzero(topas > 0.005 * np.max(topas))
        x_max = min(400.0, float(depth[visible[-1]]) + 5.0)

        dose_ax, error_ax = axes[row]
        dose_ax.plot(depth, topas * 1.0e6, color="#202124", lw=1.8, label="TOPAS 5-seed mean x10")
        dose_ax.plot(gpu_depth, gpu * 1.0e6, color="#d55e00", lw=1.3, label="GPU 1M, scale=1")
        dose_ax.set(xlim=(0.0, x_max), ylim=(0.0, None), ylabel="Absolute dose [uGy]")
        dose_ax.set_title(f"{energy} MeV/u carbon")
        dose_ax.grid(alpha=0.2, linewidth=0.6)
        dose_ax.text(
            0.015,
            0.95,
            f"peak window {item['regions']['peak_pm_2mm']['error_percent']:+.2f}%  |  "
            f"R80 {item['distal_r80_error_mm']:+.2f} mm  |  "
            f"total {item['regions']['total']['error_percent']:+.2f}%",
            transform=dose_ax.transAxes,
            va="top",
            fontsize=8.5,
        )
        if row == 0:
            dose_ax.legend(frameon=False, loc="upper right", fontsize=8.5)

        error_ax.axhspan(-1.0, 1.0, color="#009e73", alpha=0.12, label="+/-1%")
        error_ax.axhline(0.0, color="#202124", lw=0.8)
        error_ax.plot(depth, error, color="#0072b2", lw=1.0)
        error_ax.set(
            xlim=(0.0, x_max),
            ylim=(-6.0, 6.0),
            yticks=(-6, -3, 0, 3, 6),
            ylabel="GPU / TOPAS - 1 [%]",
        )
        error_ax.set_title(
            f"ROI MAE {item['roi_mae_percent']:.2f}%  |  "
            f"p95 {item['roi_p95_absolute_error_percent']:.2f}%"
        )
        error_ax.grid(alpha=0.2, linewidth=0.6)
        if row == len(ENERGIES) - 1:
            dose_ax.set_xlabel("Depth in water [mm]")
            error_ax.set_xlabel("Depth in water [mm]")

    figure.suptitle(
        "Carbon full-physics IDD without empirical straggling scaling\n"
        "Absolute dose; GPU 1M vs TOPAS five-seed 100k mean x10; error ROI >1% peak",
        fontsize=13,
    )
    png = root / "carbon_scale_free_idd_absolute_multienergy.png"
    pdf = root / "carbon_scale_free_idd_absolute_multienergy.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "scope": "carbon water full-physics absolute IDD without empirical scale",
        "normalization": "none; TOPAS five-seed 100k mean multiplied by 10 for history-count equivalence",
        "gpu_histories": 1_000_000,
        "topas_histories_per_seed": 100_000,
        "topas_seed_count": len(SEEDS),
        "straggling_scale": 1.0,
        "primary_package": str(PRIMARY_PACKAGE),
        "cascade_package": str(CASCADE_PACKAGE),
        "energies": [audit_energy(args.root, energy) for energy in ENERGIES],
    }
    json_path = args.root / "carbon_scale_free_idd_audit.json"
    md_path = args.root / "carbon_scale_free_idd_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(make_markdown(report), encoding="utf-8")
    png, pdf = plot_report(report, args.root)
    print(make_markdown(report))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
