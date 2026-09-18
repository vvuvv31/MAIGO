#!/usr/bin/env python3
"""Summarize TOPAS minibeam phase space at the water entrance.

The analysis intentionally keeps the collimator-path classification geometric:
a primary is "direct" only when its entrance ray remains inside one and the
same slit for the full 60 mm Copper thickness.  All other downstream primary
C12 tracks are labelled Copper-touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C12_PDG = 1000060120
SLIT_PITCH_MM = 3.6
SLIT_HALF_WIDTH_MM = 0.25
COLLIMATOR_THICKNESS_MM = 60.0
DEFAULT_STOPPING_POWER = Path("data/stopping_power_water_geant4_11_3_2.csv")


def load_phsp(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 14:
        raise ValueError(f"Expected at least 14 TOPAS columns in {path}")
    return data


def primary_c12(data: np.ndarray) -> np.ndarray:
    return data[(data[:, 7] == C12_PDG) & (data[:, 13] == 0)]


def directions(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ux = data[:, 3]
    uy = data[:, 4]
    uz = np.sqrt(np.maximum(0.0, 1.0 - ux * ux - uy * uy))
    uz = np.where(data[:, 8] != 0, -uz, uz)
    return ux, uy, uz


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def moments(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0}
    centered = values - np.mean(values)
    variance = np.mean(centered * centered)
    kurtosis = np.mean(centered**4) / variance**2 if variance > 0 else None
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "kurtosis": float(kurtosis) if kurtosis is not None else None,
    }


def angle_summary(rows: np.ndarray) -> dict:
    if rows.size == 0:
        return {"count": 0}
    ux, uy, uz = directions(rows)
    theta_x = np.arctan2(ux, uy) * 1.0e3
    theta_z = np.arctan2(uz, uy) * 1.0e3
    theta_r = np.hypot(theta_x, theta_z)
    abs_x = np.abs(theta_x)
    quantiles = [0.68, 0.95, 0.99, 0.999]
    qx = np.quantile(abs_x, quantiles)
    qr = np.quantile(theta_r, quantiles)
    x_mm = rows[:, 0] * 10.0
    z_mm = rows[:, 2] * 10.0
    energy = rows[:, 5]
    return {
        "count": int(rows.shape[0]),
        "theta_x_mrad": moments(theta_x),
        "theta_z_mrad": moments(theta_z),
        "theta_r_mrad_quantiles": {
            str(q): float(v) for q, v in zip(quantiles, qr)
        },
        "abs_theta_x_mrad_quantiles": {
            str(q): float(v) for q, v in zip(quantiles, qx)
        },
        "abs_theta_x_tail_ratio_q99_over_q68": float(qx[2] / qx[0]),
        "x_mm": moments(x_mm),
        "z_mm": moments(z_mm),
        "energy_MeV": moments(energy),
        "correlation_x_theta_x": corr(x_mm, theta_x),
        "correlation_z_theta_z": corr(z_mm, theta_z),
        "correlation_energy_theta_r": corr(energy, theta_r),
    }


def slit_microstructure(rows: np.ndarray, stopping: tuple[np.ndarray, np.ndarray]) -> dict:
    """Summarize primary fluence and local stopping proxy within one slit pitch."""
    if rows.size == 0:
        return {"count": 0}
    x_mm = rows[:, 0] * 10.0
    residual_mm = x_mm - np.rint(x_mm / SLIT_PITCH_MM) * SLIT_PITCH_MM
    energy_mevu = rows[:, 5] / 12.0
    stopping_power = np.interp(energy_mevu, stopping[0], stopping[1])
    window_mm = 0.25
    peak = np.abs(residual_mm) <= window_mm
    valley = np.abs(np.abs(residual_mm) - 0.5 * SLIT_PITCH_MM) <= window_mm
    peak_count = int(np.count_nonzero(peak))
    valley_count = int(np.count_nonzero(valley))
    peak_stopping = float(stopping_power[peak].sum())
    valley_stopping = float(stopping_power[valley].sum())
    return {
        "count": int(rows.shape[0]),
        "window_half_width_mm": window_mm,
        "peak_count": peak_count,
        "valley_count": valley_count,
        "valley_over_peak_fluence": valley_count / peak_count if peak_count else None,
        "peak_mean_energy_MeV": float(np.mean(rows[peak, 5])) if peak_count else None,
        "valley_mean_energy_MeV": float(np.mean(rows[valley, 5])) if valley_count else None,
        "peak_stopping_proxy_MeV_per_mm": peak_stopping,
        "valley_stopping_proxy_MeV_per_mm": valley_stopping,
        "valley_over_peak_stopping_proxy": (
            valley_stopping / peak_stopping if peak_stopping > 0.0 else None),
    }


def direct_event_ids(entrance: np.ndarray) -> set[int]:
    rows = primary_c12(entrance)
    ux, uy, _ = directions(rows)
    x0 = rows[:, 0] * 10.0
    x1 = x0 + COLLIMATOR_THICKNESS_MM * ux / uy
    slit0 = np.rint(x0 / SLIT_PITCH_MM)
    slit1 = np.rint(x1 / SLIT_PITCH_MM)
    inside0 = np.abs(x0 - slit0 * SLIT_PITCH_MM) <= SLIT_HALF_WIDTH_MM
    inside1 = np.abs(x1 - slit1 * SLIT_PITCH_MM) <= SLIT_HALF_WIDTH_MM
    valid_slit = (np.abs(slit0) <= 7) & (slit0 == slit1)
    direct = inside0 & inside1 & valid_slit
    return set(rows[direct, 11].astype(np.int64).tolist())


def species_group(pdg: int) -> str:
    if pdg == 2212:
        return "proton"
    if pdg <= 1_000_000_000:
        return {2112: "neutron", 22: "gamma", 11: "electron", -11: "positron"}.get(
            pdg, "other_nonion"
        )
    z = (pdg // 10000) % 1000
    return {1: "hydrogen_ions", 2: "helium", 3: "lithium", 4: "beryllium", 5: "boron", 6: "carbon"}.get(
        z, "other_ions"
    )


def secondary_summary(data: np.ndarray) -> dict:
    grouped: dict[str, list[np.ndarray]] = {}
    for pdg in np.unique(data[:, 7].astype(np.int64)):
        if pdg == C12_PDG:
            continue
        rows = data[data[:, 7] == pdg]
        grouped.setdefault(species_group(int(pdg)), []).append(rows)
    result = {}
    for name, pieces in sorted(grouped.items()):
        rows = np.concatenate(pieces)
        result[name] = angle_summary(rows)
    return result


def analyze_case(run_dir: Path, stopping: tuple[np.ndarray, np.ndarray]
                 ) -> tuple[dict, dict[str, np.ndarray]]:
    entrance = load_phsp(run_dir / "output/collimator_entrance.phsp")
    water = load_phsp(run_dir / "output/water_entrance.phsp")
    water_primary = primary_c12(water)
    direct_ids = direct_event_ids(entrance)
    is_direct = np.fromiter(
        (int(event) in direct_ids for event in water_primary[:, 11]),
        dtype=bool,
        count=water_primary.shape[0],
    )
    direct = water_primary[is_direct]
    touched = water_primary[~is_direct]
    summary = {
        "run_dir": str(run_dir),
        "water_primary_all": angle_summary(water_primary),
        "water_primary_direct": angle_summary(direct),
        "water_primary_copper_touched": angle_summary(touched),
        "water_primary_all_slit_microstructure": slit_microstructure(
            water_primary, stopping),
        "water_primary_direct_slit_microstructure": slit_microstructure(
            direct, stopping),
        "water_primary_copper_touched_slit_microstructure": slit_microstructure(
            touched, stopping),
        "water_primary_path_counts": {
            "all": int(water_primary.shape[0]),
            "direct": int(direct.shape[0]),
            "copper_touched": int(touched.shape[0]),
        },
        "water_secondary_species": secondary_summary(water),
    }
    return summary, {"all": water_primary, "direct": direct, "touched": touched}


def gpu_rows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map canonical GPU +z phase space onto the TOPAS +y column layout."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    data = np.atleast_1d(data)
    rows = np.zeros((data.size, 14), dtype=np.float64)
    rows[:, 0] = data["x_mm"] / 10.0
    rows[:, 2] = data["y_mm"] / 10.0
    rows[:, 3] = data["direction_x"]
    rows[:, 4] = data["direction_z"]
    rows[:, 5] = data["kinetic_energy_MeV"]
    rows[:, 8] = data["direction_y"] < 0.0
    touched = data["copper_touched"] != 0
    elastic = (data["copper_elastic"] != 0) if "copper_elastic" in data.dtype.names else np.zeros(data.size, bool)
    return rows, touched, elastic


def analyze_gpu_case(path: Path, stopping: tuple[np.ndarray, np.ndarray]) -> dict:
    rows, touched, elastic = gpu_rows(path)
    return {
        "path": str(path),
        "water_primary_all": angle_summary(rows),
        "water_primary_direct": angle_summary(rows[~touched]),
        "water_primary_copper_touched": angle_summary(rows[touched]),
        "water_primary_copper_touched_no_elastic": angle_summary(rows[touched & ~elastic]),
        "water_primary_copper_elastic": angle_summary(rows[elastic]),
        "water_primary_all_slit_microstructure": slit_microstructure(rows, stopping),
        "water_primary_direct_slit_microstructure": slit_microstructure(
            rows[~touched], stopping),
        "water_primary_copper_touched_slit_microstructure": slit_microstructure(
            rows[touched], stopping),
        "water_primary_path_counts": {
            "all": int(rows.shape[0]),
            "direct": int(np.count_nonzero(~touched)),
            "copper_touched": int(np.count_nonzero(touched)),
            "copper_elastic": int(np.count_nonzero(elastic)),
        },
    }


def comparison_ratios(topas: dict, gpu: dict) -> dict:
    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return numerator / denominator

    result = {}
    for group in ("water_primary_all", "water_primary_direct",
                  "water_primary_copper_touched"):
        t, g = topas[group], gpu[group]
        result[group] = {
            "count_gpu_over_topas": ratio(g["count"], t["count"]),
        }
        if t["count"] == 0 or g["count"] == 0:
            continue
        result[group].update({
            "theta_x_std_gpu_over_topas":
                g["theta_x_mrad"]["std"] / t["theta_x_mrad"]["std"],
            "abs_theta_x_q68_gpu_over_topas":
                g["abs_theta_x_mrad_quantiles"]["0.68"] /
                t["abs_theta_x_mrad_quantiles"]["0.68"],
            "abs_theta_x_q95_gpu_over_topas":
                g["abs_theta_x_mrad_quantiles"]["0.95"] /
                t["abs_theta_x_mrad_quantiles"]["0.95"],
            "abs_theta_x_q99_gpu_over_topas":
                g["abs_theta_x_mrad_quantiles"]["0.99"] /
                t["abs_theta_x_mrad_quantiles"]["0.99"],
            "x_std_gpu_over_topas": g["x_mm"]["std"] / t["x_mm"]["std"],
            "energy_mean_gpu_over_topas":
                g["energy_MeV"]["mean"] / t["energy_MeV"]["mean"],
            "energy_std_gpu_over_topas":
                g["energy_MeV"]["std"] / t["energy_MeV"]["std"],
        })
        micro_key = f"{group}_slit_microstructure"
        tm, gm = topas[micro_key], gpu[micro_key]
        result[group]["slit_microstructure"] = {
            "valley_fluence_gpu_over_topas":
                ratio(gm["valley_over_peak_fluence"],
                      tm["valley_over_peak_fluence"]),
            "valley_stopping_proxy_gpu_over_topas":
                ratio(gm["valley_over_peak_stopping_proxy"],
                      tm["valley_over_peak_stopping_proxy"]),
            "peak_mean_energy_gpu_over_topas":
                ratio(gm["peak_mean_energy_MeV"], tm["peak_mean_energy_MeV"]),
            "valley_mean_energy_gpu_over_topas":
                ratio(gm["valley_mean_energy_MeV"],
                      tm["valley_mean_energy_MeV"]),
        }
    return result


def plot_diagnostics(samples: dict[int, dict[str, np.ndarray]], output: Path) -> None:
    colors = {150: "tab:blue", 250: "tab:orange", 300: "tab:green"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for energy, groups in sorted(samples.items()):
        for group, linestyle in (("direct", "-"), ("touched", "--")):
            rows = groups[group]
            _, uy, _ = directions(rows)
            theta_x = np.abs(np.arctan2(rows[:, 3], uy) * 1.0e3)
            theta_x.sort()
            cdf = (np.arange(theta_x.size) + 0.5) / theta_x.size
            axes[0, 0].plot(theta_x, cdf, color=colors[energy], ls=linestyle,
                            label=f"{energy} {group}")
        rows = groups["all"]
        ux, uy, uz = directions(rows)
        tx = np.arctan2(ux, uy) * 1.0e3
        tr = np.hypot(tx, np.arctan2(uz, uy) * 1.0e3)
        axes[0, 1].hexbin(rows[:, 0] * 10.0, tx, gridsize=80, bins="log",
                          mincnt=1, cmap="viridis", alpha=0.55)
        axes[1, 0].scatter(rows[::max(1, len(rows)//4000), 5],
                           tr[::max(1, len(rows)//4000)], s=2,
                           alpha=0.25, color=colors[energy], label=str(energy))
        residual = rows[:, 0] * 10.0 - np.rint(rows[:, 0] * 10.0 / SLIT_PITCH_MM) * SLIT_PITCH_MM
        axes[1, 1].hist(residual, bins=160, range=(-1.8, 1.8), histtype="step",
                        density=True, color=colors[energy], label=str(energy))
    axes[0, 0].set(xlabel=r"$|\theta_x|$ [mrad]", ylabel="CDF", xlim=(0, 100), ylim=(0, 1))
    axes[0, 0].legend(fontsize=8, ncol=2)
    axes[0, 1].set(xlabel="water-entry x [mm]", ylabel=r"$\theta_x$ [mrad]", ylim=(-100, 100))
    axes[1, 0].set(xlabel="water-entry C12 kinetic energy [MeV]", ylabel=r"$\theta_r$ [mrad]", ylim=(0, 150))
    axes[1, 0].legend(title="MeV/u")
    axes[1, 1].set(xlabel="x relative to nearest slit centre [mm]", ylabel="density")
    axes[1, 1].legend(title="MeV/u")
    fig.suptitle("TOPAS water-entrance primary C12 phase space")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", nargs=2,
                        metavar=("MEVU", "RUN_DIR"), required=True)
    parser.add_argument("--gpu-case", action="append", nargs=2,
                        metavar=("MEVU", "CSV"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stopping-power-file", type=Path,
                        default=DEFAULT_STOPPING_POWER)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stopping_data = np.loadtxt(
        args.stopping_power_file, delimiter=",", comments="#", skiprows=2)
    stopping = (stopping_data[:, 0], stopping_data[:, 1])
    summaries = {}
    samples = {}
    for energy_text, run_text in args.case:
        energy = int(energy_text)
        summaries[str(energy)], samples[energy] = analyze_case(
            Path(run_text), stopping)
    with (args.output_dir / "phase_space_joint_summary.json").open("w") as stream:
        json.dump(summaries, stream, indent=2)
    plot_diagnostics(samples, args.output_dir / "phase_space_joint_diagnostics.png")
    if args.gpu_case:
        gpu_summaries = {
            str(int(energy)): analyze_gpu_case(Path(path), stopping)
            for energy, path in args.gpu_case
        }
        comparisons = {
            energy: comparison_ratios(summaries[energy], gpu)
            for energy, gpu in gpu_summaries.items()
        }
        with (args.output_dir / "gpu_phase_space_summary.json").open("w") as stream:
            json.dump(gpu_summaries, stream, indent=2)
        with (args.output_dir / "gpu_topas_phase_space_comparison.json").open("w") as stream:
            json.dump(comparisons, stream, indent=2)


if __name__ == "__main__":
    main()
