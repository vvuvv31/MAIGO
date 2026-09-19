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
SLIT_EDGE_BINS_MM = np.asarray([0.0, 0.25, 0.5, 0.9, 1.3, 1.55, 1.800001])


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


def slit_edge_conditional_summary(
        rows: np.ndarray, stopping: tuple[np.ndarray, np.ndarray]) -> dict:
    """Summarize the joint water-entry state versus folded slit position."""
    if rows.size == 0:
        return {"count": 0, "bins": []}
    ux, uy, _ = directions(rows)
    theta_x_mrad = np.arctan2(ux, uy) * 1.0e3
    x_mm = rows[:, 0] * 10.0
    residual_mm = x_mm - np.rint(x_mm / SLIT_PITCH_MM) * SLIT_PITCH_MM
    abs_residual_mm = np.abs(residual_mm)
    outward_theta_mrad = np.sign(residual_mm) * theta_x_mrad
    energy_MeV = rows[:, 5]
    stopping_power = np.interp(
        energy_MeV / 12.0, stopping[0], stopping[1])
    bins = []
    for index, (low, high) in enumerate(zip(
            SLIT_EDGE_BINS_MM[:-1], SLIT_EDGE_BINS_MM[1:])):
        selected = ((abs_residual_mm >= low) &
                    ((abs_residual_mm < high) if index + 2 < len(
                        SLIT_EDGE_BINS_MM) else (abs_residual_mm <= high)))
        count = int(np.count_nonzero(selected))
        record = {
            "abs_residual_low_mm": float(low),
            "abs_residual_high_mm": float(min(high, 1.8)),
            "count": count,
            "fraction": float(count / rows.shape[0]),
        }
        if count:
            absolute_theta = np.abs(theta_x_mrad[selected])
            quantiles = np.quantile(absolute_theta, [0.68, 0.95, 0.99])
            record.update({
                "energy_mean_MeV": float(np.mean(energy_MeV[selected])),
                "energy_std_MeV": float(np.std(energy_MeV[selected])),
                "stopping_sum_MeV_per_mm": float(np.sum(stopping_power[selected])),
                "stopping_mean_MeV_per_mm": float(np.mean(stopping_power[selected])),
                "outward_theta_mean_mrad": float(np.mean(
                    outward_theta_mrad[selected])),
                "outward_theta_std_mrad": float(np.std(
                    outward_theta_mrad[selected])),
                "abs_theta_q68_mrad": float(quantiles[0]),
                "abs_theta_q95_mrad": float(quantiles[1]),
                "abs_theta_q99_mrad": float(quantiles[2]),
                "abs_theta_gt20_fraction": float(np.mean(
                    absolute_theta > 20.0)),
                "abs_theta_gt40_fraction": float(np.mean(
                    absolute_theta > 40.0)),
                "correlation_energy_outward_theta": corr(
                    energy_MeV[selected], outward_theta_mrad[selected]),
            })
        bins.append(record)
    return {"count": int(rows.shape[0]), "bins": bins}


def joint_histogram_comparison(
        topas: np.ndarray, gpu: np.ndarray, nominal_energy_MeVu: float) -> dict:
    """Coarse, yield-sensitive comparison of P(|x_fold|, E, theta_out)."""
    energy_edges = np.asarray(
        [-np.inf, 0.4, 0.7, 0.85, 0.93, 0.97, 1.01, np.inf])
    theta_edges_mrad = np.asarray(
        [-np.inf, -40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40, np.inf])

    def coordinates(rows: np.ndarray) -> np.ndarray:
        ux, uy, _ = directions(rows)
        x_mm = rows[:, 0] * 10.0
        residual = x_mm - np.rint(x_mm / SLIT_PITCH_MM) * SLIT_PITCH_MM
        outward_theta = np.sign(residual) * np.arctan2(ux, uy) * 1.0e3
        return np.column_stack((
            np.abs(residual),
            rows[:, 5] / (12.0 * nominal_energy_MeVu),
            outward_theta,
        ))

    histogram_edges = (SLIT_EDGE_BINS_MM, energy_edges, theta_edges_mrad)
    topas_hist, _ = np.histogramdd(coordinates(topas), bins=histogram_edges)
    gpu_hist, _ = np.histogramdd(coordinates(gpu), bins=histogram_edges)
    topas_total = float(np.sum(topas_hist))
    gpu_total = float(np.sum(gpu_hist))
    topas_shape = topas_hist / topas_total if topas_total else topas_hist
    gpu_shape = gpu_hist / gpu_total if gpu_total else gpu_hist
    return {
        "topas_count": int(topas_total),
        "gpu_count": int(gpu_total),
        "yield_l1_over_topas": (
            float(np.sum(np.abs(gpu_hist - topas_hist)) / topas_total)
            if topas_total else None),
        "shape_total_variation": (
            float(0.5 * np.sum(np.abs(gpu_shape - topas_shape)))
            if topas_total and gpu_total else None),
        "binning": {
            "abs_folded_x_mm": [float(value) for value in SLIT_EDGE_BINS_MM],
            "energy_over_nominal": [
                "-inf" if np.isneginf(value) else
                "+inf" if np.isposinf(value) else float(value)
                for value in energy_edges],
            "outward_theta_mrad": [
                "-inf" if np.isneginf(value) else
                "+inf" if np.isposinf(value) else float(value)
                for value in theta_edges_mrad],
        },
    }


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      probability: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 0:
        return float("nan")
    index = np.searchsorted(
        cumulative, probability * cumulative[-1], side="left")
    return float(ordered_values[min(index, ordered_values.size - 1)])


def conditional_bootstrap_comparison(
        topas: np.ndarray, gpu: np.ndarray,
        stopping: tuple[np.ndarray, np.ndarray], replicates: int,
        seed: int) -> list[dict]:
    """Independent Poisson bootstrap for sparse slit-edge conditional bins."""
    rng = np.random.default_rng(seed)

    def arrays(rows: np.ndarray) -> dict[str, np.ndarray]:
        ux, uy, _ = directions(rows)
        x_mm = rows[:, 0] * 10.0
        residual = x_mm - np.rint(x_mm / SLIT_PITCH_MM) * SLIT_PITCH_MM
        energy = rows[:, 5]
        return {
            "residual": np.abs(residual),
            "energy": energy,
            "stopping": np.interp(energy / 12.0, stopping[0], stopping[1]),
            "theta": np.abs(np.arctan2(ux, uy) * 1.0e3),
        }

    topas_arrays = arrays(topas)
    gpu_arrays = arrays(gpu)
    result = []
    for bin_index, (low, high) in enumerate(zip(
            SLIT_EDGE_BINS_MM[:-1], SLIT_EDGE_BINS_MM[1:])):
        def select(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            mask = ((source["residual"] >= low) &
                    ((source["residual"] < high) if bin_index + 2 < len(
                        SLIT_EDGE_BINS_MM) else (source["residual"] <= high)))
            return {key: value[mask] for key, value in source.items()}

        topas_bin = select(topas_arrays)
        gpu_bin = select(gpu_arrays)
        estimates = {key: [] for key in (
            "count_gpu_over_topas", "energy_mean_gpu_over_topas",
            "stopping_sum_gpu_over_topas", "q68_gpu_over_topas",
            "q99_gpu_over_topas", "tail20_gpu_over_topas",
            "tail40_gpu_over_topas")}
        for _ in range(replicates):
            topas_weights = rng.poisson(1.0, topas_bin["energy"].size)
            gpu_weights = rng.poisson(1.0, gpu_bin["energy"].size)
            topas_count = float(np.sum(topas_weights))
            gpu_count = float(np.sum(gpu_weights))
            if topas_count <= 0.0 or gpu_count <= 0.0:
                continue
            estimates["count_gpu_over_topas"].append(gpu_count / topas_count)
            topas_energy = float(np.sum(
                topas_weights * topas_bin["energy"]) / topas_count)
            gpu_energy = float(np.sum(
                gpu_weights * gpu_bin["energy"]) / gpu_count)
            estimates["energy_mean_gpu_over_topas"].append(
                gpu_energy / topas_energy)
            estimates["stopping_sum_gpu_over_topas"].append(
                float(np.sum(gpu_weights * gpu_bin["stopping"]) /
                      np.sum(topas_weights * topas_bin["stopping"])))
            for label, probability in (("q68", 0.68), ("q99", 0.99)):
                topas_quantile = weighted_quantile(
                    topas_bin["theta"], topas_weights, probability)
                gpu_quantile = weighted_quantile(
                    gpu_bin["theta"], gpu_weights, probability)
                estimates[f"{label}_gpu_over_topas"].append(
                    gpu_quantile / topas_quantile)
            for label, threshold in (("tail20", 20.0), ("tail40", 40.0)):
                topas_tail = float(np.sum(
                    topas_weights * (topas_bin["theta"] > threshold)))
                gpu_tail = float(np.sum(
                    gpu_weights * (gpu_bin["theta"] > threshold)))
                if topas_tail > 0.0:
                    estimates[f"{label}_gpu_over_topas"].append(
                        (gpu_tail / gpu_count) / (topas_tail / topas_count))
        record = {
            "abs_residual_low_mm": float(low),
            "abs_residual_high_mm": float(min(high, 1.8)),
            "topas_count": int(topas_bin["energy"].size),
            "gpu_count": int(gpu_bin["energy"].size),
        }
        for key, values in estimates.items():
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            record[key] = {
                "median": float(np.median(finite)) if finite.size else None,
                "ci95": ([float(value) for value in np.quantile(
                    finite, [0.025, 0.975])] if finite.size else None),
            }
        result.append(record)
    return result


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
    water = load_phsp(run_dir / "output/water_entrance.phsp")
    water_primary = primary_c12(water)
    entrance_path = run_dir / "output/collimator_entrance.phsp"
    classification_available = entrance_path.is_file()
    if classification_available:
        direct_ids = direct_event_ids(load_phsp(entrance_path))
        is_direct = np.fromiter(
            (int(event) in direct_ids for event in water_primary[:, 11]),
            dtype=bool,
            count=water_primary.shape[0],
        )
        direct = water_primary[is_direct]
        touched = water_primary[~is_direct]
    else:
        direct = water_primary[:0]
        touched = water_primary[:0]
    summary = {
        "run_dir": str(run_dir),
        "path_classification_available": classification_available,
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


def conditional_comparison_ratios(topas: dict, gpu: dict) -> list[dict]:
    def ratio(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return numerator / denominator

    result = []
    for topas_bin, gpu_bin in zip(topas["bins"], gpu["bins"]):
        record = {
            "abs_residual_low_mm": topas_bin["abs_residual_low_mm"],
            "abs_residual_high_mm": topas_bin["abs_residual_high_mm"],
            "topas_count": topas_bin["count"],
            "gpu_count": gpu_bin["count"],
            "count_gpu_over_topas": ratio(
                gpu_bin["count"], topas_bin["count"]),
            "fraction_gpu_over_topas": ratio(
                gpu_bin["fraction"], topas_bin["fraction"]),
        }
        if topas_bin["count"] and gpu_bin["count"]:
            for key in (
                    "energy_mean_MeV", "energy_std_MeV",
                    "stopping_sum_MeV_per_mm", "stopping_mean_MeV_per_mm",
                    "outward_theta_std_mrad", "abs_theta_q68_mrad",
                    "abs_theta_q95_mrad", "abs_theta_q99_mrad",
                    "abs_theta_gt20_fraction", "abs_theta_gt40_fraction"):
                record[f"{key}_gpu_over_topas"] = ratio(
                    gpu_bin[key], topas_bin[key])
            record["outward_theta_mean_mrad"] = {
                "topas": topas_bin["outward_theta_mean_mrad"],
                "gpu": gpu_bin["outward_theta_mean_mrad"],
                "difference": (gpu_bin["outward_theta_mean_mrad"] -
                               topas_bin["outward_theta_mean_mrad"]),
            }
            record["correlation_energy_outward_theta"] = {
                "topas": topas_bin["correlation_energy_outward_theta"],
                "gpu": gpu_bin["correlation_energy_outward_theta"],
            }
        result.append(record)
    return result


def plot_conditional_ratios(comparisons: dict[str, dict], output: Path) -> None:
    metrics = (
        ("fraction_gpu_over_topas", "fluence fraction"),
        ("energy_mean_MeV_gpu_over_topas", "mean energy"),
        ("stopping_sum_MeV_per_mm_gpu_over_topas", "stopping sum"),
        ("abs_theta_q68_mrad_gpu_over_topas", "|theta x| q68"),
        ("abs_theta_q99_mrad_gpu_over_topas", "|theta x| q99"),
    )
    fig, axes = plt.subplots(
        len(comparisons), len(metrics), figsize=(16, 3.3 * len(comparisons)),
        squeeze=False, constrained_layout=True)
    for row, (energy, comparison) in enumerate(sorted(
            comparisons.items(), key=lambda item: int(item[0]))):
        bins = comparison["water_primary_copper_touched_conditional"]
        centers = np.asarray([
            0.5 * (entry["abs_residual_low_mm"] +
                   entry["abs_residual_high_mm"])
            for entry in bins])
        for column, (key, title) in enumerate(metrics):
            values = np.asarray([
                entry.get(key, np.nan) for entry in bins], dtype=float)
            axis = axes[row, column]
            axis.plot(centers, values, marker="o")
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.axvline(SLIT_HALF_WIDTH_MM, color="0.5", linestyle=":")
            axis.set(xlabel="|folded x| at water entrance [mm]",
                     ylabel="GPU / TOPAS", title=f"{energy} MeV/u: {title}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


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
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be nonnegative")
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
        for group in ("all", "direct", "touched"):
            summaries[str(energy)][
                f"water_primary_{'copper_touched' if group == 'touched' else group}_conditional"
            ] = slit_edge_conditional_summary(samples[energy][group], stopping)
    with (args.output_dir / "phase_space_joint_summary.json").open("w") as stream:
        json.dump(summaries, stream, indent=2)
    plot_diagnostics(samples, args.output_dir / "phase_space_joint_diagnostics.png")
    if args.gpu_case:
        gpu_summaries = {}
        gpu_samples = {}
        for energy_text, path in args.gpu_case:
            energy = int(energy_text)
            rows, touched, _ = gpu_rows(Path(path))
            groups = {
                "all": rows,
                "direct": rows[~touched],
                "touched": rows[touched],
            }
            gpu_samples[energy] = groups
            summary = analyze_gpu_case(Path(path), stopping)
            for group in ("all", "direct", "touched"):
                summary[
                    f"water_primary_{'copper_touched' if group == 'touched' else group}_conditional"
                ] = slit_edge_conditional_summary(groups[group], stopping)
            gpu_summaries[str(energy)] = summary
        comparisons = {
            energy: comparison_ratios(summaries[energy], gpu)
            for energy, gpu in gpu_summaries.items()
        }
        for energy, comparison in comparisons.items():
            numeric_energy = int(energy)
            for group, label in (
                    ("all", "water_primary_all"),
                    ("direct", "water_primary_direct"),
                    ("touched", "water_primary_copper_touched")):
                conditional_key = f"{label}_conditional"
                comparison[conditional_key] = conditional_comparison_ratios(
                    summaries[energy][conditional_key],
                    gpu_summaries[energy][conditional_key])
                comparison[f"{label}_joint_histogram"] = (
                    joint_histogram_comparison(
                        samples[numeric_energy][group],
                        gpu_samples[numeric_energy][group], numeric_energy))
            if args.bootstrap_replicates:
                comparison["water_primary_copper_touched_conditional_bootstrap"] = (
                    conditional_bootstrap_comparison(
                        samples[numeric_energy]["touched"],
                        gpu_samples[numeric_energy]["touched"], stopping,
                        args.bootstrap_replicates,
                        args.bootstrap_seed + numeric_energy))
        with (args.output_dir / "gpu_phase_space_summary.json").open("w") as stream:
            json.dump(gpu_summaries, stream, indent=2)
        with (args.output_dir / "gpu_topas_phase_space_comparison.json").open("w") as stream:
            json.dump(comparisons, stream, indent=2)
        plot_conditional_ratios(
            comparisons, args.output_dir / "slit_edge_conditional_ratios.png")


if __name__ == "__main__":
    main()
