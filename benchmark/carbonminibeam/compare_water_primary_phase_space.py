#!/usr/bin/env python3
"""Compare aggregate TOPAS/GPU primary-C12 transport in water.

TOPAS phase-space replay EventID is not a stable source-row identifier in a
multi-threaded run.  Consequently this analysis deliberately does not join a
downstream TOPAS record to an entrance row.  It compares quantities that are
well-defined without that join: crossing survival, energy and angle
distributions, periodic (pitch-folded) position distributions, position-angle
covariance, Fourier modulation, and fixed-region fluence/energy.  No dose or
phase-space normalization is fitted.
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrance", type=Path, required=True)
    parser.add_argument("--topas-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--depths-mm", type=float, nargs="+", default=[40, 60, 80, 100, 120])
    parser.add_argument("--pitch-mm", type=float, default=3.6)
    parser.add_argument("--field-half-width-mm", type=float, default=18.0)
    parser.add_argument(
        "--topas-surface-offset-mm", type=float, default=-0.005,
        help="Actual scorer surface depth minus the depth used in its filename")
    parser.add_argument(
        "--stopping-power", type=Path,
        default=Path("data/stopping_power_water_geant4_11_3_2.csv"))
    parser.add_argument(
        "--energy-group-edges-MeVu", type=float, nargs="+",
        default=[0, 60, 120, 180, 240, 1000])
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=0,
        help="Track-block bootstrap replicates for interval q99/q99.9")
    parser.add_argument("--bootstrap-blocks", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    return parser.parse_args()


def direction_y(rows: np.ndarray) -> np.ndarray:
    value = np.sqrt(np.maximum(
        0.0, 1.0 - rows[:, 3] * rows[:, 3] - rows[:, 4] * rows[:, 4]))
    return np.where(rows[:, 8] != 0, -value, value)


def load_entrance(path: Path) -> dict[str, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    selected = (rows[:, 7] == C12_PDG) & (rows[:, 13] == 0)
    if not np.all(selected):
        raise ValueError("Entrance replay must contain parent-0 C12 only")
    return {
        "x_mm": rows[:, 0] * 10.0,
        "y_mm": rows[:, 2] * 10.0,
        "energy_MeV": rows[:, 5],
        "direction_x": rows[:, 3],
        "direction_y": direction_y(rows),
        "direction_z": rows[:, 4],
    }


def load_topas(path: Path) -> dict[str, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    rows = rows[(rows[:, 7] == C12_PDG) & (rows[:, 13] == 0)]
    return {
        "identity": np.rint(rows[:, 10:13]).astype(np.int64),
        "x_mm": rows[:, 0] * 10.0,
        "y_mm": rows[:, 2] * 10.0,
        "energy_MeV": rows[:, 5],
        "direction_x": rows[:, 3],
        "direction_y": direction_y(rows),
        "direction_z": rows[:, 4],
    }


def load_gpu(path: Path, source_count: int) -> dict[int, dict[str, np.ndarray]]:
    rows = np.atleast_2d(np.loadtxt(
        path, delimiter=",", skiprows=1, dtype=np.float64))
    source = np.rint(rows[:, 0]).astype(np.int64)
    planes = np.rint(rows[:, 1]).astype(np.int64)
    if np.any(source < 0) or np.any(source >= source_count):
        raise ValueError("GPU source_history does not map to entrance rows")
    result = {}
    for plane in np.unique(planes):
        selected = planes == plane
        result[int(plane)] = {
            "identity": source[selected, None],
            "depth_mm": rows[selected, 2],
            "x_mm": rows[selected, 4],
            "y_mm": rows[selected, 5],
            "energy_MeV": rows[selected, 3],
            "direction_x": rows[selected, 6],
            "direction_y": rows[selected, 7],
            "direction_z": rows[selected, 8],
        }
    return result


def variance(values: np.ndarray) -> float:
    return float(np.var(values)) if values.size else float("nan")


def quantiles(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: float("nan") for key in ("q68", "q95", "q99", "q999")}
    q = np.quantile(np.abs(values), [0.68, 0.95, 0.99, 0.999])
    return dict(zip(("q68", "q95", "q99", "q999"), map(float, q)))


def fold_to_pitch(values: np.ndarray, pitch_mm: float) -> np.ndarray:
    return (values + 0.5 * pitch_mm) % pitch_mm - 0.5 * pitch_mm


def summarize(records: dict[str, np.ndarray], input_count: int,
              pitch_mm: float, field_half_width_mm: float,
              stopping: tuple[np.ndarray, np.ndarray]) -> dict[str, object]:
    count = int(records["x_mm"].size)
    if count == 0:
        return {"input": input_count, "crossings": 0}
    x = records["x_mm"]
    y = records["y_mm"]
    energy = records["energy_MeV"]
    dx = records["direction_x"]
    dy = records["direction_y"]
    dz = records["direction_z"]
    theta_x = np.arctan2(dx, dz)
    theta_y = np.arctan2(dy, dz)
    theta_r = np.hypot(theta_x, theta_y)
    local_x = fold_to_pitch(x, pitch_mm)
    centered_x = local_x - local_x.mean()
    centered_theta = theta_x - theta_x.mean()
    abs_local = np.abs(local_x)
    in_field = np.abs(x) <= field_half_width_mm
    regions = {
        "peak": in_field & (abs_local < 0.25),
        "shoulder": in_field & (abs_local >= 0.25) & (abs_local < 0.9),
        "valley": in_field & (abs_local >= 0.9) & (abs_local <= 1.8),
    }
    stopping_power = np.interp(
        energy / 12.0, stopping[0], stopping[1],
        left=stopping[1][0], right=stopping[1][-1])
    weighted_crossing = stopping_power / np.maximum(np.abs(dz), 1.0e-6)
    region_metrics = {}
    for name, mask in regions.items():
        region_count = int(np.count_nonzero(mask))
        region_metrics[name] = {
            "crossings": region_count,
            "fraction_of_crossings": float(region_count / count),
            "fluence_per_input": float(region_count / input_count),
            "mean_energy_MeV": (
                float(np.mean(energy[mask])) if np.any(mask) else None),
            "std_energy_MeV": (
                float(np.std(energy[mask])) if np.any(mask) else None),
            "stopping_weighted_crossing_MeV_per_mm": (
                float(np.sum(weighted_crossing[mask])) if np.any(mask) else 0.0),
        }
    harmonic = np.mean(np.exp(2j * np.pi * x[in_field] / pitch_mm))
    return {
        "input": input_count,
        "crossings": count,
        "in_field_crossings": int(np.count_nonzero(in_field)),
        "outside_field_crossings": int(np.count_nonzero(~in_field)),
        "survival": float(count / input_count),
        "mean_energy_MeV": float(np.mean(energy)),
        "std_energy_MeV": float(np.std(energy)),
        "global_var_x_mm2": variance(x),
        "global_var_y_mm2": variance(y),
        "folded_var_x_mm2": variance(local_x),
        "var_theta_x_rad2": variance(theta_x),
        "var_theta_y_rad2": variance(theta_y),
        "folded_x_theta_x_cov_mm_rad": float(
            np.mean(centered_x * centered_theta)),
        "folded_x_quantiles_mm": quantiles(local_x),
        "abs_theta_x_quantiles_mrad": {
            key: value * 1000.0 for key, value in quantiles(theta_x).items()},
        "radial_angle_quantiles_mrad": {
            key: value * 1000.0 for key, value in quantiles(theta_r).items()},
        "pitch_harmonic_amplitude": float(np.abs(harmonic)),
        "pitch_harmonic_phase_rad": float(np.angle(harmonic)),
        "regions": region_metrics,
    }


def matched_indices(upstream: dict[str, np.ndarray],
                    downstream: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    def packed(keys: np.ndarray) -> np.ndarray:
        keys = np.ascontiguousarray(keys, dtype=np.int64)
        return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()

    up_key = packed(upstream["identity"])
    down_key = packed(downstream["identity"])
    if np.unique(up_key).size != up_key.size:
        raise ValueError("Upstream plane identities are not unique")
    if np.unique(down_key).size != down_key.size:
        raise ValueError("Downstream plane identities are not unique")
    _, up_index, down_index = np.intersect1d(
        up_key, down_key, assume_unique=True, return_indices=True)
    return up_index, down_index


def distribution(values: np.ndarray, scale: float = 1.0) -> dict[str, object]:
    probabilities = np.asarray(
        [0.50, 0.68, 0.80, 0.90, 0.95, 0.975, 0.99, 0.995, 0.999])
    absolute = np.abs(values) * scale
    centered = values - np.mean(values)
    return {
        "mean": float(np.mean(values) * scale),
        "variance": float(np.mean(centered * centered) * scale * scale),
        "absolute_quantiles": {
            str(probability): float(value)
            for probability, value in zip(
                probabilities, np.quantile(absolute, probabilities))
        },
    }


def interval_summary(upstream: dict[str, np.ndarray],
                     downstream: dict[str, np.ndarray],
                     upstream_depth_mm: float, downstream_depth_mm: float,
                     energy_edges_MeVu: np.ndarray) -> dict[str, object]:
    up_index, down_index = matched_indices(upstream, downstream)
    up_direction = np.column_stack((
        upstream["direction_x"][up_index],
        upstream["direction_y"][up_index],
        upstream["direction_z"][up_index]))
    down_direction = np.column_stack((
        downstream["direction_x"][down_index],
        downstream["direction_y"][down_index],
        downstream["direction_z"][down_index]))
    up_direction /= np.linalg.norm(up_direction, axis=1)[:, None]
    down_direction /= np.linalg.norm(down_direction, axis=1)[:, None]

    # Build a transverse basis around the upstream direction.  These beams are
    # forward-going, so projected global x is a stable first basis vector.
    basis_u = np.column_stack((
        1.0 - up_direction[:, 0] ** 2,
        -up_direction[:, 0] * up_direction[:, 1],
        -up_direction[:, 0] * up_direction[:, 2]))
    basis_u /= np.linalg.norm(basis_u, axis=1)[:, None]
    basis_v = np.cross(up_direction, basis_u)
    forward_projection = np.sum(down_direction * up_direction, axis=1)
    delta_theta_u = np.arctan2(
        np.sum(down_direction * basis_u, axis=1), forward_projection)
    delta_theta_v = np.arctan2(
        np.sum(down_direction * basis_v, axis=1), forward_projection)

    axial_distance = downstream_depth_mm - upstream_depth_mm
    path_to_plane = axial_distance / up_direction[:, 2]
    residual = np.column_stack((
        downstream["x_mm"][down_index] - upstream["x_mm"][up_index],
        downstream["y_mm"][down_index] - upstream["y_mm"][up_index],
        np.full(up_index.size, axial_distance))) - path_to_plane[:, None] * up_direction
    displacement_u = np.sum(residual * basis_u, axis=1)
    displacement_v = np.sum(residual * basis_v, axis=1)
    up_energy_MeVu = upstream["energy_MeV"] / 12.0
    matched_energy_MeVu = upstream["energy_MeV"][up_index] / 12.0

    def one_group(mask: np.ndarray, upstream_count: int) -> dict[str, object]:
        theta_u = delta_theta_u[mask]
        theta_v = delta_theta_v[mask]
        disp_u = displacement_u[mask]
        disp_v = displacement_v[mask]
        if theta_u.size == 0 or upstream_count == 0:
            return {"upstream": upstream_count, "paired": 0}
        theta_u_centered = theta_u - np.mean(theta_u)
        disp_u_centered = disp_u - np.mean(disp_u)
        return {
            "upstream": upstream_count,
            "paired": int(theta_u.size),
            "paired_survival": float(theta_u.size / upstream_count),
            "energy_in_MeVu_mean": float(np.mean(matched_energy_MeVu[mask])),
            "energy_out_MeVu_mean": float(np.mean(
                downstream["energy_MeV"][down_index[mask]] / 12.0)),
            "energy_gain_count": int(np.count_nonzero(
                downstream["energy_MeV"][down_index[mask]] >
                upstream["energy_MeV"][up_index[mask]] + 1.0e-6)),
            "delta_theta_u_mrad": distribution(theta_u, 1000.0),
            "delta_theta_v_mrad": distribution(theta_v, 1000.0),
            "delta_theta_radial_mrad": distribution(
                np.hypot(theta_u, theta_v), 1000.0),
            "displacement_u_mm": distribution(disp_u),
            "displacement_v_mm": distribution(disp_v),
            "cov_displacement_u_theta_u_mm_rad": float(
                np.mean(disp_u_centered * theta_u_centered)),
            "correlation_displacement_u_theta_u": (
                float(np.corrcoef(disp_u, theta_u)[0, 1])
                if theta_u.size > 1 and np.std(disp_u) > 0.0 and
                np.std(theta_u) > 0.0 else None),
        }

    groups = {}
    for low, high in zip(energy_edges_MeVu[:-1], energy_edges_MeVu[1:]):
        all_mask = (up_energy_MeVu >= low) & (up_energy_MeVu < high)
        paired_mask = ((matched_energy_MeVu >= low) &
                       (matched_energy_MeVu < high))
        groups[f"{low:g}_{high:g}_MeVu"] = one_group(
            paired_mask, int(np.count_nonzero(all_mask)))
    groups["all"] = one_group(
        np.ones(up_index.size, dtype=bool), upstream["x_mm"].size)
    return {
        "upstream_depth_mm": upstream_depth_mm,
        "downstream_depth_mm": downstream_depth_mm,
        "groups": groups,
    }


def identity_blocks(identity: np.ndarray, block_count: int) -> np.ndarray:
    """Stable track-level blocks shared by every downstream interval."""
    keys = np.asarray(identity, dtype=np.int64).astype(np.uint64, copy=False)
    value = np.full(keys.shape[0], np.uint64(0x9E3779B97F4A7C15))
    constants = (
        np.uint64(0xBF58476D1CE4E5B9),
        np.uint64(0x94D049BB133111EB),
        np.uint64(0xD6E8FEB86659FD93),
    )
    for column in range(keys.shape[1]):
        value ^= keys[:, column] + constants[column % len(constants)]
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return np.asarray(value % np.uint64(block_count), dtype=np.int64)


def bootstrap_quantile(values: np.ndarray, blocks: np.ndarray,
                       block_weights: np.ndarray,
                       probability: float) -> np.ndarray:
    order = np.argsort(values)
    ordered = values[order]
    ordered_blocks = blocks[order]
    estimates = np.empty(block_weights.shape[0], dtype=np.float64)
    for replicate, weights_by_block in enumerate(block_weights):
        weights = weights_by_block[ordered_blocks]
        cumulative = np.cumsum(weights)
        if cumulative[-1] == 0:
            estimates[replicate] = np.nan
            continue
        index = np.searchsorted(
            cumulative, probability * cumulative[-1], side="left")
        estimates[replicate] = ordered[min(index, ordered.size - 1)]
    return estimates


def confidence_interval(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(finite, [0.025, 0.975])]


def interval_uncertainty(
        upstream: dict[str, np.ndarray], downstream: dict[str, np.ndarray],
        energy_edges_MeVu: np.ndarray, block_weights: np.ndarray,
        block_count: int,
        fixed_thresholds: dict[str, dict[str, float]] | None = None,
        ) -> dict[str, object]:
    """Track-block bootstrap of tails, preserving identity across planes.

    The same bootstrap block weights are reused for every interval of one
    engine.  This retains cross-plane correlation without claiming that the
    later propagation intervals are statistically independent holdout samples.
    """
    up_index, down_index = matched_indices(upstream, downstream)
    up_direction = np.column_stack((
        upstream["direction_x"][up_index],
        upstream["direction_y"][up_index],
        upstream["direction_z"][up_index]))
    down_direction = np.column_stack((
        downstream["direction_x"][down_index],
        downstream["direction_y"][down_index],
        downstream["direction_z"][down_index]))
    up_direction /= np.linalg.norm(up_direction, axis=1)[:, None]
    down_direction /= np.linalg.norm(down_direction, axis=1)[:, None]
    basis_u = np.column_stack((
        1.0 - up_direction[:, 0] ** 2,
        -up_direction[:, 0] * up_direction[:, 1],
        -up_direction[:, 0] * up_direction[:, 2]))
    basis_u /= np.linalg.norm(basis_u, axis=1)[:, None]
    forward = np.sum(down_direction * up_direction, axis=1)
    absolute_angle_mrad = np.abs(np.arctan2(
        np.sum(down_direction * basis_u, axis=1), forward)) * 1000.0
    energy_MeVu = upstream["energy_MeV"][up_index] / 12.0
    blocks = identity_blocks(upstream["identity"][up_index], block_count)

    masks = {
        f"{low:g}_{high:g}_MeVu":
            (energy_MeVu >= low) & (energy_MeVu < high)
        for low, high in zip(energy_edges_MeVu[:-1], energy_edges_MeVu[1:])
    }
    masks["all"] = np.ones(up_index.size, dtype=bool)
    groups: dict[str, object] = {}
    for name, mask in masks.items():
        values = absolute_angle_mrad[mask]
        group_blocks = blocks[mask]
        if values.size == 0:
            groups[name] = {"paired": 0}
            continue
        quantile_result = {}
        observed_thresholds = {}
        for label, probability in (("q99", 0.99), ("q999", 0.999)):
            estimate = float(np.quantile(values, probability))
            observed_thresholds[label] = estimate
            replicates = bootstrap_quantile(
                values, group_blocks, block_weights, probability)
            quantile_result[label] = {
                "estimate_mrad": estimate,
                "ci95_mrad": confidence_interval(replicates),
            }

        thresholds = (fixed_thresholds or {}).get(
            name, observed_thresholds)
        threshold_result = {}
        total_by_block = np.bincount(
            group_blocks, minlength=block_count).astype(np.float64)
        bootstrap_totals = block_weights @ total_by_block
        for label in ("q99", "q999"):
            threshold = float(thresholds[label])
            exceed_by_block = np.bincount(
                group_blocks, weights=(values > threshold),
                minlength=block_count).astype(np.float64)
            probability_replicates = np.divide(
                block_weights @ exceed_by_block, bootstrap_totals,
                out=np.full(block_weights.shape[0], np.nan),
                where=bootstrap_totals > 0.0)
            threshold_result[label] = {
                "threshold_mrad": threshold,
                "probability": float(np.mean(values > threshold)),
                "ci95": confidence_interval(probability_replicates),
            }
        groups[name] = {
            "paired": int(values.size),
            "track_block_count": block_count,
            "quantiles": quantile_result,
            "fixed_threshold_exceedance": threshold_result,
        }
    return {"groups": groups}


def finite_ratio(left: float, right: float) -> float:
    return float(left / right) if np.isfinite(left) and right != 0 else float("nan")


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stopping_table = np.loadtxt(
        args.stopping_power, delimiter=",", comments="#", skiprows=2,
        dtype=np.float64)
    stopping = (stopping_table[:, 0], stopping_table[:, 1])
    entrance = load_entrance(args.entrance)
    source_count = entrance["x_mm"].size
    gpu_planes = load_gpu(args.gpu, source_count)
    entrance_summary = summarize(
        entrance, source_count, args.pitch_mm, args.field_half_width_mm,
        stopping)
    result: dict[str, object] = {
        "entrance_file": str(args.entrance),
        "source_count": int(source_count),
        "pitch_mm": args.pitch_mm,
        "field_half_width_mm": args.field_half_width_mm,
        "topas_surface_offset_mm": args.topas_surface_offset_mm,
        "stopping_power_file": str(args.stopping_power),
        "identity_note": (
            "TOPAS MT replay EventID is not an entrance-row ID. Aggregate "
            "summaries are identity-free; interval summaries join downstream "
            "planes by their stable (RunID,EventID,TrackID) tuple."),
        "entrance": entrance_summary,
        "depths": [],
        "intervals": [],
    }
    if args.bootstrap_replicates < 0 or args.bootstrap_blocks <= 1:
        raise ValueError("bootstrap replicates must be nonnegative and blocks > 1")
    bootstrap_weights = None
    if args.bootstrap_replicates:
        bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
        bootstrap_weights = bootstrap_rng.multinomial(
            args.bootstrap_blocks,
            np.full(args.bootstrap_blocks, 1.0 / args.bootstrap_blocks),
            size=args.bootstrap_replicates).astype(np.float64)
        result["interval_tail_uncertainty"] = {
            "method": "stable-identity track-block bootstrap",
            "replicates": args.bootstrap_replicates,
            "blocks": args.bootstrap_blocks,
            "seed": args.bootstrap_seed,
            "note": (
                "One engine reuses the same track-block weights at every "
                "plane, preserving cross-plane correlation; propagation "
                "intervals share tracks and are not independent samples."),
        }
    plot_values: dict[str, dict[str, list[float]]] = {
        engine: {key: [] for key in (
            "depth", "folded_var_x", "folded_cov_x_theta", "var_theta",
            "q95", "q99", "q999", "harmonic",
            "peak", "shoulder", "valley")}
        for engine in ("topas", "gpu")
    }
    entrance_all = entrance_summary
    for engine in ("topas", "gpu"):
        values = plot_values[engine]
        values["depth"].append(0.0)
        values["folded_var_x"].append(entrance_all["folded_var_x_mm2"])
        values["folded_cov_x_theta"].append(
            entrance_all["folded_x_theta_x_cov_mm_rad"])
        values["var_theta"].append(entrance_all["var_theta_x_rad2"])
        values["harmonic"].append(entrance_all["pitch_harmonic_amplitude"])
        angle_q = entrance_all["abs_theta_x_quantiles_mrad"]
        values["q95"].append(angle_q["q95"])
        values["q99"].append(angle_q["q99"])
        values["q999"].append(angle_q["q999"])
        for region in ("peak", "shoulder", "valley"):
            values[region].append(
                entrance_all["regions"][region]["fluence_per_input"])
    topas_planes: dict[int, dict[str, np.ndarray]] = {}
    for plane, depth in enumerate(args.depths_mm):
        label = f"{int(round(depth)):03d}"
        topas = load_topas(
            args.topas_dir / "output" / f"primary_{label}.phsp")
        topas_planes[plane] = topas
        if plane not in gpu_planes:
            raise ValueError(f"GPU plane index {plane} is missing")
        gpu = gpu_planes[plane]
        record = {
            "depth_mm": depth,
            "topas_surface_depth_mm": depth + args.topas_surface_offset_mm,
            "gpu_plane_depth_mm": float(np.median(gpu["depth_mm"])),
            "topas": summarize(
                topas, source_count, args.pitch_mm,
                args.field_half_width_mm, stopping),
            "gpu": summarize(
                gpu, source_count, args.pitch_mm,
                args.field_half_width_mm, stopping),
        }
        result["depths"].append(record)
        for engine, summary in (
                ("topas", record["topas"]), ("gpu", record["gpu"])):
            values = plot_values[engine]
            values["depth"].append(depth)
            values["folded_var_x"].append(summary["folded_var_x_mm2"])
            values["folded_cov_x_theta"].append(
                summary["folded_x_theta_x_cov_mm_rad"])
            values["var_theta"].append(summary["var_theta_x_rad2"])
            values["harmonic"].append(summary["pitch_harmonic_amplitude"])
            angle_q = summary["abs_theta_x_quantiles_mrad"]
            values["q95"].append(angle_q["q95"])
            values["q99"].append(angle_q["q99"])
            values["q999"].append(angle_q["q999"])
            for region in ("peak", "shoulder", "valley"):
                values[region].append(
                    summary["regions"][region]["fluence_per_input"])

    energy_edges = np.asarray(args.energy_group_edges_MeVu, dtype=np.float64)
    if energy_edges.size < 2 or np.any(np.diff(energy_edges) <= 0.0):
        raise ValueError("Energy group edges must be strictly increasing")
    for plane in range(len(args.depths_mm) - 1):
        nominal_up = args.depths_mm[plane]
        nominal_down = args.depths_mm[plane + 1]
        interval_record = {
            "nominal_interval_mm": [nominal_up, nominal_down],
            "topas": interval_summary(
                topas_planes[plane], topas_planes[plane + 1],
                nominal_up + args.topas_surface_offset_mm,
                nominal_down + args.topas_surface_offset_mm, energy_edges),
            "gpu": interval_summary(
                gpu_planes[plane], gpu_planes[plane + 1],
                float(np.median(gpu_planes[plane]["depth_mm"])),
                float(np.median(gpu_planes[plane + 1]["depth_mm"])),
                energy_edges),
        }
        if bootstrap_weights is not None:
            topas_uncertainty = interval_uncertainty(
                topas_planes[plane], topas_planes[plane + 1], energy_edges,
                bootstrap_weights, args.bootstrap_blocks)
            topas_thresholds = {
                name: {
                    label: metrics["estimate_mrad"]
                    for label, metrics in group["quantiles"].items()
                }
                for name, group in topas_uncertainty["groups"].items()
                if group.get("paired", 0)
            }
            gpu_uncertainty = interval_uncertainty(
                gpu_planes[plane], gpu_planes[plane + 1], energy_edges,
                bootstrap_weights, args.bootstrap_blocks, topas_thresholds)
            interval_record["tail_uncertainty"] = {
                "topas": topas_uncertainty,
                "gpu_at_topas_thresholds": gpu_uncertainty,
            }
        result["intervals"].append(interval_record)

    output_json = args.output_dir / "water_primary_phase_space_metrics.json"
    output_json.write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, key, title in zip(
            axes.ravel(), ("folded_var_x", "folded_cov_x_theta", "var_theta",
                           "q95", "q99", "q999"),
            ("Var(folded x)", "Cov(folded x, theta x)", "Var(theta x)",
             "|theta x| q95", "|theta x| q99", "|theta x| q99.9")):
        for engine in ("topas", "gpu"):
            axis.plot(plot_values[engine]["depth"], plot_values[engine][key],
                      marker="o", label=engine.upper())
        axis.set(xlabel="water depth (mm)", title=title)
    axes[0, 0].legend()
    fig.savefig(args.output_dir / "water_primary_moments.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    ratio_keys = ("folded_var_x", "folded_cov_x_theta", "var_theta",
                  "q95", "q99", "q999", "harmonic", "peak", "shoulder",
                  "valley")
    for axis, key in zip(axes.ravel(), ratio_keys):
        topas = np.asarray(plot_values["topas"][key])
        gpu = np.asarray(plot_values["gpu"][key])
        ratio = np.array([finite_ratio(g, t) for g, t in zip(gpu, topas)])
        axis.plot(plot_values["gpu"]["depth"], ratio, marker="o")
        axis.axhline(1.0, color="black", linestyle="--", lw=0.8)
        axis.set(xlabel="water depth (mm)", ylabel="GPU / TOPAS", title=key)
    fig.savefig(args.output_dir / "water_primary_phase_space_ratios.png", dpi=180)
    plt.close(fig)

    interval_midpoints = []
    interval_ratios = {key: [] for key in (
        "angle_variance", "displacement_variance", "covariance",
        "q68", "q95", "q99", "q999")}
    for interval in result["intervals"]:
        interval_midpoints.append(float(np.mean(interval["nominal_interval_mm"])))
        topas = interval["topas"]["groups"]["all"]
        gpu = interval["gpu"]["groups"]["all"]
        interval_ratios["angle_variance"].append(finite_ratio(
            gpu["delta_theta_u_mrad"]["variance"],
            topas["delta_theta_u_mrad"]["variance"]))
        interval_ratios["displacement_variance"].append(finite_ratio(
            gpu["displacement_u_mm"]["variance"],
            topas["displacement_u_mm"]["variance"]))
        interval_ratios["covariance"].append(finite_ratio(
            gpu["cov_displacement_u_theta_u_mm_rad"],
            topas["cov_displacement_u_theta_u_mm_rad"]))
        for key, probability in (("q68", "0.68"), ("q95", "0.95"),
                                 ("q99", "0.99"), ("q999", "0.999")):
            interval_ratios[key].append(finite_ratio(
                gpu["delta_theta_u_mrad"]["absolute_quantiles"][probability],
                topas["delta_theta_u_mrad"]["absolute_quantiles"][probability]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key in ("angle_variance", "displacement_variance", "covariance"):
        axes[0].plot(interval_midpoints, interval_ratios[key], marker="o",
                     label=key.replace("_", " "))
    for key in ("q68", "q95", "q99", "q999"):
        axes[1].plot(interval_midpoints, interval_ratios[key], marker="o",
                     label=key)
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", lw=0.8)
        axis.set(xlabel="interval midpoint depth (mm)",
                 ylabel="GPU / TOPAS")
        axis.legend()
    axes[0].set_title("interval joint moments")
    axes[1].set_title("interval |angle increment| quantiles")
    fig.savefig(args.output_dir / "water_interval_scattering_ratios.png", dpi=180)
    plt.close(fig)
    print(output_json)


if __name__ == "__main__":
    main()
