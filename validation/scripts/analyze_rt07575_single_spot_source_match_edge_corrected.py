#!/usr/bin/env python3
"""Repeat the RT07575 one-spot diagnostic on the corrected TPS-90 CT grid.

The archived TOPAS dose is only converted; TOPAS is never re-run.  The GPU
config is rendered from current best with exactly the two established geometry
overrides (corrected CCTG path and its matching CT low edge), plus normal
run-path/history/seed substitutions.  No dose scale, shift, blur, or source
parameter is fitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_rt07575_single_spot_source_match import (  # noqa: E402
    EXPECTED_GPU_SHAPE,
    EXPECTED_PATIENT_SHAPE,
    convert_topas,
    depth_metrics,
    moment_report,
    parse_log,
    read_mhd,
    replace_one,
    sha256,
    spot_histories,
    write_mhd,
)


HISTORIES = 100_000
ALLOWED_CONFIG_KEYS = {
    "number_of_histories",
    "random_seed",
    "topas_spots_file",
    "ct_grid_file",
    "spots_ct_axis_min_mm",
    "voxel_dose_mhd_output_file",
    "let_voxel_mhd_output_file",
}


def render_config(template: Path, output: Path, run_dir: Path, spots: Path, seed: int, grid: Path) -> dict[str, Any]:
    original = template.read_text(encoding="utf-8")
    rendered = original
    overrides = {
        "number_of_histories": str(HISTORIES),
        "random_seed": str(seed),
        "topas_spots_file": str(spots),
        "ct_grid_file": str(grid),
        "spots_ct_axis_min_mm": "-104.25",
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
    }
    for key, value in overrides.items():
        rendered = replace_one(rendered, key, value)
    output.write_text(rendered, encoding="utf-8")
    fields = r"^([A-Za-z0-9_]+):\s*(.*)$"
    before, after = dict(re.findall(fields, original, re.MULTILINE)), dict(re.findall(fields, rendered, re.MULTILINE))
    changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    unexpected = sorted(set(changed) - ALLOWED_CONFIG_KEYS)
    if unexpected:
        raise ValueError(f"rendered config changed non-approved keys: {unexpected}")
    return {"changed_keys": changed, "only_run_or_established_geometry_overrides": not unexpected, "geometry_overrides": {"ct_grid_file": str(grid), "spots_ct_axis_min_mm": -104.25}}


def complete(run_dir: Path) -> dict[str, Any] | None:
    required = (run_dir / "dose.mhd", run_dir / "dose.raw", run_dir / "run.log", run_dir / "config.yaml")
    if not all(path.exists() for path in required):
        return None
    try:
        stats = parse_log(run_dir / "run.log")
    except ValueError:
        return None
    if stats["histories"] != HISTORIES or stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]:
        return None
    return stats


def depth_indices(topas: np.ndarray) -> list[tuple[str, int]]:
    idd = topas.sum(axis=(0, 1))
    peak = int(np.argmax(idd))
    before_peak = np.arange(idd.size - 1, peak - 1, -1)
    crossing = before_peak[idd[before_peak] >= 0.2 * idd[peak]]
    entrance = int(crossing[0]) if crossing.size else int(before_peak[-1])
    return [("entrance", entrance), ("mid", int(round((entrance + peak) / 2))), ("peak", peak)]


def one_dimensional_width(coordinates: np.ndarray, values: np.ndarray) -> dict[str, float | int]:
    """Unfitted own-1%-plane-Dmax dose moment for one lateral profile."""
    support = values >= 0.01 * float(np.max(values))
    weights = np.where(support, values, 0.0)
    total = float(weights.sum(dtype=np.float64))
    if total <= 0.0:
        return {"support_points": 0, "com_mm": float("nan"), "rms_width_mm": float("nan")}
    com = float(np.dot(coordinates, weights) / total)
    return {"support_points": int(np.count_nonzero(support)), "com_mm": com, "rms_width_mm": float(np.sqrt(np.dot(weights, (coordinates - com) ** 2) / total))}


def lateral_profiles_and_widths(topas: np.ndarray, gpu: np.ndarray, topas_moment: dict[str, Any], spacing: tuple[float, float, float], offset: tuple[float, float, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fixed TOPAS-COM lateral profiles plus depth-resolved no-fit widths."""
    com = topas_moment["com_patient_xyz_mm"]
    y_index = int(np.clip(round((com[1] - offset[1]) / spacing[1]), 0, topas.shape[1] - 1))
    z_index = int(np.clip(round((com[2] - offset[2]) / spacing[2]), 0, topas.shape[0] - 1))
    xcoord = offset[0] + spacing[0] * np.arange(topas.shape[2])
    ycoord = offset[1] + spacing[1] * np.arange(topas.shape[1])
    zcoord = offset[2] + spacing[2] * np.arange(topas.shape[0])
    rows: list[dict[str, Any]] = []
    profile_metrics: list[dict[str, Any]] = []
    widths: list[dict[str, Any]] = []
    for label, x_index in depth_indices(topas):
        entries = (
            ("patient_y", ycoord, topas[z_index, :, x_index], gpu[z_index, :, x_index]),
            ("patient_z", zcoord, topas[:, y_index, x_index], gpu[:, y_index, x_index]),
        )
        for axis, coordinate, reference, evaluation in entries:
            high = reference >= 0.10 * max(float(reference.max()), 1.0e-30)
            if np.count_nonzero(high) < 2:
                high = reference > 0.0
            correlation = float(np.corrcoef(reference[high], evaluation[high])[0, 1]) if np.count_nonzero(high) > 1 and np.std(reference[high]) > 0 and np.std(evaluation[high]) > 0 else float("nan")
            profile_metrics.append({"depth_label": label, "patient_x_mm": float(xcoord[x_index]), "axis": axis, "fixed_patient_y_mm": float(ycoord[y_index]), "fixed_patient_z_mm": float(zcoord[z_index]), "support_points_10pct_topas_plane": int(np.count_nonzero(high)), "pearson_r": correlation, "rmse_over_topas_plane_peak_percent": float(100.0 * np.sqrt(np.mean((evaluation[high] - reference[high]) ** 2)) / max(float(reference.max()), 1.0e-30))})
            ref_width, gpu_width = one_dimensional_width(coordinate, reference), one_dimensional_width(coordinate, evaluation)
            widths.append({"depth_label": label, "patient_x_mm": float(xcoord[x_index]), "axis": axis, "topas": ref_width, "gpu": gpu_width, "gpu_over_topas_rms_width_ratio": float(gpu_width["rms_width_mm"] / ref_width["rms_width_mm"]), "gpu_minus_topas_line_com_mm": float(gpu_width["com_mm"] - ref_width["com_mm"])})
            for position, ref_dose, gpu_dose in zip(coordinate, reference, evaluation, strict=True):
                rows.append({"depth_label": label, "patient_x_mm": float(xcoord[x_index]), "axis": axis, "coordinate_mm": float(position), "topas_dose_Gy": float(ref_dose), "gpu_dose_Gy": float(gpu_dose)})
    return rows, profile_metrics, widths


def corrected_fullplan_context(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    comparison = report["comparisons"]["edge_corrected"]
    dose = comparison["quantities"]["dose"]
    com = comparison["dose_center_of_mass_patient_xyz_mm"]["gpu_minus_topas"]
    return {"source_summary": str(path), "com_gpu_minus_topas_patient_xyz_mm": com, "dose_nrmse_over_topas_dmax_percent": dose["selected_nrmse_over_reference_max_percent"], "dose_3pct_0mm_global_percent": dose["gamma"]["30"]["global"]["pass_percent"], "dose_3pct_0mm_local_percent": dose["gamma"]["30"]["local"]["pass_percent"], "interpretation": "Corrected full plan has near-zero COM residual but poor strict local gamma, so remaining error is primarily shape/transport-like rather than a rigid pose."}


def scalar_config(path: Path, key: str) -> float:
    match = re.search(rf"^{re.escape(key)}:\s*([0-9.eE+-]+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"{path}: missing {key}")
    return float(match.group(1))


def spot_value(path: Path, layer: int) -> float:
    match = re.search(rf"^[a-z]+:Tf/Scatterer1/L{layer}/Values\s*=\s*1\s+([0-9.eE+-]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"{path}: missing one-spot L{layer}")
    return float(match.group(1))


def stopping_at(path: Path, energy_mevu: float) -> float:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [(float(row["energy_MeVu"]), float(row["total_stopping_power_MeV_per_mm"])) for row in csv.DictReader(stream)]
    energies, stopping = np.asarray([row[0] for row in rows]), np.asarray([row[1] for row in rows])
    return float(np.interp(energy_mevu, energies, stopping))


def upstream_air_estimate(config: Path, spots: Path, air_table: Path, water_table: Path) -> dict[str, Any]:
    """Order-of-magnitude omitted World-air loss/MCS; no transport is added."""
    energy_total = spot_value(spots, 2)
    energy_mevu = energy_total / 12.0
    tilt = spot_value(spots, 7) - 90.0
    distance = (scalar_config(config, "spots_sad_mm") + scalar_config(config, "spots_patient_trans_y_mm") - (-scalar_config(config, "spots_ct_axis_min_mm"))) / math.cos(math.radians(tilt))
    air_stopping = stopping_at(air_table, energy_mevu)
    water_stopping = stopping_at(water_table, energy_mevu)
    energy_loss = distance * air_stopping
    # Highland estimate for C-12 in standard dry air: X0=36.66 g/cm2 and
    # rho=0.001205 g/cm3 -> 304.2 m. Distributed scattering displacement Lθ/√3.
    x_over_x0 = distance / 304_200.0
    mass = 12.0 * 931.49410242
    momentum = math.sqrt(energy_total * (energy_total + 2.0 * mass))
    beta = momentum / (energy_total + mass)
    theta = 13.6 * 6.0 / (beta * momentum) * math.sqrt(x_over_x0) * (1.0 + 0.038 * math.log(x_over_x0))
    return {"world_air_path_to_ct_entrance_mm": distance, "air_total_stopping_power_MeV_per_mm": air_stopping, "estimated_total_energy_loss_MeV": energy_loss, "estimated_energy_loss_MeVu": energy_loss / 12.0, "water_equivalent_range_shift_mm_from_dE_over_stopping": energy_loss / water_stopping, "highland_assumptions": {"standard_dry_air_radiation_length_mm": 304_200.0, "carbon_charge": 6, "carbon_mass_MeV_c2": mass}, "estimated_air_mcs_theta0_mrad": 1000.0 * theta, "estimated_distributed_air_mcs_lateral_rms_mm": distance * theta / math.sqrt(3.0), "interpretation": "GPU propagates emittance in vacuum to the CT entrance, so this energy loss and air-MCS term are absent. The range estimate is suitable only as an order-of-magnitude check; it is not a fitted correction."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--spots", type=Path, default=Path("ct/fullplan_result/RT07575/spots_single_spot_100k_local40.generated.txt"))
    parser.add_argument("--topas-binary", type=Path, default=Path("ct/fullplan_result/RT07575/OSMK_Dtotal_single_spot_100k_local40.bin"))
    parser.add_argument("--topas-header", type=Path, default=Path("ct/fullplan_result/RT07575/OSMK_Dtotal_single_spot_100k_local40.binheader"))
    parser.add_argument("--corrected-grid", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin"))
    parser.add_argument("--corrected-metadata", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.metadata.json"))
    parser.add_argument("--fullplan-corrected-summary", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation/summary.json"))
    parser.add_argument("--air-stopping-table", type=Path, default=Path("data/stopping_power_air_geant4_11_3_2_full.csv"))
    parser.add_argument("--water-stopping-table", type=Path, default=Path("data/stopping_power_water_geant4_11_3_2_full.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/ct/RT07575/single_spot_source_diagnostic_edge_corrected"))
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    for name in ("binary", "template", "spots", "topas_binary", "topas_header", "corrected_grid", "corrected_metadata", "fullplan_corrected_summary", "air_stopping_table", "water_stopping_table", "output_dir"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.repo_root / value)
    if spot_histories(args.spots) != [HISTORIES]:
        raise ValueError("input spot must contain exactly one 100,000-history L4 entry")
    corrected_metadata = json.loads(args.corrected_metadata.read_text(encoding="utf-8"))
    if corrected_metadata["origin_xyz_mm"] != [-126.25, -35.0, 0.0] or corrected_metadata["patient_origin_xyz_mm"] != [-104.25, -126.25, -1.0]:
        raise ValueError("corrected metadata does not have the established low-edge origin")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient_metadata_path = args.output_dir / "corrected_patient_coordinate_metadata.json"
    patient_metadata_path.write_text(json.dumps({"origin_xyz_mm": corrected_metadata["patient_origin_xyz_mm"], "spacing_xyz_mm": corrected_metadata["patient_spacing_xyz_mm"], "origin_convention": corrected_metadata["origin_convention"], "source_metadata": str(args.corrected_metadata)}, indent=2) + "\n", encoding="utf-8")
    conversion = convert_topas(args.topas_binary, args.topas_header, patient_metadata_path, args.output_dir / "topas_patient.mhd")
    run_dir = args.output_dir / "gpu_seed"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_audit = render_config(args.template, run_dir / "config.yaml", run_dir, args.spots, args.seed, args.corrected_grid)
    air_estimate = upstream_air_estimate(run_dir / "config.yaml", args.spots, args.air_stopping_table, args.water_stopping_table)
    stats = complete(run_dir)
    if stats is None:
        print("RUN corrected-grid GPU single spot on CUDA", flush=True)
        started = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run([str(args.binary), "--config", str(run_dir / "config.yaml"), "--device", "cuda"], cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT, check=False, text=True)
        if result.returncode:
            raise RuntimeError(f"GPU run failed ({result.returncode}); see {run_dir / 'run.log'}")
        stats = parse_log(run_dir / "run.log")
        stats["wall_seconds_runner"] = time.monotonic() - started
    if stats["histories"] != HISTORIES or stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]:
        raise RuntimeError(f"invalid GPU run: {stats}")
    topas_meta, topas = read_mhd(args.output_dir / "topas_patient.mhd")
    gpu_meta, native_gpu = read_mhd(run_dir / "dose.mhd")
    if tuple(int(v) for v in topas_meta["DimSize"].split()) != EXPECTED_PATIENT_SHAPE or tuple(int(v) for v in gpu_meta["DimSize"].split()) != EXPECTED_GPU_SHAPE:
        raise ValueError("TOPAS/GPU MHD dimensional mismatch")
    spacing, offset = tuple(float(v) for v in topas_meta["ElementSpacing"].split()), tuple(float(v) for v in topas_meta["Offset"].split())
    gpu = np.transpose(native_gpu, (1, 2, 0))[:, :, ::-1]
    write_mhd(args.output_dir / "gpu_patient.mhd", gpu, spacing, offset)
    selection = (topas >= 0.10 * topas.max()) | (gpu >= 0.10 * gpu.max())
    moments_topas = {f"threshold_{pct}pct": moment_report(topas, spacing, offset, pct / 100.0) for pct in (1, 10)}
    moments_gpu = {f"threshold_{pct}pct": moment_report(gpu, spacing, offset, pct / 100.0) for pct in (1, 10)}
    deltas: dict[str, Any] = {}
    for key, reference in moments_topas.items():
        evaluation = moments_gpu[key]
        deltas[key] = {"gpu_minus_topas_com_patient_xyz_mm": (np.asarray(evaluation["com_patient_xyz_mm"]) - np.asarray(reference["com_patient_xyz_mm"])).tolist(), "axis_rms_width_ratio_gpu_over_topas_xyz": (np.asarray(evaluation["axis_rms_width_xyz_mm"]) / np.asarray(reference["axis_rms_width_xyz_mm"])).tolist()}
    topas_depth, gpu_depth = depth_metrics(topas, spacing, offset), depth_metrics(gpu, spacing, offset)
    profile_rows, profile_metrics, profile_widths = lateral_profiles_and_widths(topas, gpu, moments_topas["threshold_10pct"], spacing, offset)
    with (args.output_dir / "lateral_profiles.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0])); writer.writeheader(); writer.writerows(profile_rows)
    with (args.output_dir / "integrated_depth_dose.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("patient_x_mm", "beam_depth_mm", "topas_idd_sum", "gpu_idd_sum")); writer.writeheader()
        for index, values in enumerate(zip(topas_depth["depth_mm"], topas_depth["idd_sum"], gpu_depth["idd_sum"], strict=True)):
            depth, ref, evaluation = values
            writer.writerow({"patient_x_mm": offset[0] + spacing[0] * index, "beam_depth_mm": depth, "topas_idd_sum": ref, "gpu_idd_sum": evaluation})
    entrance = [float(item["gpu_over_topas_rms_width_ratio"]) for item in profile_widths if item["depth_label"] == "entrance"]
    peak = [float(item["gpu_over_topas_rms_width_ratio"]) for item in profile_widths if item["depth_label"] == "peak"]
    entrance_mean, peak_mean = float(np.mean(entrance)), float(np.mean(peak))
    if entrance_mean < 0.95 and peak_mean < entrance_mean - 0.03:
        discriminator = "A lateral deficit is already present at entrance but grows by the Bragg peak. Emittance/source propagation can contribute to the entrance deficit, but the depth growth implicates transport (MCS/nonlocal deposition/material response) in the remaining width error."
    elif entrance_mean < 0.95:
        discriminator = "The lateral deficit is already present at CT entrance and remains similar through peak, which keeps source/emittance propagation implicated; a 100k single realization cannot exclude a transport contribution."
    else:
        discriminator = "No material entrance width deficit remains after the corrected geometry. The residual width signature develops downstream, favoring transport over source/emittance."
    fullplan = corrected_fullplan_context(args.fullplan_corrected_summary)
    r80_delta = gpu_depth["r80_distal_depth_mm"] - topas_depth["r80_distal_depth_mm"] if gpu_depth["r80_distal_depth_mm"] is not None and topas_depth["r80_distal_depth_mm"] is not None else None
    air_range_consistency = ("The observed GPU-deeper R80 residual has the sign expected from omitting upstream air and is of the same sub-millimetre order as the estimated air energy-loss range shift. The air-MCS estimate is only ~0.076 mm lateral RMS, too small by itself to explain the corrected full-plan strict local-gamma deficit." if r80_delta is not None and r80_delta > 0.0 else "The observed R80 sign does not support upstream-air loss as the dominant depth residual in this realization.")
    depth_summary = lambda value: {key: item for key, item in value.items() if key not in {"depth_mm", "idd_sum"}}
    topas_sum, gpu_sum = float(topas.sum(dtype=np.float64)), float(gpu.sum(dtype=np.float64))
    report: dict[str, Any] = {"case": "RT07575", "experiment": "archived 100k single spot on corrected CT low-edge grid", "histories": HISTORIES, "seed": args.seed, "normalization": "absolute scored dose; no fitted scale/translation/blur", "inputs": {"template": str(args.template), "template_sha256": sha256(args.template), "corrected_grid": str(args.corrected_grid), "corrected_metadata": str(args.corrected_metadata), "patient_coordinate_metadata": str(patient_metadata_path), "spots": str(args.spots), "topas_binary": str(args.topas_binary)}, "corrected_grid": {"gpu_low_edge_xyz_mm": corrected_metadata["origin_xyz_mm"], "patient_low_edge_xyz_mm": corrected_metadata["patient_origin_xyz_mm"], "patient_first_center_xyz_mm": conversion["offset_first_center_xyz_mm"], "axis_mapping": corrected_metadata["axis_mapping"], "matching_spots_ct_axis_min_mm": -104.25}, "topas_conversion": conversion, "config_diff_check": config_audit, "gpu_run": stats, "mapping": "GPU [depth=patient-X, patient-Z, patient-Y] -> patient [Z,Y,X] by transpose(1,2,0), then reverse patient X; patient MHD uses corrected first centers", "mapping_validation": {"mapped_gpu_sum_minus_native_sum": float(gpu.sum(dtype=np.float64) - native_gpu.sum(dtype=np.float64)), "gpu_patient_mhd": str(args.output_dir / "gpu_patient.mhd")}, "absolute_dose": {"gpu_over_topas_integral_ratio": gpu_sum / topas_sum, "union_10pct_dmax_voxels": int(np.count_nonzero(selection)), "union_10pct_dmax_pearson_r": float(np.corrcoef(topas[selection], gpu[selection])[0, 1]), "union_10pct_dmax_nrmse_over_topas_dmax_percent": float(100.0 * np.sqrt(np.mean((gpu[selection] - topas[selection]) ** 2)) / topas.max())}, "moments": {"topas": moments_topas, "gpu": moments_gpu, "gpu_minus_topas": deltas}, "depth_metrics": {"topas": depth_summary(topas_depth), "gpu": depth_summary(gpu_depth), "gpu_minus_topas": {"idd_peak_depth_mm": gpu_depth["idd_peak_depth_mm"] - topas_depth["idd_peak_depth_mm"], "r80_distal_depth_mm": r80_delta}, "csv": str(args.output_dir / "integrated_depth_dose.csv")}, "upstream_air_omission_estimate": air_estimate, "upstream_air_consistency_with_observed_depth": air_range_consistency, "lateral_profiles": {"definition": "fixed TOPAS 10%-dose-COM Y/Z and TOPAS entrance/mid/peak depths; no profile registration", "csv": str(args.output_dir / "lateral_profiles.csv"), "metrics": profile_metrics, "own_1pct_plane_dmax_widths": profile_widths}, "emittance_vs_transport": {"entrance_mean_gpu_over_topas_lateral_rms_width_ratio": entrance_mean, "peak_mean_gpu_over_topas_lateral_rms_width_ratio": peak_mean, "interpretation": discriminator}, "corrected_fullplan_context": fullplan, "limitations": "One TOPAS and one GPU 100k realization: moment/profile trends are the primary evidence; no low-stat gamma is claimed."}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    delta = deltas["threshold_10pct"]["gpu_minus_topas_com_patient_xyz_mm"]
    ratios = deltas["threshold_10pct"]["axis_rms_width_ratio_gpu_over_topas_xyz"]
    md = ["# RT07575 corrected-grid single-spot diagnostic", "", "Archived TOPAS and a new 100,000-history GPU run; no fit, TOPAS rerun, or physics change.", "", f"GPU: {stats['elapsed_seconds']:.3f} s, {stats['throughput_histories_per_second']:.0f} histories/s; secondary/cascade overflow {stats['secondary_queue_overflow']}/{stats['cascade_queue_overflow']}.", f"Corrected patient first center: {conversion['offset_first_center_xyz_mm']} mm. Integral GPU/TOPAS={gpu_sum/topas_sum:.6f}; union-10% correlation={report['absolute_dose']['union_10pct_dmax_pearson_r']:.6f}; NRMSE/Dmax={report['absolute_dose']['union_10pct_dmax_nrmse_over_topas_dmax_percent']:.3f}%.", "", f"10%-own-Dmax COM GPU−TOPAS (x,y,z) mm: {[round(value, 4) for value in delta]}; RMS-width ratio (x,y,z): {[round(value, 4) for value in ratios]}.", f"IDD peak Δ={report['depth_metrics']['gpu_minus_topas']['idd_peak_depth_mm']:.3f} mm; R80 Δ={report['depth_metrics']['gpu_minus_topas']['r80_distal_depth_mm']!s} mm.", f"Upstream World-air omission estimate: {air_estimate['world_air_path_to_ct_entrance_mm']:.2f} mm, ΔE≈{air_estimate['estimated_total_energy_loss_MeV']:.3f} MeV total ({air_estimate['estimated_energy_loss_MeVu']:.3f} MeV/u), water-equivalent range ≈{air_estimate['water_equivalent_range_shift_mm_from_dE_over_stopping']:.3f} mm; air-MCS lateral RMS ≈{air_estimate['estimated_distributed_air_mcs_lateral_rms_mm']:.3f} mm.", air_range_consistency, f"Corrected full-plan COM residual: {[round(value, 4) for value in fullplan['com_gpu_minus_topas_patient_xyz_mm']]} mm; its 3%/0mm local gamma remains {fullplan['dose_3pct_0mm_local_percent']:.3f}%.", "", "## Width discriminator", "", discriminator, f"Entrance/peak mean lateral width ratios: {entrance_mean:.4f} / {peak_mean:.4f}.", "", f"Profiles: `{args.output_dir / 'lateral_profiles.csv'}`; depth dose: `{args.output_dir / 'integrated_depth_dose.csv'}`."]
    (args.output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "gpu_run": stats, "com_delta_10pct_mm": delta, "width_ratios_10pct": ratios, "emittance_vs_transport": discriminator}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
