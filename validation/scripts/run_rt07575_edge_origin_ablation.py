#!/usr/bin/env python3
"""Validate the RT07575 CT low-edge origin correction at equal histories.

The corrected arm changes only the TPS-90 packed CT-grid header origin and
the matching patient-X edge supplied to the spot conversion.  It deliberately
does not apply a world/minibeam translation or alter density/material payloads.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_rt07575_equal_history_groups import load_gpu_mapped, load_topas  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402
from match_gpu_to_physical_dose import dose_only_pass_rate, gamma_3d  # noqa: E402


HISTORIES = 12_963_817
SEED = 20_260_801
HEADER = struct.Struct("<IIIIIffffff")
MAGIC = 0x47544343
QUANTITIES = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def replace_one(text: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"^{re.escape(key)}:.*$", f"{key}: {value}", text, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return updated


def read_header(path: Path) -> tuple[Any, ...]:
    payload = path.read_bytes()
    if len(payload) < HEADER.size:
        raise ValueError(f"truncated CCTG header: {path}")
    return HEADER.unpack_from(payload)


def ensure_grid(args: argparse.Namespace) -> dict[str, Any]:
    """Pack the corrected patient grid, then prove it differs only in header."""
    command = [
        sys.executable,
        str(args.reorient_script),
        "--input", str(args.corrected_patient_grid),
        "--output", str(args.corrected_grid),
        "--metadata", str(args.corrected_grid_metadata),
        "--beam-patient-x-direction", "negative",
        "--beam-patient-y-direction", "positive",
        "--tps-angle-deg", "90.0",
    ]
    # Repacking is deterministic and keeps this experiment self-contained.
    subprocess.run(command, cwd=args.repo_root, check=True, text=True)
    legacy = args.legacy_grid.read_bytes()
    corrected = args.corrected_grid.read_bytes()
    patient_legacy = args.legacy_patient_grid.read_bytes()
    patient_corrected = args.corrected_patient_grid.read_bytes()
    if patient_legacy[HEADER.size:] != patient_corrected[HEADER.size:]:
        raise RuntimeError("corrected source patient payload differs from legacy")
    if legacy[HEADER.size:] != corrected[HEADER.size:]:
        raise RuntimeError("corrected TPS-90 density/material payload differs from legacy")
    header = read_header(args.corrected_grid)
    expected = (MAGIC, 3, 505, 35, 417, -126.25, -35.0, 0.0, 0.5, 2.0, 0.5)
    if header != expected:
        raise RuntimeError(f"unexpected corrected TPS-90 header: {header!r}")
    metadata = json.loads(args.corrected_grid_metadata.read_text(encoding="utf-8"))
    if metadata["origin_xyz_mm"] != [-126.25, -35.0, 0.0]:
        raise RuntimeError("metadata origin is not the expected packed low edge")
    if metadata["spacing_xyz_mm"] != [0.5, 2.0, 0.5] or metadata["shape_xyz"] != [505, 35, 417]:
        raise RuntimeError("metadata shape/spacing differs from the established TPS-90 grid")
    return {
        "mapping": metadata["axis_mapping"],
        "interpolation": metadata["interpolation"],
        "world_origin_shifts_mm": [metadata["gpu_origin_x_shift_mm"], metadata["gpu_origin_y_shift_mm"]],
        "legacy_header": list(read_header(args.legacy_grid)),
        "corrected_header": list(header),
        "corrected_metadata_origin_xyz_mm": metadata["origin_xyz_mm"],
        "corrected_metadata_spacing_xyz_mm": metadata["spacing_xyz_mm"],
        "corrected_metadata_shape_xyz": metadata["shape_xyz"],
        "patient_payload_equal_after_header": True,
        "tps90_payload_equal_after_header": True,
        "legacy_payload_sha256": hashlib.sha256(legacy[HEADER.size:]).hexdigest(),
        "corrected_payload_sha256": hashlib.sha256(corrected[HEADER.size:]).hexdigest(),
    }


def render_config(template: Path, destination: Path, run_dir: Path) -> list[str]:
    text = template.read_text(encoding="utf-8")
    overrides = {
        "number_of_histories": str(HISTORIES),
        "random_seed": str(SEED),
        "ct_grid_file": "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin",
        "spots_ct_axis_min_mm": "-104.25",
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
    }
    for key, value in overrides.items():
        text = replace_one(text, key, value)
    destination.write_text(text, encoding="utf-8")
    return list(overrides)


def number(text: str, pattern: str, default: float | None = None) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(match.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    return {
        "histories": int(number(text, r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": number(text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"),
        "energy_balance_error": number(text, r"^Energy balance error:\s*([0-9.eE+-]+)$"),
        "secondary_queue_overflow": int(number(text, r"^Secondary queue overflow:\s*(\d+)$")),
        "cascade_queue_overflow": int(number(text, r"^Cascade queue overflow:\s*(\d+)$")),
        "neutral_queue_overflow": int(number(text, r"^Neutral queue overflow:\s*(\d+)$", 0)),
        "backend": backend.group(1) if backend else "unknown",
    }


def complete(run_dir: Path) -> dict[str, Any] | None:
    required = [run_dir / filename for filename in QUANTITIES.values()] + [
        run_dir / "dose.raw", run_dir / "letd_primary_c12.raw", run_dir / "letd_all_hadron.raw",
        run_dir / "run.log", run_dir / "config.yaml",
    ]
    if not all(path.exists() for path in required):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES:
        return None
    if any(stats[key] for key in ("secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow")):
        return None
    return stats


def run_corrected(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    run_dir = args.output_root / "edge_corrected"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = complete(run_dir)
    if saved is not None:
        print("SKIP complete edge_corrected", flush=True)
        return saved, ["resumed complete output"]
    config = run_dir / "config.yaml"
    changed = render_config(args.template, config, run_dir)
    print("RUN edge_corrected", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(args.binary), "--config", str(config), "--device", "cuda"],
                                cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT,
                                check=False, text=True)
    if result.returncode:
        raise RuntimeError(f"edge_corrected failed ({result.returncode}); see {run_dir / 'run.log'}")
    saved = complete(run_dir)
    if saved is None:
        raise RuntimeError("edge_corrected is incomplete, not exact-history, or overflowed")
    return saved, changed


def dose_com(values: np.ndarray, selection: np.ndarray, shape: tuple[int, int, int], spacing: tuple[float, float, float], offset: tuple[float, float, float]) -> list[float]:
    nx, ny, _nz = shape
    linear = np.flatnonzero(selection)
    weights = values[linear]
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0:
        raise ValueError("non-positive selected dose integral")
    z = linear // (nx * ny)
    rem = linear % (nx * ny)
    y, x = rem // nx, rem % nx
    return [
        float(np.sum(weights * (offset[0] + x * spacing[0]), dtype=np.float64) / total),
        float(np.sum(weights * (offset[1] + y * spacing[1]), dtype=np.float64) / total),
        float(np.sum(weights * (offset[2] + z * spacing[2]), dtype=np.float64) / total),
    ]


def quantity_metrics(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray,
                     body: np.ndarray, shape: tuple[int, int, int], spacing: tuple[float, float, float],
                     cache_path: Path) -> dict[str, Any]:
    item: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    ref_body, eval_body = np.where(body, reference, 0.0), np.where(body, evaluation, 0.0)
    ref_selected, eval_selected = reference[selection], evaluation[selection]
    if not item:
        delta = eval_selected - ref_selected
        item = {
            "selected_voxels": int(np.count_nonzero(selection)),
            "evaluation_over_reference_body_integral": float(np.sum(eval_body, dtype=np.float64) / max(np.sum(ref_body, dtype=np.float64), 1e-30)),
            "selected_mean_reference": float(np.mean(ref_selected)),
            "selected_mean_evaluation": float(np.mean(eval_selected)),
            "selected_nrmse_over_reference_max_percent": float(100 * np.sqrt(np.mean(delta * delta)) / max(float(np.max(ref_selected)), 1e-30)),
            "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]),
            "gamma": {},
        }
        write_json(cache_path, item)
    reference_flat = array.array("f", ref_body.astype(np.float32, copy=False))
    evaluation_flat = eval_body.reshape(-1).tolist()
    gamma: dict[str, Any] = item["gamma"]
    for percent, distance, label in CRITERIA:
        criterion = gamma.setdefault(label, {})
        for local, mode in ((False, "global"), (True, "local")):
            if mode not in criterion:
                print(f"Computing {cache_path.stem}: {label} {mode}", flush=True)
                criterion[mode] = gamma_3d(reference_flat, evaluation_flat, shape, spacing, percent, distance,
                                           10.0, 50_000, 0, local_dose=local, interpolation_step_mm=0.5,
                                           selection_mask=selection.reshape(-1))
                write_json(cache_path, item)
    if "30" not in gamma:
        gamma["30"] = {
            "global": dose_only_pass_rate(reference_flat, evaluation_flat, 3.0, 10.0, selection_mask=selection.reshape(-1)),
            "local": dose_only_pass_rate(reference_flat, evaluation_flat, 3.0, 10.0, local_dose=True, selection_mask=selection.reshape(-1)),
        }
        write_json(cache_path, item)
    return item


def comparison(topas_dir: Path, gpu_dir: Path, body_mask: Path, cache_root: Path,
               established_quantities: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    topas, gpu = load_topas(topas_dir), load_gpu_mapped(gpu_dir)
    body_metadata, body_values = load(body_mask)
    topas_dose_metadata, _ = load(topas_dir / "dose.mhd")
    shape = tuple(int(v) for v in body_metadata["DimSize"].split())
    spacing = tuple(float(v) for v in body_metadata["ElementSpacing"].split())
    offset = tuple(float(v) for v in topas_dose_metadata["Offset"].split())
    if shape != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape}")
    body = body_values > 0.5
    selection = body & (topas["dose"] >= 0.10 * float(np.max(topas["dose"][body])))
    dose_reference_com = dose_com(topas["dose"], selection, shape, spacing, offset)
    dose_evaluation_com = dose_com(gpu["dose"], selection, shape, spacing, offset)
    quantities = established_quantities if established_quantities is not None else {
        name: quantity_metrics(topas[name], gpu[name], selection, body, shape, spacing,
                               cache_root / f"{name}.json") for name in QUANTITIES
    }
    result = {
        "reference": str(topas_dir), "evaluation": str(gpu_dir),
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {"3d_points": 50_000, "available_points": int(np.count_nonzero(selection)),
                           "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
        "quantities": quantities,
        "dose_center_of_mass_patient_xyz_mm": {
            "reference_topas_seed1": dose_reference_com,
            "evaluation_gpu": dose_evaluation_com,
            "gpu_minus_topas": [float(a - b) for a, b in zip(dose_evaluation_com, dose_reference_com)],
            "support": "same BODY intersect TOPAS seed-1 dose >=10% BODY-Dmax mask; single-run TOPAS seed 1 reference",
        },
    }
    return result, {"shape_xyz": list(shape), "spacing_xyz_mm": list(spacing), "offset_xyz_mm": list(offset)}


def gamma_deltas(baseline: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    return {name: {label: {mode: corrected["quantities"][name]["gamma"][label][mode]["pass_percent"] - baseline["quantities"][name]["gamma"][label][mode]["pass_percent"] for mode in ("global", "local")} for _, _, label in CRITERIA + ((3.0, 0.0, "30"),)} for name in QUANTITIES}


def scalar_deltas(baseline: dict[str, Any], corrected: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in QUANTITIES:
        before, after = baseline["quantities"][name], corrected["quantities"][name]
        out[name] = {
            "body_integral_ratio": after["evaluation_over_reference_body_integral"] - before["evaluation_over_reference_body_integral"],
            "selected_nrmse_over_reference_max_percent": after["selected_nrmse_over_reference_max_percent"] - before["selected_nrmse_over_reference_max_percent"],
        }
    before_com = baseline["dose_center_of_mass_patient_xyz_mm"]["gpu_minus_topas"]
    after_com = corrected["dose_center_of_mass_patient_xyz_mm"]["gpu_minus_topas"]
    out["dose_com_gpu_minus_topas_mm"] = {axis: after_com[i] - before_com[i] for i, axis in enumerate(("x", "y", "z"))}
    return out


def markdown(report: dict[str, Any]) -> str:
    lines = ["# RT07575 CT low-edge origin ablation", "",
             f"Both GPU arms use {HISTORIES:,} histories, seed {SEED}, and absolute equal-history scale (no fitted normalization).",
             "Corrected arm: packed CCTG header low edge and `spots_ct_axis_min_mm` move from -104.00 to -104.25 mm; no world/minibeam translation.",
             "Mask: RTSTRUCT BODY ∩ TOPAS seed-1 dose ≥10% BODY Dmax. LET uses this exact dose mask.",
             "3D gamma: deterministic 50,000 points, 0.5 mm interpolation; 3%/0 mm uses all selected voxels.", "",
             "## Grid audit", "",
             f"`{report['grid_audit']['mapping']}`, interpolation={report['grid_audit']['interpolation']}, world shifts={report['grid_audit']['world_origin_shifts_mm']} mm.",
             f"Packed origin={report['grid_audit']['corrected_metadata_origin_xyz_mm']} mm; spacing={report['grid_audit']['corrected_metadata_spacing_xyz_mm']} mm; shape={report['grid_audit']['corrected_metadata_shape_xyz']}.",
             f"Legacy/corrected packed density-material payload byte-identical: {report['grid_audit']['tps90_payload_equal_after_header']}.", "",
             "## Runtime and safety", "",
             "| Arm | Elapsed (s) | Throughput (hist/s) | Energy balance error | Secondary / cascade / neutral overflow |",
             "|---|---:|---:|---:|---:|"]
    for name, stats in report["runs"].items():
        lines.append(f"| {name} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | {stats['energy_balance_error']:.8g} | {stats['secondary_queue_overflow']} / {stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |")
    for arm, comparison_data in report["comparisons"].items():
        lines += ["", f"## {arm} vs TOPAS seed 1", "",
                  "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, metric in comparison_data["quantities"].items():
            gamma = metric["gamma"]
            gamma_cells = [f"{gamma[label]['global']['pass_percent']:.3f} / {gamma[label]['local']['pass_percent']:.3f}" for label in ("33", "22", "11", "30")]
            lines.append(f"| {name} | {metric['evaluation_over_reference_body_integral']:.6f} | {metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | {metric['selected_nrmse_over_reference_max_percent']:.3f}% | " + " | ".join(gamma_cells) + " |")
        com = comparison_data["dose_center_of_mass_patient_xyz_mm"]["gpu_minus_topas"]
        lines.append(f"Dose COM GPU − TOPAS (patient x, y, z): [{com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f}] mm.")
    lines += ["", "## Corrected minus baseline", "",
              "| Quantity | Δ integral E/R | Δ NRMSE/Dmax (pp) | Δ 3%/3mm G/L (pp) | Δ 2%/2mm G/L (pp) | Δ 1%/1mm G/L (pp) | Δ 3%/0mm G/L (pp) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in QUANTITIES:
        deltas = report["strict_deltas_vs_baseline"][name]
        gamma = report["gamma_deltas_vs_baseline_pp"][name]
        cells = [f"{gamma[label]['global']:+.3f} / {gamma[label]['local']:+.3f}" for label in ("33", "22", "11", "30")]
        lines.append(f"| {name} | {deltas['body_integral_ratio']:+.6f} | {deltas['selected_nrmse_over_reference_max_percent']:+.3f} | " + " | ".join(cells) + " |")
    com_delta = report["strict_deltas_vs_baseline"]["dose_com_gpu_minus_topas_mm"]
    lines += ["", f"Dose COM residual change (corrected − baseline), patient x/y/z mm: [{com_delta['x']:+.4f}, {com_delta['y']:+.4f}, {com_delta['z']:+.4f}].", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--reorient-script", type=Path, default=Path("validation/scripts/reorient_ct_grid_tps_90.py"))
    parser.add_argument("--legacy-patient-grid", type=Path, default=Path("ct/grid/patient_ct.bin"))
    parser.add_argument("--corrected-patient-grid", type=Path, default=Path("ct/grid/patient_ct_rt07575_edge_corrected.bin"))
    parser.add_argument("--legacy-grid", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg.bin"))
    parser.add_argument("--corrected-grid", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin"))
    parser.add_argument("--corrected-grid-metadata", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.metadata.json"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("out/ct/RT07575/equal_history_best_seeds/seed20260801"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--baseline-gamma", type=Path, default=Path("out/ct/RT07575/physics_ablation/gamma_checkpoints/baseline.json"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation"))
    args = parser.parse_args()
    for field in vars(args):
        value = getattr(args, field)
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    for field in ("binary", "template", "reorient_script", "legacy_patient_grid", "corrected_patient_grid", "legacy_grid", "baseline_dir", "topas_dir", "body_mask", "baseline_gamma"):
        if not getattr(args, field).exists():
            raise FileNotFoundError(getattr(args, field))
    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_audit = ensure_grid(args)
    baseline_stats = complete(args.baseline_dir)
    if baseline_stats is None:
        raise RuntimeError("baseline is not exact-history and zero-overflow")
    baseline_quantities = json.loads(args.baseline_gamma.read_text(encoding="utf-8"))
    for name in QUANTITIES:
        for _, _, label in CRITERIA + ((3.0, 0.0, "30"),):
            if not {"global", "local"}.issubset(baseline_quantities.get(name, {}).get("gamma", {}).get(label, {})):
                raise ValueError(f"incomplete established baseline gamma: {name}/{label}")
    corrected_stats, changed_keys = run_corrected(args)
    # Recalculate only scalar baseline/COM metrics on the established single-run
    # reference; established deterministic gamma checkpoints are reused verbatim.
    baseline_comparison, grid = comparison(args.topas_dir, args.baseline_dir, args.body_mask,
                                            args.output_root / "gamma_checkpoints" / "baseline_scalar",
                                            baseline_quantities)
    corrected_comparison, _ = comparison(args.topas_dir, args.output_root / "edge_corrected", args.body_mask,
                                          args.output_root / "gamma_checkpoints" / "edge_corrected")
    report = {
        "case": "RT07575", "experiment": "CT half-voxel low-edge origin correction",
        "histories": HISTORIES, "seed": SEED, "template": str(args.template),
        "rendered_config_changed_keys": changed_keys,
        "only_geometry_overrides": {"ct_grid_file": "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin", "spots_ct_axis_min_mm": -104.25},
        "baseline_gpu_seed1": str(args.baseline_dir), "topas_seed1": str(args.topas_dir),
        "grid_audit": grid_audit, "patient_grid": grid,
        "runs": {"baseline": baseline_stats, "edge_corrected": corrected_stats},
        "comparisons": {"baseline": baseline_comparison, "edge_corrected": corrected_comparison},
        "gamma_deltas_vs_baseline_pp": gamma_deltas(baseline_comparison, corrected_comparison),
        "strict_deltas_vs_baseline": scalar_deltas(baseline_comparison, corrected_comparison),
    }
    write_json(args.output_root / "summary.json", report)
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
