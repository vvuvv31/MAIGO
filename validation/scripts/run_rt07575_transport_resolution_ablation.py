#!/usr/bin/env python3
"""Run the RT07575 corrected-origin transport-resolution A/B on CUDA.

Only ``maximum_step_mm`` and ``maximum_relative_energy_loss`` differ from
the established corrected-origin best run.  Dose and LET are compared to the
same TOPAS seed-1 reference on the established fixed mask, with no fitted
dose scale, spatial alignment, or post-processing correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_rt07575_equal_history_groups import load_gpu_mapped, load_topas  # noqa: E402
from compare_topas_seed_gamma import load  # noqa: E402
from run_rt07575_edge_origin_ablation import (  # noqa: E402
    CRITERIA,
    HISTORIES,
    QUANTITIES,
    SEED,
    parse_log,
    quantity_metrics,
)


EXPECTED_OVERRIDES = {
    "number_of_histories": str(HISTORIES),
    "random_seed": str(SEED),
    "ct_grid_file": "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin",
    "spots_ct_axis_min_mm": "-104.25",
    "maximum_step_mm": "0.05",
    "maximum_relative_energy_loss": "0.0005",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_one(text: str, key: str, value: str) -> str:
    updated, count = re.subn(
        rf"^{re.escape(key)}:.*$", f"{key}: {value}", text, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return updated


def render_config(template: Path, output: Path, run_dir: Path) -> list[str]:
    text = template.read_text(encoding="utf-8")
    overrides = {
        **EXPECTED_OVERRIDES,
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
    }
    for key, value in overrides.items():
        text = replace_one(text, key, value)
    output.write_text(text, encoding="utf-8")
    return list(overrides)


def require_complete_gpu_run(run_dir: Path) -> dict[str, Any]:
    required = [
        run_dir / "config.yaml", run_dir / "run.log", run_dir / "dose.mhd",
        run_dir / "dose.raw", run_dir / "letd_primary_c12.mhd",
        run_dir / "letd_primary_c12.raw", run_dir / "letd_all_hadron.mhd",
        run_dir / "letd_all_hadron.raw",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"incomplete GPU run: {missing}")
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES:
        raise RuntimeError(f"expected {HISTORIES} histories, got {stats['histories']}")
    for key in ("secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow"):
        if stats[key] != 0:
            raise RuntimeError(f"{key}={stats[key]}, refusing comparison")
    if "cuda" not in stats["backend"].lower():
        raise RuntimeError(f"GPU run is not CUDA: {stats['backend']!r}")
    stats["run_log_sha256"] = sha256(run_dir / "run.log")
    return stats


def active_pid(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_run_lock(run_dir: Path) -> Path:
    """Atomically reserve this output directory for a single CUDA process.

    Deliberately retain the lock if the parent runner is interrupted.  The
    carbon_mc child can survive a terminal/session interruption; a retained
    lock therefore prevents a second process from writing the same dose files.
    A stale lock requires an explicit later ``--clear-stale-lock`` decision.
    """
    lock = run_dir / "run.lock"
    if lock.exists():
        try:
            record = json.loads(lock.read_text(encoding="utf-8"))
            pid = int(record.get("child_pid", record["pid"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"unreadable run lock {lock}; refuse CUDA launch") from error
        state = "active" if active_pid(pid) else "stale"
        raise RuntimeError(
            f"{state} run lock at {lock} for pid {pid}; refuse a concurrent or "
            "ambiguous CUDA launch"
        )
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f"race acquiring {lock}; refuse CUDA launch") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created_unix_seconds": time.time()}, stream)
        stream.write("\n")
    return lock


def record_child_pid(lock: Path, child_pid: int) -> None:
    """Replace the provisional launcher PID with the actual CUDA child PID."""
    record = json.loads(lock.read_text(encoding="utf-8"))
    record["child_pid"] = child_pid
    lock.write_text(json.dumps(record) + "\n", encoding="utf-8")


def clear_stale_lock(run_dir: Path) -> None:
    lock = run_dir / "run.lock"
    if not lock.exists():
        return
    try:
        record = json.loads(lock.read_text(encoding="utf-8"))
        pid = int(record.get("child_pid", record["pid"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unreadable run lock {lock}; remove it manually after audit") from error
    if active_pid(pid):
        raise RuntimeError(f"run lock belongs to active pid {pid}; refuse to clear it")
    lock.unlink()


def comparison(
    topas_dir: Path, gpu_dir: Path, body_mask: Path, cache_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    topas, gpu = load_topas(topas_dir), load_gpu_mapped(gpu_dir)
    body_meta, body_values = load(body_mask)
    dose_meta, _ = load(topas_dir / "dose.mhd")
    shape = tuple(int(v) for v in body_meta["DimSize"].split())
    spacing = tuple(float(v) for v in body_meta["ElementSpacing"].split())
    if shape != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape}")
    body = body_values > 0.5
    selection = body & (topas["dose"] >= 0.10 * float(np.max(topas["dose"][body])))
    quantities = {
        name: quantity_metrics(
            topas[name], gpu[name], selection, body, shape, spacing,
            cache_root / f"{name}.json",
        )
        for name in QUANTITIES
    }
    return {
        "reference": str(topas_dir),
        "evaluation": str(gpu_dir),
        "normalization": "absolute equal-history scale; no fitted dose normalization or alignment",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "topas_seed1_body_dmax": float(np.max(topas["dose"][body])),
        "gamma_sampling": {
            "3d_points": 50_000,
            "available_points": int(np.count_nonzero(selection)),
            "interpolation_step_mm": 0.5,
            "3pct_0mm_points": "all selected voxels",
        },
        "quantities": quantities,
    }, {
        "shape_xyz": list(shape),
        "spacing_xyz_mm": list(spacing),
        "topas_offset_xyz_mm": [float(v) for v in dose_meta["Offset"].split()],
        "body_voxels": int(np.count_nonzero(body)),
    }


def deltas(baseline: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in QUANTITIES:
        before, after = baseline["quantities"][name], refined["quantities"][name]
        gamma_delta = {
            label: {
                mode: after["gamma"][label][mode]["pass_percent"]
                - before["gamma"][label][mode]["pass_percent"]
                for mode in ("global", "local")
            }
            for _, _, label in CRITERIA + ((3.0, 0.0, "30"),)
        }
        output[name] = {
            "body_integral_ratio_delta": after[
                "evaluation_over_reference_body_integral"
            ] - before["evaluation_over_reference_body_integral"],
            "nrmse_over_reference_dmax_percent_delta": after[
                "selected_nrmse_over_reference_max_percent"
            ] - before["selected_nrmse_over_reference_max_percent"],
            "gamma_pass_percent_delta": gamma_delta,
        }
    return output


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RT07575 corrected-origin transport-resolution A/B",
        "",
        f"Both arms use exactly {HISTORIES:,} histories and seed {SEED} on CUDA. The fixed corrected grid/origin is `ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin` with `spots_ct_axis_min_mm=-104.25`.",
        "Only the A/B arm changes `maximum_step_mm` 0.1→0.05 and `maximum_relative_energy_loss` 0.001→0.0005. No dose scale, registration, blur, or other physics/config switch is fitted or changed.",
        "Mask: RTSTRUCT BODY ∩ TOPAS seed-1 dose ≥10% BODY Dmax. Gamma uses 50,000 deterministic points with 0.5 mm interpolation; 3%/0 mm uses all selected voxels.",
        "",
        "## CUDA runtime and integrity",
        "",
        "| Arm | Elapsed (s) | Throughput (hist/s) | Energy-balance error | Secondary / cascade / neutral overflow |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in report["runs"].items():
        lines.append(
            f"| {name} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | {stats['energy_balance_error']:.8g} | {stats['secondary_queue_overflow']} / {stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |"
        )
    for arm, comparison_data in report["comparisons"].items():
        lines += [
            "", f"## {arm} vs TOPAS seed 1", "",
            "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, metric in comparison_data["quantities"].items():
            gamma = metric["gamma"]
            cells = [
                f"{gamma[label]['global']['pass_percent']:.3f} / {gamma[label]['local']['pass_percent']:.3f}"
                for label in ("33", "22", "11", "30")
            ]
            lines.append(
                f"| {name} | {metric['evaluation_over_reference_body_integral']:.6f} | "
                f"{metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | "
                f"{metric['selected_nrmse_over_reference_max_percent']:.3f}% | "
                + " | ".join(cells) + " |"
            )
    lines += [
        "", "## Resolution A/B minus corrected baseline", "",
        "| Quantity | Δ integral E/R | Δ NRMSE/Dmax (pp) | Δ 3%/3mm G/L (pp) | Δ 2%/2mm G/L (pp) | Δ 1%/1mm G/L (pp) | Δ 3%/0mm G/L (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in QUANTITIES:
        delta = report["deltas_vs_corrected_baseline"][name]
        cells = [
            f"{delta['gamma_pass_percent_delta'][label]['global']:+.3f} / {delta['gamma_pass_percent_delta'][label]['local']:+.3f}"
            for label in ("33", "22", "11", "30")
        ]
        lines.append(
            f"| {name} | {delta['body_integral_ratio_delta']:+.6f} | "
            f"{delta['nrmse_over_reference_dmax_percent_delta']:+.3f} | "
            + " | ".join(cells) + " |"
        )
    timing = report["timing_delta_vs_corrected_baseline"]
    lines += [
        "", f"Resolution A/B runtime change: {timing['elapsed_seconds_delta']:+.3f} s ({timing['elapsed_seconds_relative_percent']:+.2f}%); throughput change {timing['throughput_relative_percent']:+.2f}%.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--corrected-grid", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin"))
    parser.add_argument("--corrected-grid-metadata", type=Path, default=Path("ct/grid/patient_ct_tps_90_xneg_edge_corrected.metadata.json"))
    parser.add_argument("--corrected-baseline-dir", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation/edge_corrected"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/transport_resolution_ablation"))
    parser.add_argument("--launch", action="store_true", help="explicitly launch the one CUDA A/B run; default is preflight only")
    parser.add_argument("--clear-stale-lock", action="store_true", help="remove only a proven-dead retained run lock before --launch")
    args = parser.parse_args()
    for field, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    for field in ("binary", "template", "corrected_grid", "corrected_grid_metadata", "corrected_baseline_dir", "topas_dir", "body_mask"):
        if not getattr(args, field).exists():
            raise FileNotFoundError(getattr(args, field))
    metadata = json.loads(args.corrected_grid_metadata.read_text(encoding="utf-8"))
    if metadata.get("origin_xyz_mm") != [-126.25, -35.0, 0.0]:
        raise RuntimeError("corrected-grid packed origin is not [-126.25, -35.0, 0.0]")
    if metadata.get("shape_xyz") != [505, 35, 417] or metadata.get("spacing_xyz_mm") != [0.5, 2.0, 0.5]:
        raise RuntimeError("corrected-grid geometry is not the established TPS-90 geometry")
    args.output_root.mkdir(parents=True, exist_ok=True)
    baseline_stats = require_complete_gpu_run(args.corrected_baseline_dir)
    run_dir = args.output_root / "resolution_half"
    config = run_dir / "config.yaml"
    try:
        refined_stats = require_complete_gpu_run(run_dir)
    except (RuntimeError, FileNotFoundError):
        refined_stats = None
    if refined_stats is None:
        if not args.launch:
            raise RuntimeError(
                "resolution_half is not complete. CUDA launch is intentionally "
                "disabled by default; after confirming the GPU is free, rerun with --launch"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.clear_stale_lock:
            clear_stale_lock(run_dir)
        lock = acquire_run_lock(run_dir)
        rendered_keys = render_config(args.template, config, run_dir)
        print("RUN resolution_half on CUDA", flush=True)
        with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                [str(args.binary), "--config", str(config), "--device", "cuda"],
                cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT,
                text=True,
            )
            record_child_pid(lock, process.pid)
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"resolution_half failed ({returncode}); see {run_dir / 'run.log'}")
        # Only an exact complete, zero-overflow process may release its lock.
        refined_stats = require_complete_gpu_run(run_dir)
        lock.unlink()
    else:
        if args.clear_stale_lock:
            raise RuntimeError("--clear-stale-lock is invalid when an exact result already exists")
        rendered_keys = ["resumed existing exact run"]
        print("RESUME resolution_half", flush=True)
    refined_stats = require_complete_gpu_run(run_dir)
    config_text = config.read_text(encoding="utf-8")
    for key, value in EXPECTED_OVERRIDES.items():
        if not re.search(rf"^{re.escape(key)}:\s*{re.escape(value)}\s*$", config_text, re.MULTILINE):
            raise RuntimeError(f"rendered config does not contain {key}: {value}")
    baseline_comparison, grid = comparison(
        args.topas_dir, args.corrected_baseline_dir, args.body_mask,
        args.output_root / "gamma_checkpoints" / "corrected_baseline",
    )
    refined_comparison, _ = comparison(
        args.topas_dir, run_dir, args.body_mask,
        args.output_root / "gamma_checkpoints" / "resolution_half",
    )
    elapsed_delta = refined_stats["elapsed_seconds"] - baseline_stats["elapsed_seconds"]
    report = {
        "case": "RT07575",
        "experiment": "corrected-origin transport resolution: halve maximum step and maximum relative energy loss together",
        "histories": HISTORIES,
        "seed": SEED,
        "normalization": "absolute equal-history scale; no fitted dose scale, alignment, or blur",
        "corrected_grid": {"path": str(args.corrected_grid), "metadata": str(args.corrected_grid_metadata), "origin_xyz_mm": metadata["origin_xyz_mm"], "spots_ct_axis_min_mm": -104.25},
        "template": str(args.template),
        "rendered_config_changed_keys": rendered_keys,
        "controlled_change": {"maximum_step_mm": {"baseline": 0.1, "resolution_half": 0.05}, "maximum_relative_energy_loss": {"baseline": 0.001, "resolution_half": 0.0005}},
        "corrected_baseline_dir": str(args.corrected_baseline_dir),
        "resolution_half_dir": str(run_dir),
        "topas_seed1": str(args.topas_dir),
        "patient_grid": grid,
        "runs": {"corrected_baseline": baseline_stats, "resolution_half": refined_stats},
        "comparisons": {"corrected_baseline": baseline_comparison, "resolution_half": refined_comparison},
        "deltas_vs_corrected_baseline": deltas(baseline_comparison, refined_comparison),
        "timing_delta_vs_corrected_baseline": {"elapsed_seconds_delta": elapsed_delta, "elapsed_seconds_relative_percent": 100.0 * elapsed_delta / baseline_stats["elapsed_seconds"], "throughput_histories_per_second_delta": refined_stats["throughput_histories_per_second"] - baseline_stats["throughput_histories_per_second"], "throughput_relative_percent": 100.0 * (refined_stats["throughput_histories_per_second"] / baseline_stats["throughput_histories_per_second"] - 1.0)},
    }
    write_json(args.output_root / "summary.json", report)
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
