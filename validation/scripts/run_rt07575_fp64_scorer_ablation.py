#!/usr/bin/env python3
"""Test FP64 voxel-dose atomics against corrected-origin RT07575 FP32 scoring.

This is a scorer-precision A/B: source, geometry, transport physics, history
sequence, and absolute dose response are fixed.  The only binary difference is
``CARBON_DOSE_FP32``.
"""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path
import re
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
SMOKE_HISTORIES = 100_000
SEED = 20_260_801
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))
REQUIRED_OUTPUTS = (
    "dose.mhd", "dose.raw", "letd_primary_c12.mhd", "letd_primary_c12.raw",
    "letd_all_hadron.mhd", "letd_all_hadron.raw", "run.log", "config.yaml",
)


def replace_one(text: str, key: str, value: str) -> str:
    updated, count = re.subn(rf"^{re.escape(key)}:.*$", f"{key}: {value}", text,
                             flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return updated


def render_config(template: Path, output: Path, run_dir: Path, histories: int) -> None:
    text = template.read_text(encoding="utf-8")
    # These reproduce the corrected-origin production arm.  Only histories and
    # scorer-output paths vary between smoke and exact FP64 runs.
    for key, value in {
        "number_of_histories": str(histories),
        "random_seed": str(SEED),
        "ct_grid_file": "ct/grid/patient_ct_tps_90_xneg_edge_corrected.bin",
        "spots_ct_axis_min_mm": "-104.25",
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
    }.items():
        text = replace_one(text, key, value)
    output.write_text(text, encoding="utf-8")


def read_number(text: str, pattern: str, default: float | None = None) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(match.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    atomic = re.search(r"^Launch plan:.*dose_atomic=(fp(?:32|64))$", text, re.MULTILINE)
    return {
        "histories": int(read_number(text, r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": read_number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": read_number(text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"),
        "energy_balance_error": read_number(text, r"^Energy balance error:\s*([0-9.eE+-]+)$"),
        "secondary_queue_overflow": int(read_number(text, r"^Secondary queue overflow:\s*(\d+)$")),
        "cascade_queue_overflow": int(read_number(text, r"^Cascade queue overflow:\s*(\d+)$")),
        "neutral_queue_overflow": int(read_number(text, r"^Neutral queue overflow:\s*(\d+)$", 0)),
        "backend": backend.group(1) if backend else "unknown",
        "dose_atomic": atomic.group(1) if atomic else "unknown",
    }


def complete(run_dir: Path, histories: int, expected_atomic: str) -> dict[str, Any] | None:
    if not all((run_dir / name).exists() for name in REQUIRED_OUTPUTS):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != histories or stats["dose_atomic"] != expected_atomic:
        return None
    if expected_atomic not in stats["backend"]:
        return None
    if any(stats[key] for key in ("secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow")):
        return None
    return stats


def run_arm(args: argparse.Namespace, name: str, histories: int) -> dict[str, Any]:
    run_dir = args.output_root / name
    saved = complete(run_dir, histories, "fp64")
    if saved is not None:
        print(f"SKIP complete {name}", flush=True)
        return saved
    run_dir.mkdir(parents=True, exist_ok=True)
    render_config(args.template, run_dir / "config.yaml", run_dir, histories)
    print(f"RUN {name}", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(args.fp64_binary), "--config", str(run_dir / "config.yaml"), "--device", "cuda"],
                                cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT,
                                check=False, text=True)
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}); see {run_dir / 'run.log'}")
    saved = complete(run_dir, histories, "fp64")
    if saved is None:
        raise RuntimeError(f"{name} output is incomplete, wrong precision/history, or overflowed")
    return saved


def cache_setting(path: Path) -> str:
    match = re.search(r"^CARBON_DOSE_FP32:BOOL=(ON|OFF)$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise ValueError(f"CARBON_DOSE_FP32 absent from {path}")
    return match.group(1)


def metrics(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray,
            body: np.ndarray, shape: tuple[int, int, int], spacing: tuple[float, float, float],
            cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    ref_body, eval_body = np.where(body, reference, 0.0), np.where(body, evaluation, 0.0)
    ref = reference[selection]
    ev = evaluation[selection]
    delta = ev - ref
    dmax = float(np.max(ref))
    item: dict[str, Any] = {
        "selected_voxels": int(ref.size),
        "evaluation_over_reference_body_integral": float(np.sum(eval_body, dtype=np.float64) / max(np.sum(ref_body, dtype=np.float64), 1e-30)),
        "selected_max_absolute_difference": float(np.max(np.abs(delta))),
        "selected_max_absolute_difference_over_reference_dmax_percent": float(100 * np.max(np.abs(delta)) / max(dmax, 1e-30)),
        "selected_rmse_over_reference_dmax_percent": float(100 * np.sqrt(np.mean(delta * delta)) / max(dmax, 1e-30)),
        "selected_mean_signed_difference_over_reference_dmax_percent": float(100 * np.mean(delta) / max(dmax, 1e-30)),
        "gamma": {},
    }
    ref_flat = array.array("f", ref_body.astype(np.float32, copy=False))
    eval_flat = eval_body.reshape(-1).tolist()
    for percent, distance, label in CRITERIA:
        item["gamma"][label] = {
            "global": gamma_3d(ref_flat, eval_flat, shape, spacing, percent, distance, 10.0, 50_000, 0,
                               interpolation_step_mm=0.5, selection_mask=selection.reshape(-1)),
            "local": gamma_3d(ref_flat, eval_flat, shape, spacing, percent, distance, 10.0, 50_000, 0,
                              local_dose=True, interpolation_step_mm=0.5, selection_mask=selection.reshape(-1)),
        }
    item["gamma"]["30"] = {
        "global": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0, selection_mask=selection.reshape(-1)),
        "local": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0, local_dose=True, selection_mask=selection.reshape(-1)),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")
    return item


def comparisons(args: argparse.Namespace) -> dict[str, Any]:
    topas = load_topas(args.topas_dir)
    fp32 = load_gpu_mapped(args.fp32_dir)
    fp64 = load_gpu_mapped(args.output_root / "fp64_exact")
    body_metadata, body_values = load(args.body_mask)
    shape = tuple(int(v) for v in body_metadata["DimSize"].split())
    spacing = tuple(float(v) for v in body_metadata["ElementSpacing"].split())
    if shape != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape}")
    body = body_values > 0.5
    selection = body & (topas["dose"] >= 0.10 * float(np.max(topas["dose"][body])))
    return {
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >=10% BODY Dmax",
        "normalization": "absolute; no fitted normalization",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {"3d_points": 50_000, "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
        "fp64_vs_topas_seed1": metrics(topas["dose"], fp64["dose"], selection, body, shape, spacing,
                                         args.output_root / "gamma_checkpoints" / "fp64_vs_topas_dose.json"),
        "fp32_vs_fp64": metrics(fp32["dose"], fp64["dose"], selection, body, shape, spacing,
                                  args.output_root / "gamma_checkpoints" / "fp32_vs_fp64_dose.json"),
    }


def markdown(report: dict[str, Any]) -> str:
    lines = ["# RT07575 FP64 voxel-dose scorer ablation", "",
             "Same corrected-origin geometry, physics, seed, histories, dose scale, and source. Only the CUDA voxel-dose atomic precision differs.",
             "Mask: RTSTRUCT BODY ∩ TOPAS seed-1 dose ≥10% BODY Dmax. No fitted normalization.", "",
             "## Build and runtime", "",
             "| Arm | Histories | Atomic precision | Elapsed (s) | Throughput (hist/s) | Secondary / cascade / neutral overflow |",
             "|---|---:|---|---:|---:|---:|"]
    for name, stats in report["runs"].items():
        lines.append(f"| {name} | {stats['histories']:,} | {stats['dose_atomic']} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | {stats['secondary_queue_overflow']} / {stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |")
    lines += ["", "## Dose comparison", "",
             "| Pair (evaluation/reference) | Integral E/R | max |Δ| / Dmax | RMSE / Dmax | mean signed Δ / Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, metric in (("FP64 / TOPAS seed 1", report["comparisons"]["fp64_vs_topas_seed1"]),
                          ("FP64 / FP32", report["comparisons"]["fp32_vs_fp64"])):
        gamma = metric["gamma"]
        cells = [f"{gamma[key]['global']['pass_percent']:.3f} / {gamma[key]['local']['pass_percent']:.3f}" for key in ("33", "22", "11", "30")]
        lines.append(f"| {label} | {metric['evaluation_over_reference_body_integral']:.8f} | {metric['selected_max_absolute_difference_over_reference_dmax_percent']:.6g}% | {metric['selected_rmse_over_reference_dmax_percent']:.6g}% | {metric['selected_mean_signed_difference_over_reference_dmax_percent']:.6g}% | " + " | ".join(cells) + " |")
    lines += ["", "FP32-vs-FP64 uses FP32 as the reference and the identical TOPAS-derived mask, so any residual is scorer-precision/numerical ordering rather than calibration.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--fp32-binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--fp64-binary", type=Path, default=Path("build/oneapi-nvidia-release-fp64/carbon_mc"))
    parser.add_argument("--fp32-cache", type=Path, default=Path("build/oneapi-nvidia-release/CMakeCache.txt"))
    parser.add_argument("--fp64-cache", type=Path, default=Path("build/oneapi-nvidia-release-fp64/CMakeCache.txt"))
    parser.add_argument("--fp32-dir", type=Path, default=Path("out/ct/RT07575/edge_origin_ablation/edge_corrected"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/fp64_scorer_ablation"))
    args = parser.parse_args()
    for field, value in vars(args).items():
        if isinstance(value, Path) and not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    for field in ("template", "fp32_binary", "fp64_binary", "fp32_cache", "fp64_cache", "fp32_dir", "topas_dir", "body_mask"):
        if not getattr(args, field).exists():
            raise FileNotFoundError(getattr(args, field))
    if cache_setting(args.fp32_cache) != "ON" or cache_setting(args.fp64_cache) != "OFF":
        raise RuntimeError("build caches do not form an FP32/FP64 scorer A/B")
    fp32_stats = complete(args.fp32_dir, HISTORIES, "fp32")
    if fp32_stats is None:
        raise RuntimeError("corrected-origin FP32 reference is not exact/history-safe/fp32")
    args.output_root.mkdir(parents=True, exist_ok=True)
    smoke = run_arm(args, "fp64_smoke_100k", SMOKE_HISTORIES)
    projected_seconds = HISTORIES / smoke["throughput_histories_per_second"]
    # A measured sub-15-minute projection is practical on the allocated GPU;
    # keeping exact histories preserves direct TOPAS comparison meaning.
    if projected_seconds > 900:
        raise RuntimeError(f"FP64 exact run projected {projected_seconds:.1f} s (>900 s); no exact comparison run")
    exact = run_arm(args, "fp64_exact", HISTORIES)
    report = {
        "case": "RT07575", "experiment": "FP32 versus FP64 voxel-dose atomics",
        "histories": HISTORIES, "seed": SEED,
        "build_audit": {"fp32_cache": str(args.fp32_cache), "fp32_setting": "ON", "fp64_cache": str(args.fp64_cache), "fp64_setting": "OFF", "fp32_binary": str(args.fp32_binary), "fp64_binary": str(args.fp64_binary)},
        "fp32_reference": str(args.fp32_dir), "topas_seed1": str(args.topas_dir),
        "smoke_projected_exact_seconds": projected_seconds,
        "runs": {"fp32_corrected_existing": fp32_stats, "fp64_smoke_100k": smoke, "fp64_exact": exact},
        "comparisons": comparisons(args),
    }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
