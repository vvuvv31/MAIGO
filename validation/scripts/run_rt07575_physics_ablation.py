#!/usr/bin/env python3
"""Run and compare exact-history single-switch RT07575 physics ablations.

Each arm is rendered from the production best-profile template.  Results are
resumable: a completed arm has its dose/LET MHD outputs, exact history count,
and zero secondary, cascade, and neutral queue overflows.
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
from compare_rt07575_equal_history_groups import (  # noqa: E402
    load_gpu_mapped,
    load_topas,
)
from compare_topas_seed_gamma import load  # noqa: E402
from match_gpu_to_physical_dose import dose_only_pass_rate, gamma_3d  # noqa: E402


HISTORIES = 12_963_817
SEED = 20_260_801
FILES_GPU = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}
# first_interaction uses the transport's documented four slots/history headroom.
NEUTRAL_QUEUE_CAPACITY = 4 * HISTORIES
VARIANTS: dict[str, dict[str, str]] = {
    "mcs_only": {"enable_ct_material_mcs": "true"},
    "secondary_straggling_only": {"enable_secondary_energy_straggling": "true"},
    "neutral_only": {
        "enable_neutral_transport": "true",
        "neutral_transport_mode": "first_interaction",
        "maximum_neutral_generations": "1",
        "neutral_queue_capacity": str(NEUTRAL_QUEUE_CAPACITY),
        "neutral_package_file": "data/packages/topas_200MeVu_neutral_development.bin",
    },
}


def replace_or_append(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count > 1:
        raise ValueError(f"expected at most one {key}, found {count}")
    return updated if count else updated.rstrip() + f"\n{key}: {value}\n"


def render_config(template: Path, output: Path, run_dir: Path, variant: str) -> None:
    text = template.read_text(encoding="utf-8")
    overrides = {
        "number_of_histories": str(HISTORIES),
        "random_seed": str(SEED),
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
        **VARIANTS[variant],
    }
    for key, value in overrides.items():
        text = replace_or_append(text, key, value)
    output.write_text(text, encoding="utf-8")


def parsed_number(text: str, pattern: str, default: float | None = None) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(match.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    cuda_queue = re.search(
        r"^CUDA backend detected: WSL-safe queue caps "
        r"secondary_queue_capacity=(\d+) neutral_queue_capacity=(\d+)$",
        text,
        re.MULTILINE,
    )
    memory = re.search(r"^Device memory (?:estimate|budget):\s*(.+)$", text, re.MULTILINE)
    return {
        "histories": int(parsed_number(text, r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": parsed_number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": parsed_number(
            text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"
        ),
        "energy_balance_error": parsed_number(
            text, r"^Energy balance error:\s*([0-9.eE+-]+)$"
        ),
        "secondary_queue_overflow": int(parsed_number(
            text, r"^Secondary queue overflow:\s*(\d+)$"
        )),
        "cascade_queue_overflow": int(parsed_number(
            text, r"^Cascade queue overflow:\s*(\d+)$"
        )),
        "neutral_queue_overflow": int(parsed_number(
            text, r"^Neutral queue overflow:\s*(\d+)$", 0.0
        )),
        "backend": backend.group(1) if backend else "unknown",
        "cuda_queue_capacities": (
            {"secondary": int(cuda_queue.group(1)), "neutral": int(cuda_queue.group(2))}
            if cuda_queue else None
        ),
        "device_memory": memory.group(1) if memory else None,
    }


def complete(run_dir: Path) -> dict[str, Any] | None:
    required = [run_dir / name for name in FILES_GPU.values()]
    required += [
        run_dir / "dose.raw",
        run_dir / "letd_primary_c12.raw",
        run_dir / "letd_all_hadron.raw",
        run_dir / "run.log",
        run_dir / "config.yaml",
    ]
    if not all(path.exists() for path in required):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES:
        return None
    if any(stats[key] for key in (
        "secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow"
    )):
        return None
    return stats


def run_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    run_dir = args.output_root / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    stats = complete(run_dir)
    if stats is not None:
        print(f"SKIP complete {variant}", flush=True)
        return stats
    config = run_dir / "config.yaml"
    render_config(args.template, config, run_dir, variant)
    print(f"RUN {variant}", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            [str(args.binary), "--config", str(config), "--device", "cuda"],
            cwd=args.repo_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{variant} failed ({completed.returncode}); see {run_dir / 'run.log'}")
    stats = complete(run_dir)
    if stats is None:
        raise RuntimeError(f"{variant} is incomplete, has wrong histories, or overflowed")
    return stats


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def cached_quantity_report(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    body: np.ndarray,
    shape_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    cache_path: Path,
    quantity: str,
) -> dict[str, Any]:
    """Calculate one gamma criterion at a time and checkpoint after each.

    A full six-criterion 50k-point 3D gamma exceeds the interactive command
    watchdog in this environment.  These checkpoints make the computational
    phase resumable independently of the (already completed) GPU transports.
    """
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    item = cache.get(quantity, {})
    reference_masked = np.where(body, reference, 0.0)
    evaluation_masked = np.where(body, evaluation, 0.0)
    ref_selected = reference[selection]
    eval_selected = evaluation[selection]
    if not item:
        delta = eval_selected - ref_selected
        item = {
            "selected_voxels": int(np.count_nonzero(selection)),
            "evaluation_over_reference_body_integral": float(
                np.sum(evaluation_masked, dtype=np.float64)
                / max(np.sum(reference_masked, dtype=np.float64), 1.0e-30)
            ),
            "selected_mean_reference": float(np.mean(ref_selected)),
            "selected_mean_evaluation": float(np.mean(eval_selected)),
            "selected_nrmse_over_reference_max_percent": float(
                100.0 * np.sqrt(np.mean(delta * delta))
                / max(float(np.max(ref_selected)), 1.0e-30)
            ),
            "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]),
            "gamma": {},
        }
        cache[quantity] = item
        write_json(cache_path, cache)
    ref_flat = array.array("f", reference_masked.astype(np.float32, copy=False))
    eval_flat = evaluation_masked.reshape(-1).tolist()
    gamma = item["gamma"]
    assert isinstance(gamma, dict)
    for dose_percent, distance_mm, label in ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11")):
        criterion = gamma.setdefault(label, {})
        assert isinstance(criterion, dict)
        for local, mode in ((False, "global"), (True, "local")):
            if mode in criterion:
                continue
            print(f"Computing {cache_path.stem}: {quantity} {label} {mode}", flush=True)
            criterion[mode] = gamma_3d(
                ref_flat, eval_flat, shape_xyz, spacing_xyz, dose_percent,
                distance_mm, 10.0, 50_000, 0, local_dose=local,
                interpolation_step_mm=0.5, selection_mask=selection.reshape(-1),
            )
            cache[quantity] = item
            write_json(cache_path, cache)
    if "30" not in gamma:
        gamma["30"] = {
            "global": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0,
                                            selection_mask=selection.reshape(-1)),
            "local": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0,
                                           local_dose=True, selection_mask=selection.reshape(-1)),
        }
        cache[quantity] = item
        write_json(cache_path, cache)
    return item


def comparison(topas_dir: Path, gpu_dir: Path, body_mask: Path, cache_path: Path) -> dict[str, Any]:
    topas = load_topas(topas_dir)
    gpu = load_gpu_mapped(gpu_dir)
    body_meta, body_values = load(body_mask)
    shape_xyz = tuple(int(v) for v in body_meta["DimSize"].split())
    spacing_xyz = tuple(float(v) for v in body_meta["ElementSpacing"].split())
    if shape_xyz != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape_xyz}")
    body = body_values > 0.5
    topas_peak = float(np.max(topas["dose"][body]))
    selection = body & (topas["dose"] >= 0.10 * topas_peak)
    quantities: dict[str, Any] = {}
    for name in FILES_GPU:
        quantities[name] = cached_quantity_report(
            topas[name], gpu[name], selection, body, shape_xyz, spacing_xyz,
            cache_path, name,
        )
    return {
        "reference": str(topas_dir),
        "evaluation": str(gpu_dir),
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {
            "3d_points": 50_000,
            "available_points": int(np.count_nonzero(selection)),
            "interpolation_step_mm": 0.5,
            "3pct_0mm_points": "all selected voxels",
        },
        "quantities": quantities,
    }


def strict_delta(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    base_gamma = baseline["quantities"]["dose"]["gamma"]
    trial_gamma = variant["quantities"]["dose"]["gamma"]
    deltas = {
        "1pct_1mm_global_pp": trial_gamma["11"]["global"]["pass_percent"] - base_gamma["11"]["global"]["pass_percent"],
        "1pct_1mm_local_pp": trial_gamma["11"]["local"]["pass_percent"] - base_gamma["11"]["local"]["pass_percent"],
        "3pct_0mm_global_pp": trial_gamma["30"]["global"]["pass_percent"] - base_gamma["30"]["global"]["pass_percent"],
        "3pct_0mm_local_pp": trial_gamma["30"]["local"]["pass_percent"] - base_gamma["30"]["local"]["pass_percent"],
    }
    values = list(deltas.values())
    direction = "improves" if min(values) > 0 else "degrades" if max(values) < 0 else "mixed"
    return {"classification": direction, "pass_rate_delta_percentage_points": deltas}


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RT07575 stage-1 equal-history physics ablation",
        "",
        f"Every result uses {HISTORIES:,} histories with seed {SEED}. No fitted normalization.",
        "Mask: RTSTRUCT BODY ∩ TOPAS seed 1 dose ≥10% BODY Dmax; dose and LET use this same mask.",
        "3D gamma: deterministic 50,000 points, 0.5 mm trilinear interpolation; 3%/0 mm uses all selected voxels.",
        "",
        "## Runtime and safety",
        "",
        "| Arm | Elapsed (s) | Throughput (hist/s) | Energy balance error | Secondary / cascade / neutral overflow |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, item in report["runs"].items():
        lines.append(
            f"| {arm} | {item['elapsed_seconds']:.3f} | {item['throughput_histories_per_second']:.1f} | "
            f"{item['energy_balance_error']:.8g} | {item['secondary_queue_overflow']} / "
            f"{item['cascade_queue_overflow']} / {item['neutral_queue_overflow']} |"
        )
    lines += ["", "## TOPAS seed-1 comparison", ""]
    for arm, item in report["comparisons"].items():
        lines += [
            f"### {arm}",
            "",
            "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for quantity, metric in item["quantities"].items():
            gamma = metric["gamma"]
            cells = [
                f"{gamma[label]['global']['pass_percent']:.3f} / {gamma[label]['local']['pass_percent']:.3f}"
                for label in ("33", "22", "11", "30")
            ]
            lines.append(
                f"| {quantity} | {metric['evaluation_over_reference_body_integral']:.6f} | "
                f"{metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | "
                f"{metric['selected_nrmse_over_reference_max_percent']:.3f}% | "
                + " | ".join(cells) + " |"
            )
        lines.append("")
    lines += ["## Strict dose-gamma deltas vs baseline", "", "| Switch | Assessment | 1%/1mm G | 1%/1mm L | 3%/0mm G | 3%/0mm L |", "|---|---|---:|---:|---:|---:|"]
    for name, item in report["strict_deltas"].items():
        delta = item["pass_rate_delta_percentage_points"]
        lines.append(
            f"| {name} | {item['classification']} | {delta['1pct_1mm_global_pp']:+.3f} pp | "
            f"{delta['1pct_1mm_local_pp']:+.3f} pp | {delta['3pct_0mm_global_pp']:+.3f} pp | "
            f"{delta['3pct_0mm_local_pp']:+.3f} pp |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("out/ct/RT07575/equal_history_best_seeds/seed20260801"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/physics_ablation"))
    args = parser.parse_args()
    for name in ("binary", "template", "topas_dir", "baseline_dir", "body_mask", "output_root"):
        value = getattr(args, name)
        if not value.is_absolute():
            setattr(args, name, args.repo_root / value)
    for required in (args.binary, args.template, args.topas_dir, args.baseline_dir, args.body_mask):
        if not required.exists():
            raise FileNotFoundError(required)

    args.output_root.mkdir(parents=True, exist_ok=True)
    runs: dict[str, Any] = {"baseline": parse_log(args.baseline_dir / "run.log")}
    for variant in VARIANTS:
        runs[variant] = run_variant(args, variant)

    cache_root = args.output_root / "gamma_checkpoints"
    comparisons: dict[str, Any] = {
        "baseline": comparison(args.topas_dir, args.baseline_dir, args.body_mask,
                               cache_root / "baseline.json")
    }
    for variant in VARIANTS:
        comparisons[variant] = comparison(args.topas_dir, args.output_root / variant,
                                          args.body_mask, cache_root / f"{variant}.json")
    report: dict[str, Any] = {
        "case": "RT07575",
        "stage": "stage-1 individual equal-history physics switches",
        "histories": HISTORIES,
        "seed": SEED,
        "template": str(args.template),
        "topas_seed1": str(args.topas_dir),
        "baseline_gpu_seed1": str(args.baseline_dir),
        "neutral_settings": {
            "package": "data/packages/topas_200MeVu_neutral_development.bin",
            "transport_mode": "first_interaction",
            "maximum_neutral_generations": 1,
            "neutral_queue_capacity": NEUTRAL_QUEUE_CAPACITY,
            "memory_note": "four slots/history documented headroom; CUDA enforces its device-memory budget",
        },
        "runs": runs,
        "comparisons": comparisons,
        "strict_deltas": {
            variant: strict_delta(comparisons["baseline"], comparisons[variant])
            for variant in VARIANTS
        },
    }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
