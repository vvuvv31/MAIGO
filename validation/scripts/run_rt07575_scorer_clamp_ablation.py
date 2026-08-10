#!/usr/bin/env python3
"""Run the exact-history RT07575 scorer-face transport-clamp A/B.

The no-clamp arm is rendered directly from the current production template,
with only ``voxel_scorer_clamps_transport: false`` added.  GPU transport and
each gamma criterion checkpoint independently, so rerunning this command is
safe after an interruption.
"""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path
import re
import shutil
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
QUANTITIES = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))


def replace_or_append(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    result, count = pattern.subn(f"{key}: {value}", text)
    if count > 1:
        raise ValueError(f"expected at most one {key}, found {count}")
    return result if count else result.rstrip() + f"\n{key}: {value}\n"


def render_config(template: Path, destination: Path, run_dir: Path) -> None:
    text = template.read_text(encoding="utf-8")
    # These reproduce the fixed experiment values even if an unrelated local
    # template edit changed an execution-path value.  No physics or source
    # setting apart from the switch below is altered.
    for key, value in {
        "number_of_histories": str(HISTORIES),
        "random_seed": str(SEED),
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
        "voxel_scorer_clamps_transport": "false",
    }.items():
        text = replace_or_append(text, key, value)
    destination.write_text(text, encoding="utf-8")


def log_number(text: str, pattern: str, default: float | None = None) -> float:
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(found.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    queue_caps = re.search(
        r"^CUDA backend detected: WSL-safe queue caps "
        r"secondary_queue_capacity=(\d+) neutral_queue_capacity=(\d+)$",
        text,
        re.MULTILINE,
    )
    return {
        "histories": int(log_number(text, r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": log_number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": log_number(
            text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"
        ),
        "energy_balance_error": log_number(
            text, r"^Energy balance error:\s*([0-9.eE+-]+)$"
        ),
        "secondary_queue_overflow": int(log_number(
            text, r"^Secondary queue overflow:\s*(\d+)$"
        )),
        "cascade_queue_overflow": int(log_number(
            text, r"^Cascade queue overflow:\s*(\d+)$"
        )),
        "neutral_queue_overflow": int(log_number(
            text, r"^Neutral queue overflow:\s*(\d+)$", 0.0
        )),
        "backend": backend.group(1) if backend else "unknown",
        "cuda_queue_capacities": (
            {"secondary": int(queue_caps.group(1)), "neutral": int(queue_caps.group(2))}
            if queue_caps else None
        ),
    }


def complete(run_dir: Path) -> dict[str, Any] | None:
    required = [run_dir / filename for filename in QUANTITIES.values()] + [
        run_dir / "dose.raw", run_dir / "letd_primary_c12.raw",
        run_dir / "letd_all_hadron.raw", run_dir / "run.log", run_dir / "config.yaml",
    ]
    if not all(path.exists() for path in required):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES or any(stats[key] for key in (
        "secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow"
    )):
        return None
    return stats


def run_no_clamp(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_root / "no_clamp"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = complete(run_dir)
    if saved is not None:
        print("SKIP complete no_clamp", flush=True)
        return saved
    config = run_dir / "config.yaml"
    render_config(args.template, config, run_dir)
    print("RUN no_clamp", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [str(args.binary), "--config", str(config), "--device", "cuda"],
            cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"no_clamp failed ({result.returncode}); see {run_dir / 'run.log'}")
    saved = complete(run_dir)
    if saved is None:
        raise RuntimeError("no_clamp is incomplete, has wrong histories, or has queue overflows")
    return saved


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def hydrate_baseline_checkpoint(source: Path, cache_root: Path) -> None:
    """Reuse the established exact baseline checkpoint, after strict checks.

    The baseline GPU result is fixed by the experiment definition and already
    has an exact-history TOPAS comparison under the same implementation as
    this runner.  This avoids needlessly recomputing its costly deterministic
    gamma samples while retaining a private, resumable checkpoint copy.
    """
    data = json.loads(source.read_text(encoding="utf-8"))
    for quantity in QUANTITIES:
        gamma = data[quantity]["gamma"]
        for _, _, label in CRITERIA:
            if not {"global", "local"}.issubset(gamma.get(label, {})):
                raise ValueError(f"incomplete baseline checkpoint: {quantity}/{label}")
        if not {"global", "local"}.issubset(gamma.get("30", {})):
            raise ValueError(f"incomplete baseline checkpoint: {quantity}/30")
    destination = cache_root / "baseline" / "all_quantities.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def checkpointed_quantity(
    reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray, body: np.ndarray,
    shape_xyz: tuple[int, int, int], spacing_xyz: tuple[float, float, float], cache_path: Path,
) -> dict[str, Any]:
    """Use the established gamma functions, checkpointing every criterion."""
    item: dict[str, Any] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
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
                100.0 * np.sqrt(np.mean(delta * delta)) / max(float(np.max(ref_selected)), 1.0e-30)
            ),
            "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]),
            "gamma": {},
        }
        write_json(cache_path, item)
    ref_flat = array.array("f", reference_masked.astype(np.float32, copy=False))
    eval_flat = evaluation_masked.reshape(-1).tolist()
    gamma: dict[str, Any] = item["gamma"]
    for dose_percent, distance_mm, label in CRITERIA:
        criterion = gamma.setdefault(label, {})
        for local, mode in ((False, "global"), (True, "local")):
            if mode not in criterion:
                print(f"Computing {cache_path.parent.parent.name}/{cache_path.stem}: {label} {mode}", flush=True)
                criterion[mode] = gamma_3d(
                    ref_flat, eval_flat, shape_xyz, spacing_xyz, dose_percent, distance_mm,
                    10.0, 50_000, 0, local_dose=local, interpolation_step_mm=0.5,
                    selection_mask=selection.reshape(-1),
                )
                write_json(cache_path, item)
    if "30" not in gamma:
        gamma["30"] = {
            "global": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0,
                                            selection_mask=selection.reshape(-1)),
            "local": dose_only_pass_rate(ref_flat, eval_flat, 3.0, 10.0, local_dose=True,
                                           selection_mask=selection.reshape(-1)),
        }
        write_json(cache_path, item)
    return item


def comparison(topas_dir: Path, gpu_dir: Path, body_mask: Path, cache_root: Path, arm: str) -> dict[str, Any]:
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
    if arm == "baseline" and (cache_root / arm / "all_quantities.json").exists():
        quantities = json.loads((cache_root / arm / "all_quantities.json").read_text(encoding="utf-8"))
    else:
        quantities = {
            name: checkpointed_quantity(topas[name], gpu[name], selection, body, shape_xyz,
                                        spacing_xyz, cache_root / arm / f"{name}.json")
            for name in QUANTITIES
        }
    return {
        "reference": str(topas_dir), "evaluation": str(gpu_dir),
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {"3d_points": 50_000, "available_points": int(np.count_nonzero(selection)),
                           "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
        "quantities": quantities,
    }


def strict_dose_deltas(baseline: dict[str, Any], no_clamp: dict[str, Any]) -> dict[str, float]:
    base = baseline["quantities"]["dose"]["gamma"]
    trial = no_clamp["quantities"]["dose"]["gamma"]
    return {
        "1pct_1mm_global_pp": trial["11"]["global"]["pass_percent"] - base["11"]["global"]["pass_percent"],
        "1pct_1mm_local_pp": trial["11"]["local"]["pass_percent"] - base["11"]["local"]["pass_percent"],
        "3pct_0mm_global_pp": trial["30"]["global"]["pass_percent"] - base["30"]["global"]["pass_percent"],
        "3pct_0mm_local_pp": trial["30"]["local"]["pass_percent"] - base["30"]["local"]["pass_percent"],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RT07575 scorer-face transport-clamp ablation", "",
        f"Both GPU arms use {HISTORIES:,} histories with seed {SEED}. No fitted normalization.",
        "The no-clamp arm is the current template with only `voxel_scorer_clamps_transport: false`.",
        "Mask: RTSTRUCT BODY ∩ TOPAS seed-1 dose ≥10% BODY Dmax; LET has no additional threshold.",
        "3D gamma: deterministic 50,000 points, 0.5 mm interpolation; 3%/0 mm uses all selected voxels.", "",
        "## Runtime and safety", "",
        "| Arm | Elapsed (s) | Throughput (hist/s) | Energy balance error | Secondary / cascade / neutral overflow |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, stats in report["runs"].items():
        lines.append(f"| {arm} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | "
                     f"{stats['energy_balance_error']:.8g} | {stats['secondary_queue_overflow']} / "
                     f"{stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |")
    for arm, result in report["comparisons"].items():
        lines += ["", f"## {arm} vs TOPAS seed 1", "",
                  "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for quantity, metric in result["quantities"].items():
            gamma = metric["gamma"]
            cells = [f"{gamma[key]['global']['pass_percent']:.3f} / {gamma[key]['local']['pass_percent']:.3f}"
                     for key in ("33", "22", "11", "30")]
            lines.append(f"| {quantity} | {metric['evaluation_over_reference_body_integral']:.6f} | "
                         f"{metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | "
                         f"{metric['selected_nrmse_over_reference_max_percent']:.3f}% | " + " | ".join(cells) + " |")
    delta = report["strict_dose_deltas_vs_baseline_pp"]
    lines += ["", "## Strict dose-gamma deltas: no clamp minus baseline", "",
              "| 1%/1mm G | 1%/1mm L | 3%/0mm G | 3%/0mm L |",
              "|---:|---:|---:|---:|",
              f"| {delta['1pct_1mm_global_pp']:+.3f} pp | {delta['1pct_1mm_local_pp']:+.3f} pp | "
              f"{delta['3pct_0mm_global_pp']:+.3f} pp | {delta['3pct_0mm_local_pp']:+.3f} pp |", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("out/ct/RT07575/equal_history_best_seeds/seed20260801"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/scorer_clamp_ablation"))
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=Path("out/ct/RT07575/physics_ablation/gamma_checkpoints/baseline.json"))
    args = parser.parse_args()
    for field in ("binary", "template", "baseline_dir", "topas_dir", "body_mask", "output_root", "baseline_checkpoint"):
        value = getattr(args, field)
        if not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    for path in (args.binary, args.template, args.baseline_dir, args.topas_dir, args.body_mask, args.baseline_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    baseline_stats = parse_log(args.baseline_dir / "run.log")
    if baseline_stats["histories"] != HISTORIES or any(baseline_stats[key] for key in (
        "secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow"
    )):
        raise RuntimeError("baseline does not meet exact-history/zero-overflow requirement")
    args.output_root.mkdir(parents=True, exist_ok=True)
    no_clamp_stats = run_no_clamp(args)
    cache = args.output_root / "gamma_checkpoints"
    hydrate_baseline_checkpoint(args.baseline_checkpoint, cache)
    comparisons = {
        "baseline": comparison(args.topas_dir, args.baseline_dir, args.body_mask, cache, "baseline"),
        "no_clamp": comparison(args.topas_dir, args.output_root / "no_clamp", args.body_mask, cache, "no_clamp"),
    }
    report = {
        "case": "RT07575", "experiment": "disable scorer-face transport clamping",
        "histories": HISTORIES, "seed": SEED, "template": str(args.template),
        "only_override": {"voxel_scorer_clamps_transport": False},
        "baseline_gpu_seed1": str(args.baseline_dir), "topas_seed1": str(args.topas_dir),
        "runs": {"baseline": baseline_stats, "no_clamp": no_clamp_stats},
        "comparisons": comparisons,
        "strict_dose_deltas_vs_baseline_pp": strict_dose_deltas(comparisons["baseline"], comparisons["no_clamp"]),
    }
    write_json(args.output_root / "summary.json", report)
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
