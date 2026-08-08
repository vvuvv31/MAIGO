#!/usr/bin/env python3
"""Exact-history RT07575 A/B: clamp every CT voxel face versus the best baseline.

Only ``ct_skip_homogeneous_face_clamp: false`` changes transport behavior.  The
run and each gamma result are independently resumable.
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
SEED = 20_260_801
QUANTITIES = {
    "dose": "dose.mhd",
    "primary_c12_letd": "letd_primary_c12.mhd",
    "all_hadron_letd": "letd_all_hadron.mhd",
}
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))


def replace_or_append(text: str, key: str, value: str) -> str:
    updated, count = re.subn(rf"^{re.escape(key)}:.*$", f"{key}: {value}", text,
                             flags=re.MULTILINE)
    if count > 1:
        raise ValueError(f"expected at most one {key}, found {count}")
    return updated if count else updated.rstrip() + f"\n{key}: {value}\n"


def render_config(template: Path, destination: Path, run_dir: Path) -> None:
    text = template.read_text(encoding="utf-8")
    # History/seed values are fixed experiment invariants, and the paths merely
    # isolate artifacts.  The sole transport/physics change is the final key.
    for key, value in {
        "number_of_histories": str(HISTORIES), "random_seed": str(SEED),
        "voxel_dose_mhd_output_file": str(run_dir / "dose.mhd"),
        "let_voxel_mhd_output_file": str(run_dir / "letd"),
        "ct_skip_homogeneous_face_clamp": "false",
    }.items():
        text = replace_or_append(text, key, value)
    destination.write_text(text, encoding="utf-8")


def number(text: str, pattern: str, default: float | None = None) -> float:
    found = re.search(pattern, text, re.MULTILINE)
    if found is None:
        if default is None:
            raise ValueError(f"missing log field {pattern}")
        return default
    return float(found.group(1))


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    profile = {}
    profile_match = re.search(r"^Transport profile counters:\n(?P<body>(?:^  .+\n)+)", text,
                              re.MULTILINE)
    if profile_match:
        for name, value in re.findall(r"^  ([a-z0-9_]+):\s*(\d+)$",
                                      profile_match.group("body"), re.MULTILINE):
            profile[name] = int(value)
    backend = re.search(r"^Backend:\s*(.+)$", text, re.MULTILINE)
    queue_caps = re.search(r"^CUDA backend detected: WSL-safe queue caps "
                           r"secondary_queue_capacity=(\d+) neutral_queue_capacity=(\d+)$",
                           text, re.MULTILINE)
    stats: dict[str, Any] = {
        "histories": int(number(text, r"^Histories:\s*(\d+)$")),
        "elapsed_seconds": number(text, r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": number(text, r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"),
        "energy_balance_error": number(text, r"^Energy balance error:\s*([0-9.eE+-]+)$"),
        "secondary_queue_overflow": int(number(text, r"^Secondary queue overflow:\s*(\d+)$")),
        "cascade_queue_overflow": int(number(text, r"^Cascade queue overflow:\s*(\d+)$")),
        "neutral_queue_overflow": int(number(text, r"^Neutral queue overflow:\s*(\d+)$", 0)),
        "backend": backend.group(1) if backend else "unknown",
        "transport_profile_counters": profile or None,
    }
    if queue_caps:
        stats["cuda_queue_capacities"] = {"secondary": int(queue_caps.group(1)),
                                          "neutral": int(queue_caps.group(2))}
    return stats


def complete(run_dir: Path) -> dict[str, Any] | None:
    needed = [run_dir / filename for filename in QUANTITIES.values()] + [
        run_dir / "dose.raw", run_dir / "letd_primary_c12.raw", run_dir / "letd_all_hadron.raw",
        run_dir / "run.log", run_dir / "config.yaml",
    ]
    if not all(path.exists() for path in needed):
        return None
    stats = parse_log(run_dir / "run.log")
    if stats["histories"] != HISTORIES or any(stats[key] for key in
            ("secondary_queue_overflow", "cascade_queue_overflow", "neutral_queue_overflow")):
        return None
    return stats


def run_exact_face(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_root / "exact_ct_faces"
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = complete(run_dir)
    if saved is not None:
        print("SKIP complete exact_ct_faces", flush=True)
        return saved
    config = run_dir / "config.yaml"
    render_config(args.template, config, run_dir)
    print("RUN exact_ct_faces", flush=True)
    with (run_dir / "run.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(args.binary), "--config", str(config), "--device", "cuda"],
                                cwd=args.repo_root, stdout=stream, stderr=subprocess.STDOUT,
                                text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"exact_ct_faces failed ({result.returncode}); see {run_dir / 'run.log'}")
    saved = complete(run_dir)
    if saved is None:
        raise RuntimeError("exact_ct_faces is incomplete, has wrong histories, or overflowed")
    return saved


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checkpoint_quantity(reference: np.ndarray, evaluation: np.ndarray, selection: np.ndarray,
                        body: np.ndarray, shape: tuple[int, int, int], spacing: tuple[float, float, float],
                        cache_path: Path) -> dict[str, Any]:
    item: dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ref_body, eval_body = np.where(body, reference, 0.0), np.where(body, evaluation, 0.0)
    ref_selected, eval_selected = reference[selection], evaluation[selection]
    if not item:
        delta = eval_selected - ref_selected
        item = {
            "selected_voxels": int(np.count_nonzero(selection)),
            "evaluation_over_reference_body_integral": float(np.sum(eval_body, dtype=np.float64) /
                                                               max(np.sum(ref_body, dtype=np.float64), 1e-30)),
            "selected_mean_reference": float(np.mean(ref_selected)),
            "selected_mean_evaluation": float(np.mean(eval_selected)),
            "selected_nrmse_over_reference_max_percent": float(100 * np.sqrt(np.mean(delta * delta)) /
                                                                 max(float(np.max(ref_selected)), 1e-30)),
            "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]), "gamma": {},
        }
        write_json(cache_path, item)
    ref_flat = array.array("f", ref_body.astype(np.float32, copy=False))
    eval_flat = eval_body.reshape(-1).tolist()
    gamma: dict[str, Any] = item["gamma"]
    for percent, distance, label in CRITERIA:
        criterion = gamma.setdefault(label, {})
        for local, mode in ((False, "global"), (True, "local")):
            if mode not in criterion:
                print(f"Computing {cache_path.stem}: {label} {mode}", flush=True)
                criterion[mode] = gamma_3d(ref_flat, eval_flat, shape, spacing, percent, distance,
                                           10.0, 50_000, 0, local_dose=local,
                                           interpolation_step_mm=0.5,
                                           selection_mask=selection.reshape(-1))
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


def comparison(topas_dir: Path, gpu_dir: Path, body_mask: Path, cache_root: Path) -> dict[str, Any]:
    topas, gpu = load_topas(topas_dir), load_gpu_mapped(gpu_dir)
    metadata, body_values = load(body_mask)
    shape = tuple(int(v) for v in metadata["DimSize"].split())
    spacing = tuple(float(v) for v in metadata["ElementSpacing"].split())
    if shape != (417, 505, 35):
        raise ValueError(f"unexpected BODY shape {shape}")
    body = body_values > 0.5
    selection = body & (topas["dose"] >= 0.10 * float(np.max(topas["dose"][body])))
    return {
        "reference": str(topas_dir), "evaluation": str(gpu_dir),
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
        "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "gamma_sampling": {"3d_points": 50_000, "available_points": int(np.count_nonzero(selection)),
                           "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
        "quantities": {name: checkpoint_quantity(topas[name], gpu[name], selection, body, shape, spacing,
                                                   cache_root / f"{name}.json") for name in QUANTITIES},
    }


def deltas(baseline: dict[str, Any], exact: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    return {quantity: {label: {mode: exact["quantities"][quantity]["gamma"][label][mode]["pass_percent"] -
                                      baseline["quantities"][quantity]["gamma"][label][mode]["pass_percent"]
                               for mode in ("global", "local")}
                       for _, _, label in CRITERIA + ((3.0, 0.0, "30"),)} for quantity in QUANTITIES}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# RT07575 exact CT-face clamping A/B", "",
             f"Both GPU arms use {HISTORIES:,} histories with seed {SEED}. No fitted normalization.",
             "Exact-face changes only `ct_skip_homogeneous_face_clamp: false`; it clamps every non-micro transport step to the next CT face.",
             "Mask: RTSTRUCT BODY ∩ TOPAS seed-1 dose ≥10% BODY Dmax; LET uses this same mask.",
             "3D gamma: deterministic 50,000 points, 0.5 mm interpolation; 3%/0 mm uses all selected voxels.", "",
             "## Runtime and safety", "",
             "| Arm | Elapsed (s) | Throughput (hist/s) | Energy balance error | Secondary / cascade / neutral overflow |",
             "|---|---:|---:|---:|---:|"]
    for arm, stats in report["runs"].items():
        lines.append(f"| {arm} | {stats['elapsed_seconds']:.3f} | {stats['throughput_histories_per_second']:.1f} | {stats['energy_balance_error']:.8g} | {stats['secondary_queue_overflow']} / {stats['cascade_queue_overflow']} / {stats['neutral_queue_overflow']} |")
    for arm, result in report["comparisons"].items():
        lines += ["", f"## {arm} vs TOPAS seed 1", "",
                  "| Quantity | BODY integral E/R | mask mean E/R | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, metric in result["quantities"].items():
            gamma = metric["gamma"]
            values = [f"{gamma[label]['global']['pass_percent']:.3f} / {gamma[label]['local']['pass_percent']:.3f}" for label in ("33", "22", "11", "30")]
            lines.append(f"| {name} | {metric['evaluation_over_reference_body_integral']:.6f} | {metric['selected_mean_evaluation'] / metric['selected_mean_reference']:.6f} | {metric['selected_nrmse_over_reference_max_percent']:.3f}% | " + " | ".join(values) + " |")
    lines += ["", "## Gamma deltas: exact-face minus baseline (pp)", "",
              "| Quantity | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |", "|---|---:|---:|---:|---:|"]
    for name, value in report["gamma_deltas_vs_baseline_pp"].items():
        vals = [f"{value[label]['global']:+.3f} / {value[label]['local']:+.3f}" for label in ("33", "22", "11", "30")]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc"))
    parser.add_argument("--template", type=Path, default=Path("config/beam_ct_fullplan_rt07575_let_soft_tissue.yaml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("out/ct/RT07575/equal_history_best_seeds/seed20260801"))
    parser.add_argument("--topas-dir", type=Path, default=Path("out/fullplan_result/RT07575/topas"))
    parser.add_argument("--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path("out/ct/RT07575/physics_ablation/gamma_checkpoints/baseline.json"))
    parser.add_argument("--output-root", type=Path, default=Path("out/ct/RT07575/exact_ct_face_ablation"))
    args = parser.parse_args()
    for field in ("binary", "template", "baseline_dir", "topas_dir", "body_mask", "baseline_checkpoint", "output_root"):
        value = getattr(args, field)
        if not value.is_absolute():
            setattr(args, field, args.repo_root / value)
    for path in (args.binary, args.template, args.baseline_dir, args.topas_dir, args.body_mask, args.baseline_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    baseline_stats = complete(args.baseline_dir)
    if baseline_stats is None:
        raise RuntimeError("baseline does not meet exact-history/zero-overflow requirement")
    baseline_quantities = json.loads(args.baseline_checkpoint.read_text(encoding="utf-8"))
    if set(baseline_quantities) != set(QUANTITIES):
        raise ValueError("baseline gamma checkpoint quantities do not match")
    for quantity in QUANTITIES:
        for _, _, label in CRITERIA + ((3.0, 0.0, "30"),):
            if not {"global", "local"}.issubset(baseline_quantities[quantity]["gamma"].get(label, {})):
                raise ValueError(f"incomplete baseline gamma: {quantity}/{label}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    exact_stats = run_exact_face(args)
    comparisons = {
        "baseline": {"reference": str(args.topas_dir), "evaluation": str(args.baseline_dir),
                     "normalization": "absolute equal-history scale; no fitted normalization",
                     "selection": "RTSTRUCT BODY and TOPAS seed 1 dose >= 10% BODY Dmax",
                     "let_selection": "same TOPAS seed 1 dose mask; no LET threshold",
                     "selected_voxels": 318711,
                     "gamma_sampling": {"3d_points": 50_000, "available_points": 318711,
                                        "interpolation_step_mm": 0.5, "3pct_0mm_points": "all selected voxels"},
                     "quantities": baseline_quantities},
        "exact_ct_faces": comparison(args.topas_dir, args.output_root / "exact_ct_faces", args.body_mask,
                                     args.output_root / "gamma_checkpoints" / "exact_ct_faces"),
    }
    report = {"case": "RT07575", "experiment": "exact CT-face clamping",
              "histories": HISTORIES, "seed": SEED, "template": str(args.template),
              "only_transport_override": {"ct_skip_homogeneous_face_clamp": False},
              "baseline_gpu_seed1": str(args.baseline_dir), "topas_seed1": str(args.topas_dir),
              "runs": {"baseline": baseline_stats, "exact_ct_faces": exact_stats},
              "comparisons": comparisons,
              "gamma_deltas_vs_baseline_pp": deltas(comparisons["baseline"], comparisons["exact_ct_faces"])}
    write_json(args.output_root / "summary.json", report)
    (args.output_root / "summary.md").write_text(markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
