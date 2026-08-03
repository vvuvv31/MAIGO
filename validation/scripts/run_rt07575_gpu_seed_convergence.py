#!/usr/bin/env python3
"""Run and analyse RT07575 non-minibeam GPU seed convergence.

The comparison is deliberately GPU-to-GPU: seed 20260801 is the reference and
seed 20260802 is the evaluation.  Both use the same profile, total plan history
budget, weighted spot allocation, CT geometry, and absolute dose scale.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from match_gpu_to_physical_dose import (  # noqa: E402
    dose_only_pass_rate,
    gamma_3d,
    read_mhd,
    write_mhd,
)


SEEDS = (20260801, 20260802)
CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))
HISTORY_LINE = re.compile(
    r"^(iv:Tf/Scatterer1/L4/Values\s*=\s*\d+\s+)(.*?)(\s*)$",
    re.MULTILINE,
)


def replace_one(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    updated, count = pattern.subn(f"{key}: {value}", text)
    if count != 1:
        raise ValueError(f"expected exactly one {key}, found {count}")
    return updated


def render_config(
    template: Path,
    output: Path,
    dose_output: Path,
    spots_file: Path,
    histories: int,
    seed: int,
) -> None:
    text = template.read_text(encoding="utf-8")
    text = replace_one(text, "number_of_histories", str(histories))
    text = replace_one(text, "random_seed", str(seed))
    text = replace_one(text, "topas_spots_file", str(spots_file))
    text = replace_one(text, "voxel_dose_mhd_output_file", str(dose_output))
    # Preserve LET scoring in best so its measured speed remains representative,
    # but omit LET files because this experiment compares dose seed convergence.
    text = replace_one(text, "let_voxel_mhd_output_file", "")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def write_scaled_spots(source: Path, output: Path, target_total: int) -> None:
    """Hamilton allocation with one history retained for every active spot."""
    text = source.read_text(encoding="utf-8")
    match = HISTORY_LINE.search(text)
    if match is None:
        raise ValueError(f"{source}: missing L4 history values")
    original = [int(value) for value in match.group(2).split()]
    active = [index for index, value in enumerate(original) if value > 0]
    if target_total < len(active):
        raise ValueError(
            f"target {target_total} is smaller than {len(active)} active spots"
        )
    original_total = sum(original)
    distributable = target_total - len(active)
    allocated = [0] * len(original)
    remainders = []
    assigned = len(active)
    for index in active:
        exact = distributable * original[index] / original_total
        base = math.floor(exact)
        allocated[index] = 1 + base
        assigned += base
        remainders.append((exact - base, original[index], -index, index))
    remainders.sort(reverse=True)
    for offset in range(target_total - assigned):
        allocated[remainders[offset][3]] += 1
    if sum(allocated) != target_total:
        raise AssertionError("scaled spot allocation does not match target")
    replacement = match.group(1) + " ".join(str(value) for value in allocated)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text[: match.start()] + replacement + text[match.end() :], encoding="utf-8")


def parse_log(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8")

    def number(pattern: str) -> float:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path}: missing {pattern}")
        return float(match.group(1))

    def integer(pattern: str) -> int:
        return int(number(pattern))

    device_match = re.search(r"^(?:Device|SYCL device):\s*(.+)$", text, re.MULTILINE)
    return {
        "device": device_match.group(1) if device_match else "unknown",
        "histories": integer(r"^Histories:\s*(\d+)$"),
        "elapsed_seconds": number(r"^Elapsed:\s*([0-9.eE+-]+)\s+s$"),
        "throughput_histories_per_second": number(
            r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$"
        ),
        "primary_kernel_seconds": number(
            r"^Kernel time:\s*primary=([0-9.eE+-]+)\s+s"
        ),
        "secondary_kernel_seconds": number(
            r"^Kernel time:.*secondary=([0-9.eE+-]+)\s+s"
        ),
        "secondary_queue_overflow": integer(
            r"^Secondary queue overflow:\s*(\d+)$"
        ),
        "cascade_queue_overflow": integer(r"^Cascade queue overflow:\s*(\d+)$"),
    }


def run_one(
    args: argparse.Namespace,
    profile: str,
    template: Path,
    histories: int,
    seed: int,
    run_dir: Path,
) -> dict[str, float | int | str]:
    dose = run_dir / "dose.mhd"
    log = run_dir / "run.log"
    config = run_dir / "config.yaml"
    spots = args.output_root / "spots" / f"spots_{histories}.txt"
    if not spots.exists():
        write_scaled_spots(args.source_spots, spots, histories)
    if dose.exists() and dose.with_suffix(".raw").exists() and log.exists():
        stats = parse_log(log)
        if stats["histories"] == histories and not (
            stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]
        ):
            print(f"SKIP complete {profile} {histories:,} seed {seed}", flush=True)
            return stats
    run_dir.mkdir(parents=True, exist_ok=True)
    render_config(template, config, dose, spots, histories, seed)
    command = [str(args.binary), "--config", str(config)]
    print(
        f"RUN {profile} {histories:,} seed {seed}: {' '.join(command)}",
        flush=True,
    )
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"run failed ({completed.returncode}); see {log}")
    stats = parse_log(log)
    if stats["histories"] != histories:
        raise RuntimeError(f"history mismatch in {log}: {stats['histories']}")
    if stats["secondary_queue_overflow"] or stats["cascade_queue_overflow"]:
        raise RuntimeError(f"queue overflow in {log}: {stats}")
    print(
        f"DONE {profile} {histories:,} seed {seed}: "
        f"{stats['elapsed_seconds']:.2f} s, "
        f"{stats['throughput_histories_per_second']:,.0f} hist/s",
        flush=True,
    )
    return stats


def combine_equal_history_doses(inputs: list[Path], output: Path) -> None:
    first_meta, first = read_mhd(inputs[0])
    shape = tuple(int(value) for value in first_meta["DimSize"].split())
    spacing = tuple(float(value) for value in first_meta["ElementSpacing"].split())
    offset = tuple(float(value) for value in first_meta.get("Offset", "0 0 0").split())
    accumulated = np.asarray(first, dtype=np.float64)
    for path in inputs[1:]:
        meta, values = read_mhd(path)
        current_shape = tuple(int(value) for value in meta["DimSize"].split())
        if current_shape != shape:
            raise ValueError(f"dose shape mismatch: {path}: {current_shape} != {shape}")
        accumulated += np.asarray(values, dtype=np.float64)
    accumulated /= float(len(inputs))
    write_mhd(
        output,
        array.array("f", accumulated.astype(np.float32).tolist()),
        shape,
        spacing,
        offset,
        units="Gy per primary (equal-history batch mean)",
        extra_lines=[f"CombinedEqualHistoryBatches = {len(inputs)}"],
    )


def aggregate_stats(
    stats: list[dict[str, float | int | str]], total_histories: int
) -> dict[str, float | int | str]:
    elapsed = sum(float(item["elapsed_seconds"]) for item in stats)
    return {
        "device": stats[0]["device"],
        "histories": total_histories,
        "elapsed_seconds": elapsed,
        "throughput_histories_per_second": total_histories / elapsed,
        "primary_kernel_seconds": sum(float(item["primary_kernel_seconds"]) for item in stats),
        "secondary_kernel_seconds": sum(float(item["secondary_kernel_seconds"]) for item in stats),
        "secondary_queue_overflow": sum(int(item["secondary_queue_overflow"]) for item in stats),
        "cascade_queue_overflow": sum(int(item["cascade_queue_overflow"]) for item in stats),
        "batches": len(stats),
        "batch_histories": [int(item["histories"]) for item in stats],
    }


def run_suite(args: argparse.Namespace) -> None:
    chunk_histories = 10_000_000
    for profile in args.profiles:
        template = args.config_root / f"RT07575_{profile}.yaml"
        if not template.exists():
            raise FileNotFoundError(template)
        for histories in args.histories:
            for group_index, seed in enumerate(SEEDS):
                level_dir = args.output_root / profile / f"{histories}" / f"seed{seed}"
                if histories <= chunk_histories:
                    stats = run_one(
                        args, profile, template, histories, seed, level_dir
                    )
                    (level_dir / "aggregate_run_stats.json").write_text(
                        json.dumps({**stats, "batches": 1, "batch_histories": [histories]}, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                    continue
                if histories % chunk_histories:
                    raise ValueError(
                        f"history budget above 10M must be an exact 10M multiple: {histories}"
                    )
                chunk_count = histories // chunk_histories
                chunk_doses = []
                chunk_stats = []
                for chunk_index in range(chunk_count):
                    if chunk_index == 0:
                        chunk_dir = (
                            args.output_root
                            / profile
                            / f"{chunk_histories}"
                            / f"seed{seed}"
                        )
                        chunk_seed = seed
                    else:
                        chunk_seed = seed + chunk_index * 100_000
                        chunk_dir = (
                            args.output_root
                            / profile
                            / "chunks_10m"
                            / f"seed_group{group_index + 1}"
                            / f"batch{chunk_index:02d}_seed{chunk_seed}"
                        )
                    chunk_stats.append(
                        run_one(
                            args,
                            profile,
                            template,
                            chunk_histories,
                            chunk_seed,
                            chunk_dir,
                        )
                    )
                    chunk_doses.append(chunk_dir / "dose.mhd")
                level_dir.mkdir(parents=True, exist_ok=True)
                combine_equal_history_doses(chunk_doses, level_dir / "dose.mhd")
                combined = aggregate_stats(chunk_stats, histories)
                (level_dir / "aggregate_run_stats.json").write_text(
                    json.dumps(combined, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"COMBINED {profile} {histories:,} seed group {group_index + 1}: "
                    f"{combined['elapsed_seconds']:.2f} s, "
                    f"{combined['throughput_histories_per_second']:,.0f} hist/s",
                    flush=True,
                )


def load_masked_pair(
    reference_path: Path,
    evaluation_path: Path,
    body_mask_path: Path,
) -> tuple[array.array, list[float], tuple[int, int, int], tuple[float, float, float], np.ndarray]:
    ref_meta, ref = read_mhd(reference_path)
    eval_meta, evaluation = read_mhd(evaluation_path)
    mask_meta, body = read_mhd(body_mask_path)
    shape = tuple(int(value) for value in ref_meta["DimSize"].split())
    spacing = tuple(float(value) for value in ref_meta["ElementSpacing"].split())
    eval_shape = tuple(int(value) for value in eval_meta["DimSize"].split())
    mask_shape = tuple(int(value) for value in mask_meta["DimSize"].split())
    if eval_shape != shape:
        raise ValueError(
            f"dose shape mismatch: reference={shape}, evaluation={eval_shape}"
        )
    if mask_shape == shape:
        inside = np.asarray(body, dtype=np.float32) > 0.5
    elif mask_shape == (shape[2], shape[0], shape[1]):
        # RT07575 BODY is patient-axis MHD [Xpatient,Ypatient,Zpatient], while
        # GPU dose is beam-axis [Ypatient,Zpatient,reverse(Xpatient)].
        patient = np.asarray(body, dtype=np.float32).reshape(
            mask_shape[2], mask_shape[1], mask_shape[0]
        )
        inside = np.transpose(patient, (2, 0, 1))[::-1, :, :].reshape(-1) > 0.5
    else:
        raise ValueError(
            f"mask shape {mask_shape} cannot map to beam dose shape {shape}"
        )
    ref_np = np.asarray(ref, dtype=np.float32)
    eval_np = np.asarray(evaluation, dtype=np.float32)
    ref_np = np.where(inside, ref_np, 0.0)
    eval_np = np.where(inside, eval_np, 0.0)
    threshold = 0.10 * float(np.max(ref_np[inside]))
    selection = inside & (ref_np >= threshold)
    return (
        array.array("f", ref_np.tolist()),
        eval_np.tolist(),
        shape,
        spacing,
        selection,
    )


def analyse_pair(
    reference_path: Path,
    evaluation_path: Path,
    body_mask_path: Path,
    gamma_points: int,
    gamma_resolution_mm: float,
) -> dict[str, object]:
    reference, evaluation, shape, spacing, selection = load_masked_pair(
        reference_path, evaluation_path, body_mask_path
    )
    ref_np = np.asarray(reference, dtype=np.float64)
    eval_np = np.asarray(evaluation, dtype=np.float64)
    selected = np.flatnonzero(selection)
    delta = eval_np[selected] - ref_np[selected]
    ref_max = float(np.max(ref_np[selected]))
    metrics: dict[str, object] = {
        "comparison_direction": "seed20260801 reference; seed20260802 evaluation",
        "body_and_reference_dose_threshold_percent": 10.0,
        "selected_voxels": int(selected.size),
        "scale": 1.0,
        "integral_evaluation_over_reference": float(eval_np.sum() / ref_np.sum()),
        "nrmse_high_dose": float(math.sqrt(np.mean(delta * delta)) / ref_max),
        "gamma": {},
    }
    gamma = metrics["gamma"]
    assert isinstance(gamma, dict)
    for dose_percent, distance_mm, label in CRITERIA:
        gamma[label] = {
            "global": gamma_3d(
                reference,
                evaluation,
                shape,
                spacing,
                dose_percent,
                distance_mm,
                10.0,
                gamma_points,
                0,
                interpolation_step_mm=gamma_resolution_mm,
                selection_mask=selection,
            ),
            "local": gamma_3d(
                reference,
                evaluation,
                shape,
                spacing,
                dose_percent,
                distance_mm,
                10.0,
                gamma_points,
                0,
                local_dose=True,
                interpolation_step_mm=gamma_resolution_mm,
                selection_mask=selection,
            ),
        }
    gamma["30"] = {
        "global": dose_only_pass_rate(
            reference, evaluation, 3.0, 10.0, selection_mask=selection
        ),
        "local": dose_only_pass_rate(
            reference,
            evaluation,
            3.0,
            10.0,
            local_dose=True,
            selection_mask=selection,
        ),
    }
    return metrics


def plot_summary(summary: dict[str, object], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    labels = (("33", "3%/3 mm"), ("22", "2%/2 mm"), ("11", "1%/1 mm"), ("30", "3%/0 mm"))
    colors = {"best": "#1f77b4", "fast": "#d62728"}
    results = summary["results"]
    assert isinstance(results, dict)
    for axis, (code, title) in zip(axes.flat, labels):
        for profile, entries in results.items():
            assert isinstance(entries, list)
            histories = [entry["histories"] for entry in entries]
            for mode, marker, linestyle in (
                ("global", "o", "-"),
                ("local", "s", "--"),
            ):
                values = [entry["metrics"]["gamma"][code][mode]["pass_percent"] for entry in entries]
                axis.plot(
                    histories,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    color=colors.get(profile),
                    label=f"{profile} {mode}",
                )
        axis.set_xscale("log")
        axis.set_ylim(0.0, 100.5)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_title(title)
        axis.set_ylabel("Gamma pass rate (%)")
        axis.set_xlabel("Histories per seed")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4)
    fig.suptitle("RT07575 non-minibeam GPU–GPU seed convergence (BODY, dose ≥10% Dmax)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_markdown(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# RT07575 non-minibeam GPU seed convergence",
        "",
        "Seed 20260801 is the reference and seed 20260802 is the evaluation. "
        "The comparison uses scale=1 and `BODY ∩ seed-1 dose >= 10% BODY Dmax`. "
        "DTA gamma uses deterministic 50k-point sampling and 0.5 mm trilinear search; "
        "3%/0 mm uses every selected voxel.",
        "",
        "| Profile | Histories/seed | seed1 s | seed2 s | mean hist/s | NRMSE | integral eval/ref | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    results = summary["results"]
    assert isinstance(results, dict)
    for profile, entries in results.items():
        assert isinstance(entries, list)
        for entry in entries:
            stats1, stats2 = entry["runs"]
            metrics = entry["metrics"]
            gamma = metrics["gamma"]
            mean_throughput = 0.5 * (
                stats1["throughput_histories_per_second"]
                + stats2["throughput_histories_per_second"]
            )
            cells = []
            for code in ("33", "22", "11", "30"):
                cells.append(
                    f"{gamma[code]['global']['pass_percent']:.3f} / "
                    f"{gamma[code]['local']['pass_percent']:.3f}"
                )
            lines.append(
                f"| {profile} | {entry['histories']:,} | "
                f"{stats1['elapsed_seconds']:.2f} | {stats2['elapsed_seconds']:.2f} | "
                f"{mean_throughput:,.0f} | {100.0 * metrics['nrmse_high_dose']:.3f}% | "
                f"{metrics['integral_evaluation_over_reference']:.6f} | "
                + " | ".join(cells)
                + " |"
            )
    lines.extend(
        [
            "",
            "The two seeds use identical weighted spot allocation at every history budget. "
            "This measures Monte Carlo repeatability, not GPU-to-TOPAS physical-model bias.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def analyse_suite(args: argparse.Namespace) -> None:
    summary: dict[str, object] = {
        "case": "RT07575",
        "minibeam": False,
        "seeds": list(SEEDS),
        "histories_per_seed": args.histories,
        "comparison_scale": 1.0,
        "selection": "BODY and seed20260801 dose >= 10% BODY Dmax",
        "gamma_sampling": {
            "dta_points": args.gamma_points,
            "interpolation_step_mm": args.gamma_resolution_mm,
            "dose_only_points": "all selected voxels",
        },
        "results": {},
    }
    results = summary["results"]
    assert isinstance(results, dict)
    for profile in args.profiles:
        entries = []
        for histories in args.histories:
            level = args.output_root / profile / f"{histories}"
            reference = level / f"seed{SEEDS[0]}" / "dose.mhd"
            evaluation = level / f"seed{SEEDS[1]}" / "dose.mhd"
            print(f"ANALYSE {profile} {histories:,}", flush=True)
            metrics = analyse_pair(
                reference,
                evaluation,
                args.body_mask,
                args.gamma_points,
                args.gamma_resolution_mm,
            )
            runs = []
            for seed in SEEDS:
                run_dir = level / f"seed{seed}"
                aggregate = run_dir / "aggregate_run_stats.json"
                runs.append(
                    json.loads(aggregate.read_text(encoding="utf-8"))
                    if aggregate.exists()
                    else parse_log(run_dir / "run.log")
                )
            entries.append({"histories": histories, "runs": runs, "metrics": metrics})
        results[profile] = entries
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.output_root / "summary.md")
    plot_summary(summary, args.output_root / "gamma_vs_histories.png")
    print(f"Wrote {summary_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--binary", type=Path, default=Path("build/oneapi-nvidia-release/carbon_mc")
    )
    parser.add_argument(
        "--config-root", type=Path, default=Path("out/fullplan_profiles/configs")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/ct/RT07575/nonminibeam_gpu_seed_convergence_fp32"),
    )
    parser.add_argument(
        "--body-mask", type=Path, default=Path("out/fullplan_result/RT07575/body_mask.mhd")
    )
    parser.add_argument(
        "--source-spots",
        type=Path,
        default=Path("ct/fullplan_result/RT07575/spots_full_plan.txt"),
    )
    parser.add_argument("--profiles", nargs="+", choices=("best", "fast"), default=["best", "fast"])
    parser.add_argument(
        "--histories",
        nargs="+",
        type=int,
        default=[1_000_000, 3_000_000, 10_000_000, 30_000_000, 60_000_000],
    )
    parser.add_argument("--gamma-points", type=int, default=50_000)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.run_only and args.analyze_only:
        parser.error("--run-only and --analyze-only are mutually exclusive")
    for attribute in (
        "binary",
        "config_root",
        "output_root",
        "body_mask",
        "source_spots",
    ):
        value = getattr(args, attribute)
        if not value.is_absolute():
            setattr(args, attribute, args.repo_root / value)
    if not args.analyze_only:
        run_suite(args)
    if not args.run_only:
        analyse_suite(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
