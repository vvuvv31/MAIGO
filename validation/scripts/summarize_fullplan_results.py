#!/usr/bin/env python3
"""Summarize valid full-plan GPU/TOPAS dose, LET, and performance results."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


CASES = ("RT06423", "RT07575")


def number(pattern: str, text: str, default: float = 0.0) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else default


def integer(pattern: str, text: str, default: int = 0) -> int:
    return int(round(number(pattern, text, float(default))))


def gpu_stats(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    return {
        "device": re.search(r"^SYCL device:\s*(.+)$", text, re.MULTILINE).group(1),
        "histories": integer(r"^Histories:\s*(\d+)$", text),
        "steps": integer(r"^Steps:\s*(\d+)$", text),
        "elapsed_seconds": number(r"^Elapsed:\s*([0-9.eE+-]+)\s+s$", text),
        "throughput_histories_per_second": number(
            r"^Throughput:\s*([0-9.eE+-]+)\s+histories/s$", text
        ),
        "primary_kernel_seconds": number(r"Kernel time: primary=([0-9.eE+-]+)\s+s", text),
        "secondary_kernel_seconds": number(
            r"Kernel time:.*secondary=([0-9.eE+-]+)\s+s", text
        ),
        "nuclear_interactions": integer(r"^Nuclear interactions:\s*(\d+)$", text),
        "transported_charged_secondaries": integer(
            r"^Transported charged secondaries:\s*(\d+)$", text
        ),
        "secondary_queue_overflow": integer(
            r"^Secondary queue overflow:\s*(\d+)$", text
        ),
        "cascade_queue_overflow": integer(r"^Cascade queue overflow:\s*(\d+)$", text),
        "energy_balance_error": number(r"^Energy balance error:\s*([0-9.eE+-]+)$", text),
        "device_memory_estimate_MiB": integer(
            r"^Device memory estimate:\s*(\d+)\s+MiB", text
        ),
    }


def topas_stats(case_dir: Path, histories: int) -> dict[str, object]:
    log_path = next(case_dir.glob("slurm_*_full.out"))
    text = log_path.read_text(encoding="utf-8", errors="replace")
    execution = number(r"Execution:.*?Real=([0-9.eE+-]+)s", text)
    total = number(r"Total:.*?Real=([0-9.eE+-]+)s", text)
    user = number(r"Execution:\s*User=([0-9.eE+-]+)s", text)
    return {
        "version": re.search(
            r"^# TOPAS Version:\s*(.+)$",
            (case_dir / "OSMK_Dtotal_full_plan.binheader").read_text(),
            re.MULTILINE,
        ).group(1),
        "geant4_version": re.search(
            r"Geant4 version Name:\s*(\S+)", text
        ).group(1),
        "allocated_cpu_tasks": 56,
        "histories": histories,
        "execution_seconds": execution,
        "total_seconds": total,
        "throughput_histories_per_second": histories / execution,
        "execution_user_cpu_seconds": user,
        "effective_cpu_cores": user / execution,
    }


def main() -> None:
    output_root = Path("out/fullplan_result")
    input_root = Path("ct/fullplan_result")
    report: dict[str, object] = {
        "comparison": (
            "GPU and TOPAS use the exact same positive per-spot integer L4 "
            "allocations from fullplan_result/spots_full_plan.txt"
        ),
        "dose_normalization": "absolute equal-history scale = 1; no fitted normalization",
        "spatial_mask": "RTSTRUCT Body/BODY; both TOPAS and GPU set to zero outside",
        "gamma": "3D inside BODY, TOPAS dose >=10% maximum, trilinear, 0.5 mm search step",
        "let_gamma": (
            "all-hadron LET_d; selection uses BODY and TOPAS dose >=10% of "
            "BODY dose maximum, with no LET threshold"
        ),
        "cases": {},
    }
    rows = []
    for case in CASES:
        case_out = output_root / case
        budget = json.loads((input_root / case / "particle_budget.json").read_text())
        whole_grid_dose = json.loads(
            (case_out / "comparison/dose_absolute/match_metrics.json").read_text()
        )
        dose = json.loads(
            (
                case_out
                / "comparison/body_masked/gamma_absolute/match_metrics.json"
            ).read_text()
        )
        let = json.loads(
            (case_out / "comparison/body_masked/let/let_metrics.json").read_text()
        )
        conversion = json.loads((case_out / "topas/conversion_metrics.json").read_text())
        body_mask = json.loads((case_out / "body_mask_metrics.json").read_text())
        histories = int(budget["total_particles_this_run"])
        gpu = gpu_stats(case_out / "gpu/run.log")
        topas = topas_stats(input_root / case, histories)
        topas_primary_nonzero = conversion["scorers"]["let_primary_c12"][
            "nonzero_voxels"
        ]
        dose_ratio = dose["integral_scaled_gpu"] / max(dose["integral_physical"], 1e-30)
        valid = topas_primary_nonzero > 0 and 0.25 <= dose_ratio <= 4.0
        status = (
            "valid"
            if valid
            else "invalid TOPAS reference: primary-C12 LET is empty and/or "
                 "equal-history dose differs by >4x"
        )
        item = {
            "status": status,
            "particle_allocation": {
                "active_spots": budget["active_spot_count"],
                "histories": histories,
            },
            "gpu": gpu,
            "topas": topas,
            "wall_clock_speedup_topas_over_gpu": (
                topas["execution_seconds"] / gpu["elapsed_seconds"]
            ),
            "dose": dose,
            "let": let,
            "body_mask": body_mask,
            "whole_ct_grid_dose_diagnostic": whole_grid_dose,
            "topas_conversion": conversion,
        }
        report["cases"][case] = item
        let_gamma = let["all_hadron_gamma"]
        rows.append(
            {
                "case": case,
                "status": status,
                "histories": histories,
                "gpu_seconds": gpu["elapsed_seconds"],
                "gpu_histories_per_second": gpu["throughput_histories_per_second"],
                "topas_execution_seconds": topas["execution_seconds"],
                "topas_histories_per_second": topas[
                    "throughput_histories_per_second"
                ],
                "wall_clock_speedup": item["wall_clock_speedup_topas_over_gpu"],
                "dose_integral_gpu_over_topas": dose_ratio,
                "dose_3pct_3mm_global_percent": dose["gamma_3pct_3mm_thr10"][
                    "pass_percent"
                ],
                "dose_3pct_3mm_local_percent": dose[
                    "gamma_local_3pct_3mm_thr10"
                ]["pass_percent"],
                "dose_2pct_2mm_global_percent": dose["gamma_2pct_2mm_thr10"][
                    "pass_percent"
                ],
                "dose_2pct_2mm_local_percent": dose[
                    "gamma_local_2pct_2mm_thr10"
                ]["pass_percent"],
                "dose_1pct_1mm_global_percent": dose["gamma_1pct_1mm_thr10"][
                    "pass_percent"
                ],
                "dose_1pct_1mm_local_percent": dose[
                    "gamma_local_1pct_1mm_thr10"
                ]["pass_percent"],
                "dose_3pct_0mm_global_percent": dose["gamma_3pct_0mm_thr10"][
                    "pass_percent"
                ],
                "dose_3pct_0mm_local_percent": dose[
                    "gamma_local_3pct_0mm_thr10"
                ]["pass_percent"],
                "primary_let_pearson_r": let["primary_c12"].get("pearson_r"),
                "primary_let_median_abs_relative_percent": let["primary_c12"].get(
                    "median_absolute_relative_percent"
                ),
                "all_hadron_let_pearson_r": let["all_hadron"].get("pearson_r"),
                "all_hadron_let_median_abs_relative_percent": let[
                    "all_hadron"
                ].get("median_absolute_relative_percent"),
                "let_3pct_3mm_global_percent": let_gamma["global_3pct_3mm"][
                    "pass_percent"
                ],
                "let_3pct_3mm_local_percent": let_gamma["local_3pct_3mm"][
                    "pass_percent"
                ],
                "let_2pct_2mm_global_percent": let_gamma["global_2pct_2mm"][
                    "pass_percent"
                ],
                "let_2pct_2mm_local_percent": let_gamma["local_2pct_2mm"][
                    "pass_percent"
                ],
                "let_1pct_1mm_global_percent": let_gamma["global_1pct_1mm"][
                    "pass_percent"
                ],
                "let_1pct_1mm_local_percent": let_gamma["local_1pct_1mm"][
                    "pass_percent"
                ],
                "let_3pct_0mm_global_percent": let_gamma["global_3pct_0mm"][
                    "pass_percent"
                ],
                "let_3pct_0mm_local_percent": let_gamma["local_3pct_0mm"][
                    "pass_percent"
                ],
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Full-plan GPU / TOPAS dose + LET validation",
        "",
        "GPU and TOPAS use the exact same positive per-spot integer L4 allocation. "
        "Dose uses absolute scale 1 with both maps set to zero outside the "
        "RTSTRUCT Body/BODY contour; no fitted normalization.",
        "",
        "| Case | Histories | GPU s | TOPAS h | Speedup | BODY GPU/TOPAS integral | Primary LET r / median ARE | All-hadron LET r / median ARE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        primary = (
            "N/A"
            if row["primary_let_pearson_r"] is None
            else f'{row["primary_let_pearson_r"]:.4f} / '
                 f'{row["primary_let_median_abs_relative_percent"]:.2f}%'
        )
        lines.append(
            f'| {row["case"]} | {row["histories"]:,} | '
            f'{row["gpu_seconds"]:.1f} | '
            f'{row["topas_execution_seconds"]/3600:.2f} | '
            f'{row["wall_clock_speedup"]:.1f}x | '
            f'{row["dose_integral_gpu_over_topas"]:.4f} | {primary} | '
            f'{row["all_hadron_let_pearson_r"]:.4f} / '
            f'{row["all_hadron_let_median_abs_relative_percent"]:.2f}% |'
        )
    lines.extend(
        [
            "",
            "## BODY dose gamma",
            "",
            "| Case | 3%/3 mm global/local | 2%/2 mm global/local | 1%/1 mm global/local | 3%/0 mm global/local |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f'| {row["case"]} | '
            f'{row["dose_3pct_3mm_global_percent"]:.3f}% / '
            f'{row["dose_3pct_3mm_local_percent"]:.3f}% | '
            f'{row["dose_2pct_2mm_global_percent"]:.3f}% / '
            f'{row["dose_2pct_2mm_local_percent"]:.3f}% | '
            f'{row["dose_1pct_1mm_global_percent"]:.3f}% / '
            f'{row["dose_1pct_1mm_local_percent"]:.3f}% | '
            f'{row["dose_3pct_0mm_global_percent"]:.3f}% / '
            f'{row["dose_3pct_0mm_local_percent"]:.3f}% |'
        )
    lines.extend(
        [
            "",
            "## All-hadron LET gamma",
            "",
            "The evaluation mask is defined only by RTSTRUCT BODY and TOPAS dose "
            ">=10% of the BODY dose maximum. No LET threshold is applied.",
            "",
            "| Case | 3%/3 mm global/local | 2%/2 mm global/local | 1%/1 mm global/local | 3%/0 mm global/local |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f'| {row["case"]} | '
            f'{row["let_3pct_3mm_global_percent"]:.3f}% / '
            f'{row["let_3pct_3mm_local_percent"]:.3f}% | '
            f'{row["let_2pct_2mm_global_percent"]:.3f}% / '
            f'{row["let_2pct_2mm_local_percent"]:.3f}% | '
            f'{row["let_1pct_1mm_global_percent"]:.3f}% / '
            f'{row["let_1pct_1mm_local_percent"]:.3f}% | '
            f'{row["let_3pct_0mm_global_percent"]:.3f}% / '
            f'{row["let_3pct_0mm_local_percent"]:.3f}% |'
        )
    lines.extend(
        [
            "",
            "Speedup is wall-clock only: TOPAS used a 56-task cluster allocation, "
            "while GPU used one NVIDIA TITAN RTX. It is not hardware-normalized.",
            "",
        ]
    )
    (output_root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
