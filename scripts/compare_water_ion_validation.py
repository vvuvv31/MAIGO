#!/usr/bin/env python3
"""Strict 1-D water validation report for a configured primary ion.

LET is compared from raw numerator and denominator moments. A depth bin is
valid when both profiles have a positive denominator. Bins where both sides
have zero denominator are skipped; a zero denominator on only one side is a
failed comparison and is never divided by. The TOPAS reference manifest is
the source of the LET coverage contract: the all-ion curve covers charged
ions and nuclei with ``Z>=1`` and excludes charged mesons and muons.

Dose and first-reaction values are normalized by their respective history counts.
The TOPAS reaction-rate profile is retained as an interaction-record diagnostic;
it is never used for the formal reaction pass/fail check. LET moments are
accumulated values and are not history-normalized here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = {
    "r80_mm": 1.0,
    "peak_mm": 1.0,
    "integral_percent": 2.0,
    "dose_mean_percent": 2.0,
    "dose_p95_percent": 5.0,
    "primary_let_percent": 3.0,
    "all_charged_ion_let_percent": 5.0,
    "reaction_percent": 2.0,
    "survival_percentage_points": 2.0,
}

ALL_CHARGED_ION_COVERAGE = (
    "charged ions and nuclei with atomic number Z>=1; "
    "charged mesons and muons excluded"
)


def read_csv(path: Path, value_column: str, *, nonnegative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    if not rows or "depth_mm" not in rows[0] or value_column not in rows[0]:
        raise ValueError(f"{path}: requires depth_mm and {value_column}")
    try:
        depth = np.asarray([float(row["depth_mm"]) for row in rows], dtype=np.float64)
        values = np.asarray([float(row[value_column]) for row in rows], dtype=np.float64)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: depth and {value_column} must be numeric") from error
    if not np.all(np.isfinite(depth)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: depth and {value_column} must be finite")
    if nonnegative and np.any(values < 0.0):
        raise ValueError(f"{path}: {value_column} must be non-negative")
    order = np.argsort(depth)
    sorted_depth = depth[order]
    if np.any(np.diff(sorted_depth) <= 0.0):
        raise ValueError(f"{path}: depth_mm must be unique and strictly increasing")
    return sorted_depth, values[order]


def ensure_same_grid(reference_depth: np.ndarray, candidate_depth: np.ndarray, label: str) -> None:
    if len(reference_depth) != len(candidate_depth) or not np.allclose(
        reference_depth, candidate_depth, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(f"{label}: depth grid mismatch")


def align(reference_depth: np.ndarray, candidate_depth: np.ndarray, candidate: np.ndarray, label: str) -> np.ndarray:
    if reference_depth[0] < candidate_depth[0] or reference_depth[-1] > candidate_depth[-1]:
        raise ValueError(f"{label}: candidate depth range does not cover reference")
    return np.interp(reference_depth, candidate_depth, candidate)


def distal_crossing(depth: np.ndarray, dose: np.ndarray, fraction: float) -> float:
    peak = int(np.argmax(dose))
    peak_value = float(dose[peak])
    if peak_value <= 0.0:
        raise ValueError("cannot determine R80 from non-positive dose")
    level = fraction * peak_value
    for right in range(peak + 1, len(depth)):
        left = right - 1
        if dose[left] >= level >= dose[right]:
            if dose[left] == dose[right]:
                return float(depth[left])
            return float(depth[left] + (level - dose[left]) * (depth[right] - depth[left]) / (dose[right] - dose[left]))
    raise ValueError("distal dose profile does not cross R80")


def percent_error(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = 100.0 * np.abs(candidate[mask] - reference[mask]) / np.abs(reference[mask])
    return float(np.mean(values)), float(np.percentile(values, 95.0))


def relative_percent(reference: float, candidate: float) -> float:
    if reference == 0.0:
        raise ValueError("relative comparison has zero reference")
    return 100.0 * (candidate / reference - 1.0)


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON manifest") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError(f"{path}: reference manifest schema_version 2 is required")
    return manifest


def _manifest_scorer(manifest: dict[str, Any], name: str, legacy_name: str | None = None) -> tuple[dict[str, Any], str]:
    scorers = manifest.get("scorers")
    if not isinstance(scorers, dict):
        raise ValueError("TOPAS manifest: scorers metadata is required")
    entry = scorers.get(name)
    if isinstance(entry, dict):
        return entry, name
    if legacy_name is not None:
        entry = scorers.get(legacy_name)
        if isinstance(entry, dict):
            return entry, legacy_name
    raise ValueError(f"TOPAS manifest: scorer metadata {name!r} is required")


def validate_reference_manifest(
    path: Path,
    primary_depth: np.ndarray,
    primary_numerator: Path,
    primary_denominator: Path,
    all_numerator: Path,
    all_denominator: Path,
) -> dict[str, Any]:
    manifest = read_manifest(path)
    depth_bins = manifest.get("depth_bins")
    if depth_bins is not None and depth_bins != len(primary_depth):
        raise ValueError(f"{path}: depth_bins does not match LET grid")
    for name, legacy in (("primary_let", None), ("all_charged_ion_let", "all_hadron_let")):
        entry, source_name = _manifest_scorer(manifest, name, legacy)
        coverage = entry.get("coverage")
        if not isinstance(coverage, str):
            raise ValueError(f"{path}: {source_name} coverage metadata is required")
        if name == "all_charged_ion_let":
            normalized = " ".join(coverage.lower().split())
            expected = " ".join(ALL_CHARGED_ION_COVERAGE.lower().split())
            if normalized != expected:
                raise ValueError(f"{path}: all_charged_ion LET coverage metadata mismatch")
        denominator = entry.get("denominator")
        if not isinstance(denominator, str):
            raise ValueError(f"{path}: {source_name} denominator metadata is required")
        if name == "primary_let" and Path(denominator).name != primary_denominator.name:
            raise ValueError(f"{path}: primary LET denominator metadata mismatch")
        if name == "all_charged_ion_let" and Path(denominator).name != all_denominator.name:
            raise ValueError(f"{path}: all_charged_ion LET denominator metadata mismatch")
    let_definition = manifest.get("let_definition")
    if not isinstance(let_definition, dict) or let_definition.get("weighting") != "dose":
        raise ValueError(f"{path}: LET weighting metadata must be dose")
    definition_coverage = let_definition.get("all_charged_ion_coverage", let_definition.get("all_hadron_coverage"))
    if not isinstance(definition_coverage, str) or "atomic number Z>=1" not in definition_coverage:
        raise ValueError(f"{path}: all_charged_ion LET coverage metadata mismatch")
    return {
        "path": str(path),
        "schema_version": manifest["schema_version"],
        "source_scorer_keys": {
            "primary_let": "primary_let",
            "all_charged_ion_let": "all_charged_ion_let"
            if "all_charged_ion_let" in manifest["scorers"]
            else "all_hadron_let",
        },
        "source_files": {
            "primary_let_numerator": primary_numerator.name,
            "primary_let_denominator": primary_denominator.name,
            "all_charged_ion_let_numerator": all_numerator.name,
            "all_charged_ion_denominator": all_denominator.name,
        },
    }


def let_moment_errors(
    reference_depth: np.ndarray,
    reference_numerator: np.ndarray,
    reference_denominator: np.ndarray,
    candidate_depth: np.ndarray,
    candidate_numerator: np.ndarray,
    candidate_denominator: np.ndarray,
    dose_mask: np.ndarray,
    label: str,
) -> tuple[float, float, dict[str, int]]:
    ensure_same_grid(reference_depth, candidate_depth, f"{label}: numerator/denominator")
    positive_reference = reference_denominator > 0.0
    positive_candidate = candidate_denominator > 0.0
    both_zero = ~positive_reference & ~positive_candidate
    one_zero = positive_reference ^ positive_candidate
    active_mismatch = one_zero & dose_mask
    valid = positive_reference & positive_candidate & dose_mask
    stats = {
        "compared_bins": int(np.count_nonzero(valid)),
        "skipped_both_zero_denominator_bins": int(np.count_nonzero(both_zero & dose_mask)),
        "zero_denominator_mismatch_bins": int(np.count_nonzero(active_mismatch)),
    }
    if np.any(active_mismatch):
        # Keep the report strict JSON while making the metric unambiguously fail.
        return 1.0e300, 1.0e300, stats
    if not np.any(valid):
        raise ValueError(f"{label}: no bins with positive LET denominator in the >1% TOPAS dose mask")
    reference_let = reference_numerator[valid] / reference_denominator[valid]
    candidate_let = candidate_numerator[valid] / candidate_denominator[valid]
    if not np.all(np.isfinite(reference_let)) or not np.all(np.isfinite(candidate_let)):
        raise ValueError(f"{label}: LET ratio is non-finite")
    mean, p95 = percent_error(reference_let, candidate_let, np.ones(reference_let.shape, dtype=bool))
    return mean, p95, stats


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    td, topas_dose_raw = read_csv(args.topas_dose, args.topas_dose_column, nonnegative=True)
    gd, gpu_dose_raw = read_csv(args.gpu_dose, args.gpu_dose_column, nonnegative=True)
    topas_dose = topas_dose_raw / args.topas_histories
    gpu_dose = align(td, gd, gpu_dose_raw / args.gpu_histories, "GPU dose")
    dose_mask = topas_dose > 0.01 * float(np.max(topas_dose))
    if not np.any(dose_mask):
        raise ValueError("no TOPAS dose samples above 1% peak")
    dose_mean, dose_p95 = percent_error(topas_dose, gpu_dose, dose_mask)

    topas_primary_depth, topas_primary_num = read_csv(args.topas_primary_let_numerator, args.topas_primary_let_numerator_column, nonnegative=True)
    topas_primary_den_depth, topas_primary_den = read_csv(args.topas_primary_let_denominator, args.topas_primary_let_denominator_column, nonnegative=True)
    topas_all_depth, topas_all_num = read_csv(args.topas_all_charged_ion_let_numerator, args.topas_all_charged_ion_let_numerator_column, nonnegative=True)
    topas_all_den_depth, topas_all_den = read_csv(args.topas_all_charged_ion_let_denominator, args.topas_all_charged_ion_let_denominator_column, nonnegative=True)
    ensure_same_grid(topas_primary_depth, topas_primary_den_depth, "TOPAS primary LET")
    ensure_same_grid(topas_all_depth, topas_all_den_depth, "TOPAS all_charged_ion LET")
    ensure_same_grid(td, topas_primary_depth, "TOPAS primary LET/dose")
    ensure_same_grid(td, topas_all_depth, "TOPAS all_charged_ion LET/dose")

    gpu_let_depth, gpu_primary_num = read_csv(args.gpu_let, args.gpu_primary_let_numerator_column, nonnegative=True)
    gpu_primary_den_depth, gpu_primary_den = read_csv(args.gpu_let, args.gpu_primary_let_denominator_column, nonnegative=True)
    gpu_all_num_depth, gpu_all_num = read_csv(args.gpu_let, args.gpu_all_charged_ion_let_numerator_column, nonnegative=True)
    gpu_all_den_depth, gpu_all_den = read_csv(args.gpu_let, args.gpu_all_charged_ion_let_denominator_column, nonnegative=True)
    ensure_same_grid(gpu_let_depth, gpu_primary_den_depth, "GPU primary LET")
    ensure_same_grid(gpu_let_depth, gpu_all_num_depth, "GPU all_charged_ion LET")
    ensure_same_grid(gpu_let_depth, gpu_all_den_depth, "GPU all_charged_ion LET")
    gpu_primary_num = align(topas_primary_depth, gpu_let_depth, gpu_primary_num, "GPU primary LET")
    gpu_primary_den = align(topas_primary_depth, gpu_let_depth, gpu_primary_den, "GPU primary LET denominator")
    gpu_all_num = align(topas_all_depth, gpu_let_depth, gpu_all_num, "GPU all_charged_ion LET")
    gpu_all_den = align(topas_all_depth, gpu_let_depth, gpu_all_den, "GPU all_charged_ion LET denominator")
    primary_let_mean, primary_let_p95, primary_let_stats = let_moment_errors(
        topas_primary_depth, topas_primary_num, topas_primary_den, topas_primary_depth, gpu_primary_num, gpu_primary_den, dose_mask, "primary LET"
    )
    all_let_mean, all_let_p95, all_let_stats = let_moment_errors(
        topas_all_depth, topas_all_num, topas_all_den, topas_all_depth, gpu_all_num, gpu_all_den, dose_mask, "all_charged_ion LET"
    )
    manifest_source = validate_reference_manifest(
        args.topas_manifest,
        topas_primary_depth,
        args.topas_primary_let_numerator,
        args.topas_primary_let_denominator,
        args.topas_all_charged_ion_let_numerator,
        args.topas_all_charged_ion_let_denominator,
    )

    sd_t, topas_survival_raw = read_csv(args.topas_survival, args.topas_survival_column, nonnegative=True)
    sd_g, gpu_survival_raw = read_csv(args.gpu_survival, args.gpu_survival_column, nonnegative=True)
    topas_survival = align(td, sd_t, topas_survival_raw / args.topas_histories, "TOPAS survival")
    gpu_survival = align(td, sd_g, gpu_survival_raw / args.gpu_histories, "GPU survival")
    survival_pp = float(100.0 * np.max(np.abs(gpu_survival - topas_survival)))
    topas_first_depth, topas_first = read_csv(
        args.topas_first_reactions, args.topas_first_reactions_column, nonnegative=True
    )
    topas_rate_depth, topas_rate = read_csv(
        args.topas_reaction_rate, args.topas_reaction_rate_column, nonnegative=True
    )
    ensure_same_grid(topas_first_depth, topas_rate_depth, "TOPAS first reactions/reaction rate")
    gpu_reaction_depth, gpu_primary_reactions = read_csv(
        args.gpu_primary_reactions, args.gpu_primary_reactions_column, nonnegative=True
    )
    topas_first_reactions = float(np.sum(topas_first))
    topas_all_primary_interaction_records = float(np.sum(topas_rate))
    continuation_excess = topas_all_primary_interaction_records - topas_first_reactions
    gpu_primary_reaction_count = float(np.sum(gpu_primary_reactions))
    first_reaction_percent = relative_percent(
        topas_first_reactions / args.topas_histories,
        gpu_primary_reaction_count / args.gpu_histories,
    )
    values = {
        "r80_difference_mm": distal_crossing(td, gpu_dose, 0.8) - distal_crossing(td, topas_dose, 0.8),
        "peak_difference_mm": float(td[int(np.argmax(gpu_dose))] - td[int(np.argmax(topas_dose))]),
        "integral_relative_percent": relative_percent(float(np.trapezoid(topas_dose, td)), float(np.trapezoid(gpu_dose, td))),
        "dose_mean_absolute_percent": dose_mean,
        "dose_p95_absolute_percent": dose_p95,
        "primary_let_mean_absolute_percent": primary_let_mean,
        "primary_let_p95_absolute_percent": primary_let_p95,
        "all_charged_ion_let_mean_absolute_percent": all_let_mean,
        "all_charged_ion_let_p95_absolute_percent": all_let_p95,
        "topas_first_reactions": topas_first_reactions,
        "topas_all_primary_interaction_records": topas_all_primary_interaction_records,
        "continuation_excess": continuation_excess,
        "gpu_primary_reactions": gpu_primary_reaction_count,
        "first_reaction_relative_difference": first_reaction_percent,
        "survival_max_absolute_percentage_points": survival_pp,
    }
    checks = {
        "R80": abs(values["r80_difference_mm"]) <= THRESHOLDS["r80_mm"],
        "peak": abs(values["peak_difference_mm"]) <= THRESHOLDS["peak_mm"],
        "integral": abs(values["integral_relative_percent"]) <= THRESHOLDS["integral_percent"],
        "dose_mean": values["dose_mean_absolute_percent"] <= THRESHOLDS["dose_mean_percent"],
        "dose_p95": values["dose_p95_absolute_percent"] <= THRESHOLDS["dose_p95_percent"],
        "primary_let": values["primary_let_mean_absolute_percent"] <= THRESHOLDS["primary_let_percent"],
        "all_charged_ion_let": values["all_charged_ion_let_mean_absolute_percent"] <= THRESHOLDS["all_charged_ion_let_percent"],
        "reaction": abs(values["first_reaction_relative_difference"]) <= THRESHOLDS["reaction_percent"],
        "survival": values["survival_max_absolute_percentage_points"] <= THRESHOLDS["survival_percentage_points"],
    }
    return {
        "schema_version": 2,
        "thresholds": THRESHOLDS,
        "metrics": values,
        "checks": checks,
        "let_denominator_policy": {
            "both_zero": "skip",
            "one_zero": "fail",
            "division": "positive-denominator-bins-only",
            "primary": primary_let_stats,
            "all_charged_ion": all_let_stats,
        },
        "sources": {
            "topas_reference": manifest_source,
            "reaction": {
                "first_reactions": {
                    "path": str(args.topas_first_reactions),
                    "column": args.topas_first_reactions_column,
                },
                "all_primary_interaction_records": {
                    "path": str(args.topas_reaction_rate),
                    "column": args.topas_reaction_rate_column,
                },
                "gpu_primary_reactions": {
                    "path": str(args.gpu_primary_reactions),
                    "column": args.gpu_primary_reactions_column,
                },
                "formal_metric": "first_reaction_relative_difference",
            },
            "gpu_let": {
                "path": str(args.gpu_let),
                "primary_numerator_column": args.gpu_primary_let_numerator_column,
                "primary_denominator_column": args.gpu_primary_let_denominator_column,
                "all_charged_ion_numerator_column": args.gpu_all_charged_ion_let_numerator_column,
                "all_charged_ion_denominator_column": args.gpu_all_charged_ion_let_denominator_column,
                "all_charged_ion_source_column_mapping": {
                    "numerator": args.gpu_all_charged_ion_let_numerator_column,
                    "denominator": args.gpu_all_charged_ion_let_denominator_column,
                },
            },
        },
        "passed": all(checks.values()),
    }


def add_csv_argument(parser: argparse.ArgumentParser, name: str, column_default: str) -> None:
    parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(f"--{name}-column", default=column_default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topas-histories", type=int, required=True)
    parser.add_argument("--gpu-histories", type=int, required=True)
    add_csv_argument(parser, "topas-dose", "energy_deposition_MeV")
    add_csv_argument(parser, "gpu-dose", "energy_deposition_MeV")
    add_csv_argument(parser, "topas-primary-let-numerator", "letd_MeV_per_mm_per_g_cm3")
    add_csv_argument(parser, "topas-primary-let-denominator", "denominator_MeV")
    add_csv_argument(parser, "topas-all-charged-ion-let-numerator", "letd_MeV_per_mm_per_g_cm3")
    add_csv_argument(parser, "topas-all-charged-ion-let-denominator", "denominator_MeV")
    add_csv_argument(parser, "gpu-let", "primary_numerator")
    parser.add_argument("--gpu-primary-let-numerator-column", default="primary_numerator")
    parser.add_argument("--gpu-primary-let-denominator-column", default="primary_denominator_MeV")
    parser.add_argument("--gpu-all-charged-ion-let-numerator-column", default="all_hadron_numerator")
    parser.add_argument("--gpu-all-charged-ion-let-denominator-column", default="all_hadron_denominator_MeV")
    parser.add_argument("--topas-manifest", type=Path, required=True)
    add_csv_argument(parser, "topas-survival", "count")
    add_csv_argument(parser, "gpu-survival", "count")
    add_csv_argument(parser, "topas-first-reactions", "count")
    add_csv_argument(parser, "topas-reaction-rate", "count")
    add_csv_argument(parser, "gpu-primary-reactions", "count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.topas_histories <= 0 or args.gpu_histories <= 0:
        raise SystemExit("history counts must be positive")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_pass and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
