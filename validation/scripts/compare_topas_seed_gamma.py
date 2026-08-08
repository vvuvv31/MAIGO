#!/usr/bin/env python3
"""Compare two equal-history TOPAS full-plan seeds on the patient grid.

The evaluation mask is fixed by the reference-seed dose and the RTSTRUCT BODY.
The same dose mask is used for dose and LET_d gamma.  No fitted
normalization is applied.  Particle-count projections assume independent
Monte Carlo noise scaling as 1/sqrt(N); they are statistical estimates, not a
claim that model/systematic errors disappear with more histories.
"""

from __future__ import annotations

import argparse
import array
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from match_gpu_to_physical_dose import (  # noqa: E402
    dose_only_pass_rate,
    gamma_3d,
    read_mhd,
)


CRITERIA = ((3.0, 3.0, "33"), (2.0, 2.0, "22"), (1.0, 1.0, "11"))
QUANTITIES = {
    "dose": "dose.mhd",
    "primary_c12_letd": "let_primary_c12.mhd",
    "all_hadron_letd": "let_all_hadron.mhd",
}


def load(path: Path) -> tuple[dict[str, str], np.ndarray]:
    metadata, values = read_mhd(path)
    return metadata, np.asarray(values, dtype=np.float64)


def require_same_grid(
    reference: dict[str, str],
    evaluation: dict[str, str],
    label: str,
    keys: tuple[str, ...] = ("DimSize", "ElementSpacing", "Offset"),
) -> None:
    for key in keys:
        if reference.get(key) != evaluation.get(key):
            raise ValueError(
                f"{label}: {key} differs: "
                f"{reference.get(key)!r} != {evaluation.get(key)!r}"
            )


def projected_histories(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    current_histories: int,
    dose_percent: float = 3.0,
) -> dict[str, object]:
    ref = reference[selection]
    delta = np.abs(evaluation[selection] - ref)
    ref_max = float(np.max(ref))
    ratios = {
        "global": delta / max(dose_percent * 0.01 * ref_max, 1.0e-30),
        "local": delta / np.maximum(
            dose_percent * 0.01 * np.abs(ref), 1.0e-30
        ),
    }
    result: dict[str, object] = {
        "criterion": f"{dose_percent:g}%/0 mm",
        "assumption": "independent Monte Carlo noise scales as 1/sqrt(N)",
        "interpretation": {
            "two_seed_repeatability": (
                "histories per seed required for two independent runs"
            ),
            "single_run_vs_infinite_mean": (
                "histories for one run versus an effectively noise-free mean; "
                "uses half the variance of a two-seed difference"
            ),
        },
        "targets": {},
    }
    targets = result["targets"]
    assert isinstance(targets, dict)
    for target in (0.95, 0.98, 0.99):
        item: dict[str, object] = {}
        for mode, ratio in ratios.items():
            quantile = float(np.quantile(ratio, target))
            repeat_multiplier = quantile * quantile
            truth_multiplier = 0.5 * repeat_multiplier
            item[mode] = {
                "observed_normalized_error_quantile": quantile,
                "two_seed_multiplier": repeat_multiplier,
                "two_seed_histories_per_seed": int(
                    math.ceil(current_histories * repeat_multiplier)
                ),
                "single_run_multiplier": truth_multiplier,
                "single_run_histories": int(
                    math.ceil(current_histories * truth_multiplier)
                ),
            }
        targets[f"{100 * target:.0f}%"] = item
    return result


def quantity_report(
    reference: np.ndarray,
    evaluation: np.ndarray,
    selection: np.ndarray,
    body: np.ndarray,
    shape_xyz: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    gamma_points: int,
    gamma_resolution_mm: float,
    histories_per_seed: int,
) -> dict[str, object]:
    reference_masked = np.where(body, reference, 0.0)
    evaluation_masked = np.where(body, evaluation, 0.0)
    ref_flat = array.array("f", reference_masked.astype(np.float32, copy=False))
    eval_flat = evaluation_masked.reshape(-1).tolist()
    selection_flat = selection.reshape(-1)
    ref_selected = reference[selection]
    eval_selected = evaluation[selection]
    delta = eval_selected - ref_selected
    ref_max = float(np.max(ref_selected))
    report: dict[str, object] = {
        "selected_voxels": int(np.count_nonzero(selection)),
        "evaluation_over_reference_body_integral": float(
            np.sum(evaluation_masked, dtype=np.float64)
            / max(np.sum(reference_masked, dtype=np.float64), 1.0e-30)
        ),
        "selected_mean_reference": float(np.mean(ref_selected)),
        "selected_mean_evaluation": float(np.mean(eval_selected)),
        "selected_nrmse_over_reference_max_percent": float(
            100.0 * np.sqrt(np.mean(delta * delta)) / max(ref_max, 1.0e-30)
        ),
        "selected_pearson_r": float(np.corrcoef(ref_selected, eval_selected)[0, 1]),
        "gamma": {},
        "particle_projection": projected_histories(
            reference,
            evaluation,
            selection,
            histories_per_seed,
        ),
    }
    gamma = report["gamma"]
    assert isinstance(gamma, dict)
    for dose_percent, distance_mm, label in CRITERIA:
        gamma[label] = {
            "global": gamma_3d(
                ref_flat,
                eval_flat,
                shape_xyz,
                spacing_xyz,
                dose_percent,
                distance_mm,
                10.0,
                gamma_points,
                0,
                interpolation_step_mm=gamma_resolution_mm,
                selection_mask=selection_flat,
            ),
            "local": gamma_3d(
                ref_flat,
                eval_flat,
                shape_xyz,
                spacing_xyz,
                dose_percent,
                distance_mm,
                10.0,
                gamma_points,
                0,
                local_dose=True,
                interpolation_step_mm=gamma_resolution_mm,
                selection_mask=selection_flat,
            ),
        }
    gamma["30"] = {
        "global": dose_only_pass_rate(
            ref_flat,
            eval_flat,
            3.0,
            10.0,
            selection_mask=selection_flat,
        ),
        "local": dose_only_pass_rate(
            ref_flat,
            eval_flat,
            3.0,
            10.0,
            local_dose=True,
            selection_mask=selection_flat,
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--body-mask", type=Path, required=True)
    parser.add_argument("--histories-per-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma-points", type=int, default=50_000)
    parser.add_argument("--gamma-resolution-mm", type=float, default=0.5)
    args = parser.parse_args()

    body_meta, body_values = load(args.body_mask)
    body = body_values > 0.5
    dose_meta, dose_reference = load(args.reference_dir / "dose.mhd")
    eval_dose_meta, dose_evaluation = load(args.evaluation_dir / "dose.mhd")
    require_same_grid(dose_meta, eval_dose_meta, "dose")
    # The BODY MHD preserves the DICOM world offset, while the TOPAS scorer
    # conversion records its local scorer origin.  Their arrays are already in
    # the same patient-axis index order, as used by the established full-plan
    # comparison; require identical dimensions and spacing here.
    require_same_grid(
        dose_meta, body_meta, "BODY", ("DimSize", "ElementSpacing")
    )
    shape_xyz = tuple(int(v) for v in dose_meta["DimSize"].split())
    spacing_xyz = tuple(float(v) for v in dose_meta["ElementSpacing"].split())
    body_peak = float(np.max(dose_reference[body]))
    selection = body & (dose_reference >= 0.10 * body_peak)

    report: dict[str, object] = {
        "reference_dir": str(args.reference_dir),
        "evaluation_dir": str(args.evaluation_dir),
        "histories_per_seed": args.histories_per_seed,
        "normalization": "absolute equal-history scale; no fitted normalization",
        "selection": "RTSTRUCT BODY and reference-seed dose >= 10% BODY Dmax",
        "let_selection": "same reference-dose mask; no LET threshold",
        "selected_voxels": int(np.count_nonzero(selection)),
        "shape_xyz": list(shape_xyz),
        "spacing_xyz_mm": list(spacing_xyz),
        "gamma_sampling": {
            "3d_points": min(args.gamma_points, int(np.count_nonzero(selection))),
            "available_points": int(np.count_nonzero(selection)),
            "interpolation_step_mm": args.gamma_resolution_mm,
            "3pct_0mm_points": "all selected voxels",
        },
        "quantities": {},
    }
    quantities = report["quantities"]
    assert isinstance(quantities, dict)
    for quantity, filename in QUANTITIES.items():
        print(f"Computing {quantity} gamma", flush=True)
        ref_meta, reference = load(args.reference_dir / filename)
        eval_meta, evaluation = load(args.evaluation_dir / filename)
        require_same_grid(ref_meta, eval_meta, quantity)
        require_same_grid(dose_meta, ref_meta, quantity)
        quantities[quantity] = quantity_report(
            reference,
            evaluation,
            selection,
            body,
            shape_xyz,
            spacing_xyz,
            args.gamma_points,
            args.gamma_resolution_mm,
            args.histories_per_seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# RT07575 TOPAS seed convergence",
        "",
        f"Each seed uses {args.histories_per_seed:,} histories. No fitted normalization.",
        "Mask: RTSTRUCT BODY and reference dose >= 10% of reference BODY Dmax; LET uses the same dose mask.",
        "",
        "| Quantity | integral B/A | NRMSE/Dmax | 3%/3mm G/L | 2%/2mm G/L | 1%/1mm G/L | 3%/0mm G/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in quantities.items():
        assert isinstance(item, dict)
        gamma = item["gamma"]
        cells = [
            f"{gamma[label]['global']['pass_percent']:.3f} / "
            f"{gamma[label]['local']['pass_percent']:.3f}"
            for label in ("33", "22", "11", "30")
        ]
        lines.append(
            f"| {name} | {item['evaluation_over_reference_body_integral']:.6f} | "
            f"{item['selected_nrmse_over_reference_max_percent']:.3f}% | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "Particle projections in the JSON use empirical 3%/0 mm error quantiles and 1/sqrt(N) scaling. They separate two-seed repeatability from a single run versus an effectively noise-free mean.",
            "",
        ]
    )
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
