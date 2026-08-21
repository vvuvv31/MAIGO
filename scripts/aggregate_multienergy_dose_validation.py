#!/usr/bin/env python3
"""Aggregate the final five-energy absolute-dose validation.

The report compares absolute TOPAS ``DoseToMedium`` and GPU dose in Gy;
neither profile nor integral is normalized.  Regional integrals use the
TOPAS proximal R50, proximal R80, and distal R80 crossings.  Each boundary is
linearly interpolated and inserted into both profiles before trapezoidal
integration, including the physical 0 and 700 mm endpoints.  Coverage is the
TOPAS-dose-above-1%-of-peak ROI through TOPAS distal R80.

The GPU provenance is read from each run's ``config.yaml`` and ``run.log``.
This deliberately does not consume an older aggregate summary, so the
generated acceptance result describes the files that are actually present.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ENERGIES = (70, 100, 150, 200, 250)
CALIBRATED_ROOT = Path(
    "out/proton_water_qgsp_bic_hp/full_physics_csda_block_fix_calibrated"
)
TOPAS_ROOT = Path("out/proton_water_qgsp_bic_hp/full_physics_compare_1M")
DEFAULT_OUTPUT_JSON = CALIBRATED_ROOT / "final_multienergy_validation.json"
DEFAULT_OUTPUT_MD = CALIBRATED_ROOT / "final_multienergy_validation.md"
REGION_NAMES = ("platform", "rise", "peak", "tail")
HISTORY_TARGET = 1_000_000


def read_profile(path: Path, column: str = "dose_Gy") -> tuple[np.ndarray, np.ndarray]:
    """Read a depth profile, accepting TOPAS comment preambles."""

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(
            csv.DictReader(
                line for line in stream if not line.lstrip().startswith("#")
            )
        )
    if not rows or "depth_mm" not in rows[0] or column not in rows[0]:
        raise ValueError(f"{path}: missing depth_mm/{column} columns")
    depth = np.asarray([float(row["depth_mm"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if depth.size < 2 or not np.all(np.isfinite(depth)) or not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: profile contains invalid values")
    order = np.argsort(depth)
    depth, values = depth[order], values[order]
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError(f"{path}: depth bins are not strictly increasing")
    return depth, values


def parse_scalar(value: str) -> Any:
    """Parse the scalar subset used by the run YAML files without PyYAML."""

    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:\d+\.?(?:\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
        return float(value)
    if (value.startswith("\"") and value.endswith("\"")) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def parse_config(path: Path) -> dict[str, Any]:
    """Read flat ``key: scalar`` values from a run configuration."""

    config: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.split(" #", 1)[0].strip()
        config[key.strip()] = parse_scalar(value)
    return config


def _number(
    text: str, pattern: str, default: float | int | None = None
) -> float | int | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return default
    value = match.group(1)
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    return float(value)


def parse_run_log(path: Path) -> dict[str, Any]:
    """Extract acceptance-critical provenance and counters from a run log."""

    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "exit_code": _number(text, r"^EXIT_CODE=([+-]?\d+)$"),
        "histories": _number(text, r"^Histories:\s*(\d+)$"),
        "depth_bins": _number(text, r"^Depth bins:\s*(\d+)$"),
        "initial_energy_MeVu": _number(
            text, r"^Initial energy:\s*([0-9.eE+-]+)\s*MeV/u"
        ),
        "energy_balance_error": _number(
            text, r"^Energy balance error:\s*([0-9.eE+-]+)"
        ),
        "primary_elastic_overflow": _number(
            text, r"^Primary elastic interactions:.*overflow=(\d+)"
        ),
        "secondary_queue_overflow": _number(
            text, r"^Secondary queue overflow:\s*(\d+)"
        ),
        "cascade_queue_overflow": _number(
            text, r"^Cascade queue overflow:\s*(\d+)"
        ),
        "neutral_queue_overflow": _number(
            text, r"^Neutral queue overflow:\s*(\d+)"
        ),
        "nuclear_interactions": _number(text, r"^Nuclear interactions:\s*(\d+)"),
        "sampled_reaction_packages": _number(
            text, r"^Sampled reaction packages:\s*(\d+)"
        ),
        "cascade_interactions": _number(text, r"^Cascade interactions:\s*(\d+)"),
        "primary_elastic_interactions": _number(
            text, r"^Primary elastic interactions:\s*(\d+)"
        ),
        "neutral_interactions": _number(text, r"^Neutral interactions:\s*(\d+)"),
    }


def crossing_depth(
    depth: np.ndarray, dose: np.ndarray, fraction: float, peak_index: int, direction: int
) -> float:
    """Find a peak-centered linear crossing in the requested direction."""

    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or +1")
    threshold = fraction * float(np.max(dose))
    start = peak_index - 1 if direction < 0 else peak_index
    stop = -1 if direction < 0 else depth.size - 1
    for index in range(start, stop, direction):
        next_index = index + direction
        if next_index < 0 or next_index >= depth.size:
            continue
        y0, y1 = dose[index], dose[next_index]
        if (y0 - threshold) * (y1 - threshold) <= 0.0 and y0 != y1:
            return float(
                depth[index]
                + (threshold - y0) * (depth[next_index] - depth[index]) / (y1 - y0)
            )
    raise ValueError(f"profile has no {fraction:g} crossing from peak")


def integral(depth: np.ndarray, values: np.ndarray) -> float:
    """Integrate aligned profile samples with NumPy 1.x/2.x support."""

    if depth.shape != values.shape:
        raise ValueError(f"integral shape mismatch: depth={depth.shape}, values={values.shape}")
    if depth.size < 2:
        return 0.0
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, depth))


def exact_crossing_integral(
    depth: np.ndarray, values: np.ndarray, start_mm: float, end_mm: float
) -> float:
    """Integrate a piecewise-linear profile, inserting both exact endpoints.

    The profiles are sampled at voxel centers (0.25 ... 699.75 mm).  At the
    physical 0/700 mm endpoints, ``np.interp`` uses the nearest edge sample,
    which is the convention used by the established 250 MeV result.  All
    interior voxel centers and both requested endpoints are then integrated as
    one aligned trapezoidal profile.
    """

    if end_mm < start_mm:
        raise ValueError(f"invalid integration interval [{start_mm}, {end_mm}]")
    interior = depth[(depth > start_mm) & (depth < end_mm)]
    points = np.concatenate(
        (
            np.asarray([start_mm], dtype=np.float64),
            interior,
            np.asarray([end_mm], dtype=np.float64),
        )
    )
    values_at_points = np.interp(points, depth, values)
    return integral(points, values_at_points)


def relative_difference(reference: float, candidate: float) -> float:
    if reference == 0.0:
        raise ValueError("cannot compute relative difference from zero reference")
    return 100.0 * (candidate / reference - 1.0)


def gpu_directory(calibrated_root: Path, topas_root: Path, energy: int) -> Path:
    """Return the promoted GPU directory, with the explicit 250 MeV location."""

    if energy == 250:
        return topas_root / "250MeV" / "gpu_full_physics_csda_block_fix"
    return calibrated_root / f"{energy}MeV"


def package_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Collect package/model fields without relying on a stale aggregate file."""

    fields = (
        "primary_reaction_package_file",
        "cascade_package_file",
        "primary_elastic_package_file",
        "neutral_package_file",
        "primary_package_physics_model",
        "cascade_package_physics_model",
        "primary_elastic_package_physics_model",
        "package_identity_validation",
        "package_identity_override_manifest_file",
        "primary_stopping_power_file",
        "primary_inelastic_cross_section_file",
        "primary_elastic_cross_section_file",
    )
    return {key: config.get(key) for key in fields if key in config}


def build_energy_report(
    calibrated_root: Path, topas_root: Path, energy: int
) -> dict[str, Any]:
    gpu_root = gpu_directory(calibrated_root, topas_root, energy)
    topas_root_for_energy = topas_root / f"{energy}MeV" / "topas_reference_1M"
    topas_depth, topas_dose = read_profile(topas_root_for_energy / "total_dose.csv")
    gpu_depth, gpu_dose = read_profile(gpu_root / "dose_Gy.csv")
    if gpu_depth.shape != topas_depth.shape or not np.allclose(
        gpu_depth, topas_depth, atol=1.0e-12
    ):
        raise ValueError(f"{energy} MeV: GPU and TOPAS depth grids differ")

    top_peak_index = int(np.argmax(topas_dose))
    gpu_peak_index = int(np.argmax(gpu_dose))
    topas_peak = float(topas_dose[top_peak_index])
    gpu_peak = float(gpu_dose[gpu_peak_index])
    top_r50p = crossing_depth(topas_depth, topas_dose, 0.50, top_peak_index, -1)
    top_r50d = crossing_depth(topas_depth, topas_dose, 0.50, top_peak_index, 1)
    top_r80p = crossing_depth(topas_depth, topas_dose, 0.80, top_peak_index, -1)
    top_r80d = crossing_depth(topas_depth, topas_dose, 0.80, top_peak_index, 1)
    gpu_r50p = crossing_depth(gpu_depth, gpu_dose, 0.50, gpu_peak_index, -1)
    gpu_r50d = crossing_depth(gpu_depth, gpu_dose, 0.50, gpu_peak_index, 1)
    gpu_r80p = crossing_depth(gpu_depth, gpu_dose, 0.80, gpu_peak_index, -1)
    gpu_r80d = crossing_depth(gpu_depth, gpu_dose, 0.80, gpu_peak_index, 1)

    boundaries = (0.0, top_r50p, top_r80p, top_r80d, 700.0)
    region_metrics: dict[str, dict[str, Any]] = {}
    for name, start, end in zip(REGION_NAMES, boundaries[:-1], boundaries[1:]):
        topas_area = exact_crossing_integral(topas_depth, topas_dose, start, end)
        gpu_area = exact_crossing_integral(gpu_depth, gpu_dose, start, end)
        region_metrics[name] = {
            "start_mm": float(start),
            "end_mm": float(end),
            "integral_method": "piecewise-linear interpolation with exact endpoints",
            "topas_integral_Gy_mm": topas_area,
            "gpu_integral_Gy_mm": gpu_area,
            "difference_Gy_mm": gpu_area - topas_area,
            "difference_percent": relative_difference(topas_area, gpu_area),
        }

    coverage_mask = (topas_dose > 0.01 * topas_peak) & (topas_depth <= top_r80d)
    if not np.any(coverage_mask):
        raise ValueError(f"{energy} MeV: TOPAS coverage ROI is empty")
    signed_errors = 100.0 * (gpu_dose[coverage_mask] / topas_dose[coverage_mask] - 1.0)
    abs_errors = np.abs(signed_errors)
    topas_integral = integral(topas_depth, topas_dose)
    gpu_integral = integral(gpu_depth, gpu_dose)

    config_path = gpu_root / "config.yaml"
    log_path = gpu_root / "run.log"
    config = parse_config(config_path)
    log = parse_run_log(log_path)
    configured_histories = config.get("number_of_histories")
    log_histories = log.get("histories")
    configured_scale = config.get("straggling_scale")
    overflow_fields = (
        "primary_elastic_overflow",
        "secondary_queue_overflow",
        "cascade_queue_overflow",
        "neutral_queue_overflow",
    )
    overflow = {key: log.get(key) for key in overflow_fields}
    checks = {
        "histories_config": configured_histories == HISTORY_TARGET,
        "histories_log": log_histories == HISTORY_TARGET,
        "exit_code": log.get("exit_code") == 0,
        "peak_depth": abs(float(gpu_depth[gpu_peak_index] - topas_depth[top_peak_index])) <= 0.5,
        "integral": abs(relative_difference(topas_integral, gpu_integral)) <= 0.5,
        "platform": abs(region_metrics["platform"]["difference_percent"]) <= 1.0,
        "rise": abs(region_metrics["rise"]["difference_percent"]) <= 1.0,
        "peak": abs(region_metrics["peak"]["difference_percent"]) <= 1.0,
        "peak_height": abs(relative_difference(topas_peak, gpu_peak)) <= 1.0,
        "queue_overflow": all(value == 0 for value in overflow.values()),
    }
    standard_pass = all(checks.values())
    status = "production proton package" if energy == 250 else (
        "pilot proton package; production-package limitation"
        if "pilot" in str(config.get("primary_reaction_package_file", ""))
        else "unknown package provenance"
    )
    return {
        "energy_MeV": energy,
        "gpu_directory": str(gpu_root),
        "topas_reference_directory": str(topas_root_for_energy),
        "config": str(config_path),
        "scale": configured_scale,
        "histories": configured_histories,
        "config_histories": configured_histories,
        "log_histories": log_histories,
        "depth_bins": int(topas_depth.size),
        "depth_bin_width_mm": float(np.median(np.diff(topas_depth))),
        "normalization": "none",
        "dose_quantity": "absolute DoseToMedium in Gy",
        "package_status": status,
        "package_provenance": package_provenance(config),
        "topas_peak_depth_mm": float(topas_depth[top_peak_index]),
        "gpu_peak_depth_mm": float(gpu_depth[gpu_peak_index]),
        "peak_depth_difference_mm": float(gpu_depth[gpu_peak_index] - topas_depth[top_peak_index]),
        "topas_peak_dose_Gy": topas_peak,
        "gpu_peak_dose_Gy": gpu_peak,
        "peak_height_difference_percent": relative_difference(topas_peak, gpu_peak),
        "topas_depth_integral_Gy_mm": topas_integral,
        "gpu_depth_integral_Gy_mm": gpu_integral,
        "integral_difference_percent": relative_difference(topas_integral, gpu_integral),
        "r50_proximal_mm": float(top_r50p),
        "r50_distal_mm": float(top_r50d),
        "r80_proximal_mm": float(top_r80p),
        "r80_distal_mm": float(top_r80d),
        "gpu_r50_proximal_mm": float(gpu_r50p),
        "gpu_r50_distal_mm": float(gpu_r50d),
        "gpu_r80_proximal_mm": float(gpu_r80p),
        "gpu_r80_distal_mm": float(gpu_r80d),
        "regions": region_metrics,
        "coverage": {
            "definition": "TOPAS dose >1% of TOPAS peak and depth <= TOPAS distal R80",
            "dose_mask_bins": int(np.count_nonzero(coverage_mask)),
            "within_1_percent": float(np.mean(abs_errors <= 1.0) * 100.0),
            "within_2_percent": float(np.mean(abs_errors <= 2.0) * 100.0),
            "within_6_percent": float(np.mean(abs_errors <= 6.0) * 100.0),
        },
        "run": log,
        "overflow": overflow,
        "standard_thresholds": {
            "peak_depth_difference_mm_max": 0.5,
            "integral_difference_percent_max": 0.5,
            "platform_rise_peak_difference_percent_max": 1.0,
            "peak_height_difference_percent_max": 1.0,
            "queue_overflow_required": 0,
            "histories_required": HISTORY_TARGET,
            "tail_is_report_only": True,
        },
        "standard_checks": checks,
        "standard_pass": bool(standard_pass),
        "tail_report_only": True,
        "tail_difference_percent": region_metrics["tail"]["difference_percent"],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Final Multienergy Absolute-Dose Validation",
        "",
        "This report compares absolute TOPAS DoseToMedium against GPU dose in Gy. No dose, profile, or integral normalization is applied.",
        "Regional integrals use TOPAS proximal R50, proximal R80, and distal R80 crossings. Each crossing and the 0/700 mm physical endpoints are inserted by piecewise-linear interpolation before integration.",
        "Coverage is the TOPAS dose >1% of peak ROI with depth <= TOPAS distal R80.",
        "",
        "## Decision Rules",
        "",
        "- Histories: configuration and log must both report 1,000,000.",
        "- Run exit code: 0.",
        "- Peak depth difference: <= 0.5 mm.",
        "- Whole-profile depth integral difference: <= 0.5%.",
        "- Platform, rise, and peak regional integral differences: <= 1%.",
        "- Peak height difference: <= 1%.",
        "- Primary-elastic, secondary, cascade, and neutral queue overflow: 0.",
        "- Tail is reported for diagnosis and is not a standard-pass gate.",
        "",
        "## Results",
        "",
        "| Energy | Scale | Peak height | Peak depth | Integral | Platform | Rise | Peak | Tail (report) | +/-1% | +/-2% | +/-6% | Standard |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in report["energies"]:
        regions = item["regions"]
        coverage = item["coverage"]
        values = [
            f"{item['energy_MeV']} MeV",
            f"{float(item['scale']):.3f}",
            f"{item['peak_height_difference_percent']:+.3f}%",
            f"{item['peak_depth_difference_mm']:+.3f} mm",
            f"{item['integral_difference_percent']:+.4f}%",
            *(f"{regions[name]['difference_percent']:+.3f}%" for name in REGION_NAMES),
            *(f"{coverage[key]:.2f}%" for key in ("within_1_percent", "within_2_percent", "within_6_percent")),
            "PASS" if item["standard_pass"] else "FAIL",
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "## Provenance",
        "",
        "The GPU configuration and run log are read independently for every energy; no legacy multienergy summary is used.",
        "",
        "| Energy | GPU directory | Package status | Histories (config/log) | Exit | Queue overflows (elastic/secondary/cascade/neutral) |",
        "|---:|:---|:---|---:|---:|:---|",
    ]
    for item in report["energies"]:
        run = item["run"]
        overflow = item["overflow"]
        lines.append(
            f"| {item['energy_MeV']} | `{item['gpu_directory']}` | {item['package_status']} | "
            f"{item['config_histories']}/{item['log_histories']} | {run['exit_code']} | "
            f"{overflow['primary_elastic_overflow']}/{overflow['secondary_queue_overflow']}/"
            f"{overflow['cascade_queue_overflow']}/{overflow['neutral_queue_overflow']} |"
        )
    lines += ["", "## Boundaries", ""]
    for item in report["energies"]:
        lines.append(
            f"- {item['energy_MeV']} MeV: R50p={item['r50_proximal_mm']:.6f} mm, "
            f"R80p={item['r80_proximal_mm']:.6f} mm, R80d={item['r80_distal_mm']:.6f} mm; "
            "regional integration uses exact crossing/endpoints."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        "--calibrated-root",
        dest="calibrated_root",
        type=Path,
        default=CALIBRATED_ROOT,
        help="GPU calibration root for 70--200 MeV; 250 MeV is taken from --topas-root",
    )
    parser.add_argument(
        "--topas-root",
        type=Path,
        default=TOPAS_ROOT,
        help="TOPAS/GPU comparison root containing the explicit 250 MeV GPU run",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    output_json = args.output_json or args.calibrated_root / "final_multienergy_validation.json"
    output_md = args.output_md or args.calibrated_root / "final_multienergy_validation.md"

    report = {
        "schema_version": 2,
        "status": "complete",
        "scope": "proton water full physics absolute DoseToMedium multienergy validation",
        "normalization": "none",
        "dose_quantity": "absolute DoseToMedium in Gy",
        "region_definition": "TOPAS proximal R50, proximal R80, distal R80; piecewise-linear exact crossing integration including 0/700 mm endpoints",
        "coverage_definition": "TOPAS dose >1% of TOPAS peak and depth <= TOPAS distal R80",
        "standard_pass_definition": "histories=1M in config and log, exit=0, integral<=0.5%, platform/rise/peak/peak-height<=1%, peak depth<=0.5 mm, all queue overflows=0; tail report-only",
        "calibrated_root": str(args.calibrated_root),
        "topas_root": str(args.topas_root),
        "energies": [
            build_energy_report(args.calibrated_root, args.topas_root, energy)
            for energy in ENERGIES
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output_md.write_text(markdown_report(report), encoding="utf-8")
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    print("Standard pass:", all(item["standard_pass"] for item in report["energies"]))


if __name__ == "__main__":
    main()
