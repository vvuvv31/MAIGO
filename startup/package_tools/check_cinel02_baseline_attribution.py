#!/usr/bin/env python3
"""Check that TOPAS-compatibility mode changes only its intended species.

The input reports are produced by ``scripts/analyze_cinel02_species_idd.py``.
This check deliberately compares GPU integrals from two runs made with the
same physics/package/rate/seed and different compatibility policy. Be and
charged-total are allowed to change because Be-6 is removed from transport;
all other scored categories are expected to be unchanged within the supplied
percentage-point tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED = ("Z4_Be", "charged_total")


def _category_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(report.get("categories", []))
    total = report.get("charged_total")
    if total is not None:
        rows.append(total)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("category")
        if not isinstance(label, str) or label in result:
            raise ValueError(f"invalid or duplicate category: {label!r}")
        result[label] = row
    return result


def _input_hashes(report: dict[str, Any]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {"package": set(), "rate": set(), "topas": set()}
    for item in report.get("inputs", []):
        path = str(item.get("path", ""))
        digest = item.get("sha256")
        if not isinstance(digest, str):
            continue
        lower = path.lower()
        name = Path(path).name.lower()
        if lower.endswith((".cinpkg", ".cinpkg03")) or "package" in lower:
            groups["package"].add(digest)
        elif "rate" in name or (name.endswith(".csv") and "cascade" in lower):
            groups["rate"].add(digest)
        # GPU output directories can contain the word ``topascompat``. Only
        # the explicit TOPAS scorer basenames are reference inputs.
        elif name.startswith("topas_") and name.endswith(".bin"):
            groups["topas"].add(digest)
    return groups


def _relative_percent(new: float, old: float) -> float | None:
    if old == 0.0:
        return 0.0 if new == 0.0 else None
    return 100.0 * (new / old - 1.0)


def check(
    off: dict[str, Any],
    on: dict[str, Any],
    *,
    tolerance_percentage_points: float = 0.01,
    allowed_categories: tuple[str, ...] = DEFAULT_ALLOWED,
) -> dict[str, Any]:
    if tolerance_percentage_points < 0.0 or not math.isfinite(tolerance_percentage_points):
        raise ValueError("tolerance_percentage_points must be finite and non-negative")

    errors: list[str] = []
    for key in ("histories", "energy_MeV_per_u", "grid", "scorer"):
        if off.get(key) != on.get(key):
            errors.append(f"metadata mismatch: {key}")

    off_hashes = _input_hashes(off)
    on_hashes = _input_hashes(on)
    for name in off_hashes:
        if off_hashes[name] != on_hashes[name]:
            errors.append(
                f"{name} input hash mismatch: off={sorted(off_hashes[name])} "
                f"on={sorted(on_hashes[name])}"
            )

    off_categories = _category_map(off)
    on_categories = _category_map(on)
    if set(off_categories) != set(on_categories):
        errors.append(
            "category mismatch: "
            f"off={sorted(off_categories)} on={sorted(on_categories)}"
        )

    rows: list[dict[str, Any]] = []
    stable_failures: list[str] = []
    for category in sorted(set(off_categories) & set(on_categories)):
        off_row = off_categories[category]
        on_row = on_categories[category]
        off_gpu = float(off_row["gpu_integral_Gy"])
        on_gpu = float(on_row["gpu_integral_Gy"])
        off_topas = float(off_row["topas_integral_Gy"])
        on_topas = float(on_row["topas_integral_Gy"])
        topas_delta = _relative_percent(on_topas, off_topas)
        gpu_delta = _relative_percent(on_gpu, off_gpu)
        allowed = category in allowed_categories
        unchanged = allowed or (
            gpu_delta is not None and abs(gpu_delta) <= tolerance_percentage_points
        )
        rows.append({
            "category": category,
            "allowed_to_change": allowed,
            "off_gpu_integral_Gy": off_gpu,
            "on_gpu_integral_Gy": on_gpu,
            "gpu_change_percent": gpu_delta,
            "off_topas_integral_Gy": off_topas,
            "on_topas_integral_Gy": on_topas,
            "topas_reference_change_percent": topas_delta,
            "unchanged_within_tolerance": unchanged,
        })
        if not unchanged:
            stable_failures.append(category)

    # TOPAS is a common reference, so any nonzero change means the reports do
    # not describe a pure compatibility A/B even if the GPU side is stable.
    reference_failures = [
        row["category"] for row in rows
        if row["topas_reference_change_percent"] is not None
        and abs(float(row["topas_reference_change_percent"])) > 1.0e-9
    ]
    if reference_failures:
        errors.append("TOPAS reference changed between reports: " + ", ".join(reference_failures))

    return {
        "schema_version": 1,
        "status": "pass" if not errors and not stable_failures else "fail",
        "tolerance_percentage_points": tolerance_percentage_points,
        "allowed_categories": list(allowed_categories),
        "errors": errors,
        "stable_category_failures": stable_failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-analysis", type=Path, required=True)
    parser.add_argument("--on-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-percentage-points", type=float, default=0.01)
    parser.add_argument(
        "--allow-category", action="append", default=list(DEFAULT_ALLOWED),
        help="Category allowed to change (repeatable; defaults to Z4_Be and charged_total)",
    )
    args = parser.parse_args()
    off = json.loads(args.off_analysis.read_text(encoding="utf-8"))
    on = json.loads(args.on_analysis.read_text(encoding="utf-8"))
    report = check(
        off,
        on,
        tolerance_percentage_points=args.tolerance_percentage_points,
        allowed_categories=tuple(args.allow_category),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["rows"]:
        change = row["gpu_change_percent"]
        rendered = "n/a" if change is None else f"{change:+.9f}%"
        print(f"{row['category']}: GPU change {rendered} (allowed={row['allowed_to_change']})")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")
    if report["stable_category_failures"]:
        print("stable category failures: " + ", ".join(report["stable_category_failures"]))
    print(f"status: {report['status']}")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
