#!/usr/bin/env python3
"""Summarize joint fragment+Cu lookup misses and package energy gaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


TOOLS = Path("/mnt/sdb/wuwei/MAIGO/startup/package_tools")
sys.path.insert(0, str(TOOLS))
import cinel03  # type: ignore


MISS_RE = re.compile(
    r"^\[fragment-cu-miss-joint\] generation=(?P<generation>\d+) "
    r"Z=(?P<z>\d+) A=(?P<a>\d+) reason=(?P<reason>\w+) "
    r"energy_bin_MeVu=\[(?P<energy_low>[^,]+),(?P<energy_high>[^)]+)\) "
    r"(?:energy_overflow=(?P<energy_overflow>[01]) )?"
    r"count=(?P<count>\d+) input_energy_MeV=(?P<input_energy>\S+) "
    r"mean_collision_depth_mm=(?P<depth>\S+) "
    r"mean_remaining_copper(?:_path_along_current_direction)?_mm="
    r"(?P<remaining>\S+)$"
)


def parse_log(path: Path) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = MISS_RE.match(line)
        if match is None:
            continue
        values = match.groupdict()
        energy_overflow = values["energy_overflow"] == "1" or \
            values["energy_high"].lower() == "inf"
        rows.append({
            "generation": int(values["generation"]),
            "z": int(values["z"]),
            "a": int(values["a"]),
            "reason": values["reason"],
            "energy_low_MeV_per_u": float(values["energy_low"]),
            "energy_high_MeV_per_u": (
                None if energy_overflow else float(values["energy_high"])
            ),
            "energy_overflow": energy_overflow,
            "count": int(values["count"]),
            "input_kinetic_energy_MeV": float(values["input_energy"]),
            "mean_collision_depth_mm": float(values["depth"]),
            "mean_remaining_copper_path_along_current_direction_mm":
                float(values["remaining"]),
        })
    if not rows:
        raise ValueError(f"no joint miss records found in {path}")
    return rows


def projectile_nodes(package: cinel03.Cinel03Package) -> dict[tuple[int, int], list[float]]:
    result: dict[tuple[int, int], list[float]] = defaultdict(list)
    for z, a, target_z, energy in package.energy_nodes:
        if target_z == 29:
            result[(z, a)].append(float(energy))
    return {key: sorted(set(values)) for key, values in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--maximum-spacing-MeVu", type=float, default=5.0)
    args = parser.parse_args()

    rows = parse_log(args.log)
    package = cinel03.Cinel03Package.read_binary(args.package)
    nodes = projectile_nodes(package)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reason_totals: dict[str, dict[str, float | int]] = {}
    for reason in sorted({str(row["reason"]) for row in rows}):
        selected = [row for row in rows if row["reason"] == reason]
        reason_totals[reason] = {
            "count": sum(int(row["count"]) for row in selected),
            "input_kinetic_energy_MeV": sum(
                float(row["input_kinetic_energy_MeV"]) for row in selected
            ),
        }

    projectile_totals: list[dict[str, object]] = []
    model_by_projectile = {
        (1, 1): "Binary Cascade",
        (1, 2): "INCL++ undefined",
        (1, 3): "INCL++ undefined",
    }
    for z, a in sorted({(int(row["z"]), int(row["a"])) for row in rows}):
        selected = [row for row in rows if (row["z"], row["a"]) == (z, a)]
        count = sum(int(row["count"]) for row in selected)
        input_energy = sum(float(row["input_kinetic_energy_MeV"]) for row in selected)
        depth_sum = sum(
            int(row["count"]) * float(row["mean_collision_depth_mm"])
            for row in selected
        )
        remaining_sum = sum(
            int(row["count"]) * float(
                row["mean_remaining_copper_path_along_current_direction_mm"])
            for row in selected
        )
        projectile_totals.append({
            "z": z,
            "a": a,
            "count": count,
            "input_kinetic_energy_MeV": input_energy,
            "mean_collision_depth_mm": depth_sum / count,
            "mean_remaining_copper_path_along_current_direction_mm":
                remaining_sum / count,
            "by_reason": {
                reason: sum(
                    int(row["count"]) for row in selected
                    if row["reason"] == reason
                )
                for reason in sorted({str(row["reason"]) for row in selected})
            },
        })

    package_gaps: list[dict[str, object]] = []
    for (z, a), energies in sorted(nodes.items()):
        for lower, upper in zip(energies, energies[1:]):
            if upper - lower <= args.maximum_spacing_MeVu:
                continue
            observed = [
                row for row in rows
                if (row["z"], row["a"], row["reason"]) == (z, a, "gap")
                and (row["energy_high_MeV_per_u"] is None or
                     float(row["energy_high_MeV_per_u"]) > lower)
                and float(row["energy_low_MeV_per_u"]) < upper
            ]
            count = sum(int(row["count"]) for row in observed)
            input_energy = sum(
                float(row["input_kinetic_energy_MeV"]) for row in observed
            )
            extraction_start = (
                math.floor(lower / args.maximum_spacing_MeVu)
                * args.maximum_spacing_MeVu
            )
            extraction_end = (
                math.ceil(upper / args.maximum_spacing_MeVu)
                * args.maximum_spacing_MeVu
            )
            package_gaps.append({
                "z": z,
                "a": a,
                "model": model_by_projectile.get((z, a), "package model"),
                "lower_package_node_MeV_per_u": lower,
                "upper_package_node_MeV_per_u": upper,
                "gap_MeV_per_u": upper - lower,
                "observed_miss_count": count,
                "observed_input_kinetic_energy_MeV": input_energy,
                "recommended_extraction": {
                    "start_MeV_per_u": extraction_start,
                    "end_MeV_per_u": extraction_end,
                    "maximum_spacing_MeV_per_u": args.maximum_spacing_MeVu,
                    "includes_overlap_nodes": True,
                },
            })

    report = {
        "schema": "MAIGO_COPPER_FRAGMENT_COVERAGE_V1",
        "source_log": str(args.log),
        "cascade_package": str(args.package),
        "joint_rows": len(rows),
        "total_misses": sum(int(row["count"]) for row in rows),
        "total_input_kinetic_energy_MeV": sum(
            float(row["input_kinetic_energy_MeV"]) for row in rows
        ),
        "reason_totals": reason_totals,
        "projectile_totals": projectile_totals,
        "package_gaps_exceeding_limit": package_gaps,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "total_misses": report["total_misses"],
        "reason_totals": reason_totals,
        "observed_package_gaps": [gap for gap in package_gaps
                                  if gap["observed_miss_count"]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
