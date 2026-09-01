#!/usr/bin/env python3
"""Census CINEL02 rate groups against CINPKG03 replay support.

This is a read-only diagnostic for Step 03B-0.  It deliberately restricts
the report to the 18 explicit runtime isotopes and the two water targets.  A
rate group can exist while all of its samples are zero; that distinction is
important for diagnosing a projectile such as Be-6 which may be produced by
the package but never receive a secondary nuclear hazard in the runtime.

The package's persisted global energy-node index is used as the authoritative
event-support source.  Each event-node energy contributes the runtime replay
window ``[E - tolerance, E + tolerance]``; overlapping windows are merged.
No event records are sampled or modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any, Iterable


PACKAGE_HEADER = struct.Struct("<8sIIIIIIIQQQQffQQ40s")
PACKAGE_NODE = struct.Struct("<hhhhf")
TOLERANCE_MEVU = 0.51
SPECIES = [
    (1, 1, "1H"),
    (1, 2, "2H"),
    (1, 3, "3H"),
    (2, 3, "3He"),
    (2, 4, "4He"),
    (2, 6, "6He"),
    (3, 6, "6Li"),
    (3, 7, "7Li"),
    (4, 6, "6Be"),
    (4, 7, "7Be"),
    (4, 9, "9Be"),
    (4, 10, "10Be"),
    (5, 8, "8B"),
    (5, 10, "10B"),
    (5, 11, "11B"),
    (6, 10, "10C"),
    (6, 11, "11C"),
    (6, 12, "12C"),
]
TARGETS = [(1, 1, "H1"), (8, 16, "O16")]
KEYS = [(z, a, tz, ta) for z, a, _ in SPECIES for tz, ta, _ in TARGETS]
LABELS = {(z, a): label for z, a, label in SPECIES}
TARGET_LABELS = {(z, a): label for z, a, label in TARGETS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rate_csv(path: Path) -> dict[tuple[int, int, int, int], list[tuple[float, float]]]:
    groups: dict[tuple[int, int, int, int], list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        if rows.fieldnames is None:
            raise ValueError(f"rate CSV has no header: {path}")
        required = {
            "projectile_z", "projectile_a", "target_z", "target_a",
            "energy_MeV_per_u", "macroscopic_cross_section_per_mm",
        }
        missing = required.difference(rows.fieldnames)
        if missing:
            raise ValueError(f"rate CSV missing fields {sorted(missing)}: {path}")
        for line_number, row in enumerate(rows, 2):
            try:
                key = tuple(int(row[name]) for name in (
                    "projectile_z", "projectile_a", "target_z", "target_a"))
                energy = float(row["energy_MeV_per_u"])
                rate = float(row["macroscopic_cross_section_per_mm"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid rate row {path}:{line_number}") from error
            if len(key) != 4 or not math.isfinite(energy) or not math.isfinite(rate):
                raise ValueError(f"non-finite/invalid rate row {path}:{line_number}")
            if energy < 0.0 or rate < 0.0:
                raise ValueError(f"negative rate energy/cross section {path}:{line_number}")
            groups.setdefault(key, []).append((energy, rate))
    for key, values in groups.items():
        values.sort()
        if any(a == b for (a, _), (b, _) in zip(values, values[1:])):
            raise ValueError(f"duplicate rate energy in group {key}")
    return groups


def _merge_intervals(intervals: Iterable[tuple[float, float]], epsilon: float = 1.0e-9) -> list[list[float]]:
    ordered = sorted((float(left), float(right)) for left, right in intervals if right >= left)
    merged: list[list[float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1] + epsilon:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return merged


def _sample_support_intervals(samples: list[tuple[float, float]]) -> list[list[float]]:
    """Represent nonzero rate samples as conservative cell intervals.

    Rate CSVs are sampled on a regular-ish energy grid.  The half-width is
    half the median positive spacing, clipped at the local edge.  These are
    diagnostic intervals, not a replacement for the runtime interpolation.
    """
    if not samples:
        return []
    energies = [energy for energy, _ in samples]
    if len(energies) == 1:
        half = 0.0
    else:
        spacings = [b - a for a, b in zip(energies, energies[1:]) if b > a]
        half = 0.5 * (sorted(spacings)[len(spacings) // 2] if spacings else 0.0)
    positive = [energy for energy, rate in samples if rate > 0.0]
    if not positive:
        return []
    return _merge_intervals((energy - half, energy + half) for energy in positive)


def _inside(energy: float, intervals: list[list[float]]) -> bool:
    return any(left - 1.0e-9 <= energy <= right + 1.0e-9 for left, right in intervals)


def _package_nodes(path: Path, requested: set[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], list[float]]:
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if len(mm) < PACKAGE_HEADER.size:
                raise ValueError(f"package is shorter than CINPKG03 header: {path}")
            values = PACKAGE_HEADER.unpack_from(mm, 0)
            (magic, version, header_size, endian, index_size, interaction_size,
             product_size, _flags, cell_count, interaction_count, product_count,
             file_size, _minimum_energy, _bin_width, _minimum_events,
             node_count, _uuid) = values
            if magic != b"CINPKG03" or version != 3 or header_size != PACKAGE_HEADER.size:
                raise ValueError(f"unsupported CINPKG03 header: {path}")
            if endian != 0x01020304 or file_size != len(mm):
                raise ValueError(f"CINPKG03 header/file mismatch: {path}")
            # The node array starts after the fixed cell/index/interaction/
            # product streams.  Validate the full extent before scanning.
            global_base = (header_size + cell_count * index_size
                           + interaction_count * interaction_size
                           + product_count * product_size)
            node_end = global_base + node_count * PACKAGE_NODE.size
            if node_end > len(mm):
                raise ValueError(f"truncated CINPKG03 global node array: {path}")
            nodes: dict[tuple[int, int, int, int], list[float]] = {
                key: [] for key in requested
            }
            for node_id in range(int(node_count)):
                z, a, target_z, target_a, energy = PACKAGE_NODE.unpack_from(
                    mm, global_base + node_id * PACKAGE_NODE.size)
                key = (int(z), int(a), int(target_z), int(target_a))
                if key in nodes:
                    nodes[key].append(float(energy))
            for values_for_key in nodes.values():
                values_for_key.sort()
            return nodes


def census(rate_path: Path, package_path: Path, tolerance: float = TOLERANCE_MEVU) -> dict[str, Any]:
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    rate_groups = _read_rate_csv(rate_path)
    package_groups = _package_nodes(package_path, set(KEYS))
    rows: list[dict[str, Any]] = []
    for key in KEYS:
        projectile_z, projectile_a, target_z, target_a = key
        rate_samples = rate_groups.get(key, [])
        package_energies = sorted(set(package_groups.get(key, [])))
        package_support = _merge_intervals(
            (energy - tolerance, energy + tolerance) for energy in package_energies
        )
        positive_rate_samples = [(energy, rate) for energy, rate in rate_samples if rate > 0.0]
        rate_nonzero_intervals = _sample_support_intervals(positive_rate_samples)
        outside_samples = [energy for energy, _ in positive_rate_samples
                           if not _inside(energy, package_support)]
        # A support interval is considered rate-covered when it intersects a
        # positive rate sample interval.  Keep the explicit interval list so
        # the boundary gaps can be inspected without a huge node dump.
        package_outside_rate = []
        for left, right in package_support:
            covered = [interval for interval in rate_nonzero_intervals
                       if interval[1] >= left and interval[0] <= right]
            if not covered:
                package_outside_rate.append([left, right])
        rows.append({
            "isotope": LABELS[(projectile_z, projectile_a)],
            "projectile_z": projectile_z,
            "projectile_a": projectile_a,
            "target": TARGET_LABELS[(target_z, target_a)],
            "target_z": target_z,
            "target_a": target_a,
            "rate_group_exists": bool(rate_samples),
            "rate_sample_count": len(rate_samples),
            "rate_energy_min_MeV_per_u": rate_samples[0][0] if rate_samples else None,
            "rate_energy_max_MeV_per_u": rate_samples[-1][0] if rate_samples else None,
            "rate_positive_sample_count": len(positive_rate_samples),
            "rate_nonzero_energy_min_MeV_per_u": positive_rate_samples[0][0] if positive_rate_samples else None,
            "rate_nonzero_energy_max_MeV_per_u": positive_rate_samples[-1][0] if positive_rate_samples else None,
            "rate_nonzero_intervals_MeV_per_u": rate_nonzero_intervals,
            "package_group_exists": bool(package_energies),
            "package_node_count": len(package_energies),
            "package_energy_min_MeV_per_u": package_energies[0] if package_energies else None,
            "package_energy_max_MeV_per_u": package_energies[-1] if package_energies else None,
            "package_support_intervals_MeV_per_u": package_support,
            "positive_rate_samples_outside_package_support": outside_samples,
            "positive_rate_sample_outside_count": len(outside_samples),
            "package_support_intervals_without_positive_rate_overlap_MeV_per_u": package_outside_rate,
            "package_support_interval_without_positive_rate_count": len(package_outside_rate),
        })
    summary = {
        "row_count": len(rows),
        "rate_group_count": sum(row["rate_group_exists"] for row in rows),
        "positive_rate_group_count": sum(row["rate_positive_sample_count"] > 0 for row in rows),
        "package_group_count": sum(row["package_group_exists"] for row in rows),
        "groups_with_positive_rate_but_no_package_support": sum(
            row["rate_positive_sample_count"] > 0 and not row["package_group_exists"] for row in rows),
        "groups_with_package_support_but_no_positive_rate": sum(
            row["package_group_exists"] and row["rate_positive_sample_count"] == 0 for row in rows),
        "positive_rate_samples_outside_package_support": sum(
            row["positive_rate_sample_outside_count"] for row in rows),
        "package_support_intervals_without_positive_rate_overlap": sum(
            row["package_support_interval_without_positive_rate_count"] for row in rows),
        "be6_rows": [
            {key: row[key] for key in (
                "isotope", "target", "rate_group_exists", "rate_sample_count",
                "rate_positive_sample_count", "rate_nonzero_energy_min_MeV_per_u",
                "rate_nonzero_energy_max_MeV_per_u", "package_group_exists",
                "package_node_count", "package_energy_min_MeV_per_u",
                "package_energy_max_MeV_per_u")}
            for row in rows if row["isotope"] == "6Be"
        ],
    }
    return {
        "format": "CINEL02_RATE_PACKAGE_COVERAGE_CENSUS_V1",
        "tolerance_MeV_per_u": tolerance,
        "species": [{"label": label, "z": z, "a": a} for z, a, label in SPECIES],
        "targets": [{"label": label, "z": z, "a": a} for z, a, label in TARGETS],
        "inputs": {
            "rate_csv": {"path": str(rate_path), "sha256": _sha256(rate_path)},
            "package": {"path": str(package_path), "sha256": _sha256(package_path)},
        },
        "rows": rows,
        "summary": summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "isotope", "projectile_z", "projectile_a", "target", "target_z", "target_a",
        "rate_group_exists", "rate_sample_count", "rate_energy_min_MeV_per_u",
        "rate_energy_max_MeV_per_u", "rate_positive_sample_count",
        "rate_nonzero_energy_min_MeV_per_u", "rate_nonzero_energy_max_MeV_per_u",
        "package_group_exists", "package_node_count", "package_energy_min_MeV_per_u",
        "package_energy_max_MeV_per_u", "positive_rate_sample_outside_count",
        "package_support_interval_without_positive_rate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    rows = report["rows"]
    lines = [
        "# CINEL02 rate/package coverage census",
        "",
        f"- Format: `{report['format']}`; replay tolerance: `{report['tolerance_MeV_per_u']}` MeV/u",
        f"- Rate groups: `{summary['rate_group_count']}/36`; positive-rate groups: `{summary['positive_rate_group_count']}/36`",
        f"- Package groups: `{summary['package_group_count']}/36`",
        f"- Positive rate samples outside package support: `{summary['positive_rate_samples_outside_package_support']}`",
        "",
        "| isotope | target | rate group | positive samples | rate nonzero range (MeV/u) | package nodes | package range (MeV/u) | rate samples outside support | support intervals without positive rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rate_range = (
            f"{row['rate_nonzero_energy_min_MeV_per_u']:.3g}–{row['rate_nonzero_energy_max_MeV_per_u']:.3g}"
            if row["rate_positive_sample_count"] else "—")
        package_range = (
            f"{row['package_energy_min_MeV_per_u']:.3g}–{row['package_energy_max_MeV_per_u']:.3g}"
            if row["package_node_count"] else "—")
        lines.append(
            f"| {row['isotope']} | {row['target']} | {'yes' if row['rate_group_exists'] else 'no'} "
            f"| {row['rate_positive_sample_count']} | {rate_range} | {row['package_node_count']} "
            f"| {package_range} | {row['positive_rate_sample_outside_count']} "
            f"| {row['package_support_interval_without_positive_rate_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", required=True, type=Path)
    parser.add_argument("--package", required=True, dest="package_path", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_MEVU)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = census(args.rate, args.package_path, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output:
        _write_csv(args.csv_output, report["rows"])
    if args.markdown_output:
        _write_markdown(args.markdown_output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
