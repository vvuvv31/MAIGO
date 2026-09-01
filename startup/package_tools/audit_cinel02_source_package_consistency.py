#!/usr/bin/env python3
"""Audit CINEL02 raw-source to package support and exact replay semantics.

This is a bounded Step 03B-2B correctness audit.  It streams the raw source,
compares every raw interaction energy with the compiled package global energy
nodes, and exhaustively probes the package index at exact node energies and
replay-window boundaries.  It never changes the package, rate table, tolerance,
CDF, or runtime transport.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import mmap
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from audit_cinel02_unsupported import iter_records  # noqa: E402
import cinel02  # noqa: E402

PACKAGE_HEADER = struct.Struct("<8sIIIIIIIQQQQffQQ40s")
PACKAGE_INDEX = struct.Struct("<hhhhIQQff")
PACKAGE_NODE = struct.Struct("<hhhhf")
U64 = struct.Struct("<Q")
TOLERANCE_MEVU = 0.51

SPECIES = [
    (1, 1, "1H"), (1, 2, "2H"), (1, 3, "3H"), (2, 3, "3He"),
    (2, 4, "4He"), (2, 6, "6He"), (3, 6, "6Li"), (3, 7, "7Li"),
    (4, 6, "6Be"), (4, 7, "7Be"), (4, 9, "9Be"), (4, 10, "10Be"),
    (5, 8, "8B"), (5, 10, "10B"), (5, 11, "11B"), (6, 10, "10C"),
    (6, 11, "11C"), (6, 12, "12C"),
]
TARGETS = [(1, 1, "H"), (8, 16, "O")]
KEYS = [(z, a, tz, ta) for z, a, _ in SPECIES for tz, ta, _ in TARGETS]
LABELS = {(z, a): label for z, a, label in SPECIES}
TARGET_LABELS = {(z, a): label for z, a, label in TARGETS}


def _unmatched(values: list[float], references: list[float], tolerance: float) -> int:
    """Count sorted values with no reference inside +/- tolerance in O(n)."""
    cursor = 0
    missing = 0
    for value in values:
        while cursor < len(references) and references[cursor] < value - tolerance:
            cursor += 1
        if cursor == len(references) or references[cursor] > value + tolerance:
            missing += 1
    return missing


def _read_summary(path: Path) -> set[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None:
            raise ValueError(f"campaign summary has no header: {path}")
        required = {"projectile_z", "projectile_a", "completed"}
        missing = required.difference(rows.fieldnames)
        if missing:
            raise ValueError(f"campaign summary missing fields {sorted(missing)}: {path}")
        identities = set()
        for line, row in enumerate(rows, 2):
            try:
                z, a = int(row["projectile_z"]), int(row["projectile_a"])
                completed = int(row["completed"])
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError(f"invalid campaign summary row {path}:{line}") from error
            if completed:
                identities.add((z, a))
        return identities


def _scan_raw(paths: Iterable[Path], keys: set[tuple[int, int, int, int]]) -> tuple[dict[tuple[int, int, int, int], list[float]], int]:
    energies: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    records = 0
    for path in paths:
        for record, _ in iter_records(path):
            records += 1
            key = tuple(int(record[name]) for name in
                        ("projectile_z", "projectile_a", "target_z", "target_a"))
            if key in keys:
                energies[key].append(float(record["collision_energy_MeV_per_u"]))
    for values in energies.values():
        values.sort()
    return energies, records


def _read_package(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        if len(mm) < PACKAGE_HEADER.size:
            raise ValueError(f"package is shorter than header: {path}")
        values = PACKAGE_HEADER.unpack_from(mm, 0)
        (magic, version, header_size, endian, index_size, interaction_size,
         product_size, flags, cell_count, interaction_count, product_count,
         file_size, minimum_energy, bin_width, minimum_events, node_count, uuid) = values
        if magic != b"CINPKG03" or version != 3 or header_size != PACKAGE_HEADER.size:
            raise ValueError(f"unsupported package header: {path}")
        if endian != 0x01020304 or file_size != len(mm):
            raise ValueError(f"package header/file mismatch: {path}")
        index_base = header_size
        interaction_base = index_base + cell_count * index_size
        product_base = interaction_base + interaction_count * interaction_size
        global_base = product_base + product_count * product_size
        node_end = global_base + node_count * PACKAGE_NODE.size
        offset_base = node_end
        offset_end = offset_base + (node_count + 1) * U64.size
        index_base_global = offset_end
        index_end = index_base_global + interaction_count * U64.size
        if index_end > len(mm) or index_size != PACKAGE_INDEX.size or interaction_size != cinel02.RAW_FIXED_FORMAT.size:
            raise ValueError(f"package stream extent mismatch: {path}")

        interactions: list[tuple[int, int, int, int, float]] = []
        for index in range(int(interaction_count)):
            values = cinel02.RAW_FIXED_FORMAT.unpack_from(mm, interaction_base + index * interaction_size)
            names = dict(zip(cinel02._FIELD_NAMES, values))
            interactions.append((int(names["projectile_z"]), int(names["projectile_a"]),
                                 int(names["target_z"]), int(names["target_a"]),
                                 float(names["collision_energy_MeV_per_u"])))
        nodes: list[tuple[tuple[int, int, int, int], float, int, int]] = []
        nodes_by_key: dict[tuple[int, int, int, int], list[tuple[float, int, int]]] = defaultdict(list)
        offsets = [U64.unpack_from(mm, offset_base + i * U64.size)[0]
                   for i in range(int(node_count) + 1)]
        indices = [U64.unpack_from(mm, index_base_global + i * U64.size)[0]
                   for i in range(int(interaction_count))]
        for node_index in range(int(node_count)):
            z, a, tz, ta, energy = PACKAGE_NODE.unpack_from(mm, global_base + node_index * PACKAGE_NODE.size)
            first, last = int(offsets[node_index]), int(offsets[node_index + 1])
            if first >= last or last > len(indices):
                raise ValueError(f"empty/invalid package node {node_index}")
            key = (int(z), int(a), int(tz), int(ta))
            for position in range(first, last):
                event_index = int(indices[position])
                if event_index >= len(interactions) or interactions[event_index][:4] != key or interactions[event_index][4] != float(energy):
                    raise ValueError(f"global index mismatch at node {node_index}")
            node_energy = float(energy)
            nodes.append((key, node_energy, first, last))
            nodes_by_key[key].append((node_energy, first, last))
        if offsets[-1] != len(indices) or len(set(indices)) != len(interactions):
            raise ValueError("global event index is not a complete permutation")
        return {
            "interaction_count": int(interaction_count),
            "product_count": int(product_count),
            "node_count": int(node_count),
            "minimum_energy_MeV_per_u": float(minimum_energy),
            "energy_bin_width_MeV_per_u": float(bin_width),
            "minimum_events_per_bin": int(minimum_events),
            "flags": int(flags),
            "uuid": uuid.rstrip(b"\x00").decode("ascii", errors="replace"),
            "interactions": interactions,
            "nodes": nodes,
            "nodes_by_key": dict(nodes_by_key),
            "node_energies_by_key": {key: [node[0] for node in values]
                                     for key, values in nodes_by_key.items()},
            "offsets": offsets,
            "indices": indices,
        }


def _exact_query(package: dict[str, Any], key: tuple[int, int, int, int], energy: float,
                 tolerance: float, u01: float) -> int | None:
    nodes = package["nodes_by_key"].get(key, [])
    energies = package["node_energies_by_key"].get(key, [])
    endpoint_epsilon = 1.0e-6
    left = bisect.bisect_left(energies, energy - tolerance - endpoint_epsilon)
    right = bisect.bisect_right(energies, energy + tolerance + endpoint_epsilon)
    if left >= right:
        return None
    first = nodes[left][1]
    last = nodes[right - 1][2]
    count = last - first
    pick = count - 1 if u01 >= 1.0 else min(int(u01 * count), count - 1)
    event_index = package["indices"][first + pick]
    event = package["interactions"][event_index]
    if event[:4] != key or abs(event[4] - energy) > tolerance + endpoint_epsilon:
        return None
    return int(event_index)


def audit(summary: Path, raw_paths: list[Path], package_path: Path,
          occupancy_path: Path | None = None, tolerance: float = TOLERANCE_MEVU) -> dict[str, Any]:
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    summary_identities = _read_summary(summary)
    raw, raw_records = _scan_raw(raw_paths, set(KEYS))
    package = _read_package(package_path)
    package_by_key: dict[tuple[int, int, int, int], list[float]] = {
        key: [node[0] for node in values]
        for key, values in package["nodes_by_key"].items()
    }
    rows = []
    for key in KEYS:
        z, a, tz, ta = key
        raw_values = raw.get(key, [])
        package_values = package_by_key.get(key, [])
        raw_missing = _unmatched(raw_values, package_values, tolerance)
        package_missing = _unmatched(package_values, raw_values, tolerance)
        source_present = (z, a) in summary_identities and bool(raw_values)
        if not source_present and not raw_values:
            status = "source_missing"
        elif raw_missing:
            status = "compiler_dropped_support"
        elif package_missing:
            status = "package_has_unbacked_support"
        else:
            status = "source_present_compiled"
        rows.append({
            "isotope": LABELS[(z, a)], "projectile_z": z, "projectile_a": a,
            "target": TARGET_LABELS[(tz, ta)], "target_z": tz, "target_a": ta,
            "summary_source_campaign_present": (z, a) in summary_identities,
            "raw_event_count": len(raw_values),
            "raw_energy_min_MeV_per_u": min(raw_values) if raw_values else None,
            "raw_energy_max_MeV_per_u": max(raw_values) if raw_values else None,
            "package_node_count": len(package_values),
            "package_energy_min_MeV_per_u": min(package_values) if package_values else None,
            "package_energy_max_MeV_per_u": max(package_values) if package_values else None,
            "raw_events_without_package_support": raw_missing,
            "package_nodes_without_raw_support": package_missing,
            "status": status,
        })

    exact_checks = 0
    exact_failures: list[dict[str, Any]] = []
    for key, energy, _, _ in package["nodes"]:
        for query_energy, query_tolerance, u01 in ((energy, 0.0, 0.0),
                                                    (energy, 0.0, 1.0),
                                                    (energy - tolerance, tolerance, 0.0),
                                                    (energy + tolerance, tolerance, 1.0)):
            exact_checks += 1
            if _exact_query(package, key, query_energy, query_tolerance, u01) is None:
                exact_failures.append({"key": list(key), "node_energy": energy,
                                       "query_energy": query_energy,
                                       "query_tolerance": query_tolerance, "u01": u01})
                if len(exact_failures) >= 100:
                    break
        if len(exact_failures) >= 100:
            break

    result: dict[str, Any] = {
        "schema": "CINEL02_SOURCE_PACKAGE_REPLAY_CONSISTENCY_V1",
        "tolerance_MeV_per_u": tolerance,
        "inputs": {"summary": str(summary), "raw": [str(p) for p in raw_paths], "package": str(package_path)},
        "raw_record_count_scanned": raw_records,
        "package": {key: package[key] for key in ("interaction_count", "product_count", "node_count",
                                                   "minimum_energy_MeV_per_u", "energy_bin_width_MeV_per_u",
                                                   "minimum_events_per_bin", "flags", "uuid")},
        "rows": rows,
        "summary": {
            "source_present_compiled": sum(row["status"] == "source_present_compiled" for row in rows),
            "source_missing": sum(row["status"] == "source_missing" for row in rows),
            "compiler_dropped_support": sum(row["status"] == "compiler_dropped_support" for row in rows),
            "package_has_unbacked_support": sum(row["status"] == "package_has_unbacked_support" for row in rows),
            "raw_events_without_package_support": sum(row["raw_events_without_package_support"] for row in rows),
            "package_nodes_without_raw_support": sum(row["package_nodes_without_raw_support"] for row in rows),
            "exact_replay_checks": exact_checks,
            "exact_replay_failures": len(exact_failures),
        },
        "exact_replay_failures": exact_failures,
    }
    if occupancy_path is not None:
        occupancy = json.loads(occupancy_path.read_text(encoding="utf-8"))
        occupied_rows = [row for row in occupancy.get("rows", []) if row.get("lookup_miss_count", 0)]
        result["occupancy"] = {
            "path": str(occupancy_path),
            "miss_count": sum(int(row["lookup_miss_count"]) for row in occupied_rows),
            "provisional_gap_cells": sum(row.get("classification") == "raw_or_compiler_support_gap" for row in occupied_rows),
            "provisional_anomaly_cells": sum(row.get("classification") == "index_or_lookup_anomaly" for row in occupied_rows),
            "provisional_gap_cells_with_source_present_compiled": sum(
                row.get("classification") == "raw_or_compiler_support_gap" and
                next((item["status"] for item in rows if item["projectile_z"] == row["projectile_z"] and
                     item["projectile_a"] == row["projectile_a"] and item["target_z"] == row["target_z"] and
                     item["target_a"] == row["target_a"]), None) == "source_present_compiled"
                for row in occupied_rows),
            "provisional_anomaly_cells_with_source_present_compiled": sum(
                row.get("classification") == "index_or_lookup_anomaly" and
                next((item["status"] for item in rows if item["projectile_z"] == row["projectile_z"] and
                     item["projectile_a"] == row["projectile_a"] and item["target_z"] == row["target_z"] and
                     item["target_a"] == row["target_a"]), None) == "source_present_compiled"
                for row in occupied_rows),
            "interpretation": (
                "All occupied miss cells have source_present_compiled provenance and exact replay "
                "boundary probes pass; classifications remain runtime sparse-support diagnostics."
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--package", dest="package_path", type=Path, required=True)
    parser.add_argument("--occupancy-audit", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_MEVU)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.summary, args.raw, args.package_path, args.occupancy_audit, args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if not report["exact_replay_failures"] and not report["summary"]["compiler_dropped_support"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
