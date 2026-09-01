#!/usr/bin/env python3
"""Deterministic CINPKG03 replay-support/yield auditor.

The auditor consumes a runtime energy ledger and a CINPKG03 package.  It uses
exactly the persisted global energy-node index and the runtime +/- tolerance
window to aggregate package final-state expectations at the ledger's observed
replay energies.  No Monte Carlo sampling or package mutation is performed.

The ledger only stores per-cell energy sums, rather than every query energy, so
expectations are evaluated at the cell mean replay energy.  This limitation is
reported explicitly in the JSON output; it is still sufficient to distinguish
runtime occupancy/support problems from large package-yield errors.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import mmap
import re
import struct
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_HEADER = struct.Struct("<8sIIIIIIIQQQQffQQ40s")
PACKAGE_INDEX = struct.Struct("<hhhhIQQff")
PACKAGE_NODE = struct.Struct("<hhhhf")
PACKAGE_OFFSET = struct.Struct("<Q")
PACKAGE_EVENT_INDEX = struct.Struct("<Q")
INTERACTION = struct.Struct(
    "<QIQIIIihhfff" + "ff" + "fff" + "fff" + "ffff" + "ii" + "hhI" +
    "iii" + "iihh" + "f" * 13 + "ff" + "IIf" + "f" * 10 +
    "64s32s64s64s"
)
PRODUCT = struct.Struct("<ihh" + "f" * 15 + "i")
STATUS_COUNT = 5
ENERGY_BIN_COUNT = 40
ENERGY_BIN_WIDTH_MEVU = 10.0
STATUS_CANDIDATE = 0
STATUS_VALID = 1
STATUS_MISS = 2
STATUS_INVALID = 3
STATUS_CUTOFF = 4
TOLERANCE_MEVU = 0.51

# Keep this in the same order as Cinel02ReplayLedgerSchema/get_charged_species_idx.
SPECIES = [
    "1H", "2H", "3H", "3He", "4He", "6He", "6Li", "7Li", "7Be",
    "9Be", "10Be", "8B", "10B", "11B", "10C", "11C", "12C", "6Be",
]
ELEMENT_Z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6}
SPECIES_ZA = []
for name in SPECIES:
    match = re.fullmatch(r"(\d+)([A-Z][a-z]?)", name)
    if not match or match.group(2) not in ELEMENT_Z:
        raise RuntimeError(f"invalid species name: {name}")
    SPECIES_ZA.append((ELEMENT_Z[match.group(2)], int(match.group(1))))
SPECIES_INDEX = {za: i for i, za in enumerate(SPECIES_ZA)}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_queries(ledger: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layout = ledger.get("cinel02_replay_status_layout")
    if not isinstance(layout, dict) or layout.get("shape") != [18, 2, 3, ENERGY_BIN_COUNT, STATUS_COUNT]:
        raise ValueError("ledger does not contain the Step 01B.1 replay-status shape [18,2,3,40,5]")
    counts = ledger.get("cinel02_replay_status_counts")
    replay_energy = ledger.get("cinel02_replay_status_replay_query_energy_MeV")
    if not isinstance(counts, list) or not isinstance(replay_energy, list):
        raise ValueError("ledger is missing replay status counts/replay energies")
    expected_len = 18 * 2 * 3 * ENERGY_BIN_COUNT * STATUS_COUNT
    if len(counts) != expected_len or len(replay_energy) != expected_len:
        raise ValueError("replay status arrays have an unexpected length")
    queries = []
    for species in range(18):
        z, a = SPECIES_ZA[species]
        for target in range(2):
            target_z, target_a = ((1, 1), (8, 16))[target]
            for generation in range(3):
                for energy_bin in range(ENERGY_BIN_COUNT):
                    base = ((((species * 2) + target) * 3 + generation) * ENERGY_BIN_COUNT + energy_bin) * STATUS_COUNT
                    candidate = int(counts[base + STATUS_CANDIDATE])
                    valid = int(counts[base + STATUS_VALID])
                    replay_sum = float(replay_energy[base + STATUS_CANDIDATE])
                    mean_energy = replay_sum / candidate if candidate else 0.0
                    if candidate:
                        queries.append({
                            "index": len(queries),
                            "species": species,
                            "projectile_z": z,
                            "projectile_a": a,
                            "target": target,
                            "target_z": target_z,
                            "target_a": target_a,
                            "reaction_generation": generation,
                            "energy_bin": energy_bin,
                            "candidate_count": candidate,
                            "valid_count": valid,
                            "lookup_miss_count": int(counts[base + STATUS_MISS]),
                            "invalid_count": int(counts[base + STATUS_INVALID]),
                            "cutoff_count": int(counts[base + STATUS_CUTOFF]),
                            "mean_replay_energy_MeV_per_u": mean_energy / max(1, a),
                        })
    return queries, layout


def _package_layout(mm: mmap.mmap) -> dict[str, int | float | str]:
    if len(mm) < PACKAGE_HEADER.size:
        raise ValueError("package is shorter than CINPKG03 header")
    values = PACKAGE_HEADER.unpack_from(mm, 0)
    (magic, version, header_size, endian, index_size, interaction_size,
     product_size, flags, cell_count, interaction_count, product_count,
     file_size, minimum_energy, bin_width, minimum_events, node_count, uuid) = values
    if magic != b"CINPKG03" or version != 3 or header_size != PACKAGE_HEADER.size:
        raise ValueError("unsupported CINPKG03 header")
    if endian != 0x01020304 or index_size != PACKAGE_INDEX.size or interaction_size != INTERACTION.size or product_size != PRODUCT.size:
        raise ValueError("CINPKG03 record-size/endian mismatch")
    if file_size != len(mm):
        raise ValueError(f"package file-size mismatch: header={file_size}, actual={len(mm)}")
    if node_count <= 0:
        raise ValueError("package has no global energy nodes")
    interaction_base = header_size + cell_count * PACKAGE_INDEX.size
    product_base = interaction_base + interaction_count * INTERACTION.size
    global_base = product_base + product_count * PRODUCT.size
    expected_global = node_count * PACKAGE_NODE.size + (node_count + 1) * PACKAGE_OFFSET.size + interaction_count * PACKAGE_EVENT_INDEX.size
    if global_base + expected_global != len(mm):
        raise ValueError("CINPKG03 global-index extent mismatch")
    return {
        "cell_count": int(cell_count), "interaction_count": int(interaction_count),
        "product_count": int(product_count), "node_count": int(node_count),
        "interaction_base": int(interaction_base), "product_base": int(product_base),
        "global_base": int(global_base), "node_count_u64": int(node_count),
        "minimum_energy_MeV_per_u": float(minimum_energy),
        "energy_bin_width_MeV_per_u": float(bin_width),
        # The fixed-width UUID field is NUL padded.  Use a real NUL byte here;
        # splitting on the two-character string ``\\0`` leaves padding in the
        # report and makes package identity comparisons fail spuriously.
        "campaign_uuid": uuid.split(b"\0", 1)[0].decode("ascii", "replace"),
        "flags": int(flags), "minimum_events_per_bin": int(minimum_events),
    }


def _build_global_index(mm: mmap.mmap, layout: dict[str, int | float | str]):
    node_count = int(layout["node_count"])
    global_base = int(layout["global_base"])
    offsets_base = global_base + node_count * PACKAGE_NODE.size
    indices_base = offsets_base + (node_count + 1) * PACKAGE_OFFSET.size
    nodes_by_key: dict[tuple[int, int, int, int], list[tuple[float, int]]] = defaultdict(list)
    node_ranges: list[tuple[int, int]] = []
    for node_id in range(node_count):
        z, a, tz, ta, energy = PACKAGE_NODE.unpack_from(mm, global_base + node_id * PACKAGE_NODE.size)
        nodes_by_key[(int(z), int(a), int(tz), int(ta))].append((float(energy), node_id))
        start = PACKAGE_OFFSET.unpack_from(mm, offsets_base + node_id * 8)[0]
        end = PACKAGE_OFFSET.unpack_from(mm, offsets_base + (node_id + 1) * 8)[0]
        node_ranges.append((int(start), int(end)))
    for values in nodes_by_key.values():
        values.sort()
    event_to_node = array("i", [-1]) * int(layout["interaction_count"])
    for node_id, (start, end) in enumerate(node_ranges):
        if end < start or end > int(layout["interaction_count"]):
            raise ValueError("invalid global energy-node offset")
        for offset in range(start, end):
            event_index = PACKAGE_EVENT_INDEX.unpack_from(mm, indices_base + offset * 8)[0]
            if event_index >= len(event_to_node) or event_to_node[event_index] != -1:
                raise ValueError("global energy index is not a permutation")
            event_to_node[event_index] = node_id
    if any(value < 0 for value in event_to_node):
        raise ValueError("global energy index does not cover all interactions")
    return nodes_by_key, node_ranges, event_to_node


def _matching_nodes(nodes_by_key, query: dict[str, Any], tolerance: float):
    values = nodes_by_key.get((query["projectile_z"], query["projectile_a"], query["target_z"], query["target_a"]), [])
    energy = query["mean_replay_energy_MeV_per_u"]
    left = bisect.bisect_left(values, (energy - tolerance, -1))
    right = bisect.bisect_right(values, (energy + tolerance, 2**63 - 1))
    return [node_id for _, node_id in values[left:right]]


def _empty_accumulator():
    return {
        "events": 0, "parent_continued": 0, "parent_killed": 0,
        "incident_energy_MeV_per_u": 0.0, "parent_after_energy_MeV": 0.0,
        "local_deposit_MeV": 0.0, "products": [0] * len(SPECIES),
        "product_count_sq": [0.0] * len(SPECIES),
        "product_kinetic_MeV": [0.0] * len(SPECIES),
    }


def _audit_package(mm: mmap.mmap, layout: dict[str, int | float | str], queries: list[dict[str, Any]], tolerance: float):
    nodes_by_key, node_ranges, event_to_node = _build_global_index(mm, layout)
    node_queries: dict[int, list[int]] = defaultdict(list)
    query_support: list[list[int]] = []
    for query in queries:
        node_ids = _matching_nodes(nodes_by_key, query, tolerance)
        query_support.append(node_ids)
        for node_id in node_ids:
            node_queries[node_id].append(query["index"])
    accum = [_empty_accumulator() for _ in queries]
    interaction_base = int(layout["interaction_base"])
    product_base = int(layout["product_base"])
    product_offset = 0
    needed_events = 0
    for event_index in range(int(layout["interaction_count"])):
        values = INTERACTION.unpack_from(mm, interaction_base + event_index * INTERACTION.size)
        direct_count = int(values[51])
        node_id = event_to_node[event_index]
        qids = node_queries.get(node_id)
        if qids:
            needed_events += 1
            projectile_z, projectile_a = int(values[7]), int(values[8])
            target_z, target_a = int(values[26]), int(values[27])
            energy = float(values[13])
            parent_status = int(values[32])
            parent_energy = float(values[39])
            local = float(values[49])
            per_event = []
            for product_index in range(direct_count):
                product = PRODUCT.unpack_from(mm, product_base + (product_offset + product_index) * PRODUCT.size)
                pz, pa, kinetic, role = int(product[1]), int(product[2]), float(product[6]), int(product[18])
                if role == 0:
                    species = SPECIES_INDEX.get((pz, pa))
                    if species is not None:
                        per_event.append((species, kinetic))
            # Keep the per-event multiplicity so the auditor can report an
            # approximate sampling standard error for the expected yield.
            # This is a package-event variance, not a TOPAS/GPU uncertainty
            # estimate; it is explicitly labelled as such in the report.
            event_counts = [0] * len(SPECIES)
            for species, _ in per_event:
                event_counts[species] += 1
            for qid in qids:
                item = accum[qid]
                item["events"] += 1
                item["incident_energy_MeV_per_u"] += energy
                item["parent_after_energy_MeV"] += max(0.0, parent_energy) if parent_status == 0 else 0.0
                item["local_deposit_MeV"] += max(0.0, local)
                if parent_status == 0:
                    item["parent_continued"] += 1
                else:
                    item["parent_killed"] += 1
                for species, kinetic in per_event:
                    item["products"][species] += 1
                    item["product_kinetic_MeV"][species] += max(0.0, kinetic)
                for species, count in enumerate(event_counts):
                    item["product_count_sq"][species] += float(count * count)
        product_offset += direct_count
    if product_offset != int(layout["product_count"]):
        raise ValueError(f"package product stream mismatch: {product_offset} != {layout['product_count']}")
    rows = []
    for query, support, item in zip(queries, query_support, accum):
        events = item["events"]
        valid = query["valid_count"]
        candidate = query["candidate_count"]
        expected_valid_counts = [valid * n / events for n in item["products"]] if events else [0.0] * len(SPECIES)
        expected_candidate_counts = [candidate * n / events for n in item["products"]] if events else [0.0] * len(SPECIES)
        expected_valid_ke = [valid * value / events for value in item["product_kinetic_MeV"]] if events else [0.0] * len(SPECIES)
        expected_valid_count_stddev = []
        for species in range(len(SPECIES)):
            if not events:
                expected_valid_count_stddev.append(0.0)
                continue
            mean = item["products"][species] / events
            second_moment = item["product_count_sq"][species] / events
            variance = max(0.0, second_moment - mean * mean)
            expected_valid_count_stddev.append(math.sqrt(valid * variance))
        rows.append({
            **query,
            "support_node_count": len(support),
            "support_event_count": events,
            "valid_reuse_weight": (valid / events if events else None),
            "candidate_reuse_weight": (candidate / events if events else None),
            "package_parent_continue_probability": (item["parent_continued"] / events if events else None),
            "package_parent_kill_probability": (item["parent_killed"] / events if events else None),
            "package_incident_energy_mean_MeV_per_u": (item["incident_energy_MeV_per_u"] / events if events else None),
            "package_products_per_event": item["products"],
            "package_product_kinetic_MeV_per_event": item["product_kinetic_MeV"],
            "expected_valid_product_counts": expected_valid_counts,
            "expected_valid_product_count_stddev": expected_valid_count_stddev,
            "expected_candidate_product_counts": expected_candidate_counts,
            "expected_valid_product_kinetic_MeV": expected_valid_ke,
            "expected_valid_parent_continue_count": valid * item["parent_continued"] / events if events else 0.0,
            "expected_valid_parent_kill_count": valid * item["parent_killed"] / events if events else 0.0,
            "support_qualified": events > 0,
        })
    return rows, {"needed_interactions": needed_events, "query_count": len(queries), "node_count": int(layout["node_count"]), "interaction_count": int(layout["interaction_count"])}


def _actual_transition(ledger: dict[str, Any], key: str) -> list[int]:
    values = ledger.get(key)
    if not isinstance(values, list) or len(values) != 18 * 18:
        raise ValueError(f"ledger missing {key}")
    return [int(value) for value in values]


def _summarise(rows: list[dict[str, Any]], ledger: dict[str, Any]) -> dict[str, Any]:
    expected = [0.0] * (18 * 18)
    expected_variance = [0.0] * (18 * 18)
    expected_ke = [0.0] * (18 * 18)
    expected_parent_continue = [0.0] * 18
    expected_parent_kill = [0.0] * 18
    expected_parent_continue_cell = [0.0] * (18 * 2 * 3)
    expected_parent_kill_cell = [0.0] * (18 * 2 * 3)
    support_missing = []
    for row in rows:
        parent = int(row["species"])
        for child, value in enumerate(row["expected_valid_product_counts"]):
            expected[parent * 18 + child] += value
        for child, value in enumerate(row["expected_valid_product_kinetic_MeV"]):
            expected_ke[parent * 18 + child] += value
        for child, value in enumerate(row["expected_valid_product_count_stddev"]):
            expected_variance[parent * 18 + child] += float(value) * float(value)
        expected_parent_continue[parent] += float(row["expected_valid_parent_continue_count"])
        expected_parent_kill[parent] += float(row["expected_valid_parent_kill_count"])
        outcome_cell = ((parent * 2 + int(row["target"])) * 3 + int(row["reaction_generation"]))
        expected_parent_continue_cell[outcome_cell] += float(row["expected_valid_parent_continue_count"])
        expected_parent_kill_cell[outcome_cell] += float(row["expected_valid_parent_kill_count"])
        if not row["support_qualified"]:
            support_missing.append({"species": SPECIES[parent], "target": row["target"], "reaction_generation": row["reaction_generation"], "energy_bin": row["energy_bin"], "valid_count": row["valid_count"]})
    actual_generated = _actual_transition(ledger, "cinel02_generated_transition_counts")
    actual_queued = _actual_transition(ledger, "cinel02_queued_transition_counts")
    transition_rows = []
    for parent in range(18):
        for child in range(18):
            index = parent * 18 + child
            transition_rows.append({
                "parent_species": SPECIES[parent], "child_species": SPECIES[child],
                "actual_generated_count": actual_generated[index],
                "actual_queued_count": actual_queued[index],
                "expected_valid_generated_count": expected[index],
                "expected_valid_generated_count_stddev": math.sqrt(expected_variance[index]),
                "expected_valid_generated_kinetic_MeV": expected_ke[index],
                "count_difference_generated_minus_expected": actual_generated[index] - expected[index],
                "count_difference_z_score": (
                    (actual_generated[index] - expected[index]) /
                    math.sqrt(expected_variance[index])
                    if expected_variance[index] > 0.0 else None
                ),
            })
    child_summary = []
    for child, name in enumerate(SPECIES):
        actual = sum(actual_generated[parent * 18 + child] for parent in range(18))
        expected_child = sum(expected[parent * 18 + child] for parent in range(18))
        stddev = math.sqrt(sum(expected_variance[parent * 18 + child] for parent in range(18)))
        child_summary.append({
            "species": name,
            "actual_generated_count": actual,
            "expected_valid_generated_count": expected_child,
            "expected_valid_generated_count_stddev": stddev,
            "count_difference": actual - expected_child,
            "count_difference_z_score": (actual - expected_child) / stddev if stddev > 0.0 else None,
            "relative_difference_percent": (actual - expected_child) / expected_child * 100.0 if expected_child > 0.0 else None,
        })
    actual_parent_outcome = ledger.get("cinel02_parent_outcome_counts")
    if not isinstance(actual_parent_outcome, list) or len(actual_parent_outcome) != 18 * 2 * 3 * 2:
        raise ValueError("ledger missing isotope×target×generation parent outcome counts")
    parent_outcome_rows = []
    for parent in range(18):
        for target in range(2):
            for generation in range(3):
                cell = (parent * 2 + target) * 3 + generation
                actual_continue = int(actual_parent_outcome[cell * 2])
                actual_kill = int(actual_parent_outcome[cell * 2 + 1])
                expected_continue = expected_parent_continue_cell[cell]
                expected_kill = expected_parent_kill_cell[cell]
                expected_total = expected_continue + expected_kill
                actual_total = actual_continue + actual_kill
                parent_outcome_rows.append({
                    "projectile": SPECIES[parent],
                    "target": ("H", "O")[target],
                    "reaction_generation": generation,
                    "actual_valid_count": actual_total,
                    "actual_continue_count": actual_continue,
                    "actual_kill_count": actual_kill,
                    "actual_continue_fraction": (actual_continue / actual_total if actual_total else None),
                    "expected_valid_count": expected_total,
                    "expected_continue_count": expected_continue,
                    "expected_kill_count": expected_kill,
                    "expected_continue_fraction": (expected_continue / expected_total if expected_total else None),
                    "continue_count_difference": actual_continue - expected_continue,
                    "kill_count_difference": actual_kill - expected_kill,
                })
    total_expected = sum(expected)
    total_stddev = math.sqrt(sum(expected_variance))
    total_actual = sum(actual_generated)
    return {
        "species": SPECIES,
        "query_count": len(rows),
        "support_missing_query_count": len(support_missing),
        "support_missing_queries": support_missing,
        "expected_valid_generated_count_total": total_expected,
        "expected_valid_generated_count_total_stddev": total_stddev,
        "actual_generated_count_total": total_actual,
        "actual_queued_count_total": sum(actual_queued),
        "total_count_difference_z_score": (total_actual - total_expected) / total_stddev if total_stddev > 0.0 else None,
        "child_species_summary": child_summary,
        "transition_rows": transition_rows,
        "parent_outcome_rows": parent_outcome_rows,
        "note": "Expectation uses each occupied replay cell's mean post-EM energy; standard deviations are approximate package-event sampling errors. Ledger does not retain per-collision energies, so support and energy-occupancy conclusions remain provisional for cells with sparse nodes.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--tolerance-MeVu", type=float, default=TOLERANCE_MEVU)
    args = parser.parse_args()
    if args.tolerance_MeVu <= 0.0 or not math.isfinite(args.tolerance_MeVu):
        parser.error("--tolerance-MeVu must be finite and positive")
    ledger = _read_json(args.ledger)
    queries, status_layout = _load_queries(ledger)
    with args.package.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            package_layout = _package_layout(mm)
            rows, scan_summary = _audit_package(mm, package_layout, queries, args.tolerance_MeVu)
    summary = _summarise(rows, ledger)
    report = {
        "schema": "CINEL02_RUNTIME_PACKAGE_AUDIT_V1",
        "package": str(args.package), "ledger": str(args.ledger),
        "tolerance_MeV_per_u": args.tolerance_MeVu,
        "package_layout": package_layout,
        "replay_status_layout": status_layout,
        "scan": scan_summary,
        "queries": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w") as stream:
            stream.write("projectile,target,reaction_generation,energy_bin,candidate,valid,lookup_miss,cutoff,mean_replay_energy_MeV_per_u,support_nodes,support_events,valid_reuse_weight,package_continue_probability,package_kill_probability\\n")
            for row in rows:
                def value(key):
                    return "" if row[key] is None else row[key]
                stream.write(",".join(str(value(key)) for key in (
                    "projectile_z", "target", "reaction_generation", "energy_bin",
                    "candidate_count", "valid_count", "lookup_miss_count", "cutoff_count",
                    "mean_replay_energy_MeV_per_u", "support_node_count", "support_event_count",
                    "valid_reuse_weight", "package_parent_continue_probability",
                    "package_parent_kill_probability")) + "\\n")
    print(json.dumps({
        "package_uuid": package_layout["campaign_uuid"],
        "query_count": len(rows),
        "support_missing_query_count": summary["support_missing_query_count"],
        "expected_valid_generated_count_total": summary["expected_valid_generated_count_total"],
        "actual_generated_count_total": summary["actual_generated_count_total"],
        "actual_queued_count_total": summary["actual_queued_count_total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
