#!/usr/bin/env python3
"""Build a generation-resolved Be cascade ledger from CINEL02 worker raw files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cinel02  # noqa: E402


ENERGY_EDGES = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, math.inf)
CHANNELS = {4: "Be_to_Be", 5: "B_to_Be", 6: "C_to_Be"}


def energy_bin(value: float) -> int:
    for index, upper in enumerate(ENERGY_EDGES[1:]):
        if value < upper:
            return index
    return len(ENERGY_EDGES) - 2


def track_generation(track_id: int, parents: dict[int, int], memo: dict[int, int]) -> int | None:
    if track_id in memo:
        return memo[track_id]
    visited: list[int] = []
    current = track_id
    while current not in memo:
        if current == 1:
            memo[current] = 0
            break
        if current in visited or current not in parents:
            return None
        visited.append(current)
        current = parents[current]
    generation = memo[current]
    for child in reversed(visited):
        generation += 1
        memo[child] = generation
    return memo.get(track_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.raw_directory.glob("worker_*.cinel02"))
    if not paths:
        raise SystemExit(f"no worker CINEL02 files in {args.raw_directory}")

    ledger: dict[tuple[int, int, int, str, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "kinetic_energy_MeV": 0.0}
    )
    incident: dict[tuple[int, int, int, int], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "kinetic_energy_MeV": 0.0}
    )
    generation_counts: Counter[int | str] = Counter()
    projectile_counts: Counter[tuple[int, int]] = Counter()
    target_counts: Counter[int] = Counter()
    unresolved_examples: list[dict[str, int]] = []
    interaction_count = product_count = unsupported_products = 0

    for path in paths:
        records = cinel02.read_raw(path)
        interaction_count += len(records)
        product_count += sum(len(products) for _, products in records)
        events: dict[tuple[int, int, int], list[tuple[dict, list[dict]]]] = defaultdict(list)
        for record, products in records:
            key = (int(record["run_id"]), int(record["thread_id"]), int(record["event_id"]))
            events[key].append((record, products))
        for event_key, members in events.items():
            parents: dict[int, int] = {}
            for record, _ in members:
                track = int(record["track_id"])
                parent = int(record["parent_track_id"])
                if track in parents and parents[track] != parent:
                    raise ValueError(f"inconsistent parent for {event_key} track {track}")
                parents[track] = parent
            memo = {1: 0}
            for record, products in members:
                track = int(record["track_id"])
                generation = track_generation(track, parents, memo)
                if generation is None:
                    generation_counts["unresolved"] += 1
                    if len(unresolved_examples) < 20:
                        unresolved_examples.append({
                            "run_id": event_key[0], "thread_id": event_key[1],
                            "event_id": event_key[2], "track_id": track,
                            "parent_track_id": int(record["parent_track_id"]),
                        })
                    continue
                generation_counts[generation] += 1
                parent_z = int(record["projectile_z"])
                parent_a = int(record["projectile_a"])
                projectile_counts[(parent_z, parent_a)] += 1
                target_z = int(record["target_z"])
                target_counts[target_z] += 1
                bin_index = energy_bin(float(record["collision_energy_MeV_per_u"]))
                channel = CHANNELS.get(parent_z)
                if channel is None:
                    continue
                item = incident[(generation, target_z, bin_index, parent_z)]
                item["count"] += 1
                item["kinetic_energy_MeV"] += float(record["collision_energy_MeV"])

                for product in products:
                    if int(product["role"]) == cinel02.PRODUCT_UNSUPPORTED_BUT_RECORDED:
                        unsupported_products += 1
                    if int(product["role"]) != cinel02.PRODUCT_DIRECT_SECONDARY or int(product["z"]) != 4:
                        continue
                    isotope = int(product["a"])
                    if isotope not in (6, 7, 9, 10):
                        continue
                    item = ledger[(generation, target_z, bin_index, channel, isotope)]
                    item["count"] += 1
                    item["kinetic_energy_MeV"] += float(product["kinetic_energy_MeV"])

    args.output_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    generations = sorted({key[0] for key in ledger} | {key[0] for key in incident})
    for generation in generations:
        for target_z in (1, 8):
            for bin_index in range(len(ENERGY_EDGES) - 1):
                for parent_z, channel in CHANNELS.items():
                    inc = incident[(generation, target_z, bin_index, parent_z)]
                    for isotope in (6, 7, 9, 10):
                        value = ledger[(generation, target_z, bin_index, channel, isotope)]
                        rows.append({
                        "generation": generation,
                        "target": "H" if target_z == 1 else "O",
                        "energy_bin_low_MeV_per_u": ENERGY_EDGES[bin_index],
                        "energy_bin_high_MeV_per_u": ENERGY_EDGES[bin_index + 1],
                            "channel": channel,
                            "be_isotope_a": isotope,
                        "be_incident_count": int(inc["count"]),
                        "be_incident_energy_MeV": inc["kinetic_energy_MeV"],
                        "be_birth_count": int(value["count"]),
                            "be_birth_energy_MeV": value["kinetic_energy_MeV"],
                            "be_births_per_reaction": (value["count"] / inc["count"] if inc["count"] else None),
                            "mean_be_birth_energy_MeV_per_u": (value["kinetic_energy_MeV"] / value["count"] / isotope if value["count"] else None),
                            "be_birth_energy_per_reaction_MeV": (value["kinetic_energy_MeV"] / inc["count"] if inc["count"] else None),
                        })
    csv_path = args.output_directory / "topas_be_channel_ledger.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "CINEL02_CASCADE_BE_ISOTOPE_LEDGER_V2",
        "raw_directory": str(args.raw_directory),
        "worker_files": len(paths),
        "interaction_count": interaction_count,
        "product_count": product_count,
        "generation_interaction_counts": {str(key): value for key, value in generation_counts.items()},
        "unresolved_generation_examples": unresolved_examples,
        "target_interaction_counts": {"H": target_counts[1], "O": target_counts[8]},
        "projectile_interaction_counts": {
            f"Z{z}A{a}": count for (z, a), count in sorted(projectile_counts.items())
        },
        "unsupported_products_in_analyzed_channels": unsupported_products,
        "ledger_csv": str(csv_path),
    }
    (args.output_directory / "topas_cascade_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
