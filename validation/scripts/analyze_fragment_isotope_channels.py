#!/usr/bin/env python3
"""Audit isotope production by source table, projectile, and reaction generation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FOCUS_ISOTOPES = ((1, 1), (1, 2), (1, 3), (2, 3), (2, 4),
                  (3, 6), (3, 7), (3, 8), (3, 9),
                  (5, 8), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"{label} SHA-256 mismatch: {actual} != {expected}")


def isotope_name(z: int, a: int) -> str:
    names = {(1, 1): "proton", (1, 2): "deuteron", (1, 3): "triton",
             (2, 3): "He3", (2, 4): "alpha"}
    if (z, a) in names:
        return names[(z, a)]
    symbols = {3: "Li", 4: "Be", 5: "B", 6: "C"}
    return f"{symbols.get(z, f'Z{z}A')}{a}"


def dose_category(z: int, a: int) -> str:
    if (z, a) == (1, 1):
        return "proton"
    return {2: "helium", 3: "lithium", 4: "beryllium", 5: "boron",
            6: "secondary_carbon"}.get(z, "other_charged")


def new_accumulator() -> dict[str, float | int]:
    return {"count": 0, "energy_MeV": 0.0, "direction_z_sum": 0.0,
            "energy_direction_z_sum_MeV": 0.0, "forward_count": 0,
            "forward_energy_MeV": 0.0}


def add_product(accumulator: dict[str, float | int], energy: float, direction_z: float) -> None:
    accumulator["count"] += 1
    accumulator["energy_MeV"] += energy
    accumulator["direction_z_sum"] += direction_z
    accumulator["energy_direction_z_sum_MeV"] += energy * direction_z
    if direction_z > 0.0:
        accumulator["forward_count"] += 1
        accumulator["forward_energy_MeV"] += energy


def summarized(accumulator: dict[str, float | int], interactions: int) -> dict[str, float | int | None]:
    count = int(accumulator["count"])
    energy = float(accumulator["energy_MeV"])
    return {
        "product_count": count,
        "products_per_interaction": count / interactions if interactions else None,
        "kinetic_energy_sum_MeV": energy,
        "kinetic_energy_MeV_per_interaction": energy / interactions if interactions else None,
        "mean_kinetic_energy_MeV": energy / count if count else None,
        "mean_direction_z": float(accumulator["direction_z_sum"]) / count if count else None,
        "energy_weighted_mean_direction_z":
            float(accumulator["energy_direction_z_sum_MeV"]) / energy if energy else None,
        "forward_product_fraction": int(accumulator["forward_count"]) / count if count else None,
        "forward_energy_fraction": float(accumulator["forward_energy_MeV"]) / energy if energy else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-metadata", type=Path, required=True)
    parser.add_argument("--primary-products", type=Path, required=True)
    parser.add_argument("--cascade-metadata", type=Path, required=True)
    parser.add_argument("--cascade-interactions", type=Path, required=True)
    parser.add_argument("--cascade-products", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    primary_metadata = json.loads(args.primary_metadata.read_text(encoding="utf-8"))
    cascade_metadata = json.loads(args.cascade_metadata.read_text(encoding="utf-8"))
    verify(args.primary_products,
           primary_metadata["output_files"]["secondaries"]["sha256"], "Primary products")
    verify(args.cascade_interactions,
           cascade_metadata["outputs"]["interactions"]["sha256"], "Cascade interactions")
    verify(args.cascade_products,
           cascade_metadata["outputs"]["products"]["sha256"], "Cascade products")

    interactions: dict[int, dict[str, int | float | tuple[int, int, int]]] = {}
    with gzip.open(args.cascade_interactions, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            interaction_id = int(row["interaction_id"])
            interactions[interaction_id] = {
                "scope": (int(row["run_id"]), int(row["thread_id"]), int(row["event_id"])),
                "track_id": int(row["interaction_track_id"]),
                "z": int(row["projectile_Z"]),
                "a": int(row["projectile_A"]),
                "energy_MeV_per_u": float(row["incident_energy_MeV_per_u"]),
            }

    birth_parent: dict[tuple[tuple[int, int, int], int], int] = {}
    duplicate_births = 0
    with gzip.open(args.cascade_products, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            interaction_id = int(row["interaction_id"])
            if int(row["Z"]) <= 0 or int(row["A"]) <= 0:
                continue
            key = (interactions[interaction_id]["scope"], int(row["track_id"]))
            if key in birth_parent and birth_parent[key] != interaction_id:
                duplicate_births += 1
            else:
                birth_parent[key] = interaction_id

    generation_cache: dict[int, int] = {}
    active: set[int] = set()

    def generation(interaction_id: int) -> int:
        if interaction_id in generation_cache:
            return generation_cache[interaction_id]
        if interaction_id in active:
            raise SystemExit(f"Cycle in cascade lineage at interaction {interaction_id}")
        active.add(interaction_id)
        interaction = interactions[interaction_id]
        parent_id = birth_parent.get((interaction["scope"], interaction["track_id"]))
        value = generation(parent_id) + 1 if parent_id is not None else 0
        active.remove(interaction_id)
        generation_cache[interaction_id] = value
        return value

    interaction_denominators: Counter[tuple[str, int, int]] = Counter()
    unlinked_generation_zero: Counter[tuple[int, int]] = Counter()
    for interaction_id, interaction in interactions.items():
        stage = f"cascade_generation_{generation(interaction_id)}"
        parent = (int(interaction["z"]), int(interaction["a"]))
        interaction_denominators[(stage, *parent)] += 1
        interaction_denominators[(stage, -1, -1)] += 1
        if generation_cache[interaction_id] == 0 and parent != (6, 12):
            unlinked_generation_zero[parent] += 1

    aggregates: defaultdict[tuple[str, int, int, int, int], dict[str, float | int]] = defaultdict(
        new_accumulator
    )
    primary_interactions = int(primary_metadata["reaction_count"])
    interaction_denominators[("primary_reaction_table", 6, 12)] = primary_interactions
    interaction_denominators[("primary_reaction_table", -1, -1)] = primary_interactions

    with gzip.open(args.primary_products, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            z, a = int(row["atomic_number_Z"]), int(row["mass_number_A"])
            if z <= 0 or a <= 0:
                continue
            energy, direction_z = float(row["kinetic_energy_MeV"]), float(row["direction_z"])
            for parent in ((6, 12), (-1, -1)):
                add_product(aggregates[("primary_reaction_table", *parent, z, a)],
                            energy, direction_z)

    with gzip.open(args.cascade_products, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            z, a = int(row["Z"]), int(row["A"])
            if z <= 0 or a <= 0:
                continue
            interaction_id = int(row["interaction_id"])
            interaction = interactions[interaction_id]
            stage = f"cascade_generation_{generation_cache[interaction_id]}"
            parent = (int(interaction["z"]), int(interaction["a"]))
            energy, direction_z = float(row["kinetic_energy_MeV"]), float(row["direction_z"])
            for parent_key in (parent, (-1, -1)):
                add_product(aggregates[(stage, *parent_key, z, a)], energy, direction_z)

    rows: list[dict[str, object]] = []
    for (stage, parent_z, parent_a, product_z, product_a), accumulator in sorted(aggregates.items()):
        denominator = interaction_denominators[(stage, parent_z, parent_a)]
        row = {
            "stage": stage,
            "parent_Z": parent_z,
            "parent_A": parent_a,
            "parent_isotope": "all" if parent_z < 0 else isotope_name(parent_z, parent_a),
            "interaction_count": denominator,
            "product_Z": product_z,
            "product_A": product_a,
            "product_isotope": isotope_name(product_z, product_a),
            "dose_category": dose_category(product_z, product_a),
            **summarized(accumulator, denominator),
        }
        rows.append(row)

    primary_lookup = {(int(row["product_Z"]), int(row["product_A"])): row for row in rows
                      if row["stage"] == "primary_reaction_table" and row["parent_Z"] == 6}
    cascade_zero_lookup = {(int(row["product_Z"]), int(row["product_A"])): row for row in rows
                           if row["stage"] == "cascade_generation_0" and row["parent_Z"] == 6
                           and row["parent_A"] == 12}
    comparison: dict[str, object] = {}
    for isotope in FOCUS_ISOTOPES:
        primary = primary_lookup.get(isotope)
        cascade = cascade_zero_lookup.get(isotope)
        primary_yield = float(primary["products_per_interaction"]) if primary else 0.0
        cascade_yield = float(cascade["products_per_interaction"]) if cascade else 0.0
        primary_energy = float(primary["kinetic_energy_MeV_per_interaction"]) if primary else 0.0
        cascade_energy = float(cascade["kinetic_energy_MeV_per_interaction"]) if cascade else 0.0
        comparison[isotope_name(*isotope)] = {
            "Z": isotope[0], "A": isotope[1],
            "primary_table_products_per_interaction": primary_yield,
            "cascade_generation0_products_per_interaction": cascade_yield,
            "cascade_over_primary_yield_ratio": cascade_yield / primary_yield if primary_yield else None,
            "primary_table_energy_MeV_per_interaction": primary_energy,
            "cascade_generation0_energy_MeV_per_interaction": cascade_energy,
            "cascade_over_primary_energy_ratio": cascade_energy / primary_energy if primary_energy else None,
        }

    focus_parent_energy: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not str(row["stage"]).startswith("cascade_generation_") or row["parent_Z"] < 0:
            continue
        if int(row["product_Z"]) not in (1, 2, 3, 5):
            continue
        focus_parent_energy[dose_category(int(row["product_Z"]), int(row["product_A"]))].append({
            "stage": row["stage"], "parent": row["parent_isotope"],
            "product": row["product_isotope"],
            "energy_MeV_per_interaction": row["kinetic_energy_MeV_per_interaction"],
            "products_per_interaction": row["products_per_interaction"],
        })
    for values_ in focus_parent_energy.values():
        values_.sort(key=lambda item: float(item["energy_MeV_per_interaction"]), reverse=True)
        del values_[20:]

    output = {
        "sources": {
            "primary_metadata": args.primary_metadata.as_posix(),
            "cascade_metadata": args.cascade_metadata.as_posix(),
        },
        "lineage_reconstruction": {
            "interactions": len(interactions),
            "generation_counts": dict(sorted(Counter(generation_cache.values()).items())),
            "maximum_generation": max(generation_cache.values()),
            "duplicate_charged_track_birth_records": duplicate_births,
            "unlinked_generation_zero_non_primary_projectiles": {
                isotope_name(*key): value for key, value in sorted(unlinked_generation_zero.items())
            },
        },
        "primary_table_vs_cascade_generation0_c12": comparison,
        "top_cascade_focus_channel_energy": dict(focus_parent_energy),
        "global_scale_applied": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(output, indent=2) + "\n")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
