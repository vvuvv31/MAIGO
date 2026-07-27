#!/usr/bin/env python3
"""Compile validated multi-projectile TOPAS cascade tables for SYCL transport."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import struct
from collections import defaultdict
from pathlib import Path


MAGIC = b"CCAS001\0"
VERSION = 3
HEADER = struct.Struct("<8sIIIIIIIIQQQQ")
PROJECTILE = struct.Struct("<hhIIII")
XS_SAMPLE = struct.Struct("<ff")
INTERACTION = struct.Struct("<ffII")
PRODUCT = struct.Struct("<ihhffff")
CONDITION_ENERGY_BIN_MEVU = 2.0
CONDITION_DEPTH_BIN_MM = 10.0


def read_unit_direction(
    row: dict[str, str], columns: tuple[str, str, str], label: str
) -> tuple[float, float, float]:
    direction = tuple(float(row[column]) for column in columns)
    norm_squared = sum(component * component for component in direction)
    if (not all(math.isfinite(component) for component in direction)
            or abs(norm_squared - 1.0) > 2.0e-3):
        raise SystemExit(f"Invalid unit direction for {label}: {direction}")
    inverse_norm = 1.0 / math.sqrt(norm_squared)
    return tuple(component * inverse_norm for component in direction)


def direction_in_parent_frame(
    direction: tuple[float, float, float],
    parent: tuple[float, float, float],
) -> tuple[float, float, float]:
    wx, wy, wz = parent
    reference = (1.0, 0.0, 0.0) if abs(wx) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(reference[index] * parent[index] for index in range(3))
    u_raw = tuple(reference[index] - projection * parent[index] for index in range(3))
    u_norm = math.sqrt(sum(component * component for component in u_raw))
    u = tuple(component / u_norm for component in u_raw)
    v = (
        wy * u[2] - wz * u[1],
        wz * u[0] - wx * u[2],
        wx * u[1] - wy * u[0],
    )
    local = (
        sum(direction[index] * u[index] for index in range(3)),
        sum(direction[index] * v[index] for index in range(3)),
        sum(direction[index] * parent[index] for index in range(3)),
    )
    local_norm = math.sqrt(sum(component * component for component in local))
    return tuple(component / local_norm for component in local)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--xs-energy-quantum-mevu", type=float, default=0.25)
    args = parser.parse_args()
    if args.xs_energy_quantum_mevu <= 0.0:
        raise SystemExit("--xs-energy-quantum-mevu must be positive")

    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    interactions = read_gzip_csv(args.interactions)
    products = read_gzip_csv(args.products)
    if sha256(args.interactions) != source["outputs"]["interactions"]["sha256"]:
        raise SystemExit("Interaction table SHA-256 mismatch")
    if sha256(args.products) != source["outputs"]["products"]["sha256"]:
        raise SystemExit("Product table SHA-256 mismatch")

    products_by_interaction: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in products:
        products_by_interaction[int(row["interaction_id"])].append(row)
    interactions_by_species: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    xs_by_species_energy: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    quantum = args.xs_energy_quantum_mevu
    def add_xs(z: int, a: int, energy_mevu: float, value: float) -> None:
        if z <= 0 or a <= 0 or energy_mevu < 0.0 or value <= 0.0:
            return
        energy_key = int(round(energy_mevu / quantum))
        xs_by_species_energy[(z, a)][energy_key].append(value)

    expected_offset = 0
    for row in interactions:
        interaction_id = int(row["interaction_id"])
        count = int(row["product_count"])
        offset = int(row["product_offset_zero_based"])
        if offset != expected_offset or len(products_by_interaction[interaction_id]) != count:
            raise SystemExit(f"Product range mismatch for interaction {interaction_id}")
        expected_offset += count
        key = (int(row["projectile_Z"]), int(row["projectile_A"]))
        interactions_by_species[key].append(row)
        add_xs(key[0], key[1], float(row["incident_energy_MeV_per_u"]),
               float(row["macro_inelastic_per_mm"]))
    if expected_offset != len(products):
        raise SystemExit("Interaction product ranges do not cover the product table")
    for row in products:
        if row["kinetic_energy_MeV_per_u"]:
            add_xs(int(row["Z"]), int(row["A"]),
                   float(row["kinetic_energy_MeV_per_u"]),
                   float(row["macro_inelastic_per_mm_at_birth"]))

    binary_projectiles: list[tuple[int, int, int, int, int, int]] = []
    binary_xs: list[tuple[float, float]] = []
    binary_interactions: list[tuple[float, float, int, int]] = []
    binary_products: list[tuple[int, int, int, float, float, float, float]] = []
    species_metadata: dict[str, object] = {}
    skipped_zero_cross_section: dict[str, object] = {}
    for (z, a), species_interactions in sorted(interactions_by_species.items()):
        if not xs_by_species_energy[(z, a)]:
            skipped_zero_cross_section[f"Z{z}A{a}"] = {
                "interactions": len(species_interactions),
                "reason": "TOPAS returned no positive macroscopic inelastic cross section",
            }
            continue
        xs_offset = len(binary_xs)
        for energy_key, values in sorted(xs_by_species_energy[(z, a)].items()):
            binary_xs.append((energy_key * quantum, statistics.median(values)))
        interaction_offset = len(binary_interactions)
        def condition_key(item: dict[str, str]) -> tuple[int, int, float, float]:
            energy = float(item["incident_energy_MeV_per_u"])
            depth = float(item["depth_mm"])
            return (
                max(0, int(energy / CONDITION_ENERGY_BIN_MEVU)),
                max(0, int(depth / CONDITION_DEPTH_BIN_MM)),
                energy,
                depth,
            )

        for row in sorted(species_interactions, key=condition_key):
            interaction_id = int(row["interaction_id"])
            incident_direction = read_unit_direction(
                row, ("direction_x", "direction_y", "direction_z"),
                f"interaction {interaction_id} incident particle",
            )
            product_offset = len(binary_products)
            members = products_by_interaction[interaction_id]
            for product in members:
                energy = float(product["kinetic_energy_MeV"])
                if energy < 0.0 or not math.isfinite(energy):
                    raise SystemExit(f"Invalid product in interaction {interaction_id}")
                global_direction = read_unit_direction(
                    product, ("direction_x", "direction_y", "direction_z"),
                    f"interaction {interaction_id} product",
                )
                local_direction = direction_in_parent_frame(
                    global_direction, incident_direction)
                binary_products.append((int(product["pdg_id"]), int(product["Z"]),
                                        int(product["A"]), energy, *local_direction))
            binary_interactions.append((
                float(row["incident_energy_MeV_per_u"]),
                float(row["depth_mm"]),
                product_offset,
                len(members),
            ))
        xs_count = len(binary_xs) - xs_offset
        interaction_count = len(binary_interactions) - interaction_offset
        if xs_count == 0 or interaction_count == 0:
            raise SystemExit(f"Species Z{z}A{a} has no usable cross sections or interactions")
        binary_projectiles.append((z, a, xs_offset, xs_count,
                                   interaction_offset, interaction_count))
        species_metadata[f"Z{z}A{a}"] = {
            "cross_section_samples": xs_count,
            "interactions": interaction_count,
            "minimum_energy_MeV_per_u": min(float(r["incident_energy_MeV_per_u"])
                                             for r in species_interactions),
            "maximum_energy_MeV_per_u": max(float(r["incident_energy_MeV_per_u"])
                                             for r in species_interactions),
        }

    expected_size = (HEADER.size + len(binary_projectiles) * PROJECTILE.size +
                     len(binary_xs) * XS_SAMPLE.size +
                     len(binary_interactions) * INTERACTION.size +
                     len(binary_products) * PRODUCT.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, HEADER.size, PROJECTILE.size,
                                 XS_SAMPLE.size, INTERACTION.size, PRODUCT.size,
                                 len(binary_projectiles), 0, len(binary_xs),
                                 len(binary_interactions), len(binary_products), expected_size))
        for row in binary_projectiles:
            stream.write(PROJECTILE.pack(*row))
        for row in binary_xs:
            stream.write(XS_SAMPLE.pack(*row))
        for row in binary_interactions:
            stream.write(INTERACTION.pack(*row))
        for row in binary_products:
            stream.write(PRODUCT.pack(*row))
    if args.output.stat().st_size != expected_size:
        raise SystemExit("Compiled cascade binary size mismatch")

    compiled = {
        "format": "charged-fragment cascade package", "version": VERSION,
        "final_state_conditioning": [
            "projectile_Z",
            "projectile_A",
            "incident_energy_MeV_per_u",
            "reference_depth_mm",
        ],
        "condition_bins": {
            "incident_energy_MeV_per_u": CONDITION_ENERGY_BIN_MEVU,
            "reference_depth_mm": CONDITION_DEPTH_BIN_MM,
            "interaction_order": "projectile × energy_bin × depth_bin × exact_energy",
        },
        "direction_coordinates": "projectile-local orthonormal frame",
        "direction_components": ["local_x", "local_y", "along_projectile"],
        "source_metadata": args.metadata.as_posix(),
        "source_metadata_sha256": sha256(args.metadata),
        "records": {"projectiles": len(binary_projectiles),
                    "cross_section_samples": len(binary_xs),
                    "interactions": len(binary_interactions), "products": len(binary_products)},
        "species": species_metadata,
        "skipped_zero_cross_section_species": skipped_zero_cross_section,
        "output": {"path": args.output.as_posix(), "bytes": expected_size,
                   "sha256": sha256(args.output)},
    }
    with args.output_metadata.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(compiled, indent=2) + "\n")
    print(f"Compiled {len(binary_projectiles)} projectile species, "
          f"{len(binary_interactions)} interactions, and {len(binary_products)} products")


if __name__ == "__main__":
    main()
