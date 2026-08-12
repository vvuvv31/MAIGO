#!/usr/bin/env python3
"""Compile validated neutron/gamma TOPAS tables for SYCL transport."""

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


MAGIC = b"CNPK001\0"
VERSION = 1
HEADER = struct.Struct("<8sIIIIIIIIQQQQ")
PROJECTILE = struct.Struct("<iIIII")  # pdg, xs_offset, xs_count, ix_offset, ix_count
XS_SAMPLE = struct.Struct("<ff")  # energy_MeV, macroscopic_total_per_mm
INTERACTION = struct.Struct("<ffffffiiII")
# incident_E, continuation_E, local_deposit, cont_dx, cont_dy, cont_dz,
# process_type, process_subtype, product_offset, product_count
PRODUCT = struct.Struct("<ihhffff")  # pdg, Z, A, E, local_dx, local_dy, local_dz


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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


def metadata_sha(source: dict, kind: str) -> str:
    files = source.get("files") or source.get("outputs") or {}
    entry = files.get(kind)
    if not isinstance(entry, dict) or "sha256" not in entry:
        raise SystemExit(f"Metadata is missing files.{kind}.sha256")
    return str(entry["sha256"])


def projectile_label(pdg: int) -> str:
    if pdg == 2112:
        return "neutron"
    if pdg == 22:
        return "gamma"
    return f"pdg{pdg}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--xs-energy-quantum-mev", type=float, default=0.25)
    parser.add_argument(
        "--material",
        default="water",
        help="Material represented by the sampled macroscopic cross sections",
    )
    args = parser.parse_args()
    if args.xs_energy_quantum_mev <= 0.0:
        raise SystemExit("--xs-energy-quantum-mev must be positive")

    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    if (source.get("topas_version") != "4.2.p3" or
            source.get("geant4_version") != "geant4-11-03-patch-02"):
        raise SystemExit(
            "Neutral packages require TOPAS 4.2.p3 / Geant4 11.3.2 provenance"
        )
    interactions = read_gzip_csv(args.interactions)
    products = read_gzip_csv(args.products)
    if sha256(args.interactions) != metadata_sha(source, "interactions"):
        raise SystemExit("Interaction table SHA-256 mismatch")
    if sha256(args.products) != metadata_sha(source, "products"):
        raise SystemExit("Product table SHA-256 mismatch")

    products_by_interaction: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in products:
        products_by_interaction[int(row["interaction_index"])].append(row)

    interactions_by_pdg: dict[int, list[dict[str, str]]] = defaultdict(list)
    xs_by_pdg_energy: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    quantum = args.xs_energy_quantum_mev

    expected_offset = 0
    for row in interactions:
        interaction_index = int(row["interaction_index"])
        count = int(row["product_count"])
        offset = int(row["product_offset_zero_based"])
        if offset != expected_offset or len(products_by_interaction[interaction_index]) != count:
            raise SystemExit(
                f"Product range mismatch for interaction {interaction_index}"
            )
        expected_offset += count
        pdg = int(row["projectile_pdg_id"])
        if pdg not in (22, 2112):
            raise SystemExit(f"Unsupported neutral projectile PDG {pdg}")
        energy = float(row["incident_energy_MeV"])
        xs = float(row["macroscopic_total_per_mm"])
        if energy <= 0.0 or not math.isfinite(energy):
            raise SystemExit(f"Invalid incident energy for interaction {interaction_index}")
        if xs <= 0.0 or not math.isfinite(xs):
            raise SystemExit(f"Invalid total cross section for interaction {interaction_index}")
        continuation = float(row["continuation_energy_MeV"])
        deposit = float(row["local_deposit_MeV"])
        if continuation < 0.0 or deposit < 0.0:
            raise SystemExit(f"Negative continuation/deposit for interaction {interaction_index}")
        interactions_by_pdg[pdg].append(row)
        energy_key = int(round(energy / quantum))
        xs_by_pdg_energy[pdg][energy_key].append(xs)
    if expected_offset != len(products):
        raise SystemExit("Interaction product ranges do not cover the product table")
    # The grouping dictionaries now own all row references. Drop the two large
    # CSV reader lists before constructing compact binary records.
    del interactions
    del products

    binary_projectiles: list[tuple[int, int, int, int, int]] = []
    binary_xs: list[tuple[float, float]] = []
    binary_interactions: list[tuple[float, float, float, float, float, float, int, int, int, int]] = []
    binary_products: list[tuple[int, int, int, float, float, float, float]] = []
    species_metadata: dict[str, object] = {}

    for pdg in sorted(interactions_by_pdg):
        species_interactions = interactions_by_pdg[pdg]
        xs_table = xs_by_pdg_energy[pdg]
        if not xs_table:
            raise SystemExit(f"No positive cross sections for {projectile_label(pdg)}")
        xs_offset = len(binary_xs)
        for energy_key, values in sorted(xs_table.items()):
            binary_xs.append((energy_key * quantum, statistics.median(values)))
        interaction_offset = len(binary_interactions)
        for row in sorted(
            species_interactions, key=lambda item: float(item["incident_energy_MeV"])
        ):
            interaction_index = int(row["interaction_index"])
            incident_direction = read_unit_direction(
                row,
                ("incident_direction_x", "incident_direction_y", "incident_direction_z"),
                f"interaction {interaction_index} incident",
            )
            continuation_global = read_unit_direction(
                row,
                (
                    "continuation_direction_x",
                    "continuation_direction_y",
                    "continuation_direction_z",
                ),
                f"interaction {interaction_index} continuation",
            )
            continuation_local = direction_in_parent_frame(
                continuation_global, incident_direction
            )
            product_offset = len(binary_products)
            # Release verbose CSV dictionaries as soon as their compact binary
            # representation has been built, limiting peak memory on 100k runs.
            members = products_by_interaction.pop(interaction_index, [])
            for product in members:
                energy = float(product["kinetic_energy_MeV"])
                if energy < 0.0 or not math.isfinite(energy):
                    raise SystemExit(f"Invalid product in interaction {interaction_index}")
                product_global = read_unit_direction(
                    product,
                    ("direction_x", "direction_y", "direction_z"),
                    f"interaction {interaction_index} product",
                )
                product_local = direction_in_parent_frame(
                    product_global, incident_direction
                )
                binary_products.append((
                    int(product["pdg_id"]),
                    int(product["Z"]),
                    int(product["A"]),
                    energy,
                    *product_local,
                ))
            binary_interactions.append((
                float(row["incident_energy_MeV"]),
                float(row["continuation_energy_MeV"]),
                float(row["local_deposit_MeV"]),
                *continuation_local,
                int(row["process_type"]),
                int(row["process_subtype"]),
                product_offset,
                len(members),
            ))
        xs_count = len(binary_xs) - xs_offset
        interaction_count = len(binary_interactions) - interaction_offset
        if xs_count == 0 or interaction_count == 0:
            raise SystemExit(
                f"{projectile_label(pdg)} has no usable cross sections or interactions"
            )
        binary_projectiles.append(
            (pdg, xs_offset, xs_count, interaction_offset, interaction_count)
        )
        energies = [float(row["incident_energy_MeV"]) for row in species_interactions]
        species_metadata[projectile_label(pdg)] = {
            "pdg_id": pdg,
            "cross_section_samples": xs_count,
            "interactions": interaction_count,
            "minimum_energy_MeV": min(energies),
            "maximum_energy_MeV": max(energies),
            "process_counts": dict(
                sorted(
                    {
                        name: sum(1 for row in species_interactions
                                  if row["process_name"] == name)
                        for name in {row["process_name"] for row in species_interactions}
                    }.items()
                )
            ),
        }
        species_interactions.clear()

    if products_by_interaction:
        raise SystemExit("Unconsumed neutral product groups remain after compilation")

    expected_size = (
        HEADER.size
        + len(binary_projectiles) * PROJECTILE.size
        + len(binary_xs) * XS_SAMPLE.size
        + len(binary_interactions) * INTERACTION.size
        + len(binary_products) * PRODUCT.size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(HEADER.pack(
            MAGIC, VERSION, HEADER.size, PROJECTILE.size, XS_SAMPLE.size,
            INTERACTION.size, PRODUCT.size, len(binary_projectiles), 0,
            len(binary_xs), len(binary_interactions), len(binary_products),
            expected_size,
        ))
        for row in binary_projectiles:
            stream.write(PROJECTILE.pack(*row))
        for row in binary_xs:
            stream.write(XS_SAMPLE.pack(*row))
        for row in binary_interactions:
            stream.write(INTERACTION.pack(*row))
        for row in binary_products:
            stream.write(PRODUCT.pack(*row))
    if args.output.stat().st_size != expected_size:
        raise SystemExit("Compiled neutral binary size mismatch")

    compiled = {
        "format": "neutron-gamma neutral package",
        "version": VERSION,
        "energy_units": "MeV absolute (not MeV/u)",
        "cross_section":
            f"macroscopic total per mm in {args.material}",
        "direction_coordinates": "incident-projectile local orthonormal frame",
        "direction_components": ["local_x", "local_y", "along_projectile"],
        "xs_energy_quantum_MeV": args.xs_energy_quantum_mev,
        "source_metadata": args.metadata.as_posix(),
        "source_metadata_sha256": sha256(args.metadata),
        "source_runtime": {
            "topas_version": source.get("topas_version"),
            "geant4_version": source.get("geant4_version"),
            "runtime_log_sha256": (source.get("files", {}).get("runtime_log", {})
                                   .get("sha256")),
        },
        "records": {
            "projectiles": len(binary_projectiles),
            "cross_section_samples": len(binary_xs),
            "interactions": len(binary_interactions),
            "products": len(binary_products),
        },
        "species": species_metadata,
        "output": {
            "path": args.output.as_posix(),
            "bytes": expected_size,
            "sha256": sha256(args.output),
        },
        "global_scale_applied": False,
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(
        json.dumps(compiled, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Compiled {len(binary_projectiles)} neutral projectiles, "
        f"{len(binary_interactions)} interactions, and {len(binary_products)} products"
    )


if __name__ == "__main__":
    main()
