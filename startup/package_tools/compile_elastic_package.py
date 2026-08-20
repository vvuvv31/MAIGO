#!/usr/bin/env python3
"""Compile a generator/model-independent TOPAS elastic event package.

ELPKG v1 is deliberately independent from CRPKG/CCAS.  The input tables are
validated CSV files (usually produced by ``prepare_topas_elastic.py``): one
interaction row owns a contiguous range of product rows.  Runtime energy bins
store explicit event offsets; nearest-bin aliasing is opt-in and is recorded
in the sidecar.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import struct
from pathlib import Path


MAGIC = b"ELPKG01\0"
VERSION = 1
# magic, version, header/record sizes, bin count, flags/reserved, counts, size
HEADER = struct.Struct("<8sIIIIIIIQQQ")
BIN = struct.Struct("<ffII")
# Z, A, incident/outgoing MeV/u, outgoing local direction, local deposit,
# product offset/count, continuation disposition, generation.
EVENT = struct.Struct("<hhffffffIIii")
# PDG, Z, A, kinetic MeV, local direction, charge, generation,
# transport disposition.  Charge is retained so a neutral recoil cannot be
# reclassified as charged merely because it has an ion-like Z/A tuple.
PRODUCT = struct.Struct("<ihhfffffii")

DISPOSITION_CODES = {
    "transport": 1,
    "queue": 2,
    "local_deposit": 3,
    "discard": 4,
    "primary_continuation": 5,
    "recoil": 6,
}
# Elastic is a scattering event: the incident projectile remains transportable.
# A kill disposition is intentionally not part of ELPKG v1.
CONTINUATION_CODES = {"continue": 1, "primary_continuation": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value: str, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise SystemExit(f"Non-finite {label}: {value}")
    return number


def unit(row: dict[str, str], prefix: str, label: str) -> tuple[float, float, float]:
    values = tuple(finite(row[f"{prefix}{axis}"], f"{label} direction") for axis in "xyz")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or abs(norm - 1.0) > 2.0e-3:
        raise SystemExit(f"Invalid unit direction for {label}: {values}")
    return tuple(value / norm for value in values)


def relative_direction(
    direction: tuple[float, float, float], parent: tuple[float, float, float]
) -> tuple[float, float, float]:
    wx, wy, wz = parent
    reference = (1.0, 0.0, 0.0) if abs(wx) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(reference[i] * parent[i] for i in range(3))
    u_raw = tuple(reference[i] - projection * parent[i] for i in range(3))
    u_norm = math.sqrt(sum(value * value for value in u_raw))
    u = tuple(value / u_norm for value in u_raw)
    v = (wy * u[2] - wz * u[1], wz * u[0] - wx * u[2], wx * u[1] - wy * u[0])
    local = (
        sum(direction[i] * u[i] for i in range(3)),
        sum(direction[i] * v[i] for i in range(3)),
        sum(direction[i] * parent[i] for i in range(3)),
    )
    norm = math.sqrt(sum(value * value for value in local))
    return tuple(value / norm for value in local)


def disposition_code(value: str, label: str) -> int:
    try:
        return DISPOSITION_CODES[value]
    except KeyError as error:
        raise SystemExit(f"Unknown {label} disposition {value!r}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--projectile-z", type=int)
    parser.add_argument("--projectile-a", type=int)
    parser.add_argument("--material")
    parser.add_argument("--physics-model")
    parser.add_argument("--energy-bin-min-mevu", type=float, default=0.0)
    parser.add_argument("--energy-bin-width-mevu", type=float, default=1.0)
    parser.add_argument("--energy-bin-count", type=int, default=201)
    parser.add_argument("--fill-empty", choices=("none", "nearest"), default="none")
    parser.add_argument("--energy-tolerance-mev", type=float, default=1.0e-2)
    parser.add_argument("--energy-relative-tolerance", type=float, default=2.0e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.energy_bin_width_mevu <= 0.0 or args.energy_bin_count <= 0:
        raise SystemExit("Energy bin width/count must be positive")
    if args.energy_tolerance_mev < 0.0 or args.energy_relative_tolerance < 0.0:
        raise SystemExit("Energy tolerances must be non-negative")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata_projectile = metadata.get("projectile", {})
    if not isinstance(metadata_projectile, dict):
        metadata_projectile = {}
    projectile_z = args.projectile_z if args.projectile_z is not None else metadata_projectile.get("Z")
    projectile_a = args.projectile_a if args.projectile_a is not None else metadata_projectile.get("A")
    if (not isinstance(projectile_z, int) or not isinstance(projectile_a, int) or
            projectile_z <= 0 or projectile_a < projectile_z or
            projectile_z > 32767 or projectile_a > 32767):
        raise SystemExit("ELPKG requires a valid projectile Z/A")
    material = args.material or metadata.get("material")
    physics_model = args.physics_model or metadata.get("physics_model")
    provenance = metadata.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    if not isinstance(material, str) or not material:
        raise SystemExit("ELPKG sidecar requires a material")
    if not isinstance(physics_model, str) or not physics_model:
        raise SystemExit("ELPKG sidecar requires a physics model")

    interactions = read_gzip_csv(args.interactions)
    products = read_gzip_csv(args.products)
    required_interaction = {
        "interaction_id", "projectile_Z", "projectile_A", "incident_energy_MeV_per_u",
        "outgoing_projectile_energy_MeV_per_u", "outgoing_direction_x",
        "outgoing_direction_y", "outgoing_direction_z", "incident_direction_x",
        "incident_direction_y", "incident_direction_z", "local_deposit_MeV",
        "product_count", "product_offset_zero_based", "continuation_disposition",
    }
    required_product = {
        "interaction_id", "product_index", "pdg_id", "atomic_number_Z", "mass_number_A",
        "charge_e", "kinetic_energy_MeV", "direction_x", "direction_y", "direction_z",
        "generation", "transport_disposition",
    }
    if not interactions:
        raise SystemExit("Elastic interaction table is empty")
    missing = required_interaction - set(interactions[0])
    if missing:
        raise SystemExit(f"Elastic interaction table missing columns: {sorted(missing)}")
    if products:
        missing = required_product - set(products[0])
        if missing:
            raise SystemExit(f"Elastic product table missing columns: {sorted(missing)}")

    products_by_id: dict[int, list[dict[str, str]]] = {}
    for product in products:
        interaction_id = int(product["interaction_id"])
        products_by_id.setdefault(interaction_id, []).append(product)

    bins: list[list[int]] = [[] for _ in range(args.energy_bin_count)]
    parsed_events: list[tuple[dict[str, str], list[dict[str, str]], tuple[float, float, float]]] = []
    expected_product_offset = 0
    for row_index, row in enumerate(interactions):
        interaction_id = int(row["interaction_id"])
        if interaction_id != row_index + 1:
            raise SystemExit("Interaction IDs must be contiguous starting at one")
        z, a = int(row["projectile_Z"]), int(row["projectile_A"])
        if (z, a) != (projectile_z, projectile_a):
            raise SystemExit(f"Wrong projectile identity in interaction {interaction_id}: Z{z}A{a}")
        energy = finite(row["incident_energy_MeV_per_u"], f"interaction {interaction_id} energy")
        outgoing_energy = finite(row["outgoing_projectile_energy_MeV_per_u"], f"interaction {interaction_id} outgoing energy")
        local_deposit = finite(row["local_deposit_MeV"], f"interaction {interaction_id} local deposit")
        if energy < 0.0 or outgoing_energy < 0.0 or local_deposit < 0.0:
            raise SystemExit(f"Negative energy/deposit in interaction {interaction_id}")
        incident = unit(row, "incident_direction_", f"interaction {interaction_id} incident")
        outgoing = unit(row, "outgoing_direction_", f"interaction {interaction_id} outgoing")
        relative_outgoing = relative_direction(outgoing, incident)
        offset, count = int(row["product_offset_zero_based"]), int(row["product_count"])
        members = products_by_id.get(interaction_id, [])
        if offset != expected_product_offset or count != len(members):
            raise SystemExit(f"Non-contiguous product range for interaction {interaction_id}")
        expected_product_offset += count
        product_energy = 0.0
        for expected_index, product in enumerate(members, 1):
            if int(product["product_index"]) != expected_index:
                raise SystemExit(f"Product indices are not contiguous for interaction {interaction_id}")
            pz, pa = int(product["atomic_number_Z"]), int(product["mass_number_A"])
            if pz < 0 or pa < 0 or (pz == 0 and pa != 0) or (pa and pz > pa):
                raise SystemExit(f"Invalid product Z/A in interaction {interaction_id}")
            charge = finite(product["charge_e"], f"interaction {interaction_id} product charge")
            if pz == 0 and pa == 0 and abs(charge) > 1.0e-3:
                raise SystemExit(f"Neutral product has non-zero charge in interaction {interaction_id}")
            if int(product["pdg_id"]) in (22, 2112) and abs(charge) > 1.0e-3:
                raise SystemExit(f"Neutral PDG product has non-zero charge in interaction {interaction_id}")
            kinetic = finite(product["kinetic_energy_MeV"], f"interaction {interaction_id} product energy")
            if kinetic < 0.0:
                raise SystemExit(f"Negative product energy in interaction {interaction_id}")
            unit(product, "direction_", f"interaction {interaction_id} product")
            generation = int(product["generation"])
            if generation < 0:
                raise SystemExit(f"Negative product generation in interaction {interaction_id}")
            disposition_code(product["transport_disposition"], "product")
            product_energy += kinetic
        total_in = energy * projectile_a
        total_out = outgoing_energy * projectile_a + product_energy + local_deposit
        residual = abs(total_in - total_out)
        tolerance = max(args.energy_tolerance_mev, args.energy_relative_tolerance * max(total_in, 1.0))
        if residual > tolerance:
            raise SystemExit(
                f"Energy balance failure in interaction {interaction_id}: "
                f"incident={total_in:g} outgoing={total_out:g} residual={residual:g} tolerance={tolerance:g}"
            )
        continuation = row["continuation_disposition"]
        if continuation not in CONTINUATION_CODES:
            raise SystemExit(f"Unknown continuation disposition {continuation!r}")
        parsed_events.append((row, members, relative_outgoing))
        bin_index = math.floor((energy - args.energy_bin_min_mevu) / args.energy_bin_width_mevu)
        if bin_index < 0 or bin_index >= args.energy_bin_count:
            raise SystemExit(f"Interaction {interaction_id} energy is outside the bin grid")
        bins[bin_index].append(row_index)
    if expected_product_offset != len(products):
        raise SystemExit("Product ranges do not cover the complete product table")

    aliases: list[dict[str, int]] = []
    empty = [index for index, values in enumerate(bins) if not values]
    if empty:
        if args.fill_empty != "nearest":
            raise SystemExit("Empty elastic energy bins require --fill-empty nearest")
        occupied = [index for index, values in enumerate(bins) if values]
        for index in empty:
            source = min(occupied, key=lambda candidate: (abs(candidate - index), 0 if candidate > index else 1, -candidate))
            bins[index] = list(bins[source])
            aliases.append({"empty_bin": index, "copied_from_bin": source})

    binary_bins: list[tuple[float, float, int, int]] = []
    binary_events: list[tuple[int, int, float, float, float, float, float, float, int, int, int, int]] = []
    binary_products: list[tuple[int, int, int, float, float, float, float, int, int]] = []
    for bin_index, members in enumerate(bins):
        event_offset = len(binary_events)
        for source_index in members:
            row, product_rows, relative_outgoing = parsed_events[source_index]
            product_offset = len(binary_products)
            for product in product_rows:
                direction = unit(product, "direction_", f"interaction {row['interaction_id']} product")
                local_direction = relative_direction(direction, unit(row, "incident_direction_", f"interaction {row['interaction_id']} incident"))
                binary_products.append((
                    int(product["pdg_id"]), int(product["atomic_number_Z"]), int(product["mass_number_A"]),
                    finite(product["kinetic_energy_MeV"], "product energy"), *local_direction,
                    finite(product["charge_e"], "product charge"),
                    int(product["generation"]), disposition_code(product["transport_disposition"], "product"),
                ))
            binary_events.append((
                projectile_z, projectile_a, finite(row["incident_energy_MeV_per_u"], "incident energy"),
                finite(row["outgoing_projectile_energy_MeV_per_u"], "outgoing energy"), *relative_outgoing,
                finite(row["local_deposit_MeV"], "local deposit"), product_offset, len(product_rows),
                CONTINUATION_CODES[row["continuation_disposition"]], 0,
            ))
        binary_bins.append((
            args.energy_bin_min_mevu + bin_index * args.energy_bin_width_mevu,
            args.energy_bin_min_mevu + (bin_index + 1) * args.energy_bin_width_mevu,
            event_offset, len(binary_events) - event_offset,
        ))

    expected_size = HEADER.size + len(binary_bins) * BIN.size + len(binary_events) * EVENT.size + len(binary_products) * PRODUCT.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        stream.write(HEADER.pack(MAGIC, VERSION, HEADER.size, BIN.size, EVENT.size, PRODUCT.size, len(binary_bins), 0, len(binary_events), len(binary_products), expected_size))
        for record in binary_bins:
            stream.write(BIN.pack(*record))
        for record in binary_events:
            stream.write(EVENT.pack(*record))
        for record in binary_products:
            stream.write(PRODUCT.pack(*record))
    if args.output.stat().st_size != expected_size:
        raise SystemExit("ELPKG output size does not match header")
    sidecar = {
        "sidecar_schema_version": 2,
        "kind": "elastic",
        "format": "ELPKG elastic event package",
        "format_version": VERSION,
        "byte_order": "little-endian",
        "projectile": {"Z": projectile_z, "A": projectile_a},
        "material": material,
        "physics_model": physics_model,
        "physics": {"model": physics_model, "topas_version": metadata.get("topas_version", provenance.get("topas_version")), "geant4_version": metadata.get("geant4_version", provenance.get("geant4_version"))},
        "energy_range_MeV_per_u": {"minimum": args.energy_bin_min_mevu, "maximum": args.energy_bin_min_mevu + args.energy_bin_width_mevu * args.energy_bin_count},
        "bin_semantics": {"interval": "[minimum, maximum)", "selection": "floor((E-minimum)/width)", "empty_bin_policy": args.fill_empty, "nearest_fill_aliases": aliases, "alias_is_explicit": bool(aliases)},
        "energy_balance": {"tolerance_MeV": args.energy_tolerance_mev, "relative_tolerance": args.energy_relative_tolerance, "definition": "incident kinetic = outgoing projectile + visible products + local deposit"},
        "continuation": {"code": 1, "meaning": "primary continuation survives elastic interaction and remains transportable"},
        "product_dispositions": {str(code): name for name, code in DISPOSITION_CODES.items()},
        "records": {"bins": len(binary_bins), "events": len(binary_events), "products": len(binary_products)},
        "record_sizes_bytes": {"header": HEADER.size, "bin": BIN.size, "event": EVENT.size, "product": PRODUCT.size},
        "record_layout": {
            "header": "<8sIIIIIIIQQQ: magic,version,header_size,bin_size,event_size,product_size,bin_count,flags,event_count,product_count,output_bytes",
            "bin": "<ffII: minimum_MeV_per_u,maximum_MeV_per_u,event_offset,event_count",
            "event": "<hhffffffIIii: projectile_Z,projectile_A,incident_MeV_per_u,outgoing_MeV_per_u,outgoing_local_direction_xyz,local_deposit_MeV,product_offset,product_count,continuation,generation",
            "product": "<ihhfffffii: pdg_id,Z,A,kinetic_MeV,local_direction_xyz,charge_e,generation,transport_disposition",
        },
        "source_metadata": args.metadata.as_posix(), "source_metadata_sha256": sha256(args.metadata),
        "source_interactions_sha256": sha256(args.interactions), "source_products_sha256": sha256(args.products),
        "provenance": provenance,
        "output": {"path": args.output.as_posix(), "bytes": expected_size, "sha256": sha256(args.output)},
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"Compiled ELPKG v1: {len(binary_bins)} bins, {len(binary_events)} events, {len(binary_products)} products")


if __name__ == "__main__":
    main()
