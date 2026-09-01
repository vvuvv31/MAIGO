#!/usr/bin/env python3
"""Summarize C12+O16 -> Be10 content in a CINPKG03 package."""

from __future__ import annotations

import argparse
import json
import math
import mmap
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cinel02  # noqa: E402


EDGES = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, math.inf)
BE_ISOTOPES = (6, 7, 9, 10)


def energy_bin(value: float) -> int:
    return next(index for index, upper in enumerate(EDGES[1:]) if value < upper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.package.open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        header = cinel02.PACKAGE_HEADER.unpack_from(data)
        (
            magic, version, header_size, endian, index_size, fixed_size,
            product_size, _flags, cell_count, interaction_count, product_count,
            file_size, _energy_min, _energy_width, _minimum_events, node_count,
            _campaign,
        ) = header
        if magic != cinel02.PACKAGE_MAGIC or version != cinel02.PACKAGE_VERSION:
            raise ValueError("unsupported package")
        if endian != 0x01020304 or file_size != len(data):
            raise ValueError("invalid package header")
        if fixed_size != cinel02.RAW_FIXED_FORMAT.size or product_size != cinel02.PRODUCT_FORMAT.size:
            raise ValueError("package record sizes disagree")

        fixed_start = header_size + cell_count * index_size
        product_start = fixed_start + interaction_count * fixed_size
        selected: list[tuple[int, int]] = []
        reactions = [0] * 8
        parent_energy = [0.0] * 8
        source_reactions = defaultdict(int)
        product_offset = 0
        for interaction_index in range(interaction_count):
            offset = fixed_start + interaction_index * fixed_size
            values = cinel02.RAW_FIXED_FORMAT.unpack_from(data, offset)
            record = dict(zip(cinel02._FIELD_NAMES, values))
            direct_count = int(record["direct_product_count"])
            if (
                int(record["projectile_z"]) == 6
                and int(record["projectile_a"]) == 12
                and int(record["target_z"]) == 8
                and int(record["target_a"]) == 16
                and 0.0 <= float(record["collision_energy_MeV_per_u"]) < 400.0
            ):
                bin_index = energy_bin(float(record["collision_energy_MeV_per_u"]))
                reactions[bin_index] += 1
                parent_energy[bin_index] += float(record["collision_energy_MeV_per_u"])
                source_energy = round(float(record["source_initial_energy_MeV"]), 3)
                source_reactions[(bin_index, source_energy)] += 1
                selected.append((product_offset, direct_count))
            product_offset += direct_count
        if product_offset != product_count:
            raise ValueError("product prefix sum disagrees with package header")

        counts = {isotope: [0] * 8 for isotope in BE_ISOTOPES}
        kinetic = {isotope: [0.0] * 8 for isotope in BE_ISOTOPES}
        source_births = defaultdict(int)
        source_birth_energy = defaultdict(float)
        # Recover the parent bin from each selected interaction in the same pass.
        selected_index = 0
        product_offset = 0
        for interaction_index in range(interaction_count):
            offset = fixed_start + interaction_index * fixed_size
            values = cinel02.RAW_FIXED_FORMAT.unpack_from(data, offset)
            record = dict(zip(cinel02._FIELD_NAMES, values))
            direct_count = int(record["direct_product_count"])
            is_selected = (
                int(record["projectile_z"]) == 6
                and int(record["projectile_a"]) == 12
                and int(record["target_z"]) == 8
                and int(record["target_a"]) == 16
                and 0.0 <= float(record["collision_energy_MeV_per_u"]) < 400.0
            )
            if is_selected:
                bin_index = energy_bin(float(record["collision_energy_MeV_per_u"]))
                for product_index in range(product_offset, product_offset + direct_count):
                    product_values = cinel02.PRODUCT_FORMAT.unpack_from(
                        data, product_start + product_index * product_size
                    )
                    product = dict(zip(cinel02._PRODUCT_NAMES, product_values))
                    if (
                        int(product["role"]) == cinel02.PRODUCT_DIRECT_SECONDARY
                        and int(product["z"]) == 4
                        and int(product["a"]) in BE_ISOTOPES
                    ):
                        isotope = int(product["a"])
                        counts[isotope][bin_index] += 1
                        kinetic[isotope][bin_index] += float(product["kinetic_energy_MeV"])
                        source_energy = round(float(record["source_initial_energy_MeV"]), 3)
                        source_births[(bin_index, source_energy, isotope)] += 1
                        source_birth_energy[(bin_index, source_energy, isotope)] += float(product["kinetic_energy_MeV"])
                selected_index += 1
            product_offset += direct_count

    rows = []
    for isotope in BE_ISOTOPES:
        for index in range(8):
            count = counts[isotope][index]
            energy = kinetic[isotope][index]
            rows.append({
            "be_isotope_a": isotope,
            "energy_bin_low_MeV_per_u": EDGES[index],
            "energy_bin_high_MeV_per_u": EDGES[index + 1],
            "reaction_count": reactions[index],
            "mean_parent_energy_MeV_per_u": parent_energy[index] / reactions[index] if reactions[index] else None,
            "be_birth_count": count,
            "be_birth_kinetic_energy_MeV": energy,
            "be_births_per_reaction": count / reactions[index] if reactions[index] else None,
            "mean_be_birth_energy_MeV_per_u": energy / count / isotope if count else None,
            "be_birth_kinetic_energy_per_reaction_MeV": energy / reactions[index] if reactions[index] else None,
        })
    source_rows = []
    source_keys = sorted(source_reactions)
    for bin_index, source_energy in source_keys:
        reactions_at_source = source_reactions[(bin_index, source_energy)]
        for isotope in BE_ISOTOPES:
            count = source_births[(bin_index, source_energy, isotope)]
            energy = source_birth_energy[(bin_index, source_energy, isotope)]
            source_rows.append({
                "energy_bin_low_MeV_per_u": EDGES[bin_index],
                "energy_bin_high_MeV_per_u": EDGES[bin_index + 1],
                "source_initial_energy_MeV": source_energy,
                "reaction_count": reactions_at_source,
                "be_isotope_a": isotope,
                "be_birth_count": count,
                "be_births_per_reaction": count / reactions_at_source,
                "mean_be_birth_energy_MeV_per_u": energy / count / isotope if count else None,
            })
    result = {
        "schema": "CINEL02_PACKAGE_BE_ISOTOPE_PARENT_ENERGY_CELLS_V2",
        "package": str(args.package),
        "projectile": "C12",
        "target": "O16",
        "package_node_count": node_count,
        "rows": rows,
        "source_campaign_rows": source_rows,
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
