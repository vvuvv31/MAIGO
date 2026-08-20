#!/usr/bin/env python3
"""Per-bin first-inelastic yields from a compiled-source primary reaction CSV.

Used to measure how far a dedicated-E package can be reused off-energy.
Counts real recorded reactions only (does not include compile-time empty-bin fills).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


SPECIES = (
    ("proton", 1, 1),
    ("helium", 2, None),
    ("lithium", 3, None),
    ("beryllium", 4, None),
    ("boron", 5, None),
    ("carbon", 6, None),
    ("nitrogen", 7, None),
    ("oxygen", 8, None),
)


def open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--secondaries", type=Path, required=True)
    parser.add_argument("--source-energy-mevu", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-bin-min-mevu", type=float, default=0.0)
    parser.add_argument("--energy-bin-width-mevu", type=float, default=4.0)
    parser.add_argument("--energy-bin-count", type=int, required=True)
    args = parser.parse_args()

    reactions: list[dict[str, str]] = []
    with open_csv(args.reactions) as stream:
        reactions = list(csv.DictReader(stream))
    products_by_reaction: dict[int, list[dict[str, str]]] = defaultdict(list)
    with open_csv(args.secondaries) as stream:
        for row in csv.DictReader(stream):
            products_by_reaction[int(row["reaction_id"])].append(row)

    width = args.energy_bin_width_mevu
    origin = args.energy_bin_min_mevu
    n_bins = args.energy_bin_count
    bins: list[dict[str, object]] = []
    for index in range(n_bins):
        lo = origin + index * width
        bins.append(
            {
                "bin": index,
                "energy_lo_MeVu": lo,
                "energy_hi_MeVu": lo + width,
                "n_reactions": 0,
                "mean_incident_MeVu": 0.0,
                "multiplicity": {name: 0.0 for name, _, _ in SPECIES},
                "mean_ke_MeV": {name: 0.0 for name, _, _ in SPECIES},
                "counts": {name: 0 for name, _, _ in SPECIES},
                "ke_sum_MeV": {name: 0.0 for name, _, _ in SPECIES},
            }
        )

    incident_sum = [0.0] * n_bins
    skipped_outside = 0
    for reaction in reactions:
        energy_column = "incident_energy_MeV_per_u"
        if energy_column not in reaction:
            energy_column = "incident_c12_energy_MeV_per_u"
        energy = float(reaction[energy_column])
        index = int((energy - origin) / width)
        if index < 0 or index >= n_bins:
            skipped_outside += 1
            continue
        slot = bins[index]
        slot["n_reactions"] += 1
        incident_sum[index] += energy
        rid = int(reaction["reaction_id"])
        for product in products_by_reaction[rid]:
            z = int(product["atomic_number_Z"])
            a = int(product["mass_number_A"])
            ke = float(product["kinetic_energy_MeV"])
            for name, want_z, want_a in SPECIES:
                if z != want_z:
                    continue
                if want_a is not None and a != want_a:
                    continue
                slot["counts"][name] += 1
                slot["ke_sum_MeV"][name] += ke
                break

    for index, slot in enumerate(bins):
        n = int(slot["n_reactions"])
        if n == 0:
            continue
        slot["mean_incident_MeVu"] = incident_sum[index] / n
        for name, _, _ in SPECIES:
            count = int(slot["counts"][name])
            slot["multiplicity"][name] = count / n
            slot["mean_ke_MeV"][name] = (
                float(slot["ke_sum_MeV"][name]) / count if count else 0.0
            )
        del slot["counts"]
        del slot["ke_sum_MeV"]

    occupied = [slot for slot in bins if int(slot["n_reactions"]) > 0]
    payload = {
        "source_energy_MeVu": args.source_energy_mevu,
        "histories_expected": 1000000,
        "reaction_count": len(reactions),
        "skipped_outside_grid": skipped_outside,
        "energy_bins": {
            "minimum_MeV_per_u": origin,
            "width_MeV_per_u": width,
            "count": n_bins,
        },
        "occupied_bin_count": len(occupied),
        "empty_bin_count": n_bins - len(occupied),
        "on_energy_bin": max(
            int((args.source_energy_mevu - 1.0e-9 - origin) / width), 0
        ),
        "bins": bins,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output}: {len(reactions)} reactions, "
        f"{len(occupied)}/{n_bins} bins occupied"
    )


if __name__ == "__main__":
    main()
