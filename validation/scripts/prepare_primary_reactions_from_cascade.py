#!/usr/bin/env python3
"""Extract primary track-1 C-12 reaction packages from a cascade baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


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


def write_gzip_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-metadata", type=Path, required=True)
    parser.add_argument("--runtime-reference-metadata", type=Path, required=True)
    parser.add_argument("--interactions", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--reactions-output", type=Path, required=True)
    parser.add_argument("--secondaries-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    args = parser.parse_args()

    source_metadata = json.loads(args.cascade_metadata.read_text(encoding="utf-8"))
    runtime_metadata = json.loads(args.runtime_reference_metadata.read_text(encoding="utf-8"))
    verify(args.interactions, source_metadata["outputs"]["interactions"]["sha256"],
           "Cascade interactions")
    verify(args.products, source_metadata["outputs"]["products"]["sha256"],
           "Cascade products")

    source_products: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
    with gzip.open(args.products, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            source_products[int(row["interaction_id"])].append(row)

    selected: list[dict[str, str]] = []
    with gzip.open(args.interactions, "rt", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if (int(row["projectile_Z"]), int(row["projectile_A"])) != (6, 12):
                continue
            if int(row["interaction_track_id"]) != 1 or int(row["event_interaction_id"]) != 0:
                continue
            selected.append(row)
    if not selected:
        raise SystemExit("No primary track-1 C-12 interactions were found")

    reaction_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    for reaction_id, interaction in enumerate(selected, start=1):
        interaction_id = int(interaction["interaction_id"])
        products = source_products[interaction_id]
        if len(products) != int(interaction["product_count"]):
            raise SystemExit(f"Product-count mismatch for source interaction {interaction_id}")
        reaction_rows.append({
            "reaction_id": reaction_id,
            "incident_c12_energy_MeV_per_u": interaction["incident_energy_MeV_per_u"],
            "reaction_depth_mm": interaction["depth_mm"],
            "secondary_count": len(products),
            "secondary_offset_zero_based": len(secondary_rows),
            "incident_direction_x": interaction["direction_x"],
            "incident_direction_y": interaction["direction_y"],
            "incident_direction_z": interaction["direction_z"],
            "source_interaction_id": interaction_id,
            "source_event_id": interaction["event_id"],
        })
        for secondary_index, product in enumerate(products, start=1):
            secondary_rows.append({
                "reaction_id": reaction_id,
                "secondary_index": secondary_index,
                "pdg_id": product["pdg_id"],
                "atomic_number_Z": product["Z"],
                "mass_number_A": product["A"],
                "kinetic_energy_MeV": product["kinetic_energy_MeV"],
                "direction_x": product["direction_x"],
                "direction_y": product["direction_y"],
                "direction_z": product["direction_z"],
                "source_track_id": product["track_id"],
                "particle_name": product["particle_name"],
            })

    reaction_fields = list(reaction_rows[0])
    secondary_fields = list(secondary_rows[0])
    write_gzip_csv(args.reactions_output, reaction_fields, reaction_rows)
    write_gzip_csv(args.secondaries_output, secondary_fields, secondary_rows)
    metadata = {
        "case": "cascade-aligned-primary-c12",
        "histories": int(source_metadata["histories"]),
        "reaction_definition": "first inelastic interaction of source track-1 primary C-12",
        "source_cascade_metadata": args.cascade_metadata.as_posix(),
        "source_cascade_metadata_sha256": sha256(args.cascade_metadata),
        "runtime_reference": {
            "metadata": args.runtime_reference_metadata.as_posix(),
            "topas_version": runtime_metadata["topas_version"],
            "geant4_version": runtime_metadata["topas_log"]["geant4_version"],
        },
        "reaction_count": len(reaction_rows),
        "secondary_count": len(secondary_rows),
        "selection": {
            "projectile_Z": 6,
            "projectile_A": 12,
            "interaction_track_id": 1,
            "event_interaction_id": 0,
        },
        "output_files": {
            "reactions": {
                "path": args.reactions_output.as_posix(),
                "sha256": sha256(args.reactions_output),
                "bytes": args.reactions_output.stat().st_size,
                "rows": len(reaction_rows),
                "compression": "gzip",
            },
            "secondaries": {
                "path": args.secondaries_output.as_posix(),
                "sha256": sha256(args.secondaries_output),
                "bytes": args.secondaries_output.stat().st_size,
                "rows": len(secondary_rows),
                "compression": "gzip",
            },
        },
        "global_scale_applied": False,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_output.open("w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(metadata, indent=2) + "\n")
    print(f"Extracted {len(reaction_rows)} primary C-12 reactions and "
          f"{len(secondary_rows)} correlated products")


if __name__ == "__main__":
    main()
