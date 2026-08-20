#!/usr/bin/env python3
"""Stitch dedicated-E first-inelastic CSVs into one 0–400 MeV/u primary package.

Each 4 MeV/u bin keeps reactions from the dedicated source nearest the bin
center. If that source has no events in the bin, the source with the most
events in that bin is used instead.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


REACTION_FIELDS = (
    "reaction_id",
    "incident_energy_MeV_per_u",
    "reaction_depth_mm",
    "local_deposit_MeV",
    "secondary_count",
    "secondary_offset_zero_based",
    "incident_direction_x",
    "incident_direction_y",
    "incident_direction_z",
    "source_interaction_id",
    "source_event_id",
)
SECONDARY_FIELDS = (
    "reaction_id",
    "secondary_index",
    "pdg_id",
    "atomic_number_Z",
    "mass_number_A",
    "kinetic_energy_MeV",
    "direction_x",
    "direction_y",
    "direction_z",
    "source_track_id",
    "particle_name",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="")


def write_gzip_csv(path: Path, fieldnames: tuple[str, ...], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nearest_source(center: float, sources: tuple[int, ...]) -> int:
    return min(sources, key=lambda energy: (abs(energy - center), energy))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="topas_water_inclxx_1M_stitch7")
    parser.add_argument("--sources-mevu", default="100,150,200,250,300,350,400")
    parser.add_argument("--projectile-z", type=int, required=True)
    parser.add_argument("--projectile-a", type=int, required=True)
    parser.add_argument("--histories-per-source", type=int, default=1000000)
    parser.add_argument("--energy-bin-width-mevu", type=float, default=4.0)
    parser.add_argument("--energy-bin-count", type=int, default=101)
    args = parser.parse_args()
    sources = tuple(int(value) for value in args.sources_mevu.split(",") if value)
    if not sources or args.projectile_z <= 0 or args.projectile_a < args.projectile_z:
        raise SystemExit("sources and projectile Z/A must be valid")

    width = args.energy_bin_width_mevu
    n_bins = args.energy_bin_count
    counts: dict[int, list[int]] = {
        energy: [0] * n_bins for energy in sources
    }
    paths: dict[int, tuple[Path, Path, Path]] = {}
    runtime = None
    for energy in sources:
        run = args.grid_root / f"run_{energy}MeVu_1M"
        tag = f"topas_{energy}MeVu_water_inclxx_1M"
        reactions = run / f"{tag}_primary_reactions.csv.gz"
        secondaries = run / f"{tag}_primary_secondaries.csv.gz"
        metadata = args.grid_root / "packages" / f"{tag}_primary.metadata.json"
        if not reactions.is_file() or not secondaries.is_file():
            raise SystemExit(f"missing CSVs for {energy} MeV/u under {run}")
        paths[energy] = (reactions, secondaries, metadata)
        with open_csv(reactions) as stream:
            for row in csv.DictReader(stream):
                value = float(row["incident_energy_MeV_per_u"])
                index = int((value - 0.0) / width)
                if 0 <= index < n_bins:
                    counts[energy][index] += 1
        if runtime is None and metadata.is_file():
            runtime = json.loads(metadata.read_text(encoding="utf-8")).get(
                "runtime_reference", {}
            )

    assignment: list[int | None] = [None] * n_bins
    assign_log: list[dict[str, object]] = []
    for index in range(n_bins):
        center = (index + 0.5) * width
        preferred = nearest_source(center, sources)
        chosen = preferred if counts[preferred][index] > 0 else None
        if chosen is None:
            ranked = sorted(
                sources,
                key=lambda energy: (-counts[energy][index], abs(energy - center)),
            )
            if counts[ranked[0]][index] > 0:
                chosen = ranked[0]
        assignment[index] = chosen
        assign_log.append(
            {
                "bin": index,
                "energy_lo_MeVu": index * width,
                "energy_hi_MeVu": (index + 1) * width,
                "preferred_source_MeVu": preferred,
                "chosen_source_MeVu": chosen,
                "n_from_chosen": (
                    0 if chosen is None else counts[chosen][index]
                ),
                "fallback": chosen != preferred,
            }
        )

    keep: dict[int, set[int]] = {energy: set() for energy in sources}
    for index, source in enumerate(assignment):
        if source is None:
            continue
        keep[source].add(index)

    reaction_rows: list[dict[str, object]] = []
    secondary_rows: list[dict[str, object]] = []
    next_id = 1
    for energy in sources:
        if not keep[energy]:
            continue
        reactions_path, secondaries_path, _ = paths[energy]
        selected: list[tuple[int, dict[str, str]]] = []
        with open_csv(reactions_path) as stream:
            for row in csv.DictReader(stream):
                value = float(row["incident_energy_MeV_per_u"])
                index = int(value / width)
                if index not in keep[energy]:
                    continue
                selected.append((int(row["reaction_id"]), row))
        wanted_old = {old_id for old_id, _ in selected}
        products_by_old = defaultdict(list)
        with open_csv(secondaries_path) as stream:
            for row in csv.DictReader(stream):
                old_id = int(row["reaction_id"])
                if old_id in wanted_old:
                    products_by_old[old_id].append(row)
        for old_id, row in selected:
            new_id = next_id
            next_id += 1
            products = products_by_old[old_id]
            reaction_rows.append(
                {
                    "reaction_id": new_id,
                    "incident_energy_MeV_per_u": row[
                        "incident_energy_MeV_per_u"
                    ],
                    "reaction_depth_mm": row["reaction_depth_mm"],
                    "local_deposit_MeV": row["local_deposit_MeV"],
                    "secondary_count": len(products),
                    "secondary_offset_zero_based": len(secondary_rows),
                    "incident_direction_x": row["incident_direction_x"],
                    "incident_direction_y": row["incident_direction_y"],
                    "incident_direction_z": row["incident_direction_z"],
                    "source_interaction_id": row["source_interaction_id"],
                    "source_event_id": row["source_event_id"],
                }
            )
            for secondary_index, product in enumerate(products, start=1):
                secondary_rows.append(
                    {
                        "reaction_id": new_id,
                        "secondary_index": secondary_index,
                        "pdg_id": product["pdg_id"],
                        "atomic_number_Z": product["atomic_number_Z"],
                        "mass_number_A": product["mass_number_A"],
                        "kinetic_energy_MeV": product["kinetic_energy_MeV"],
                        "direction_x": product["direction_x"],
                        "direction_y": product["direction_y"],
                        "direction_z": product["direction_z"],
                        "source_track_id": product["source_track_id"],
                        "particle_name": product["particle_name"],
                    }
                )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    reactions_out = out / f"{args.tag}_primary_reactions.csv.gz"
    secondaries_out = out / f"{args.tag}_primary_secondaries.csv.gz"
    write_gzip_csv(reactions_out, REACTION_FIELDS, reaction_rows)
    write_gzip_csv(secondaries_out, SECONDARY_FIELDS, secondary_rows)
    metadata = {
        "case": "stitched-dedicated-primary",
        "histories": args.histories_per_source,
        "reaction_definition": (
            "first inelastic of track-1 source primary, Voronoi-stitched from "
            "dedicated source-energy samples"
        ),
        "runtime_reference": runtime
        or {
            "topas_version": "4.2.p3",
            "geant4_version": "geant4-11-03-patch-02",
        },
        "reaction_count": len(reaction_rows),
        "secondary_count": len(secondary_rows),
        "selection": {
            "projectile_Z": args.projectile_z,
            "projectile_A": args.projectile_a,
            "interaction_track_id": 1,
            "event_interaction_id": 0,
            "stitch": "nearest dedicated source to 4 MeV/u bin center",
        },
        "output_files": {
            "reactions": {
                "path": reactions_out.as_posix(),
                "sha256": sha256(reactions_out),
                "bytes": reactions_out.stat().st_size,
                "rows": len(reaction_rows),
                "compression": "gzip",
            },
            "secondaries": {
                "path": secondaries_out.as_posix(),
                "sha256": sha256(secondaries_out),
                "bytes": secondaries_out.stat().st_size,
                "rows": len(secondary_rows),
                "compression": "gzip",
            },
        },
        "bin_assignment": assign_log,
        "global_scale_applied": False,
    }
    metadata_out = out / f"{args.tag}_primary.metadata.json"
    metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(reaction_rows)} reactions, {len(secondary_rows)} secondaries")
    print(f"Empty bins after stitch: {sum(1 for item in assign_log if item['chosen_source_MeVu'] is None)}")
    print(f"Fallback bins: {sum(1 for item in assign_log if item['fallback'])}")
    print(metadata_out)


if __name__ == "__main__":
    main()
