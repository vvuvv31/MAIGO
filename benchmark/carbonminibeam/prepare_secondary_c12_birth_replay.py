#!/usr/bin/env python3
"""Convert exported C12 queue births to the secondary replay contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REGIONS = {"copper": 1, "water": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("birth_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--birth-region", choices=("copper", "water", "all"),
                        default="water")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    wanted = None if args.birth_region == "all" else REGIONS[args.birth_region]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    read_count = 0
    written = 0
    energy_MeV = 0.0
    with args.birth_csv.open(newline="", encoding="utf-8") as source, \
            args.output.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        required = {
            "replay_particle_id", "source_history", "generation",
            "birth_region", "Z", "A", "kinetic_energy_MeV", "weight",
            "x_mm", "y_mm", "z_mm", "direction_x", "direction_y",
            "direction_z", "rng_stream",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("C12 birth CSV does not satisfy the replay contract")
        writer = csv.writer(target)
        writer.writerow([
            "origin", "run_id", "event_id", "track_id", "parent_id",
            "pdg", "atomic_number", "mass_number", "kinetic_energy_MeV",
            "weight", "x_mm", "y_mm", "z_mm", "direction_x",
            "direction_y", "direction_z", "rng_stream", "generation",
            "birth_region",
        ])
        for row in reader:
            read_count += 1
            if wanted is not None and int(row["birth_region"]) != wanted:
                continue
            if args.max_records is not None and written >= args.max_records:
                break
            replay_id = int(row["replay_particle_id"])
            history = int(row["source_history"])
            energy = float(row["kinetic_energy_MeV"])
            writer.writerow([
                "fragment", 0, history, replay_id + 1, history + 1,
                1000060120, 6, 12, row["kinetic_energy_MeV"], row["weight"],
                row["x_mm"], row["y_mm"], row["z_mm"],
                row["direction_x"], row["direction_y"], row["direction_z"],
                row["rng_stream"], row["generation"], row["birth_region"],
            ])
            written += 1
            energy_MeV += energy

    metadata = {
        "source": str(args.birth_csv),
        "output": str(args.output),
        "birth_region": args.birth_region,
        "source_records_examined": read_count,
        "replay_records": written,
        "replay_kinetic_energy_MeV": energy_MeV,
        "note": "Records retain true birth position/direction; incident-history "
                "normalization is configured independently in the replay YAML.",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
