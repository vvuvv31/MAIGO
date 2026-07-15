#!/usr/bin/env python3
"""Validate and standardize CarbonNeutralNtuple phase-space output."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


COLUMNS = (
    "record_kind", "run_id", "thread_id", "event_id", "interaction_id",
    "interaction_track_id", "track_id", "parent_id", "projectile_pdg_id",
    "pdg_id", "particle_name", "atomic_number_Z", "mass_number_A",
    "charge_e", "incident_energy_MeV", "kinetic_energy_MeV",
    "local_deposit_MeV", "macroscopic_total_per_mm", "vertex_x_mm",
    "vertex_y_mm", "vertex_z_mm", "incident_direction_x",
    "incident_direction_y", "incident_direction_z", "direction_x",
    "direction_y", "direction_z", "process_name", "process_type",
    "process_subtype", "creator_model_id",
)
INTEGER_COLUMNS = {
    "run_id", "thread_id", "event_id", "interaction_id",
    "interaction_track_id", "track_id", "parent_id", "projectile_pdg_id",
    "pdg_id", "atomic_number_Z", "mass_number_A", "process_type",
    "process_subtype", "creator_model_id",
}
FLOAT_COLUMNS = set(COLUMNS) - INTEGER_COLUMNS - {
    "record_kind", "particle_name", "process_name"
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_histories(path: Path) -> tuple[int, int]:
    histories = entries = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Number of Original Histories:"):
            histories = int(line.split(":", 1)[1])
        elif line.startswith("Number of Scored Entries:"):
            entries = int(line.split(":", 1)[1])
    if histories is None or entries is None:
        raise SystemExit(f"Missing history/entry counts in {path}")
    return histories, entries


def read_rows(path: Path) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            values = line.split()
            if not values:
                continue
            if len(values) != len(COLUMNS):
                raise SystemExit(
                    f"{path}:{line_number}: expected {len(COLUMNS)} fields, "
                    f"found {len(values)}"
                )
            row: dict[str, object] = dict(zip(COLUMNS, values))
            for name in INTEGER_COLUMNS:
                row[name] = int(row[name])
            for name in FLOAT_COLUMNS:
                row[name] = float(row[name])
            result.append(row)
    return result


def direction(row: dict[str, object], prefix: str) -> tuple[float, float, float]:
    names = (
        f"{prefix}direction_x",
        f"{prefix}direction_y",
        f"{prefix}direction_z",
    )
    values = tuple(float(row[name]) for name in names)
    norm = math.sqrt(sum(value * value for value in values))
    if not all(math.isfinite(value) for value in values) or abs(norm - 1.0) > 2e-3:
        raise SystemExit(f"Invalid {prefix}direction: {values}")
    return values


def write_gzip_csv(path: Path, fieldnames: tuple[str, ...], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            import io
            text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(fieldnames)
            writer.writerows(rows)
            text.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--phsp", type=Path, required=True)
    parser.add_argument("--interactions-output", type=Path, required=True)
    parser.add_argument("--products-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()

    histories, declared_entries = read_histories(args.header)
    rows = read_rows(args.phsp)
    if len(rows) != declared_entries:
        raise SystemExit(
            f"Header declares {declared_entries} entries, found {len(rows)}"
        )

    interactions: dict[tuple[int, int, int, int], dict[str, object]] = {}
    products: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    process_counts: Counter[str] = Counter()
    projectile_counts: Counter[str] = Counter()
    product_species: Counter[str] = Counter()
    for row in rows:
        key = (
            int(row["run_id"]), int(row["thread_id"]),
            int(row["event_id"]), int(row["interaction_id"]),
        )
        if row["record_kind"] == "interaction":
            if key in interactions:
                raise SystemExit(f"Duplicate interaction key: {key}")
            if int(row["projectile_pdg_id"]) not in (22, 2112):
                raise SystemExit(f"Unsupported neutral projectile: {key}")
            if float(row["incident_energy_MeV"]) <= 0.0:
                raise SystemExit(f"Non-positive incident energy: {key}")
            if float(row["kinetic_energy_MeV"]) < 0.0:
                raise SystemExit(f"Negative continuation energy: {key}")
            if float(row["local_deposit_MeV"]) < 0.0:
                raise SystemExit(f"Negative local deposit: {key}")
            if float(row["macroscopic_total_per_mm"]) <= 0.0:
                raise SystemExit(f"Non-positive total cross section: {key}")
            direction(row, "incident_")
            direction(row, "")
            interactions[key] = row
            process_counts[str(row["process_name"])] += 1
            projectile_counts[str(row["projectile_pdg_id"])] += 1
        elif row["record_kind"] == "product":
            direction(row, "incident_")
            direction(row, "")
            products[key].append(row)
            product_species[
                f"{row['pdg_id']}:{row['particle_name']}"
            ] += 1
        else:
            raise SystemExit(f"Unknown record kind: {row['record_kind']}")

    orphaned = set(products) - set(interactions)
    if orphaned:
        raise SystemExit(f"Orphan product keys: {len(orphaned)}")

    interaction_fields = (
        "interaction_index", "run_id", "thread_id", "event_id",
        "interaction_sequence_id", "interaction_track_id", "projectile_pdg_id",
        "incident_energy_MeV", "continuation_energy_MeV",
        "local_deposit_MeV", "macroscopic_total_per_mm",
        "incident_direction_x", "incident_direction_y", "incident_direction_z",
        "continuation_direction_x", "continuation_direction_y",
        "continuation_direction_z", "process_name", "process_type",
        "process_subtype", "product_count", "product_offset_zero_based",
    )
    product_fields = (
        "interaction_index", "product_index", "pdg_id", "particle_name",
        "Z", "A", "charge_e", "kinetic_energy_MeV",
        "direction_x", "direction_y", "direction_z", "creator_model_id",
    )
    interaction_rows: list[list[object]] = []
    product_rows: list[list[object]] = []
    product_offset = 0
    sorted_interactions = sorted(interactions.items())
    for interaction_index, (key, row) in enumerate(sorted_interactions):
        members = products.get(key, [])
        vertex = tuple(float(row[f"vertex_{axis}_mm"]) for axis in "xyz")
        for product_index, product in enumerate(members, 1):
            product_vertex = tuple(
                float(product[f"vertex_{axis}_mm"]) for axis in "xyz"
            )
            if max(abs(a - b) for a, b in zip(vertex, product_vertex)) > 1e-3:
                raise SystemExit(f"Product vertex mismatch: {key}")
            product_rows.append([
                interaction_index, product_index, product["pdg_id"],
                product["particle_name"], product["atomic_number_Z"],
                product["mass_number_A"], product["charge_e"],
                product["kinetic_energy_MeV"], product["direction_x"],
                product["direction_y"], product["direction_z"],
                product["creator_model_id"],
            ])
        interaction_rows.append([
            interaction_index, *key[:3], key[3], row["interaction_track_id"],
            row["projectile_pdg_id"], row["incident_energy_MeV"],
            row["kinetic_energy_MeV"], row["local_deposit_MeV"],
            row["macroscopic_total_per_mm"], row["incident_direction_x"],
            row["incident_direction_y"], row["incident_direction_z"],
            row["direction_x"], row["direction_y"], row["direction_z"],
            row["process_name"], row["process_type"], row["process_subtype"],
            len(members), product_offset,
        ])
        product_offset += len(members)

    write_gzip_csv(args.interactions_output, interaction_fields, interaction_rows)
    write_gzip_csv(args.products_output, product_fields, product_rows)
    def rel(path: Path) -> str:
        return path.as_posix()

    metadata = {
        "case": args.case,
        "histories": histories,
        "raw_entries": len(rows),
        "interactions": len(interaction_rows),
        "products": len(product_rows),
        "projectile_counts": dict(projectile_counts),
        "process_counts": dict(process_counts),
        "product_species": dict(product_species.most_common()),
        "all_interaction_cross_sections_positive": True,
        "global_scale_applied": False,
        "files": {
            "header": {"path": rel(args.header), "sha256": sha256(args.header)},
            "phsp": {"path": rel(args.phsp), "sha256": sha256(args.phsp)},
            "interactions": {
                "path": rel(args.interactions_output),
                "sha256": sha256(args.interactions_output),
            },
            "products": {
                "path": rel(args.products_output),
                "sha256": sha256(args.products_output),
            },
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Validated {len(interaction_rows)} neutral interactions and "
        f"{len(product_rows)} products from {histories} histories"
    )


if __name__ == "__main__":
    main()
