#!/usr/bin/env python3
"""Validate a TOPAS charged-fragment cascade n-tuple and emit compact tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_header(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    histories = re.search(r"Number of Original Histories:\s+(\d+)", text)
    entries = re.search(r"Number of Scored Entries:\s+(\d+)", text)
    if histories is None or entries is None:
        raise ValueError(f"Cannot parse history/entry counts from {path}")
    return {"histories": int(histories.group(1)), "entries": int(entries.group(1))}


def parse_row(line: str, path: Path, line_number: int) -> dict[str, object]:
    values = line.split()
    if len(values) != 29:
        raise ValueError(f"Expected 29 columns at {path}:{line_number}, got {len(values)}")
    integer_indices = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 26, 27, 28)
    float_indices = (12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24)
    for index in integer_indices:
        int(values[index])
    for index in float_indices:
        if not math.isfinite(float(values[index])):
            raise ValueError(f"Non-finite value at {path}:{line_number}")
    return {
        "kind": values[0], "run": int(values[1]), "thread": int(values[2]),
        "event": int(values[3]), "interaction_id": int(values[4]),
        "interaction_track": int(values[5]), "track": int(values[6]),
        "parent": int(values[7]), "pdg": int(values[8]), "name": values[9],
        "z": int(values[10]), "a": int(values[11]), "charge": float(values[12]),
        "energy": float(values[13]), "incident_energy": float(values[14]),
        "macro_xs": float(values[15]), "projectile_z": int(values[16]),
        "projectile_a": int(values[17]), "x": float(values[18]),
        "y": float(values[19]), "z_mm": float(values[20]), "dx": float(values[21]),
        "dy": float(values[22]), "dz": float(values[23]), "weight": float(values[24]),
        "process": values[25], "process_type": int(values[26]),
        "process_subtype": int(values[27]), "model": int(values[28]),
    }


def write_csv_gz(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(f"{value:.12g}" if isinstance(value, float) else value for value in row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("smoke", "development"), required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/topas/output"))
    parser.add_argument("--interactions-output", type=Path, required=True)
    parser.add_argument("--products-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--phantom-half-length-mm", type=float, default=200.0)
    args = parser.parse_args()

    stem = args.input_dir / f"cascade_{args.case}_reactions"
    phsp = stem.with_suffix(".phsp")
    header_path = stem.with_suffix(".header")
    log_path = args.input_dir / f"cascade-{args.case}_topas.log"
    for path in (phsp, header_path, log_path):
        if not path.exists():
            raise SystemExit(f"Required TOPAS output not found: {path}")
    counts = parse_header(header_path)
    if counts["histories"] != args.histories:
        raise SystemExit(f"Header histories {counts['histories']} != {args.histories}")

    interactions: dict[tuple[int, int, int, int], dict[str, object]] = {}
    products: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    parsed = 0
    with phsp.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = parse_row(line, phsp, line_number)
            key = (int(row["run"]), int(row["thread"]), int(row["event"]),
                   int(row["interaction_id"]))
            if row["kind"] == "interaction":
                if key in interactions:
                    raise SystemExit(f"Duplicate interaction key {key}")
                if row["track"] != row["interaction_track"]:
                    raise SystemExit(f"Interaction track mismatch for {key}")
                interactions[key] = row
            elif row["kind"] == "product":
                if row["parent"] != row["interaction_track"]:
                    raise SystemExit(f"Product parent mismatch for {key}")
                products[key].append(row)
            else:
                raise SystemExit(f"Unknown record kind {row['kind']}")
            parsed += 1
    if parsed != counts["entries"]:
        raise SystemExit(f"Parsed {parsed} entries, header reports {counts['entries']}")
    orphaned = set(products) - set(interactions)
    if orphaned:
        raise SystemExit(f"Products without interaction: {sorted(orphaned)[:5]}")

    interaction_rows: list[list[object]] = []
    product_rows: list[list[object]] = []
    interaction_species: Counter[str] = Counter()
    product_species: Counter[str] = Counter()
    process_counts: Counter[str] = Counter()
    max_vertex_error = 0.0
    product_offset = 0
    for interaction_id, key in enumerate(sorted(interactions), 1):
        row = interactions[key]
        species = f"Z{row['projectile_z']}A{row['projectile_a']}"
        interaction_species[species] += 1
        process_counts[str(row["process"])] += 1
        members = products[key]
        interaction_rows.append([
            interaction_id, key[0], key[1], key[2], key[3], row["interaction_track"],
            row["pdg"], row["name"], row["projectile_z"],
            row["projectile_a"], row["incident_energy"],
            float(row["incident_energy"]) / int(row["projectile_a"]), row["macro_xs"],
            float(row["z_mm"]) + args.phantom_half_length_mm, row["dx"], row["dy"],
            row["dz"], row["process"], row["process_type"], row["process_subtype"],
            len(members), product_offset,
        ])
        for product_index, product in enumerate(members, 1):
            vertex_error = max(abs(float(product[c]) - float(row[c])) for c in ("x", "y", "z_mm"))
            max_vertex_error = max(max_vertex_error, vertex_error)
            if vertex_error > 2.0e-4:
                raise SystemExit(f"Interaction/product vertex mismatch for {key}")
            if abs(float(product["incident_energy"]) - float(row["incident_energy"])) > 2.0e-3:
                raise SystemExit(f"Interaction/product incident-energy mismatch for {key}")
            if (product["projectile_z"], product["projectile_a"]) != (
                row["projectile_z"], row["projectile_a"]
            ):
                raise SystemExit(f"Interaction/product projectile mismatch for {key}")
            product_species[f"Z{product['z']}A{product['a']}"] += 1
            product_rows.append([
                interaction_id, product_index, product["track"], product["pdg"],
                product["name"], product["z"], product["a"], product["charge"],
                product["energy"],
                float(product["energy"]) / int(product["a"]) if int(product["a"]) > 0 else "",
                product["macro_xs"], product["dx"], product["dy"], product["dz"],
                product["model"],
            ])
        product_offset += len(members)

    write_csv_gz(args.interactions_output, (
        "interaction_id", "run_id", "thread_id", "event_id", "event_interaction_id",
        "interaction_track_id", "projectile_pdg",
        "projectile_name", "projectile_Z", "projectile_A", "incident_energy_MeV",
        "incident_energy_MeV_per_u", "macro_inelastic_per_mm", "depth_mm",
        "direction_x", "direction_y", "direction_z", "process", "process_type",
        "process_subtype", "product_count", "product_offset_zero_based",
    ), interaction_rows)
    write_csv_gz(args.products_output, (
        "interaction_id", "product_index", "track_id", "pdg_id", "particle_name", "Z", "A",
        "charge_e", "kinetic_energy_MeV", "kinetic_energy_MeV_per_u",
        "macro_inelastic_per_mm_at_birth", "direction_x", "direction_y", "direction_z",
        "creator_model_id",
    ), product_rows)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    elapsed = re.search(r"^\s*Total:.*?Real=([0-9.]+)s", log_text, re.MULTILINE)
    metadata = {
        "case": args.case, "histories": args.histories, "entries": parsed,
        "interactions": len(interaction_rows), "products": len(product_rows),
        "interaction_species": dict(interaction_species.most_common()),
        "product_species": dict(product_species.most_common()),
        "process_counts": dict(process_counts.most_common()),
        "maximum_vertex_error_mm": max_vertex_error,
        "inputs": {"phsp_sha256": sha256(phsp), "header_sha256": sha256(header_path),
                   "log_sha256": sha256(log_path)},
        "outputs": {
            "interactions": {"path": args.interactions_output.as_posix(),
                             "sha256": sha256(args.interactions_output)},
            "products": {"path": args.products_output.as_posix(),
                         "sha256": sha256(args.products_output)},
        },
        "elapsed_real_s": float(elapsed.group(1)) if elapsed else None,
        "global_scale_applied": False,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {len(interaction_rows)} interactions and {len(product_rows)} products")
    print(f"Interaction species: {dict(interaction_species.most_common())}")


if __name__ == "__main__":
    main()
