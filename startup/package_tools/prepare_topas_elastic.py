#!/usr/bin/env python3
"""Normalize ``CarbonElasticNtuple`` output for the ELPKG compiler.

The extension keeps its historical ``Carbon`` class namespace for TOPAS build
compatibility, but the scorer is parameterized by the requested projectile
Z/A.  Rows are split into interaction and product tables while preserving
neutral products and an explicit primary-continuation field on interactions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


COLUMNS = (
    "record_kind", "run_id", "thread_id", "event_id", "interaction_id",
    "interaction_track_id", "track_id", "parent_id", "projectile_pdg_id",
    "pdg_id", "particle_name", "atomic_number_Z", "mass_number_A", "charge_e",
    "incident_energy_MeV_per_u", "outgoing_projectile_energy_MeV_per_u",
    "kinetic_energy_MeV", "local_deposit_MeV", "macroscopic_elastic_per_mm",
    "vertex_x_mm", "vertex_y_mm", "vertex_z_mm", "incident_direction_x",
    "incident_direction_y", "incident_direction_z", "direction_x", "direction_y",
    "direction_z", "generation", "transport_disposition", "process_name",
    "process_type", "process_subtype", "creator_model_id",
)
INTEGER_COLUMNS = {
    "run_id", "thread_id", "event_id", "interaction_id", "interaction_track_id",
    "track_id", "parent_id", "projectile_pdg_id", "pdg_id", "atomic_number_Z",
    "mass_number_A", "process_type", "process_subtype", "creator_model_id", "generation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_header(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    histories = re.search(r"Number of Original Histories:\s+(\d+)", text)
    entries = re.search(r"Number of Scored Entries:\s+(\d+)", text)
    if histories is None or entries is None:
        raise SystemExit(f"Missing history/entry counts in {path}")
    return int(histories.group(1)), int(entries.group(1))


def direction(row: dict[str, object], prefix: str, label: str) -> None:
    values = tuple(float(row[f"{prefix}direction_{axis}"]) for axis in "xyz")
    norm = math.sqrt(sum(value * value for value in values))
    if not all(math.isfinite(value) for value in values) or norm <= 0.0 or abs(norm - 1.0) > 2.0e-3:
        raise SystemExit(f"Invalid {label} direction: {values}")


def write_csv(path: Path, fields: tuple[str, ...], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--histories", type=int, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("validation/topas/output"))
    parser.add_argument("--stem", default=None)
    parser.add_argument("--projectile-z", type=int, required=True)
    parser.add_argument("--projectile-a", type=int, required=True)
    parser.add_argument("--material", default="G4_WATER")
    parser.add_argument("--physics-model", default="G4HadronElasticPhysicsHP")
    parser.add_argument("--interactions-output", type=Path, required=True)
    parser.add_argument("--products-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--phantom-half-length-mm", type=float, default=200.0)
    parser.add_argument("--runtime-log", type=Path)
    parser.add_argument("--extraction-mode", default="transport-degrading")
    parser.add_argument("--sampling-purpose", default="elastic-package-extraction")
    parser.add_argument("--continuous-em-loss-enabled", choices=("true", "false"), default="true")
    args = parser.parse_args()
    if args.projectile_z <= 0 or args.projectile_a < args.projectile_z:
        raise SystemExit("Projectile Z/A must satisfy Z > 0 and A >= Z")
    stem = args.input_dir / (args.stem or f"elastic_{args.case}")
    phsp, header = stem.with_suffix(".phsp"), stem.with_suffix(".header")
    if not phsp.exists() or not header.exists():
        raise SystemExit(f"Required TOPAS files not found: {phsp}, {header}")
    histories, declared_entries = parse_header(header)
    if histories != args.histories:
        raise SystemExit(f"Header histories {histories} != requested {args.histories}")

    rows: list[dict[str, object]] = []
    with phsp.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != len(COLUMNS):
                raise SystemExit(f"{phsp}:{line_number}: expected {len(COLUMNS)} columns, found {len(values)}")
            row: dict[str, object] = dict(zip(COLUMNS, values))
            for name in INTEGER_COLUMNS:
                row[name] = int(row[name])
            for name in set(COLUMNS) - INTEGER_COLUMNS - {"record_kind", "particle_name", "transport_disposition", "process_name"}:
                row[name] = float(row[name])
            for name in set(COLUMNS) - INTEGER_COLUMNS - {"record_kind", "particle_name", "transport_disposition", "process_name"}:
                if not math.isfinite(float(row[name])):
                    raise SystemExit(f"{phsp}:{line_number}: non-finite {name}")
            if float(row["macroscopic_elastic_per_mm"]) < 0.0:
                raise SystemExit(f"{phsp}:{line_number}: negative macroscopic elastic cross section")
            rows.append(row)
    if len(rows) != declared_entries:
        raise SystemExit(f"Header declares {declared_entries} entries, found {len(rows)}")

    interactions: dict[tuple[int, int, int, int], dict[str, object]] = {}
    products: dict[tuple[int, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (int(row["run_id"]), int(row["thread_id"]), int(row["event_id"]), int(row["interaction_id"]))
        direction(row, "incident_", f"interaction {key} incident")
        direction(row, "", f"interaction {key} outgoing/product")
        if row["record_kind"] == "interaction":
            if key in interactions:
                raise SystemExit(f"Duplicate elastic interaction {key}")
            if (int(row["atomic_number_Z"]), int(row["mass_number_A"])) != (args.projectile_z, args.projectile_a):
                raise SystemExit(f"Wrong projectile identity in {key}")
            if "elastic" not in str(row["process_name"]).lower():
                raise SystemExit(f"Non-elastic process in {key}: {row['process_name']}")
            if str(row["transport_disposition"]) not in {"continue", "primary_continuation"}:
                raise SystemExit(f"Elastic interaction must continue primary in {key}")
            for field in ("incident_energy_MeV_per_u", "outgoing_projectile_energy_MeV_per_u", "local_deposit_MeV"):
                if not math.isfinite(float(row[field])) or float(row[field]) < 0.0:
                    raise SystemExit(f"Invalid {field} in {key}")
            interactions[key] = row
        elif row["record_kind"] == "product":
            if int(row["parent_id"]) != int(row["interaction_track_id"]):
                raise SystemExit(f"Elastic product parent mismatch in {key}")
            if int(row["atomic_number_Z"]) < 0 or int(row["mass_number_A"]) < 0:
                raise SystemExit(f"Invalid product Z/A in {key}")
            if float(row["kinetic_energy_MeV"]) < 0.0 or not math.isfinite(float(row["kinetic_energy_MeV"])):
                raise SystemExit(f"Invalid product energy in {key}")
            if int(row["generation"]) < 0:
                raise SystemExit(f"Invalid product generation in {key}")
            if not str(row["transport_disposition"]):
                raise SystemExit(f"Missing product disposition in {key}")
            products[key].append(row)
        else:
            raise SystemExit(f"Unknown elastic record kind {row['record_kind']!r}")
    orphaned = set(products) - set(interactions)
    if orphaned:
        raise SystemExit(f"Orphan elastic products: {sorted(orphaned)[:3]}")

    interaction_fields = (
        "interaction_id", "run_id", "thread_id", "event_id", "projectile_Z", "projectile_A",
        "incident_energy_MeV_per_u", "outgoing_projectile_energy_MeV_per_u",
        "macroscopic_elastic_per_mm",
        "incident_direction_x", "incident_direction_y", "incident_direction_z",
        "outgoing_direction_x", "outgoing_direction_y", "outgoing_direction_z",
        "local_deposit_MeV", "product_count", "product_offset_zero_based",
        "continuation_disposition", "process_name", "process_type", "process_subtype",
    )
    product_fields = (
        "interaction_id", "product_index", "pdg_id", "atomic_number_Z", "mass_number_A",
        "charge_e", "kinetic_energy_MeV", "direction_x", "direction_y", "direction_z",
        "generation", "transport_disposition",
    )
    interaction_rows: list[list[object]] = []
    product_rows: list[list[object]] = []
    for index, (key, row) in enumerate(sorted(interactions.items()), 1):
        members = products[key]
        offset = len(product_rows)
        interaction_rows.append([
            index, key[0], key[1], key[2], args.projectile_z, args.projectile_a,
            row["incident_energy_MeV_per_u"], row["outgoing_projectile_energy_MeV_per_u"],
            row["macroscopic_elastic_per_mm"],
            row["incident_direction_x"], row["incident_direction_y"], row["incident_direction_z"],
            row["direction_x"], row["direction_y"], row["direction_z"], row["local_deposit_MeV"],
            len(members), offset, "continue", row["process_name"], row["process_type"], row["process_subtype"],
        ])
        for product_index, product in enumerate(members, 1):
            product_rows.append([
                index, product_index, product["pdg_id"], product["atomic_number_Z"], product["mass_number_A"],
                product["charge_e"], product["kinetic_energy_MeV"], product["direction_x"], product["direction_y"],
                product["direction_z"], product["generation"], product["transport_disposition"],
            ])
    write_csv(args.interactions_output, interaction_fields, interaction_rows)
    write_csv(args.products_output, product_fields, product_rows)
    topas_version = geant4_version = None
    runtime_log_metadata = None
    if args.runtime_log is not None:
        log_text = args.runtime_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Welcome to TOPAS.*?Version\s+([^\)]+)\)", log_text)
        topas_version = match.group(1).strip() if match else None
        match = re.search(r"Geant4 version Name:\s+(\S+)", log_text)
        geant4_version = match.group(1) if match else None
        runtime_log_metadata = {"path": args.runtime_log.as_posix(), "sha256": sha256(args.runtime_log)}
    metadata = {
        "case": args.case, "histories": histories, "entries": len(rows),
        "projectile": {"Z": args.projectile_z, "A": args.projectile_a},
        "material": args.material, "physics_model": args.physics_model,
        "provenance": {
            "scorer": "CarbonElasticNtuple",
            "processes": sorted({str(row["process_name"]) for row in interactions.values()}),
            "topas_version": topas_version, "geant4_version": geant4_version,
            "runtime_log": runtime_log_metadata,
            "extraction_mode": args.extraction_mode,
            "sampling_purpose": args.sampling_purpose,
            "continuous_em_loss_enabled": args.continuous_em_loss_enabled == "true",
            "not_for_dose_reference": args.continuous_em_loss_enabled == "false",
        },
        "interactions": len(interaction_rows), "products": len(product_rows),
        "files": {"header": {"path": header.as_posix(), "sha256": sha256(header)}, "phsp": {"path": phsp.as_posix(), "sha256": sha256(phsp)},
                  "interactions": {"path": args.interactions_output.as_posix(), "sha256": sha256(args.interactions_output)},
                  "products": {"path": args.products_output.as_posix(), "sha256": sha256(args.products_output)}},
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(interaction_rows)} elastic interactions and {len(product_rows)} products")


if __name__ == "__main__":
    main()
